# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

C11 translation of SLICOT (Subroutine Library In Control Theory) from Fortran77.

**Reference:** `SLICOT-Reference/src/` (Fortran77), `SLICOT-Reference/doc/` (HTML docs)

## Build Commands

```bash
# Setup (one-time)
uv venv && uv pip install ".[test]"

# Development (build + install + test)
uv pip install . && .venv/bin/pytest tests/python/test_ROUTINE.py -v

# Full test suite
uv pip install . && .venv/bin/pytest tests/python/ -n auto

# Debug build (standalone, without Python)
meson setup build --buildtype=debug && meson compile -C build

# Release build
meson setup build-release --buildtype=release && meson compile -C build-release
```

**Note:** Uses Meson build system via meson-python (PEP 517). Use `.venv/bin/pytest` directly (not `uv run`) to avoid meson editable rebuild issues.

## Memory Debugging

```bash
# ASAN build (native)
meson setup build-asan -Db_sanitize=address -Db_lundef=false
meson compile -C build-asan

# ASAN via Docker (required before PR)
docker build --platform linux/arm64 -t slicot-asan -f docker/Dockerfile.asan docker/
./scripts/run_asan_docker.sh --no-build tests/python/test_x.py -v  # single file
./scripts/run_asan_docker.sh --no-build                             # full suite

# macOS quick checks (assumes venv active)
MallocScribble=1 pytest tests/python/ -n auto          # use-after-free
DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib pytest tests/python/test_x.py -v  # overflow
```

**Note:** `docker build` fails in Claude Code sandbox. Use `--no-build` after pre-building.

## Test Options

```bash
# Single file (during development)
pytest tests/python/test_ab01md.py -v

# Full suite (parallel with retry)
pytest tests/python/ -n auto --reruns 2 --only-rerun "worker .* crashed"
```

## Directory Structure

```
src/XX/routine.c              # C11 implementation (XX=AB,MB,MC...)
include/slicot/*.h            # Family headers (ab.h, mb01.h, sb.h, etc.)
python/wrappers/py_*.c        # Python wrappers by family
python/data/docstrings.json   # Docstrings (source of truth)
python/data/docstrings.h      # AUTO-GENERATED from JSON by Meson
tests/python/test_*.py        # pytest tests
```

**Naming:** `AB01MD.f` → `src/AB/ab01md.c`

## Adding a New Routine

Use `slicot-fortran-translator` agent - handles full TDD workflow including all 9 files.

## Critical Patterns

### Types
- `INTEGER` → `i32`, `DOUBLE PRECISION` → `f64`, `LOGICAL` → `bool`
- **Exception:** LAPACK callbacks (DGEES/DGGES SELECT) MUST use `int`, not `bool` (ABI: FORTRAN LOGICAL=4 bytes, C bool=1 byte)

### Column-Major Arrays
```c
// Index: a[i + j*lda] (row i, col j)
// Memory layout: column-by-column
double a[] = {1.0, 3.0, 2.0, 4.0};  // [[1,2], [3,4]]
```
**NumPy tests:** Always use `order='F'`

### Index Conversion (CRITICAL - Security)
SLICOT returns 1-based indices. Always convert & validate:

```c
// CORRECT
k = iwork[j] - 1;
if (k < 0 || k >= n) break;  // REQUIRED bounds check
// Now safe: iwork[k]

// WRONG - Buffer overflow risk!
k = iwork[j] - 1;
if (iwork[k] < 0) { }  // Missing bounds check before access
```

### Fortran Index Arithmetic (CRITICAL - Heap Corruption)
When translating Fortran index calculations, account for 1-based to 0-based conversion:

```c
// Fortran: NC = MAX(J2-J-1, 0)  where J is 1-based loop variable
// C: j_idx is 0-based, so j_idx = J - 1, meaning J = j_idx + 1

// WRONG - causes buffer overflow!
i32 nc = (j2 - j_idx - 1 > 0) ? j2 - j_idx - 1 : 0;

// CORRECT - substitute J = j_idx + 1 into Fortran formula
// NC = J2 - (j_idx+1) - 1 = J2 - j_idx - 2
i32 nc = (j2 - j_idx - 2 > 0) ? j2 - j_idx - 2 : 0;
```

**Rule**: For `Fortran_expr - J`, substitute `J = j_idx + 1` → result is `Fortran_expr - j_idx - 1`

### Python Wrapper (CRITICAL - Memory)
**In-place modification:**
```c
// CORRECT: Return modified input array
PyObject *result = Py_BuildValue("Odi", b_array, scale, info);
Py_DECREF(a_array);
Py_DECREF(b_array);

// WRONG: Double-free crash!
// PyObject *u_array = PyArray_New(..., b_data, ...);
// PyArray_ENABLEFLAGS(u_array, NPY_ARRAY_OWNDATA);
```

**Input arrays:**
```c
a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
```

**Output arrays (CRITICAL - OWNDATA crash):**
```c
// CORRECT: Let NumPy allocate, then get pointer
npy_intp dims[2] = {m, n};
npy_intp strides[2] = {sizeof(f64), m * sizeof(f64)};
arr = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
f64 *data = (f64*)PyArray_DATA(arr);  // NumPy owns this memory

// WRONG: calloc + OWNDATA causes "free on address not malloc()-ed" under ASAN
// f64 *data = (f64*)calloc(m * n, sizeof(f64));
// arr = PyArray_New(..., data, ...);
// PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);  // CRASH!
```
**Why:** With F-order arrays, NumPy may internally offset the base pointer. When OWNDATA triggers deallocation, NumPy frees the offset address, not the original calloc pointer. See `docs/BUG_PATTERN_OWNDATA_FREE.md`.

**Temporary arrays (ASAN-compatible):**
```c
// CORRECT: Use PyMem for temp arrays allocated/freed in wrapper
c128 *temp = (c128 *)PyMem_Calloc(n, sizeof(c128));
// ... use temp ...
PyMem_Free(temp);

// WRONG: Causes "not malloc()-ed" errors under Docker ASAN
c128 *temp = (c128 *)calloc(n, sizeof(c128));
free(temp);  // ASAN may not track this allocation
```
**Why:** ASAN via `LD_PRELOAD` intercepts malloc/free but Python uses its own memory pool. Mixing allocators causes tracking mismatches. Use `PyMem_*` for wrapper-local temp arrays; standard `malloc`/`free` for workspace passed to C routines.

### BLAS/LAPACK
- Use `SLC_DGEMM()` etc. from `slicot_blas.h`
- Pass scalars by pointer: `SLC_DGEMM("N", "N", &m, &n, &k, &alpha, a, &lda, ...)`

### Error Codes
`info = 0` success, `info < 0` param error, `info > 0` algorithm error

## Test Data Strategy

**Threshold rule**: Use NPZ files for datasets with ≥50 values or >10 lines of data.

| Scenario | Strategy | Example |
|----------|----------|---------|
| Small data (<50 values) | Embed inline | 3x3 matrix, short vector |
| Large data (≥50 values) | NPZ file in `tests/python/data/` | 1000-sample time series |
| Shared between tests | ALWAYS use NPZ | Same I/O data for IB01AD/IB01BD |

**NPZ file pattern:**
```python
# Creating test data file (one-time)
np.savez('tests/python/data/routine_test_data.npz', u=u, y=y, expected_a=a)

# Loading in test
def load_test_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'routine_test_data.npz')
    data = np.load(data_path)
    return data['u'], data['y'], data['expected_a']
```

**Why**: Keeps test files readable (<400 lines), prevents data duplication, enables data sharing between related tests.

## Docs

- `fortran_diag/README.md` - C vs Fortran debugging
- `tools/README.md` - Workflow examples
- `.claude/skills/slicot-knowledge/SKILL.md` - Detailed translation knowledge (invoke with `/slicot-knowledge`)
