# CLAUDE.md

C11 implementation of SLICOT (Subroutine Library In Control Theory). Package name: `ctrlsys`. Internal C code uses `slicot` prefixes.

## Build Commands

```bash
# Setup (one-time)
uv venv && uv pip install ".[test]"  # installs as 'ctrlsys' package
meson setup build --buildtype=debug  # generates compile_commands.json for clangd

# Development (build + install + test)
uv pip install . && .venv/bin/pytest tests/python/test_ROUTINE.py -v

# Full test suite
uv pip install . && .venv/bin/pytest tests/python/ -n auto

# Full suite with retry
.venv/bin/pytest tests/python/ -n auto --reruns 2 --only-rerun "worker .* crashed"

# Debug build (standalone, without Python)
meson setup build --buildtype=debug && meson compile -C build

# Release build
meson setup build-release --buildtype=release && meson compile -C build-release

# Bump version
python scripts/bump_version.py X.Y.Z

# ASAN build (native)
meson setup build-asan -Db_sanitize=address -Db_lundef=false && meson compile -C build-asan

# ASAN via Docker (required before PR) — pre-build image first
docker build --platform linux/arm64 -t slicot-asan -f docker/Dockerfile.asan docker/
./scripts/run_asan_docker.sh --no-build tests/python/test_x.py -v  # single file
./scripts/run_asan_docker.sh --no-build                             # full suite

# macOS quick checks
MallocScribble=1 pytest tests/python/ -n auto
DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib pytest tests/python/test_x.py -v
```

**Note:** Uses meson-python (PEP 517). Use `.venv/bin/pytest` directly (not `uv run`). `docker build` fails in Claude Code sandbox — use `--no-build`.

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

## Critical Patterns

- **Types:** `INTEGER` → `i32`, `DOUBLE PRECISION` → `f64`, `LOGICAL` → `bool`. Exception: LAPACK SELECT callbacks use `int` not `bool` (FORTRAN LOGICAL=4 bytes)
- **Column-major:** `a[i + j*lda]`. NumPy tests use `order='F'`
- **Index conversion:** 1-based → 0-based. Always bounds-check before using converted index: `k = iwork[j] - 1; if (k < 0 || k >= n) break;`
- **Fortran index arithmetic:** For `Fortran_expr - J` where J is 1-based, substitute `J = j_idx + 1` → result is `Fortran_expr - j_idx - 1`
- **BLAS/LAPACK:** Use `SLC_DGEMM()` etc. from `slicot_blas.h`, scalars by pointer
- **Error codes:** `info = 0` success, `< 0` param error, `> 0` algorithm error

## Python Wrapper Memory Rules

- **Input arrays:** `PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY)`
- **Output arrays:** Let NumPy allocate via `PyArray_New` with `NULL` data pointer, then use `PyArray_DATA()`. Never use `calloc` + `NPY_ARRAY_OWNDATA` (F-order pointer offset causes free crash)
- **In-place modification:** Return the modified input array directly. Never wrap input data in a new array with OWNDATA
- **Temp arrays in wrappers:** Use `PyMem_Calloc`/`PyMem_Free` (ASAN-compatible). Use standard `malloc`/`free` only for workspace passed to C routines

## Test Data

Use NPZ files (`tests/python/data/`) for datasets with ≥50 values or >10 lines. Small data inline.
