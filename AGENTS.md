# AGENTS.md

`ctrlsys` is a C11 translation of SLICOT with Python bindings. C routines map as
`AB01MD.f` -> `src/AB/ab01md.c`; headers are under `include/slicot/`; wrappers
are under `python/wrappers/`.

## Commands

Use `.venv/bin/pytest`, not `uv run`; rebuild first with `uv pip install .`.

- Targeted: `.venv/bin/pytest tests/python/test_ROUTINE.py -v`
- Full: `.venv/bin/pytest tests/python/ -n auto`
- Full with crash retry: `.venv/bin/pytest tests/python/ -n auto --reruns 2 --only-rerun "worker .* crashed"`
- Translation checks: `scripts/check_translation_coverage.py --report-only` and `scripts/check_translation_smells.py`
- ASAN Docker before PR: `docker build --platform linux/arm64 -t slicot-asan -f docker/Dockerfile.asan docker/`, then `./scripts/run_asan_docker.sh --no-build`

## Translation Rules

- Types: `INTEGER` -> `i32`, `DOUBLE PRECISION` -> `f64`, SLICOT `LOGICAL` ->
  `bool`; LAPACK/FORTRAN logical ABI slots such as `DGEES` `BWORK`/selection use
  `int`/`i32`.
- Column-major indexing is `a[i + j*lda]`; preserve Fortran slice starts and
  strides exactly, e.g. `C(I,1), LDC` -> `&c[i]`, `ldc`; `C(1,I), 1` ->
  `&c[i*ldc]`, `1`.
- Convert 1-based indices before use and bounds-check converted values before
  indexing.
- For `Fortran_expr - J` with 1-based `J`, substitute `J = j_idx + 1`.
- Use `SLC_DGEMM()` and other `SLC_*` BLAS/LAPACK wrappers with scalar pointers.

## Wrapper Memory

- Inputs: `PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY)`.
- Outputs: allocate with NumPy (`PyArray_New`/`PyArray_ZEROS`) and use
  `PyArray_DATA()`; do not combine `calloc` with `NPY_ARRAY_OWNDATA`.
- Return modified input arrays directly; do not wrap input data in a new owning
  array.
- Use `PyMem_Calloc`/`PyMem_Free` for wrapper temporaries; plain `malloc`/`free`
  is okay for workspaces passed to C routines.

## Review And Tests

Green Python tests are not enough for complex translated branches. For risky
translations, prove target C lines with `scripts/check_translation_coverage.py`
or add a differential pytest using `tests/python/fortran_reference.py` against
`SLICOT-Reference/build` or `CTRLSYS_FORTRAN_REFERENCE_LIB`.

Use NPZ files in `tests/python/data/` for datasets with 50+ values or more than
10 lines. End plans with `Unresolved questions: none` or the open questions.
