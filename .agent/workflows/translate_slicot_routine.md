---
description: Translate a SLICOT Fortran routine to C following strict TDD and memory safety guidelines.
---

1. **Parse Documentation**
   - Follow the workflow: `.agent/workflows/parse_slicot_documentation.md`
   - **Goal**: Understand the routine's purpose, inputs, outputs, and algorithm.

2. **Analyze Dependencies**
   - Follow the workflow: `.agent/workflows/analyze_slicot_dependencies.md`
   - **Goal**: Ensure the routine is ready for translation (Level 0 or all dependencies met).

3. **Extract Test Data (RED Phase)**
   - Follow the workflow: `.agent/workflows/extract_slicot_test_data.md`
   - **Goal**: Create accurate test data (NPZ or inline) with correct column/row-major parsing.

4. **Write Tests (RED Phase)**
   - Create `tests/python/test_[routine].py`.
   - **Requirements**:
     - `np.random.seed(N)` for ANY random data.
     - Validate actual numerical values (not just shapes).
     - Mathematical property tests (e.g. eigenvalue preservation, residual checks).
     - Minimum 3 tests: Basic, Edge Case, Error Handling.
   - Run `pytest tests/python/test_[routine].py` and confirm it FAILS.

5. **Implement C Code (GREEN Phase)**
   - Create `src/[XX]/[routine].c` (lowercase).
   - **Checklist**:
     - [ ] Includes: `slicot.h` and `slicot_blas.h`.
     - [ ] Use `SLC_*` macros for BLAS/LAPACK.
     - [ ] Verify **Critical Patterns** (Types, Indexing, Memory) in `.agent/rules/slicot-fortran-c11-translator.md`.
     - [ ] Bounds check ALL array accesses.
     - [ ] Validate 1-based to 0-based index conversions.
   - Update `src/CMakeLists.txt` to include the new file.
   - Update `include/slicot.h` with the function declaration and Doxygen comments.
   - Update `python/slicot_module.c` to wrap the function.
   - Update `python/slicot/__init__.py` to export it.

6. **Build and Verify (GREEN Phase)**
   - Run `cmake --build --preset linux-x64-debug-build`.
   - Run `uv pip install .`.
   - Run `.venv/bin/pytest tests/python/test_[routine].py`.
   - Ensure all tests PASS.

7. **Refactor and Finalize**
   - Clean up code, remove debug prints.
   - Ensure no double-free issues in Python wrappers.
   - Run full test suite: `pytest tests/python/`.
