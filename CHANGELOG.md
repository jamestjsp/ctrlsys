# Changelog

## [1.0.19] - 2026-02-15

### Changed

- Package renamed from `slicot` to `ctrlsys` on PyPI
- GitHub repo moved from `jamestjsp/slicot` to `jamestjsp/ctrlsys`
- `pip install slicot` now installs a shim that depends on `ctrlsys` and emits a deprecation warning
- All `import slicot` → `import ctrlsys` in user-facing code
- C internals (headers, guards, extension module name) unchanged

## [1.0.18] - 2026-02-08

### Changed

- CI: consolidated 5 workflow files into 3 — eliminated redundant test runs on push
- CI: split publish pipeline into fast/slow phases — x86_64/macOS/Windows wheels on PyPI in ~12min, aarch64 follows ~90min later
- CI: CodeQL moved to weekly schedule only (was per-push)

### Added

- Documentation badge and link in README and PyPI metadata

## [1.0.15] - 2026-02-07

### Added

- Complete 11 C translations from stubs: ab09iy, ab13dd, ab13hd, mb01uy, mb01vd, mb02pd, mb03bd, mb03xp, mb03yd, mb04pb, mb04rb, mb04tb (+5.5k lines)
- mb04pb test with expanded coverage
- LSP configs (.clangd, pyrightconfig.json)

### Fixed

- Compiler warnings eliminated across 141 files (964 → 0)
- ab13hd workspace query in Python wrapper

## [1.0.14] - 2026-02-07

### Fixed

- sg03bx: E-standardization bugs — AS(2,2) typo and missing B rotation (upstream [SLICOT-Reference#39](https://github.com/SLICOT/SLICOT-Reference/issues/39))
- sg03bu/bv/bw: Lyapunov solver indexing and DSCAL pointer fixes
- mb04hd: remove unnecessary bool cast
- py_mb04: calloc→PyMem_Calloc for ASAN compatibility
- mb04rd: Linux OpenBLAS crash resolved — skip marker removed

### Added

- Expanded test coverage across 70 files (AB/AG/IB/MB/MC/MD/SB/SG/TB families)
- Residual verification tests for sg03bx
- User guide restructured around python-control workflows, added llms.txt

## [1.0.13] - 2026-02-05

### Added

- sb02rd: condition estimation via SB02QD/SB02SD
- C11 vs Fortran77 benchmark infrastructure
- `scripts/bump_version.py` for atomic version updates with drift detection

### Changed

- Migrated build system to uv
- Single source of truth for package version — reduced from 5 manual touch points to 2 (`pyproject.toml` + `meson.build` line 2), with `__init__.py` derived via `importlib.metadata`, `slicot_config.h` generated via meson `configure_file()`, and version macros parsed from `meson.project_version()`
- Python version requirement updated to 3.11+

### Fixed

- Buffer overflows in 7 routines causing parallel test crashes
- Standalone `meson setup build` broken by `include_directories()` rejecting absolute numpy paths
- CI gaps after build system migration to uv
- test_mb03rd: replaced `scipy.linalg.solve` with numpy
- Publish CI failures: LAPACK detection conflicting with scipy-openblas32, PEP 668 sdist build, missing pytest-xdist

## [1.0.12] - 2025-02-01

### Fixed

- ab13dd: L-infinity norm algorithm bugs causing 8.8% error for MIMO systems
  - Tolerance formula for imaginary-axis eigenvalue detection
  - Iteration loop count (off-by-one)
  - Algorithm structure: frequency collection, sorting, midpoint computation
  - MB03XD BALANC parameter ('B' not 'N')
  - Hamiltonian scaling (divide by gamma not gamma²)
  - MB03XD SCALE array buffer overflow (crash on Linux)

## [1.0.11] - 2025-01-31

### Fixed

- ab13dd: segfault for n=1 single-state systems (mb03xd requires T workspace even for JOB='E')

## [1.0.10] - 2025-01-30

### Fixed

- mb03rd: transformation matrix X now correctly satisfies X⁻¹AX = A_out

## [1.0.0] - 2025-01-23

### Added

- Initial release
- 550+ SLICOT routines (AB, SB, MB, IB, TG, SG families)
- Pre-built wheels: Linux (x86_64, aarch64), macOS (arm64), Windows (AMD64, ARM64)
- Python 3.11-3.13, NumPy 2.x
- Type stubs (PEP 561)
