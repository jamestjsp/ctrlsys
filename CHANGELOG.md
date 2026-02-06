# Changelog

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
