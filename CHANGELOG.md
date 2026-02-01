# Changelog

## [1.0.12] - 2025-02-01

### Fixed

- ab13dd: L-infinity norm algorithm bugs causing 8.8% error for MIMO systems
  - Tolerance formula for imaginary-axis eigenvalue detection
  - Iteration loop count (off-by-one)
  - Algorithm structure: frequency collection, sorting, midpoint computation
  - MB03XD BALANC parameter ('B' not 'N')
  - Hamiltonian scaling (divide by gamma not gamma²)

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
