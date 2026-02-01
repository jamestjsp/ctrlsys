# SLICOT Benchmarks

Two benchmark suites: **Python wrapper benchmarks** (pytest-benchmark) and **C vs Fortran benchmarks** (native).

## Python Wrapper Benchmarks

pytest-benchmark tests for the Python API.

```bash
# Install pytest-benchmark
pip install pytest-benchmark

# Run benchmarks
pytest benchmarks/ --benchmark-only
```

## C11 vs Fortran77 Benchmarks

Compares native C11 implementation against original Fortran77.

### Prerequisites

1. **Build C library with benchmarks**
   ```bash
   meson setup build && meson compile -C build
   ```

2. **Build Fortran reference library**
   ```bash
   cd SLICOT-Reference
   cmake -B build_bench
   cmake --build build_bench
   cd ..
   ```

### Running

```bash
# Full comparison (generates markdown report)
python scripts/benchmark_c_vs_fortran.py --output benchmark_report.md

# Custom build directory
python scripts/benchmark_c_vs_fortran.py --c-build-dir build --output report.md

# Specific routines only
python scripts/benchmark_c_vs_fortran.py -r bb01ad sb02md

# Run native C benchmarks via meson
meson benchmark -C build
```

### Output

Generates `benchmark_report.md` with:
- Executive summary (speedup statistics)
- System configuration
- Per-routine comparison tables
- Raw benchmark data (CSV)

### Routines Benchmarked

| Routine | Description |
|---------|-------------|
| BB01AD | CAREX - Continuous-time Riccati examples |
| BB02AD | DAREX - Discrete-time Riccati examples |
| BB03AD | CTLEX - Continuous-time Lyapunov examples |
| BB04AD | DTLEX - Discrete-time Lyapunov examples |
| BD01AD | CTDSX - Continuous-time descriptor examples |
| BD02AD | DTDSX - Discrete-time descriptor examples |
| SB03MD | Continuous-time Lyapunov solver |
| SB03OD | Discrete-time Lyapunov solver (Cholesky) |

### Requirements

- Python 3.11+
- gfortran (for Fortran benchmarks)
- meson, ninja (for C build)
- BLAS/LAPACK (Accelerate on macOS, OpenBLAS on Linux)
