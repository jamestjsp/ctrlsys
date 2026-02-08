# SLICOT for Python

C11 translation of [SLICOT](https://github.com/SLICOT/SLICOT-Reference) (Subroutine Library In Control Theory) with Python bindings.

**552/627 routines translated (88%)** — Riccati/Lyapunov solvers, system identification, model reduction, and more.

## Install

```bash
pip install slicot
```

## Usage

```python
import numpy as np
import slicot

# Continuous-time Algebraic Riccati Equation (CARE)
# A'X + XA - XBR^{-1}B'X + Q = 0
A = np.array([[0, 1], [-2, -3]], dtype=np.float64, order='F')
B = np.array([[0], [1]], dtype=np.float64, order='F')
Q = np.array([[1, 0], [0, 2]], dtype=np.float64, order='F')
R = np.array([[1]], dtype=np.float64, order='F')
G = B @ np.linalg.solve(R, B.T)

X, scale, sep, ferr, _, _ = slicot.sb02md(
    'C', 'N', 'U', 'N', A, G, Q
)
```

## Features

- **Column-major storage** — Fortran-compatible, works with `order='F'` NumPy arrays
- **552 routines** — Riccati, Lyapunov, Sylvester solvers; system identification (MOESP, N4SID); model reduction (balanced truncation, Hankel norm); eigenvalue assignment; H-infinity norm
- **BLAS/LAPACK backend** — Links against system OpenBLAS or scipy-openblas32

## Routine Families

| Family | Count | Description |
|--------|-------|-------------|
| AB | 55 | Analysis (controllability, observability, norms) |
| SB | 113 | Synthesis (Riccati, Lyapunov, pole placement) |
| MB | 229 | Matrix operations (decompositions, transformations) |
| MA | 37 | Matrix analysis (norms, condition numbers) |
| IB | 16 | System identification (MOESP, N4SID) |
| Others | 83 | Benchmarks, data generation, transforms |

## License

BSD 3-Clause — based on [SLICOT-Reference](https://github.com/SLICOT/SLICOT-Reference)

## Links

- [Source Code](https://github.com/jamestjsp/slicot)
- [SLICOT Documentation](http://slicot.org/objects/software/shared/doc/SLICOT-Reference.pdf)
- [Original SLICOT-Reference](https://github.com/SLICOT/SLICOT-Reference)
