# SLICOT

[![PyPI version](https://img.shields.io/pypi/v/slicot)](https://pypi.org/project/slicot/)
[![Build Status](https://github.com/jamestjsp/slicot/actions/workflows/test.yml/badge.svg)](https://github.com/jamestjsp/slicot/actions)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Python bindings for **SLICOT** (Subroutine Library In COntrol Theory) - numerical routines for control systems analysis and design.

## Installation

```bash
pip install slicot
```

## Features

- **600+ routines** for control systems
- **State-space methods**: Riccati, Lyapunov, pole placement
- **Model reduction**: Balance & Truncate, Hankel-norm
- **System identification**: MOESP, N4SID
- **NumPy integration**: Column-major arrays

## Quick Start

```python
import numpy as np
import slicot

# Controllability analysis
A = np.array([[1, 2], [3, 4]], order='F')
B = np.array([[1], [0]], order='F')

a_out, b_out, ncont, z, tau, info = slicot.ab01md('I', A, B.flatten(), 0.0)
print(f"Controllable dimension: {ncont}")
```

## Column-Major Arrays

SLICOT uses Fortran conventions:

```python
A = np.array([[1, 2], [3, 4]], order='F')  # Required!
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
