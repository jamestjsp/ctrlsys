# ctrlsys

[![PyPI version](https://img.shields.io/pypi/v/ctrlsys)](https://pypi.org/project/ctrlsys/)
[![Build Status](https://github.com/jamestjsp/ctrlsys/actions/workflows/test.yml/badge.svg)](https://github.com/jamestjsp/ctrlsys/actions)
[![Documentation](https://readthedocs.org/projects/ctrlsys/badge/?version=latest)](https://ctrlsys.readthedocs.io/en/latest/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Python bindings for **SLICOT** (Subroutine Library In COntrol Theory) - numerical routines for control systems analysis and design. This is a low-level API primarily targeting AI agents and higher-level libraries.

> **Note:** This package was previously published as `slicot` on PyPI. Install with `pip install ctrlsys`.

## Installation

```bash
pip install ctrlsys
```

## Features

- **600+ routines** for control systems
- **State-space methods**: Riccati, Lyapunov, pole placement
- **Model reduction**: Balance & Truncate, Hankel-norm
- **System identification**: MOESP, N4SID
- **NumPy integration**: Column-major arrays

## Usage

While you can use this library directly, it's recommended to access it through AI coding agents that understand control theory conventions and SLICOT's API patterns.

For agent skills (Claude Code, Cursor, etc.), see [control-skills](https://github.com/jamestjsp/control-skills).

## Quick Start

```python
import numpy as np
import ctrlsys

# Controllability analysis
A = np.array([[1, 2], [3, 4]], order='F')
B = np.array([[1], [0]], order='F')

a_out, b_out, ncont, z, tau, info = ctrlsys.ab01md('I', A, B.flatten(), 0.0)
print(f"Controllable dimension: {ncont}")
```

## Column-Major Arrays

SLICOT uses Fortran conventions:

```python
A = np.array([[1, 2], [3, 4]], order='F')  # Required!
```

## Contributions

I don't accept direct contributions. Issues and PRs are welcome for illustration, but won't be merged directly. An AI agent reviews submissions and independently decides whether/how to address them. Bug reports appreciated.

## License

This project (C11 translation and Python bindings) is BSD-3-Clause. See [LICENSE](LICENSE).

The original SLICOT library is copyright © 1996-2025 The SLICOT Team, maintained by [NICONET e.V.](https://www.slicot.org/), and licensed under BSD-3-Clause. See [LICENSE-SLICOT](LICENSE-SLICOT).
