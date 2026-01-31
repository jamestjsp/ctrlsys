# SLICOT

[![PyPI version](https://img.shields.io/pypi/v/slicot)](https://pypi.org/project/slicot/)
[![Build Status](https://github.com/jamestjsp/slicot/actions/workflows/test.yml/badge.svg)](https://github.com/jamestjsp/slicot/actions)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Python bindings for **SLICOT** (Subroutine Library In COntrol Theory) - numerical routines for control systems analysis and design. This is a low-level API primarily targeting AI agents and higher-level libraries.

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

## Usage

While you can use this library directly, it's recommended to access it through AI coding agents that understand control theory conventions and SLICOT's API patterns.

**Compatible agents:** Claude Code, Codex, GitHub Copilot, Cursor, and other [Agent Skills](https://agentskills.io)-compatible tools.

### Claude Code Plugin

Add the marketplace and install plugins:

```bash
/plugin marketplace add jamestjsp/slicot
/plugin install slicot-control@jamestjsp-slicot
```

Available plugins:
- **slicot-control** - Navigate 578 SLICOT functions across 27 categories
- **control-theory** - General control theory concepts and workflows
- **pid-loop-tuning** - PID controller design and tuning

Install all three:
```bash
/plugin install slicot-control@jamestjsp-slicot
/plugin install control-theory@jamestjsp-slicot
/plugin install pid-loop-tuning@jamestjsp-slicot
```

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

## Contributions

I don't accept direct contributions. Issues and PRs are welcome for illustration, but won't be merged directly. An AI agent reviews submissions and independently decides whether/how to address them. Bug reports appreciated.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
