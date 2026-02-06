# slicot.c

C11 translation of SLICOT (Subroutine Library In Control Theory) from Fortran77.

**Based on:** [SLICOT-Reference](https://github.com/SLICOT/SLICOT-Reference) (BSD 3-Clause License)
**License:** BSD 3-Clause (see [LICENSE](LICENSE))

## Quick Start

```bash
# Install with uv
uv pip install ".[test]"

# Run tests
.venv/bin/pytest tests/python/ -v
```

## Translation Status

**552/627 routines translated (88%)**

| Family | Translated |
|--------|------------|
| AB | 55 |
| MB | 229 |
| SB | 113 |
| MA | 37 |
| MC | 19 |
| IB | 16 |
| Others | 83 |

513 test files covering translated routines.

## Features

- Column-major storage (Fortran-compatible)
- Python bindings (NumPy arrays)
- TDD workflow (RED→GREEN→REFACTOR)

## Docs

- **[CLAUDE.md](CLAUDE.md)** - Development workflow & translation patterns

## Contributions

I don't accept direct contributions. Issues and PRs are welcome for illustration, but won't be merged directly. An AI agent reviews submissions and independently decides whether/how to address them. Bug reports appreciated.
