# slicot Python Package Specification

Pure C11 implementation of SLICOT (Subroutine Library In Control Theory). No Fortran compiler required.

## Package Identity

| Attribute | Value |
|-----------|-------|
| **PyPI name** | `slicot` |
| **Import** | `import slicot` |
| **License** | BSD-3-Clause |
| **Version** | 1.0.2 |
| **Status** | Released |

## Platform Support

| Platform | Architectures | Notes |
|----------|---------------|-------|
| **Linux** | x86_64 | manylinux wheels |
| **macOS** | ARM64 | Apple Silicon |
| **Windows** | x64 | |

## Dependencies

| Dependency | Constraint | Strategy |
|------------|------------|----------|
| **Python** | ≥3.11 | |
| **NumPy** | ≥2.0 | NumPy 1.x not supported |
| **OpenBLAS** | via scipy-openblas32 | Vendored in wheels |
| **Meson** | ≥1.1.0 | Build-time only |

### sdist
- Uses system BLAS/LAPACK or scipy-openblas32
- Self-contained build with pkg-config detection

## API Design

### Namespace
Flat namespace matching slycot:
```python
import slicot
a_out, b_out, ncont, z, tau, info = slicot.ab01md('I', a, b, tol)
```

### Compatibility
- **slycot API compatible**: Drop-in replacement where routines overlap
- **SemVer strict**: Breaking changes only in major versions
- **Deprecation**: 1 minor version notice before removal

### Index Convention
- **0-based**: All returned indices converted from SLICOT's 1-based to Python's 0-based
- Users cannot directly map to Fortran docs (intentional trade-off for Pythonic API)

### Array Mutation
- **In-place**: Input arrays modified like Fortran SLICOT
- Documented clearly; users copy if immutability needed

### Error Handling
| SLICOT INFO | Python behavior |
|-------------|-----------------|
| INFO = 0 | Success, return results |
| INFO < 0 | `ValueError` with parameter name (e.g., "Invalid value for parameter N (position 3)") |
| INFO > 0 | Raise exception (e.g., `SlicotConvergenceError`) |

### Thread Safety
- Thread-safe: workspace allocated per call
- Safe for concurrent use from multiple threads

## Scope

| Metric | Count |
|--------|-------|
| **C routines implemented** | 624 |
| **Python wrappers** | 578 functions |
| **Categories** | 27 (AB, AG, BB, BD, DE, DF, DG, DK, DL, FB, FD, IB, MA, MB, MC, MD, NF, SB, SG, TB, TC, TD, TF, TG, UD, UE, ZG) |
| **Complex routines** | Included (Z-prefix) |

### Not included in v1.0
- WASM/Pyodide support (dropped due to BLAS/LAPACK complexity)
- scipy.linalg compatibility wrappers
- Jupyter example notebooks
- Type stubs (.pyi files)

## Repository Structure

### GitHub
| Setting | Value |
|---------|-------|
| **URL** | https://github.com/jamestjsp/slicot |
| **Visibility** | Public |
| **Structure** | Monorepo (C + Python) |

### Contributions Policy
No direct contributions accepted. Issues and PRs welcome for illustration but won't be merged directly. AI agent reviews submissions and independently decides whether/how to address them. Bug reports appreciated.

## CI/CD

### GitHub Actions Structure
- **Reusable workflows** in `.github/workflows/`
- **Staged execution**: Tests pass first → build all wheels

### Workflows
| Workflow | Trigger |
|----------|---------|
| `test.yml` | Push/PR to main |
| `build.yml` | Reusable, builds wheels |
| `publish.yml` | Git tag `v*` or manual dispatch |
| `codeql.yml` | Push/PR to main |

### Test Strategy
| Phase | Tests run |
|-------|-----------|
| **Quick Test** | `test_ab01md.py` (smoke test) |
| **Full Suite** | All 6400+ tests (Ubuntu only) |
| **macOS** | Quick test only (Accelerate issues) |

### Build Tools
- **meson-python** for Python packaging
- **cibuildwheel** for wheel building
- **delvewheel** for Windows DLL bundling

## Release Process

### Triggers
- Git tag (e.g., `v1.0.1`) triggers release workflow
- Manual dispatch with `skip_tests` option

### Pipeline
1. Run tests (skippable)
2. Build wheels for all platforms
3. Publish directly to **PyPI**

### Authentication
- **Trusted Publishers (OIDC)**: No API tokens stored in secrets

## Documentation

### Hosting
| Resource | URL |
|----------|-----|
| **Docs** | https://slicot.readthedocs.io/ |
| **PyPI** | https://pypi.org/project/slicot/ |
| **GitHub** | https://github.com/jamestjsp/slicot |

### Content
- Installation guide
- Quick start examples
- API reference (578 functions with full docstrings)

### No notebooks
- Examples in documentation only
- No Jupyter notebooks in repo

## Build Configuration

### BLAS/LAPACK Detection
Meson auto-detects in order:
1. OpenBLAS
2. scipy-openblas
3. System BLAS
4. Accelerate (macOS)

### Fortran Calling Convention
Auto-detected at build time:
- `scipy_dgemm_` (scipy-openblas)
- `dgemm_` (standard)
- `dgemm` (no underscore)

## Quality Gates

| Check | Blocking |
|-------|----------|
| Tests pass (Ubuntu + macOS) | ✓ |
| CodeQL findings | ✓ |

## Future Roadmap

- [ ] Type stubs (.pyi files)
- [ ] conda-forge package
- [ ] Linux ARM64 wheels
- [ ] Windows ARM64 wheels
- [ ] scipy.linalg-style high-level wrappers

## Comparison with slycot

| Feature | slicot (this) | slycot |
|---------|---------------|--------|
| **Language** | C11 | Fortran77 |
| **Fortran compiler** | Not needed | Required |
| **Wheels** | Pre-built all platforms | Source-only on PyPI |
| **License** | BSD-3-Clause | GPLv2 |
| **Routine count** | 578 | ~40 |
| **NumPy** | 2.x only | 1.x + 2.x |
