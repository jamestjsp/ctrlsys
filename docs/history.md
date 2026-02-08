# The History of SLICOT, Slycot, and python-control

## 1. SLICOT: The Original Fortran Library (1996-present)

**SLICOT** (Subroutine Library In COntrol Theory) is a comprehensive Fortran 77 library for numerical computations in systems and control theory.

### Origins
- **Created:** 1996-1997 by a team of numerical linear algebra experts
- **Key authors:** P. Benner, V. Mehrmann, V. Sima, S. Van Huffel, A. Varga
- **Organization:** Developed under the **NICONET e.V.** (Network for Numerics in Control), a European thematic network
- **First reference:** NICONET Report 97-3 (June 1997)
- **Website:** https://www.slicot.org/

### What It Does
- 280+ user-callable subroutines built on top of BLAS and LAPACK
- Riccati, Lyapunov, and Sylvester equation solvers
- H-infinity and H2 optimal control synthesis
- Model reduction (balanced truncation, Hankel-norm)
- System identification (MOESP, N4SID)
- Pole placement, frequency response, state-space transformations
- Used by **MATLAB** (since v7, June 2004), Octave, and Scilab

### Licensing History (Critical)
| Period | Version | License |
|--------|---------|---------|
| 1996-2009 | v5.0 and earlier | Permissive / open |
| 2012+ | v5.5+ | **Restricted** - free for academic use, commercial license required |
| Dec 2020 | v5.7 (SLICOT-Reference on GitHub) | **BSD-3-Clause** - fully open source again |

The mid-period licensing restriction created massive friction in the open-source ecosystem and was a primary driver of the "move away from SLICOT" debate.

---

## 2. python-control: The Python Control Systems Library (2009-present)

### Origins and Growth
- **Created:** ~2009 on SourceForge, moved to GitHub Aug 9, 2014
- **Founder:** Richard Murray (Caltech) with Clancy Rowley (Princeton)
- **Published:** 60th IEEE Conference on Decision and Control (CDC), 2021
- **Current stats:** 1,962 stars, 448 forks, 73 contributors
- **Latest release:** v0.10.2 (July 2025)

### Key Contributors
| Contributor | Role | Commits |
|------------|------|---------|
| Richard Murray (@murrayrm) | Founder, primary maintainer | 1,417 |
| Ben Greiner (@bnavigator) | Packaging/build expert, Slycot maintainer | 302 |
| Sawyer Fuller (@sawyerbfuller) | Core contributor | 194 |
| Rory Yorke (@roryyorke) | Core contributor, release management | 123 |
| Rene van Paassen (@repagh) | Core contributor, Slycot maintainer | 95 |
| Clancy Rowley (@cwrowley) | Co-founder | 89 |

### SLICOT Dependency Architecture
python-control uses SLICOT (via Slycot) as an **optional** dependency:

**Functions requiring Slycot (no fallback):**
- `hinfsyn()` - H-infinity synthesis
- `h2syn()` - H2 synthesis
- `mixsyn()` - Mixed-sensitivity synthesis
- Advanced model reduction
- Some MIMO frequency response computations

**Functions with SciPy fallback:**
- `lyap()`, `dlyap()` - Lyapunov equations
- `care()`, `dare()` - Riccati equations
- H2/Hinf norms (partial)

---

## 3. Slycot: The Troubled Bridge (2011-present)

### History
- **Original:** Created by @avventi (~2011), unmaintained since 2013
- **Fork:** @jgoppert maintained a fork (linked from PyPI 0.2.0), inactive since 2016
- **Consolidation:** python-control/Slycot became the "official" repo in 2018 (Issue #16)
- **Current stats:** 148 stars, 45 forks
- **Latest release:** v0.6.1 (Nov 2025)

### Key Releases
| Version | Date | Highlights |
|---------|------|------------|
| v0.3.3 | June 2018 | First major consolidated release |
| v0.3.4 | May 2019 | Switched to scikit-build/CMake |
| v0.4.0 | May 2020 | 6 new SLICOT wrappers |
| v0.5.0 | July 2022 | Major cleanup, dropped Python 3.7 |
| v0.6.0 | April 2024 | Updated to SLICOT 5.9 |
| v0.6.1 | Nov 2025 | Latest |

### The Installation Problem (The Core Pain Point)

Slycot requires a full Fortran compilation toolchain:
1. **C compiler** (gcc, MSVC, clang)
2. **Fortran compiler** (gfortran, ifort, flang)
3. **BLAS/LAPACK** (OpenBLAS, MKL, ATLAS)

**No binary wheels on PyPI** - only source distributions. Users must compile from source.

> "We don't publish wheels because we don't know how to build them" - @roryyorke

**Platform-specific nightmares:**
- **Windows:** Requires Visual Studio + Fortran compiler + compatible BLAS. DLL loading failures. "Compiling Slycot is unfortunately a bit tricky, especially on Windows" - README
- **macOS:** Segfaults on ARM (M1/M2) Macs. OpenBLAS path issues between OS versions.
- **Linux:** MKL conflicts, architecture-specific failures (aarch64)

**Only reliable installation:** `conda install -c conda-forge slycot`

This installation barrier is the #1 reason students and users abandon python-control and return to MATLAB.

### The License Problem

| Component | License |
|-----------|---------|
| SLICOT upstream (v5.7+) | BSD-3-Clause |
| Slycot wrappers | **GPL-2.0** |
| python-control | BSD-3-Clause |

Slycot **cannot** change its license without permission from all contributors. This means:
- python-control users who install Slycot are subject to GPL restrictions
- Corporate users with GPL-incompatible policies are blocked from using advanced features
- Issue #1057 (Nov 2024): Corporate users explicitly reported being blocked

---

## 4. The Decade-Long "Move Away From SLICOT" Debate

### Issue #27 (Aug 2014 - still open): "Move away from SLICOT?"

This is perhaps the most significant discussion in python-control's history.

**The Proposal (Clancy Rowley, Aug 2014):**
> Rewrite routines to eliminate SLICOT dependency. SciPy now has Lyapunov and Riccati solvers.

**The Pro-Replacement Camp (@murrayrm, @cwrowley, @ilayn):**
- Installation difficulties drive users away
- Students gave up and returned to MATLAB
- Closed-source licensing (pre-2020) is unacceptable
- "Clunky interface"
- SciPy has added key solvers
- Want pure Python/Cython for ease of maintenance

**The Pro-SLICOT Camp (@pmli, passionate 2016 defense):**
- Numerical stability is critically important; disasters from errors cost money and lives
- SLICOT developed by experts over 10+ years, battle-tested since 1997
- SciPy implementations are suboptimal (e.g., `solve_lyapunov` computes Schur decomposition twice)
- python-control itself has "dubious numerical codes" (eigenvalues via characteristic polynomial)
- Pole placement is "the classic example of a numerically difficult problem"
- "Placing Plenty of Poles is Pretty Preposterous"
- Replacing SLICOT requires deep numerical linear algebra expertise
- Known to outperform MATLAB functions, often by orders of magnitude

**The Middle Ground (@repagh, @jgoppert):**
- Make Slycot optional, not required
- Provide fallback implementations
- Keep SLICOT as backend for performance-critical users

**Outcome:** Slycot became optional. No replacement was developed. The issue remains open after 12 years.

### @ilayn's Contributions
Ilhan Polat (@ilayn, SciPy maintainer) made notable contributions:
- Implemented some SLICOT MIMO equivalents in Python
- Coded BBBS algorithm for H-infinity norm (MIT licensed)
- Argued open source implementations can be good with enough motivation
- Later proposed rewriting SLICOT in Python, pointing to SciPy's successful Fortran-to-Python conversions (30K+ LOC of F77 already converted in scipy/scipy#18566)

---

## 5. The jamestjsp/slicot C11 Translation (Jan 2026-present)

### Project Overview
- **Created:** January 24, 2026
- **Author:** James Joseph (@jamestjsp)
- **Approach:** C11 translation of SLICOT from Fortran 77, with Python bindings
- **Stats:** 600+ routines, v1.0.18, BSD-3-Clause
- **Build system:** Meson + meson-python (PEP 517)
- **Key advantage:** `pip install slicot` - no Fortran compiler needed

### PR #1200: "Replace slycot with slicot" (Jan 31 - Feb 6, 2026)

The first attempt to integrate the C11 translation into python-control.

**What was proposed:**
- Add `control/slicot_compat.py` compatibility layer
- Support both slicot (C11) and slycot (Fortran) with automatic fallback
- Replace all slycot imports

**Timeline:**
| Date | Event |
|------|-------|
| Jan 31 | PR opened. Initial commit replaces slycot throughout. |
| Feb 1 | @bnavigator reviews, finds wrapper signature bugs (ab13bd expects 7 args, got 10; tb05ad incompatible) |
| Feb 1 | Numerical bug found: ab13dd L-inf norm returns 4.65 vs expected 4.28 |
| Feb 1 | Fixed in slicot v1.0.12, published to PyPI within hours |
| Feb 2 | @bnavigator runs benchmark: **Fortran 8.97s vs C11 14.53s** (~2x slower) |
| Feb 2 | @ilayn raises licensing/naming concerns: cannot use "SLICOT" name without permission |
| Feb 2 | @bnavigator: "This greatly diminishes my confidence... Did you actually test your code or are we dealing with AI slop?" |
| Feb 3 | @bnavigator identifies root cause: wrapper infers dimensions from array shapes, but python-control passes pre-padded arrays |
| Feb 3 | @jamestjsp confesses: "I have not read one character of this Claude-generated code. Sorry @bnavigator you were talking to Claude so far." |
| Feb 4 | @bnavigator responds: "Your overly verbose and factually wrong LLM output is extremely annoying." Identifies the bug is in the wrapper, not python-control. |
| Feb 6 | PR closed by author. "No plan to fix it." |

### Key Technical Issues Exposed

1. **API incompatibility:** Slycot passes explicit `n, m, p` dimensions; slicot C11 infers from array shapes. When python-control pre-pads arrays (B to max(m,p) columns), slicot infers wrong dimensions.

2. **Performance regression:** C11 version ~2x slower than Fortran in python-control's test suite (14.53s vs 8.97s).

3. **Numerical precision:** Some routines (ab13dd) produced different results. Fixed in v1.0.12 but revealed insufficient testing.

### Key Stakeholder Positions

**@bnavigator (Slycot maintainer):**
- Deeply skeptical of AI-generated code quality
- Found real bugs through code review
- Insisted wrapper bug was in slicot's Python bindings, not python-control
- Performance concern: won't accept 2x regression

**@ilayn (SciPy maintainer):**
- Raised legal concern: "SLICOT" name requires permission from original authors
- License separation: original SLICOT copyright must be in separate file
- Has own views on rewriting SLICOT in Python instead

**@murrayrm (python-control founder):**
- Did not comment on PR #1200

---

## 6. Current State of the Ecosystem (Feb 2026)

### python-control
- Stable, well-maintained, regular releases
- 96 open issues, active development
- Slycot remains the only SLICOT backend
- Issue #27 ("move away from SLICOT") still open after 12 years

### Slycot
- Moderately active maintenance (small team)
- GPL-2.0 license remains unchanged
- No binary wheels on PyPI (conda-forge only)
- 15 open issues

### jamestjsp/slicot (C11)
- Active development (v1.0.18)
- Not integrated into python-control (PR #1200 closed)
- Available on PyPI with prebuilt wheels
- 2 GitHub stars

---

## 7. The Fundamental Tensions

### What the community wants:
- BSD-compatible license
- Easy installation (`pip install`)
- Full SLICOT functionality (H-inf synthesis, etc.)
- Numerical accuracy matching MATLAB
- Good performance
- Reliable, well-tested code

### What exists today:
- **Slycot:** Full functionality, battle-tested, but GPL + installation nightmare
- **SciPy fallback:** Easy to install, BSD, but missing advanced control functions
- **slicot C11:** Easy to install, BSD, 600+ routines, but untrusted by python-control maintainers

### Unresolved Issues
1. **Naming:** Can the C11 translation use the "SLICOT" name? No explicit permission from SLICOT Team/NICONET e.V.
2. **API design:** Should a replacement match Slycot's API exactly, or design a clean Python-first interface?
3. **Trust:** How to establish confidence in AI-translated numerical code?
4. **Performance:** C11 currently ~2x slower than Fortran - acceptable tradeoff for easier installation?
5. **Governance:** Who decides when/if python-control switches backends?

---

## Sources

### Repositories
- [python-control/python-control](https://github.com/python-control/python-control) - 1,962 stars
- [python-control/Slycot](https://github.com/python-control/Slycot) - 148 stars
- [jamestjsp/slicot](https://github.com/jamestjsp/slicot) - C11 translation
- [SLICOT/SLICOT-Reference](https://github.com/SLICOT/SLICOT-Reference) - Official Fortran reference

### Key Issues and PRs
- [python-control #27](https://github.com/python-control/python-control/issues/27) - "Move away from SLICOT?" (2014, still open)
- [python-control #850](https://github.com/python-control/python-control/issues/850) - Installation instructions (2023)
- [python-control #1057](https://github.com/python-control/python-control/issues/1057) - GPL license concern (2024)
- [python-control #1200](https://github.com/python-control/python-control/pull/1200) - "Replace slycot with slicot" (2026, closed)
- [Slycot #164](https://github.com/python-control/Slycot/issues/164) - Build wheels in CI (2021, open)
- [Slycot #146](https://github.com/python-control/Slycot/issues/146) - SLICOT 5.7 BSD license (2020)

### Publications
- SLICOT: NICONET Report 97-3 (1997)
- python-control: 60th IEEE CDC (2021) - Fuller, Greiner, Moore, Murray, van Paassen, Yorke
- [SLICOT.org](https://www.slicot.org/) - Official SLICOT website
