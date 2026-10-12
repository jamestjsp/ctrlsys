"""Differential tests for AB13CD against the SLICOT Fortran reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fortran_reference import fortran_matrix, run_fortran_driver


def _ab13cd_driver(a, b, c, d) -> str:
    n, m = b.shape
    p = c.shape[0]
    return f"""
program main
  implicit none
  integer, parameter :: n = {n}, m = {m}, p = {p}
  integer, parameter :: ldwork = 200000, lcwork = 200000
  integer info
  integer iwork(n)
  logical bwork(2*n)
  double precision a(n,n), b(n,m), c(p,n), d(p,m), dwork(ldwork), hnorm
  double complex cwork(lcwork)
  double precision AB13CD
  external AB13CD

{fortran_matrix("a", a)}{fortran_matrix("b", b)}{fortran_matrix("c", c)}{fortran_matrix("d", d)}
  hnorm = AB13CD(n, m, p, a, n, b, n, c, p, d, p, 1.0d-9, iwork, dwork, &
       ldwork, cwork, lcwork, bwork, info)
  print '(I0,1X,ES26.17E3,1X,ES26.17E3)', info, hnorm, dwork(2)
end program main
"""


def _run(a, b, c, d, tmp_path):
    from ctrlsys import ab13cd

    info_f, hnorm_f, fpeak_f = run_fortran_driver(
        _ab13cd_driver(a, b, c, d), tmp_path
    ).split()
    n, m = b.shape
    p = c.shape[0]
    hnorm, fpeak, info = ab13cd(
        n, m, p, *(np.asfortranarray(x, dtype=float) for x in (a, b, c, d)), 1e-9
    )
    return (int(info_f), float(hnorm_f), float(fpeak_f)), (info, hnorm, fpeak)


DATA = Path(__file__).parent / "data" / "ab13cd_fortran_reference.npz"

# Pre-DLAPY2 C (hypot) returned exactly half the norm for half_norm_stall and
# was off by 1.5% for lightly_damped_mode.
CASES = ["half_norm_stall", "lightly_damped_mode"]


@pytest.mark.parametrize("name", CASES)
def test_ab13cd_hnorm_matches_fortran(name, tmp_path):
    with np.load(DATA) as data:
        a, b, c, d = (data[f"{name}_{k}"] for k in "abcd")
    (info_f, hnorm_f, _), (info, hnorm, _) = _run(a, b, c, d, tmp_path)

    assert info == info_f == 0
    np.testing.assert_allclose(hnorm, hnorm_f, rtol=1e-10, atol=0)


def _oscillator(zeta):
    a = np.array([[0.0, 1.0], [-1.0, -2.0 * zeta]])
    b = np.array([[0.0], [1.0]])
    c = np.array([[1.0, 0.0]])
    d = np.array([[0.0]])
    return a, b, c, d


def _mimo(seed):
    rng = np.random.default_rng(seed)
    n, m, p = 5, 2, 3
    a = rng.standard_normal((n, n))
    a -= (np.max(np.linalg.eigvals(a).real) + 0.5) * np.eye(n)
    return a, rng.standard_normal((n, m)), rng.standard_normal((p, n)), rng.standard_normal((p, m))


# AB13CD stops when no Hamiltonian eigenvalue satisfies SB02CX's absolute
# |Re| < 100*eps test, so cases whose on-axis eigenvalues carry ~1e-14 rounding
# noise can legitimately flip between compilers/BLAS (e.g. clang FMA
# contraction). These cases keep that noise well below the threshold.
@pytest.mark.parametrize(
    "system",
    [_oscillator(0.1), _oscillator(0.3), _mimo(5)],
    ids=["oscillator-0.1", "oscillator-0.3", "mimo-5"],
)
def test_ab13cd_matches_fortran(tmp_path, system):
    (info_f, hnorm_f, fpeak_f), (info, hnorm, fpeak) = _run(*system, tmp_path)

    assert info == info_f
    np.testing.assert_allclose(hnorm, hnorm_f, rtol=1e-10)
    np.testing.assert_allclose(fpeak, fpeak_f, rtol=1e-8)
