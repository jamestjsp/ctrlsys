"""Differential tests for AB13CD against the SLICOT Fortran reference."""

from __future__ import annotations

import numpy as np
import pytest

from fortran_reference import run_fortran_driver


def _real_assign(name, mat):
    return "\n".join(
        f"  {name}({i + 1},{j + 1}) = {v:.17e}".replace("e", "d")
        for (i, j), v in np.ndenumerate(mat)
    )


def _driver(a, b, c, d, tol):
    n, m = b.shape
    p = c.shape[0]
    tol_f = f"{tol:.17e}".replace("e", "d")
    return f"""
program main
  implicit none
  integer, parameter :: n = {n}, m = {m}, p = {p}
  integer, parameter :: ldwork = 100000, lcwork = 100000
  double precision :: a(n,n), b(n,m), c(p,n), d(p,m), dwork(ldwork), hnorm
  complex*16 :: cwork(lcwork)
  integer :: iwork(n), info
  logical :: bwork(2*n)
  double precision, external :: AB13CD

{_real_assign("a", a)}
{_real_assign("b", b)}
{_real_assign("c", c)}
{_real_assign("d", d)}

  hnorm = AB13CD(n, m, p, a, n, b, n, c, p, d, p, {tol_f}, iwork, dwork, &
       ldwork, cwork, lcwork, bwork, info)
  print '(I0,1X,ES26.17,1X,ES26.17)', info, hnorm, dwork(2)
end program main
"""


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
    from ctrlsys import ab13cd

    a, b, c, d = system
    n, m = b.shape
    p = c.shape[0]
    tol = 1e-9

    output = run_fortran_driver(_driver(a, b, c, d, tol), tmp_path)
    info_f, hnorm_f, fpeak_f = output.split()

    hnorm, fpeak, info = ab13cd(
        n, m, p, *(np.asfortranarray(x, dtype=float) for x in (a, b, c, d)), tol
    )

    assert info == int(info_f)
    np.testing.assert_allclose(hnorm, float(hnorm_f), rtol=1e-10)
    np.testing.assert_allclose(fpeak, float(fpeak_f), rtol=1e-8)
