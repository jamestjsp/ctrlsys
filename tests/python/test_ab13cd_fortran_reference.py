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
  print '(I0,1X,ES26.17E3)', info, hnorm
end program main
"""


DATA = Path(__file__).parent / "data" / "ab13cd_fortran_reference.npz"

# Pre-DLAPY2 C (hypot) returned exactly half the norm for half_norm_stall and
# was off by 1.5% for lightly_damped_mode.
CASES = ["half_norm_stall", "lightly_damped_mode"]


@pytest.mark.parametrize("name", CASES)
def test_ab13cd_hnorm_matches_fortran(name, tmp_path):
    from ctrlsys import ab13cd

    with np.load(DATA) as data:
        a, b, c, d = (data[f"{name}_{k}"] for k in "abcd")
    info_f, hnorm_f = run_fortran_driver(_ab13cd_driver(a, b, c, d), tmp_path).split()

    n, m = b.shape
    p = c.shape[0]
    hnorm, _, info = ab13cd(
        n, m, p, np.asfortranarray(a), np.asfortranarray(b),
        np.asfortranarray(c), np.asfortranarray(d), 1e-9,
    )

    assert info == int(info_f) == 0
    np.testing.assert_allclose(hnorm, float(hnorm_f), rtol=1e-10, atol=0)
