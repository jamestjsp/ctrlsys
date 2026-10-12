"""Differential tests for AB13CD against the SLICOT Fortran reference."""

from __future__ import annotations

import numpy as np
import pytest

from fortran_reference import run_fortran_driver


def _fortran_matrix(name: str, x: np.ndarray) -> str:
    values = ", &\n       ".join(
        repr(float(v)).replace("e", "d") + ("" if "e" in repr(float(v)) else "d0")
        for v in x.ravel(order="F")
    )
    return f"  {name} = reshape([ &\n       {values}], [{x.shape[0]}, {x.shape[1]}])\n"


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

{_fortran_matrix("a", a)}{_fortran_matrix("b", b)}{_fortran_matrix("c", c)}{_fortran_matrix("d", d)}
  hnorm = AB13CD(n, m, p, a, n, b, n, c, p, d, p, 1.0d-9, iwork, dwork, &
       ldwork, cwork, lcwork, bwork, info)
  print '(I0,1X,ES26.17E3)', info, hnorm
end program main
"""


# Pre-DLAPY2 C (hypot) returned exactly half the norm for the first case and
# was off by 1.5% for the second.
CASES = {
    "half_norm_stall": (
        np.array([
            [-2.4806072299368775, -2.407212482054229, 0.8018213545600507, 0.4260616374782699],
            [-1.055675857029951, -3.6520640088637113, -0.09929006930764216, 1.7061105352508932],
            [-0.5569396445768086, -0.7888444337019473, -2.074598955071876, -0.6659977766106081],
            [-1.0753599718641065, -0.16526639527322654, 0.8008859350425591, -2.8898165815984473],
        ]),
        np.array([[1.4182540728247504], [1.3720966923682074], [-1.6375198743585355], [-1.1911231447012907]]),
        np.array([[-0.9543702008875902, -0.2462614906767239, 0.14201339999900708, -0.1058324984281836]]),
        np.array([[0.04624856015859313]]),
    ),
    "lightly_damped_mode": (
        np.array([[0.0, 1.0], [-50.93481311320884, -0.44149932291655924]]),
        np.array([[0.32779564692010765], [1.8853016298620688]]),
        np.array([[-0.06520236907632664, 0.3067134673568813], [1.1453540836655265, -0.4273319589531791]]),
        np.array([[-0.5825483609114824], [0.2835565871203473]]),
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_ab13cd_hnorm_matches_fortran(name, tmp_path):
    from ctrlsys import ab13cd

    a, b, c, d = CASES[name]
    info_f, hnorm_f = run_fortran_driver(_ab13cd_driver(a, b, c, d), tmp_path).split()

    n, m = b.shape
    p = c.shape[0]
    hnorm, _, info = ab13cd(
        n, m, p, np.asfortranarray(a), np.asfortranarray(b),
        np.asfortranarray(c), np.asfortranarray(d), 1e-9,
    )

    assert info == int(info_f) == 0
    np.testing.assert_allclose(hnorm, float(hnorm_f), rtol=1e-10, atol=0)
