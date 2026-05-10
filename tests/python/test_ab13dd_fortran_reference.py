"""Differential tests for AB13DD against the SLICOT Fortran reference."""

from __future__ import annotations

import numpy as np

from fortran_reference import run_fortran_driver


AB13DD_CONTINUOUS_SORTED_FREQUENCY_CASE = r"""
program main
  implicit none
  integer, parameter :: n = 6, m = 1, p = 1
  integer, parameter :: lda = 6, lde = 6, ldb = 6, ldc = 1, ldd = 1
  integer, parameter :: liwork = 10000, ldwork = 200000, lcwork = 200000
  character dico, jobe, equil, jobd
  integer info
  integer iwork(liwork)
  double precision a(lda,n), e(lde,n), b(ldb,m), c(ldc,n), d(ldd,m)
  double precision fpeak(2), gpeak(2), tol, dwork(ldwork)
  double complex cwork(lcwork)

  dico = 'C'
  jobe = 'I'
  equil = 'N'
  jobd = 'D'
  tol = 1.0d-9
  info = -99

  a = 0.0d0
  a(1,2) = 1.0d0
  a(2,1) = -0.5d0
  a(2,2) = -0.0002d0
  a(3,4) = 1.0d0
  a(4,3) = -1.0d0
  a(4,4) = -0.00002d0
  a(5,6) = 1.0d0
  a(6,5) = -2.0d0
  a(6,6) = -0.000002d0

  e = 0.0d0
  e(1,1) = 1.0d0
  e(2,2) = 1.0d0
  e(3,3) = 1.0d0
  e(4,4) = 1.0d0
  e(5,5) = 1.0d0
  e(6,6) = 1.0d0

  b = 0.0d0
  b(1,1) = 1.0d0
  b(3,1) = 1.0d0
  b(5,1) = 1.0d0

  c = 0.0d0
  c(1,1) = 1.0d0
  c(1,3) = 1.0d0
  c(1,5) = 1.0d0

  d(1,1) = 0.0d0
  fpeak(1) = 0.0d0
  fpeak(2) = 1.0d0
  gpeak(1) = 0.0d0
  gpeak(2) = 1.0d0

  call AB13DD(dico, jobe, equil, jobd, n, m, p, fpeak, a, lda, e, lde, &
       b, ldb, c, ldc, d, ldd, gpeak, tol, iwork, dwork, ldwork, &
       cwork, lcwork, info)

  print '(I0,1X,ES24.16,1X,ES24.16,1X,ES24.16,1X,ES24.16)', &
       info, gpeak(1), gpeak(2), fpeak(1), fpeak(2)
end program main
"""


def test_ab13dd_continuous_sorted_frequency_candidates_match_fortran(tmp_path):
    from ctrlsys import ab13dd

    output = run_fortran_driver(AB13DD_CONTINUOUS_SORTED_FREQUENCY_CASE, tmp_path)
    info_f, g0_f, g1_f, f0_f, f1_f = output.split()

    n, m, p = 6, 1, 1
    a = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5, -0.0002, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -0.00002, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -2.0, -0.000002],
    ], order="F", dtype=float)
    e = np.eye(n, order="F", dtype=float)
    b = np.array([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]], order="F", dtype=float)
    c = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], order="F", dtype=float)
    d = np.array([[0.0]], order="F", dtype=float)
    fpeak = np.array([0.0, 1.0], order="F", dtype=float)

    gpeak, fpeak_out, info = ab13dd(
        "C", "I", "N", "D", n, m, p, fpeak, a, e, b, c, d, 1e-9
    )

    assert info == int(info_f)
    np.testing.assert_allclose(gpeak, [float(g0_f), float(g1_f)], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fpeak_out, [float(f0_f), float(f1_f)], rtol=1e-10, atol=1e-10)
