"""Differential tests for AB13HD against the SLICOT Fortran reference."""

from __future__ import annotations

import numpy as np

from fortran_reference import run_fortran_driver


AB13HD_DISCRETE_ROW_CASE = r"""
program main
  implicit none
  integer, parameter :: n = 2, m = 1, p = 2
  integer, parameter :: lda = 2, lde = 2, ldb = 2, ldc = 2, ldd = 2
  integer, parameter :: liwork = 10000, ldwork = 200000, lzwork = 200000
  character dico, jobe, equil, jobd, ckprop, reduce, poles
  integer ranke, nr, iwarn, info
  integer iwork(liwork)
  double precision a(lda,n), e(lde,n), b(ldb,m), c(ldc,n), d(ldd,m)
  double precision fpeak(2), gpeak(2), tol(2), dwork(ldwork)
  double complex zwork(lzwork)
  logical bwork(liwork)

  dico = 'D'
  jobe = 'I'
  equil = 'N'
  jobd = 'D'
  ckprop = 'N'
  reduce = 'N'
  poles = 'A'
  ranke = 0
  nr = -1
  iwarn = -99
  info = -99

  a = 0.0d0
  a(1,1) = 0.50d0
  a(2,2) = 0.30d0
  e = 0.0d0
  e(1,1) = 1.0d0
  e(2,2) = 1.0d0
  b(1,1) = 1.0d0
  b(2,1) = 0.5d0
  c(1,1) = 1.0d0
  c(2,1) = 10.0d0
  c(1,2) = 2.0d0
  c(2,2) = -3.0d0
  d(1,1) = 0.1d0
  d(2,1) = -0.2d0
  fpeak(1) = 0.0d0
  fpeak(2) = 1.0d0
  tol(1) = 1.0d-9
  tol(2) = -1.0d0

  call AB13HD(dico, jobe, equil, jobd, ckprop, reduce, poles, n, m, p, &
       ranke, fpeak, a, lda, e, lde, b, ldb, c, ldc, d, ldd, nr, &
       gpeak, tol, iwork, dwork, ldwork, zwork, lzwork, bwork, iwarn, info)

  print '(I0,1X,I0,1X,I0,1X,ES24.16,1X,ES24.16,1X,ES24.16,1X,ES24.16)', &
       info, iwarn, nr, gpeak(1), gpeak(2), fpeak(1), fpeak(2)
end program main
"""


def test_ab13hd_discrete_mimo_matches_fortran_reference(tmp_path):
    """Catches regressions where Fortran C(I,1),LDC is translated as a column walk."""
    from ctrlsys import ab13hd

    output = run_fortran_driver(AB13HD_DISCRETE_ROW_CASE, tmp_path)
    info_f, iwarn_f, nr_f, g0_f, g1_f, f0_f, f1_f = output.split()

    n, m, p = 2, 1, 2
    a = np.array([[0.5, 0.0], [0.0, 0.3]], order="F", dtype=float)
    e = np.eye(n, order="F", dtype=float)
    b = np.array([[1.0], [0.5]], order="F", dtype=float)
    c = np.array([[1.0, 2.0], [10.0, -3.0]], order="F", dtype=float)
    d = np.array([[0.1], [-0.2]], order="F", dtype=float)
    fpeak = np.array([0.0, 1.0], order="F", dtype=float)
    tol = np.array([1e-9, -1.0], order="F", dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        "D", "I", "N", "D", "N", "N", "A",
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info == int(info_f)
    assert iwarn == int(iwarn_f)
    assert nr == int(nr_f)
    np.testing.assert_allclose(gpeak, [float(g0_f), float(g1_f)], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fpeak_out, [float(f0_f), float(f1_f)], rtol=1e-10, atol=1e-10)
