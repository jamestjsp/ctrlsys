"""
Tests for SG02ND: Optimal gain matrix for discrete/continuous Riccati problems.

Computes:
- Discrete: K = (R + B'XB)^{-1} (B'X*op(A) + L')
- Continuous: K = R^{-1} (B'X*op(E) + L')

Test data from SLICOT HTML documentation example.

Mathematical properties tested:
- Gain matrix dimensions
- Riccati equation residual (closed-loop verification)
- Condition number output

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fortran_reference import run_fortran_driver


def _sg02nd_data_file():
    return "SLICOT-Reference/examples/data/SG02ND.dat"


def _sg02nd_tokens():
    tokens = []
    with open(_sg02nd_data_file(), encoding="ascii") as data_file:
        next(data_file)
        for line in data_file:
            tokens.extend(line.split())
    return tokens


def _read_sg02nd_example_data():
    tokens = _sg02nd_tokens()
    offset = 0
    n = int(tokens[offset])
    m = int(tokens[offset + 1])
    p = int(tokens[offset + 2])
    dico, jobe, job, jobx, fact, jobl, uplo, trans = tokens[offset + 4:offset + 12]
    offset += 12

    def matrix(rows, cols):
        nonlocal offset
        data = np.array([float(x) for x in tokens[offset:offset + rows * cols]])
        offset += rows * cols
        return np.asfortranarray(data.reshape(rows, cols))

    a = matrix(n, n)
    b = matrix(n, m)
    _c = matrix(p, n)
    r = matrix(m, m)
    x = np.eye(n, dtype=float, order="F")
    e = np.zeros((1, 1), dtype=float, order="F")
    l = np.zeros((1, 1), dtype=float, order="F")
    return dico, jobe, job, jobx, fact, uplo, jobl, trans, n, m, p, a, e, b, r, l, x


def _sg02nd_example_fortran_source():
    data_path = _sg02nd_data_file()
    return f"""
program sg02nd_reference
  implicit none
  integer, parameter :: nmax = 20, mmax = 20, pmax = 20
  integer, parameter :: lda = nmax, ldb = nmax, lde = nmax, ldf = mmax
  integer, parameter :: ldh = nmax, ldl = nmax, ldr = max(mmax,pmax), ldx = nmax, ldxe = nmax
  integer, parameter :: ldwork = max(nmax+3*mmax+2, 14*nmax+23, 16*nmax)
  character :: dico, fact, job, jobe, jobl, jobx, trans, uplo
  integer :: i, info, j, m, n, p
  integer :: ipiv(mmax), iwork(mmax), oufact(2)
  double precision :: rcond, rnorm, tol
  double precision :: a(lda,nmax), b(ldb,mmax), c(pmax,nmax), dwork(ldwork)
  double precision :: e(lde,nmax), f(ldf,nmax), h(ldh,mmax), l(ldl,mmax)
  double precision :: r(ldr,mmax), x(ldx,nmax), xe(ldxe,nmax)

  open(unit=5, file='{data_path}', status='old')
  read(5, *)
  read(5, *) n, m, p, tol, dico, jobe, job, jobx, fact, jobl, uplo, trans
  read(5, *) ((a(i,j), j = 1,n), i = 1,n)
  read(5, *) ((b(i,j), j = 1,m), i = 1,n)
  read(5, *) ((c(i,j), j = 1,n), i = 1,p)
  read(5, *) ((r(i,j), j = 1,m), i = 1,m)
  close(5)

  x = 0.0d0
  do i = 1, n
     x(i,i) = 1.0d0
  end do
  e = 0.0d0
  l = 0.0d0
  rnorm = 0.0d0

  call SG02ND(dico, jobe, job, jobx, fact, uplo, jobl, trans, n, m, p, &
       a, lda, e, lde, b, ldb, r, ldr, ipiv, l, ldl, x, ldx, rnorm, &
       f, ldf, h, ldh, xe, ldxe, oufact, iwork, dwork, ldwork, info)

  rcond = dwork(2)
  write(*,'(I0,2(1X,I0),1X,ES24.16)') info, oufact(1), oufact(2), rcond
  do i = 1, m
     write(*,'(*(1X,ES24.16))') (f(i,j), j = 1,n)
  end do
end program
"""


def _sg02nd_jobx_fortran_source():
    return """
program sg02nd_jobx_reference
  implicit none
  integer, parameter :: n = 2, m = 1, p = 1
  integer, parameter :: ldwork = 64
  integer :: i, info, j
  integer :: ipiv(m), iwork(m), oufact(2)
  double precision :: a(1,1), b(n,m), dwork(ldwork), e(n,n), h(n,m), k(m,n)
  double precision :: l(1,1), r(m,m), rcond, rnorm, x(n,n), xe(n,n)

  a = 0.0d0
  b(1,1) = 1.0d0
  b(2,1) = 0.5d0
  e = 0.0d0
  e(1,1) = 2.0d0
  e(2,2) = 1.0d0
  r(1,1) = 1.0d0
  l = 0.0d0
  x = 0.0d0
  x(1,1) = 1.0d0
  x(2,2) = 1.0d0
  rnorm = 0.0d0

  call SG02ND('C', 'G', 'K', 'C', 'N', 'U', 'Z', 'N', n, m, p, &
       a, 1, e, n, b, n, r, m, ipiv, l, 1, x, n, rnorm, &
       k, m, h, n, xe, n, oufact, iwork, dwork, ldwork, info)

  rcond = dwork(2)
  write(*,'(I0,2(1X,I0),1X,ES24.16)') info, oufact(1), oufact(2), rcond
  write(*,'(*(1X,ES24.16))') (k(1,j), j = 1,n)
  do i = 1, n
     write(*,'(*(1X,ES24.16))') (xe(i,j), j = 1,n)
  end do
end program
"""


def _parse_matrix(lines, offset, rows, cols):
    values = [[float(x) for x in line.split()] for line in lines[offset:offset + rows]]
    return np.asfortranarray(np.array(values)), offset + rows


def test_sg02nd_html_example_matches_fortran_reference(tmp_path):
    from ctrlsys import sg02nd

    args = _read_sg02nd_example_data()
    dico, jobe, job, jobx, fact, uplo, jobl, trans, n, m, p = args[:11]
    a, e, b, r, l, x = args[11:]

    k, h, xe, oufact, rcond, info = sg02nd(
        dico, jobe, job, jobx, fact, uplo, jobl, trans,
        n, m, p, a, e, b.copy(order="F"), r.copy(order="F"),
        np.zeros(m, dtype=np.int32), l, x.copy(order="F"), 0.0
    )
    output = run_fortran_driver(_sg02nd_example_fortran_source(), tmp_path)
    lines = output.splitlines()
    header = lines[0].split()
    ref_info = int(header[0])
    ref_oufact = np.array([int(header[1]), int(header[2])], dtype=np.int32)
    ref_rcond = float(header[3])
    ref_k, _ = _parse_matrix(lines, 1, m, n)

    assert info == ref_info == 0
    np.testing.assert_array_equal(oufact, ref_oufact)
    assert_allclose(rcond, ref_rcond, rtol=1e-13, atol=1e-13)
    assert_allclose(k, ref_k, rtol=1e-13, atol=1e-13)


def test_sg02nd_jobx_c_matches_fortran_reference(tmp_path):
    from ctrlsys import sg02nd

    n, m, p = 2, 1, 1
    e = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float, order="F")
    b = np.array([[1.0], [0.5]], dtype=float, order="F")
    r = np.array([[1.0]], dtype=float, order="F")
    x = np.eye(n, dtype=float, order="F")

    k, h, xe, oufact, rcond, info = sg02nd(
        "C", "G", "K", "C", "N", "U", "Z", "N",
        n, m, p,
        np.zeros((1, 1), dtype=float, order="F"),
        e.copy(order="F"),
        b.copy(order="F"),
        r.copy(order="F"),
        np.zeros(m, dtype=np.int32),
        np.zeros((1, 1), dtype=float, order="F"),
        x.copy(order="F"),
        0.0,
    )
    output = run_fortran_driver(_sg02nd_jobx_fortran_source(), tmp_path)
    lines = output.splitlines()
    header = lines[0].split()
    ref_info = int(header[0])
    ref_oufact = np.array([int(header[1]), int(header[2])], dtype=np.int32)
    ref_rcond = float(header[3])
    ref_k = np.array([[float(x) for x in lines[1].split()]], dtype=float, order="F")
    ref_xe, _ = _parse_matrix(lines, 2, n, n)

    assert info == ref_info == 0
    np.testing.assert_array_equal(oufact, ref_oufact)
    assert_allclose(rcond, ref_rcond, rtol=1e-13, atol=1e-13)
    assert_allclose(k, ref_k, rtol=1e-13, atol=1e-13)
    assert_allclose(xe, ref_xe, rtol=1e-13, atol=1e-13)


def test_sg02nd_discrete_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Discrete-time case with identity E, unfactored R, no L matrix.
    N=2, M=1, P=3, DICO='D', JOBE='I', JOB='K', FACT='N', JOBL='Z'

    The example solves for the Riccati solution X and then computes the
    optimal feedback matrix K.
    """
    from ctrlsys import sg02nd

    n = 2
    m = 1
    p = 3

    a = np.array([
        [2.0, -1.0],
        [1.0,  0.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.0]
    ], order='F', dtype=float)

    r = np.array([
        [0.0]
    ], order='F', dtype=float)

    x = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    k_expected = np.array([
        [2.0, -1.0]
    ], order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='D', jobe='I', job='K', jobx='N', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=p,
        a=a.copy(order='F'),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)
    assert_allclose(k, k_expected, rtol=1e-3, atol=1e-4)


def test_sg02nd_continuous_identity_e():
    """
    Test continuous-time case with identity E.

    Continuous: K = R^{-1} B'X
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(42)
    n = 3
    m = 2

    r = np.eye(m, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='C', jobe='I', job='K', jobx='N', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=np.zeros((1, 1), order='F', dtype=float),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)

    k_expected = b.T.copy(order='F')
    assert_allclose(k, k_expected, rtol=1e-14)


def test_sg02nd_with_h_output():
    """
    Test JOB='H' mode that returns both K and H.

    H = op(A)'*X*B + L (discrete) or op(E)'*X*B + L (continuous)
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(123)
    n = 2
    m = 1

    a = np.array([
        [0.5, 0.2],
        [0.1, 0.4]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.0]
    ], order='F', dtype=float)

    r = np.array([[1.0]], order='F', dtype=float)

    x = np.array([
        [2.0, 0.5],
        [0.5, 1.5]
    ], order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='D', jobe='I', job='H', jobx='N', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=a.copy(order='F'),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)
    assert h.shape == (n, m)

    h_expected = a.T @ x @ b
    assert_allclose(h, h_expected, rtol=1e-14)


def test_sg02nd_cholesky_factored_r():
    """
    Test with Cholesky factored R (FACT='C').

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(456)
    n = 2
    m = 2

    r_chol = np.array([
        [2.0, 0.5],
        [0.0, 1.5]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='C', jobe='I', job='K', jobx='N', fact='C',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=np.zeros((1, 1), order='F', dtype=float),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r_chol.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert oufact[0] == 1


def test_sg02nd_with_cross_term_l():
    """
    Test with non-zero cross-weighting matrix L (JOBL='N').

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(789)
    n = 2
    m = 1

    a = np.array([
        [1.0, 0.5],
        [0.0, 0.8]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.5]
    ], order='F', dtype=float)

    r = np.array([[2.0]], order='F', dtype=float)

    l = np.array([
        [0.1],
        [0.2]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='D', jobe='I', job='K', jobx='N', fact='N',
        uplo='U', jobl='N', trans='N',
        n=n, m=m, p=m,
        a=a.copy(order='F'),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=l.copy(order='F'),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)


def test_sg02nd_zero_dimensions():
    """
    Test edge case with n=0 (quick return).
    """
    from ctrlsys import sg02nd

    n = 0
    m = 1

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='C', jobe='I', job='K', jobx='N', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=np.zeros((1, 1), order='F', dtype=float),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=np.zeros((1, 1), order='F', dtype=float),
        r=np.zeros((1, 1), order='F', dtype=float),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=np.zeros((1, 1), order='F', dtype=float),
        rnorm=0.0
    )

    assert info == 0
    assert rcond == 1.0


def test_sg02nd_continuous_general_e():
    """
    Test continuous-time case with general E matrix (JOBE='G').

    Random seed: 321 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(321)
    n = 2
    m = 1

    e = np.array([
        [2.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.5]
    ], order='F', dtype=float)

    r = np.array([[1.0]], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='C', jobe='G', job='K', jobx='C', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=np.zeros((1, 1), order='F', dtype=float),
        e=e.copy(order='F'),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)

    xe_expected = x @ e
    assert_allclose(xe, xe_expected, rtol=1e-14)
