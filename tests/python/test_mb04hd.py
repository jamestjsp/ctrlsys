"""Tests for MB04HD - Reducing skew-Hamiltonian/Hamiltonian pencil to generalized Schur form."""

import numpy as np
import pytest

from fortran_reference import run_fortran_driver


MB04HD_I_TRANSFORM_CASE = r"""
program main
  implicit none
  integer, parameter :: n=4, lda=n, ldb=n, ldq1=n, ldq2=n, liwork=32, ldwork=2*n*n+272
  integer info
  integer iwork(liwork)
  logical bwork(n/2)
  double precision a(lda,n), b(ldb,n), q1(ldq1,n), q2(ldq2,n), dwork(ldwork)

  a=0.0d0
  b=0.0d0
  q1=0.0d0
  q2=0.0d0

  a(1,1)=1.40d0
  a(1,2)=-0.20d0
  a(2,2)=0.90d0
  a(3,3)=-0.70d0
  a(3,4)=0.25d0
  a(4,4)=1.10d0

  b(1,3)=0.50d0
  b(1,4)=-0.10d0
  b(2,4)=0.70d0
  b(3,1)=-0.30d0
  b(3,2)=0.20d0
  b(4,1)=0.15d0
  b(4,2)=0.90d0

  call MB04HD('I','I',n,a,lda,b,ldb,q1,ldq1,q2,ldq2,iwork,liwork,dwork,ldwork,bwork,info)

  print '(I0)', info
  print '(*(ES24.16,1X))', a
  print '(*(ES24.16,1X))', b
  print '(*(ES24.16,1X))', q1
  print '(*(ES24.16,1X))', q2
end program main
"""


def _mb04hd_reference_inputs():
    n = 4
    a = np.zeros((n, n), order='F')
    b = np.zeros((n, n), order='F')

    a[0, 0] = 1.40
    a[0, 1] = -0.20
    a[1, 1] = 0.90
    a[2, 2] = -0.70
    a[2, 3] = 0.25
    a[3, 3] = 1.10

    b[0, 2] = 0.50
    b[0, 3] = -0.10
    b[1, 3] = 0.70
    b[2, 0] = -0.30
    b[2, 1] = 0.20
    b[3, 0] = 0.15
    b[3, 1] = 0.90
    return a, b


def _take_matrix(values, offset, shape):
    count = shape[0] * shape[1]
    matrix = np.array(values[offset:offset + count]).reshape(shape, order='F')
    return matrix, offset + count


def test_mb04hd_i_transform_matches_fortran_reference(tmp_path):
    """Compare COMPQ1/COMPQ2='I' outputs with the original SLICOT MB04HD."""
    from ctrlsys import mb04hd

    a, b = _mb04hd_reference_inputs()
    a_out, b_out, q1, q2, info = mb04hd('I', 'I', a, b)

    output = run_fortran_driver(MB04HD_I_TRANSFORM_CASE, tmp_path)
    tokens = output.split()
    assert info == int(tokens[0])

    values = np.array(tokens[1:], dtype=float)
    expected_a, offset = _take_matrix(values, 0, (4, 4))
    expected_b, offset = _take_matrix(values, offset, (4, 4))
    expected_q1, offset = _take_matrix(values, offset, (4, 4))
    expected_q2, offset = _take_matrix(values, offset, (4, 4))
    assert offset == len(values)

    np.testing.assert_allclose(a_out, expected_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b_out, expected_b, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q1, expected_q1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q2, expected_q2, rtol=1e-12, atol=1e-12)


def test_mb04hd_basic():
    """
    Test MB04HD with COMPQ1='I', COMPQ2='I'.

    Tests transformation of block diagonal/anti-diagonal pencil to generalized Schur form.
    The input pencil has structure:
      A = diag(A11, A22) with A11, A22 upper triangular
      B = anti-diag(B21, B12) with B12 upper triangular, B21 upper quasi-triangular

    N must be even (N >= 0).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04hd

    np.random.seed(42)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b12 = np.triu(np.random.randn(m, m))
    b21 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, m:] = b12
    b[m:, :m] = b21

    result = mb04hd('I', 'I', a, b)
    a_out, b_out, q1, q2, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    assert a_out.shape == (n, n)
    assert b_out.shape == (n, n)
    assert q1.shape == (n, n)
    assert q2.shape == (n, n)


def test_mb04hd_orthogonality():
    """
    Validate Q1 and Q2 are orthogonal matrices.

    For COMPQ1='I', COMPQ2='I', the transformation matrices
    Q1 and Q2 must satisfy Q'Q = QQ' = I.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04hd

    np.random.seed(123)
    n = 6
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b12 = np.triu(np.random.randn(m, m))
    b21 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, m:] = b12
    b[m:, :m] = b21

    result = mb04hd('I', 'I', a, b)
    a_out, b_out, q1, q2, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q1.T @ q1, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q2.T @ q2, np.eye(n), rtol=1e-12, atol=1e-12)


def test_mb04hd_upper_triangular_a():
    """
    Validate transformed A is upper triangular.

    After transformation, Q2' A Q1 should be upper triangular.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04hd

    np.random.seed(456)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b12 = np.triu(np.random.randn(m, m))
    b21 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, m:] = b12
    b[m:, :m] = b21

    result = mb04hd('I', 'I', a, b)
    a_out, b_out, q1, q2, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(np.tril(a_out, -1), 0, atol=1e-10)


def test_mb04hd_n_zero():
    """Test MB04HD with N=0 (quick return)."""
    from ctrlsys import mb04hd

    a = np.array([], dtype=float, order='F').reshape(0, 0)
    b = np.array([], dtype=float, order='F').reshape(0, 0)

    result = mb04hd('N', 'N', a, b)
    a_out, b_out, q1, q2, info = result

    assert info == 0


def test_mb04hd_no_orthogonal_matrices():
    """
    Test MB04HD with COMPQ1='N', COMPQ2='N'.

    Q1 and Q2 are not computed.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04hd

    np.random.seed(789)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b12 = np.triu(np.random.randn(m, m))
    b21 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, m:] = b12
    b[m:, :m] = b21

    result = mb04hd('N', 'N', a, b)
    a_out, b_out, q1, q2, info = result

    assert info == 0 or info in [1, 2, 3, 4]
    assert a_out.shape == (n, n)
    assert b_out.shape == (n, n)


def test_mb04hd_small_n2():
    """
    Test MB04HD with N=2 (smallest even case).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04hd

    np.random.seed(111)
    n = 2
    m = n // 2

    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 1.5
    a[1, 1] = 2.5

    b = np.zeros((n, n), order='F', dtype=float)
    b[0, 1] = 0.3
    b[1, 0] = 0.4

    result = mb04hd('I', 'I', a, b)
    a_out, b_out, q1, q2, info = result

    assert info == 0 or info in [1, 2, 3, 4]
    assert a_out.shape == (n, n)
    assert b_out.shape == (n, n)


def test_mb04hd_update_mode():
    """
    Test MB04HD with COMPQ1='U', COMPQ2='U' (update mode).

    When update mode, initial orthogonal matrices are provided and multiplied
    by the transformation matrices.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb04hd

    np.random.seed(222)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b12 = np.triu(np.random.randn(m, m))
    b21 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, m:] = b12
    b[m:, :m] = b21

    q1_init = np.eye(n, dtype=float, order='F')
    q2_init = np.eye(n, dtype=float, order='F')

    result = mb04hd('U', 'U', a, b, q1=q1_init.copy(), q2=q2_init.copy())
    a_out, b_out, q1, q2, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-12, atol=1e-12)


def test_mb04hd_transformation_property():
    """
    Validate transformation: Q2' * A_in * Q1 = A_out and Q2' * B_in * Q1 = B_out.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb04hd

    np.random.seed(333)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a_in = np.zeros((n, n), order='F', dtype=float)
    a_in[:m, :m] = a11
    a_in[m:, m:] = a22
    a_copy = a_in.copy()

    b12 = np.triu(np.random.randn(m, m))
    b21 = np.triu(np.random.randn(m, m))
    b_in = np.zeros((n, n), order='F', dtype=float)
    b_in[:m, m:] = b12
    b_in[m:, :m] = b21
    b_copy = b_in.copy()

    result = mb04hd('I', 'I', a_in, b_in)
    a_out, b_out, q1, q2, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        a_transformed = q2.T @ a_copy @ q1
        np.testing.assert_allclose(a_transformed, a_out, rtol=1e-10, atol=1e-10)

        b_transformed = q2.T @ b_copy @ q1
        np.testing.assert_allclose(b_transformed, b_out, rtol=1e-10, atol=1e-10)


def test_mb04hd_larger_n():
    """
    Test MB04HD with larger N=8.

    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb04hd

    np.random.seed(444)
    n = 8
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b12 = np.triu(np.random.randn(m, m))
    b21 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, m:] = b12
    b[m:, :m] = b21

    result = mb04hd('I', 'I', a, b)
    a_out, b_out, q1, q2, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.tril(a_out, -1), 0, atol=1e-10)


def test_mb04hd_larger_n_reorder_transformation_property():
    """Exercise the M > 1 reorder path where later logical workspace entries matter."""
    from ctrlsys import mb04hd

    np.random.seed(0)
    n = 8
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a_in = np.zeros((n, n), order='F', dtype=float)
    a_in[:m, :m] = a11
    a_in[m:, m:] = a22
    a_copy = a_in.copy()

    b12 = np.triu(np.random.randn(m, m))
    b21 = np.triu(np.random.randn(m, m))
    b_in = np.zeros((n, n), order='F', dtype=float)
    b_in[:m, m:] = b12
    b_in[m:, :m] = b21
    b_copy = b_in.copy()

    a_out, b_out, q1, q2, info = mb04hd('I', 'I', a_in, b_in)

    assert info == 0
    assert np.linalg.norm(q1 - np.eye(n)) > 1e-8
    assert np.linalg.norm(q2 - np.eye(n)) > 1e-8
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(q2.T @ a_copy @ q1, a_out, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q2.T @ b_copy @ q1, b_out, rtol=1e-10, atol=1e-10)
