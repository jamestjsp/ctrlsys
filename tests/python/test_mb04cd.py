"""Tests for MB04CD - Reducing skew-Hamiltonian/Hamiltonian pencil to Schur form."""

import ctypes

import numpy as np
import pytest

from fortran_reference import run_fortran_driver


MB04CD_I_TRANSFORM_CASE = r"""
program main
  implicit none
  integer, parameter :: n = 4, lda = n, ldb = n, ldd = n
  integer, parameter :: ldq1 = n, ldq2 = n, ldq3 = n
  integer, parameter :: liwork = 48, ldwork = 3*n*n + 432
  integer info
  integer iwork(liwork)
  logical bwork(n/2)
  double precision a(lda,n), b(ldb,n), d(ldd,n)
  double precision q1(ldq1,n), q2(ldq2,n), q3(ldq3,n), dwork(ldwork)

  a = 0.0d0
  b = 0.0d0
  d = 0.0d0
  q1 = 0.0d0
  q2 = 0.0d0
  q3 = 0.0d0

  a(1,1) = 1.40d0
  a(1,2) = -0.20d0
  a(2,2) = 0.90d0
  a(3,3) = -0.70d0
  a(3,4) = 0.25d0
  a(4,4) = 1.10d0

  b(1,1) = 0.80d0
  b(1,2) = 0.35d0
  b(2,2) = 1.30d0
  b(3,3) = 1.20d0
  b(3,4) = -0.45d0
  b(4,4) = 0.60d0

  d(1,3) = 0.50d0
  d(1,4) = -0.10d0
  d(2,4) = 0.70d0
  d(3,1) = -0.30d0
  d(3,2) = 0.20d0
  d(4,1) = 0.15d0
  d(4,2) = 0.90d0

  call MB04CD('I', 'I', 'I', n, a, lda, b, ldb, d, ldd, &
       q1, ldq1, q2, ldq2, q3, ldq3, iwork, liwork, dwork, &
       ldwork, bwork, info)

  print '(I0)', info
  print '(*(ES24.16,1X))', a
  print '(*(ES24.16,1X))', b
  print '(*(ES24.16,1X))', d
  print '(*(ES24.16,1X))', q1
  print '(*(ES24.16,1X))', q2
  print '(*(ES24.16,1X))', q3
end program main
"""


def _mb04cd_reference_inputs():
    n = 4
    a = np.zeros((n, n), order="F", dtype=float)
    b = np.zeros((n, n), order="F", dtype=float)
    d = np.zeros((n, n), order="F", dtype=float)

    a[0, 0] = 1.40
    a[0, 1] = -0.20
    a[1, 1] = 0.90
    a[2, 2] = -0.70
    a[2, 3] = 0.25
    a[3, 3] = 1.10

    b[0, 0] = 0.80
    b[0, 1] = 0.35
    b[1, 1] = 1.30
    b[2, 2] = 1.20
    b[2, 3] = -0.45
    b[3, 3] = 0.60

    d[0, 2] = 0.50
    d[0, 3] = -0.10
    d[1, 3] = 0.70
    d[2, 0] = -0.30
    d[2, 1] = 0.20
    d[3, 0] = 0.15
    d[3, 1] = 0.90

    return a, b, d


def _take_matrix(values, offset, n):
    end = offset + n * n
    return values[offset:end].reshape((n, n), order="F"), end


def _mb04cd_raw_call(a, lda, b, ldb, d, ldd):
    from ctrlsys import _slicot

    n = 4
    m = n // 2
    liwork = 48
    ldwork = 3 * n * n + 432

    q1 = np.zeros((n, n), order="F", dtype=np.float64)
    q2 = np.zeros((n, n), order="F", dtype=np.float64)
    q3 = np.zeros((n, n), order="F", dtype=np.float64)
    iwork = np.zeros(liwork, dtype=np.int32)
    dwork = np.zeros(ldwork, dtype=np.float64)
    bwork = np.zeros(m, dtype=np.bool_)
    info = ctypes.c_int32(-999)

    lib = ctypes.CDLL(_slicot.__file__)
    lib.mb04cd.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.bool_, flags="C_CONTIGUOUS"),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.mb04cd.restype = None

    lib.mb04cd(
        b"I",
        b"I",
        b"I",
        n,
        a.ctypes.data_as(ctypes.c_void_p),
        lda,
        b.ctypes.data_as(ctypes.c_void_p),
        ldb,
        d.ctypes.data_as(ctypes.c_void_p),
        ldd,
        q1,
        n,
        q2,
        n,
        q3,
        n,
        iwork,
        liwork,
        dwork,
        ldwork,
        bwork,
        ctypes.byref(info),
    )
    return info.value, q1, q2, q3


def test_mb04cd_transformations_match_fortran_reference(tmp_path):
    from ctrlsys import mb04cd

    output = run_fortran_driver(MB04CD_I_TRANSFORM_CASE, tmp_path)
    tokens = output.split()
    info_f = int(tokens[0])
    values = np.array(tokens[1:], dtype=float)

    n = 4
    offset = 0
    a_f, offset = _take_matrix(values, offset, n)
    b_f, offset = _take_matrix(values, offset, n)
    d_f, offset = _take_matrix(values, offset, n)
    q1_f, offset = _take_matrix(values, offset, n)
    q2_f, offset = _take_matrix(values, offset, n)
    q3_f, offset = _take_matrix(values, offset, n)

    a, b, d = _mb04cd_reference_inputs()
    a_out, b_out, d_out, q1, q2, q3, info = mb04cd("I", "I", "I", a, b, d)

    assert info == info_f == 0
    assert offset == len(values)
    np.testing.assert_allclose(a_out, a_f, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b_out, b_f, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(d_out, d_f, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q1, q1_f, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q2, q2_f, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q3, q3_f, rtol=1e-10, atol=1e-10)


def test_mb04cd_recompute_outputs_with_padded_leading_dimensions():
    n = 4
    lda = n + 3
    ldb = n + 2
    ldd = n + 4

    a_ref, b_ref, d_ref = _mb04cd_reference_inputs()
    compact_a = a_ref.copy(order="F")
    compact_b = b_ref.copy(order="F")
    compact_d = d_ref.copy(order="F")

    padded_a = np.full((lda, n), 12345.0, order="F", dtype=np.float64)
    padded_b = np.full((ldb, n), -23456.0, order="F", dtype=np.float64)
    padded_d = np.full((ldd, n), 34567.0, order="F", dtype=np.float64)
    padded_a[:n, :] = a_ref
    padded_b[:n, :] = b_ref
    padded_d[:n, :] = d_ref

    info_c, q1_c, q2_c, q3_c = _mb04cd_raw_call(compact_a, n, compact_b, n, compact_d, n)
    info_p, q1_p, q2_p, q3_p = _mb04cd_raw_call(padded_a, lda, padded_b, ldb, padded_d, ldd)

    assert info_p == info_c == 0
    np.testing.assert_allclose(padded_a[:n, :], compact_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(padded_b[:n, :], compact_b, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(padded_d[:n, :], compact_d, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q1_p, q1_c, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q2_p, q2_c, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q3_p, q3_c, rtol=1e-12, atol=1e-12)


def test_mb04cd_basic():
    """
    Test MB04CD with COMPQ1='I', COMPQ2='I', COMPQ3='I'.

    Tests transformation of block diagonal pencil to generalized Schur form.
    N must be even (N >= 0).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(42)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21

    result = mb04cd('I', 'I', 'I', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0

    assert a_out.shape == (n, n)
    assert b_out.shape == (n, n)
    assert d_out.shape == (n, n)
    assert q1.shape == (n, n)
    assert q2.shape == (n, n)
    assert q3.shape == (n, n)


def test_mb04cd_orthogonality():
    """
    Validate Q1, Q2, Q3 are orthogonal matrices.

    For COMPQ1='I', COMPQ2='I', COMPQ3='I', the transformation matrices
    Q1, Q2, Q3 must satisfy Q'Q = QQ' = I.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(123)
    n = 6
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21

    result = mb04cd('I', 'I', 'I', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q3 @ q3.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q1.T @ q1, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q2.T @ q2, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q3.T @ q3, np.eye(n), rtol=1e-12, atol=1e-12)


def test_mb04cd_outputs_match_returned_transformations():
    """
    Validate transformed outputs are consistent with returned Q factors.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(456)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22
    a_in = a.copy()

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22
    b_in = b.copy()

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21
    d_in = d.copy()

    result = mb04cd('I', 'I', 'I', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(a_out, q3.T @ a_in @ q2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(b_out, q2.T @ b_in @ q1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d_out, q3.T @ d_in @ q1, rtol=1e-12, atol=1e-12)


def test_mb04cd_n_zero():
    """Test MB04CD with N=0 (quick return)."""
    from ctrlsys import mb04cd

    a = np.array([], dtype=float, order='F').reshape(0, 0)
    b = np.array([], dtype=float, order='F').reshape(0, 0)
    d = np.array([], dtype=float, order='F').reshape(0, 0)

    result = mb04cd('N', 'N', 'N', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0


def test_mb04cd_no_orthogonal_matrices():
    """
    Test MB04CD with COMPQ1='N', COMPQ2='N', COMPQ3='N'.

    Q1, Q2, Q3 are not computed.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(789)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21

    result = mb04cd('N', 'N', 'N', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]
    assert a_out.shape == (n, n)
    assert b_out.shape == (n, n)
    assert d_out.shape == (n, n)


def test_mb04cd_small_n2():
    """
    Test MB04CD with N=2 (smallest even case).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(111)
    n = 2
    m = n // 2

    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 1.5
    a[1, 1] = 2.5

    b = np.zeros((n, n), order='F', dtype=float)
    b[0, 0] = 0.8
    b[1, 1] = 1.2

    d = np.zeros((n, n), order='F', dtype=float)
    d[0, 1] = 0.3
    d[1, 0] = 0.4

    result = mb04cd('I', 'I', 'I', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]
    assert a_out.shape == (n, n)
    assert b_out.shape == (n, n)
    assert d_out.shape == (n, n)


def test_mb04cd_update_mode():
    """
    Test MB04CD with COMPQ1='U', COMPQ2='U', COMPQ3='U' (update mode).

    When update mode, initial orthogonal matrices are provided and multiplied
    by the transformation matrices.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(222)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21

    q1_init = np.eye(n, dtype=float, order='F')
    q2_init = np.eye(n, dtype=float, order='F')
    q3_init = np.eye(n, dtype=float, order='F')

    result = mb04cd('U', 'U', 'U', a, b, d,
                    q1=q1_init.copy(), q2=q2_init.copy(), q3=q3_init.copy())
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q3 @ q3.T, np.eye(n), rtol=1e-12, atol=1e-12)
