"""Tests for MB03OD - Incremental condition/rank estimation for QR"""
import numpy as np
import pytest
from ctrlsys import mb03od


def test_mb03od_basic_example():
    """Test MB03OD with HTML doc example - rank deficient matrix"""
    # HTML doc example: M=6, N=5, JOBQR='Q', RCOND=5e-16, SVLMAX=0.0
    # READ format: ((A(I,J), J=1,N), I=1,M) - row-wise
    m, n = 6, 5
    a = np.array([
        [ 1.,  2.,  6.,  3.,  5.],
        [-2., -1., -1.,  0., -2.],
        [ 5.,  5.,  1.,  5.,  1.],
        [-2., -1., -1.,  0., -2.],
        [ 4.,  8.,  4., 20.,  4.],
        [-2., -1., -1.,  0., -2.]
    ], dtype=float, order='F')

    rcond = 5e-16
    svlmax = 0.0

    # Expected results
    expected_rank = 4
    expected_jpvt = np.array([4, 3, 1, 5, 2], dtype=np.int32)
    expected_sval = np.array([22.7257, 1.4330, 0.0])

    jpvt, rank, sval, info = mb03od(m, n, a, rcond, svlmax, jobqr='Q')

    assert info == 0, f"mb03od failed with info={info}"
    assert rank == expected_rank, f"Expected rank {expected_rank}, got {rank}"
    np.testing.assert_array_equal(jpvt, expected_jpvt)
    np.testing.assert_allclose(sval, expected_sval, rtol=1e-3, atol=1e-14)


def test_mb03od_full_rank():
    """Test MB03OD with full rank matrix"""
    m, n = 4, 3
    # Create full rank matrix
    a = np.array([
        [1., 0., 0.],
        [0., 2., 0.],
        [0., 0., 3.],
        [1., 1., 1.]
    ], dtype=float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    jpvt, rank, sval, info = mb03od(m, n, a, rcond, svlmax, jobqr='Q')

    assert info == 0
    assert rank == n  # Full rank
    assert sval[0] > 0  # Largest singular value positive
    assert sval[1] > 0  # Smallest of R11 positive


def test_mb03od_zero_matrix():
    """Test MB03OD with zero matrix"""
    m, n = 3, 3
    a = np.zeros((m, n), dtype=float, order='F')
    rcond = 1e-10
    svlmax = 0.0

    jpvt, rank, sval, info = mb03od(m, n, a, rcond, svlmax, jobqr='Q')

    assert info == 0
    assert rank == 0  # Zero rank
    np.testing.assert_array_equal(sval, [0., 0., 0.])


def test_mb03od_no_qr():
    """Test MB03OD with JOBQR='N' - precomputed QR"""
    m, n = 3, 3
    # Use identity as R (already upper triangular)
    a = np.eye(m, n, dtype=float, order='F')
    rcond = 1e-10
    svlmax = 0.0

    # JOBQR='N' assumes A already contains R from QR factorization
    jpvt, rank, sval, info = mb03od(m, n, a, rcond, svlmax, jobqr='N')

    assert info == 0
    assert rank == n  # Full rank identity


def test_mb03od_svlmax_effect():
    """Test MB03OD with SVLMAX > 0 - relative rank decision"""
    m, n = 4, 3
    a = np.array([
        [10., 0., 0.],
        [ 0., 5., 0.],
        [ 0., 0., 1.],
        [ 1., 1., 1.]
    ], dtype=float, order='F')

    rcond = 0.1

    # With SVLMAX=0, rank based on R only
    jpvt1, rank1, sval1, info1 = mb03od(m, n, a.copy(order='F'), rcond, 0.0, jobqr='Q')
    assert info1 == 0

    # With SVLMAX=100, may reduce rank (stricter threshold)
    jpvt2, rank2, sval2, info2 = mb03od(m, n, a.copy(order='F'), rcond, 100.0, jobqr='Q')
    assert info2 == 0
    assert rank2 <= rank1  # SVLMAX can only reduce rank


def test_mb03od_empty_matrix():
    """Test MB03OD with zero dimensions - quick return"""
    # M=0
    a = np.empty((0, 3), dtype=float, order='F')
    jpvt, rank, sval, info = mb03od(0, 3, a, 1e-10, 0.0, jobqr='Q')
    assert info == 0
    assert rank == 0
    np.testing.assert_array_equal(sval, [0., 0., 0.])

    # N=0
    a = np.empty((3, 0), dtype=float, order='F')
    jpvt, rank, sval, info = mb03od(3, 0, a, 1e-10, 0.0, jobqr='Q')
    assert info == 0
    assert rank == 0
    np.testing.assert_array_equal(sval, [0., 0., 0.])


def test_mb03od_invalid_params():
    """Test MB03OD parameter validation"""
    m, n = 3, 3
    a = np.eye(m, n, dtype=float, order='F')

    # Invalid RCOND < 0
    jpvt, rank, sval, info = mb03od(m, n, a, -0.1, 0.0, jobqr='Q')
    assert info == -7

    # Invalid SVLMAX < 0
    jpvt, rank, sval, info = mb03od(m, n, a, 1e-10, -1.0, jobqr='Q')
    assert info == -8
