#!/usr/bin/env python3
"""
pytest tests for MB3OYZ - Complex rank-revealing QR factorization with column pivoting.

MB3OYZ computes a truncated QR factorization A*P = Q*R with column pivoting,
estimating effective rank using incremental condition estimation.
"""
import pytest
import numpy as np
from ctrlsys import mb3oyz


def test_mb3oyz_rank_deficient_complex():
    """Test MB3OYZ with rank-deficient complex matrix.

    Random seed: 42 (for reproducibility)
    Construct rank-3 matrix via SVD with known singular values.
    """
    m, n = 6, 4
    rcond = 1.0e-10
    svlmax = 0.0

    np.random.seed(42)
    U = np.linalg.qr(np.random.randn(m, m) + 1j * np.random.randn(m, m))[0]
    V = np.linalg.qr(np.random.randn(n, n) + 1j * np.random.randn(n, n))[0]

    sigma = np.array([10.0, 5.0, 2.0, 1e-14])
    Sigma = np.zeros((m, n), dtype=complex)
    for i in range(4):
        Sigma[i, i] = sigma[i]

    a = U @ Sigma @ V.conj().T
    a = np.asfortranarray(a)

    true_rank = np.linalg.matrix_rank(a, tol=rcond * sigma[0])
    assert true_rank == 3

    a_result, rank, info, sval, jpvt, tau = mb3oyz(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 3
    assert sval[0] > 0.0
    assert sval[1] > 0.0
    assert sval[2] < sval[1]


def test_mb3oyz_full_rank_identity():
    """Test MB3OYZ with complex identity matrix."""
    m, n = 4, 4
    rcond = 1.0e-10
    svlmax = 0.0

    a = np.eye(4, dtype=complex, order='F')

    a_result, rank, info, sval, jpvt, tau = mb3oyz(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 4
    np.testing.assert_allclose(sval[0], 1.0, rtol=1e-10)
    np.testing.assert_allclose(sval[1], 1.0, rtol=1e-10)
    np.testing.assert_allclose(sval[2], 1.0, rtol=1e-10)


def test_mb3oyz_full_rank_random():
    """Test MB3OYZ with full-rank random complex matrix.

    Random seed: 123 (for reproducibility)
    """
    m, n = 5, 4
    rcond = 1.0e-10
    svlmax = 0.0

    np.random.seed(123)
    a = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    a = np.asfortranarray(a)

    a_result, rank, info, sval, jpvt, tau = mb3oyz(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 4
    assert sval[0] > 0.0
    assert sval[1] > 0.0
    assert sval[2] > 0.0


def test_mb3oyz_edge_case_zero_rows():
    """Test MB3OYZ with zero rows."""
    m, n = 0, 4
    rcond = 1.0e-10
    svlmax = 0.0

    a = np.array([[]], dtype=complex, order='F').reshape(0, 4)

    a_result, rank, info, sval, jpvt, tau = mb3oyz(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 0


def test_mb3oyz_edge_case_zero_cols():
    """Test MB3OYZ with zero columns."""
    m, n = 4, 0
    rcond = 1.0e-10
    svlmax = 0.0

    a = np.array([[]], dtype=complex, order='F').reshape(4, 0)

    a_result, rank, info, sval, jpvt, tau = mb3oyz(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 0


def test_mb3oyz_error_negative_m():
    """Test MB3OYZ error handling: negative m."""
    m, n = -1, 4
    rcond = 1.0e-10
    svlmax = 0.0

    a = np.zeros((1, 4), dtype=complex, order='F')

    with pytest.raises(ValueError, match="Dimensions must be non-negative"):
        mb3oyz(m, n, a, rcond, svlmax)


def test_mb3oyz_error_negative_n():
    """Test MB3OYZ error handling: negative n."""
    m, n = 4, -1
    rcond = 1.0e-10
    svlmax = 0.0

    a = np.zeros((4, 1), dtype=complex, order='F')

    with pytest.raises(ValueError, match="Dimensions must be non-negative"):
        mb3oyz(m, n, a, rcond, svlmax)


def test_mb3oyz_singular_value_estimates():
    """Test MB3OYZ singular value estimate accuracy.

    Random seed: 456 (for reproducibility)
    Compare SVAL estimates to true singular values from SVD.
    """
    m, n = 6, 4
    rcond = 1.0e-12
    svlmax = 0.0

    np.random.seed(456)
    a = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    a = np.asfortranarray(a)

    true_svd = np.linalg.svd(a, compute_uv=False)

    a_result, rank, info, sval, jpvt, tau = mb3oyz(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == min(m, n)
    np.testing.assert_allclose(sval[0], true_svd[0], rtol=0.15)
    np.testing.assert_allclose(sval[1], true_svd[rank - 1], rtol=0.3)


def test_mb3oyz_permutation_valid():
    """Test MB3OYZ returns valid permutation.

    Random seed: 789 (for reproducibility)
    Verify jpvt contains valid 1-based column indices.
    """
    m, n = 4, 4
    rcond = 1.0e-10
    svlmax = 0.0

    np.random.seed(789)
    a = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    a = np.asfortranarray(a)

    a_result, rank, info, sval, jpvt, tau = mb3oyz(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert len(jpvt) == n
    assert set(jpvt) == set(range(1, n + 1))


def test_mb3oyz_numerical_rank_detection():
    """Test MB3OYZ detects numerical rank correctly.

    Random seed: 999 (for reproducibility)
    """
    m, n = 4, 3
    rcond = 1.0e-8
    svlmax = 0.0

    np.random.seed(999)
    U = np.random.randn(m, 2) + 1j * np.random.randn(m, 2)
    V = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)
    a = U @ V.conj().T
    a += 1e-12 * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
    a = np.asfortranarray(a)

    a_result, rank, info, sval, jpvt, tau = mb3oyz(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
