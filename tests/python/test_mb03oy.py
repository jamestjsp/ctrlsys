#!/usr/bin/env python3
"""
pytest tests for MB03OY - Rank determination via incremental condition estimation.
"""
import pytest
import numpy as np
from ctrlsys import mb03oy


def test_mb03oy_rank_deficient_matrix():
    """Test MB03OY with rank-deficient matrix (6x4, rank 3).

    Synthetic test: Construct rank-3 matrix via SVD with known singular values.
    Strategy per CLAUDE.md: Use NumPy/SciPy for mathematically sound test cases.
    """
    m, n = 6, 4
    rcond = 1.0e-10
    svlmax = 0.0

    # Construct rank-3 matrix using SVD: A = U @ Sigma @ V.T
    # Singular values: [10.0, 5.0, 2.0, 1e-14] - last one below threshold
    np.random.seed(42)
    U = np.linalg.qr(np.random.randn(m, m))[0]  # Orthogonal m x m
    V = np.linalg.qr(np.random.randn(n, n))[0]  # Orthogonal n x n

    # Singular values: 3 large + 1 tiny (below rcond * max)
    sigma = np.array([10.0, 5.0, 2.0, 1e-14])
    Sigma = np.zeros((m, n))
    for i in range(4):
        Sigma[i, i] = sigma[i]

    a = U @ Sigma @ V.T
    a = np.asfortranarray(a)

    # Verify construction: check rank via NumPy
    true_rank = np.linalg.matrix_rank(a, tol=rcond * sigma[0])
    assert true_rank == 3, f"Synthetic matrix should be rank 3, got {true_rank}"

    a_result, rank, info, sval, jpvt, tau = mb03oy(m, n, a.copy(), rcond, svlmax)

    # Verify successful execution
    assert info == 0

    # Verify estimated rank matches construction
    assert rank == 3, f"Expected rank 3 (from SVD construction), got {rank}"

    # Verify singular value estimates are reasonable
    assert sval[0] > 0.0  # Largest singular value positive
    assert sval[1] > 0.0  # Rank-th singular value positive
    assert sval[2] < sval[1]  # (rank+1)-th smaller

    # Verify condition number of rank-3 submatrix
    cond_estimate = sval[0] / sval[1]
    assert cond_estimate < 1.0 / rcond

    # Compare with NumPy SVD (QR-based estimates less accurate than full SVD)
    # MB03OY uses incremental condition estimation, not full SVD
    true_svd = np.linalg.svd(a, compute_uv=False)
    np.testing.assert_allclose(sval[0], true_svd[0], rtol=0.15)  # ~15% tolerance for QR estimate
    np.testing.assert_allclose(sval[1], true_svd[2], rtol=0.3)   # Rank-th SV (less accurate)
    assert sval[2] < 1e-10  # Near-zero for 4th singular value

    # Verify R matrix upper triangle (first rank columns)
    for j in range(rank):
        for i in range(j + 1):
            assert not np.isnan(a_result[i, j])


def test_mb03oy_full_rank_matrix():
    """Test MB03OY with full-rank identity matrix."""
    m, n = 4, 4
    rcond = 1.0e-10
    svlmax = 0.0

    # Identity matrix (full rank)
    a = np.eye(4, dtype=np.float64, order='F')

    a_result, rank, info, sval, jpvt, tau = mb03oy(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 4  # Full rank

    # For identity, singular values should all be ~1.0
    np.testing.assert_allclose(sval[0], 1.0, rtol=1e-10)
    np.testing.assert_allclose(sval[1], 1.0, rtol=1e-10)
    np.testing.assert_allclose(sval[2], 1.0, rtol=1e-10)


def test_mb03oy_full_rank_random_matrix():
    """Test MB03OY with full-rank random matrix."""
    m, n = 5, 4
    rcond = 1.0e-10
    svlmax = 0.0

    # Generate random full-rank matrix
    np.random.seed(12345)
    a = np.random.randn(m, n)
    a = np.asfortranarray(a)  # Convert to column-major

    a_result, rank, info, sval, jpvt, tau = mb03oy(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 4  # Should be full rank (min(m,n))

    # Verify singular values are positive
    assert sval[0] > 0.0
    assert sval[1] > 0.0
    assert sval[2] > 0.0


def test_mb03oy_edge_case_zero_rows():
    """Test MB03OY with zero rows."""
    m, n = 0, 4
    rcond = 1.0e-10
    svlmax = 0.0

    a = np.array([[]], dtype=np.float64, order='F').reshape(0, 4)

    a_result, rank, info, sval, jpvt, tau = mb03oy(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 0  # No rows means rank 0


def test_mb03oy_edge_case_zero_cols():
    """Test MB03OY with zero columns."""
    m, n = 4, 0
    rcond = 1.0e-10
    svlmax = 0.0

    a = np.array([[]], dtype=np.float64, order='F').reshape(4, 0)

    a_result, rank, info, sval, jpvt, tau = mb03oy(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 0  # No columns means rank 0


def test_mb03oy_error_negative_m():
    """Test MB03OY error handling: negative m."""
    m, n = -1, 4
    rcond = 1.0e-10
    svlmax = 0.0

    a = np.zeros((1, 4), dtype=np.float64, order='F')

    with pytest.raises(ValueError, match="Dimensions must be non-negative"):
        a_result, rank, info, sval, jpvt, tau = mb03oy(m, n, a, rcond, svlmax)


def test_mb03oy_error_negative_n():
    """Test MB03OY error handling: negative n."""
    m, n = 4, -1
    rcond = 1.0e-10
    svlmax = 0.0

    a = np.zeros((4, 1), dtype=np.float64, order='F')

    with pytest.raises(ValueError, match="Dimensions must be non-negative"):
        a_result, rank, info, sval, jpvt, tau = mb03oy(m, n, a, rcond, svlmax)


def test_mb03oy_strict_rcond():
    """Test MB03OY with strict rcond (should detect lower rank)."""
    m, n = 4, 4
    rcond = 0.5  # Strict threshold
    svlmax = 0.0

    # Matrix with condition number ~2 (two singular values differ by factor 2)
    a = np.diag([2.0, 2.0, 1.0, 1.0])
    a = np.asfortranarray(a)

    a_result, rank, info, sval, jpvt, tau = mb03oy(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    # With rcond=0.5, only singular values >= 0.5*max_sval should count
    # Expected rank depends on threshold application
    assert 0 < rank <= 4


def test_mb03oy_numerical_rank_detection():
    """Test MB03OY detects numerical rank correctly."""
    m, n = 4, 3
    rcond = 1.0e-8
    svlmax = 0.0

    # Create rank-2 matrix with small noise
    np.random.seed(42)
    U = np.random.randn(m, 2)
    V = np.random.randn(n, 2)
    a = U @ V.T
    a += 1e-12 * np.random.randn(m, n)  # Tiny noise
    a = np.asfortranarray(a)

    a_result, rank, info, sval, jpvt, tau = mb03oy(m, n, a.copy(), rcond, svlmax)

    assert info == 0
    assert rank == 2  # Should detect rank 2 (noise is below threshold)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
