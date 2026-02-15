"""
Tests for MB03PY - Rank-revealing RQ factorization with row pivoting.

MB03PY computes a truncated RQ factorization with row pivoting:
    P * A = R * Q
where R22 is the largest trailing upper triangular submatrix with
estimated condition number less than 1/RCOND.
"""
import numpy as np
import pytest
from ctrlsys import mb03py


"""Basic functionality tests for mb03py."""

def test_full_rank_square():
    """
    Full rank square matrix.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n = 4, 4
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == min(m, n)
    assert len(sval) == 3
    assert len(jpvt) == m
    assert len(tau) == min(m, n)
    assert sval[0] >= sval[1] >= 0

def test_full_rank_tall():
    """
    Full rank tall matrix (m > n).
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, n = 6, 4
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == min(m, n)
    assert len(jpvt) == m
    assert len(tau) == min(m, n)

def test_full_rank_wide():
    """
    Full rank wide matrix (m < n).
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, n = 4, 6
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == min(m, n)
    assert len(jpvt) == m
    assert len(tau) == min(m, n)


"""Tests for rank-deficient matrices."""

def test_rank_deficient_by_zero_row():
    """
    Matrix with zero row should have reduced rank.
    """
    m, n = 4, 4
    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [0.0, 0.0, 0.0, 0.0],
        [9.0, 10.0, 11.0, 12.0]
    ], dtype=float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank < min(m, n)
    assert rank <= 3

def test_rank_1_matrix():
    """
    Rank-1 matrix: outer product of two vectors.
    """
    m, n = 5, 5
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    a = np.outer(u, v).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == 1

def test_rank_2_matrix():
    """
    Rank-2 matrix: sum of two outer products.
    """
    m, n = 5, 5
    u1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    u2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    v2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    a = (np.outer(u1, v1) + np.outer(u2, v2)).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == 2


"""Mathematical property tests for mb03py."""

def test_orthogonal_q_property():
    """
    Validate Q is orthogonal: Q * Q^T = I.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m, n = 4, 6
    a_orig = np.random.randn(m, n).astype(float, order='F')
    a = a_orig.copy(order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a, rcond, svlmax)

    assert info == 0
    assert rank == m

    q = np.eye(n, dtype=float)
    for i in range(rank):
        row_idx = m - rank + i
        col_idx = n - rank + i

        v = np.zeros(n)
        v[:col_idx] = a_out[row_idx, :col_idx]
        v[col_idx] = 1.0

        tau_i = tau[m - rank + i]
        h = np.eye(n) - tau_i * np.outer(v, v)
        q = h @ q

    q_qt = q @ q.T
    np.testing.assert_allclose(q_qt, np.eye(n), rtol=1e-10, atol=1e-12)

def test_permutation_property():
    """
    JPVT defines a valid permutation.
    Random seed: 321 (for reproducibility)
    """
    np.random.seed(321)
    m, n = 5, 5
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    jpvt_set = set(jpvt)
    assert len(jpvt_set) == m
    assert all(1 <= j <= m for j in jpvt)

def test_sval_estimates_ordering():
    """
    SVAL estimates should satisfy ordering: sval[0] >= sval[1].
    Random seed: 654 (for reproducibility)
    """
    np.random.seed(654)
    m, n = 4, 4
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert sval[0] >= sval[1]

def test_rank_vs_numpy_svd():
    """
    Compare rank determination with NumPy SVD.
    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    m, n = 5, 5
    true_rank = 3
    u = np.random.randn(m, true_rank)
    v = np.random.randn(true_rank, n)
    a = (u @ v).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03py(m, n, a.copy(order='F'), rcond, svlmax)

    s_numpy = np.linalg.svd(a, compute_uv=False)
    numpy_rank = np.sum(s_numpy > 1e-10)

    assert info == 0
    assert rank == numpy_rank


"""Edge case tests for mb03py."""

def test_1x1_matrix():
    """Test 1x1 matrix."""
    a = np.array([[5.0]], dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03py(1, 1, a.copy(order='F'), 1e-10, 0.0)

    assert info == 0
    assert rank == 1
    assert jpvt[0] == 1

def test_single_row():
    """Test 1xN matrix (single row)."""
    a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03py(1, 4, a.copy(order='F'), 1e-10, 0.0)

    assert info == 0
    assert rank == 1

def test_single_column():
    """Test Mx1 matrix (single column)."""
    a = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03py(4, 1, a.copy(order='F'), 1e-10, 0.0)

    assert info == 0
    assert rank == 1

def test_zero_matrix():
    """Test zero matrix has rank 0."""
    a = np.zeros((3, 3), dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03py(3, 3, a.copy(order='F'), 1e-10, 0.0)

    assert info == 0
    assert rank == 0
    assert sval[0] == 0.0
    assert sval[1] == 0.0

def test_identity_matrix():
    """Test identity matrix has full rank."""
    n = 4
    a = np.eye(n, dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03py(n, n, a.copy(order='F'), 1e-10, 0.0)

    assert info == 0
    assert rank == n


"""Tests for parameter variations."""

def test_rcond_affects_rank():
    """
    Larger rcond should reduce detected rank for ill-conditioned matrices.
    """
    m, n = 4, 4
    a = np.diag([1.0, 0.1, 0.01, 0.001]).astype(float, order='F')

    _, rank_tight, _, _, _, _ = mb03py(m, n, a.copy(order='F'), 1e-10, 0.0)

    _, rank_loose, _, _, _, _ = mb03py(m, n, a.copy(order='F'), 0.1, 0.0)

    assert rank_tight >= rank_loose

def test_svlmax_positive():
    """
    Test with positive svlmax (estimated max singular value).
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m, n = 4, 4
    a = np.random.randn(m, n).astype(float, order='F')

    frob_norm = np.linalg.norm(a, 'fro')

    a_out, rank, info, sval, jpvt, tau = mb03py(
        m, n, a.copy(order='F'), 1e-10, frob_norm
    )

    assert info == 0
    assert rank == min(m, n)


"""Error handling tests for mb03py."""

def test_negative_m():
    """Negative m should raise error."""
    a = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError, match="non-negative"):
        mb03py(-1, 3, a, 1e-10, 0.0)

def test_negative_n():
    """Negative n should raise error."""
    a = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError, match="non-negative"):
        mb03py(3, -1, a, 1e-10, 0.0)

def test_rcond_out_of_range_high():
    """rcond > 1 should return error."""
    a = np.eye(3, dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03py(3, 3, a.copy(order='F'), 1.5, 0.0)

    assert info == -5

def test_rcond_negative():
    """rcond < 0 should return error."""
    a = np.eye(3, dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03py(3, 3, a.copy(order='F'), -0.1, 0.0)

    assert info == -5

def test_svlmax_negative():
    """svlmax < 0 should return error."""
    a = np.eye(3, dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03py(3, 3, a.copy(order='F'), 1e-10, -1.0)

    assert info == -6
