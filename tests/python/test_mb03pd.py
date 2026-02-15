"""
Tests for MB03PD - Rank-revealing RQ factorization with row pivoting.

MB03PD computes (optionally) a rank-revealing RQ factorization:
    P * A = R * Q
where R22 is the largest trailing upper triangular submatrix with
estimated condition number less than 1/RCOND.

JOBRQ = 'R': Perform RQ factorization with row pivoting
JOBRQ = 'N': Assume RQ factorization already done
"""
import numpy as np
import pytest
from ctrlsys import mb03pd


def test_html_doc_example():
    """
    Test from SLICOT HTML documentation example.

    M=6, N=5, JOBRQ='R', RCOND=5e-16, SVLMAX=0.0
    Expected: rank=4, jpvt=[2,4,6,3,1,5], sval=[24.5744, 0.9580, 0.0000]
    """
    m, n = 6, 5
    a = np.array([
        [1., 2., 6., 3., 5.],
        [-2., -1., -1., 0., -2.],
        [5., 5., 1., 5., 1.],
        [-2., -1., -1., 0., -2.],
        [4., 8., 4., 20., 4.],
        [-2., -1., -1., 0., -2.]
    ], dtype=float, order='F')

    rcond = 5e-16
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == 4

    expected_jpvt = np.array([2, 4, 6, 3, 1, 5], dtype=np.int32)
    np.testing.assert_array_equal(jpvt, expected_jpvt)

    expected_sval = np.array([24.5744, 0.9580, 0.0000])
    np.testing.assert_allclose(sval, expected_sval, rtol=1e-3, atol=1e-4)


def test_jobrq_r_full_rank_square():
    """
    Full rank square matrix with JOBRQ='R'.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n = 4, 4
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == min(m, n)
    assert len(sval) == 3
    assert len(jpvt) == m
    assert len(tau) == min(m, n)
    assert sval[0] >= sval[1] >= 0


def test_jobrq_r_tall_matrix():
    """
    Full rank tall matrix (m > n) with JOBRQ='R'.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, n = 6, 4
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == min(m, n)
    assert len(jpvt) == m
    assert len(tau) == min(m, n)


def test_jobrq_r_wide_matrix():
    """
    Full rank wide matrix (m < n) with JOBRQ='R'.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, n = 4, 6
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == min(m, n)
    assert len(jpvt) == m
    assert len(tau) == min(m, n)


def test_jobrq_n_precomputed_factorization():
    """
    Test JOBRQ='N': use precomputed RQ factorization.
    First do RQ with 'R', then use result with 'N'.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m, n = 4, 4
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_rq, rank1, info1, sval1, jpvt1, tau1 = mb03pd('R', m, n, a.copy(order='F'), rcond, svlmax)
    assert info1 == 0

    a_out, rank2, info2, sval2, jpvt2, tau2 = mb03pd('N', m, n, a_rq.copy(order='F'), rcond, svlmax)
    assert info2 == 0

    assert rank1 == rank2
    np.testing.assert_allclose(sval1, sval2, rtol=1e-14)


def test_rank_deficient_matrix():
    """
    Matrix with linearly dependent rows should have reduced rank.
    """
    m, n = 4, 4
    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [2.0, 4.0, 6.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ], dtype=float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', m, n, a.copy(order='F'), rcond, svlmax)

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

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert rank == 1


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

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    jpvt_set = set(jpvt)
    assert len(jpvt_set) == m
    assert all(1 <= j <= m for j in jpvt)


def test_sval_ordering():
    """
    SVAL estimates should satisfy: sval[0] >= sval[1].
    Random seed: 654 (for reproducibility)
    """
    np.random.seed(654)
    m, n = 4, 4
    a = np.random.randn(m, n).astype(float, order='F')

    rcond = 1e-10
    svlmax = 0.0

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', m, n, a.copy(order='F'), rcond, svlmax)

    assert info == 0
    assert sval[0] >= sval[1]


def test_1x1_matrix():
    """Test 1x1 matrix."""
    a = np.array([[5.0]], dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', 1, 1, a.copy(order='F'), 1e-10, 0.0)

    assert info == 0
    assert rank == 1
    assert jpvt[0] == 1


def test_zero_matrix():
    """Test zero matrix has rank 0."""
    a = np.zeros((3, 3), dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', 3, 3, a.copy(order='F'), 1e-10, 0.0)

    assert info == 0
    assert rank == 0
    assert sval[0] == 0.0
    assert sval[1] == 0.0


def test_identity_matrix():
    """Test identity matrix has full rank."""
    n = 4
    a = np.eye(n, dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', n, n, a.copy(order='F'), 1e-10, 0.0)

    assert info == 0
    assert rank == n


def test_zero_dimensions():
    """Test m=0 or n=0 returns rank=0."""
    a = np.zeros((0, 3), dtype=float, order='F')
    a_out, rank, info, sval, jpvt, tau = mb03pd('R', 0, 3, a, 1e-10, 0.0)
    assert info == 0
    assert rank == 0


def test_rcond_affects_rank():
    """
    Larger rcond should reduce detected rank for ill-conditioned matrices.
    """
    m, n = 4, 4
    a = np.diag([1.0, 0.1, 0.01, 0.001]).astype(float, order='F')

    _, rank_tight, _, _, _, _ = mb03pd('R', m, n, a.copy(order='F'), 1e-10, 0.0)
    _, rank_loose, _, _, _, _ = mb03pd('R', m, n, a.copy(order='F'), 0.1, 0.0)

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

    a_out, rank, info, sval, jpvt, tau = mb03pd(
        'R', m, n, a.copy(order='F'), 1e-10, frob_norm
    )

    assert info == 0
    assert rank == min(m, n)


def test_invalid_jobrq():
    """Invalid JOBRQ should return error."""
    a = np.eye(3, dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03pd('X', 3, 3, a.copy(order='F'), 1e-10, 0.0)

    assert info == -1


def test_negative_m():
    """Negative m should raise error."""
    a = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError, match="non-negative"):
        mb03pd('R', -1, 3, a, 1e-10, 0.0)


def test_negative_n():
    """Negative n should raise error."""
    a = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError, match="non-negative"):
        mb03pd('R', 3, -1, a, 1e-10, 0.0)


def test_rcond_negative():
    """rcond < 0 should return error."""
    a = np.eye(3, dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', 3, 3, a.copy(order='F'), -0.1, 0.0)

    assert info == -7


def test_svlmax_negative():
    """svlmax < 0 should return error."""
    a = np.eye(3, dtype=float, order='F')

    a_out, rank, info, sval, jpvt, tau = mb03pd('R', 3, 3, a.copy(order='F'), 1e-10, -1.0)

    assert info == -8
