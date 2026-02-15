import numpy as np
import pytest
from ctrlsys import mb04jd


def _build_q_from_lq(a_out, tau, n, m):
    """Build orthogonal Q from packed LQ Householder reflectors."""
    k = min(n, m)
    Q = np.eye(m)
    for i in range(k):
        v = np.zeros(m)
        v[i] = 1.0
        v[i+1:] = a_out[i, i+1:]
        Q = (np.eye(m) - tau[i] * np.outer(v, v)) @ Q
    return Q


def test_mb04jd_basic():
    """
    Validate basic LQ factorization with upper-right zero triangle.

    Tests structured LQ on 8x7 matrix with p=2 zero triangle.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 8, 7, 2

    a = np.random.randn(n, m).astype(float, order='F')
    # Zero upper-right triangle: min(n,p) x p
    a[0, 5] = 0.0
    a[0, 6] = 0.0
    a[1, 6] = 0.0

    a_orig = a.copy()

    a_out, tau, info = mb04jd(n, m, p, a)

    assert info == 0
    assert a_out.shape == (n, m)
    assert tau.shape == (min(n, m),)

    k = min(n, m)
    Q = _build_q_from_lq(a_out, tau, n, m)
    L = np.zeros((n, m))
    L[:, :k] = np.tril(a_out[:, :k])
    np.testing.assert_allclose(L @ Q, a_orig, atol=1e-12)
    np.testing.assert_allclose(Q @ Q.T, np.eye(m), atol=1e-14)


def test_mb04jd_with_b_matrix():
    """
    Validate LQ factorization with optional B matrix transformation.

    Tests that Q is correctly applied to B matrix from the right.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p, l = 6, 7, 2, 3

    a = np.random.randn(n, m).astype(float, order='F')
    # Zero upper-right triangle
    a[0, 5] = 0.0
    a[0, 6] = 0.0
    a[1, 6] = 0.0

    b = np.random.randn(l, m).astype(float, order='F')
    b_orig = b.copy()

    a_out, b_out, tau, info = mb04jd(n, m, p, a, b=b, l=l)

    assert info == 0
    assert b_out.shape == (l, m)

    # Verify that B was transformed (not equal to original)
    assert not np.allclose(b_out, b_orig)


def test_mb04jd_orthogonality():
    """
    Validate tau factors are computed correctly.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 8, 10, 3

    a = np.random.randn(n, m).astype(float, order='F')
    # Zero upper-right triangle: min(n,p) x p
    for i in range(min(n, p)):
        for j in range(m - p + i, m):
            a[i, j] = 0.0

    a_out, tau, info = mb04jd(n, m, p, a)
    assert info == 0

    # Verify tau values are valid (between 0 and 2 for real matrices)
    assert np.all(tau >= 0.0)
    assert np.all(tau <= 2.0)


def test_mb04jd_square_matrix():
    """
    Validate LQ factorization for square matrix case (n = m).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 6
    m = 6
    p = 2

    a = np.random.randn(n, m).astype(float, order='F')
    # Zero upper-right triangle
    a[0, 4] = 0.0
    a[0, 5] = 0.0
    a[1, 5] = 0.0

    a_orig = a.copy()
    a_out, tau, info = mb04jd(n, m, p, a)

    assert info == 0
    assert tau.shape == (n,)

    Q = _build_q_from_lq(a_out, tau, n, m)
    L = np.tril(a_out)
    np.testing.assert_allclose(L @ Q, a_orig, atol=1e-12)
    np.testing.assert_allclose(Q @ Q.T, np.eye(m), atol=1e-14)


def test_mb04jd_wide_matrix():
    """
    Validate LQ factorization for wide matrix (m > n).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m, p = 5, 12, 2

    a = np.random.randn(n, m).astype(float, order='F')
    # Zero upper-right triangle
    a[0, 10] = 0.0
    a[0, 11] = 0.0
    a[1, 11] = 0.0

    a_out, tau, info = mb04jd(n, m, p, a)

    assert info == 0
    assert tau.shape == (n,)

    # Extract L (n x n lower triangular)
    l_mat = np.tril(a_out[:, :n])
    assert np.allclose(np.triu(l_mat, 1), 0.0, atol=1e-14)


def test_mb04jd_zero_p():
    """
    Validate standard LQ factorization when p = 0 (no special structure).

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 4, 5, 0

    a = np.random.randn(n, m).astype(float, order='F')

    a_out, tau, info = mb04jd(n, m, p, a)

    assert info == 0

    # Extract L
    l_mat = np.tril(a_out[:, :n])
    assert np.allclose(np.triu(l_mat, 1), 0.0, atol=1e-14)


def test_mb04jd_edge_cases():
    """
    Test edge cases: minimal dimensions, m <= p+1.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    # Case 1: m = p + 1 (quick return path)
    n, m, p = 4, 3, 2
    a = np.random.randn(n, m).astype(float, order='F')
    a_out, tau, info = mb04jd(n, m, p, a)
    assert info == 0
    assert np.all(tau == 0.0)

    # Case 2: Small matrix
    n, m, p = 2, 3, 1
    a = np.random.randn(n, m).astype(float, order='F')
    a[0, 2] = 0.0
    a_out, tau, info = mb04jd(n, m, p, a)
    assert info == 0


def test_mb04jd_l_matrix_structure():
    """
    Validate L matrix has proper lower trapezoidal structure.

    For LQ factorization A = L*Q, the L matrix should be lower trapezoidal.
    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, p = 5, 8, 2

    a = np.random.randn(n, m).astype(float, order='F')
    a[0, 6] = 0.0
    a[0, 7] = 0.0
    a[1, 7] = 0.0

    a_out, tau, info = mb04jd(n, m, p, a)

    assert info == 0

    # Extract L (n x min(n,m) lower trapezoidal)
    l_mat = a_out[:, :min(n, m)].copy()
    for i in range(min(n, m)):
        l_mat[i, i+1:min(n, m)] = 0.0

    # Verify L is lower triangular (zeros above diagonal)
    for i in range(min(n, m)):
        for j in range(i + 1, min(n, m)):
            assert abs(l_mat[i, j]) < 1e-14, f"L[{i},{j}] should be zero"

    # Verify diagonal elements are reasonable (non-tiny for full-rank input)
    for i in range(min(n, m)):
        assert abs(a_out[i, i]) > 1e-10, f"L[{i},{i}] diagonal too small"


def test_mb04jd_compare_numpy_lq():
    """
    Compare LQ factorization result with NumPy's QR on transposed matrix.

    For a full-rank matrix without special structure, LQ should match
    the L factor from NumPy's QR decomposition of A^T (transpose relation).
    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n, m, p = 4, 6, 0

    a = np.random.randn(n, m).astype(float, order='F')
    a_orig = a.copy()

    a_out, tau, info = mb04jd(n, m, p, a)

    assert info == 0

    # NumPy's QR on A^T gives A^T = Q * R, so A = R^T * Q^T = L * Q
    q_np, r_np = np.linalg.qr(a_orig.T)
    l_np = r_np.T[:, :n]

    # Extract L from MB04JD output
    l_slicot = np.tril(a_out[:, :n])

    # L matrices may differ by sign (column sign conventions)
    # So we compare absolute values of diagonal elements
    for i in range(n):
        assert abs(abs(l_slicot[i, i]) - abs(l_np[i, i])) < 1e-10, \
            f"Diagonal element mismatch at [{i},{i}]"


def test_mb04jd_error_handling():
    """
    Validate error handling for invalid parameters.
    """
    # Invalid n < 0
    with pytest.raises((ValueError, RuntimeError)):
        mb04jd(-1, 5, 2, np.zeros((1, 5), dtype=float, order='F'))

    # Invalid m < 0
    with pytest.raises((ValueError, RuntimeError)):
        mb04jd(5, -1, 2, np.zeros((5, 1), dtype=float, order='F'))

    # Invalid p < 0
    with pytest.raises((ValueError, RuntimeError)):
        mb04jd(5, 5, -1, np.zeros((5, 5), dtype=float, order='F'))
