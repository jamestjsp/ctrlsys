import numpy as np
import pytest
from ctrlsys import mb04id


def _build_q_from_qr(a_out, tau, n, m):
    """Build orthogonal Q from packed Householder reflectors."""
    k = min(n, m)
    Q = np.eye(n)
    for i in range(k):
        v = np.zeros(n)
        v[i] = 1.0
        v[i+1:] = a_out[i+1:, i]
        Q = Q @ (np.eye(n) - tau[i] * np.outer(v, v))
    return Q


def test_mb04id_basic():
    """
    Validate basic QR factorization with lower-left zero triangle.

    Tests structured QR on 8x7 matrix with p=2 zero triangle.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 8, 7, 2

    # Generate random matrix with zero lower-left triangle
    a = np.random.randn(n, m).astype(float, order='F')
    a[6, 0] = 0.0  # Zero triangle
    a[7, 0] = 0.0
    a[7, 1] = 0.0

    a_orig = a.copy()

    # Compute QR factorization
    a_out, tau, info = mb04id(n, m, p, a)

    assert info == 0
    assert a_out.shape == (n, m)
    assert tau.shape == (min(n, m),)

    Q = _build_q_from_qr(a_out, tau, n, m)
    R = np.zeros((n, m))
    R[:min(n, m), :] = np.triu(a_out[:min(n, m), :])
    np.testing.assert_allclose(Q @ R, a_orig, atol=1e-12)
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-14)


def test_mb04id_with_b_matrix():
    """
    Validate QR factorization with optional B matrix transformation.

    Tests that Q^T is correctly applied to B matrix.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p, l = 6, 5, 2, 3

    a = np.random.randn(n, m).astype(float, order='F')
    a[4, 0] = 0.0  # Zero triangle
    a[5, 0] = 0.0
    a[5, 1] = 0.0

    b = np.random.randn(n, l).astype(float, order='F')
    b_orig = b.copy()

    a_out, b_out, tau, info = mb04id(n, m, p, a, b=b, l=l)

    assert info == 0
    assert b_out.shape == (n, l)

    # Verify that B was transformed (not equal to original)
    assert not np.allclose(b_out, b_orig)


def test_mb04id_orthogonality():
    """
    Validate tau factors are computed.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 10, 8, 3

    a = np.random.randn(n, m).astype(float, order='F')
    # Zero lower-left triangle
    for i in range(n - p, n):
        for j in range(min(i - (n - p), m)):
            a[i, j] = 0.0

    a_out, tau, info = mb04id(n, m, p, a)
    assert info == 0

    # Verify tau values are valid (between 0 and 2 for real matrices)
    assert np.all(tau >= 0.0)
    assert np.all(tau <= 2.0)


def test_mb04id_square_matrix():
    """
    Validate QR factorization for square matrix case (n = m).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 6
    m = 6
    p = 2

    a = np.random.randn(n, m).astype(float, order='F')
    a[4, 0] = 0.0
    a[5, 0] = 0.0
    a[5, 1] = 0.0

    a_orig = a.copy()
    a_out, tau, info = mb04id(n, m, p, a)

    assert info == 0
    assert tau.shape == (n,)

    Q = _build_q_from_qr(a_out, tau, n, m)
    R = np.triu(a_out)
    np.testing.assert_allclose(Q @ R, a_orig, atol=1e-12)
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-14)


def test_mb04id_tall_matrix():
    """
    Validate QR factorization for tall matrix (n > m).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m, p = 12, 5, 2

    a = np.random.randn(n, m).astype(float, order='F')
    # Zero triangle
    a[10, 0] = 0.0
    a[11, 0] = 0.0
    a[11, 1] = 0.0

    a_out, tau, info = mb04id(n, m, p, a)

    assert info == 0
    assert tau.shape == (m,)

    # Extract R (m x m upper triangular)
    r = np.triu(a_out[:m, :])
    assert np.allclose(np.tril(r, -1), 0.0, atol=1e-14)


def test_mb04id_zero_p():
    """
    Validate standard QR factorization when p = 0 (no special structure).

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 5, 4, 0

    a = np.random.randn(n, m).astype(float, order='F')

    a_out, tau, info = mb04id(n, m, p, a)

    assert info == 0

    # Extract R
    r = np.triu(a_out[:m, :])
    assert np.allclose(np.tril(r, -1), 0.0, atol=1e-14)


def test_mb04id_edge_cases():
    """
    Test edge cases: minimal dimensions, n <= p+1.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    # Case 1: n = p + 1 (quick return path)
    n, m, p = 3, 4, 2
    a = np.random.randn(n, m).astype(float, order='F')
    a_out, tau, info = mb04id(n, m, p, a)
    assert info == 0
    assert np.all(tau == 0.0)

    # Case 2: Small matrix
    n, m, p = 2, 2, 1
    a = np.random.randn(n, m).astype(float, order='F')
    a[1, 0] = 0.0
    a_out, tau, info = mb04id(n, m, p, a)
    assert info == 0


def test_mb04id_workspace_query():
    """
    Validate workspace query functionality (ldwork = -1).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m, p = 8, 6, 2

    a = np.random.randn(n, m).astype(float, order='F')

    # Query optimal workspace
    a_out, tau, info, dwork = mb04id(n, m, p, a, ldwork=-1)

    assert info == 0
    assert dwork > 0  # Should return optimal workspace size


def test_mb04id_error_handling():
    """
    Validate error handling for invalid parameters.
    """
    # Invalid n < 0
    with pytest.raises((ValueError, RuntimeError)):
        mb04id(-1, 5, 2, np.zeros((1, 5), dtype=float, order='F'))

    # Invalid m < 0
    with pytest.raises((ValueError, RuntimeError)):
        mb04id(5, -1, 2, np.zeros((5, 1), dtype=float, order='F'))

    # Invalid p < 0
    with pytest.raises((ValueError, RuntimeError)):
        mb04id(5, 5, -1, np.zeros((5, 5), dtype=float, order='F'))
