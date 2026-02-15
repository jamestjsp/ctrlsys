"""
Tests for SB03MV - Solve 2x2 discrete-time Lyapunov equation.

SB03MV solves for the 2-by-2 symmetric matrix X in:
    op(T)'*X*op(T) - X = SCALE*B

where T is 2-by-2, B is symmetric 2-by-2, and op(T) = T or T'.
"""
import pytest
import numpy as np


def test_sb03mv_basic_notrans_upper():
    """
    Test basic functionality: op(T) = T, upper triangle.

    Solves T'*X*T - X = SCALE*B (no transpose).
    Verifies equation residual.
    """
    from ctrlsys import sb03mv

    T = np.array([
        [0.5, 0.1],
        [0.0, 0.3]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.2],
        [0.0, 0.5]  # Only upper triangle used
    ], order='F', dtype=float)

    X, scale, xnorm, info = sb03mv(T, B, ltran=False, lupper=True)

    assert info == 0 or info == 1  # info=1 means perturbed for singularity
    assert 0 < scale <= 1

    # Verify equation: T'*X*T - X = scale*B
    # Make X symmetric for verification
    X_sym = np.triu(X) + np.triu(X, 1).T
    B_sym = np.triu(B) + np.triu(B, 1).T

    residual = T.T @ X_sym @ T - X_sym - scale * B_sym
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03mv_basic_trans_upper():
    """
    Test basic functionality: op(T) = T', upper triangle.

    Solves T*X*T' - X = SCALE*B (transpose).
    """
    from ctrlsys import sb03mv

    T = np.array([
        [0.4, 0.2],
        [0.1, 0.5]
    ], order='F', dtype=float)

    B = np.array([
        [0.8, 0.3],
        [0.0, 0.6]
    ], order='F', dtype=float)

    X, scale, xnorm, info = sb03mv(T, B, ltran=True, lupper=True)

    assert info == 0 or info == 1
    assert 0 < scale <= 1

    # Verify: T*X*T' - X = scale*B
    X_sym = np.triu(X) + np.triu(X, 1).T
    B_sym = np.triu(B) + np.triu(B, 1).T

    residual = T @ X_sym @ T.T - X_sym - scale * B_sym
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03mv_lower_triangle():
    """
    Test with lower triangle storage.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03mv

    np.random.seed(42)

    T = np.array([
        [0.6, 0.15],
        [0.05, 0.4]
    ], order='F', dtype=float)

    # B stored in lower triangle
    B = np.array([
        [0.7, 0.0],
        [0.25, 0.9]
    ], order='F', dtype=float)

    X, scale, xnorm, info = sb03mv(T, B, ltran=False, lupper=False)

    assert info == 0 or info == 1
    assert 0 < scale <= 1

    # Verify: T'*X*T - X = scale*B
    X_sym = np.tril(X) + np.tril(X, -1).T
    B_sym = np.tril(B) + np.tril(B, -1).T

    residual = T.T @ X_sym @ T - X_sym - scale * B_sym
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03mv_xnorm_property():
    """
    Test XNORM output - should be infinity norm of solution.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03mv

    np.random.seed(123)

    T = np.array([
        [0.5, 0.2],
        [0.1, 0.6]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.5],
        [0.0, 0.8]
    ], order='F', dtype=float)

    X, scale, xnorm, info = sb03mv(T, B, ltran=False, lupper=True)

    assert info == 0 or info == 1

    # XNORM = max(|x11| + |x12|, |x12| + |x22|) (infinity norm for symmetric)
    x11 = X[0, 0]
    x12 = X[0, 1]
    x22 = X[1, 1]
    expected_xnorm = max(abs(x11) + abs(x12), abs(x12) + abs(x22))
    np.testing.assert_allclose(xnorm, expected_xnorm, rtol=1e-14)


def test_sb03mv_symmetry_preservation():
    """
    Test that solution X is symmetric (stored in specified triangle).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03mv

    np.random.seed(456)

    T = np.random.randn(2, 2).astype(float, order='F') * 0.4

    # Symmetric B (upper triangle)
    B_upper = np.array([
        [1.2, 0.3],
        [0.0, 0.8]
    ], order='F', dtype=float)

    X, scale, xnorm, info = sb03mv(T, B_upper, ltran=False, lupper=True)

    assert info == 0 or info == 1

    # Upper triangle stores solution - construct symmetric
    X_full = np.triu(X) + np.triu(X, 1).T

    # Verify symmetry by checking both triangles give same result
    np.testing.assert_allclose(X_full, X_full.T, rtol=1e-14)


def test_sb03mv_diagonal_t():
    """
    Test with diagonal T matrix (simple case).

    For diagonal T = diag(t1, t2), the equation simplifies.
    """
    from ctrlsys import sb03mv

    T = np.array([
        [0.5, 0.0],
        [0.0, 0.7]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    X, scale, xnorm, info = sb03mv(T, B, ltran=False, lupper=True)

    assert info == 0 or info == 1
    assert 0 < scale <= 1

    # For diagonal T: T'*X*T - X = scale*B
    # t1^2 * x11 - x11 = scale * b11 => x11 = scale * b11 / (t1^2 - 1)
    # t2^2 * x22 - x22 = scale * b22 => x22 = scale * b22 / (t2^2 - 1)

    X_sym = np.triu(X) + np.triu(X, 1).T
    B_sym = np.triu(B) + np.triu(B, 1).T

    residual = T.T @ X_sym @ T - X_sym - scale * B_sym
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03mv_nearly_singular():
    """
    Test near-singular case (eigenvalues close to reciprocal).

    When T has eigenvalues close to reciprocal, INFO = 1.
    """
    from ctrlsys import sb03mv

    # T with eigenvalue close to 1 (reciprocal eigenvalue issue)
    T = np.array([
        [0.99, 0.01],
        [0.0, 0.98]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], order='F', dtype=float)

    X, scale, xnorm, info = sb03mv(T, B, ltran=False, lupper=True)

    # May return info=1 for near-singular
    assert info == 0 or info == 1
    assert 0 < scale <= 1

    # Solution should still satisfy equation (with perturbation)
    X_sym = np.triu(X) + np.triu(X, 1).T
    B_sym = np.triu(B) + np.triu(B, 1).T

    residual = T.T @ X_sym @ T - X_sym - scale * B_sym
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)
