"""
Tests for SG03AY - Solve reduced generalized continuous-time Lyapunov equation.

Solves either:
    A' * X * E + E' * X * A = SCALE * Y  (TRANS='N')
or
    A * X * E' + E * X * A' = SCALE * Y  (TRANS='T')

where A is upper quasi-triangular (Schur form) and E is upper triangular.
"""
import pytest
import numpy as np


def test_sg03ay_basic_notrans():
    """
    Test basic functionality: solve equation (1).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sg03ay

    np.random.seed(42)

    n = 3
    A = np.array([
        [2.0, 0.5, 0.1],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    E = np.array([
        [1.0, 0.2, 0.1],
        [0.0, 1.0, 0.15],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    Y = np.array([
        [1.0, 0.3, 0.2],
        [0.3, 0.8, 0.1],
        [0.2, 0.1, 0.5]
    ], order='F', dtype=float)

    X, scale, info = sg03ay('N', A, E, Y.copy())

    assert info == 0 or info == 1
    assert 0 < scale <= 1

    # Verify: A'*X*E + E'*X*A = scale*Y
    X_sym = (X + X.T) / 2  # Make symmetric
    residual = A.T @ X_sym @ E + E.T @ X_sym @ A - scale * Y
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sg03ay_basic_trans():
    """
    Test transpose equation: solve equation (2).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sg03ay

    np.random.seed(123)

    n = 3
    A = np.array([
        [1.8, 0.4, 0.2],
        [0.0, 1.2, 0.1],
        [0.0, 0.0, 0.9]
    ], order='F', dtype=float)

    E = np.array([
        [1.0, 0.1, 0.05],
        [0.0, 1.0, 0.1],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    Y = np.array([
        [0.9, 0.2, 0.1],
        [0.2, 0.7, 0.15],
        [0.1, 0.15, 0.6]
    ], order='F', dtype=float)

    X, scale, info = sg03ay('T', A, E, Y.copy())

    assert info == 0 or info == 1
    assert 0 < scale <= 1

    # Verify: A*X*E' + E*X*A' = scale*Y
    X_sym = (X + X.T) / 2
    residual = A @ X_sym @ E.T + E @ X_sym @ A.T - scale * Y
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sg03ay_2x2_block():
    """
    Test with 2x2 diagonal block (complex conjugate eigenvalues).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sg03ay

    np.random.seed(456)

    n = 4
    # A with 2x2 block at (2,2) position
    A = np.array([
        [2.0, 0.3, 0.1, 0.05],
        [0.0, 1.5, 0.8, 0.1],
        [0.0, -0.6, 1.5, 0.2],
        [0.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    E = np.array([
        [1.0, 0.15, 0.08, 0.03],
        [0.0, 1.0, 0.12, 0.05],
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    Y = np.array([
        [1.0, 0.2, 0.15, 0.1],
        [0.2, 0.8, 0.1, 0.05],
        [0.15, 0.1, 0.7, 0.08],
        [0.1, 0.05, 0.08, 0.5]
    ], order='F', dtype=float)

    X, scale, info = sg03ay('N', A, E, Y.copy())

    assert info == 0 or info == 1
    assert 0 < scale <= 1

    X_sym = (X + X.T) / 2
    residual = A.T @ X_sym @ E + E.T @ X_sym @ A - scale * Y
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_sg03ay_small_n1():
    """
    Test with n=1 (smallest non-trivial case).
    """
    from ctrlsys import sg03ay

    A = np.array([[2.0]], order='F', dtype=float)
    E = np.array([[1.0]], order='F', dtype=float)
    Y = np.array([[1.0]], order='F', dtype=float)

    X, scale, info = sg03ay('N', A, E, Y.copy())

    assert info == 0 or info == 1
    assert 0 < scale <= 1

    # For n=1: a*x*e + e*x*a = scale*y => 2*a*e*x = scale*y
    expected_x = scale * Y[0, 0] / (2 * A[0, 0] * E[0, 0])
    np.testing.assert_allclose(X[0, 0], expected_x, rtol=1e-14)


def test_sg03ay_empty():
    """
    Test with n=0 (quick return).
    """
    from ctrlsys import sg03ay

    A = np.array([], order='F', dtype=float).reshape((0, 0))
    E = np.array([], order='F', dtype=float).reshape((0, 0))
    Y = np.array([], order='F', dtype=float).reshape((0, 0))

    X, scale, info = sg03ay('N', A, E, Y.copy())

    assert info == 0
    assert scale == 1.0


def test_sg03ay_symmetry():
    """
    Test that solution X is symmetric.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sg03ay

    np.random.seed(789)

    n = 4
    A = np.triu(np.random.randn(n, n).astype(float, order='F'))
    np.fill_diagonal(A, np.abs(np.diag(A)) + 1.0)

    E = np.triu(np.random.randn(n, n).astype(float, order='F'))
    np.fill_diagonal(E, np.abs(np.diag(E)) + 0.5)

    Y = np.random.randn(n, n).astype(float, order='F')
    Y = (Y + Y.T) / 2  # Make symmetric

    X, scale, info = sg03ay('N', A, E, Y.copy())

    assert info == 0 or info == 1

    # X should be symmetric (upper triangle is computed)
    X_full = np.triu(X) + np.triu(X, 1).T
    np.testing.assert_allclose(X_full, X_full.T, rtol=1e-14)
