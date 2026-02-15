"""
Tests for AB13DX: Transfer function singular value at specific frequency.

Computes maximum singular value of G(lambda) = C*inv(lambda*E - A)*B + D.
"""

import numpy as np


def test_ab13dx_continuous_identity():
    """
    Test AB13DX with continuous-time, E=I, omega=1.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab13dx

    np.random.seed(42)
    n, m, p = 2, 1, 1

    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)
    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)

    omega = 1.0

    result, info = ab13dx('C', 'I', 'Z', n, m, p, a, e, b, c, d, omega)

    assert info == 0
    assert result > 0


def test_ab13dx_continuous_omega_zero():
    """
    Test AB13DX with omega=0 (DC gain) for continuous-time system.

    For omega=0, G(0) = -C*inv(A)*B + D (assuming A stable).
    Uses special path via MB02SD/MB02RD for real linear solve.
    """
    from ctrlsys import ab13dx

    n, m, p = 2, 1, 1

    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)
    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)

    omega = 0.0

    result, info = ab13dx('C', 'I', 'Z', n, m, p, a, e, b, c, d, omega)

    assert info == 0
    assert result > 0


def test_ab13dx_discrete():
    """
    Test AB13DX with discrete-time system.

    For discrete-time, lambda = exp(j*omega).
    """
    from ctrlsys import ab13dx

    n, m, p = 2, 1, 1

    a = np.array([
        [0.5, 0.0],
        [0.0, 0.3]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)
    b = np.array([[1.0], [0.5]], order='F', dtype=float)
    c = np.array([[1.0, 0.5]], order='F', dtype=float)
    d = np.array([[0.1]], order='F', dtype=float)

    omega = 0.5

    result, info = ab13dx('D', 'I', 'D', n, m, p, a, e, b, c, d, omega)

    assert info == 0
    assert result > 0


def test_ab13dx_with_d_matrix():
    """
    Test AB13DX with nonzero D matrix.
    """
    from ctrlsys import ab13dx

    n, m, p = 2, 1, 1

    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)
    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.5]], order='F', dtype=float)

    omega = 1.0

    result, info = ab13dx('C', 'I', 'D', n, m, p, a, e, b, c, d, omega)

    assert info == 0
    assert result > 0


def test_ab13dx_n0():
    """Test AB13DX with n=0 (static gain only)."""
    from ctrlsys import ab13dx

    n, m, p = 0, 1, 1

    a = np.zeros((1, 1), order='F', dtype=float)
    e = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, 1), order='F', dtype=float)
    c = np.zeros((1, 1), order='F', dtype=float)
    d = np.array([[2.0]], order='F', dtype=float)

    omega = 1.0

    result, info = ab13dx('C', 'I', 'D', n, m, p, a, e, b, c, d, omega)

    assert info == 0
    np.testing.assert_allclose(result, 2.0, rtol=1e-14)


def test_ab13dx_m0_or_p0():
    """Test AB13DX with m=0 or p=0 (zero transfer function)."""
    from ctrlsys import ab13dx

    n, m, p = 2, 0, 1

    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.zeros((n, 1), order='F', dtype=float)
    c = np.zeros((1, n), order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    omega = 1.0

    result, info = ab13dx('C', 'I', 'D', n, m, p, a, e, b, c, d, omega)

    assert info == 0
    np.testing.assert_allclose(result, 0.0, atol=1e-14)


def test_ab13dx_invalid_dico():
    """Test AB13DX with invalid DICO parameter."""
    from ctrlsys import ab13dx

    n, m, p = 2, 1, 1

    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)

    omega = 1.0

    result, info = ab13dx('X', 'I', 'Z', n, m, p, a, e, b, c, d, omega)

    assert info == -1
