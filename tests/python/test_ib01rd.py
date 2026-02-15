"""Tests for IB01RD - Initial state estimation for discrete-time LTI systems.

IB01RD estimates the initial state x(0) given system matrices (A,B,C,D)
and input/output trajectories. The model is:
    x(k+1) = A*x(k) + B*u(k)
    y(k)   = C*x(k) + D*u(k)

Matrix A must be in real Schur form.
"""

import numpy as np
import pytest
from ctrlsys import ib01rd


def _simulate_system(a, b, c, d, u, x0):
    """Simulate discrete-time LTI system with given initial state."""
    nsmp, m = u.shape if u.ndim == 2 else (u.shape[0], 1)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    n = a.shape[0]
    l = c.shape[0]

    y = np.zeros((nsmp, l), order='F', dtype=float)
    x = x0.copy()

    for k in range(nsmp):
        y[k, :] = c @ x
        if m > 0 and d.size > 0:
            y[k, :] += d @ u[k, :]
        x_next = a @ x
        if m > 0 and b.size > 0:
            x_next += b @ u[k, :]
        x = x_next

    return y


def test_ib01rd_basic():
    """
    Validate basic functionality with simple SISO system.

    Test a stable 2nd order discrete system with known initial state.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, l = 2, 1, 1

    # A in real Schur form (upper quasi-triangular)
    a = np.array([[0.9, 0.2],
                  [0.0, 0.8]], order='F', dtype=float)
    b = np.array([[0.5],
                  [0.3]], order='F', dtype=float)
    c = np.array([[1.0, 0.5]], order='F', dtype=float)
    d = np.array([[0.1]], order='F', dtype=float)

    x0_true = np.array([1.0, -0.5], dtype=float)

    nsmp = 20
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = _simulate_system(a, b, c, d, u, x0_true)

    x0_est, rcond, iwarn, info = ib01rd('N', n, m, l, nsmp, a, b, c, d, u, y, 0.0)

    assert info == 0, f"IB01RD returned info={info}"
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-10, atol=1e-12)


def test_ib01rd_jobz():
    """
    Test JOB='Z' mode (D matrix is zero).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, l = 3, 2, 2

    # A in real Schur form
    a = np.array([[0.7, 0.3, 0.1],
                  [0.0, 0.8, 0.2],
                  [0.0, 0.0, 0.6]], order='F', dtype=float)
    b = np.array([[0.5, 0.1],
                  [0.3, 0.2],
                  [0.1, 0.4]], order='F', dtype=float)
    c = np.array([[1.0, 0.5, 0.2],
                  [0.3, 1.0, 0.4]], order='F', dtype=float)
    d = np.zeros((l, m), order='F', dtype=float)

    x0_true = np.array([0.5, -0.3, 0.8], dtype=float)

    nsmp = 30
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = _simulate_system(a, b, c, d, u, x0_true)

    x0_est, rcond, iwarn, info = ib01rd('Z', n, m, l, nsmp, a, b, c, d, u, y, 0.0)

    assert info == 0, f"IB01RD returned info={info}"
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-10, atol=1e-12)


def test_ib01rd_zero_initial_state():
    """
    Test with zero initial state - should return near-zero x0.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, l = 2, 1, 1

    a = np.array([[0.9, 0.1],
                  [0.0, 0.85]], order='F', dtype=float)
    b = np.array([[0.4],
                  [0.2]], order='F', dtype=float)
    c = np.array([[1.0, 0.3]], order='F', dtype=float)
    d = np.array([[0.05]], order='F', dtype=float)

    x0_true = np.zeros(n, dtype=float)

    nsmp = 25
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = _simulate_system(a, b, c, d, u, x0_true)

    x0_est, rcond, iwarn, info = ib01rd('N', n, m, l, nsmp, a, b, c, d, u, y, 0.0)

    assert info == 0
    np.testing.assert_allclose(x0_est, x0_true, atol=1e-12)


def test_ib01rd_mimo_system():
    """
    Test MIMO system (multiple inputs, multiple outputs).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, l = 4, 2, 3

    # A in real Schur form (upper triangular for simplicity)
    a = np.array([[0.8, 0.2, 0.1, 0.05],
                  [0.0, 0.75, 0.15, 0.1],
                  [0.0, 0.0, 0.7, 0.2],
                  [0.0, 0.0, 0.0, 0.65]], order='F', dtype=float)

    b = np.random.randn(n, m).astype(float, order='F') * 0.3
    c = np.random.randn(l, n).astype(float, order='F') * 0.5
    d = np.random.randn(l, m).astype(float, order='F') * 0.1

    x0_true = np.array([1.0, -0.5, 0.3, -0.2], dtype=float)

    nsmp = 40
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = _simulate_system(a, b, c, d, u, x0_true)

    x0_est, rcond, iwarn, info = ib01rd('N', n, m, l, nsmp, a, b, c, d, u, y, 0.0)

    assert info == 0
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-8, atol=1e-10)


def test_ib01rd_no_input():
    """
    Test autonomous system (M=0, no inputs).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, l = 3, 0, 2

    a = np.array([[0.9, 0.1, 0.05],
                  [0.0, 0.85, 0.1],
                  [0.0, 0.0, 0.8]], order='F', dtype=float)
    b = np.zeros((n, 1), order='F', dtype=float)
    c = np.array([[1.0, 0.5, 0.2],
                  [0.3, 1.0, 0.4]], order='F', dtype=float)
    d = np.zeros((l, 1), order='F', dtype=float)

    x0_true = np.array([2.0, -1.0, 0.5], dtype=float)

    nsmp = 20
    u = np.zeros((nsmp, 1), order='F', dtype=float)

    # Simulate without input
    y = np.zeros((nsmp, l), order='F', dtype=float)
    x = x0_true.copy()
    for k in range(nsmp):
        y[k, :] = c @ x
        x = a @ x

    x0_est, rcond, iwarn, info = ib01rd('Z', n, m, l, nsmp, a, b, c, d, u, y, 0.0)

    assert info == 0
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-10, atol=1e-12)


def test_ib01rd_state_space_property():
    """
    Validate mathematical property: recovered x0 produces correct output trajectory.

    Tests: y(k) = C*A^k*x0 + sum_{i=0}^{k-1} C*A^{k-1-i}*B*u(i) + D*u(k)
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m, l = 3, 1, 2

    a = np.array([[0.85, 0.15, 0.05],
                  [0.0, 0.8, 0.1],
                  [0.0, 0.0, 0.75]], order='F', dtype=float)
    b = np.array([[0.4],
                  [0.3],
                  [0.2]], order='F', dtype=float)
    c = np.array([[1.0, 0.5, 0.3],
                  [0.2, 1.0, 0.4]], order='F', dtype=float)
    d = np.array([[0.1],
                  [0.05]], order='F', dtype=float)

    x0_true = np.array([1.5, -0.8, 0.4], dtype=float)

    nsmp = 25
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = _simulate_system(a, b, c, d, u, x0_true)

    x0_est, rcond, iwarn, info = ib01rd('N', n, m, l, nsmp, a, b, c, d, u, y, 0.0)

    assert info == 0

    y_reconstructed = _simulate_system(a, b, c, d, u, x0_est)
    np.testing.assert_allclose(y_reconstructed, y, rtol=1e-10, atol=1e-12)


def test_ib01rd_2x2_block_schur():
    """
    Test with A having 2x2 block on diagonal (complex conjugate eigenvalues).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, l = 4, 1, 1

    # A with 2x2 block representing complex eigenvalues 0.7 +/- 0.3i
    # and two real eigenvalues
    a = np.array([[0.7, -0.3, 0.1, 0.05],
                  [0.3,  0.7, 0.1, 0.05],
                  [0.0,  0.0, 0.8, 0.1],
                  [0.0,  0.0, 0.0, 0.6]], order='F', dtype=float)

    b = np.array([[0.3],
                  [0.2],
                  [0.4],
                  [0.1]], order='F', dtype=float)
    c = np.array([[1.0, 0.5, 0.3, 0.2]], order='F', dtype=float)
    d = np.array([[0.05]], order='F', dtype=float)

    x0_true = np.array([1.0, 0.5, -0.3, 0.8], dtype=float)

    nsmp = 30
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = _simulate_system(a, b, c, d, u, x0_true)

    x0_est, rcond, iwarn, info = ib01rd('N', n, m, l, nsmp, a, b, c, d, u, y, 0.0)

    assert info == 0
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-9, atol=1e-11)


def test_ib01rd_invalid_job():
    """Test error handling for invalid JOB parameter."""
    n, m, l = 2, 1, 1
    a = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((l, n), order='F', dtype=float)
    d = np.ones((l, m), order='F', dtype=float)
    u = np.ones((n, m), order='F', dtype=float)
    y = np.ones((n, l), order='F', dtype=float)

    with pytest.raises(ValueError):
        ib01rd('X', n, m, l, n, a, b, c, d, u, y, 0.0)


def test_ib01rd_invalid_dimensions():
    """Test error handling for NSMP < N."""
    n, m, l = 5, 1, 1
    nsmp = 3  # Less than n

    a = np.eye(n, order='F', dtype=float) * 0.9
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((l, n), order='F', dtype=float)
    d = np.ones((l, m), order='F', dtype=float)
    u = np.ones((nsmp, m), order='F', dtype=float)
    y = np.ones((nsmp, l), order='F', dtype=float)

    with pytest.raises(ValueError):
        ib01rd('N', n, m, l, nsmp, a, b, c, d, u, y, 0.0)


def test_ib01rd_n_zero():
    """Test quick return when N=0."""
    n, m, l = 0, 1, 1
    nsmp = 5

    a = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, m), order='F', dtype=float)
    c = np.zeros((l, 1), order='F', dtype=float)
    d = np.ones((l, m), order='F', dtype=float)
    u = np.ones((nsmp, m), order='F', dtype=float)
    y = np.ones((nsmp, l), order='F', dtype=float)

    x0_est, rcond, iwarn, info = ib01rd('N', n, m, l, nsmp, a, b, c, d, u, y, 0.0)

    assert info == 0
    assert len(x0_est) == 0
