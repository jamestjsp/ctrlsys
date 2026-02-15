"""
Tests for TF01MY - Output sequence of linear time-invariant open-loop system

TF01MY computes output sequence y(1),...,y(NY) of discrete-time state-space model
(A,B,C,D) given initial state x(1) and input sequence u(1),...,u(NY).

The system is: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k)

Storage format differs from TF01MD:
- U is NY-by-M (k-th row = u(k)')
- Y is NY-by-P (k-th row = y(k)')
"""
import numpy as np
import pytest

try:
    from ctrlsys import tf01my
except ImportError:
    pytest.skip("tf01my not available", allow_module_level=True)


def test_tf01my_basic():
    """
    Test TF01MY with simple 2x2 system.

    Validates:
    - State evolution x(k+1) = A*x(k) + B*u(k)
    - Output equation y(k) = C*x(k) + D*u(k)
    - Final state computation

    Random seed: N/A (hand-designed test case)
    """
    n = 2
    m = 1
    p = 1
    ny = 3

    a = np.array([
        [0.5, 0.1],
        [0.0, 0.8]
    ], dtype=float, order='F')

    b = np.array([
        [1.0],
        [0.5]
    ], dtype=float, order='F')

    c = np.array([
        [1.0, 0.0]
    ], dtype=float, order='F')

    d = np.array([
        [0.0]
    ], dtype=float, order='F')

    # Input sequence u(k) for k=1,2,3 - rows are u(k)'
    u = np.array([
        [1.0],
        [0.5],
        [0.0]
    ], dtype=float, order='F')

    x = np.array([1.0, 0.5], dtype=float, order='F')

    # Expected outputs computed manually:
    # k=1: y(1) = C*x(1) + D*u(1) = 1.0*1.0 + 0.0*0.5 + 0.0*1.0 = 1.0
    #      x(2) = A*x(1) + B*u(1) = [0.5*1.0+0.1*0.5; 0.0*1.0+0.8*0.5] + [1.0; 0.5]
    #           = [0.55; 0.4] + [1.0; 0.5] = [1.55; 0.9]
    # k=2: y(2) = C*x(2) + D*u(2) = 1.0*1.55 + 0.0*0.9 + 0.0*0.5 = 1.55
    #      x(3) = A*x(2) + B*u(2) = [0.5*1.55+0.1*0.9; 0.0*1.55+0.8*0.9] + [0.5; 0.25]
    #           = [0.865; 0.72] + [0.5; 0.25] = [1.365; 0.97]
    # k=3: y(3) = C*x(3) + D*u(3) = 1.0*1.365 = 1.365
    #      x(4) = A*x(3) + B*u(3) = [0.5*1.365+0.1*0.97; 0.0*1.365+0.8*0.97]
    #           = [0.7795; 0.776]

    y_expected = np.array([
        [1.0],
        [1.55],
        [1.365]
    ], dtype=float, order='F')

    x_final_expected = np.array([0.7795, 0.776], dtype=float, order='F')

    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0
    np.testing.assert_allclose(y, y_expected, rtol=1e-14)
    np.testing.assert_allclose(x_out, x_final_expected, rtol=1e-14)


def test_tf01my_state_space_equations():
    """
    Test that state-space equations hold exactly.

    Mathematical property test:
    - x(k+1) = A*x(k) + B*u(k) for all k
    - y(k) = C*x(k) + D*u(k) for all k

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n = 3
    m = 2
    p = 2
    ny = 10

    a = np.random.randn(n, n).astype(float, order='F') * 0.5
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')
    u = np.random.randn(ny, m).astype(float, order='F')
    x0 = np.random.randn(n).astype(float, order='F')

    x = x0.copy()
    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0

    # Verify equations hold by manual simulation
    x_sim = x0.copy()
    y_sim = np.zeros((ny, p), dtype=float, order='F')

    for k in range(ny):
        u_k = u[k, :]
        y_sim[k, :] = c @ x_sim + d @ u_k
        x_sim = a @ x_sim + b @ u_k

    np.testing.assert_allclose(y, y_sim, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(x_out, x_sim, rtol=1e-14, atol=1e-15)


def test_tf01my_markov_parameters():
    """
    Test Markov parameter property for impulse response.

    For impulse input u = [1, 0, 0, ...]:
    - y(0) = D (direct feedthrough)
    - y(k) = C * A^(k-1) * B for k >= 1

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n = 3
    m = 1
    p = 1
    ny = 8

    a = np.diag([0.9, 0.7, 0.5]).astype(float, order='F')
    b = np.array([[1.0], [1.0], [1.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.5, 0.3]], dtype=float, order='F')
    d = np.array([[0.1]], dtype=float, order='F')

    # Impulse input
    u = np.zeros((ny, m), dtype=float, order='F')
    u[0, 0] = 1.0

    x0 = np.zeros(n, dtype=float, order='F')
    x = x0.copy()

    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0

    # y(0) = D
    np.testing.assert_allclose(y[0], d.flatten(), rtol=1e-14)

    # y(k) = C * A^(k-1) * B for k >= 1
    for k in range(1, min(5, ny)):
        markov_k = c @ np.linalg.matrix_power(a, k - 1) @ b
        np.testing.assert_allclose(y[k], markov_k.flatten(), rtol=1e-14, atol=1e-15)


def test_tf01my_zero_states():
    """Test TF01MY with N=0 (non-dynamic system)."""
    n = 0
    m = 2
    p = 1
    ny = 3

    a = np.zeros((0, 0), dtype=float, order='F')
    b = np.zeros((0, m), dtype=float, order='F')
    c = np.zeros((p, 0), dtype=float, order='F')
    d = np.array([[1.5, 0.5]], dtype=float, order='F')

    u = np.array([
        [1.0, 2.0],
        [0.5, 1.0],
        [0.0, 0.5]
    ], dtype=float, order='F')

    x = np.zeros(0, dtype=float, order='F')

    # y(k) = D * u(k)
    y_expected = np.array([
        [1.5 * 1.0 + 0.5 * 2.0],
        [1.5 * 0.5 + 0.5 * 1.0],
        [1.5 * 0.0 + 0.5 * 0.5]
    ], dtype=float, order='F')

    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0
    np.testing.assert_allclose(y, y_expected, rtol=1e-14)
    assert len(x_out) == 0


def test_tf01my_zero_inputs():
    """Test TF01MY with M=0 (no inputs)."""
    n = 2
    m = 0
    p = 1
    ny = 3

    a = np.array([
        [0.9, 0.0],
        [0.0, 0.8]
    ], dtype=float, order='F')

    b = np.zeros((n, 0), dtype=float, order='F')
    c = np.array([[1.0, 1.0]], dtype=float, order='F')
    d = np.zeros((p, 0), dtype=float, order='F')
    u = np.zeros((ny, 0), dtype=float, order='F')

    x = np.array([2.0, 1.0], dtype=float, order='F')

    # k=1: y(1) = C*x(1) = 2.0 + 1.0 = 3.0, x(2) = A*x(1) = [1.8, 0.8]
    # k=2: y(2) = C*x(2) = 1.8 + 0.8 = 2.6, x(3) = A*x(2) = [1.62, 0.64]
    # k=3: y(3) = C*x(3) = 1.62 + 0.64 = 2.26, x(4) = [1.458, 0.512]

    y_expected = np.array([
        [3.0],
        [2.6],
        [2.26]
    ], dtype=float, order='F')

    x_final_expected = np.array([1.458, 0.512], dtype=float, order='F')

    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0
    np.testing.assert_allclose(y, y_expected, rtol=1e-14)
    np.testing.assert_allclose(x_out, x_final_expected, rtol=1e-14)


def test_tf01my_zero_outputs():
    """Test TF01MY with NY=0."""
    n = 2
    m = 1
    p = 1
    ny = 0

    a = np.array([[0.5, 0.1], [0.0, 0.8]], dtype=float, order='F')
    b = np.array([[1.0], [0.5]], dtype=float, order='F')
    c = np.array([[1.0, 0.0]], dtype=float, order='F')
    d = np.array([[0.0]], dtype=float, order='F')
    u = np.zeros((0, m), dtype=float, order='F')
    x = np.array([1.0, 0.5], dtype=float, order='F')

    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0
    assert y.shape[0] == 0


def test_tf01my_zero_output_channels():
    """Test TF01MY with P=0."""
    n = 2
    m = 1
    p = 0
    ny = 3

    a = np.array([[0.5, 0.1], [0.0, 0.8]], dtype=float, order='F')
    b = np.array([[1.0], [0.5]], dtype=float, order='F')
    c = np.zeros((0, n), dtype=float, order='F')
    d = np.zeros((0, m), dtype=float, order='F')
    u = np.array([[1.0], [0.5], [0.0]], dtype=float, order='F')
    x = np.array([1.0, 0.5], dtype=float, order='F')

    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0
    assert y.shape[1] == 0


def test_tf01my_mimo():
    """
    Test TF01MY with MIMO system.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n = 4
    m = 2
    p = 3
    ny = 15

    a = np.random.randn(n, n).astype(float, order='F') * 0.3
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    u = np.random.randn(ny, m).astype(float, order='F')
    x0 = np.random.randn(n).astype(float, order='F')

    x = x0.copy()
    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0
    assert y.shape == (ny, p)
    assert x_out.shape == (n,)

    # Verify by manual simulation
    x_sim = x0.copy()
    for k in range(ny):
        y_k = c @ x_sim + d @ u[k, :]
        np.testing.assert_allclose(y[k, :], y_k, rtol=1e-13, atol=1e-14)
        x_sim = a @ x_sim + b @ u[k, :]

    np.testing.assert_allclose(x_out, x_sim, rtol=1e-13, atol=1e-14)


def test_tf01my_large_workspace():
    """
    Test TF01MY with larger problem to trigger BLAS3 path.

    The Fortran code uses BLAS3 when LDWORK >= 2*N and problem is large.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n = 8
    m = 3
    p = 4
    ny = 50

    a = np.random.randn(n, n).astype(float, order='F') * 0.2
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    u = np.random.randn(ny, m).astype(float, order='F')
    x0 = np.random.randn(n).astype(float, order='F')

    x = x0.copy()
    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0

    # Verify by manual simulation
    x_sim = x0.copy()
    for k in range(ny):
        y_k = c @ x_sim + d @ u[k, :]
        np.testing.assert_allclose(y[k, :], y_k, rtol=1e-12, atol=1e-13)
        x_sim = a @ x_sim + b @ u[k, :]

    np.testing.assert_allclose(x_out, x_sim, rtol=1e-12, atol=1e-13)


def test_tf01my_workspace_query():
    """
    Test workspace query functionality (LDWORK = -1).

    This is handled internally by Python wrapper.
    """
    n = 4
    m = 2
    p = 2
    ny = 10

    a = np.eye(n, dtype=float, order='F') * 0.5
    b = np.ones((n, m), dtype=float, order='F')
    c = np.ones((p, n), dtype=float, order='F')
    d = np.zeros((p, m), dtype=float, order='F')
    u = np.ones((ny, m), dtype=float, order='F')
    x = np.zeros(n, dtype=float, order='F')

    y, x_out, info = tf01my(a, b, c, d, u, x)

    assert info == 0
    assert y.shape == (ny, p)
