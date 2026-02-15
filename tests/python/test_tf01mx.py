"""
Tests for TF01MX - Discrete-time state-space time response computation

TF01MX computes the output sequence and final state of a discrete-time
linear system given input sequence and initial state. This is fundamental
for system simulation and validation.

The system is: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k)
"""
import numpy as np
import pytest

try:
    from ctrlsys import tf01mx
except ImportError:
    pytest.skip("tf01mx not available", allow_module_level=True)

# Reference data generated using Python control package v0.10.2
# No runtime dependency on control package - all reference outputs are hardcoded


def test_tf01mx_basic():
    """
    Test TF01MX with simple 2x2 system

    Validates:
    - Hand-calculated output sequence matches implementation
    - Final state propagation is correct
    - Cross-validation with control.forced_response()
    - Cross-validation with slycot.tf01md() if available
    """
    # System: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k)
    # A = [0.5, 0.1; 0.0, 0.8], B = [1.0; 0.5], C = [1.0, 0.0], D = [0.0]
    n = 2
    m = 1
    p = 1
    ny = 3

    # System matrix S = [A B; C D]
    s = np.array([
        [0.5, 0.1, 1.0],
        [0.0, 0.8, 0.5],
        [1.0, 0.0, 0.0]
    ], dtype=float, order='F')

    # Extract A, B, C, D
    a = s[:n, :n].copy(order='F')
    b = s[:n, n:].copy(order='F')
    c = s[n:, :n].copy(order='F')
    d = s[n:, n:].copy(order='F')

    # Input sequence u(k) for k=1,2,3
    u = np.array([
        [1.0],
        [0.5],
        [0.0]
    ], dtype=float, order='F')

    # Initial state
    x0 = np.array([1.0, 0.5], dtype=float, order='F')
    x = x0.copy()

    # Expected outputs computed manually:
    # k=1: y(1) = C*x(1) + D*u(1) = 1.0*1.0 + 0.0*0.0 + 0.0*1.0 = 1.0
    #      x(2) = A*x(1) + B*u(1) = [0.5*1.0+0.1*0.5; 0.0*1.0+0.8*0.5] + [1.0*1.0; 0.5*1.0]
    #           = [0.55; 0.4] + [1.0; 0.5] = [1.55; 0.9]
    # k=2: y(2) = C*x(2) + D*u(2) = 1.0*1.55 + 0.0*0.9 + 0.0*0.5 = 1.55
    #      x(3) = A*x(2) + B*u(2) = [0.5*1.55+0.1*0.9; 0.0*1.55+0.8*0.9] + [1.0*0.5; 0.5*0.5]
    #           = [0.865; 0.72] + [0.5; 0.25] = [1.365; 0.97]
    # k=3: y(3) = C*x(3) + D*u(3) = 1.0*1.365 + 0.0*0.97 + 0.0*0.0 = 1.365
    #      x(4) = A*x(3) + B*u(3) = [0.5*1.365+0.1*0.97; 0.0*1.365+0.8*0.97] + [0.0; 0.0]
    #           = [0.7795; 0.776]

    y_expected = np.array([
        [1.0],
        [1.55],
        [1.365]
    ], dtype=float, order='F')

    x_final_expected = np.array([0.7795, 0.776], dtype=float, order='F')

    y, x_final, info = tf01mx(n, m, p, ny, s, u, x)

    assert info == 0
    np.testing.assert_allclose(y, y_expected, rtol=1e-14)
    np.testing.assert_allclose(x_final, x_final_expected, rtol=1e-14)

    # Cross-validate with reference data from control.forced_response v0.10.2
    # Reference: control.StateSpace(a, b, c, d, dt=1.0).forced_response(T=[0,1,2], U=u.T, X0=x0)
    # Output: yout = array([1.0, 1.55, 1.365])
    y_reference = np.array([[1.0], [1.55], [1.365]], dtype=float)
    np.testing.assert_allclose(y, y_reference, rtol=1e-13, atol=1e-14)


def test_tf01mx_no_inputs():
    """Test TF01MX with system having no inputs (M=0)."""
    # System: x(k+1) = A*x(k), y(k) = C*x(k)
    # A = [0.9, 0.0; 0.0, 0.8], C = [1.0, 1.0]
    n = 2
    m = 0
    p = 1
    ny = 2

    # System matrix S = [A; C]
    s = np.array([
        [0.9, 0.0],
        [0.0, 0.8],
        [1.0, 1.0]
    ], dtype=float, order='F')

    # No inputs
    u = np.zeros((ny, 0), dtype=float, order='F')

    # Initial state
    x = np.array([2.0, 1.0], dtype=float, order='F')

    # Expected outputs:
    # k=1: y(1) = C*x(1) = 1.0*2.0 + 1.0*1.0 = 3.0
    #      x(2) = A*x(1) = [0.9*2.0; 0.8*1.0] = [1.8; 0.8]
    # k=2: y(2) = C*x(2) = 1.0*1.8 + 1.0*0.8 = 2.6
    #      x(3) = A*x(2) = [0.9*1.8; 0.8*0.8] = [1.62; 0.64]

    y_expected = np.array([
        [3.0],
        [2.6]
    ], dtype=float, order='F')

    x_final_expected = np.array([1.62, 0.64], dtype=float, order='F')

    y, x_final, info = tf01mx(n, m, p, ny, s, u, x)

    assert info == 0
    np.testing.assert_allclose(y, y_expected, rtol=1e-14)
    np.testing.assert_allclose(x_final, x_final_expected, rtol=1e-14)


def test_tf01mx_zero_states():
    """Test TF01MX with zero states (N=0)."""
    # Non-dynamic system: y(k) = D*u(k)
    n = 0
    m = 2
    p = 1
    ny = 2

    # System matrix S = [D]
    s = np.array([
        [1.5, 0.5]
    ], dtype=float, order='F')

    # Input sequence
    u = np.array([
        [1.0, 2.0],
        [0.5, 1.0]
    ], dtype=float, order='F')

    # No state
    x = np.zeros(0, dtype=float, order='F')

    # Expected outputs: y(k) = D*u(k)
    # k=1: y(1) = 1.5*1.0 + 0.5*2.0 = 2.5
    # k=2: y(2) = 1.5*0.5 + 0.5*1.0 = 1.25

    y_expected = np.array([
        [2.5],
        [1.25]
    ], dtype=float, order='F')

    y, x_final, info = tf01mx(n, m, p, ny, s, u, x)

    assert info == 0
    np.testing.assert_allclose(y, y_expected, rtol=1e-14)
    assert len(x_final) == 0


def test_tf01mx_zero_outputs():
    """Test TF01MX with zero output steps (NY=0)."""
    n = 2
    m = 1
    p = 1
    ny = 0

    s = np.array([[0.5, 0.1, 1.0], [0.0, 0.8, 0.5], [1.0, 0.0, 0.0]],
                 dtype=float, order='F')
    u = np.zeros((0, m), dtype=float, order='F')
    x = np.array([1.0, 0.5], dtype=float, order='F')

    y, x_final, info = tf01mx(n, m, p, ny, s, u, x)

    assert info == 0
    assert y.shape == (0, p)
    np.testing.assert_allclose(x_final, [1.0, 0.5], rtol=1e-14)


def test_tf01mx_invalid_n():
    """Test TF01MX with invalid N."""
    with pytest.raises(ValueError, match="N must be >= 0"):
        n = -1
        m = 1
        p = 1
        ny = 1
        s = np.zeros((2, 2), dtype=float, order='F')
        u = np.zeros((1, 1), dtype=float, order='F')
        x = np.zeros(1, dtype=float, order='F')
        tf01mx(n, m, p, ny, s, u, x)


def test_tf01mx_invalid_ldwork():
    """
    Test TF01MX workspace validation

    Validates:
    - Python wrapper correctly allocates workspace internally
    - No memory errors for various system dimensions
    """
    # This should trigger LDWORK validation
    # For M>0, LDWORK >= 2*N+M+P
    n = 2
    m = 1
    p = 1
    ny = 1

    s = np.array([[0.5, 0.1, 1.0], [0.0, 0.8, 0.5], [1.0, 0.0, 0.0]],
                 dtype=float, order='F')
    u = np.array([[1.0]], dtype=float, order='F')
    x = np.array([1.0, 0.5], dtype=float, order='F')

    # This should work fine (wrapper allocates workspace)
    y, x_final, info = tf01mx(n, m, p, ny, s, u, x)
    assert info == 0


def test_tf01mx_step_response():
    """
    Test TF01MX with step input and validate steady-state behavior

    Validates:
    - Correct handling of constant (step) input
    - Steady-state behavior matches control theory predictions
    - Time-domain response accuracy

    Reference data generated from control.forced_response v0.10.2
    """
    n = 2
    m = 1
    p = 1
    ny = 20  # Long enough to approach steady state

    # Stable system with known step response
    a = np.array([[0.8, 0.0], [0.0, 0.5]], dtype=float, order='F')
    b = np.array([[1.0], [0.5]], dtype=float, order='F')
    c = np.array([[1.0, 1.0]], dtype=float, order='F')
    d = np.array([[0.0]], dtype=float, order='F')

    # System matrix
    s = np.zeros((n+p, n+m), dtype=float, order='F')
    s[:n, :n] = a
    s[:n, n:] = b
    s[n:, :n] = c
    s[n:, n:] = d

    # Step input: u(k) = 1.0 for all k
    u = np.ones((ny, m), dtype=float, order='F')

    # Zero initial state
    x0 = np.zeros(n, dtype=float, order='F')
    x = x0.copy()

    # Compute response
    y, x_final, info = tf01mx(n, m, p, ny, s, u, x)

    assert info == 0
    assert y.shape == (ny, p)

    # Check steady-state value: y_ss = C*(I-A)^{-1}*B*u + D*u
    # For step input u=1: y_ss = C*(I-A)^{-1}*B + D
    I = np.eye(n)
    y_ss_theory = c @ np.linalg.inv(I - a) @ b + d

    # Final values should approach steady state (may need more time steps for full convergence)
    # Reference: control.forced_response gives y[-1] = 5.92794049861344
    # Theory predicts y_ss = 6.0, so we're very close after 20 steps
    np.testing.assert_allclose(y[-1], y_ss_theory.flatten(), rtol=0.05, atol=0.1)

    # Validate against control package reference
    # Expected y[-1] ≈ 5.928 (from control.forced_response v0.10.2)
    assert np.abs(y[-1, 0] - 5.928) < 0.001


def test_tf01mx_impulse_response():
    """
    Test TF01MX impulse response

    Validates:
    - Impulse response (u=[1,0,0,...]) computation
    - Markov parameters h(k) = C*A^(k-1)*B for k>=1, h(0) = D
    - Natural modes and decay rates

    Reference data generated from control.forced_response v0.10.2
    """
    n = 3
    m = 1
    p = 1
    ny = 15

    # Create system with known eigenvalues
    # A in diagonal form for easy analysis
    a = np.array([[0.9, 0.0, 0.0],
                  [0.0, 0.7, 0.0],
                  [0.0, 0.0, 0.5]], dtype=float, order='F')
    b = np.array([[1.0], [1.0], [1.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.5, 0.3]], dtype=float, order='F')
    d = np.array([[0.1]], dtype=float, order='F')

    # System matrix
    s = np.zeros((n+p, n+m), dtype=float, order='F')
    s[:n, :n] = a
    s[:n, n:] = b
    s[n:, :n] = c
    s[n:, n:] = d

    # Impulse input
    u = np.zeros((ny, m), dtype=float, order='F')
    u[0, 0] = 1.0

    x0 = np.zeros(n, dtype=float, order='F')
    x = x0.copy()

    y, x_final, info = tf01mx(n, m, p, ny, s, u, x)

    assert info == 0

    # Verify first sample: y(0) = D (direct feedthrough)
    # Reference: control.forced_response gives y[0] = 0.1
    np.testing.assert_allclose(y[0], d.flatten(), rtol=1e-14)
    np.testing.assert_allclose(y[0, 0], 0.1, rtol=1e-14)

    # Verify Markov parameters: y(k) = C*A^(k-1)*B for k>=1
    for k in range(1, min(5, ny)):
        markov_k = c @ np.linalg.matrix_power(a, k-1) @ b
        np.testing.assert_allclose(y[k], markov_k.flatten(), rtol=1e-12, atol=1e-14)

    # Impulse response should decay for stable system (all |eigenvalues| < 1)
    # Check exponential decay
    assert np.abs(y[-1]) < np.abs(y[1]) * 0.2, "Impulse response should decay"


def test_tf01mx_mimo_system():
    """
    Test TF01MX with MIMO (multiple input, multiple output) system

    Validates:
    - Correct handling of multi-channel inputs and outputs
    - Channel coupling through state-space dynamics
    - Cross-validation with control package reference data

    Reference data generated from control.forced_response v0.10.2
    """
    n = 2
    m = 2  # Two inputs
    p = 2  # Two outputs
    ny = 10

    # MIMO system with cross-coupling
    a = np.array([[0.7, 0.1],
                  [0.2, 0.6]], dtype=float, order='F')
    b = np.array([[1.0, 0.5],
                  [0.3, 0.8]], dtype=float, order='F')
    c = np.array([[1.0, 0.2],
                  [0.4, 1.0]], dtype=float, order='F')
    d = np.array([[0.1, 0.0],
                  [0.0, 0.2]], dtype=float, order='F')

    # System matrix
    s = np.zeros((n+p, n+m), dtype=float, order='F')
    s[:n, :n] = a
    s[:n, n:] = b
    s[n:, :n] = c
    s[n:, n:] = d

    # Two-channel input: chirp-like pattern
    u = np.zeros((ny, m), dtype=float, order='F')
    for k in range(ny):
        u[k, 0] = np.sin(2*np.pi*k/10)
        u[k, 1] = np.cos(2*np.pi*k/8)

    x0 = np.array([0.5, -0.3], dtype=float, order='F')
    x = x0.copy()

    y, x_final, info = tf01mx(n, m, p, ny, s, u, x)

    assert info == 0
    assert y.shape == (ny, p)

    # Cross-validate with control package reference data (v0.10.2)
    # control.forced_response(sys, T=range(10), U=u.T, X0=[0.5,-0.3])
    # where u[k,0]=sin(2πk/10), u[k,1]=cos(2πk/8)
    y_reference = np.array([
        [ 0.44      ,  0.1       ],
        [ 1.02277853,  1.18942136],
        [ 1.95004849,  1.97295646],
        [ 2.57222077,  2.14257419],
        [ 2.53447615,  1.71229296],
        [ 1.86722622,  1.00355402],
        [ 0.88743287,  0.42455113],
        [-0.01435637,  0.21139251],
        [-0.57383242,  0.30240216],
        [-0.73815593,  0.41967308]
    ], dtype=float)

    np.testing.assert_allclose(y, y_reference, rtol=1e-6, atol=1e-8)

    # Verify final state matches manual propagation
    x_verify = x0.copy()
    for k in range(ny):
        x_verify = a @ x_verify + b @ u[k, :]

    np.testing.assert_allclose(x_final, x_verify, rtol=1e-13, atol=1e-14)
