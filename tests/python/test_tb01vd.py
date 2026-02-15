"""
Tests for TB01VD - Output normal form conversion for discrete-time systems.

TB01VD converts (A, B, C, D, x0) to parameter vector THETA using output normal form.
The matrix A must be stable (all eigenvalues inside unit circle for discrete-time).

Algorithm:
1. Solve Lyapunov equation A'*Q*A - Q = -scale^2*C'*C in Cholesky factor T
2. Transform system using T
3. QR factorization of transposed observability matrix
4. Extract parameters via N orthogonal transformations
"""
import numpy as np
import pytest

from ctrlsys import tb01vd, tb01vy


"""Test basic functionality."""

def test_2x2_stable_discrete_system():
    """
    Test 2x2 stable discrete-time system conversion.

    Random seed: 42 (for reproducibility)
    Uses small discrete-time stable system.
    """
    np.random.seed(42)
    n, m, l = 2, 1, 1

    a = np.array([
        [0.5, 0.1],
        [0.0, 0.3]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.5]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.5]
    ], order='F', dtype=float)

    d = np.array([
        [0.0]
    ], order='F', dtype=float)

    x0 = np.array([0.1, 0.2], dtype=float)

    theta, a_out, b_out, c_out, scale, info = tb01vd(
        n, m, l, a, b, c, d, x0, apply='N'
    )

    assert info == 0
    assert scale > 0.0

    ltheta_expected = n * (l + m + 1) + l * m
    assert theta.shape[0] >= ltheta_expected

    assert a_out.shape == (n, n)
    assert b_out.shape == (n, m)
    assert c_out.shape == (l, n)

def test_3x2x1_system():
    """
    Test 3-state, 2-input, 1-output system.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, l = 3, 2, 1

    a = 0.4 * np.random.randn(n, n)
    a = np.asfortranarray(a)

    b = np.random.randn(n, m)
    b = np.asfortranarray(b)

    c = np.random.randn(l, n)
    c = np.asfortranarray(c)

    d = np.random.randn(l, m)
    d = np.asfortranarray(d)

    x0 = np.random.randn(n)

    theta, a_out, b_out, c_out, scale, info = tb01vd(
        n, m, l, a, b, c, d, x0, apply='N'
    )

    assert info == 0
    assert scale > 0.0

    ltheta = n * (l + m + 1) + l * m
    assert len(theta) >= ltheta


"""Test round-trip: TB01VD -> TB01VY should recover original system."""

def test_roundtrip_2x2_system():
    """
    Test VD->VY round-trip preserves system behavior.

    TB01VD: (A,B,C,D,x0) -> THETA
    TB01VY: THETA -> (A',B',C',D',x0')

    Systems should produce identical outputs for same input sequence.
    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n, m, l = 2, 1, 1

    a = np.array([
        [0.6, 0.1],
        [-0.1, 0.4]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.5]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0]
    ], order='F', dtype=float)

    d = np.array([
        [0.1]
    ], order='F', dtype=float)

    x0 = np.array([0.0, 0.0], dtype=float)

    theta, a_vd, b_vd, c_vd, scale, info = tb01vd(
        n, m, l, a.copy(), b.copy(), c.copy(), d.copy(), x0.copy(), apply='N'
    )
    assert info == 0

    ltheta = n * (l + m + 1) + l * m
    a_vy, b_vy, c_vy, d_vy, x0_vy, info2 = tb01vy(n, m, l, theta[:ltheta], apply='N')
    assert info2 == 0

    u_seq = np.array([[1.0], [0.5], [0.2], [-0.1], [0.3]], dtype=float)

    def simulate(A, B, C, D, x_init, u_sequence):
        """Simulate discrete-time system."""
        y_out = []
        x = x_init.copy()
        for u in u_sequence:
            y = C @ x + D @ u.reshape(-1, 1)
            y_out.append(y.flatten())
            x = A @ x + B @ u.reshape(-1, 1)
        return np.array(y_out)

    y_orig = simulate(a, b, c, d, x0.reshape(-1, 1), u_seq)
    y_recon = simulate(a_vy, b_vy, c_vy, d_vy, x0_vy.reshape(-1, 1), u_seq)

    np.testing.assert_allclose(y_orig, y_recon, rtol=1e-10, atol=1e-12)

def test_roundtrip_3x2x2_apply_bijective():
    """
    Test round-trip with bijective mapping (apply='A').

    The bijective mapping relaxes constraint norm(THETAi) < 1.
    Random seed: 300 (for reproducibility)
    """
    np.random.seed(300)
    n, m, l = 3, 2, 2

    a = 0.3 * np.random.randn(n, n)
    a = np.asfortranarray(a)

    b = np.random.randn(n, m)
    b = np.asfortranarray(b)

    c = np.random.randn(l, n)
    c = np.asfortranarray(c)

    d = np.random.randn(l, m)
    d = np.asfortranarray(d)

    x0 = np.random.randn(n)

    theta, a_vd, b_vd, c_vd, scale, info = tb01vd(
        n, m, l, a.copy(), b.copy(), c.copy(), d.copy(), x0.copy(), apply='A'
    )
    assert info == 0

    ltheta = n * (l + m + 1) + l * m
    a_vy, b_vy, c_vy, d_vy, x0_vy, info2 = tb01vy(n, m, l, theta[:ltheta], apply='A')
    assert info2 == 0

    u_seq = np.random.randn(5, m)

    def simulate_mimo(A, B, C, D, x_init, u_sequence):
        y_out = []
        x = x_init.reshape(-1, 1)
        for u in u_sequence:
            y = C @ x + D @ u.reshape(-1, 1)
            y_out.append(y.flatten())
            x = A @ x + B @ u.reshape(-1, 1)
        return np.array(y_out)

    y_orig = simulate_mimo(a, b, c, d, x0, u_seq)
    y_recon = simulate_mimo(a_vy, b_vy, c_vy, d_vy, x0_vy, u_seq)

    np.testing.assert_allclose(y_orig, y_recon, rtol=1e-10, atol=1e-12)


"""Test edge cases."""

def test_n_zero():
    """Test N=0: quick return with D copied to THETA."""
    n, m, l = 0, 2, 2

    a = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, m), order='F', dtype=float)
    c = np.zeros((l, 1), order='F', dtype=float)
    d = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)
    x0 = np.zeros(1, dtype=float)

    theta, a_out, b_out, c_out, scale, info = tb01vd(
        n, m, l, a, b, c, d, x0, apply='N'
    )

    assert info == 0
    np.testing.assert_allclose(theta[:l*m].reshape((l, m), order='F'), d, rtol=1e-14)

def test_l_zero():
    """Test L=0: THETA contains B and x0."""
    n, m, l = 2, 2, 0

    a = np.array([
        [0.5, 0.1],
        [0.0, 0.3]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    c = np.zeros((1, n), order='F', dtype=float)
    d = np.zeros((1, m), order='F', dtype=float)
    x0 = np.array([0.5, 0.6], dtype=float)

    theta, a_out, b_out, c_out, scale, info = tb01vd(
        n, m, l, a, b, c, d, x0, apply='N'
    )

    assert info == 0
    b_from_theta = theta[:n*m].reshape((n, m), order='F')
    np.testing.assert_allclose(b_from_theta, b, rtol=1e-14)
    np.testing.assert_allclose(theta[n*m:n*m+n], x0, rtol=1e-14)

def test_m_zero():
    """Test M=0: only autonomous system with output."""
    n, m, l = 2, 0, 1

    a = np.array([
        [0.5, 0.1],
        [0.0, 0.3]
    ], order='F', dtype=float)

    b = np.zeros((n, 1), order='F', dtype=float)

    c = np.array([
        [1.0, 0.5]
    ], order='F', dtype=float)

    d = np.zeros((l, 1), order='F', dtype=float)
    x0 = np.array([0.1, 0.2], dtype=float)

    theta, a_out, b_out, c_out, scale, info = tb01vd(
        n, m, l, a, b, c, d, x0, apply='N'
    )

    assert info == 0


"""Test error conditions."""

def test_unstable_system():
    """Test unstable A (eigenvalue outside unit circle) returns info=2."""
    n, m, l = 2, 1, 1

    a = np.array([
        [1.5, 0.0],
        [0.0, 0.5]
    ], order='F', dtype=float)

    b = np.array([[1.0], [0.5]], order='F', dtype=float)
    c = np.array([[1.0, 0.5]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)
    x0 = np.array([0.0, 0.0], dtype=float)

    theta, a_out, b_out, c_out, scale, info = tb01vd(
        n, m, l, a, b, c, d, x0, apply='N'
    )

    assert info == 2

def test_invalid_apply():
    """Test invalid apply parameter raises error."""
    n, m, l = 2, 1, 1

    a = np.array([[0.5, 0.0], [0.0, 0.3]], order='F', dtype=float)
    b = np.array([[1.0], [0.5]], order='F', dtype=float)
    c = np.array([[1.0, 0.5]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)
    x0 = np.array([0.0, 0.0], dtype=float)

    with pytest.raises(ValueError):
        tb01vd(n, m, l, a, b, c, d, x0, apply='X')


"""Test THETA vector structure."""

def test_theta_layout():
    """
    Verify THETA layout matches documentation:
    - THETA(1:N*L)                      : parameters for A, C
    - THETA(N*L+1:N*(L+M))              : parameters for B
    - THETA(N*(L+M)+1:N*(L+M)+L*M)      : parameters for D
    - THETA(N*(L+M)+L*M+1:N*(L+M+1)+L*M): parameters for x0

    Random seed: 400 (for reproducibility)
    """
    np.random.seed(400)
    n, m, l = 3, 2, 2

    a = 0.3 * np.random.randn(n, n)
    a = np.asfortranarray(a)

    b = np.random.randn(n, m)
    b = np.asfortranarray(b)

    c = np.random.randn(l, n)
    c = np.asfortranarray(c)

    d = np.random.randn(l, m)
    d = np.asfortranarray(d)

    x0 = np.random.randn(n)

    theta, a_out, b_out, c_out, scale, info = tb01vd(
        n, m, l, a.copy(), b.copy(), c.copy(), d.copy(), x0.copy(), apply='N'
    )

    assert info == 0

    ltheta = n * (l + m + 1) + l * m
    assert len(theta) >= ltheta

    theta_b = theta[n*l : n*(l+m)].reshape((n, m), order='F')
    np.testing.assert_allclose(theta_b, b_out, rtol=1e-14)

    theta_d = theta[n*(l+m) : n*(l+m)+l*m].reshape((l, m), order='F')
    np.testing.assert_allclose(theta_d, d, rtol=1e-14)


"""Test larger systems for robustness."""

def test_5x3x2_system():
    """
    Test 5-state, 3-input, 2-output system.

    Random seed: 500 (for reproducibility)
    """
    np.random.seed(500)
    n, m, l = 5, 3, 2

    a = 0.2 * np.random.randn(n, n)
    a = np.asfortranarray(a)

    b = np.random.randn(n, m)
    b = np.asfortranarray(b)

    c = np.random.randn(l, n)
    c = np.asfortranarray(c)

    d = np.random.randn(l, m)
    d = np.asfortranarray(d)

    x0 = np.random.randn(n)

    theta, a_out, b_out, c_out, scale, info = tb01vd(
        n, m, l, a.copy(), b.copy(), c.copy(), d.copy(), x0.copy(), apply='N'
    )

    assert info == 0
    assert scale > 0.0

    ltheta = n * (l + m + 1) + l * m
    a_vy, b_vy, c_vy, d_vy, x0_vy, info2 = tb01vy(n, m, l, theta[:ltheta], apply='N')
    assert info2 == 0

    u_seq = np.random.randn(10, m)

    def simulate_mimo(A, B, C, D, x_init, u_sequence):
        y_out = []
        x = x_init.reshape(-1, 1)
        for u in u_sequence:
            y = C @ x + D @ u.reshape(-1, 1)
            y_out.append(y.flatten())
            x = A @ x + B @ u.reshape(-1, 1)
        return np.array(y_out)

    y_orig = simulate_mimo(a, b, c, d, x0, u_seq)
    y_recon = simulate_mimo(a_vy, b_vy, c_vy, d_vy, x0_vy, u_seq)

    np.testing.assert_allclose(y_orig, y_recon, rtol=1e-9, atol=1e-10)
