"""
Tests for AB05MD - Cascade (series) inter-connection of two systems.

AB05MD computes the state-space model (A,B,C,D) for the cascaded
inter-connection of two systems G1 and G2, where output of G1 feeds input of G2.
"""

import numpy as np
import pytest
from ctrlsys import ab05md


"""Basic functionality tests using SLICOT HTML doc example."""

def test_lower_block_diagonal_form():
    """
    Test UPLO='L' using SLICOT HTML doc example.

    System 1: N1=3, M1=2, P1=2
    System 2: N2=3, P2=2

    For UPLO='L':
    A = [A1,    0   ]    B = [ B1   ]
        [B2*C1, A2  ]        [B2*D1 ]

    C = [D2*C1, C2]      D = D2*D1
    """
    a1 = np.array([
        [1.0, 0.0, -1.0],
        [0.0, -1.0, 1.0],
        [1.0, 1.0, 2.0]
    ], order='F', dtype=float)

    b1 = np.array([
        [1.0, 2.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    c1 = np.array([
        [3.0, -2.0, 1.0],
        [0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    d1 = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a2 = np.array([
        [-3.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, -1.0, 2.0]
    ], order='F', dtype=float)

    b2 = np.array([
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    c2 = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, -1.0]
    ], order='F', dtype=float)

    d2 = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a, b, c, d, n, info = ab05md('L', 'N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == 6

    a_expected = np.array([
        [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -3.0, 0.0, 0.0],
        [-3.0, 2.0, -1.0, 1.0, 0.0, 1.0],
        [0.0, 2.0, 0.0, 0.0, -1.0, 2.0]
    ], order='F', dtype=float)

    b_expected = np.array([
        [1.0, 2.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    c_expected = np.array([
        [3.0, -1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, -1.0]
    ], order='F', dtype=float)

    d_expected = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b, b_expected, rtol=1e-14)
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)

def test_upper_block_diagonal_form():
    """
    Test UPLO='U' using same input data.

    For UPLO='U':
    A = [A2,    B2*C1]    B = [B2*D1]
        [0,     A1   ]        [ B1  ]

    C = [C2, D2*C1]      D = D2*D1
    """
    a1 = np.array([
        [1.0, 0.0, -1.0],
        [0.0, -1.0, 1.0],
        [1.0, 1.0, 2.0]
    ], order='F', dtype=float)

    b1 = np.array([
        [1.0, 2.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    c1 = np.array([
        [3.0, -2.0, 1.0],
        [0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    d1 = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a2 = np.array([
        [-3.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, -1.0, 2.0]
    ], order='F', dtype=float)

    b2 = np.array([
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    c2 = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, -1.0]
    ], order='F', dtype=float)

    d2 = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a, b, c, d, n, info = ab05md('U', 'N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == 6

    b2c1 = b2 @ c1
    b2d1 = b2 @ d1
    d2c1 = d2 @ c1
    d2d1 = d2 @ d1

    a_expected = np.zeros((6, 6), order='F', dtype=float)
    a_expected[:3, :3] = a2
    a_expected[:3, 3:] = b2c1
    a_expected[3:, 3:] = a1

    b_expected = np.zeros((6, 2), order='F', dtype=float)
    b_expected[:3, :] = b2d1
    b_expected[3:, :] = b1

    c_expected = np.zeros((2, 6), order='F', dtype=float)
    c_expected[:, :3] = c2
    c_expected[:, 3:] = d2c1

    d_expected = d2d1

    np.testing.assert_allclose(a, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b, b_expected, rtol=1e-14)
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)


"""Test mathematical properties of cascade connection."""

def test_cascade_state_space_equations():
    """
    Validate that cascaded system satisfies state-space equations.

    For cascade G2*G1:
    - y = G2(G1(u)) = (C2*C1)*x + D2*D1*u (steady state through)
    - Markov parameters: h(0) = D = D2*D1

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n1, m1, p1 = 2, 1, 2
    n2, p2 = 3, 1

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, p1).astype(float, order='F')
    c2 = np.random.randn(p2, n2).astype(float, order='F')
    d2 = np.random.randn(p2, p1).astype(float, order='F')

    a, b, c, d, n, info = ab05md('L', 'N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n1 + n2

    d_expected = d2 @ d1
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)

    cb = c @ b
    c1b1 = c1 @ b1
    c2b2 = c2 @ b2
    h1_expected = d2 @ c1b1 + c2 @ (b2 @ d1)
    np.testing.assert_allclose(cb, h1_expected, rtol=1e-13)

def test_cascade_transfer_function_equivalence():
    """
    Verify cascade connection preserves transfer function composition.

    For discrete-time system, simulate:
    y1 = G1(u), y2 = G2(y1) should equal y = G_cascade(u)

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n1, m1, p1 = 2, 1, 2
    n2, p2 = 2, 1

    a1 = np.diag([0.5, 0.3]).astype(float, order='F')
    b1 = np.array([[1.0], [0.5]], order='F', dtype=float)
    c1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    d1 = np.array([[0.1], [0.0]], order='F', dtype=float)

    a2 = np.diag([0.4, 0.6]).astype(float, order='F')
    b2 = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=float)
    c2 = np.array([[1.0, 1.0]], order='F', dtype=float)
    d2 = np.array([[0.5, 0.2]], order='F', dtype=float)

    a, b, c, d, n, info = ab05md('L', 'N', a1, b1, c1, d1, a2, b2, c2, d2)
    assert info == 0

    nsteps = 10
    u = np.ones((m1, nsteps), order='F', dtype=float)

    x1 = np.zeros((n1, 1), order='F', dtype=float)
    x2 = np.zeros((n2, 1), order='F', dtype=float)
    y_separate = np.zeros((p2, nsteps), order='F', dtype=float)

    for k in range(nsteps):
        uk = u[:, k:k+1]
        y1 = c1 @ x1 + d1 @ uk
        y2 = c2 @ x2 + d2 @ y1
        y_separate[:, k:k+1] = y2
        x1 = a1 @ x1 + b1 @ uk
        x2 = a2 @ x2 + b2 @ y1

    x_casc = np.zeros((n, 1), order='F', dtype=float)
    y_cascade = np.zeros((p2, nsteps), order='F', dtype=float)

    for k in range(nsteps):
        uk = u[:, k:k+1]
        y_cascade[:, k:k+1] = c @ x_casc + d @ uk
        x_casc = a @ x_casc + b @ uk

    np.testing.assert_allclose(y_cascade, y_separate, rtol=1e-14, atol=1e-15)


"""Edge cases and boundary conditions."""

def test_zero_state_first_system():
    """
    Test cascade with N1=0 (first system is static gain D1 only).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m1, p1 = 2, 2
    n2, p2 = 3, 2

    a1 = np.zeros((0, 0), order='F', dtype=float)
    b1 = np.zeros((0, m1), order='F', dtype=float)
    c1 = np.zeros((p1, 0), order='F', dtype=float)
    d1 = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=float)

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, p1).astype(float, order='F')
    c2 = np.random.randn(p2, n2).astype(float, order='F')
    d2 = np.random.randn(p2, p1).astype(float, order='F')

    a, b, c, d, n, info = ab05md('L', 'N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n2

    np.testing.assert_allclose(a, a2, rtol=1e-14)
    np.testing.assert_allclose(b, b2 @ d1, rtol=1e-14)
    np.testing.assert_allclose(c, c2, rtol=1e-14)
    np.testing.assert_allclose(d, d2 @ d1, rtol=1e-14)

def test_zero_state_second_system():
    """
    Test cascade with N2=0 (second system is static gain D2 only).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n1, m1, p1 = 3, 2, 2
    p2 = 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.zeros((0, 0), order='F', dtype=float)
    b2 = np.zeros((0, p1), order='F', dtype=float)
    c2 = np.zeros((p2, 0), order='F', dtype=float)
    d2 = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=float)

    a, b, c, d, n, info = ab05md('L', 'N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n1

    np.testing.assert_allclose(a, a1, rtol=1e-14)
    np.testing.assert_allclose(b, b1, rtol=1e-14)
    np.testing.assert_allclose(c, d2 @ c1, rtol=1e-14)
    np.testing.assert_allclose(d, d2 @ d1, rtol=1e-14)

def test_single_state_systems():
    """
    Test with minimal N1=1, N2=1 systems.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n1, m1, p1 = 1, 1, 1
    n2, p2 = 1, 1

    a1 = np.array([[0.5]], order='F', dtype=float)
    b1 = np.array([[1.0]], order='F', dtype=float)
    c1 = np.array([[2.0]], order='F', dtype=float)
    d1 = np.array([[0.1]], order='F', dtype=float)

    a2 = np.array([[0.3]], order='F', dtype=float)
    b2 = np.array([[1.5]], order='F', dtype=float)
    c2 = np.array([[0.8]], order='F', dtype=float)
    d2 = np.array([[0.2]], order='F', dtype=float)

    a, b, c, d, n, info = ab05md('L', 'N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == 2

    a_expected = np.array([
        [0.5, 0.0],
        [1.5 * 2.0, 0.3]
    ], order='F', dtype=float)

    b_expected = np.array([
        [1.0],
        [1.5 * 0.1]
    ], order='F', dtype=float)

    c_expected = np.array([
        [0.2 * 2.0, 0.8]
    ], order='F', dtype=float)

    d_expected = np.array([
        [0.2 * 0.1]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b, b_expected, rtol=1e-14)
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)


"""Test error handling and invalid input detection."""

def test_invalid_uplo():
    """Test invalid UPLO parameter."""
    a1 = np.array([[1.0]], order='F', dtype=float)
    b1 = np.array([[1.0]], order='F', dtype=float)
    c1 = np.array([[1.0]], order='F', dtype=float)
    d1 = np.array([[1.0]], order='F', dtype=float)
    a2 = np.array([[1.0]], order='F', dtype=float)
    b2 = np.array([[1.0]], order='F', dtype=float)
    c2 = np.array([[1.0]], order='F', dtype=float)
    d2 = np.array([[1.0]], order='F', dtype=float)

    with pytest.raises(ValueError, match="uplo"):
        ab05md('X', 'N', a1, b1, c1, d1, a2, b2, c2, d2)

def test_invalid_over():
    """Test invalid OVER parameter."""
    a1 = np.array([[1.0]], order='F', dtype=float)
    b1 = np.array([[1.0]], order='F', dtype=float)
    c1 = np.array([[1.0]], order='F', dtype=float)
    d1 = np.array([[1.0]], order='F', dtype=float)
    a2 = np.array([[1.0]], order='F', dtype=float)
    b2 = np.array([[1.0]], order='F', dtype=float)
    c2 = np.array([[1.0]], order='F', dtype=float)
    d2 = np.array([[1.0]], order='F', dtype=float)

    with pytest.raises(ValueError, match="over"):
        ab05md('L', 'X', a1, b1, c1, d1, a2, b2, c2, d2)

def test_dimension_mismatch_p1():
    """Test dimension mismatch: P1 (output of G1 != input of G2)."""
    a1 = np.array([[1.0]], order='F', dtype=float)
    b1 = np.array([[1.0]], order='F', dtype=float)
    c1 = np.array([[1.0], [2.0]], order='F', dtype=float)
    d1 = np.array([[1.0], [2.0]], order='F', dtype=float)
    a2 = np.array([[1.0]], order='F', dtype=float)
    b2 = np.array([[1.0, 2.0, 3.0]], order='F', dtype=float)
    c2 = np.array([[1.0]], order='F', dtype=float)
    d2 = np.array([[1.0, 2.0, 3.0]], order='F', dtype=float)

    with pytest.raises(ValueError, match="[pP]1"):
        ab05md('L', 'N', a1, b1, c1, d1, a2, b2, c2, d2)
