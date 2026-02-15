"""
Tests for AB05OD - Rowwise concatenation of two systems in state-space form.

AB05OD computes the state-space model (A,B,C,D) for rowwise concatenation
(parallel inter-connection on outputs, with separate inputs) of two systems:
    Y = G1*U1 + alpha*G2*U2

The interconnected system has:
- A = [[A1, 0], [0, A2]]  (block diagonal)
- B = [[B1, 0], [0, B2]]  (block diagonal)
- C = [C1, alpha*C2]      (rowwise concatenation)
- D = [D1, alpha*D2]      (rowwise concatenation)
"""

import numpy as np
import pytest
from ctrlsys import ab05od


"""Basic functionality tests using SLICOT HTML doc example."""

def test_html_example_alpha_one():
    """
    Test rowwise concatenation using SLICOT HTML doc example.

    System 1: N1=3, M1=2, P1=2
    System 2: N2=3, M2=2
    ALPHA=1.0

    From AB05OD.html Program Data and Results.
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

    alpha = 1.0

    a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

    assert info == 0
    assert n == 6
    assert m == 4

    a_expected = np.array([
        [1.0000, 0.0000, -1.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, -1.0000, 1.0000, 0.0000, 0.0000, 0.0000],
        [1.0000, 1.0000, 2.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, -3.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, -1.0000, 2.0000]
    ], order='F', dtype=float)

    b_expected = np.array([
        [1.0000, 2.0000, 0.0000, 0.0000],
        [1.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, -1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 2.0000]
    ], order='F', dtype=float)

    c_expected = np.array([
        [3.0000, -2.0000, 1.0000, 1.0000, 1.0000, 0.0000],
        [0.0000, 1.0000, 0.0000, 1.0000, 1.0000, -1.0000]
    ], order='F', dtype=float)

    d_expected = np.array([
        [1.0000, 0.0000, 1.0000, 1.0000],
        [0.0000, 1.0000, 0.0000, 1.0000]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a, a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b, b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(c, c_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(d, d_expected, rtol=1e-3, atol=1e-4)

def test_alpha_negative():
    """
    Test rowwise concatenation with alpha=-2.0 (scaled second system).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n1, m1, p1 = 2, 2, 2
    n2, m2 = 2, 2

    a1 = np.random.randn(n1, n1).astype(float, order='F')
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F')
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p1, n2).astype(float, order='F')
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    alpha = -2.0

    a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

    assert info == 0
    assert n == n1 + n2
    assert m == m1 + m2

    a_expected = np.zeros((n, n), order='F', dtype=float)
    a_expected[:n1, :n1] = a1
    a_expected[n1:, n1:] = a2

    b_expected = np.zeros((n, m), order='F', dtype=float)
    b_expected[:n1, :m1] = b1
    b_expected[n1:, m1:] = b2

    c_expected = np.hstack([c1, alpha * c2])
    d_expected = np.hstack([d1, alpha * d2])

    np.testing.assert_allclose(a, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b, b_expected, rtol=1e-14)
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)


"""Test mathematical properties of rowwise concatenation."""

def test_block_diagonal_structure_a():
    """
    Validate A matrix is block diagonal: A = [[A1, 0], [0, A2]].

    The state transition matrix must have zero off-diagonal blocks.
    Random seed: 100 (for reproducibility)
    """
    np.random.seed(100)
    n1, m1, p1 = 3, 2, 2
    n2, m2 = 2, 2

    a1 = np.random.randn(n1, n1).astype(float, order='F')
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F')
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p1, n2).astype(float, order='F')
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    alpha = 1.5

    a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

    assert info == 0

    np.testing.assert_allclose(a[:n1, :n1], a1, rtol=1e-14)
    np.testing.assert_allclose(a[n1:, n1:], a2, rtol=1e-14)
    np.testing.assert_allclose(a[:n1, n1:], np.zeros((n1, n2)), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(a[n1:, :n1], np.zeros((n2, n1)), rtol=1e-14, atol=1e-15)

def test_block_diagonal_structure_b():
    """
    Validate B matrix is block diagonal: B = [[B1, 0], [0, B2]].

    The input matrix must have zero off-diagonal blocks.
    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n1, m1, p1 = 3, 2, 2
    n2, m2 = 2, 3

    a1 = np.random.randn(n1, n1).astype(float, order='F')
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F')
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p1, n2).astype(float, order='F')
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    alpha = 0.5

    a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

    assert info == 0

    np.testing.assert_allclose(b[:n1, :m1], b1, rtol=1e-14)
    np.testing.assert_allclose(b[n1:, m1:], b2, rtol=1e-14)
    np.testing.assert_allclose(b[:n1, m1:], np.zeros((n1, m2)), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(b[n1:, :m1], np.zeros((n2, m1)), rtol=1e-14, atol=1e-15)

def test_output_matrix_concatenation():
    """
    Validate C matrix: C = [C1, alpha*C2] (rowwise concatenation).

    Random seed: 300 (for reproducibility)
    """
    np.random.seed(300)
    n1, m1, p1 = 3, 2, 2
    n2, m2 = 2, 2

    a1 = np.random.randn(n1, n1).astype(float, order='F')
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F')
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p1, n2).astype(float, order='F')
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    for alpha in [1.0, -1.0, 2.5, 0.0]:
        a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

        assert info == 0

        np.testing.assert_allclose(c[:, :n1], c1, rtol=1e-14)
        np.testing.assert_allclose(c[:, n1:], alpha * c2, rtol=1e-14)

def test_feedthrough_matrix_concatenation():
    """
    Validate D matrix: D = [D1, alpha*D2] (rowwise concatenation).

    Random seed: 400 (for reproducibility)
    """
    np.random.seed(400)
    n1, m1, p1 = 2, 2, 3
    n2, m2 = 3, 2

    a1 = np.random.randn(n1, n1).astype(float, order='F')
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F')
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p1, n2).astype(float, order='F')
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    for alpha in [1.0, -1.0, 3.14159, 0.0]:
        a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

        assert info == 0

        np.testing.assert_allclose(d[:, :m1], d1, rtol=1e-14)
        np.testing.assert_allclose(d[:, m1:], alpha * d2, rtol=1e-14)

def test_state_space_simulation():
    """
    Validate state-space equations for parallel system.

    Test that:
        x(k+1) = A*x(k) + B*u(k)
        y(k) = C*x(k) + D*u(k)

    equals:
        x1(k+1) = A1*x1(k) + B1*u1(k)
        x2(k+1) = A2*x2(k) + B2*u2(k)
        y(k) = C1*x1(k) + D1*u1(k) + alpha*(C2*x2(k) + D2*u2(k))

    Random seed: 500 (for reproducibility)
    """
    np.random.seed(500)
    n1, m1, p1 = 2, 2, 2
    n2, m2 = 2, 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p1, n2).astype(float, order='F')
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    alpha = 1.5

    a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

    assert info == 0

    x1 = np.array([[1.0], [0.5]], order='F', dtype=float)
    x2 = np.array([[-0.5], [1.0]], order='F', dtype=float)
    x_combined = np.vstack([x1, x2])

    u1 = np.array([[0.5], [-0.3]], order='F', dtype=float)
    u2 = np.array([[0.2], [0.8]], order='F', dtype=float)
    u_combined = np.vstack([u1, u2])

    y_parallel = c1 @ x1 + d1 @ u1 + alpha * (c2 @ x2 + d2 @ u2)
    y_combined = c @ x_combined + d @ u_combined
    np.testing.assert_allclose(y_combined, y_parallel, rtol=1e-14)

    x1_next = a1 @ x1 + b1 @ u1
    x2_next = a2 @ x2 + b2 @ u2
    x_next_parallel = np.vstack([x1_next, x2_next])
    x_next_combined = a @ x_combined + b @ u_combined
    np.testing.assert_allclose(x_next_combined, x_next_parallel, rtol=1e-14)


"""Edge cases and boundary conditions."""

def test_zero_state_first_system():
    """
    Test with N1=0 (first system is static gain D1 only).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m1, p1 = 2, 2
    n2, m2 = 3, 2

    a1 = np.zeros((0, 0), order='F', dtype=float)
    b1 = np.zeros((0, m1), order='F', dtype=float)
    c1 = np.zeros((p1, 0), order='F', dtype=float)
    d1 = np.array([[1.0, 0.5], [0.2, 1.0]], order='F', dtype=float)

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p1, n2).astype(float, order='F')
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    alpha = 2.0

    a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

    assert info == 0
    assert n == n2
    assert m == m1 + m2

    np.testing.assert_allclose(a, a2, rtol=1e-14)
    np.testing.assert_allclose(b[:, m1:], b2, rtol=1e-14)
    np.testing.assert_allclose(b[:, :m1], np.zeros((n2, m1)), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(c, alpha * c2, rtol=1e-14)
    np.testing.assert_allclose(d[:, :m1], d1, rtol=1e-14)
    np.testing.assert_allclose(d[:, m1:], alpha * d2, rtol=1e-14)

def test_zero_state_second_system():
    """
    Test with N2=0 (second system is static gain D2 only).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n1, m1, p1 = 3, 2, 2
    m2 = 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.zeros((0, 0), order='F', dtype=float)
    b2 = np.zeros((0, m2), order='F', dtype=float)
    c2 = np.zeros((p1, 0), order='F', dtype=float)
    d2 = np.array([[0.5, 0.2], [0.1, 0.4]], order='F', dtype=float)

    alpha = -1.5

    a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

    assert info == 0
    assert n == n1
    assert m == m1 + m2

    np.testing.assert_allclose(a, a1, rtol=1e-14)
    np.testing.assert_allclose(b[:, :m1], b1, rtol=1e-14)
    np.testing.assert_allclose(b[:, m1:], np.zeros((n1, m2)), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(c, c1, rtol=1e-14)
    np.testing.assert_allclose(d[:, :m1], d1, rtol=1e-14)
    np.testing.assert_allclose(d[:, m1:], alpha * d2, rtol=1e-14)

def test_both_systems_zero_state():
    """
    Test with N1=0, N2=0 (both systems are static gains only).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    m1, m2, p1 = 2, 2, 2

    a1 = np.zeros((0, 0), order='F', dtype=float)
    b1 = np.zeros((0, m1), order='F', dtype=float)
    c1 = np.zeros((p1, 0), order='F', dtype=float)
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.zeros((0, 0), order='F', dtype=float)
    b2 = np.zeros((0, m2), order='F', dtype=float)
    c2 = np.zeros((p1, 0), order='F', dtype=float)
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    alpha = 3.0

    a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

    assert info == 0
    assert n == 0
    assert m == m1 + m2

    np.testing.assert_allclose(d[:, :m1], d1, rtol=1e-14)
    np.testing.assert_allclose(d[:, m1:], alpha * d2, rtol=1e-14)

def test_single_state_systems():
    """
    Test with minimal N1=1, N2=1 systems.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n1, m1, p1 = 1, 1, 1
    n2, m2 = 1, 1

    a1 = np.array([[0.5]], order='F', dtype=float)
    b1 = np.array([[1.0]], order='F', dtype=float)
    c1 = np.array([[2.0]], order='F', dtype=float)
    d1 = np.array([[0.1]], order='F', dtype=float)

    a2 = np.array([[0.3]], order='F', dtype=float)
    b2 = np.array([[1.5]], order='F', dtype=float)
    c2 = np.array([[0.8]], order='F', dtype=float)
    d2 = np.array([[0.2]], order='F', dtype=float)

    for alpha in [1.0, -1.0, 0.0, 2.5]:
        a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

        assert info == 0
        assert n == 2
        assert m == 2

        np.testing.assert_allclose(a[0, 0], a1[0, 0], rtol=1e-14)
        np.testing.assert_allclose(a[1, 1], a2[0, 0], rtol=1e-14)
        np.testing.assert_allclose(a[0, 1], 0.0, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(a[1, 0], 0.0, rtol=1e-14, atol=1e-15)

        np.testing.assert_allclose(c[0, 0], c1[0, 0], rtol=1e-14)
        np.testing.assert_allclose(c[0, 1], alpha * c2[0, 0], rtol=1e-14)

def test_alpha_zero():
    """
    Test with alpha=0.0 (second system contributes nothing to output).

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n1, m1, p1 = 2, 2, 2
    n2, m2 = 2, 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p1, n2).astype(float, order='F')
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    alpha = 0.0

    a, b, c, d, n, m, info = ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, alpha)

    assert info == 0

    np.testing.assert_allclose(c[:, n1:], np.zeros((p1, n2)), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(d[:, m1:], np.zeros((p1, m2)), rtol=1e-14, atol=1e-15)


"""Test error handling and invalid input detection."""

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
        ab05od('X', a1, b1, c1, d1, a2, b2, c2, d2, 1.0)

def test_dimension_mismatch_p1():
    """Test dimension mismatch: P1 from C1 != P1 from C2."""
    a1 = np.array([[1.0]], order='F', dtype=float)
    b1 = np.array([[1.0]], order='F', dtype=float)
    c1 = np.array([[1.0], [2.0]], order='F', dtype=float)  # P1=2
    d1 = np.array([[1.0], [2.0]], order='F', dtype=float)

    a2 = np.array([[1.0]], order='F', dtype=float)
    b2 = np.array([[1.0]], order='F', dtype=float)
    c2 = np.array([[1.0], [2.0], [3.0]], order='F', dtype=float)  # P1=3 (mismatch)
    d2 = np.array([[1.0]], order='F', dtype=float)

    with pytest.raises(ValueError, match="[pP]1"):
        ab05od('N', a1, b1, c1, d1, a2, b2, c2, d2, 1.0)


"""Test OVER='O' overlap mode."""

def test_overlap_mode_same_results():
    """
    Verify OVER='O' produces same results as OVER='N'.

    The Python wrapper allocates separate output arrays, so the results
    should be identical regardless of OVER mode.

    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    n1, m1, p1 = 2, 2, 2
    n2, m2 = 2, 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p1, n2).astype(float, order='F')
    d2 = np.random.randn(p1, m2).astype(float, order='F')

    alpha = -1.0

    a_n, b_n, c_n, d_n, n_n, m_n, info_n = ab05od(
        'N', a1.copy(), b1.copy(), c1.copy(), d1.copy(),
        a2.copy(), b2.copy(), c2.copy(), d2.copy(), alpha)

    a_o, b_o, c_o, d_o, n_o, m_o, info_o = ab05od(
        'O', a1.copy(), b1.copy(), c1.copy(), d1.copy(),
        a2.copy(), b2.copy(), c2.copy(), d2.copy(), alpha)

    assert info_n == 0
    assert info_o == 0
    assert n_n == n_o
    assert m_n == m_o

    np.testing.assert_allclose(a_o, a_n, rtol=1e-14)
    np.testing.assert_allclose(b_o, b_n, rtol=1e-14)
    np.testing.assert_allclose(c_o, c_n, rtol=1e-14)
    np.testing.assert_allclose(d_o, d_n, rtol=1e-14)
