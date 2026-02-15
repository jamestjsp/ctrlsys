"""
Tests for AB05QD - Appending two systems in state-space form (block diagonal).

AB05QD constructs G = diag(G1, G2) where:
- A = [[A1, 0], [0, A2]]
- B = [[B1, 0], [0, B2]]
- C = [[C1, 0], [0, C2]]
- D = [[D1, 0], [0, D2]]
"""

import numpy as np
import pytest
from ctrlsys import ab05qd


"""Basic functionality tests using SLICOT HTML doc example."""

def test_html_doc_example():
    """
    Test using SLICOT HTML doc example data.

    System 1: N1=3, M1=2, P1=2
    System 2: N2=3, M2=2, P2=2
    Result: N=6, M=4, P=4

    Forms G = diag(G1, G2) with block diagonal structure.
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

    a, b, c, d, n, m, p, info = ab05qd('N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == 6
    assert m == 4
    assert p == 4

    a_expected = np.array([
        [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 2.0]
    ], order='F', dtype=float)

    b_expected = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    c_expected = np.array([
        [3.0, -2.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, -1.0]
    ], order='F', dtype=float)

    d_expected = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b, b_expected, rtol=1e-14)
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)


"""Test mathematical properties of block diagonal structure."""

def test_block_diagonal_structure():
    """
    Validate block diagonal structure is correctly formed.

    The appended system G = diag(G1, G2) should have:
    - A[0:N1, 0:N1] = A1, A[N1:N, N1:N] = A2, off-diagonals = 0
    - Similar for B, C, D

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n1, m1, p1 = 2, 1, 2
    n2, m2, p2 = 3, 2, 1

    a1 = np.random.randn(n1, n1).astype(float, order='F')
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F')
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p2, n2).astype(float, order='F')
    d2 = np.random.randn(p2, m2).astype(float, order='F')

    a, b, c, d, n, m, p, info = ab05qd('N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n1 + n2
    assert m == m1 + m2
    assert p == p1 + p2

    np.testing.assert_allclose(a[:n1, :n1], a1, rtol=1e-14)
    np.testing.assert_allclose(a[n1:, n1:], a2, rtol=1e-14)
    np.testing.assert_allclose(a[:n1, n1:], np.zeros((n1, n2)), rtol=1e-14)
    np.testing.assert_allclose(a[n1:, :n1], np.zeros((n2, n1)), rtol=1e-14)

    np.testing.assert_allclose(b[:n1, :m1], b1, rtol=1e-14)
    np.testing.assert_allclose(b[n1:, m1:], b2, rtol=1e-14)
    np.testing.assert_allclose(b[:n1, m1:], np.zeros((n1, m2)), rtol=1e-14)
    np.testing.assert_allclose(b[n1:, :m1], np.zeros((n2, m1)), rtol=1e-14)

    np.testing.assert_allclose(c[:p1, :n1], c1, rtol=1e-14)
    np.testing.assert_allclose(c[p1:, n1:], c2, rtol=1e-14)
    np.testing.assert_allclose(c[:p1, n1:], np.zeros((p1, n2)), rtol=1e-14)
    np.testing.assert_allclose(c[p1:, :n1], np.zeros((p2, n1)), rtol=1e-14)

    np.testing.assert_allclose(d[:p1, :m1], d1, rtol=1e-14)
    np.testing.assert_allclose(d[p1:, m1:], d2, rtol=1e-14)
    np.testing.assert_allclose(d[:p1, m1:], np.zeros((p1, m2)), rtol=1e-14)
    np.testing.assert_allclose(d[p1:, :m1], np.zeros((p2, m1)), rtol=1e-14)

def test_eigenvalue_preservation():
    """
    Validate eigenvalues of A are union of A1 and A2 eigenvalues.

    For block diagonal A = diag(A1, A2), eigenvalues are preserved.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n1, m1, p1 = 3, 1, 1
    n2, m2, p2 = 2, 1, 1

    a1 = np.random.randn(n1, n1).astype(float, order='F')
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.random.randn(n2, n2).astype(float, order='F')
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p2, n2).astype(float, order='F')
    d2 = np.random.randn(p2, m2).astype(float, order='F')

    a, b, c, d, n, m, p, info = ab05qd('N', a1, b1, c1, d1, a2, b2, c2, d2)
    assert info == 0

    eig1 = np.linalg.eigvals(a1)
    eig2 = np.linalg.eigvals(a2)
    eig_combined = np.concatenate([eig1, eig2])

    eig_result = np.linalg.eigvals(a)

    eig_combined_sorted = np.sort(eig_combined.real) + 1j * np.sort(eig_combined.imag)
    eig_result_sorted = np.sort(eig_result.real) + 1j * np.sort(eig_result.imag)

    np.testing.assert_allclose(
        np.sort(eig_combined.real),
        np.sort(eig_result.real),
        rtol=1e-13
    )

def test_independent_subsystem_simulation():
    """
    Validate that block diagonal system simulates as independent subsystems.

    For G = diag(G1, G2):
    - u = [u1; u2] -> y = [y1; y2]
    - y1 depends only on u1, y2 depends only on u2

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n1, m1, p1 = 2, 1, 1
    n2, m2, p2 = 2, 1, 1

    a1 = np.diag([0.5, 0.3]).astype(float, order='F')
    b1 = np.array([[1.0], [0.5]], order='F', dtype=float)
    c1 = np.array([[1.0, 0.5]], order='F', dtype=float)
    d1 = np.array([[0.1]], order='F', dtype=float)

    a2 = np.diag([0.4, 0.6]).astype(float, order='F')
    b2 = np.array([[0.8], [1.2]], order='F', dtype=float)
    c2 = np.array([[0.5, 1.0]], order='F', dtype=float)
    d2 = np.array([[0.2]], order='F', dtype=float)

    a, b, c, d, n, m, p, info = ab05qd('N', a1, b1, c1, d1, a2, b2, c2, d2)
    assert info == 0

    nsteps = 10
    u1 = np.ones((m1, nsteps), order='F', dtype=float)
    u2 = np.ones((m2, nsteps), order='F', dtype=float) * 2.0

    x1 = np.zeros((n1, 1), order='F', dtype=float)
    y1_sep = np.zeros((p1, nsteps), order='F', dtype=float)
    for k in range(nsteps):
        uk = u1[:, k:k+1]
        y1_sep[:, k:k+1] = c1 @ x1 + d1 @ uk
        x1 = a1 @ x1 + b1 @ uk

    x2 = np.zeros((n2, 1), order='F', dtype=float)
    y2_sep = np.zeros((p2, nsteps), order='F', dtype=float)
    for k in range(nsteps):
        uk = u2[:, k:k+1]
        y2_sep[:, k:k+1] = c2 @ x2 + d2 @ uk
        x2 = a2 @ x2 + b2 @ uk

    u_combined = np.vstack([u1, u2])
    x_comb = np.zeros((n, 1), order='F', dtype=float)
    y_comb = np.zeros((p, nsteps), order='F', dtype=float)
    for k in range(nsteps):
        uk = u_combined[:, k:k+1]
        y_comb[:, k:k+1] = c @ x_comb + d @ uk
        x_comb = a @ x_comb + b @ uk

    np.testing.assert_allclose(y_comb[:p1, :], y1_sep, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(y_comb[p1:, :], y2_sep, rtol=1e-14, atol=1e-15)


"""Edge cases and boundary conditions."""

def test_zero_state_first_system():
    """
    Test with N1=0 (first system is static gain D1 only).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m1, p1 = 2, 2
    n2, m2, p2 = 3, 2, 2

    a1 = np.zeros((0, 0), order='F', dtype=float)
    b1 = np.zeros((0, m1), order='F', dtype=float)
    c1 = np.zeros((p1, 0), order='F', dtype=float)
    d1 = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=float)

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, m2).astype(float, order='F')
    c2 = np.random.randn(p2, n2).astype(float, order='F')
    d2 = np.random.randn(p2, m2).astype(float, order='F')

    a, b, c, d, n, m, p, info = ab05qd('N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n2
    assert m == m1 + m2
    assert p == p1 + p2

    np.testing.assert_allclose(a, a2, rtol=1e-14)

    np.testing.assert_allclose(b[:, m1:], b2, rtol=1e-14)
    np.testing.assert_allclose(b[:, :m1], np.zeros((n2, m1)), rtol=1e-14)

    np.testing.assert_allclose(c[p1:, :], c2, rtol=1e-14)
    np.testing.assert_allclose(c[:p1, :], np.zeros((p1, n2)), rtol=1e-14)

def test_zero_state_second_system():
    """
    Test with N2=0 (second system is static gain D2 only).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n1, m1, p1 = 3, 2, 2
    m2, p2 = 2, 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F')

    a2 = np.zeros((0, 0), order='F', dtype=float)
    b2 = np.zeros((0, m2), order='F', dtype=float)
    c2 = np.zeros((p2, 0), order='F', dtype=float)
    d2 = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=float)

    a, b, c, d, n, m, p, info = ab05qd('N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n1
    assert m == m1 + m2
    assert p == p1 + p2

    np.testing.assert_allclose(a, a1, rtol=1e-14)

    np.testing.assert_allclose(b[:, :m1], b1, rtol=1e-14)
    np.testing.assert_allclose(b[:, m1:], np.zeros((n1, m2)), rtol=1e-14)

    np.testing.assert_allclose(c[:p1, :], c1, rtol=1e-14)
    np.testing.assert_allclose(c[p1:, :], np.zeros((p2, n1)), rtol=1e-14)

def test_single_state_systems():
    """
    Test with minimal N1=1, N2=1 systems.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n1, m1, p1 = 1, 1, 1
    n2, m2, p2 = 1, 1, 1

    a1 = np.array([[0.5]], order='F', dtype=float)
    b1 = np.array([[1.0]], order='F', dtype=float)
    c1 = np.array([[2.0]], order='F', dtype=float)
    d1 = np.array([[0.1]], order='F', dtype=float)

    a2 = np.array([[0.3]], order='F', dtype=float)
    b2 = np.array([[1.5]], order='F', dtype=float)
    c2 = np.array([[0.8]], order='F', dtype=float)
    d2 = np.array([[0.2]], order='F', dtype=float)

    a, b, c, d, n, m, p, info = ab05qd('N', a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == 2
    assert m == 2
    assert p == 2

    a_expected = np.array([
        [0.5, 0.0],
        [0.0, 0.3]
    ], order='F', dtype=float)

    b_expected = np.array([
        [1.0, 0.0],
        [0.0, 1.5]
    ], order='F', dtype=float)

    c_expected = np.array([
        [2.0, 0.0],
        [0.0, 0.8]
    ], order='F', dtype=float)

    d_expected = np.array([
        [0.1, 0.0],
        [0.0, 0.2]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b, b_expected, rtol=1e-14)
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)


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
        ab05qd('X', a1, b1, c1, d1, a2, b2, c2, d2)
