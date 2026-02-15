"""Tests for AB05PD - Parallel inter-connection of two systems."""

import numpy as np
import pytest


def test_ab05pd_html_doc_example():
    """
    Test AB05PD using example from SLICOT HTML documentation.

    System 1: N1=3, M=2, P=2
    System 2: N2=3
    ALPHA = 1.0 (coefficient for G2)

    G = G1 + alpha*G2 (parallel connection with same inputs)
    """
    from ctrlsys import ab05pd

    n1, m, p, n2 = 3, 2, 2, 3
    alpha = 1.0

    # System 1 matrices (column-major order)
    a1 = np.array([
        [1.0, 0.0, -1.0],
        [0.0, -1.0, 1.0],
        [1.0, 1.0, 2.0]
    ], order='F', dtype=float)

    # B1 read column-wise in Fortran: col1=[1,1,0], col2=[2,0,1]
    b1 = np.array([
        [1.0, 2.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    # C1 read row-wise: row1=[3,-2,1], row2=[0,1,0]
    c1 = np.array([
        [3.0, -2.0, 1.0],
        [0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    # D1 read row-wise: row1=[1,0], row2=[0,1]
    d1 = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    # System 2 matrices
    a2 = np.array([
        [-3.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, -1.0, 2.0]
    ], order='F', dtype=float)

    # B2 read column-wise: col1=[0,-1,0], col2=[1,0,2]
    b2 = np.array([
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    # C2 read row-wise: row1=[1,1,0], row2=[1,1,-1]
    c2 = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, -1.0]
    ], order='F', dtype=float)

    # D2 read row-wise: row1=[1,1], row2=[0,1]
    d2 = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    # Call routine
    n, a, b, c, d, info = ab05pd(n1, m, p, n2, alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n1 + n2  # 6

    # Expected output A (block diagonal)
    a_expected = np.array([
        [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 2.0]
    ], order='F', dtype=float)

    # Expected B (stacked vertically)
    b_expected = np.array([
        [1.0, 2.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    # Expected C (concatenated horizontally: [C1, alpha*C2])
    c_expected = np.array([
        [3.0, -2.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, -1.0]
    ], order='F', dtype=float)

    # Expected D (D1 + alpha*D2)
    d_expected = np.array([
        [2.0, 1.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b, b_expected, rtol=1e-14)
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)


def test_ab05pd_alpha_scaling():
    """
    Test AB05PD with alpha != 1.0 to verify scaling.

    Uses simple 2x2 systems with alpha = 0.5.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab05pd

    np.random.seed(42)
    n1, m, p, n2 = 2, 1, 1, 2
    alpha = 0.5

    # Simple diagonal systems for easy verification
    a1 = np.array([[0.5, 0.0], [0.0, -0.5]], order='F', dtype=float)
    b1 = np.array([[1.0], [0.0]], order='F', dtype=float)
    c1 = np.array([[1.0, 0.0]], order='F', dtype=float)
    d1 = np.array([[0.0]], order='F', dtype=float)

    a2 = np.array([[-0.3, 0.0], [0.0, 0.3]], order='F', dtype=float)
    b2 = np.array([[0.0], [2.0]], order='F', dtype=float)
    c2 = np.array([[0.0, 1.0]], order='F', dtype=float)
    d2 = np.array([[2.0]], order='F', dtype=float)

    n, a, b, c, d, info = ab05pd(n1, m, p, n2, alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == 4

    # Verify block diagonal A
    a_expected = np.array([
        [0.5, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0],
        [0.0, 0.0, -0.3, 0.0],
        [0.0, 0.0, 0.0, 0.3]
    ], order='F', dtype=float)
    np.testing.assert_allclose(a, a_expected, rtol=1e-14)

    # Verify stacked B
    b_expected = np.array([[1.0], [0.0], [0.0], [2.0]], order='F', dtype=float)
    np.testing.assert_allclose(b, b_expected, rtol=1e-14)

    # Verify C = [C1, alpha*C2] = [[1,0,0,0.5]]
    c_expected = np.array([[1.0, 0.0, 0.0, 0.5]], order='F', dtype=float)
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)

    # Verify D = D1 + alpha*D2 = 0 + 0.5*2 = 1
    d_expected = np.array([[1.0]], order='F', dtype=float)
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)


def test_ab05pd_transfer_function_property():
    """
    Validate mathematical property: G(s) = G1(s) + alpha*G2(s).

    At s=0 (DC gain): G(0) = D1 + alpha*D2 (for strictly proper systems) or
    full DC gain = D - C*inv(A)*B for proper systems.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab05pd

    np.random.seed(123)
    n1, m, p, n2 = 2, 2, 2, 2
    alpha = 2.0

    # Use stable diagonal systems (negative eigenvalues)
    a1 = np.diag([-1.0, -2.0]).astype(float, order='F')
    b1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    c1 = np.array([[1.0, 1.0], [0.0, 1.0]], order='F', dtype=float)
    d1 = np.array([[0.1, 0.0], [0.0, 0.1]], order='F', dtype=float)

    a2 = np.diag([-0.5, -1.5]).astype(float, order='F')
    b2 = np.array([[0.5, 0.0], [0.0, 0.5]], order='F', dtype=float)
    c2 = np.array([[1.0, 0.0], [0.5, 1.0]], order='F', dtype=float)
    d2 = np.array([[0.2, 0.1], [0.1, 0.2]], order='F', dtype=float)

    n, a, b, c, d, info = ab05pd(n1, m, p, n2, alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == 4

    # Verify D = D1 + alpha*D2
    d_expected = d1 + alpha * d2
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)

    # Verify DC gain: G(0) = D - C*inv(A)*B
    # Combined system must equal sum of individual DC gains
    dc_gain_1 = d1 - c1 @ np.linalg.solve(a1, b1)
    dc_gain_2 = d2 - c2 @ np.linalg.solve(a2, b2)
    dc_gain_expected = dc_gain_1 + alpha * dc_gain_2

    dc_gain_combined = d - c @ np.linalg.solve(a, b)
    np.testing.assert_allclose(dc_gain_combined, dc_gain_expected, rtol=1e-13)


def test_ab05pd_eigenvalue_preservation():
    """
    Validate eigenvalue preservation in block diagonal structure.

    The combined A matrix eigenvalues should be union of A1 and A2 eigenvalues.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import ab05pd

    np.random.seed(456)
    n1, m, p, n2 = 3, 1, 1, 2
    alpha = 1.0

    # Create systems with known eigenvalues
    a1 = np.array([
        [-1.0, 0.5, 0.0],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=float)  # Upper triangular, eigs: -1, -2, -3

    a2 = np.array([
        [-0.5, 0.2],
        [0.0, -1.5]
    ], order='F', dtype=float)  # Upper triangular, eigs: -0.5, -1.5

    b1 = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
    c1 = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
    d1 = np.array([[0.0]], order='F', dtype=float)

    b2 = np.array([[1.0], [1.0]], order='F', dtype=float)
    c2 = np.array([[1.0, 1.0]], order='F', dtype=float)
    d2 = np.array([[0.0]], order='F', dtype=float)

    n, a, b, c, d, info = ab05pd(n1, m, p, n2, alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0

    # Eigenvalues of combined A should be union of A1 and A2 eigenvalues
    eig_a1 = np.linalg.eigvals(a1)
    eig_a2 = np.linalg.eigvals(a2)
    eig_a = np.linalg.eigvals(a)

    expected_eigs = np.concatenate([eig_a1, eig_a2])

    np.testing.assert_allclose(
        sorted(eig_a.real), sorted(expected_eigs.real), rtol=1e-14
    )
    np.testing.assert_allclose(
        sorted(eig_a.imag), sorted(expected_eigs.imag), rtol=1e-14, atol=1e-14
    )


def test_ab05pd_n2_zero():
    """
    Test edge case: N2=0 (second system has no states).

    Result should be G1 + alpha*D2 for D matrix only.
    """
    from ctrlsys import ab05pd

    n1, m, p, n2 = 2, 2, 2, 0
    alpha = 1.5

    a1 = np.array([[1.0, 0.0], [0.0, -1.0]], order='F', dtype=float)
    b1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    c1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    d1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)

    # N2=0 means empty A2, B2, C2 but D2 still contributes
    a2 = np.array([], order='F', dtype=float).reshape(0, 0)
    b2 = np.array([], order='F', dtype=float).reshape(0, m)
    c2 = np.array([], order='F', dtype=float).reshape(p, 0)
    d2 = np.array([[2.0, 0.0], [0.0, 2.0]], order='F', dtype=float)

    n, a, b, c, d, info = ab05pd(n1, m, p, n2, alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n1  # Only states from system 1

    np.testing.assert_allclose(a, a1, rtol=1e-14)
    np.testing.assert_allclose(b, b1, rtol=1e-14)
    np.testing.assert_allclose(c, c1, rtol=1e-14)

    # D = D1 + alpha*D2 = I + 1.5*2*I = I + 3*I = 4*I
    d_expected = d1 + alpha * d2
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)


def test_ab05pd_n1_zero():
    """
    Test edge case: N1=0 (first system has no states).

    Result should be D1 + alpha*G2.
    """
    from ctrlsys import ab05pd

    n1, m, p, n2 = 0, 2, 2, 2
    alpha = 0.5

    # N1=0 means empty A1, B1, C1 but D1 still contributes
    a1 = np.array([], order='F', dtype=float).reshape(0, 0)
    b1 = np.array([], order='F', dtype=float).reshape(0, m)
    c1 = np.array([], order='F', dtype=float).reshape(p, 0)
    d1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)

    a2 = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
    b2 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    c2 = np.array([[2.0, 0.0], [0.0, 2.0]], order='F', dtype=float)
    d2 = np.array([[0.0, 1.0], [1.0, 0.0]], order='F', dtype=float)

    n, a, b, c, d, info = ab05pd(n1, m, p, n2, alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n2  # Only states from system 2

    np.testing.assert_allclose(a, a2, rtol=1e-14)
    np.testing.assert_allclose(b, b2, rtol=1e-14)

    # C = [C1, alpha*C2] = [empty, 0.5*C2] = 0.5*C2
    c_expected = alpha * c2
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)

    # D = D1 + alpha*D2
    d_expected = d1 + alpha * d2
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)


def test_ab05pd_alpha_zero():
    """
    Test edge case: alpha=0 (second system contribution is zero).

    Result should be exactly G1.
    """
    from ctrlsys import ab05pd

    n1, m, p, n2 = 2, 1, 1, 2
    alpha = 0.0

    a1 = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    b1 = np.array([[1.0], [2.0]], order='F', dtype=float)
    c1 = np.array([[1.0, 1.0]], order='F', dtype=float)
    d1 = np.array([[0.5]], order='F', dtype=float)

    a2 = np.array([[5.0, 6.0], [7.0, 8.0]], order='F', dtype=float)
    b2 = np.array([[3.0], [4.0]], order='F', dtype=float)
    c2 = np.array([[2.0, 2.0]], order='F', dtype=float)
    d2 = np.array([[1.0]], order='F', dtype=float)

    n, a, b, c, d, info = ab05pd(n1, m, p, n2, alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n1 + n2

    # A is block diagonal but A2 still appears
    np.testing.assert_allclose(a[:n1, :n1], a1, rtol=1e-14)
    np.testing.assert_allclose(a[n1:, n1:], a2, rtol=1e-14)
    np.testing.assert_allclose(a[:n1, n1:], np.zeros((n1, n2)), rtol=1e-14)
    np.testing.assert_allclose(a[n1:, :n1], np.zeros((n2, n1)), rtol=1e-14)

    # B is stacked
    np.testing.assert_allclose(b[:n1, :], b1, rtol=1e-14)
    np.testing.assert_allclose(b[n1:, :], b2, rtol=1e-14)

    # C = [C1, 0*C2]
    c_expected = np.hstack([c1, np.zeros((p, n2))])
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)

    # D = D1 + 0*D2 = D1
    np.testing.assert_allclose(d, d1, rtol=1e-14)


def test_ab05pd_negative_alpha():
    """
    Test with negative alpha (subtraction of systems).

    G = G1 - G2 when alpha = -1.0
    """
    from ctrlsys import ab05pd

    n1, m, p, n2 = 2, 1, 1, 2
    alpha = -1.0

    a1 = np.array([[0.5, 0.0], [0.0, -0.5]], order='F', dtype=float)
    b1 = np.array([[1.0], [1.0]], order='F', dtype=float)
    c1 = np.array([[1.0, 1.0]], order='F', dtype=float)
    d1 = np.array([[1.0]], order='F', dtype=float)

    a2 = np.array([[-0.3, 0.0], [0.0, 0.3]], order='F', dtype=float)
    b2 = np.array([[0.5], [0.5]], order='F', dtype=float)
    c2 = np.array([[1.0, 1.0]], order='F', dtype=float)
    d2 = np.array([[0.5]], order='F', dtype=float)

    n, a, b, c, d, info = ab05pd(n1, m, p, n2, alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0

    # C has negative scaling for C2
    c_expected = np.array([[1.0, 1.0, -1.0, -1.0]], order='F', dtype=float)
    np.testing.assert_allclose(c, c_expected, rtol=1e-14)

    # D = D1 - D2 = 1.0 - 0.5 = 0.5
    d_expected = np.array([[0.5]], order='F', dtype=float)
    np.testing.assert_allclose(d, d_expected, rtol=1e-14)
