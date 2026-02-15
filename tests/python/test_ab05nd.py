"""
Tests for AB05ND - Feedback inter-connection of two systems.

AB05ND computes the state-space model (A,B,C,D) for the feedback
inter-connection of two systems G1 and G2, where:
- U = U1 + alpha*Y2 (input with feedback)
- Y = Y1 = U2 (output feeds G2 input)
- alpha = +1: positive feedback
- alpha = -1: negative feedback

The interconnection uses:
E21 = (I + alpha*D1*D2)^-1
E12 = (I + alpha*D2*D1)^-1 = I - alpha*D2*E21*D1
"""

import numpy as np
import pytest
from ctrlsys import ab05nd


"""Basic functionality tests using SLICOT HTML doc example."""

def test_positive_feedback_html_example():
    """
    Test ALPHA=+1 (positive feedback) using SLICOT HTML doc example.

    System 1: N1=3, M1=2, P1=2
    System 2: N2=3

    From AB05ND.html Program Data and Results.
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

    alpha = 1.0  # Positive feedback

    a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == 6

    a_expected = np.array([
        [-0.5000, -0.2500, -1.5000, -1.2500, -1.2500,  0.7500],
        [-1.5000, -0.2500,  0.5000, -0.2500, -0.2500, -0.2500],
        [ 1.0000,  0.5000,  2.0000, -0.5000, -0.5000,  0.5000],
        [ 0.0000,  0.5000,  0.0000, -3.5000, -0.5000,  0.5000],
        [-1.5000,  1.2500, -0.5000,  1.2500,  0.2500,  1.2500],
        [ 0.0000,  1.0000,  0.0000, -1.0000, -2.0000,  3.0000]
    ], order='F', dtype=float)

    b_expected = np.array([
        [ 0.5000,  0.7500],
        [ 0.5000, -0.2500],
        [ 0.0000,  0.5000],
        [ 0.0000,  0.5000],
        [-0.5000,  0.2500],
        [ 0.0000,  1.0000]
    ], order='F', dtype=float)

    c_expected = np.array([
        [ 1.5000, -1.2500,  0.5000, -0.2500, -0.2500, -0.2500],
        [ 0.0000,  0.5000,  0.0000, -0.5000, -0.5000,  0.5000]
    ], order='F', dtype=float)

    d_expected = np.array([
        [ 0.5000, -0.2500],
        [ 0.0000,  0.5000]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a, a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b, b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(c, c_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(d, d_expected, rtol=1e-3, atol=1e-4)

def test_negative_feedback_modified_d_matrices():
    """
    Test ALPHA=-1 (negative feedback) with modified D matrices.

    For negative feedback, the formulas use E21 = (I - D1*D2)^-1.
    Use smaller D1, D2 to ensure non-singular feedback matrix.
    Random seed: 100 (for reproducibility)
    """
    np.random.seed(100)
    n1, m1, p1 = 3, 2, 2
    n2 = 3

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
        [0.3, 0.0],
        [0.0, 0.3]
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
        [0.5, 0.2],
        [0.1, 0.4]
    ], order='F', dtype=float)

    alpha = -1.0  # Negative feedback

    a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == 6

    E21 = np.linalg.inv(np.eye(2) + alpha * d1 @ d2)
    E12 = np.eye(2) - alpha * d2 @ E21 @ d1

    d_expected = E21 @ d1
    np.testing.assert_allclose(d, d_expected, rtol=1e-13)

    c_expected = np.hstack([
        E21 @ c1,
        -alpha * E21 @ d1 @ c2
    ])
    np.testing.assert_allclose(c, c_expected, rtol=1e-13)


"""Test mathematical properties of feedback connection."""

def test_feedback_matrix_identity():
    """
    Validate E12 = I - alpha*D2*E21*D1 relationship.

    This is a core mathematical identity in the feedback algorithm.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    p1, m1 = 2, 3

    d1 = np.random.randn(p1, m1).astype(float, order='F') * 0.3
    d2 = np.random.randn(m1, p1).astype(float, order='F') * 0.3
    alpha = -1.0

    E21 = np.linalg.inv(np.eye(p1) + alpha * d1 @ d2)
    E12_computed = np.eye(m1) - alpha * d2 @ E21 @ d1
    E12_direct = np.linalg.inv(np.eye(m1) + alpha * d2 @ d1)

    np.testing.assert_allclose(E12_computed, E12_direct, rtol=1e-13)

def test_feedthrough_matrix_formula():
    """
    Validate D = E21*D1 for the connected system.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n1, m1, p1 = 2, 2, 2
    n2 = 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F') * 0.3

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, p1).astype(float, order='F')
    c2 = np.random.randn(m1, n2).astype(float, order='F')
    d2 = np.random.randn(m1, p1).astype(float, order='F') * 0.3

    for alpha in [1.0, -1.0]:
        a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)
        assert info == 0

        E21 = np.linalg.inv(np.eye(p1) + alpha * d1 @ d2)
        d_expected = E21 @ d1
        np.testing.assert_allclose(d, d_expected, rtol=1e-13)

def test_output_matrix_formula():
    """
    Validate C = [E21*C1, -alpha*E21*D1*C2] for the connected system.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n1, m1, p1 = 2, 2, 2
    n2 = 3

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F') * 0.3

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, p1).astype(float, order='F')
    c2 = np.random.randn(m1, n2).astype(float, order='F')
    d2 = np.random.randn(m1, p1).astype(float, order='F') * 0.3

    for alpha in [1.0, -1.0]:
        a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)
        assert info == 0

        E21 = np.linalg.inv(np.eye(p1) + alpha * d1 @ d2)
        c_expected = np.hstack([
            E21 @ c1,
            -alpha * E21 @ d1 @ c2
        ])
        np.testing.assert_allclose(c, c_expected, rtol=1e-13)

def test_input_matrix_formula():
    """
    Validate B = [B1*E12; B2*E21*D1] for the connected system.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n1, m1, p1 = 2, 2, 2
    n2 = 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F') * 0.3

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, p1).astype(float, order='F')
    c2 = np.random.randn(m1, n2).astype(float, order='F')
    d2 = np.random.randn(m1, p1).astype(float, order='F') * 0.3

    for alpha in [1.0, -1.0]:
        a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)
        assert info == 0

        E21 = np.linalg.inv(np.eye(p1) + alpha * d1 @ d2)
        E12 = np.eye(m1) - alpha * d2 @ E21 @ d1
        b_expected = np.vstack([
            b1 @ E12,
            b2 @ E21 @ d1
        ])
        np.testing.assert_allclose(b, b_expected, rtol=1e-13)

def test_state_matrix_formula():
    """
    Validate A matrix structure for the connected system.

    A = [A1 - alpha*B1*E12*D2*C1,    -alpha*B1*E12*C2    ]
        [B2*E21*C1,                   A2 - alpha*B2*E21*D1*C2]

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n1, m1, p1 = 2, 2, 2
    n2 = 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F') * 0.3

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, p1).astype(float, order='F')
    c2 = np.random.randn(m1, n2).astype(float, order='F')
    d2 = np.random.randn(m1, p1).astype(float, order='F') * 0.3

    for alpha in [1.0, -1.0]:
        a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)
        assert info == 0

        E21 = np.linalg.inv(np.eye(p1) + alpha * d1 @ d2)
        E12 = np.eye(m1) - alpha * d2 @ E21 @ d1

        a11_expected = a1 - alpha * b1 @ E12 @ d2 @ c1
        a12_expected = -alpha * b1 @ E12 @ c2
        a21_expected = b2 @ E21 @ c1
        a22_expected = a2 - alpha * b2 @ E21 @ d1 @ c2

        a_expected = np.block([
            [a11_expected, a12_expected],
            [a21_expected, a22_expected]
        ])
        np.testing.assert_allclose(a, a_expected, rtol=1e-13)


"""Edge cases and boundary conditions."""

def test_zero_state_first_system():
    """
    Test feedback with N1=0 (first system is static gain D1 only).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m1, p1 = 2, 2
    n2 = 3

    a1 = np.zeros((0, 0), order='F', dtype=float)
    b1 = np.zeros((0, m1), order='F', dtype=float)
    c1 = np.zeros((p1, 0), order='F', dtype=float)
    d1 = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=float)

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, p1).astype(float, order='F')
    c2 = np.random.randn(m1, n2).astype(float, order='F')
    d2 = np.random.randn(m1, p1).astype(float, order='F') * 0.3

    alpha = -1.0
    a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n2

    E21 = np.linalg.inv(np.eye(p1) + alpha * d1 @ d2)
    d_expected = E21 @ d1
    np.testing.assert_allclose(d, d_expected, rtol=1e-13)

def test_zero_state_second_system():
    """
    Test feedback with N2=0 (second system is static gain D2 only).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n1, m1, p1 = 3, 2, 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F') * 0.3

    a2 = np.zeros((0, 0), order='F', dtype=float)
    b2 = np.zeros((0, p1), order='F', dtype=float)
    c2 = np.zeros((m1, 0), order='F', dtype=float)
    d2 = np.array([[0.5, 0.2], [0.1, 0.4]], order='F', dtype=float)

    alpha = 1.0
    a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n1

    E21 = np.linalg.inv(np.eye(p1) + alpha * d1 @ d2)
    E12 = np.eye(m1) - alpha * d2 @ E21 @ d1

    d_expected = E21 @ d1
    c_expected = E21 @ c1
    b_expected = b1 @ E12
    a_expected = a1 - alpha * b1 @ E12 @ d2 @ c1

    np.testing.assert_allclose(d, d_expected, rtol=1e-13)
    np.testing.assert_allclose(c, c_expected, rtol=1e-13)
    np.testing.assert_allclose(b, b_expected, rtol=1e-13)
    np.testing.assert_allclose(a, a_expected, rtol=1e-13)

def test_single_state_systems():
    """
    Test with minimal N1=1, N2=1 systems.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n1, m1, p1 = 1, 1, 1
    n2 = 1

    a1 = np.array([[0.5]], order='F', dtype=float)
    b1 = np.array([[1.0]], order='F', dtype=float)
    c1 = np.array([[2.0]], order='F', dtype=float)
    d1 = np.array([[0.1]], order='F', dtype=float)

    a2 = np.array([[0.3]], order='F', dtype=float)
    b2 = np.array([[1.5]], order='F', dtype=float)
    c2 = np.array([[0.8]], order='F', dtype=float)
    d2 = np.array([[0.2]], order='F', dtype=float)

    for alpha in [1.0, -1.0]:
        a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)

        assert info == 0
        assert n == 2

        E21 = 1.0 / (1.0 + alpha * d1[0, 0] * d2[0, 0])
        E12 = 1.0 - alpha * d2[0, 0] * E21 * d1[0, 0]

        d_expected = E21 * d1
        np.testing.assert_allclose(d, d_expected, rtol=1e-13)

def test_identity_feedthrough():
    """
    Test with D1=I, D2=0 (no algebraic feedback loop).

    With D2=0, E21 = I and E12 = I, simplifying formulas.
    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n1, m1, p1 = 2, 2, 2
    n2 = 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.eye(p1, m1, order='F', dtype=float)

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, p1).astype(float, order='F')
    c2 = np.random.randn(m1, n2).astype(float, order='F')
    d2 = np.zeros((m1, p1), order='F', dtype=float)

    alpha = -1.0
    a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info == 0
    assert n == n1 + n2

    np.testing.assert_allclose(d, d1, rtol=1e-14)


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
        ab05nd('X', 1.0, a1, b1, c1, d1, a2, b2, c2, d2)

def test_singular_feedback_matrix():
    """
    Test singular feedback matrix (I + alpha*D1*D2) detection.

    When D1*D2 has eigenvalue = -1/alpha, the feedback is singular.
    """
    d1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    d2 = np.array([[-1.0, 0.0], [0.0, 0.5]], order='F', dtype=float)
    alpha = 1.0  # (I + D1*D2) has eigenvalue 0

    a1 = np.array([[0.5, 0.0], [0.0, 0.3]], order='F', dtype=float)
    b1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    c1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)

    a2 = np.array([[0.4, 0.0], [0.0, 0.2]], order='F', dtype=float)
    b2 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    c2 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)

    a, b, c, d, n, info = ab05nd('N', alpha, a1, b1, c1, d1, a2, b2, c2, d2)

    assert info > 0

def test_dimension_mismatch_p1_c2():
    """Test dimension mismatch: P1 from C1 != rows of B2."""
    a1 = np.array([[1.0]], order='F', dtype=float)
    b1 = np.array([[1.0]], order='F', dtype=float)
    c1 = np.array([[1.0], [2.0]], order='F', dtype=float)  # P1=2
    d1 = np.array([[1.0], [2.0]], order='F', dtype=float)

    a2 = np.array([[1.0]], order='F', dtype=float)
    b2 = np.array([[1.0, 2.0, 3.0]], order='F', dtype=float)  # expects P1=3
    c2 = np.array([[1.0]], order='F', dtype=float)
    d2 = np.array([[1.0, 2.0, 3.0]], order='F', dtype=float)

    with pytest.raises(ValueError, match="[pP]1"):
        ab05nd('N', 1.0, a1, b1, c1, d1, a2, b2, c2, d2)

def test_dimension_mismatch_m1_c2():
    """Test dimension mismatch: M1 != rows of C2."""
    a1 = np.array([[1.0]], order='F', dtype=float)
    b1 = np.array([[1.0, 2.0]], order='F', dtype=float)  # M1=2
    c1 = np.array([[1.0]], order='F', dtype=float)
    d1 = np.array([[1.0, 2.0]], order='F', dtype=float)

    a2 = np.array([[1.0]], order='F', dtype=float)
    b2 = np.array([[1.0]], order='F', dtype=float)
    c2 = np.array([[1.0], [2.0], [3.0]], order='F', dtype=float)  # M1=3 rows
    d2 = np.array([[1.0]], order='F', dtype=float)

    with pytest.raises(ValueError, match="[mM]1"):
        ab05nd('N', 1.0, a1, b1, c1, d1, a2, b2, c2, d2)


"""Test OVER='O' overlap mode."""

def test_overlap_mode_same_results():
    """
    Verify OVER='O' produces same results as OVER='N'.

    When OVER='O' (overlap mode) but arrays don't actually overlap,
    the C code now correctly detects this and falls back to the non-overlap path.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n1, m1, p1 = 2, 2, 2
    n2 = 2

    a1 = np.random.randn(n1, n1).astype(float, order='F') * 0.5
    b1 = np.random.randn(n1, m1).astype(float, order='F')
    c1 = np.random.randn(p1, n1).astype(float, order='F')
    d1 = np.random.randn(p1, m1).astype(float, order='F') * 0.3

    a2 = np.random.randn(n2, n2).astype(float, order='F') * 0.5
    b2 = np.random.randn(n2, p1).astype(float, order='F')
    c2 = np.random.randn(m1, n2).astype(float, order='F')
    d2 = np.random.randn(m1, p1).astype(float, order='F') * 0.3

    alpha = -1.0

    a_n, b_n, c_n, d_n, n_n, info_n = ab05nd('N', alpha,
        a1.copy(), b1.copy(), c1.copy(), d1.copy(),
        a2.copy(), b2.copy(), c2.copy(), d2.copy())

    a_o, b_o, c_o, d_o, n_o, info_o = ab05nd('O', alpha,
        a1.copy(), b1.copy(), c1.copy(), d1.copy(),
        a2.copy(), b2.copy(), c2.copy(), d2.copy())

    assert info_n == 0
    assert info_o == 0
    assert n_n == n_o

    np.testing.assert_allclose(a_o, a_n, rtol=1e-14)
    np.testing.assert_allclose(b_o, b_n, rtol=1e-14)
    np.testing.assert_allclose(c_o, c_n, rtol=1e-14)
    np.testing.assert_allclose(d_o, d_n, rtol=1e-14)
