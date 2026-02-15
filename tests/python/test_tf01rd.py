"""Tests for TF01RD: Markov parameters from state-space representation.

TF01RD computes N Markov parameters M(1), M(2), ..., M(N) from system
matrices (A, B, C), where M(k) = C * A^(k-1) * B.
"""

import numpy as np
import pytest
from ctrlsys import tf01rd


"""Basic functionality tests using HTML doc example."""

def test_html_doc_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Tests NA=3, NB=2, NC=2, N=5 system.
    Data from SLICOT-Reference/doc/TF01RD.html
    """
    # System matrices from HTML doc (column-major)
    a = np.array([
        [0.0, 1.0, 0.0],
        [-0.07, 0.8, 0.0],
        [0.015, -0.15, 0.5]
    ], order='F', dtype=float)

    b = np.array([
        [0.0, -1.0],
        [2.0, -0.1],
        [1.0, 1.0]
    ], order='F', dtype=float)

    c = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ], order='F', dtype=float)

    n = 5  # Number of Markov parameters

    # Call routine
    h, info = tf01rd(a, b, c, n)

    assert info == 0, f"Expected info=0, got {info}"

    # Verify output shape: H is NC x (N*NB) = 2 x 10
    assert h.shape == (2, 10), f"Expected shape (2, 10), got {h.shape}"

    # Expected Markov parameters from HTML doc (4-decimal precision)
    # M(1) = CB
    m1_expected = np.array([
        [1.0, 1.0],
        [0.0, -1.0]
    ], order='F', dtype=float)

    # M(2) = CAB
    m2_expected = np.array([
        [0.2, 0.5],
        [2.0, -0.1]
    ], order='F', dtype=float)

    # M(3) = CA^2 B
    m3_expected = np.array([
        [-0.11, 0.25],
        [1.6, -0.01]
    ], order='F', dtype=float)

    # M(4) = CA^3 B
    m4_expected = np.array([
        [-0.202, 0.125],
        [1.14, -0.001]
    ], order='F', dtype=float)

    # M(5) = CA^4 B
    m5_expected = np.array([
        [-0.2039, 0.0625],
        [0.8, -0.0001]
    ], order='F', dtype=float)

    # Extract each M(k) from H and validate
    nb = 2
    np.testing.assert_allclose(h[:, 0:2], m1_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 2:4], m2_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 4:6], m3_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 6:8], m4_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 8:10], m5_expected, rtol=1e-3, atol=1e-4)


"""Mathematical property validation tests."""

def test_markov_parameter_definition():
    """
    Validate M(k) = C * A^(k-1) * B exactly.

    Tests that routine correctly computes the matrix power series.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    na, nb, nc = 4, 2, 3
    n = 6

    a = np.random.randn(na, na).astype(float, order='F')
    b = np.random.randn(na, nb).astype(float, order='F')
    c = np.random.randn(nc, na).astype(float, order='F')

    h, info = tf01rd(a, b, c, n)
    assert info == 0

    # Verify each Markov parameter manually
    a_power = np.eye(na, order='F', dtype=float)  # A^0 = I
    for k in range(1, n + 1):
        # M(k) = C * A^(k-1) * B
        m_expected = c @ a_power @ b
        m_actual = h[:, (k - 1) * nb:k * nb]
        np.testing.assert_allclose(m_actual, m_expected, rtol=1e-14)
        a_power = a_power @ a  # A^k for next iteration

def test_impulse_response_interpretation():
    """
    Validate Markov parameters as impulse response matrices.

    For a discrete-time system, M(k) represents the output at time k
    when input is a unit impulse at time 0 (assuming zero initial state).
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    na, nb, nc = 3, 1, 2
    n = 10

    # Create stable system (scale eigenvalues < 1)
    a = np.random.randn(na, na).astype(float, order='F')
    a = a / (np.max(np.abs(np.linalg.eigvals(a))) + 0.1)  # Make stable

    b = np.random.randn(na, nb).astype(float, order='F')
    c = np.random.randn(nc, na).astype(float, order='F')

    h, info = tf01rd(a, b, c, n)
    assert info == 0

    # Simulate impulse response manually
    x = np.zeros((na, 1), order='F', dtype=float)
    u_impulse = np.ones((nb, 1), order='F', dtype=float)

    for k in range(1, n + 1):
        if k == 1:
            # First step: x(1) = A*x(0) + B*u(0) = B*1 = B
            x = b.copy()
        else:
            # Subsequent steps: x(k) = A*x(k-1)
            x = a @ x

        # y(k) = C * x(k)
        y = c @ x
        m_k = h[:, (k - 1) * nb:k * nb]
        np.testing.assert_allclose(m_k, y, rtol=1e-14)

def test_linearity_in_c():
    """
    Validate linearity: If C is scaled by alpha, H is scaled by alpha.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    na, nb, nc = 3, 2, 2
    n = 4
    alpha = 2.5

    a = np.random.randn(na, na).astype(float, order='F')
    b = np.random.randn(na, nb).astype(float, order='F')
    c = np.random.randn(nc, na).astype(float, order='F')

    h1, info1 = tf01rd(a, b, c, n)
    h2, info2 = tf01rd(a, b, alpha * c, n)

    assert info1 == 0 and info2 == 0
    np.testing.assert_allclose(h2, alpha * h1, rtol=1e-14)

def test_linearity_in_b():
    """
    Validate linearity: If B is scaled by alpha, H is scaled by alpha.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    na, nb, nc = 3, 2, 2
    n = 4
    alpha = 3.0

    a = np.random.randn(na, na).astype(float, order='F')
    b = np.random.randn(na, nb).astype(float, order='F')
    c = np.random.randn(nc, na).astype(float, order='F')

    h1, info1 = tf01rd(a, b, c, n)
    h2, info2 = tf01rd(a, alpha * b, c, n)

    assert info1 == 0 and info2 == 0
    np.testing.assert_allclose(h2, alpha * h1, rtol=1e-14)


"""Edge case tests."""

def test_single_markov_parameter():
    """
    Validate N=1 case: M(1) = C*B.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    na, nb, nc = 3, 2, 2
    n = 1

    a = np.random.randn(na, na).astype(float, order='F')
    b = np.random.randn(na, nb).astype(float, order='F')
    c = np.random.randn(nc, na).astype(float, order='F')

    h, info = tf01rd(a, b, c, n)
    assert info == 0
    assert h.shape == (nc, nb)

    # M(1) = C * A^0 * B = C * B
    m1_expected = c @ b
    np.testing.assert_allclose(h, m1_expected, rtol=1e-14)

def test_siso_system():
    """
    Validate SISO system (NB=1, NC=1).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    na, nb, nc = 4, 1, 1
    n = 5

    a = np.random.randn(na, na).astype(float, order='F')
    b = np.random.randn(na, nb).astype(float, order='F')
    c = np.random.randn(nc, na).astype(float, order='F')

    h, info = tf01rd(a, b, c, n)
    assert info == 0
    assert h.shape == (1, 5)  # NC x (N*NB) = 1 x 5

    # Verify each Markov parameter
    a_power = np.eye(na, dtype=float)
    for k in range(1, n + 1):
        m_expected = c @ a_power @ b
        np.testing.assert_allclose(h[0, k - 1], m_expected[0, 0], rtol=1e-14)
        a_power = a_power @ a

def test_identity_a_matrix():
    """
    Validate A=I case: All Markov parameters are C*B.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    na, nb, nc = 3, 2, 2
    n = 4

    a = np.eye(na, order='F', dtype=float)
    b = np.random.randn(na, nb).astype(float, order='F')
    c = np.random.randn(nc, na).astype(float, order='F')

    h, info = tf01rd(a, b, c, n)
    assert info == 0

    # With A=I, all M(k) = C*I*B = C*B
    cb = c @ b
    for k in range(n):
        np.testing.assert_allclose(h[:, k * nb:(k + 1) * nb], cb, rtol=1e-14)

def test_nilpotent_a_matrix():
    """
    Validate nilpotent A: Markov parameters become zero after some k.

    A nilpotent matrix of index p satisfies A^p = 0, so M(k) = 0 for k > p.
    """
    na, nb, nc = 3, 1, 1
    n = 5

    # Strictly upper triangular matrix is nilpotent
    a = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    b = np.array([[1.0], [0.0], [0.0]], order='F', dtype=float)
    c = np.array([[0.0, 0.0, 1.0]], order='F', dtype=float)

    h, info = tf01rd(a, b, c, n)
    assert info == 0

    # A^3 = 0, so M(k) = 0 for k >= 4 (since M(k) = C * A^(k-1) * B)
    # M(1) = C*B, M(2) = C*A*B, M(3) = C*A^2*B, M(4) = M(5) = 0
    assert h[0, 3] == 0.0  # M(4)
    assert h[0, 4] == 0.0  # M(5)


"""Error handling tests."""

def test_empty_n_returns_empty():
    """
    Validate N=0 returns empty output.
    """
    np.random.seed(444)
    na, nb, nc = 3, 2, 2
    n = 0

    a = np.random.randn(na, na).astype(float, order='F')
    b = np.random.randn(na, nb).astype(float, order='F')
    c = np.random.randn(nc, na).astype(float, order='F')

    h, info = tf01rd(a, b, c, n)
    assert info == 0
    assert h.shape[1] == 0  # Empty output

def test_negative_n_error():
    """
    Validate negative N raises error.
    """
    np.random.seed(555)
    na, nb, nc = 3, 2, 2

    a = np.random.randn(na, na).astype(float, order='F')
    b = np.random.randn(na, nb).astype(float, order='F')
    c = np.random.randn(nc, na).astype(float, order='F')

    with pytest.raises(ValueError):
        tf01rd(a, b, c, -1)

def test_incompatible_dimensions():
    """
    Validate incompatible matrix dimensions raise error.
    """
    np.random.seed(666)

    # A is 3x3, B is 4x2 - incompatible
    a = np.random.randn(3, 3).astype(float, order='F')
    b = np.random.randn(4, 2).astype(float, order='F')
    c = np.random.randn(2, 3).astype(float, order='F')

    with pytest.raises(ValueError):
        tf01rd(a, b, c, 5)
