"""
Tests for SB04QD: Discrete-time Sylvester equation solver X + AXB = C
using Hessenberg-Schur method.

Tests numerical correctness using:
1. HTML doc example (N=3, M=3)
2. Property test: residual X + AXB - C = 0
3. Edge cases (1x1, empty)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb04qd_html_doc_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Problem: X + AXB = C
    Data from SB04QD.html Program Data/Results.
    N=3, M=3
    """
    from ctrlsys import sb04qd

    n = 3
    m = 3

    # A matrix (3x3) from HTML doc
    a = np.array([
        [1.0, 2.0, 3.0],
        [6.0, 7.0, 8.0],
        [9.0, 2.0, 3.0]
    ], dtype=float, order='F')

    # B matrix (3x3) from HTML doc
    b = np.array([
        [7.0, 2.0, 3.0],
        [2.0, 1.0, 2.0],
        [3.0, 4.0, 1.0]
    ], dtype=float, order='F')

    # C matrix (3x3) from HTML doc
    c = np.array([
        [271.0, 135.0, 147.0],
        [923.0, 494.0, 482.0],
        [578.0, 383.0, 287.0]
    ], dtype=float, order='F')

    # Expected solution from HTML doc
    x_expected = np.array([
        [2.0, 3.0, 6.0],
        [4.0, 7.0, 1.0],
        [5.0, 3.0, 2.0]
    ], dtype=float, order='F')

    x, z, info = sb04qd(a, b, c)

    assert info == 0, f"sb04qd failed with info={info}"
    assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)

    # Verify Z is orthogonal
    assert_allclose(z @ z.T, np.eye(m), rtol=1e-10, atol=1e-14)


def test_sb04qd_residual_property():
    """
    Validate mathematical property: X + AXB = C holds for solution X.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04qd

    np.random.seed(42)
    n = 4
    m = 3

    # Random general matrices - ensure system has unique solution
    a = 0.5 * np.random.randn(n, n).astype(float, order='F')
    b = 0.5 * np.random.randn(m, m).astype(float, order='F')
    c = np.random.randn(n, m).astype(float, order='F')

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    x, z, info = sb04qd(a, b, c)

    assert info == 0, f"sb04qd failed with info={info}"

    # Verify residual: X + A*X*B - C = 0 (using original matrices)
    residual = x + a_copy @ x @ b_copy - c_copy
    assert_allclose(residual, np.zeros((n, m)), atol=1e-10)


def test_sb04qd_square_system():
    """
    Test square system N=M with known solution.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04qd

    np.random.seed(123)
    n = 3

    # Generate random solution
    x_true = np.random.randn(n, n).astype(float, order='F')

    # Random matrices scaled to ensure unique solution
    a = 0.4 * np.random.randn(n, n).astype(float, order='F')
    b = 0.4 * np.random.randn(n, n).astype(float, order='F')

    # Compute C = X + A*X*B
    c = (x_true + a @ x_true @ b).astype(float, order='F')

    a_copy = a.copy()
    b_copy = b.copy()

    x, z, info = sb04qd(a, b, c)

    assert info == 0
    assert_allclose(x, x_true, rtol=1e-10)

    # Verify Z is orthogonal
    assert_allclose(z @ z.T, np.eye(n), rtol=1e-12, atol=1e-15)


def test_sb04qd_1x1():
    """
    Test 1x1 case (simplest non-trivial case).

    X + AXB = C => x + a*x*b = c => x*(1 + a*b) = c => x = c / (1 + a*b)
    """
    from ctrlsys import sb04qd

    a = np.array([[2.0]], dtype=float, order='F')
    b = np.array([[3.0]], dtype=float, order='F')
    c = np.array([[14.0]], dtype=float, order='F')

    x, z, info = sb04qd(a, b, c)

    assert info == 0
    # x = 14 / (1 + 2*3) = 14 / 7 = 2.0
    assert_allclose(x, np.array([[2.0]]), rtol=1e-14)
    assert_allclose(z, np.array([[1.0]]), rtol=1e-12)


def test_sb04qd_empty_dimensions():
    """
    Test with zero dimensions (quick return case).
    """
    from ctrlsys import sb04qd

    # N=0, M=2
    a = np.zeros((0, 0), dtype=float, order='F')
    b = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float, order='F')
    c = np.zeros((0, 2), dtype=float, order='F')

    x, z, info = sb04qd(a, b, c)
    assert info == 0
    assert x.shape == (0, 2)


def test_sb04qd_n_larger_than_m():
    """
    Test case where N > M.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04qd

    np.random.seed(456)
    n = 5
    m = 2

    a = 0.4 * np.random.randn(n, n).astype(float, order='F')
    b = 0.4 * np.random.randn(m, m).astype(float, order='F')
    c = np.random.randn(n, m).astype(float, order='F')

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    x, z, info = sb04qd(a, b, c)

    assert info == 0

    # Verify residual
    residual = x + a_copy @ x @ b_copy - c_copy
    assert_allclose(residual, np.zeros((n, m)), atol=1e-10)


def test_sb04qd_m_larger_than_n():
    """
    Test case where M > N.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04qd

    np.random.seed(789)
    n = 2
    m = 5

    a = 0.4 * np.random.randn(n, n).astype(float, order='F')
    b = 0.4 * np.random.randn(m, m).astype(float, order='F')
    c = np.random.randn(n, m).astype(float, order='F')

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    x, z, info = sb04qd(a, b, c)

    assert info == 0

    # Verify residual
    residual = x + a_copy @ x @ b_copy - c_copy
    assert_allclose(residual, np.zeros((n, m)), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
