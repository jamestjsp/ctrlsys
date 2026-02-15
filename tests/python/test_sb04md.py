"""
Tests for SB04MD: Continuous-time Sylvester equation solver AX + XB = C
using Hessenberg-Schur method.

Tests numerical correctness using:
1. HTML doc example (N=3, M=2)
2. Property test: residual AX + XB - C = 0
3. Edge cases (1x1, empty)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb04md_html_doc_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Problem: AX + XB = C
    Data from SB04MD.html Program Data/Results.
    N=3, M=2
    """
    from ctrlsys import sb04md

    n = 3
    m = 2

    # A matrix (3x3) from HTML doc
    a = np.array([
        [2.0, 1.0, 3.0],
        [0.0, 2.0, 1.0],
        [6.0, 1.0, 2.0]
    ], dtype=float, order='F')

    # B matrix (2x2) from HTML doc
    b = np.array([
        [2.0, 1.0],
        [1.0, 6.0]
    ], dtype=float, order='F')

    # C matrix (3x2) from HTML doc
    c = np.array([
        [2.0, 1.0],
        [1.0, 4.0],
        [0.0, 5.0]
    ], dtype=float, order='F')

    # Expected solution from HTML doc
    x_expected = np.array([
        [-2.7685, 0.5498],
        [-1.0531, 0.6865],
        [4.5257, -0.4389]
    ], dtype=float, order='F')

    x, z, info = sb04md(a, b, c)

    assert info == 0, f"sb04md failed with info={info}"
    assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)

    # Verify Z is orthogonal
    assert_allclose(z @ z.T, np.eye(m), rtol=1e-10, atol=1e-14)


def test_sb04md_residual_property():
    """
    Validate mathematical property: AX + XB = C holds for solution X.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04md

    np.random.seed(42)
    n = 4
    m = 3

    # Random general matrices
    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    c = np.random.randn(n, m).astype(float, order='F')

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    x, z, info = sb04md(a, b, c)

    assert info == 0, f"sb04md failed with info={info}"

    # Verify residual: A*X + X*B - C = 0 (using original matrices)
    residual = a_copy @ x + x @ b_copy - c_copy
    assert_allclose(residual, np.zeros((n, m)), atol=1e-10)


def test_sb04md_square_system():
    """
    Test square system N=M with known solution.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04md

    np.random.seed(123)
    n = 3

    # Generate random solution
    x_true = np.random.randn(n, n).astype(float, order='F')

    # Random well-conditioned matrices
    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, n).astype(float, order='F')

    # Compute C = A*X + X*B
    c = (a @ x_true + x_true @ b).astype(float, order='F')

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    x, z, info = sb04md(a, b, c)

    assert info == 0
    assert_allclose(x, x_true, rtol=1e-10)

    # Verify Z is orthogonal
    assert_allclose(z @ z.T, np.eye(n), rtol=1e-10, atol=1e-14)


def test_sb04md_1x1():
    """
    Test 1x1 case (simplest non-trivial case).

    AX + XB = C => a*x + x*b = c => x = c / (a + b)
    """
    from ctrlsys import sb04md

    a = np.array([[3.0]], dtype=float, order='F')
    b = np.array([[2.0]], dtype=float, order='F')
    c = np.array([[10.0]], dtype=float, order='F')

    x, z, info = sb04md(a, b, c)

    assert info == 0
    # x = 10 / (3 + 2) = 2.0
    assert_allclose(x, np.array([[2.0]]), rtol=1e-14)
    assert_allclose(z, np.array([[1.0]]), rtol=1e-12)


def test_sb04md_empty_dimensions():
    """
    Test with zero dimensions (quick return case).
    """
    from ctrlsys import sb04md

    # N=0, M=2
    a = np.zeros((0, 0), dtype=float, order='F')
    b = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float, order='F')
    c = np.zeros((0, 2), dtype=float, order='F')

    x, z, info = sb04md(a, b, c)
    assert info == 0
    assert x.shape == (0, 2)


def test_sb04md_n_larger_than_m():
    """
    Test case where N > M.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04md

    np.random.seed(456)
    n = 5
    m = 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    c = np.random.randn(n, m).astype(float, order='F')

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    x, z, info = sb04md(a, b, c)

    assert info == 0

    # Verify residual
    residual = a_copy @ x + x @ b_copy - c_copy
    assert_allclose(residual, np.zeros((n, m)), atol=1e-10)


def test_sb04md_m_larger_than_n():
    """
    Test case where M > N.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04md

    np.random.seed(789)
    n = 2
    m = 5

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    c = np.random.randn(n, m).astype(float, order='F')

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    x, z, info = sb04md(a, b, c)

    assert info == 0

    # Verify residual
    residual = a_copy @ x + x @ b_copy - c_copy
    assert_allclose(residual, np.zeros((n, m)), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
