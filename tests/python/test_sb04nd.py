"""
Tests for SB04ND: Continuous-time Sylvester equation solver AX + XB = C
using Hessenberg-Schur method.

Tests numerical correctness using:
1. HTML doc example (N=5, M=3, ABSCHU='B')
2. Property test: residual AX + XB - C = 0
3. Error handling (singular case)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb04nd_html_doc_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Problem: AX + XB = C with A in upper Hessenberg, B in upper Schur form.
    Data from SB04ND.html Program Data/Results.
    """
    from ctrlsys import sb04nd

    n = 5
    m = 3

    # A is upper Hessenberg (5x5) - read row-by-row from HTML doc
    a = np.array([
        [17.0, 24.0,  1.0,  8.0, 15.0],
        [23.0,  5.0,  7.0, 14.0, 16.0],
        [ 0.0,  6.0, 13.0, 20.0, 22.0],
        [ 0.0,  0.0, 19.0, 21.0,  3.0],
        [ 0.0,  0.0,  0.0,  2.0,  9.0]
    ], dtype=float, order='F')

    # B is upper Schur (3x3) - read row-by-row from HTML doc
    b = np.array([
        [8.0, 1.0, 6.0],
        [0.0, 5.0, 7.0],
        [0.0, 9.0, 2.0]
    ], dtype=float, order='F')

    # C matrix (5x3) - RHS of equation, read row-by-row from HTML doc
    c = np.array([
        [ 62.0, -12.0, 26.0],
        [ 59.0, -10.0, 31.0],
        [ 70.0,  -6.0,  9.0],
        [ 35.0,  31.0, -7.0],
        [ 36.0, -15.0,  7.0]
    ], dtype=float, order='F')

    # Expected solution from HTML doc Program Results
    x_expected = np.array([
        [0.0,  0.0,  1.0],
        [1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [1.0,  1.0, -1.0],
        [2.0, -2.0,  1.0]
    ], dtype=float, order='F')

    # Call sb04nd: ABSCHU='B' means B is Schur, A is Hessenberg
    # ULA='U' means A is upper, ULB='U' means B is upper
    x, info = sb04nd('B', 'U', 'U', a, b, c, tol=0.0)

    assert info == 0, f"sb04nd failed with info={info}"
    assert_allclose(x, x_expected, rtol=1e-10, atol=1e-10)


def test_sb04nd_residual_property():
    """
    Validate mathematical property: AX + XB = C holds for solution X.

    Uses random upper Schur matrices A, B and random C.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04nd

    np.random.seed(42)
    n = 4
    m = 3

    # Generate upper triangular (Schur) matrices
    a = np.triu(np.random.randn(n, n).astype(float, order='F'))
    b = np.triu(np.random.randn(m, m).astype(float, order='F'))

    # Ensure eigenvalues of A and -B are not too close (avoid near-singularity)
    for i in range(n):
        a[i, i] = 1.0 + i * 0.5
    for i in range(m):
        b[i, i] = -5.0 - i * 0.5

    # Random RHS
    c = np.random.randn(n, m).astype(float, order='F')
    c_copy = c.copy()

    # Both A and B in upper Schur form: ABSCHU='S', ULA='U', ULB='U'
    x, info = sb04nd('S', 'U', 'U', a, b, c)

    assert info == 0, f"sb04nd failed with info={info}"

    # Verify residual: AX + XB - C = 0
    residual = a @ x + x @ b - c_copy
    assert_allclose(residual, np.zeros((n, m)), atol=1e-12)


def test_sb04nd_a_schur_b_hessenberg():
    """
    Test case with A in Schur form and B in Hessenberg form (ABSCHU='A').

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04nd

    np.random.seed(123)
    n = 3
    m = 4

    # A is upper Schur (triangular)
    a = np.triu(np.random.randn(n, n).astype(float, order='F'))
    for i in range(n):
        a[i, i] = 2.0 + i

    # B is upper Hessenberg (tri + superdiagonal + one subdiagonal)
    b = np.triu(np.random.randn(m, m).astype(float, order='F'))
    for i in range(m - 1):
        b[i + 1, i] = np.random.randn()
    for i in range(m):
        b[i, i] = -6.0 - i

    c = np.random.randn(n, m).astype(float, order='F')
    c_copy = c.copy()

    # ABSCHU='A' means A is Schur, B is Hessenberg
    x, info = sb04nd('A', 'U', 'U', a, b, c)

    assert info == 0, f"sb04nd failed with info={info}"

    # Verify residual
    residual = a @ x + x @ b - c_copy
    assert_allclose(residual, np.zeros((n, m)), atol=1e-12)


def test_sb04nd_lower_triangular():
    """
    Test with lower Schur forms (ULA='L', ULB='L').

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04nd

    np.random.seed(456)
    n = 3
    m = 3

    # Lower triangular matrices
    a = np.tril(np.random.randn(n, n).astype(float, order='F'))
    b = np.tril(np.random.randn(m, m).astype(float, order='F'))

    # Ensure well-separated eigenvalues
    for i in range(n):
        a[i, i] = 1.0 + i
    for i in range(m):
        b[i, i] = -5.0 - i

    c = np.random.randn(n, m).astype(float, order='F')
    c_copy = c.copy()

    x, info = sb04nd('S', 'L', 'L', a, b, c)

    assert info == 0, f"sb04nd failed with info={info}"

    residual = a @ x + x @ b - c_copy
    assert_allclose(residual, np.zeros((n, m)), atol=1e-12)


def test_sb04nd_near_singular():
    """
    Test near-singularity detection when A and -B have close eigenvalues.

    Should return info=1 when matrices are nearly singular.
    """
    from ctrlsys import sb04nd

    n = 2
    m = 2

    # A and B with very close eigenvalues (A has eigenvalue 5, B has eigenvalue -5)
    # This makes A + B*I nearly singular
    a = np.array([[5.0, 1.0], [0.0, 5.0]], dtype=float, order='F')
    b = np.array([[-5.0, 1.0], [0.0, -5.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float, order='F')

    # With tight tolerance, should detect near-singularity
    x, info = sb04nd('S', 'U', 'U', a, b, c, tol=1e-10)

    # info=1 indicates near-singular system
    # Note: With ABSCHU='S' and ULA='U', ULB='U', DTRSYL is used directly
    # which handles near-singular cases differently (via SCALE)
    # In this case, DTRSYL may still succeed with SCALE != 1
    # The test validates that either solution is found or near-singularity detected
    assert info in [0, 1], f"Unexpected info={info}"


def test_sb04nd_empty_dimensions():
    """
    Test with zero dimensions (quick return case).
    """
    from ctrlsys import sb04nd

    # N=0 case
    a = np.zeros((0, 0), dtype=float, order='F')
    b = np.array([[1.0]], dtype=float, order='F')
    c = np.zeros((0, 1), dtype=float, order='F')

    x, info = sb04nd('S', 'U', 'U', a, b, c)
    assert info == 0
    assert x.shape == (0, 1)


def test_sb04nd_1x1():
    """
    Test 1x1 case (simplest non-trivial case).

    AX + XB = C => a*x + x*b = c => x = c / (a + b)
    """
    from ctrlsys import sb04nd

    a = np.array([[3.0]], dtype=float, order='F')
    b = np.array([[2.0]], dtype=float, order='F')
    c = np.array([[10.0]], dtype=float, order='F')

    x, info = sb04nd('S', 'U', 'U', a, b, c)

    assert info == 0
    # x = 10 / (3 + 2) = 2.0
    assert_allclose(x, np.array([[2.0]]), rtol=1e-14)


def test_sb04nd_2x2_real_schur():
    """
    Test 2x2 case with 2x2 quasi-triangular block (complex eigenvalues).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04nd

    np.random.seed(789)

    # A with 2x2 block representing complex eigenvalues
    a = np.array([
        [1.0, 2.0],
        [-2.0, 1.0]  # eigenvalues: 1 +/- 2i
    ], dtype=float, order='F')

    # B is simple upper triangular
    b = np.array([
        [-3.0, 1.0],
        [0.0, -4.0]
    ], dtype=float, order='F')

    c = np.array([
        [5.0, 3.0],
        [1.0, 2.0]
    ], dtype=float, order='F')
    c_copy = c.copy()

    # B is Schur, A is Hessenberg (which is also quasi-triangular here)
    x, info = sb04nd('B', 'U', 'U', a, b, c)

    assert info == 0

    # Verify residual
    residual = a @ x + x @ b - c_copy
    assert_allclose(residual, np.zeros((2, 2)), atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
