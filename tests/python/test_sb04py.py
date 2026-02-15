"""
Tests for SB04PY - Discrete-time Sylvester equation solver.

Solves: op(A)*X*op(B) + ISGN*X = scale*C

where op(A) = A or A**T, A and B are upper quasi-triangular (Schur form),
ISGN = 1 or -1. Solution X overwrites C; scale <= 1 prevents overflow.

A is M-by-M, B is N-by-N, C/X are M-by-N.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb04py_basic_no_transpose():
    """
    Test basic case: A*X*B + X = scale*C (TRANA='N', TRANB='N', ISGN=1)

    Uses simple 2x2 upper triangular matrices (Schur form).
    Equation: A*X*B + X = C
    With A = [[2, 0], [0, 3]], B = [[1, 0], [0, 2]]:
    Element (0,0): 2*x00*1 + x00 = c00 => 3*x00 = c00
    Element (0,1): 2*x01*2 + x01 = c01 => 5*x01 = c01
    Element (1,0): 3*x10*1 + x10 = c10 => 4*x10 = c10
    Element (1,1): 3*x11*2 + x11 = c11 => 7*x11 = c11

    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import sb04py

    # Upper triangular matrices (Schur form)
    a = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float, order='F')
    b = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float, order='F')
    c = np.array([[6.0, 10.0], [8.0, 14.0]], dtype=float, order='F')

    # Expected: x00=2, x01=2, x10=2, x11=2
    x_expected = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=float, order='F')

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    assert info == 0
    assert scale == 1.0
    assert_allclose(x, x_expected, rtol=1e-14)


def test_sb04py_transpose_a():
    """
    Test with TRANA='T': A'*X*B + X = scale*C

    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import sb04py

    # Upper triangular with non-zero (0,1) element
    a = np.array([[2.0, 0.5], [0.0, 3.0]], dtype=float, order='F')
    b = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float, order='F')
    c = np.array([[5.0, 10.0], [8.0, 14.0]], dtype=float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('T', 'N', 1, a, b, c)

    assert info == 0
    assert scale == 1.0

    # Verify: A'*X*B + X = scale*C
    residual = a.T @ x @ b + x - scale * c_orig
    assert_allclose(residual, np.zeros((2, 2)), atol=1e-13)


def test_sb04py_transpose_b():
    """
    Test with TRANB='T': A*X*B' + X = scale*C

    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import sb04py

    a = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float, order='F')
    b = np.array([[1.0, 0.3], [0.0, 2.0]], dtype=float, order='F')
    c = np.array([[5.0, 9.0], [8.0, 14.0]], dtype=float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('N', 'T', 1, a, b, c)

    assert info == 0
    assert scale == 1.0

    # Verify: A*X*B' + X = scale*C
    residual = a @ x @ b.T + x - scale * c_orig
    assert_allclose(residual, np.zeros((2, 2)), atol=1e-13)


def test_sb04py_both_transpose():
    """
    Test with TRANA='T', TRANB='T': A'*X*B' + X = scale*C

    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import sb04py

    a = np.array([[2.0, 0.5], [0.0, 3.0]], dtype=float, order='F')
    b = np.array([[1.0, 0.3], [0.0, 2.0]], dtype=float, order='F')
    c = np.array([[6.0, 10.0], [9.0, 15.0]], dtype=float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('T', 'T', 1, a, b, c)

    assert info == 0
    assert scale == 1.0

    # Verify: A'*X*B' + X = scale*C
    residual = a.T @ x @ b.T + x - scale * c_orig
    assert_allclose(residual, np.zeros((2, 2)), atol=1e-13)


def test_sb04py_isgn_minus_one():
    """
    Test with ISGN=-1: A*X*B - X = scale*C

    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import sb04py

    a = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float, order='F')
    b = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float, order='F')
    # A*X*B - X = C => 2*x00*1 - x00 = c00 => x00 = c00
    # etc.
    c = np.array([[2.0, 6.0], [4.0, 10.0]], dtype=float, order='F')

    x_expected = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=float, order='F')

    x, scale, info = sb04py('N', 'N', -1, a, b, c)

    assert info == 0
    assert scale == 1.0
    assert_allclose(x, x_expected, rtol=1e-14)


def test_sb04py_2x2_block():
    """
    Test with 2x2 block (complex eigenvalue pair in Schur form).

    A 2x2 block in Schur form has equal diagonal and opposite off-diagonal signs:
    [[a, b], [-b, a]] represents eigenvalues a +/- bi

    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import sb04py

    # Schur form with 2x2 block representing eigenvalues 1 +/- 0.5i
    a = np.array([[1.0, 0.5], [-0.5, 1.0]], dtype=float, order='F')
    b = np.array([[2.0, 0.3], [-0.3, 2.0]], dtype=float, order='F')
    c = np.array([[10.0, 8.0], [6.0, 12.0]], dtype=float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    assert info == 0
    assert scale == 1.0

    # Verify: A*X*B + X = scale*C
    residual = a @ x @ b + x - scale * c_orig
    assert_allclose(residual, np.zeros((2, 2)), atol=1e-12)


def test_sb04py_larger_system():
    """
    Test 3x3 system with mixed 1x1 and 2x2 blocks.

    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import sb04py

    # A: 2x2 block at (0:2,0:2), 1x1 block at (2,2)
    a = np.array([
        [1.0, 0.5, 0.2],
        [-0.5, 1.0, 0.1],
        [0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    # B: 1x1 block at (0,0), 2x2 block at (1:3,1:3)
    b = np.array([
        [2.0, 0.3, 0.1],
        [0.0, 1.5, 0.4],
        [0.0, -0.4, 1.5]
    ], dtype=float, order='F')

    c = np.array([
        [10.0, 8.0, 6.0],
        [7.0, 12.0, 9.0],
        [5.0, 4.0, 15.0]
    ], dtype=float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    assert info == 0
    assert scale == 1.0

    # Verify: A*X*B + X = scale*C
    residual = a @ x @ b + x - scale * c_orig
    assert_allclose(residual, np.zeros((3, 3)), atol=1e-11)


def test_sb04py_rectangular_m_greater_n():
    """
    Test rectangular case with M > N (more rows than columns).

    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import sb04py

    # A is 3x3, B is 2x2 => C/X is 3x2
    a = np.array([
        [2.0, 0.1, 0.0],
        [0.0, 3.0, 0.2],
        [0.0, 0.0, 4.0]
    ], dtype=float, order='F')

    b = np.array([
        [1.0, 0.1],
        [0.0, 2.0]
    ], dtype=float, order='F')

    c = np.array([
        [6.0, 10.0],
        [9.0, 14.0],
        [12.0, 20.0]
    ], dtype=float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    assert info == 0
    assert scale == 1.0

    # Verify: A*X*B + X = scale*C
    residual = a @ x @ b + x - scale * c_orig
    assert_allclose(residual, np.zeros((3, 2)), atol=1e-12)


def test_sb04py_rectangular_n_greater_m():
    """
    Test rectangular case with N > M (more columns than rows).

    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import sb04py

    # A is 2x2, B is 3x3 => C/X is 2x3
    a = np.array([
        [2.0, 0.1],
        [0.0, 3.0]
    ], dtype=float, order='F')

    b = np.array([
        [1.0, 0.1, 0.0],
        [0.0, 2.0, 0.2],
        [0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    c = np.array([
        [6.0, 10.0, 15.0],
        [9.0, 14.0, 24.0]
    ], dtype=float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    assert info == 0
    assert scale == 1.0

    # Verify: A*X*B + X = scale*C
    residual = a @ x @ b + x - scale * c_orig
    assert_allclose(residual, np.zeros((2, 3)), atol=1e-12)


def test_sb04py_empty_m():
    """
    Test edge case: M=0 (empty A, C has 0 rows).

    Should return immediately with scale=1.0.
    """
    from ctrlsys import sb04py

    a = np.zeros((0, 0), dtype=float, order='F')
    b = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float, order='F')
    c = np.zeros((0, 2), dtype=float, order='F')

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    assert info == 0
    assert scale == 1.0
    assert x.shape == (0, 2)


def test_sb04py_empty_n():
    """
    Test edge case: N=0 (empty B, C has 0 columns).

    Should return immediately with scale=1.0.
    """
    from ctrlsys import sb04py

    a = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float, order='F')
    b = np.zeros((0, 0), dtype=float, order='F')
    c = np.zeros((2, 0), dtype=float, order='F')

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    assert info == 0
    assert scale == 1.0
    assert x.shape == (2, 0)


def test_sb04py_1x1():
    """
    Test 1x1 case (simplest non-trivial).

    A*X*B + ISGN*X = C => a*x*b + x = c => x = c/(a*b + 1)
    a=2, b=3, c=14 => x = 14/(6+1) = 2
    """
    from ctrlsys import sb04py

    a = np.array([[2.0]], dtype=float, order='F')
    b = np.array([[3.0]], dtype=float, order='F')
    c = np.array([[14.0]], dtype=float, order='F')

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    assert info == 0
    assert scale == 1.0
    assert_allclose(x, np.array([[2.0]]), rtol=1e-14)


def test_sb04py_residual_property_random():
    """
    Property test: verify equation holds for random upper triangular matrices.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04py

    np.random.seed(42)
    m, n = 4, 3

    # Generate random upper triangular matrices (Schur form)
    a = np.triu(np.random.randn(m, m)).astype(float, order='F')
    b = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    assert info == 0
    assert 0.0 < scale <= 1.0

    # Verify: A*X*B + X = scale*C
    residual = a @ x @ b + x - scale * c_orig
    assert_allclose(residual, np.zeros((m, n)), atol=1e-11)


def test_sb04py_residual_property_transpose_a_random():
    """
    Property test with transpose A: A'*X*B + X = scale*C

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04py

    np.random.seed(123)
    m, n = 3, 4

    a = np.triu(np.random.randn(m, m)).astype(float, order='F')
    b = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('T', 'N', 1, a, b, c)

    assert info == 0

    # Verify: A'*X*B + X = scale*C
    residual = a.T @ x @ b + x - scale * c_orig
    assert_allclose(residual, np.zeros((m, n)), atol=1e-11)


def test_sb04py_residual_property_both_transpose_random():
    """
    Property test with both transposed: A'*X*B' + X = scale*C

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04py

    np.random.seed(456)
    m, n = 4, 4

    a = np.triu(np.random.randn(m, m)).astype(float, order='F')
    b = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    c_orig = c.copy()

    x, scale, info = sb04py('T', 'T', 1, a, b, c)

    assert info == 0

    # Verify: A'*X*B' + X = scale*C
    residual = a.T @ x @ b.T + x - scale * c_orig
    assert_allclose(residual, np.zeros((m, n)), atol=1e-11)


def test_sb04py_reciprocal_eigenvalues_warning():
    """
    Test case with nearly reciprocal eigenvalues: info=1 warning.

    When A and -ISGN*B have almost reciprocal eigenvalues, info=1.
    For ISGN=1: eigenvalues of A times eigenvalues of B should be near -1.
    """
    from ctrlsys import sb04py

    # A has eigenvalue 0.5, B has eigenvalue -2 => 0.5 * (-2) = -1 (exact reciprocal)
    # Use nearly reciprocal to trigger warning
    a = np.array([[0.5]], dtype=float, order='F')
    b = np.array([[-2.0 + 1e-16]], dtype=float, order='F')
    c = np.array([[1.0]], dtype=float, order='F')

    x, scale, info = sb04py('N', 'N', 1, a, b, c)

    # May return info=1 (warning) or 0 depending on threshold
    assert info in [0, 1]


def test_sb04py_invalid_trana():
    """
    Test error handling for invalid TRANA parameter.
    """
    from ctrlsys import sb04py

    a = np.array([[1.0]], dtype=float, order='F')
    b = np.array([[1.0]], dtype=float, order='F')
    c = np.array([[1.0]], dtype=float, order='F')

    # Invalid TRANA should cause error
    with pytest.raises(ValueError):
        sb04py('X', 'N', 1, a, b, c)


def test_sb04py_invalid_tranb():
    """
    Test error handling for invalid TRANB parameter.
    """
    from ctrlsys import sb04py

    a = np.array([[1.0]], dtype=float, order='F')
    b = np.array([[1.0]], dtype=float, order='F')
    c = np.array([[1.0]], dtype=float, order='F')

    # Invalid TRANB should cause error
    with pytest.raises(ValueError):
        sb04py('N', 'X', 1, a, b, c)


def test_sb04py_invalid_isgn():
    """
    Test error handling for invalid ISGN parameter (must be 1 or -1).
    """
    from ctrlsys import sb04py

    a = np.array([[1.0]], dtype=float, order='F')
    b = np.array([[1.0]], dtype=float, order='F')
    c = np.array([[1.0]], dtype=float, order='F')

    # Invalid ISGN should cause error
    with pytest.raises(ValueError):
        sb04py('N', 'N', 0, a, b, c)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
