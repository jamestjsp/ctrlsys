"""
Tests for SB04NY: Solve Hessenberg system with one offdiagonal and one RHS.

This routine solves a system of equations in Hessenberg form with one
offdiagonal and one right-hand side. Used by SB04ND for 1x1 diagonal blocks.

Tests numerical correctness using:
1. Upper Hessenberg, row transforms (RC='R', UL='U')
2. Upper Hessenberg, column transforms (RC='C', UL='U')
3. Lower Hessenberg cases
4. Mathematical property: solution satisfies (A + lambda*I) * x = d
5. Error handling (singular matrix detection)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb04ny_upper_row_basic():
    """
    Validate basic functionality with upper Hessenberg, row transforms.

    Solves (A + lambda*I) * x = d.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04ny

    np.random.seed(42)
    m = 4

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    for i in range(m - 1):
        a[i + 1, i] = np.random.randn() * 0.5

    for i in range(m):
        a[i, i] = 2.0 + i

    lambda_val = 1.0

    d = np.random.randn(m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04ny('R', 'U', a, d, lambda_val, tol)

    assert info == 0, f"sb04ny failed with info={info}"

    h = a.copy()
    for i in range(m):
        h[i, i] += lambda_val

    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(m), atol=1e-12)


def test_sb04ny_upper_col_transforms():
    """
    Validate with upper Hessenberg, column transforms (RC='C').

    When RC='C', the routine solves (A + lambda*I)^T * x = d.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04ny

    np.random.seed(123)
    m = 4

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    for i in range(m - 1):
        a[i + 1, i] = np.random.randn() * 0.5

    for i in range(m):
        a[i, i] = 3.0 + i

    lambda_val = 2.0

    d = np.random.randn(m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04ny('C', 'U', a, d, lambda_val, tol)

    assert info == 0, f"sb04ny failed with info={info}"

    h = a.copy()
    for i in range(m):
        h[i, i] += lambda_val

    residual = h.T @ d_result - d_orig
    assert_allclose(residual, np.zeros(m), atol=1e-12)


def test_sb04ny_lower_row():
    """
    Validate with lower Hessenberg matrix (UL='L'), row transforms.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04ny

    np.random.seed(456)
    m = 4

    a = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)
    for i in range(m - 1):
        a[i, i + 1] = np.random.randn() * 0.5

    for i in range(m):
        a[i, i] = 4.0 + i

    lambda_val = 1.5

    d = np.random.randn(m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04ny('R', 'L', a, d, lambda_val, tol)

    assert info == 0, f"sb04ny failed with info={info}"

    h = a.copy()
    for i in range(m):
        h[i, i] += lambda_val

    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(m), atol=1e-12)


def test_sb04ny_lower_col():
    """
    Validate with lower Hessenberg matrix (UL='L'), column transforms.

    When RC='C', the routine solves (A + lambda*I)^T * x = d.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04ny

    np.random.seed(789)
    m = 4

    a = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)
    for i in range(m - 1):
        a[i, i + 1] = np.random.randn() * 0.5

    for i in range(m):
        a[i, i] = 5.0 + i

    lambda_val = 0.5

    d = np.random.randn(m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04ny('C', 'L', a, d, lambda_val, tol)

    assert info == 0, f"sb04ny failed with info={info}"

    h = a.copy()
    for i in range(m):
        h[i, i] += lambda_val

    residual = h.T @ d_result - d_orig
    assert_allclose(residual, np.zeros(m), atol=1e-12)


def test_sb04ny_1x1():
    """
    Test with 1x1 matrix.

    (a + lambda) * x = d => x = d / (a + lambda)
    """
    from ctrlsys import sb04ny

    a = np.array([[3.0]], dtype=float, order='F')
    lambda_val = 2.0
    d = np.array([10.0], dtype=float, order='F')

    tol = 1e-10

    d_result, info = sb04ny('R', 'U', a, d, lambda_val, tol)

    assert info == 0

    x_expected = 10.0 / (3.0 + 2.0)
    assert_allclose(d_result, np.array([x_expected]), rtol=1e-14)


def test_sb04ny_2x2():
    """
    Test with 2x2 upper Hessenberg matrix.

    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import sb04ny

    np.random.seed(101)

    a = np.array([
        [2.0, 1.0],
        [0.5, 3.0]
    ], dtype=float, order='F')

    lambda_val = 1.0

    d = np.array([5.0, 7.0], dtype=float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04ny('R', 'U', a, d, lambda_val, tol)

    assert info == 0

    h = a + lambda_val * np.eye(2)
    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(2), atol=1e-12)


def test_sb04ny_triangular():
    """
    Test with strictly upper triangular + diagonal (no subdiagonal).

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import sb04ny

    np.random.seed(202)
    m = 5

    a = np.triu(np.random.randn(m, m).astype(float, order='F'))
    for i in range(m):
        a[i, i] = 5.0 + i

    lambda_val = 2.0

    d = np.random.randn(m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04ny('R', 'U', a, d, lambda_val, tol)

    assert info == 0

    h = a + lambda_val * np.eye(m)
    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(m), atol=1e-12)


def test_sb04ny_near_singular():
    """
    Test near-singular matrix detection.

    When lambda makes diagonal near-zero, should return info=1.
    """
    from ctrlsys import sb04ny

    a = np.array([[1.0]], dtype=float, order='F')
    lambda_val = -1.0

    d = np.array([1.0], dtype=float, order='F')

    tol = 1e-10

    d_result, info = sb04ny('R', 'U', a, d, lambda_val, tol)

    assert info == 1


def test_sb04ny_negative_lambda():
    """
    Test with negative lambda value.

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import sb04ny

    np.random.seed(303)
    m = 3

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    for i in range(m - 1):
        a[i + 1, i] = np.random.randn() * 0.3

    for i in range(m):
        a[i, i] = 10.0 + i

    lambda_val = -3.0

    d = np.random.randn(m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04ny('R', 'U', a, d, lambda_val, tol)

    assert info == 0

    h = a + lambda_val * np.eye(m)
    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(m), atol=1e-12)


def test_sb04ny_larger_system():
    """
    Test with larger Hessenberg matrix.

    Random seed: 404 (for reproducibility)
    """
    from ctrlsys import sb04ny

    np.random.seed(404)
    m = 8

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    for i in range(m - 1):
        a[i + 1, i] = np.random.randn() * 0.2

    for i in range(m):
        a[i, i] = 15.0 + i

    lambda_val = 2.5

    d = np.random.randn(m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-12

    d_result, info = sb04ny('R', 'U', a, d, lambda_val, tol)

    assert info == 0

    h = a + lambda_val * np.eye(m)
    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(m), atol=1e-10)


def test_sb04ny_compare_row_col():
    """
    Verify row and column transforms solve related systems.

    RC='R' solves: (A + lambda*I) * x = d
    RC='C' solves: (A + lambda*I)^T * x = d

    Both should correctly solve their respective systems.

    Random seed: 505 (for reproducibility)
    """
    from ctrlsys import sb04ny

    np.random.seed(505)
    m = 4

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    for i in range(m - 1):
        a[i + 1, i] = np.random.randn() * 0.3

    for i in range(m):
        a[i, i] = 8.0 + i

    lambda_val = 1.0

    d_row = np.random.randn(m).astype(float, order='F')
    d_col = d_row.copy()
    d_orig = d_row.copy()

    tol = 1e-10

    d_row_result, info_row = sb04ny('R', 'U', a, d_row, lambda_val, tol)
    d_col_result, info_col = sb04ny('C', 'U', a, d_col, lambda_val, tol)

    assert info_row == 0
    assert info_col == 0

    h = a.copy()
    for i in range(m):
        h[i, i] += lambda_val

    residual_row = h @ d_row_result - d_orig
    residual_col = h.T @ d_col_result - d_orig

    assert_allclose(residual_row, np.zeros(m), atol=1e-12)
    assert_allclose(residual_col, np.zeros(m), atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
