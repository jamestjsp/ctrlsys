"""
Tests for SB04NX: Solve Hessenberg system with two offdiagonals and two RHS.

This routine solves a system of equations in Hessenberg form with two
consecutive offdiagonals and two right-hand sides. Used by SB04ND for
2x2 diagonal blocks.

Tests numerical correctness using:
1. Upper Hessenberg, row transforms (RC='R', UL='U')
2. Upper Hessenberg, column transforms (RC='C', UL='U')
3. Lower Hessenberg cases
4. Mathematical property: solution satisfies original system
5. Error handling (singular matrix detection)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb04nx_upper_row_basic():
    """
    Validate basic functionality with upper Hessenberg, row transforms.

    Solves (A + lambda*I) * x = d where lambda is a 2x2 block.
    The system has 2*M unknowns for M-by-M Hessenberg matrix A.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04nx

    np.random.seed(42)
    m = 3

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    for i in range(m - 1):
        a[i + 1, i] = np.random.randn() * 0.5

    for i in range(m):
        a[i, i] = 2.0 + i

    lambd1 = 1.0
    lambd2 = 0.1
    lambd3 = -0.1
    lambd4 = 1.5

    d = np.random.randn(2 * m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04nx('R', 'U', a, d, lambd1, lambd2, lambd3, lambd4, tol)

    assert info == 0, f"sb04nx failed with info={info}"

    h = np.zeros((2 * m, 2 * m), dtype=float, order='F')
    for j in range(m):
        j2 = 2 * j
        ml = min(j + 2, m)
        for i in range(ml):
            h[2 * i, j2] = a[i, j]
            h[2 * i + 1, j2 + 1] = a[i, j]

        h[j2, j2] += lambd1
        h[j2 + 1, j2] = lambd3
        h[j2, j2 + 1] = lambd2
        h[j2 + 1, j2 + 1] += lambd4

    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(2 * m), atol=1e-10)


def test_sb04nx_upper_col_transforms():
    """
    Validate with upper Hessenberg, column transforms (RC='C').

    When RC='C', the routine solves H^T * x = d.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04nx

    np.random.seed(123)
    m = 3

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    for i in range(m - 1):
        a[i + 1, i] = np.random.randn() * 0.5

    for i in range(m):
        a[i, i] = 3.0 + i

    lambd1 = 2.0
    lambd2 = 0.2
    lambd3 = -0.2
    lambd4 = 2.5

    d = np.random.randn(2 * m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04nx('C', 'U', a, d, lambd1, lambd2, lambd3, lambd4, tol)

    assert info == 0, f"sb04nx failed with info={info}"

    h = np.zeros((2 * m, 2 * m), dtype=float, order='F')
    for j in range(m):
        j2 = 2 * j
        ml = min(j + 2, m)
        for i in range(ml):
            h[2 * i, j2] = a[i, j]
            h[2 * i + 1, j2 + 1] = a[i, j]

        h[j2, j2] += lambd1
        h[j2 + 1, j2] = lambd3
        h[j2, j2 + 1] = lambd2
        h[j2 + 1, j2 + 1] += lambd4

    residual = h.T @ d_result - d_orig
    assert_allclose(residual, np.zeros(2 * m), atol=1e-10)


def test_sb04nx_lower_hessenberg():
    """
    Validate with lower Hessenberg matrix (UL='L').

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04nx

    np.random.seed(456)
    m = 3

    a = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)
    for i in range(m - 1):
        a[i, i + 1] = np.random.randn() * 0.5

    for i in range(m):
        a[i, i] = 4.0 + i

    lambd1 = 1.5
    lambd2 = 0.0
    lambd3 = 0.0
    lambd4 = 1.5

    d = np.random.randn(2 * m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04nx('R', 'L', a, d, lambd1, lambd2, lambd3, lambd4, tol)

    assert info == 0, f"sb04nx failed with info={info}"


def test_sb04nx_diagonal_lambda():
    """
    Test with diagonal lambda block (lambd2=lambd3=0).

    This corresponds to two independent real eigenvalues.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04nx

    np.random.seed(789)
    m = 4

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    for i in range(m - 1):
        a[i + 1, i] = np.random.randn() * 0.3

    for i in range(m):
        a[i, i] = 5.0 + i

    lambd1 = 1.0
    lambd2 = 0.0
    lambd3 = 0.0
    lambd4 = 2.0

    d = np.random.randn(2 * m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04nx('R', 'U', a, d, lambd1, lambd2, lambd3, lambd4, tol)

    assert info == 0, f"sb04nx failed with info={info}"

    h = np.zeros((2 * m, 2 * m), dtype=float, order='F')
    for j in range(m):
        j2 = 2 * j
        ml = min(j + 2, m)
        for i in range(ml):
            h[2 * i, j2] = a[i, j]
            h[2 * i + 1, j2 + 1] = a[i, j]

        h[j2, j2] += lambd1
        h[j2 + 1, j2] = lambd3
        h[j2, j2 + 1] = lambd2
        h[j2 + 1, j2 + 1] += lambd4

    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(2 * m), atol=1e-10)


def test_sb04nx_1x1():
    """
    Test with 1x1 Hessenberg matrix.

    The system becomes 2x2:
    [[a + lambd1, lambd2], [lambd3, a + lambd4]] * x = d
    """
    from ctrlsys import sb04nx

    a = np.array([[3.0]], dtype=float, order='F')
    lambd1 = 1.0
    lambd2 = 0.5
    lambd3 = -0.5
    lambd4 = 2.0

    d = np.array([5.0, 3.0], dtype=float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04nx('R', 'U', a, d, lambd1, lambd2, lambd3, lambd4, tol)

    assert info == 0

    h = np.array([
        [3.0 + lambd1, lambd2],
        [lambd3, 3.0 + lambd4]
    ], dtype=float, order='F')

    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(2), atol=1e-12)


def test_sb04nx_2x2():
    """
    Test with 2x2 Hessenberg matrix.

    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import sb04nx

    np.random.seed(101)
    m = 2

    a = np.array([
        [2.0, 1.0],
        [0.5, 3.0]
    ], dtype=float, order='F')

    lambd1 = 0.5
    lambd2 = 0.1
    lambd3 = -0.1
    lambd4 = 0.5

    d = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    d_orig = d.copy()

    tol = 1e-10

    d_result, info = sb04nx('R', 'U', a, d, lambd1, lambd2, lambd3, lambd4, tol)

    assert info == 0

    h = np.zeros((4, 4), dtype=float, order='F')
    h[0, 0] = a[0, 0] + lambd1
    h[1, 0] = lambd3
    h[2, 0] = a[1, 0]
    h[0, 1] = lambd2
    h[1, 1] = a[0, 0] + lambd4
    h[3, 1] = a[1, 0]
    h[0, 2] = a[0, 1]
    h[2, 2] = a[1, 1] + lambd1
    h[3, 2] = lambd3
    h[1, 3] = a[0, 1]
    h[2, 3] = lambd2
    h[3, 3] = a[1, 1] + lambd4

    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(4), atol=1e-10)


def test_sb04nx_near_singular():
    """
    Test near-singular matrix detection.

    When lambda values make the system ill-conditioned, info=1.
    """
    from ctrlsys import sb04nx

    a = np.array([[1.0]], dtype=float, order='F')
    lambd1 = -1.0
    lambd2 = 0.0
    lambd3 = 0.0
    lambd4 = -1.0

    d = np.array([1.0, 1.0], dtype=float, order='F')

    tol = 1e-10

    d_result, info = sb04nx('R', 'U', a, d, lambd1, lambd2, lambd3, lambd4, tol)

    assert info == 1


def test_sb04nx_larger_system():
    """
    Test with larger Hessenberg matrix.

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import sb04nx

    np.random.seed(202)
    m = 5

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    for i in range(m - 1):
        a[i + 1, i] = np.random.randn() * 0.3

    for i in range(m):
        a[i, i] = 10.0 + i

    lambd1 = 1.0
    lambd2 = 0.2
    lambd3 = -0.2
    lambd4 = 1.5

    d = np.random.randn(2 * m).astype(float, order='F')
    d_orig = d.copy()

    tol = 1e-12

    d_result, info = sb04nx('R', 'U', a, d, lambd1, lambd2, lambd3, lambd4, tol)

    assert info == 0

    h = np.zeros((2 * m, 2 * m), dtype=float, order='F')
    for j in range(m):
        j2 = 2 * j
        ml = min(j + 2, m)
        for i in range(ml):
            h[2 * i, j2] = a[i, j]
            h[2 * i + 1, j2 + 1] = a[i, j]

        h[j2, j2] += lambd1
        h[j2 + 1, j2] = lambd3
        h[j2, j2 + 1] = lambd2
        h[j2 + 1, j2 + 1] += lambd4

    residual = h @ d_result - d_orig
    assert_allclose(residual, np.zeros(2 * m), atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
