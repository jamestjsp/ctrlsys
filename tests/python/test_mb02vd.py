"""
Tests for MB02VD: Solution of X * op(A) = B via LU factorization.

Tests:
1. Basic functionality: solve X * A = B
2. Transpose case: solve X * A' = B
3. Singular matrix detection: info > 0

Random seed: 42, 123, 456 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb02vd_basic():
    """
    Validate basic functionality: solve X * A = B.

    Tests numerical correctness by verifying the solution satisfies X * A = B.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb02vd

    np.random.seed(42)

    n = 4
    m = 3

    # Create non-singular matrix A
    a = np.array([
        [4.0, 2.0, 1.0, 3.0],
        [1.0, 5.0, 2.0, 1.0],
        [2.0, 1.0, 6.0, 2.0],
        [1.0, 2.0, 1.0, 7.0]
    ], order='F', dtype=float)

    # Create right-hand side B
    b = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    b_orig = b.copy()

    x, ipiv, info = mb02vd('N', a.copy(order='F'), b.copy(order='F'))

    assert info == 0

    # Verify X * A = B (within tolerance)
    residual = x @ a_orig - b_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)


def test_mb02vd_transpose():
    """
    Validate transpose case: solve X * A' = B.

    Tests numerical correctness by verifying the solution satisfies X * A' = B.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb02vd

    np.random.seed(123)

    n = 3
    m = 2

    # Create non-singular matrix A
    a = np.array([
        [3.0, 1.0, 2.0],
        [1.0, 4.0, 1.0],
        [2.0, 1.0, 5.0]
    ], order='F', dtype=float)

    # Create right-hand side B
    b = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    b_orig = b.copy()

    x, ipiv, info = mb02vd('T', a.copy(order='F'), b.copy(order='F'))

    assert info == 0

    # Verify X * A' = B (within tolerance)
    residual = x @ a_orig.T - b_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)


def test_mb02vd_singular_matrix():
    """
    Validate detection of singular matrix.

    When A is singular, info > 0 indicates the index of the zero pivot.
    """
    from ctrlsys import mb02vd

    n = 3
    m = 2

    # Create singular matrix A (row 3 = row 1 + row 2)
    a = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 2.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], order='F', dtype=float)

    x, ipiv, info = mb02vd('N', a.copy(order='F'), b.copy(order='F'))

    # info > 0 means the matrix is singular
    assert info > 0


def test_mb02vd_random_system():
    """
    Validate with random well-conditioned system.

    Generates a random diagonally dominant matrix (guaranteed non-singular)
    and verifies the solution.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb02vd

    np.random.seed(456)

    n = 5
    m = 4

    # Create random diagonally dominant matrix (guaranteed non-singular)
    a = np.random.randn(n, n).astype(float, order='F')
    a = a + n * np.eye(n)  # Make diagonally dominant

    # Random right-hand side
    b = np.random.randn(m, n).astype(float, order='F')

    a_orig = a.copy()
    b_orig = b.copy()

    x, ipiv, info = mb02vd('N', a.copy(order='F'), b.copy(order='F'))

    assert info == 0

    # Verify solution
    residual = x @ a_orig - b_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-12)


def test_mb02vd_identity():
    """
    Validate with identity matrix: X * I = B implies X = B.
    """
    from ctrlsys import mb02vd

    n = 3
    m = 2

    a = np.eye(n, order='F', dtype=float)
    b = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], order='F', dtype=float)

    b_orig = b.copy()

    x, ipiv, info = mb02vd('N', a.copy(order='F'), b.copy(order='F'))

    assert info == 0
    assert_allclose(x, b_orig, rtol=1e-14)
