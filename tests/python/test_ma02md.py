"""
Tests for MA02MD - Compute norms of a real skew-symmetric matrix.

MA02MD computes the value of the one norm, or the Frobenius norm, or
the infinity norm, or the element of largest absolute value
of a real skew-symmetric matrix.

Note that for skew-symmetric matrices, the infinity norm equals the one norm.
"""
import numpy as np
import pytest
from ctrlsys import ma02md


def test_ma02md_max_norm_upper():
    """
    Test max norm (M) with upper triangular storage.

    For a skew-symmetric matrix, max norm is max(|a_ij|).
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    a = np.zeros((n, n), order='F', dtype=float)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn()
            a[j, i] = -a[i, j]

    expected = np.max(np.abs(a))

    norm = ma02md('M', 'U', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02md_max_norm_lower():
    """
    Test max norm (M) with lower triangular storage.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 5

    a = np.zeros((n, n), order='F', dtype=float)
    for j in range(n - 1):
        for i in range(j + 1, n):
            a[i, j] = np.random.randn()
            a[j, i] = -a[i, j]

    expected = np.max(np.abs(a))

    norm = ma02md('M', 'L', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02md_one_norm_upper():
    """
    Test one norm (1/O) with upper triangular storage.

    The one norm of a matrix is the maximum column sum of absolute values.
    For skew-symmetric matrices, this equals the infinity norm.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4

    a = np.zeros((n, n), order='F', dtype=float)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn()
            a[j, i] = -a[i, j]

    expected = np.linalg.norm(a, 1)

    norm = ma02md('1', 'U', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02md_one_norm_lower():
    """
    Test one norm with lower triangular storage.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5

    a = np.zeros((n, n), order='F', dtype=float)
    for j in range(n - 1):
        for i in range(j + 1, n):
            a[i, j] = np.random.randn()
            a[j, i] = -a[i, j]

    expected = np.linalg.norm(a, 1)

    norm = ma02md('O', 'L', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02md_infinity_norm():
    """
    Test infinity norm (I).

    For skew-symmetric matrices, infinity norm equals one norm.
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 4

    a = np.zeros((n, n), order='F', dtype=float)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn()
            a[j, i] = -a[i, j]

    expected = np.linalg.norm(a, np.inf)

    norm = ma02md('I', 'U', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02md_frobenius_norm_upper():
    """
    Test Frobenius norm (F/E) with upper triangular storage.

    Frobenius norm is sqrt(sum of squares of all elements).
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 4

    a = np.zeros((n, n), order='F', dtype=float)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn()
            a[j, i] = -a[i, j]

    expected = np.linalg.norm(a, 'fro')

    norm = ma02md('F', 'U', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-13)


def test_ma02md_frobenius_norm_lower():
    """
    Test Frobenius norm with lower triangular storage.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 5

    a = np.zeros((n, n), order='F', dtype=float)
    for j in range(n - 1):
        for i in range(j + 1, n):
            a[i, j] = np.random.randn()
            a[j, i] = -a[i, j]

    expected = np.linalg.norm(a, 'fro')

    norm = ma02md('E', 'L', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-13)


def test_ma02md_n_zero():
    """
    Edge case: n = 0 should return 0.
    """
    a = np.array([], order='F', dtype=float).reshape(0, 0)

    norm = ma02md('F', 'U', a)

    assert norm == 0.0


def test_ma02md_n_one():
    """
    Edge case: n = 1.

    A 1x1 skew-symmetric matrix is just [0], so all norms are 0.
    """
    a = np.array([[5.0]], order='F', dtype=float)

    norm_m = ma02md('M', 'U', a)
    norm_1 = ma02md('1', 'U', a)
    norm_f = ma02md('F', 'U', a)

    assert norm_m == 0.0
    assert norm_1 == 0.0
    assert norm_f == 0.0


def test_ma02md_skew_symmetry_property():
    """
    Mathematical property: verify that norm equality for skew-symmetric matrices.

    For skew-symmetric matrices:
    - ||A||_1 = ||A||_inf (one norm equals infinity norm)
    - ||A||_F = sqrt(2) * ||strictly_upper(A)||_F

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 6

    a = np.zeros((n, n), order='F', dtype=float)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn()
            a[j, i] = -a[i, j]

    norm_1 = ma02md('1', 'U', a)
    norm_inf = ma02md('I', 'U', a)

    np.testing.assert_allclose(norm_1, norm_inf, rtol=1e-14)


def test_ma02md_simple_known_matrix():
    """
    Test with a simple known skew-symmetric matrix.

    A = [  0,  2, -3]
        [ -2,  0,  4]
        [  3, -4,  0]

    Max norm: max(2, 3, 4) = 4
    One norm: max(0+2+3, 2+0+4, 3+4+0) = max(5, 6, 7) = 7
    Frobenius norm: sqrt(2*(2^2 + 3^2 + 4^2)) = sqrt(2*29) = sqrt(58)
    """
    a = np.array([
        [0.0,  2.0, -3.0],
        [-2.0, 0.0,  4.0],
        [3.0, -4.0,  0.0]
    ], order='F', dtype=float)

    norm_m = ma02md('M', 'U', a)
    norm_1 = ma02md('1', 'U', a)
    norm_f = ma02md('F', 'U', a)

    np.testing.assert_allclose(norm_m, 4.0, rtol=1e-14)
    np.testing.assert_allclose(norm_1, 7.0, rtol=1e-14)
    np.testing.assert_allclose(norm_f, np.sqrt(58.0), rtol=1e-14)
