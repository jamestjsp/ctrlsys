"""
Tests for MA02MZ - Compute norms of a complex skew-Hermitian matrix.

MA02MZ computes the value of the one norm, or the Frobenius norm, or
the infinity norm, or the element of largest absolute value
of a complex skew-Hermitian matrix.

Note that for skew-Hermitian matrices:
- Diagonal elements are pure imaginary (real part = 0)
- a[j,i] = -conj(a[i,j]) for off-diagonal elements
- The infinity norm equals the one norm
"""
import numpy as np
import pytest
from ctrlsys import ma02mz


def test_ma02mz_max_norm_upper():
    """
    Test max norm (M) with upper triangular storage.

    For a skew-Hermitian matrix, max norm is max(|a_ij|).
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    a = np.zeros((n, n), order='F', dtype=complex)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn() + 1j * np.random.randn()
            a[j, i] = -np.conj(a[i, j])
    for i in range(n):
        a[i, i] = 1j * np.random.randn()

    expected = np.max(np.abs(a))

    norm = ma02mz('M', 'U', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02mz_max_norm_lower():
    """
    Test max norm (M) with lower triangular storage.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 5

    a = np.zeros((n, n), order='F', dtype=complex)
    for j in range(n - 1):
        for i in range(j + 1, n):
            a[i, j] = np.random.randn() + 1j * np.random.randn()
            a[j, i] = -np.conj(a[i, j])
    for i in range(n):
        a[i, i] = 1j * np.random.randn()

    expected = np.max(np.abs(a))

    norm = ma02mz('M', 'L', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02mz_one_norm_upper():
    """
    Test one norm (1/O) with upper triangular storage.

    The one norm of a matrix is the maximum column sum of absolute values.
    For skew-Hermitian matrices, this equals the infinity norm.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4

    a = np.zeros((n, n), order='F', dtype=complex)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn() + 1j * np.random.randn()
            a[j, i] = -np.conj(a[i, j])
    for i in range(n):
        a[i, i] = 1j * np.random.randn()

    expected = np.linalg.norm(a, 1)

    norm = ma02mz('1', 'U', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02mz_one_norm_lower():
    """
    Test one norm with lower triangular storage.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5

    a = np.zeros((n, n), order='F', dtype=complex)
    for j in range(n - 1):
        for i in range(j + 1, n):
            a[i, j] = np.random.randn() + 1j * np.random.randn()
            a[j, i] = -np.conj(a[i, j])
    for i in range(n):
        a[i, i] = 1j * np.random.randn()

    expected = np.linalg.norm(a, 1)

    norm = ma02mz('O', 'L', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02mz_infinity_norm():
    """
    Test infinity norm (I).

    For skew-Hermitian matrices, infinity norm equals one norm.
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 4

    a = np.zeros((n, n), order='F', dtype=complex)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn() + 1j * np.random.randn()
            a[j, i] = -np.conj(a[i, j])
    for i in range(n):
        a[i, i] = 1j * np.random.randn()

    expected = np.linalg.norm(a, np.inf)

    norm = ma02mz('I', 'U', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-14)


def test_ma02mz_frobenius_norm_upper():
    """
    Test Frobenius norm (F/E) with upper triangular storage.

    Frobenius norm is sqrt(sum of squares of all elements).
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 4

    a = np.zeros((n, n), order='F', dtype=complex)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn() + 1j * np.random.randn()
            a[j, i] = -np.conj(a[i, j])
    for i in range(n):
        a[i, i] = 1j * np.random.randn()

    expected = np.linalg.norm(a, 'fro')

    norm = ma02mz('F', 'U', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-13)


def test_ma02mz_frobenius_norm_lower():
    """
    Test Frobenius norm with lower triangular storage.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 5

    a = np.zeros((n, n), order='F', dtype=complex)
    for j in range(n - 1):
        for i in range(j + 1, n):
            a[i, j] = np.random.randn() + 1j * np.random.randn()
            a[j, i] = -np.conj(a[i, j])
    for i in range(n):
        a[i, i] = 1j * np.random.randn()

    expected = np.linalg.norm(a, 'fro')

    norm = ma02mz('E', 'L', a)

    np.testing.assert_allclose(norm, expected, rtol=1e-13)


def test_ma02mz_n_zero():
    """
    Edge case: n = 0 should return 0.
    """
    a = np.array([], order='F', dtype=complex).reshape(0, 0)

    norm = ma02mz('F', 'U', a)

    assert norm == 0.0


def test_ma02mz_skew_hermitian_property():
    """
    Mathematical property: verify norm equality for skew-Hermitian matrices.

    For skew-Hermitian matrices:
    - ||A||_1 = ||A||_inf (one norm equals infinity norm)

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 6

    a = np.zeros((n, n), order='F', dtype=complex)
    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn() + 1j * np.random.randn()
            a[j, i] = -np.conj(a[i, j])
    for i in range(n):
        a[i, i] = 1j * np.random.randn()

    norm_1 = ma02mz('1', 'U', a)
    norm_inf = ma02mz('I', 'U', a)

    np.testing.assert_allclose(norm_1, norm_inf, rtol=1e-14)


def test_ma02mz_simple_known_matrix():
    """
    Test with a simple known skew-Hermitian matrix.

    A = [  2i,       1+i,    -2+3i]
        [ -1+i,      -i,      4-i ]
        [  2+3i,   -4-i,      3i  ]

    Verify A is skew-Hermitian: A^H = -A
    - a[0,0] = 2i, -conj(a[0,0]) = -(-2i) = 2i, pure imaginary: OK
    - a[0,1] = 1+i, a[1,0] = -1+i = -(1-i) = -conj(1+i): OK
    - a[0,2] = -2+3i, a[2,0] = 2+3i = -(-2-3i) = -conj(-2+3i): OK
    - etc.

    Max norm: max(|2i|, |1+i|, |-2+3i|, |-i|, |4-i|, |3i|)
            = max(2, sqrt(2), sqrt(13), 1, sqrt(17), 3)
            = sqrt(17) approx 4.123

    One norm: max of column sums of absolute values
    """
    a = np.array([
        [2j,        1+1j,   -2+3j],
        [-1+1j,     -1j,     4-1j],
        [2+3j,      -4-1j,   3j]
    ], order='F', dtype=complex)

    np.testing.assert_allclose(a + a.conj().T, 0, atol=1e-14)

    norm_m = ma02mz('M', 'U', a)
    norm_1 = ma02mz('1', 'U', a)
    norm_f = ma02mz('F', 'U', a)

    expected_max = np.max(np.abs(a))
    expected_1 = np.linalg.norm(a, 1)
    expected_f = np.linalg.norm(a, 'fro')

    np.testing.assert_allclose(norm_m, expected_max, rtol=1e-14)
    np.testing.assert_allclose(norm_1, expected_1, rtol=1e-14)
    np.testing.assert_allclose(norm_f, expected_f, rtol=1e-13)


def test_ma02mz_diagonal_imaginary_only():
    """
    Test that diagonal contribution comes only from imaginary parts.

    For skew-Hermitian matrices, diagonal elements are pure imaginary.
    The routine assumes real parts of diagonal elements are zero.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n = 3

    a = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        a[i, i] = 1j * np.random.randn()

    for j in range(1, n):
        for i in range(j):
            a[i, j] = np.random.randn() + 1j * np.random.randn()
            a[j, i] = -np.conj(a[i, j])

    norm_m = ma02mz('M', 'U', a)
    expected = np.max(np.abs(a))

    np.testing.assert_allclose(norm_m, expected, rtol=1e-14)
