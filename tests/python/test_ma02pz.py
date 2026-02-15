"""
Tests for MA02PZ - Count zero rows and zero columns of a complex matrix

MA02PZ computes the number of zero rows and zero columns of a complex M-by-N matrix.
This is the complex version of MA02PD.

Property tests verify:
- Zero matrices have all rows/cols as zero
- Identity matrix has no zero rows/cols
- Single zero row/col detection
- Edge cases (empty, single element)
- Complex values with both real and imaginary parts
"""
import numpy as np
import pytest
from ctrlsys import ma02pz


def test_ma02pz_basic():
    """
    Test basic functionality with complex matrix containing zero rows and columns.

    Matrix:
    [[1+1j, 0+0j, 2+2j],
     [0+0j, 0+0j, 0+0j],   <- zero row
     [3+3j, 0+0j, 4+4j]]
            ^
            zero column
    """
    a = np.array([[1+1j, 0+0j, 2+2j],
                  [0+0j, 0+0j, 0+0j],
                  [3+3j, 0+0j, 4+4j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 1
    assert nzc == 1


def test_ma02pz_zero_matrix():
    """
    Test zero matrix - all rows and all columns are zero.

    A 3x4 zero matrix has 3 zero rows and 4 zero columns.
    """
    a = np.zeros((3, 4), order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 3
    assert nzc == 4


def test_ma02pz_identity():
    """
    Test complex identity matrix - no zero rows or columns.
    """
    a = np.eye(4, order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pz_full_nonzero():
    """
    Test matrix with no zero rows or columns.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    a = np.random.randn(5, 3) + 1j * np.random.randn(5, 3) + 0.1
    a = np.asfortranarray(a)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pz_multiple_zero_rows():
    """
    Test matrix with multiple zero rows.

    Matrix:
    [[1+0j, 2+0j],
     [0+0j, 0+0j],   <- zero row
     [0+0j, 0+0j],   <- zero row
     [3+0j, 4+0j]]
    """
    a = np.array([[1+0j, 2+0j],
                  [0+0j, 0+0j],
                  [0+0j, 0+0j],
                  [3+0j, 4+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 2
    assert nzc == 0


def test_ma02pz_multiple_zero_cols():
    """
    Test matrix with multiple zero columns.

    Matrix:
    [[0+0j, 1+0j, 0+0j, 2+0j],
     [0+0j, 3+0j, 0+0j, 4+0j]]
       ^          ^
       zero cols
    """
    a = np.array([[0+0j, 1+0j, 0+0j, 2+0j],
                  [0+0j, 3+0j, 0+0j, 4+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 2


def test_ma02pz_single_element_zero():
    """
    Test single element zero matrix.
    """
    a = np.array([[0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 1
    assert nzc == 1


def test_ma02pz_single_element_nonzero():
    """
    Test single element nonzero matrix.
    """
    a = np.array([[5+3j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pz_imaginary_only():
    """
    Test matrix with pure imaginary values.

    Rows/cols with only imaginary parts are NOT zero.
    """
    a = np.array([[0+1j, 0+0j],
                  [0+2j, 0+0j],
                  [0+3j, 0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 1


def test_ma02pz_real_only():
    """
    Test matrix with pure real values (as complex).

    Rows/cols with only real parts are NOT zero.
    """
    a = np.array([[1+0j, 0+0j],
                  [2+0j, 0+0j],
                  [3+0j, 0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 1


def test_ma02pz_row_vector():
    """
    Test row vector (1 x n complex matrix).

    Vector: [[0+0j, 1+1j, 0+0j, 2+2j]]
              ^          ^
              zero cols

    The single row is not zero (contains nonzero elements).
    """
    a = np.array([[0+0j, 1+1j, 0+0j, 2+2j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 2


def test_ma02pz_zero_row_vector():
    """
    Test zero row vector.
    """
    a = np.array([[0+0j, 0+0j, 0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 1
    assert nzc == 3


def test_ma02pz_col_vector():
    """
    Test column vector (m x 1 complex matrix).

    Vector: [[0+0j],
             [1+1j],
             [0+0j],
             [2+2j]]

    The single column is not zero (contains nonzero elements).
    Zero rows: 2 (rows 0 and 2)
    """
    a = np.array([[0+0j],
                  [1+1j],
                  [0+0j],
                  [2+2j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 2
    assert nzc == 0


def test_ma02pz_zero_col_vector():
    """
    Test zero column vector.
    """
    a = np.array([[0+0j],
                  [0+0j],
                  [0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 3
    assert nzc == 1


def test_ma02pz_rectangular_tall():
    """
    Test tall rectangular complex matrix (m > n).

    Matrix 5x2 with 1 zero row and 1 zero col.
    """
    a = np.array([[1+1j, 0+0j],
                  [2+2j, 0+0j],
                  [0+0j, 0+0j],
                  [3+3j, 0+0j],
                  [4+4j, 0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 1
    assert nzc == 1


def test_ma02pz_rectangular_wide():
    """
    Test wide rectangular complex matrix (m < n).

    Matrix 2x5 with 1 zero row and 2 zero cols.
    """
    a = np.array([[1+0j, 0+0j, 2+0j, 0+0j, 3+0j],
                  [0+0j, 0+0j, 0+0j, 0+0j, 0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 1
    assert nzc == 2


def test_ma02pz_large_matrix():
    """
    Test larger complex matrix with known structure.

    Random seed: 123 (for reproducibility)

    Creates a 20x15 complex matrix and sets specific rows/cols to zero.
    """
    np.random.seed(123)
    a = np.random.randn(20, 15) + 1j * np.random.randn(20, 15)

    a[3, :] = 0.0
    a[7, :] = 0.0
    a[15, :] = 0.0

    a[:, 2] = 0.0
    a[:, 9] = 0.0

    a = np.asfortranarray(a)

    nzr, nzc = ma02pz(a)

    assert nzr == 3
    assert nzc == 2


def test_ma02pz_near_zero_values():
    """
    Test that near-zero complex values are NOT counted as zero.

    Only exact zeros are counted.
    """
    a = np.array([[1e-100+0j, 1+0j],
                  [0+1e-200j, 2+0j],
                  [1e-300+1e-300j, 3+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pz_mixed_zeros_near_zeros():
    """
    Test complex matrix with both exact zeros and near-zeros.

    Row 1 has exact zeros -> zero row
    Column 0 has near-zeros -> NOT zero column
    """
    a = np.array([[1e-15+0j, 1+1j, 2+2j],
                  [0+0j, 0+0j, 0+0j],
                  [0+1e-15j, 3+3j, 4+4j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 1
    assert nzc == 0


def test_ma02pz_diagonal_zeros():
    """
    Test complex matrix with zeros only on diagonal.

    No zero rows or columns since off-diagonal elements are nonzero.
    """
    a = np.array([[0+0j, 1+0j, 2+0j],
                  [3+0j, 0+0j, 4+0j],
                  [5+0j, 6+0j, 0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pz_first_row_zero():
    """
    Test complex matrix with only first row zero.
    """
    a = np.array([[0+0j, 0+0j, 0+0j],
                  [1+1j, 2+2j, 3+3j],
                  [4+4j, 5+5j, 6+6j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 1
    assert nzc == 0


def test_ma02pz_last_row_zero():
    """
    Test complex matrix with only last row zero.
    """
    a = np.array([[1+1j, 2+2j, 3+3j],
                  [4+4j, 5+5j, 6+6j],
                  [0+0j, 0+0j, 0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 1
    assert nzc == 0


def test_ma02pz_first_col_zero():
    """
    Test complex matrix with only first column zero.
    """
    a = np.array([[0+0j, 1+1j, 2+2j],
                  [0+0j, 3+3j, 4+4j],
                  [0+0j, 5+5j, 6+6j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 1


def test_ma02pz_last_col_zero():
    """
    Test complex matrix with only last column zero.
    """
    a = np.array([[1+1j, 2+2j, 0+0j],
                  [3+3j, 4+4j, 0+0j],
                  [5+5j, 6+6j, 0+0j]], order='F', dtype=complex)

    nzr, nzc = ma02pz(a)

    assert nzr == 0
    assert nzc == 1


def test_ma02pz_conjugate_symmetric():
    """
    Test Hermitian (conjugate-symmetric) matrix.

    Random seed: 789 (for reproducibility)

    Verify routine works with Hermitian structure.
    """
    np.random.seed(789)
    n = 4
    a = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    a = (a + a.conj().T) / 2
    a[:, 1] = 0.0
    a[1, :] = 0.0
    a = np.asfortranarray(a)

    nzr, nzc = ma02pz(a)

    assert nzr == 1
    assert nzc == 1
