"""
Tests for MA02PD - Count zero rows and zero columns of a real matrix

MA02PD computes the number of zero rows and zero columns of a real M-by-N matrix.

Property tests verify:
- Zero matrices have all rows/cols as zero
- Identity matrix has no zero rows/cols
- Single zero row/col detection
- Edge cases (empty, single element)
"""
import numpy as np
import pytest
from ctrlsys import ma02pd


def test_ma02pd_basic():
    """
    Test basic functionality with matrix containing zero rows and columns.

    Matrix:
    [[1.0, 0.0, 2.0],
     [0.0, 0.0, 0.0],   <- zero row
     [3.0, 0.0, 4.0]]
           ^
           zero column
    """
    a = np.array([[1.0, 0.0, 2.0],
                  [0.0, 0.0, 0.0],
                  [3.0, 0.0, 4.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 1
    assert nzc == 1


def test_ma02pd_zero_matrix():
    """
    Test zero matrix - all rows and all columns are zero.

    A 3x4 zero matrix has 3 zero rows and 4 zero columns.
    """
    a = np.zeros((3, 4), order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 3
    assert nzc == 4


def test_ma02pd_identity():
    """
    Test identity matrix - no zero rows or columns.
    """
    a = np.eye(4, order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pd_full_nonzero():
    """
    Test matrix with no zero rows or columns.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    a = np.random.randn(5, 3) + 0.1
    a = np.asfortranarray(a)

    nzr, nzc = ma02pd(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pd_multiple_zero_rows():
    """
    Test matrix with multiple zero rows.

    Matrix:
    [[1.0, 2.0],
     [0.0, 0.0],   <- zero row
     [0.0, 0.0],   <- zero row
     [3.0, 4.0]]
    """
    a = np.array([[1.0, 2.0],
                  [0.0, 0.0],
                  [0.0, 0.0],
                  [3.0, 4.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 2
    assert nzc == 0


def test_ma02pd_multiple_zero_cols():
    """
    Test matrix with multiple zero columns.

    Matrix:
    [[0.0, 1.0, 0.0, 2.0],
     [0.0, 3.0, 0.0, 4.0]]
       ^        ^
       zero cols
    """
    a = np.array([[0.0, 1.0, 0.0, 2.0],
                  [0.0, 3.0, 0.0, 4.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 0
    assert nzc == 2


def test_ma02pd_single_element_zero():
    """
    Test single element zero matrix.
    """
    a = np.array([[0.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 1
    assert nzc == 1


def test_ma02pd_single_element_nonzero():
    """
    Test single element nonzero matrix.
    """
    a = np.array([[5.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pd_row_vector():
    """
    Test row vector (1 x n matrix).

    Vector: [[0.0, 1.0, 0.0, 2.0]]
              ^        ^
              zero cols

    The single row is not zero (contains nonzero elements).
    """
    a = np.array([[0.0, 1.0, 0.0, 2.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 0
    assert nzc == 2


def test_ma02pd_zero_row_vector():
    """
    Test zero row vector.
    """
    a = np.array([[0.0, 0.0, 0.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 1
    assert nzc == 3


def test_ma02pd_col_vector():
    """
    Test column vector (m x 1 matrix).

    Vector: [[0.0],
             [1.0],
             [0.0],
             [2.0]]

    The single column is not zero (contains nonzero elements).
    Zero rows: 2 (rows 0 and 2)
    """
    a = np.array([[0.0],
                  [1.0],
                  [0.0],
                  [2.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 2
    assert nzc == 0


def test_ma02pd_zero_col_vector():
    """
    Test zero column vector.
    """
    a = np.array([[0.0],
                  [0.0],
                  [0.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 3
    assert nzc == 1


def test_ma02pd_rectangular_tall():
    """
    Test tall rectangular matrix (m > n).

    Matrix 5x2 with 1 zero row and 1 zero col.
    """
    a = np.array([[1.0, 0.0],
                  [2.0, 0.0],
                  [0.0, 0.0],
                  [3.0, 0.0],
                  [4.0, 0.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 1
    assert nzc == 1


def test_ma02pd_rectangular_wide():
    """
    Test wide rectangular matrix (m < n).

    Matrix 2x5 with 1 zero row and 2 zero cols.
    """
    a = np.array([[1.0, 0.0, 2.0, 0.0, 3.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 1
    assert nzc == 2


def test_ma02pd_large_matrix():
    """
    Test larger matrix with known structure.

    Random seed: 123 (for reproducibility)

    Creates a 20x15 matrix and sets specific rows/cols to zero.
    """
    np.random.seed(123)
    a = np.random.randn(20, 15)

    a[3, :] = 0.0
    a[7, :] = 0.0
    a[15, :] = 0.0

    a[:, 2] = 0.0
    a[:, 9] = 0.0

    a = np.asfortranarray(a)

    nzr, nzc = ma02pd(a)

    assert nzr == 3
    assert nzc == 2


def test_ma02pd_near_zero_values():
    """
    Test that near-zero values are NOT counted as zero.

    Only exact zeros are counted.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    a = np.array([[1e-100, 1.0],
                  [1e-200, 2.0],
                  [1e-300, 3.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pd_mixed_zeros_near_zeros():
    """
    Test matrix with both exact zeros and near-zeros.

    Row 1 has exact zeros -> zero row
    Column 1 has near-zeros -> NOT zero column
    """
    a = np.array([[1e-15, 1.0, 2.0],
                  [0.0, 0.0, 0.0],
                  [1e-15, 3.0, 4.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 1
    assert nzc == 0


def test_ma02pd_diagonal_zeros():
    """
    Test matrix with zeros only on diagonal.

    No zero rows or columns since off-diagonal elements are nonzero.
    """
    a = np.array([[0.0, 1.0, 2.0],
                  [3.0, 0.0, 4.0],
                  [5.0, 6.0, 0.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 0
    assert nzc == 0


def test_ma02pd_first_row_zero():
    """
    Test matrix with only first row zero.
    """
    a = np.array([[0.0, 0.0, 0.0],
                  [1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 1
    assert nzc == 0


def test_ma02pd_last_row_zero():
    """
    Test matrix with only last row zero.
    """
    a = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [0.0, 0.0, 0.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 1
    assert nzc == 0


def test_ma02pd_first_col_zero():
    """
    Test matrix with only first column zero.
    """
    a = np.array([[0.0, 1.0, 2.0],
                  [0.0, 3.0, 4.0],
                  [0.0, 5.0, 6.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 0
    assert nzc == 1


def test_ma02pd_last_col_zero():
    """
    Test matrix with only last column zero.
    """
    a = np.array([[1.0, 2.0, 0.0],
                  [3.0, 4.0, 0.0],
                  [5.0, 6.0, 0.0]], order='F')

    nzr, nzc = ma02pd(a)

    assert nzr == 0
    assert nzc == 1
