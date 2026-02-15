"""
Tests for UD01DD: Reading/constructing a sparse matrix.

UD01DD reads the elements of a sparse matrix. The original Fortran routine
reads from a file, but the C implementation accepts sparse COO format data
directly (row indices, column indices, values).

The routine initializes the matrix to zero, then assigns nonzero elements
at specified positions.
"""

import numpy as np
import pytest


def test_ud01dd_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    From HTML doc:
    - M=6, N=5 (6x5 sparse matrix)
    - Diagonal elements at (1,1) to (5,5) = -1.1, -2.2, -3.3, -4.4, -5.5
    - Row 6 elements: 1.5, 2.5, 3.5, 4.5, 5.5
    """
    from ctrlsys import ud01dd

    m, n = 6, 5

    # Sparse data from HTML example (1-based indices as in Fortran)
    rows = np.array([1, 6, 2, 6, 3, 6, 4, 6, 5, 6], dtype=np.int32)
    cols = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int32)
    vals = np.array([-1.1, 1.5, -2.2, 2.5, -3.3, 3.5, -4.4, 4.5, -5.5, 5.5])

    a, info = ud01dd(m, n, rows, cols, vals)

    assert info == 0

    # Expected matrix from HTML results
    expected = np.array([
        [-1.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, -2.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, -3.3, 0.0, 0.0],
        [0.0, 0.0, 0.0, -4.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, -5.5],
        [1.5, 2.5, 3.5, 4.5, 5.5]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a, expected, rtol=1e-14)


def test_ud01dd_empty_sparse():
    """
    Test edge case: no nonzero elements (all zeros).
    """
    from ctrlsys import ud01dd

    m, n = 3, 4

    # No sparse entries
    rows = np.array([], dtype=np.int32)
    cols = np.array([], dtype=np.int32)
    vals = np.array([], dtype=float)

    a, info = ud01dd(m, n, rows, cols, vals)

    assert info == 0
    assert a.shape == (m, n)
    np.testing.assert_allclose(a, np.zeros((m, n), order='F'), rtol=1e-14)


def test_ud01dd_zero_dimensions():
    """
    Test edge case: M=0 or N=0 (quick return).
    """
    from ctrlsys import ud01dd

    rows = np.array([], dtype=np.int32)
    cols = np.array([], dtype=np.int32)
    vals = np.array([], dtype=float)

    # M=0
    a, info = ud01dd(0, 5, rows, cols, vals)
    assert info == 0
    assert a.shape == (0, 5)

    # N=0
    a, info = ud01dd(4, 0, rows, cols, vals)
    assert info == 0
    assert a.shape == (4, 0)


def test_ud01dd_out_of_bounds_warning():
    """
    Test warning case: index out of bounds returns info=1.

    From HTML doc: INFO=1 if i<1, i>M, j<1, or j>N.
    """
    from ctrlsys import ud01dd

    m, n = 3, 3

    # Valid entries
    rows = np.array([1, 2, 10], dtype=np.int32)  # 10 > m, out of bounds
    cols = np.array([1, 2, 1], dtype=np.int32)
    vals = np.array([1.0, 2.0, 3.0])

    a, info = ud01dd(m, n, rows, cols, vals)

    assert info == 1  # Warning for out-of-bounds

    # Valid entries should still be assigned
    expected = np.zeros((m, n), order='F')
    expected[0, 0] = 1.0
    expected[1, 1] = 2.0
    np.testing.assert_allclose(a, expected, rtol=1e-14)


def test_ud01dd_error_m_negative():
    """
    Test error: M < 0.
    """
    from ctrlsys import ud01dd

    rows = np.array([1], dtype=np.int32)
    cols = np.array([1], dtype=np.int32)
    vals = np.array([1.0])

    a, info = ud01dd(-1, 3, rows, cols, vals)

    assert info == -1


def test_ud01dd_error_n_negative():
    """
    Test error: N < 0.
    """
    from ctrlsys import ud01dd

    rows = np.array([1], dtype=np.int32)
    cols = np.array([1], dtype=np.int32)
    vals = np.array([1.0])

    a, info = ud01dd(3, -1, rows, cols, vals)

    assert info == -2


def test_ud01dd_single_element():
    """
    Test edge case: single nonzero element.
    """
    from ctrlsys import ud01dd

    m, n = 5, 4

    rows = np.array([3], dtype=np.int32)  # 1-based
    cols = np.array([2], dtype=np.int32)
    vals = np.array([7.5])

    a, info = ud01dd(m, n, rows, cols, vals)

    assert info == 0
    assert a.shape == (m, n)

    expected = np.zeros((m, n), order='F')
    expected[2, 1] = 7.5  # Convert to 0-based
    np.testing.assert_allclose(a, expected, rtol=1e-14)


def test_ud01dd_overwrite_same_position():
    """
    Test behavior: multiple entries at same position (last one wins).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ud01dd

    m, n = 3, 3

    # Multiple entries at (2,2) - 1-based
    rows = np.array([2, 2, 2], dtype=np.int32)
    cols = np.array([2, 2, 2], dtype=np.int32)
    vals = np.array([1.0, 5.0, 9.0])  # Last value (9.0) should win

    a, info = ud01dd(m, n, rows, cols, vals)

    assert info == 0
    assert a[1, 1] == 9.0  # 0-based index


def test_ud01dd_full_dense():
    """
    Test: sparse format representing a full dense matrix.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ud01dd

    np.random.seed(123)

    m, n = 4, 3

    # Create all entries
    rows_list = []
    cols_list = []
    vals_list = []

    expected = np.random.randn(m, n).astype(float, order='F')

    for i in range(m):
        for j in range(n):
            rows_list.append(i + 1)  # 1-based
            cols_list.append(j + 1)
            vals_list.append(expected[i, j])

    rows = np.array(rows_list, dtype=np.int32)
    cols = np.array(cols_list, dtype=np.int32)
    vals = np.array(vals_list)

    a, info = ud01dd(m, n, rows, cols, vals)

    assert info == 0
    np.testing.assert_allclose(a, expected, rtol=1e-14)


def test_ud01dd_boundary_indices():
    """
    Test boundary indices (corners of matrix).
    """
    from ctrlsys import ud01dd

    m, n = 5, 4

    # Four corners (1-based)
    rows = np.array([1, 1, m, m], dtype=np.int32)
    cols = np.array([1, n, 1, n], dtype=np.int32)
    vals = np.array([1.0, 2.0, 3.0, 4.0])

    a, info = ud01dd(m, n, rows, cols, vals)

    assert info == 0

    expected = np.zeros((m, n), order='F')
    expected[0, 0] = 1.0
    expected[0, n-1] = 2.0
    expected[m-1, 0] = 3.0
    expected[m-1, n-1] = 4.0

    np.testing.assert_allclose(a, expected, rtol=1e-14)


def test_ud01dd_column_major_storage():
    """
    Mathematical property: output array uses column-major (Fortran) storage.

    Verify memory layout is column-major by checking strides.
    """
    from ctrlsys import ud01dd

    m, n = 4, 3

    rows = np.array([1, 2], dtype=np.int32)
    cols = np.array([1, 2], dtype=np.int32)
    vals = np.array([1.0, 2.0])

    a, info = ud01dd(m, n, rows, cols, vals)

    assert info == 0
    assert a.flags['F_CONTIGUOUS'], "Output should be Fortran (column-major) contiguous"
