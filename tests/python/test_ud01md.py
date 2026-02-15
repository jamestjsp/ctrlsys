"""
Tests for UD01MD: Printing a real matrix.

UD01MD prints an M-by-N real matrix A row by row to a string with
7 significant figures. In Python, returns a formatted string instead
of writing to a file.

Format:
- Title line: "Matrix A (MxN)"
- Column headers every L columns
- Values in scientific notation with 7 significant figures
"""

import numpy as np
import pytest


def test_ud01md_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    From HTML doc:
    - M=4, N=4, L=4 (4x4 matrix, 4 elements per line)
    - Matrix values 1-16
    """
    from ctrlsys import ud01md

    m, n, l = 4, 4, 4

    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], order='F', dtype=float)

    text = 'Matrix A'

    result, info = ud01md(m, n, l, a, text)

    assert info == 0
    assert isinstance(result, str)

    assert 'Matrix A' in result
    assert '4' in result

    lines = result.strip().split('\n')
    assert len(lines) >= m + 2


def test_ud01md_small_l():
    """
    Test with L=2 (2 elements per line, requiring column blocks).

    This tests the column blocking feature where N > L.
    """
    from ctrlsys import ud01md

    m, n, l = 2, 4, 2

    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0]
    ], order='F', dtype=float)

    text = 'Test Matrix'

    result, info = ud01md(m, n, l, a, text)

    assert info == 0
    assert 'Test Matrix' in result


def test_ud01md_single_column():
    """
    Test edge case: single column matrix (N=1).
    """
    from ctrlsys import ud01md

    m, n, l = 3, 1, 1

    a = np.array([
        [1.0],
        [2.0],
        [3.0]
    ], order='F', dtype=float)

    text = 'Column Vector'

    result, info = ud01md(m, n, l, a, text)

    assert info == 0
    assert 'Column Vector' in result
    assert '3' in result
    assert '1' in result


def test_ud01md_single_row():
    """
    Test edge case: single row matrix (M=1).
    """
    from ctrlsys import ud01md

    m, n, l = 1, 4, 4

    a = np.array([
        [1.0, 2.0, 3.0, 4.0]
    ], order='F', dtype=float)

    text = 'Row Vector'

    result, info = ud01md(m, n, l, a, text)

    assert info == 0
    assert 'Row Vector' in result


def test_ud01md_l_equals_5():
    """
    Test with maximum allowed L=5.
    """
    from ctrlsys import ud01md

    m, n, l = 2, 5, 5

    a = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0]
    ], order='F', dtype=float)

    text = 'Max L Matrix'

    result, info = ud01md(m, n, l, a, text)

    assert info == 0
    assert 'Max L Matrix' in result


def test_ud01md_scientific_notation():
    """
    Test that values are formatted in scientific notation with 7 figures.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ud01md

    np.random.seed(42)

    m, n, l = 2, 2, 2

    a = np.array([
        [1.234567e5, 9.876543e-3],
        [5.555555e10, 1.111111e-8]
    ], order='F', dtype=float)

    text = 'Scientific'

    result, info = ud01md(m, n, l, a, text)

    assert info == 0
    assert 'D' in result or 'E' in result or 'e' in result


def test_ud01md_error_m_invalid():
    """
    Test error: M < 1.
    """
    from ctrlsys import ud01md

    m, n, l = 0, 4, 4

    a = np.array([[1.0, 2.0, 3.0, 4.0]], order='F', dtype=float)

    text = 'Test'

    result, info = ud01md(m, n, l, a, text)

    assert info == -1


def test_ud01md_error_n_invalid():
    """
    Test error: N < 1.
    """
    from ctrlsys import ud01md

    m, n, l = 4, 0, 4

    a = np.array([[1.0], [2.0], [3.0], [4.0]], order='F', dtype=float)

    text = 'Test'

    result, info = ud01md(m, n, l, a, text)

    assert info == -2


def test_ud01md_error_l_too_small():
    """
    Test error: L < 1.
    """
    from ctrlsys import ud01md

    m, n, l = 4, 4, 0

    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], order='F', dtype=float)

    text = 'Test'

    result, info = ud01md(m, n, l, a, text)

    assert info == -3


def test_ud01md_error_l_too_large():
    """
    Test error: L > 5.
    """
    from ctrlsys import ud01md

    m, n, l = 4, 4, 6

    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], order='F', dtype=float)

    text = 'Test'

    result, info = ud01md(m, n, l, a, text)

    assert info == -3


def test_ud01md_long_text():
    """
    Test with long text title (truncated to 72 chars).
    """
    from ctrlsys import ud01md

    m, n, l = 2, 2, 2

    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    text = 'A' * 100

    result, info = ud01md(m, n, l, a, text)

    assert info == 0


def test_ud01md_empty_text():
    """
    Test with empty text title.
    """
    from ctrlsys import ud01md

    m, n, l = 2, 2, 2

    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    text = ''

    result, info = ud01md(m, n, l, a, text)

    assert info == 0
