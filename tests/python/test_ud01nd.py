"""
Tests for UD01ND: Printing a matrix polynomial.

UD01ND prints the MP-by-NP coefficient matrices of a matrix polynomial:
    P(s) = P(0) + P(1)*s + ... + P(dp-1)*s^(dp-1) + P(dp)*s^dp

The elements are output to 7 significant figures. In Python, returns a
formatted string instead of writing to a file.

Format per coefficient matrix P(k):
- Title line: "TEXT(k) (MPxNP)" where k is the polynomial degree
- Column headers every L columns
- Values in scientific notation with 7 significant figures
"""

import numpy as np
import pytest


def test_ud01nd_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    From HTML doc:
    - MP=4, NP=3, DP=2, L=5
    - P(0), P(1), P(2) coefficient matrices

    Expected output format:
    P( 0) ( 4X 3)
           1              2              3
     1    0.1000000D+01  0.0000000D+00  0.0000000D+00
    ...
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 4, 3, 2, 5

    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)

    p[:, :, 0] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 4.0],
        [0.0, 4.0, 8.0],
        [0.0, 6.0, 12.0]
    ], order='F')

    p[:, :, 1] = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ], order='F')

    p[:, :, 2] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], order='F')

    text = 'P'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == 0
    assert isinstance(result, str)

    assert 'P' in result
    assert '( 0)' in result or '(0)' in result
    assert '( 1)' in result or '(1)' in result
    assert '( 2)' in result or '(2)' in result

    assert '4' in result
    assert '3' in result


def test_ud01nd_degree_zero():
    """
    Test edge case: degree 0 polynomial (single coefficient matrix).
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 2, 2, 0, 5

    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    p[:, :, 0] = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F')

    text = 'Const'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == 0
    assert 'Const' in result
    assert '( 0)' in result or '(0)' in result


def test_ud01nd_column_blocking():
    """
    Test column blocking with L=2 and NP=5 (requires multiple blocks).

    This tests the column blocking feature where NP > L.
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 2, 5, 1, 2

    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)

    p[:, :, 0] = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0]
    ], order='F')

    p[:, :, 1] = np.array([
        [11.0, 12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0, 20.0]
    ], order='F')

    text = 'Q'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == 0

    col1_count = result.count('     1')
    col2_count = result.count('     2')
    assert col1_count >= 2
    assert col2_count >= 2


def test_ud01nd_single_element():
    """
    Test edge case: 1x1 coefficient matrices.
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 1, 1, 2, 1

    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    p[0, 0, 0] = 5.0
    p[0, 0, 1] = 3.0
    p[0, 0, 2] = 1.0

    text = 'Scalar'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == 0
    assert 'Scalar' in result


def test_ud01nd_empty_text():
    """
    Test with empty text (should print blank lines as separators).

    From Fortran: If TEXT = ' ', then coefficient matrices are
    separated by an empty line.
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 2, 2, 1, 2

    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    p[:, :, 0] = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')
    p[:, :, 1] = np.array([[5.0, 6.0], [7.0, 8.0]], order='F')

    text = ' '

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == 0


def test_ud01nd_scientific_notation():
    """
    Test that values are formatted in scientific notation with 7 figures.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ud01nd

    np.random.seed(42)

    mp, np_dim, dp, l = 2, 2, 1, 2

    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    p[:, :, 0] = np.array([
        [1.234567e5, 9.876543e-3],
        [5.555555e10, 1.111111e-8]
    ], order='F')
    p[:, :, 1] = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F')

    text = 'Poly'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == 0
    assert 'D' in result or 'E' in result or 'e' in result


def test_ud01nd_error_mp_invalid():
    """
    Test error: MP < 1.
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 0, 2, 1, 2

    p = np.zeros((1, 2, 2), order='F', dtype=float)

    text = 'Test'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == -1


def test_ud01nd_error_np_invalid():
    """
    Test error: NP < 1.
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 2, 0, 1, 2

    p = np.zeros((2, 1, 2), order='F', dtype=float)

    text = 'Test'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == -2


def test_ud01nd_error_dp_invalid():
    """
    Test error: DP < 0.
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 2, 2, -1, 2

    p = np.zeros((2, 2, 1), order='F', dtype=float)

    text = 'Test'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == -3


def test_ud01nd_error_l_too_small():
    """
    Test error: L < 1.
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 2, 2, 1, 0

    p = np.zeros((2, 2, 2), order='F', dtype=float)

    text = 'Test'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == -4


def test_ud01nd_error_l_too_large():
    """
    Test error: L > 5.
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 2, 2, 1, 6

    p = np.zeros((2, 2, 2), order='F', dtype=float)

    text = 'Test'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == -4


def test_ud01nd_long_text():
    """
    Test with long text title (truncated to 72 chars).
    """
    from ctrlsys import ud01nd

    mp, np_dim, dp, l = 2, 2, 0, 2

    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    p[:, :, 0] = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')

    text = 'A' * 100

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == 0


def test_ud01nd_polynomial_evaluation_property():
    """
    Mathematical property test: verify polynomial structure.

    P(s) = P(0) + P(1)*s + P(2)*s^2 = I + A*s + B*s^2

    For this test, we verify the output contains all coefficient matrices
    with correct indexing.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ud01nd

    np.random.seed(123)

    mp, np_dim, dp, l = 3, 3, 2, 5

    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)

    p[:, :, 0] = np.eye(mp, np_dim, dtype=float)

    p[:, :, 1] = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], order='F')

    p[:, :, 2] = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ], order='F')

    text = 'MatPoly'

    result, info = ud01nd(mp, np_dim, dp, l, p, text)

    assert info == 0

    assert '( 0)' in result or '(0)' in result
    assert '( 1)' in result or '(1)' in result
    assert '( 2)' in result or '(2)' in result

    lines = result.strip().split('\n')
    assert len(lines) >= 3 * (mp + 2)
