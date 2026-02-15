"""
Tests for MA02SD: Compute smallest nonzero absolute value of matrix elements

Random seeds used for reproducibility:
- test_ma02sd_basic: N/A (fixed data)
- test_ma02sd_random: 42
- test_ma02sd_single_element: N/A
- test_ma02sd_all_zeros: N/A
- test_ma02sd_property: 123
"""

import numpy as np
import pytest


def test_ma02sd_basic():
    """
    Test MA02SD with simple known values.

    Matrix: [[1, 2, 3], [4, 5, 6]]
    Smallest nonzero absolute value = 1.0
    """
    from ctrlsys import ma02sd

    a = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], order='F', dtype=float)

    result = ma02sd(a)

    assert result == 1.0


def test_ma02sd_with_zeros():
    """
    Test MA02SD ignoring zero elements.

    Matrix: [[0, 0, 3], [0, 5, 0]]
    Smallest nonzero absolute value = 3.0
    """
    from ctrlsys import ma02sd

    a = np.array([[0.0, 0.0, 3.0],
                  [0.0, 5.0, 0.0]], order='F', dtype=float)

    result = ma02sd(a)

    assert result == 3.0


def test_ma02sd_with_negative():
    """
    Test MA02SD with negative values (uses absolute value).

    Matrix: [[-0.5, 2], [3, -4]]
    Smallest nonzero absolute value = 0.5
    """
    from ctrlsys import ma02sd

    a = np.array([[-0.5, 2.0],
                  [3.0, -4.0]], order='F', dtype=float)

    result = ma02sd(a)

    assert result == 0.5


def test_ma02sd_random():
    """
    Test MA02SD with random matrix.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n = 5, 4

    a = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import ma02sd
    result = ma02sd(a)

    expected = np.min(np.abs(a[a != 0]))
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_ma02sd_single_element():
    """
    Test MA02SD with 1x1 matrix.
    """
    from ctrlsys import ma02sd

    a = np.array([[7.5]], order='F', dtype=float)

    result = ma02sd(a)

    assert result == 7.5


def test_ma02sd_single_zero():
    """
    Test MA02SD with 1x1 zero matrix.

    Returns overflow value (very large) since no nonzero elements.
    """
    from ctrlsys import ma02sd

    a = np.array([[0.0]], order='F', dtype=float)

    result = ma02sd(a)

    assert result > 1e100


def test_ma02sd_all_zeros():
    """
    Test MA02SD with all-zero matrix.

    Returns overflow value (very large) since no nonzero elements.
    """
    from ctrlsys import ma02sd

    a = np.zeros((3, 3), order='F', dtype=float)

    result = ma02sd(a)

    assert result > 1e100


def test_ma02sd_empty_m_zero():
    """
    Test MA02SD with M=0 (empty matrix).

    Returns 0.0 per Fortran spec.
    """
    from ctrlsys import ma02sd

    a = np.array([], dtype=float, order='F').reshape(0, 3)

    result = ma02sd(a)

    assert result == 0.0


def test_ma02sd_empty_n_zero():
    """
    Test MA02SD with N=0 (empty matrix).

    Returns 0.0 per Fortran spec.
    """
    from ctrlsys import ma02sd

    a = np.array([], dtype=float, order='F').reshape(3, 0)

    result = ma02sd(a)

    assert result == 0.0


def test_ma02sd_property_minimum():
    """
    Property test: result <= all nonzero absolute values.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, n = 10, 8

    a = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import ma02sd
    result = ma02sd(a)

    nonzero_abs = np.abs(a[a != 0])
    assert np.all(result <= nonzero_abs + 1e-14)


def test_ma02sd_column_vector():
    """
    Test MA02SD with column vector (N=1).
    """
    from ctrlsys import ma02sd

    a = np.array([[1.0], [0.5], [2.0]], order='F', dtype=float)

    result = ma02sd(a)

    assert result == 0.5


def test_ma02sd_row_vector():
    """
    Test MA02SD with row vector (M=1).
    """
    from ctrlsys import ma02sd

    a = np.array([[1.0, 0.5, 2.0]], order='F', dtype=float)

    result = ma02sd(a)

    assert result == 0.5
