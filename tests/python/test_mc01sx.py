"""
Tests for MC01SX - Compute variation of exponents in floating-point series.

Computes V = max(E(j)) - min(E(j)) for j where MANT(j) != 0.
This is used internally by SLICOT for polynomial scaling operations.
"""

import numpy as np
import pytest


def test_mc01sx_basic():
    """
    Test basic exponent variation calculation.

    Exponents: [5, 10, 3, 8]
    Mantissas: [1.0, 2.0, 3.0, 4.0] (all non-zero)
    Expected variation: 10 - 3 = 7
    """
    from ctrlsys import mc01sx

    e = np.array([5, 10, 3, 8], dtype=np.int32)
    mant = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    v = mc01sx(e, mant)

    assert v == 7


def test_mc01sx_with_zero_mantissa():
    """
    Test that zero mantissas are ignored.

    Exponents: [5, 100, 3, 8]  (100 should be ignored)
    Mantissas: [1.0, 0.0, 3.0, 4.0]
    Expected variation: 8 - 3 = 5 (ignoring the 100 at index 1)
    """
    from ctrlsys import mc01sx

    e = np.array([5, 100, 3, 8], dtype=np.int32)
    mant = np.array([1.0, 0.0, 3.0, 4.0], dtype=np.float64)

    v = mc01sx(e, mant)

    assert v == 5


def test_mc01sx_single_element():
    """
    Test with single element.

    Variation should be 0.
    """
    from ctrlsys import mc01sx

    e = np.array([42], dtype=np.int32)
    mant = np.array([1.5], dtype=np.float64)

    v = mc01sx(e, mant)

    assert v == 0


def test_mc01sx_all_same_exponent():
    """
    Test when all non-zero entries have the same exponent.

    Variation should be 0.
    """
    from ctrlsys import mc01sx

    e = np.array([5, 5, 5, 5], dtype=np.int32)
    mant = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    v = mc01sx(e, mant)

    assert v == 0


def test_mc01sx_negative_exponents():
    """
    Test with negative exponents.

    Exponents: [-10, -5, -15, -3]
    Mantissas: [1.0, 1.0, 1.0, 1.0]
    Expected variation: -3 - (-15) = 12
    """
    from ctrlsys import mc01sx

    e = np.array([-10, -5, -15, -3], dtype=np.int32)
    mant = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    v = mc01sx(e, mant)

    assert v == 12


def test_mc01sx_mixed_sign_exponents():
    """
    Test with mixed positive and negative exponents.

    Exponents: [-5, 0, 10, -10]
    Mantissas: [1.0, 1.0, 1.0, 1.0]
    Expected variation: 10 - (-10) = 20
    """
    from ctrlsys import mc01sx

    e = np.array([-5, 0, 10, -10], dtype=np.int32)
    mant = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    v = mc01sx(e, mant)

    assert v == 20


def test_mc01sx_first_only_nonzero():
    """
    Test when only first mantissa is non-zero.

    Variation should be 0.
    """
    from ctrlsys import mc01sx

    e = np.array([5, 100, 200, -50], dtype=np.int32)
    mant = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    v = mc01sx(e, mant)

    assert v == 0


def test_mc01sx_matches_numpy():
    from ctrlsys import mc01sx

    e = np.array([3, -7, 15, 0, -2, 8], dtype=np.int32)
    mant = np.array([1.5, 0.0, 2.3, 0.0, -0.5, 1.0], dtype=np.float64)

    v = mc01sx(e, mant)

    active = e[mant != 0.0]
    expected = int(active.max() - active.min())
    assert v == expected
