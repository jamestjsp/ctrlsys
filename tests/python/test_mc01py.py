"""
Tests for MC01PY - Compute polynomial coefficients from zeros (decreasing order).

Computes coefficients of real polynomial P(x) = (x - r1)(x - r2)...(x - rk)
from given zeros. Coefficients stored in DECREASING order of powers of x.
Complex conjugate zeros must appear consecutively.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mc01py_single_real_zero():
    """
    Test with single real zero.

    Zero: r = 3
    P(x) = (x - 3) = x - 3
    Decreasing order: [1, -3] (x^1 coefficient first, then x^0)
    """
    from ctrlsys import mc01py

    rez = np.array([3.0], order='F', dtype=float)
    imz = np.array([0.0], order='F', dtype=float)

    p, info = mc01py(rez, imz)

    assert info == 0
    expected_p = np.array([1.0, -3.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01py_two_real_zeros():
    """
    Test with two real zeros.

    Zeros: r1 = 1, r2 = 2
    P(x) = (x - 1)(x - 2) = x^2 - 3*x + 2
    Decreasing order: [1, -3, 2] (x^2, x^1, x^0)
    """
    from ctrlsys import mc01py

    rez = np.array([1.0, 2.0], order='F', dtype=float)
    imz = np.array([0.0, 0.0], order='F', dtype=float)

    p, info = mc01py(rez, imz)

    assert info == 0
    expected_p = np.array([1.0, -3.0, 2.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01py_complex_conjugate_pair():
    """
    Test with complex conjugate pair.

    Zeros: 1+2i, 1-2i
    P(x) = (x - (1+2i))(x - (1-2i)) = x^2 - 2x + 5
    Decreasing order: [1, -2, 5]
    """
    from ctrlsys import mc01py

    rez = np.array([1.0, 1.0], order='F', dtype=float)
    imz = np.array([2.0, -2.0], order='F', dtype=float)

    p, info = mc01py(rez, imz)

    assert info == 0
    expected_p = np.array([1.0, -2.0, 5.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01py_zero_degree():
    """
    Test with K=0 (constant polynomial).

    No zeros, P(x) = 1
    Decreasing order: [1]
    """
    from ctrlsys import mc01py

    rez = np.array([], order='F', dtype=float)
    imz = np.array([], order='F', dtype=float)

    p, info = mc01py(rez, imz)

    assert info == 0
    assert len(p) == 1
    assert_allclose(p[0], 1.0, rtol=1e-14)


def test_mc01py_imaginary_zeros():
    """
    Test with pure imaginary zeros.

    Zeros: i, -i (conjugate pair)
    P(x) = (x - i)(x + i) = x^2 + 1
    Decreasing order: [1, 0, 1]
    """
    from ctrlsys import mc01py

    rez = np.array([0.0, 0.0], order='F', dtype=float)
    imz = np.array([1.0, -1.0], order='F', dtype=float)

    p, info = mc01py(rez, imz)

    assert info == 0
    expected_p = np.array([1.0, 0.0, 1.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01py_error_unpaired_complex():
    """
    Test error when complex zero is not paired with conjugate.

    If last zero is complex (IMZ != 0), it has no pair -> error.
    INFO = K (the last index).
    """
    from ctrlsys import mc01py

    rez = np.array([1.0], order='F', dtype=float)
    imz = np.array([2.0], order='F', dtype=float)

    p, info = mc01py(rez, imz)

    assert info == 1


def test_mc01py_error_wrong_conjugate():
    """
    Test error when conjugate pair is incorrect.

    Zeros: (1, 2i), (2, -2i) - not conjugates (real parts differ)
    Should return INFO = 2.
    """
    from ctrlsys import mc01py

    rez = np.array([1.0, 2.0], order='F', dtype=float)
    imz = np.array([2.0, -2.0], order='F', dtype=float)

    p, info = mc01py(rez, imz)

    assert info == 2


def test_mc01py_mixed_real_complex():
    """
    Test with mix of real and complex conjugate zeros.

    Zeros: r1 = -1 (real), r2,r3 = 1+i, 1-i (conjugate)
    (x + 1)(x - 1 - i)(x - 1 + i) = (x + 1)(x^2 - 2x + 2)
    = x^3 - 2x^2 + 2x + x^2 - 2x + 2
    = x^3 - x^2 + 2
    Decreasing order: [1, -1, 0, 2]
    """
    from ctrlsys import mc01py

    rez = np.array([-1.0, 1.0, 1.0], order='F', dtype=float)
    imz = np.array([0.0, 1.0, -1.0], order='F', dtype=float)

    p, info = mc01py(rez, imz)

    assert info == 0
    expected_p = np.array([1.0, -1.0, 0.0, 2.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01py_five_zeros():
    """
    Test with five zeros (mix of real and complex).

    Zeros: (0, 1), (0, -1), (2, 0), (1, 3), (1, -3)
    (x - i)(x + i) = x^2 + 1
    (x - 2) = x - 2
    (x - 1 - 3i)(x - 1 + 3i) = x^2 - 2x + 10

    P(x) = (x^2 + 1)(x - 2)(x^2 - 2x + 10)
    = (x^2 + 1)(x^3 - 2x^2 + 10x - 2x^2 + 4x - 20)
    = (x^2 + 1)(x^3 - 4x^2 + 14x - 20)
    = x^5 - 4x^4 + 14x^3 - 20x^2 + x^3 - 4x^2 + 14x - 20
    = x^5 - 4x^4 + 15x^3 - 24x^2 + 14x - 20

    Decreasing order: [1, -4, 15, -24, 14, -20]
    """
    from ctrlsys import mc01py

    rez = np.array([0.0, 0.0, 2.0, 1.0, 1.0], order='F', dtype=float)
    imz = np.array([1.0, -1.0, 0.0, 3.0, -3.0], order='F', dtype=float)

    p, info = mc01py(rez, imz)

    assert info == 0
    expected_p = np.array([1.0, -4.0, 15.0, -24.0, 14.0, -20.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01py_numpy_polyval_validation():
    """
    Validate polynomial at zeros evaluates to zero.

    Mathematical property: P(r_i) = 0 for all zeros r_i.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc01py

    np.random.seed(42)

    zeros = np.array([-1.5, -0.5, 2.0, 3.5], order='F', dtype=float)
    rez = zeros.copy()
    imz = np.zeros_like(zeros)

    p, info = mc01py(rez, imz)
    assert info == 0

    for z in zeros:
        val = np.polyval(p, z)
        assert abs(val) < 1e-12


def test_mc01py_complex_zeros_polyval():
    """
    Validate polynomial at complex zeros evaluates to zero.

    Mathematical property: P(r_i) = 0 for all zeros r_i.
    """
    from ctrlsys import mc01py

    rez = np.array([1.0, 1.0, -3.0], order='F', dtype=float)
    imz = np.array([2.0, -2.0, 0.0], order='F', dtype=float)

    p, info = mc01py(rez, imz)
    assert info == 0

    zeros_complex = [1 + 2j, 1 - 2j, -3 + 0j]
    for z in zeros_complex:
        val = np.polyval(p, z)
        assert abs(val) < 1e-12
