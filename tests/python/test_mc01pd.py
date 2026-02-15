"""
Tests for MC01PD - Compute polynomial coefficients from zeros.

Computes coefficients of real polynomial P(x) = (x - r1)(x - r2)...(x - rk)
from given zeros. Complex conjugate zeros must appear consecutively.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mc01pd_basic_html_example():
    """
    Test using SLICOT HTML doc example.

    Zeros: (0, 1), (0, -1), (2, 0), (1, 3), (1, -3)
    Expected coefficients (increasing powers of x):
    p[0] = -20.0, p[1] = 14.0, p[2] = -24.0, p[3] = 15.0, p[4] = -4.0, p[5] = 1.0
    """
    from ctrlsys import mc01pd

    k = 5
    rez = np.array([0.0, 0.0, 2.0, 1.0, 1.0], order='F', dtype=float)
    imz = np.array([1.0, -1.0, 0.0, 3.0, -3.0], order='F', dtype=float)

    p, info = mc01pd(rez, imz)

    assert info == 0
    assert len(p) == k + 1

    expected_p = np.array([-20.0, 14.0, -24.0, 15.0, -4.0, 1.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01pd_single_real_zero():
    """
    Test with single real zero.

    Zero: r = 3
    P(x) = (x - 3) = -3 + x
    Coefficients: [-3, 1]
    """
    from ctrlsys import mc01pd

    rez = np.array([3.0], order='F', dtype=float)
    imz = np.array([0.0], order='F', dtype=float)

    p, info = mc01pd(rez, imz)

    assert info == 0
    expected_p = np.array([-3.0, 1.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01pd_two_real_zeros():
    """
    Test with two real zeros.

    Zeros: r1 = 1, r2 = 2
    P(x) = (x - 1)(x - 2) = 2 - 3*x + x^2
    Coefficients: [2, -3, 1]
    """
    from ctrlsys import mc01pd

    rez = np.array([1.0, 2.0], order='F', dtype=float)
    imz = np.array([0.0, 0.0], order='F', dtype=float)

    p, info = mc01pd(rez, imz)

    assert info == 0
    expected_p = np.array([2.0, -3.0, 1.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01pd_complex_conjugate_pair():
    """
    Test with complex conjugate pair.

    Zeros: 1+2i, 1-2i
    P(x) = (x - (1+2i))(x - (1-2i)) = x^2 - 2x + 5
    Coefficients: [5, -2, 1]
    """
    from ctrlsys import mc01pd

    rez = np.array([1.0, 1.0], order='F', dtype=float)
    imz = np.array([2.0, -2.0], order='F', dtype=float)

    p, info = mc01pd(rez, imz)

    assert info == 0
    expected_p = np.array([5.0, -2.0, 1.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01pd_zero_degree():
    """
    Test with K=0 (constant polynomial).

    No zeros, P(x) = 1
    """
    from ctrlsys import mc01pd

    rez = np.array([], order='F', dtype=float)
    imz = np.array([], order='F', dtype=float)

    p, info = mc01pd(rez, imz)

    assert info == 0
    assert len(p) == 1
    assert_allclose(p[0], 1.0, rtol=1e-14)


def test_mc01pd_imaginary_zeros():
    """
    Test with pure imaginary zeros.

    Zeros: i, -i (conjugate pair)
    P(x) = (x - i)(x + i) = x^2 + 1
    Coefficients: [1, 0, 1]
    """
    from ctrlsys import mc01pd

    rez = np.array([0.0, 0.0], order='F', dtype=float)
    imz = np.array([1.0, -1.0], order='F', dtype=float)

    p, info = mc01pd(rez, imz)

    assert info == 0
    expected_p = np.array([1.0, 0.0, 1.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01pd_error_unpaired_complex():
    """
    Test error when complex zero is not paired with conjugate.

    If last zero is complex (IMZ != 0), it has no pair -> error.
    """
    from ctrlsys import mc01pd

    rez = np.array([1.0], order='F', dtype=float)
    imz = np.array([2.0], order='F', dtype=float)

    p, info = mc01pd(rez, imz)

    assert info == 1


def test_mc01pd_error_wrong_conjugate():
    """
    Test error when conjugate pair is incorrect.

    Zeros: (1, 2i), (2, -2i) - not conjugates
    Should return INFO = 2.
    """
    from ctrlsys import mc01pd

    rez = np.array([1.0, 2.0], order='F', dtype=float)
    imz = np.array([2.0, -2.0], order='F', dtype=float)

    p, info = mc01pd(rez, imz)

    assert info == 2


def test_mc01pd_mixed_real_complex():
    """
    Test with mix of real and complex conjugate zeros.

    Zeros: r1 = -1 (real), r2,r3 = 1+i, 1-i (conjugate)
    P(x) = (x + 1)(x^2 - 2x + 2) = x^3 - x^2 + 2
    = 2 + 0*x - 1*x^2 + 1*x^3
    Coefficients: [2, 0, -1, 1]

    Verification:
    (x - 1 - i)(x - 1 + i) = x^2 - (1+i)x - (1-i)x + (1+i)(1-i)
                           = x^2 - 2x + 2
    (x + 1)(x^2 - 2x + 2) = x^3 - 2x^2 + 2x + x^2 - 2x + 2
                          = x^3 - x^2 + 2
    """
    from ctrlsys import mc01pd

    rez = np.array([-1.0, 1.0, 1.0], order='F', dtype=float)
    imz = np.array([0.0, 1.0, -1.0], order='F', dtype=float)

    p, info = mc01pd(rez, imz)

    assert info == 0
    expected_p = np.array([2.0, 0.0, -1.0, 1.0])
    assert_allclose(p, expected_p, rtol=1e-14)


def test_mc01pd_numpy_roots_validation():
    """
    Validate polynomial expansion against NumPy.

    Generate random real zeros, compute poly with MC01PD,
    verify roots match original zeros.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc01pd

    np.random.seed(42)

    zeros = np.array([-1.5, -0.5, 2.0, 3.5], order='F', dtype=float)
    rez = zeros.copy()
    imz = np.zeros_like(zeros)

    p, info = mc01pd(rez, imz)
    assert info == 0

    roots = np.roots(p[::-1])
    roots_sorted = np.sort(roots.real)
    zeros_sorted = np.sort(zeros)

    assert_allclose(roots_sorted, zeros_sorted, rtol=1e-13)


def test_mc01pd_complex_roots_validation():
    """
    Validate polynomial with complex zeros against NumPy.

    Zeros: 1+2i, 1-2i, -3 (real)
    P(x) = (x^2 - 2x + 5)(x + 3) = x^3 + x^2 - x + 15

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01pd

    rez = np.array([1.0, 1.0, -3.0], order='F', dtype=float)
    imz = np.array([2.0, -2.0, 0.0], order='F', dtype=float)

    p, info = mc01pd(rez, imz)
    assert info == 0

    roots = np.roots(p[::-1])
    roots_sorted = sorted(roots, key=lambda x: (x.real, x.imag))

    expected_roots = [1 + 2j, 1 - 2j, -3 + 0j]
    expected_sorted = sorted(expected_roots, key=lambda x: (x.real, x.imag))

    for r, e in zip(roots_sorted, expected_sorted):
        assert_allclose([r.real, r.imag], [e.real, e.imag], rtol=1e-13)
