"""
Tests for MC01OD - Compute complex polynomial coefficients from zeros.

Computes coefficients of complex polynomial P(x) = (x - r(1))...(x - r(K))
where r(i) = REZ(i) + j*IMZ(i) are complex zeros.

Unlike MC01PD which produces real polynomials from conjugate-paired zeros,
MC01OD produces complex polynomial coefficients (REP + j*IMP).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mc01od_basic_html_example():
    """
    Test using SLICOT HTML doc example.

    Zeros (5 complex):
      z1 = 1.1 + 0.9j
      z2 = 0.6 - 0.7j
      z3 = -2.0 + 0.3j
      z4 = -0.8 + 2.5j
      z5 = -0.3 - 0.4j

    Expected coefficients (increasing powers of x):
      p[0] = 2.7494 - 2.1300j
      p[1] = -1.7590 - 5.4205j
      p[2] = 0.0290 + 2.8290j
      p[3] = -1.6500 - 1.7300j
      p[4] = 1.4000 - 2.6000j
      p[5] = 1.0000 + 0.0000j
    """
    from ctrlsys import mc01od

    rez = np.array([1.1, 0.6, -2.0, -0.8, -0.3], order='F', dtype=float)
    imz = np.array([0.9, -0.7, 0.3, 2.5, -0.4], order='F', dtype=float)

    rep, imp, info = mc01od(rez, imz)

    assert info == 0
    assert len(rep) == 6
    assert len(imp) == 6

    expected_rep = np.array([2.7494, -1.7590, 0.0290, -1.6500, 1.4000, 1.0000])
    expected_imp = np.array([-2.1300, -5.4205, 2.8290, -1.7300, -2.6000, 0.0000])

    assert_allclose(rep, expected_rep, rtol=1e-3, atol=1e-4)
    assert_allclose(imp, expected_imp, rtol=1e-3, atol=1e-4)


def test_mc01od_single_real_zero():
    """
    Test with single real zero.

    Zero: r = 3 (i.e., 3 + 0j)
    P(x) = (x - 3)
    Coefficients: p[0] = -3, p[1] = 1 (both real)
    """
    from ctrlsys import mc01od

    rez = np.array([3.0], order='F', dtype=float)
    imz = np.array([0.0], order='F', dtype=float)

    rep, imp, info = mc01od(rez, imz)

    assert info == 0
    expected_rep = np.array([-3.0, 1.0])
    expected_imp = np.array([0.0, 0.0])

    assert_allclose(rep, expected_rep, rtol=1e-14)
    assert_allclose(imp, expected_imp, atol=1e-14)


def test_mc01od_single_complex_zero():
    """
    Test with single complex zero.

    Zero: r = 2 + 3j
    P(x) = (x - (2 + 3j)) = x + (-2 - 3j)
    Coefficients: p[0] = -2 - 3j, p[1] = 1 + 0j
    """
    from ctrlsys import mc01od

    rez = np.array([2.0], order='F', dtype=float)
    imz = np.array([3.0], order='F', dtype=float)

    rep, imp, info = mc01od(rez, imz)

    assert info == 0
    expected_rep = np.array([-2.0, 1.0])
    expected_imp = np.array([-3.0, 0.0])

    assert_allclose(rep, expected_rep, rtol=1e-14)
    assert_allclose(imp, expected_imp, rtol=1e-14)


def test_mc01od_two_real_zeros():
    """
    Test with two real zeros.

    Zeros: r1 = 1, r2 = 2
    P(x) = (x - 1)(x - 2) = x^2 - 3x + 2
    Coefficients: [2, -3, 1]
    """
    from ctrlsys import mc01od

    rez = np.array([1.0, 2.0], order='F', dtype=float)
    imz = np.array([0.0, 0.0], order='F', dtype=float)

    rep, imp, info = mc01od(rez, imz)

    assert info == 0
    expected_rep = np.array([2.0, -3.0, 1.0])
    expected_imp = np.array([0.0, 0.0, 0.0])

    assert_allclose(rep, expected_rep, rtol=1e-14)
    assert_allclose(imp, expected_imp, atol=1e-14)


def test_mc01od_two_complex_zeros():
    """
    Test with two complex zeros (not conjugates).

    Zeros: r1 = 1 + j, r2 = 2 + 3j
    P(x) = (x - (1+j))(x - (2+3j))
         = x^2 - (3+4j)x + (1+j)(2+3j)
         = x^2 - (3+4j)x + (2 + 3j + 2j + 3j^2)
         = x^2 - (3+4j)x + (2 + 5j - 3)
         = x^2 - (3+4j)x + (-1 + 5j)

    Coefficients: [(-1+5j), (-3-4j), (1+0j)]
    """
    from ctrlsys import mc01od

    rez = np.array([1.0, 2.0], order='F', dtype=float)
    imz = np.array([1.0, 3.0], order='F', dtype=float)

    rep, imp, info = mc01od(rez, imz)

    assert info == 0
    expected_rep = np.array([-1.0, -3.0, 1.0])
    expected_imp = np.array([5.0, -4.0, 0.0])

    assert_allclose(rep, expected_rep, rtol=1e-14)
    assert_allclose(imp, expected_imp, rtol=1e-14)


def test_mc01od_zero_degree():
    """
    Test with K=0 (constant polynomial).

    No zeros, P(x) = 1
    REP = [1.0], IMP = [0.0]
    """
    from ctrlsys import mc01od

    rez = np.array([], order='F', dtype=float)
    imz = np.array([], order='F', dtype=float)

    rep, imp, info = mc01od(rez, imz)

    assert info == 0
    assert len(rep) == 1
    assert len(imp) == 1
    assert_allclose(rep[0], 1.0, rtol=1e-14)
    assert_allclose(imp[0], 0.0, atol=1e-14)


def test_mc01od_pure_imaginary_zeros():
    """
    Test with pure imaginary zeros.

    Zeros: r1 = j, r2 = 2j
    P(x) = (x - j)(x - 2j) = x^2 - 3jx + j*2j = x^2 - 3jx - 2

    Coefficients: [(-2+0j), (0-3j), (1+0j)]
    """
    from ctrlsys import mc01od

    rez = np.array([0.0, 0.0], order='F', dtype=float)
    imz = np.array([1.0, 2.0], order='F', dtype=float)

    rep, imp, info = mc01od(rez, imz)

    assert info == 0
    expected_rep = np.array([-2.0, 0.0, 1.0])
    expected_imp = np.array([0.0, -3.0, 0.0])

    assert_allclose(rep, expected_rep, rtol=1e-14)
    assert_allclose(imp, expected_imp, rtol=1e-14)


def test_mc01od_conjugate_pair():
    """
    Test with complex conjugate pair (should produce real polynomial).

    Zeros: r1 = 1 + 2j, r2 = 1 - 2j
    P(x) = (x - (1+2j))(x - (1-2j))
         = x^2 - 2x + 5

    All imaginary coefficients should be zero.
    """
    from ctrlsys import mc01od

    rez = np.array([1.0, 1.0], order='F', dtype=float)
    imz = np.array([2.0, -2.0], order='F', dtype=float)

    rep, imp, info = mc01od(rez, imz)

    assert info == 0
    expected_rep = np.array([5.0, -2.0, 1.0])
    expected_imp = np.array([0.0, 0.0, 0.0])

    assert_allclose(rep, expected_rep, rtol=1e-14)
    assert_allclose(imp, expected_imp, atol=1e-14)


def test_mc01od_polynomial_evaluation_property():
    """
    Validate P(r_i) = 0 for each zero r_i.

    The polynomial should evaluate to zero at each of its zeros.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc01od

    np.random.seed(42)

    k = 4
    rez = np.random.randn(k).astype(float, order='F')
    imz = np.random.randn(k).astype(float, order='F')

    rep, imp, info = mc01od(rez, imz)
    assert info == 0

    for i in range(k):
        z = complex(rez[i], imz[i])
        p_at_z = complex(0, 0)
        for j in range(k + 1):
            p_at_z += complex(rep[j], imp[j]) * (z ** j)

        assert_allclose([p_at_z.real, p_at_z.imag], [0.0, 0.0], atol=1e-12)


def test_mc01od_numpy_poly_validation():
    """
    Validate against NumPy polynomial from roots.

    Build polynomial using numpy.poly and compare coefficients.
    Note: numpy.poly returns coefficients in decreasing powers.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mc01od

    np.random.seed(123)

    k = 3
    rez = np.array([1.0, -2.0, 0.5], order='F', dtype=float)
    imz = np.array([0.5, 1.0, -1.5], order='F', dtype=float)

    rep, imp, info = mc01od(rez, imz)
    assert info == 0

    zeros = rez + 1j * imz
    np_poly = np.poly(zeros)

    slicot_poly = rep + 1j * imp
    slicot_poly_reversed = slicot_poly[::-1]

    assert_allclose(slicot_poly_reversed, np_poly, rtol=1e-13)


def test_mc01od_leading_coefficient_unity():
    """
    Validate leading coefficient is always 1.

    For monic polynomial P(x) = (x - r1)...(x - rk), the leading
    coefficient (x^k term) should be 1 + 0j.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mc01od

    np.random.seed(789)

    for k in [1, 3, 5, 7]:
        rez = np.random.randn(k).astype(float, order='F')
        imz = np.random.randn(k).astype(float, order='F')

        rep, imp, info = mc01od(rez, imz)
        assert info == 0

        assert_allclose(rep[k], 1.0, rtol=1e-14)
        assert_allclose(imp[k], 0.0, atol=1e-14)


def test_mc01od_vieta_sum_of_roots():
    """
    Validate Vieta's formula: sum of roots = -p_{k-1}/p_k.

    For P(x) = x^k + a_{k-1}*x^{k-1} + ... + a_0,
    sum of roots = -a_{k-1}

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mc01od

    np.random.seed(456)

    k = 5
    rez = np.random.randn(k).astype(float, order='F')
    imz = np.random.randn(k).astype(float, order='F')

    rep, imp, info = mc01od(rez, imz)
    assert info == 0

    sum_zeros = complex(np.sum(rez), np.sum(imz))
    coeff_km1 = complex(rep[k-1], imp[k-1])

    assert_allclose(
        [sum_zeros.real, sum_zeros.imag],
        [-coeff_km1.real, -coeff_km1.imag],
        rtol=1e-13
    )
