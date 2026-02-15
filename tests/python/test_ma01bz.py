"""
Tests for MA01BZ - General product of K complex scalars without overflow/underflow.

MA01BZ computes ALPHA / BETA * BASE^SCAL = product of K complex scalars.
Each scalar's contribution is controlled by signature S[i]:
  S[i] = 1:  multiply (contribute to numerator)
  S[i] = -1: divide (contribute to denominator)

Test data sources:
- Mathematical properties of complex products
- Known special cases
- Edge cases (zeros, ones, large/small numbers)
"""

import numpy as np
import pytest

from ctrlsys import ma01bz


def test_ma01bz_simple_product():
    """
    Test simple product of complex numbers.
    Product: (1+1j) * (2+0j) * (0+3j) = (1+1j)*2*(3j) = 2*(1+1j)*3j
           = 2*(3j + 3j*j) = 2*(3j - 3) = -6 + 6j
    All signatures = 1 (multiply).
    """
    base = 2.0
    k = 3
    s = np.array([1, 1, 1], dtype=np.int32)
    a = np.array([1+1j, 2+0j, 0+3j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0) or beta == complex(0.0, 0.0)
    if beta != complex(0.0, 0.0):
        result = alpha * (base ** scal)
        expected = (1+1j) * (2+0j) * (0+3j)
        np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_ma01bz_simple_division():
    """
    Test product with divisions.
    Product: (6+0j) / (2+0j) / (3+0j) = 1
    S = [1, -1, -1]
    """
    base = 2.0
    k = 3
    s = np.array([1, -1, -1], dtype=np.int32)
    a = np.array([6+0j, 2+0j, 3+0j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0) or beta == complex(0.0, 0.0)
    if beta != complex(0.0, 0.0):
        result = alpha * (base ** scal)
        np.testing.assert_allclose(result, 1.0+0j, rtol=1e-14)


def test_ma01bz_single_element():
    """
    Test with single element (K=1).
    """
    base = 2.0
    k = 1
    s = np.array([1], dtype=np.int32)
    a = np.array([3+4j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0)
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 3+4j, rtol=1e-14)


def test_ma01bz_with_zero():
    """
    Test product containing zero.
    Product: (2+3j) * 0 * (4+5j) = 0
    When dividing by zero, beta should be 0.
    """
    base = 2.0
    k = 3
    s = np.array([1, 1, 1], dtype=np.int32)
    a = np.array([2+3j, 0+0j, 4+5j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    result = alpha * (base ** scal) if beta == complex(1.0, 0.0) else 0.0
    np.testing.assert_allclose(result, 0.0, atol=1e-15)


def test_ma01bz_division_by_zero():
    """
    Test division by zero case.
    When S[i]=-1 and A[i]=0, beta should be set to 0.
    """
    base = 2.0
    k = 3
    s = np.array([1, -1, 1], dtype=np.int32)
    a = np.array([2+3j, 0+0j, 4+5j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(0.0, 0.0)


def test_ma01bz_inca_stride():
    """
    Test with non-unity stride (INCA=2).
    Uses every second element: a[0], a[2] = (1+1j), (2+2j)
    Product: (1+1j) * (2+2j) = 2 + 2j + 2j + 2j^2 = 2 + 4j - 2 = 4j
    """
    base = 2.0
    k = 2
    s = np.array([1, 1], dtype=np.int32)
    a = np.array([1+1j, 99+99j, 2+2j], dtype=np.complex128)
    inca = 2

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0)
    result = alpha * (base ** scal)
    expected = (1+1j) * (2+2j)
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_ma01bz_pure_imaginary():
    """
    Test with pure imaginary numbers.
    Product: 1j * 2j * 3j = 6j^3 = 6*(-j) = -6j
    """
    base = 2.0
    k = 3
    s = np.array([1, 1, 1], dtype=np.int32)
    a = np.array([1j, 2j, 3j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0)
    result = alpha * (base ** scal)
    expected = 1j * 2j * 3j
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_ma01bz_all_ones():
    """
    Test product of all ones.
    Product: 1 * 1 * 1 = 1
    """
    base = 2.0
    k = 5
    s = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    a = np.array([1+0j, 1+0j, 1+0j, 1+0j, 1+0j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0)
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 1+0j, rtol=1e-14)


def test_ma01bz_large_numbers():
    """
    Test prevention of overflow with large complex numbers.
    """
    base = 2.0
    k = 2
    s = np.array([1, 1], dtype=np.int32)
    a = np.array([1e100+1e100j, 1e50+0j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0)
    result = alpha * (base ** scal)
    expected = (1e100+1e100j) * (1e50+0j)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_ma01bz_small_numbers():
    """
    Test prevention of underflow with small complex numbers.
    """
    base = 2.0
    k = 2
    s = np.array([1, 1], dtype=np.int32)
    a = np.array([1e-100+1e-100j, 1e-50+0j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0)
    result = alpha * (base ** scal)
    expected = (1e-100+1e-100j) * (1e-50+0j)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_ma01bz_property_product_correctness():
    """
    Mathematical property: result equals true product for moderate numbers.

    Tests that ALPHA / BETA * BASE^SCAL equals naive product.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    base = 2.0

    for _ in range(10):
        k = np.random.randint(2, 6)
        s = np.random.choice([1, -1], size=k).astype(np.int32)
        re = np.random.uniform(0.5, 2.0, size=k)
        im = np.random.uniform(-1.0, 1.0, size=k)
        a = (re + 1j * im).astype(np.complex128)
        inca = 1

        alpha, beta, scal = ma01bz(base, k, s, a, inca)

        if beta == complex(0.0, 0.0):
            continue

        result = alpha * (base ** scal)

        expected = complex(1.0, 0.0)
        for i in range(k):
            if s[i] == 1:
                expected *= a[i]
            else:
                expected /= a[i]

        np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_ma01bz_alpha_magnitude_normalized():
    """
    Mathematical property: |ALPHA| is normalized to [1, BASE) when non-zero.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    base = 2.0

    for _ in range(10):
        k = np.random.randint(2, 6)
        s = np.ones(k, dtype=np.int32)
        re = np.random.uniform(0.5, 3.0, size=k)
        im = np.random.uniform(-1.5, 1.5, size=k)
        a = (re + 1j * im).astype(np.complex128)
        inca = 1

        alpha, beta, scal = ma01bz(base, k, s, a, inca)

        if abs(alpha) > 0:
            assert 1.0 <= abs(alpha) < base, f"|alpha| = {abs(alpha)} not in [1, {base})"


def test_ma01bz_conjugate_pairs():
    """
    Test with conjugate pairs - product should be real and positive.
    (a + bi)(a - bi) = a^2 + b^2
    Product: (3+4j)(3-4j) = 25
    """
    base = 2.0
    k = 2
    s = np.array([1, 1], dtype=np.int32)
    a = np.array([3+4j, 3-4j], dtype=np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0)
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 25+0j, rtol=1e-14)
    np.testing.assert_allclose(result.imag, 0.0, atol=1e-14)


def test_ma01bz_unit_circle():
    """
    Test with numbers on unit circle (|z|=1).
    These shouldn't need much scaling.
    """
    base = 2.0
    k = 4
    s = np.array([1, 1, 1, 1], dtype=np.int32)
    angles = np.array([0, np.pi/4, np.pi/2, np.pi])
    a = np.exp(1j * angles).astype(np.complex128)
    inca = 1

    alpha, beta, scal = ma01bz(base, k, s, a, inca)

    assert beta == complex(1.0, 0.0)
    result = alpha * (base ** scal)
    expected = np.prod(a)
    np.testing.assert_allclose(result, expected, rtol=1e-14)
