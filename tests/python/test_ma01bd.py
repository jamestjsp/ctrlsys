"""
Tests for MA01BD - General product of K real scalars without overflow/underflow.

MA01BD computes ALPHA / BETA * BASE^SCAL = product of K scalars.
Each scalar's contribution is controlled by signature S[i]:
  S[i] = 1:  multiply (contribute to numerator)
  S[i] = -1: divide (contribute to denominator)

Test data sources:
- Mathematical properties of products
- Known special cases
- Edge cases (zeros, ones, large/small numbers)
"""

import math
import numpy as np
import pytest

from ctrlsys import ma01bd


def test_ma01bd_simple_product():
    """
    Test simple product of positive numbers.
    Product: 2 * 3 * 4 = 24
    All signatures = 1 (multiply).
    """
    base = 2.0
    lgbas = math.log(base)
    k = 3
    s = np.array([1, 1, 1], dtype=np.int32)
    a = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 24.0, rtol=1e-14)


def test_ma01bd_simple_division():
    """
    Test product with divisions.
    Product: 24 / 3 / 4 = 2
    S = [1, -1, -1] means: a[0] * (1/a[1]) * (1/a[2])
    """
    base = 2.0
    lgbas = math.log(base)
    k = 3
    s = np.array([1, -1, -1], dtype=np.int32)
    a = np.array([24.0, 3.0, 4.0], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 2.0, rtol=1e-14)


def test_ma01bd_single_element():
    """
    Test with single element (K=1).
    """
    base = 2.0
    lgbas = math.log(base)
    k = 1
    s = np.array([1], dtype=np.int32)
    a = np.array([5.0], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 5.0, rtol=1e-14)


def test_ma01bd_with_zero():
    """
    Test product containing zero.
    Product: 2 * 0 * 4 = 0
    """
    base = 2.0
    lgbas = math.log(base)
    k = 3
    s = np.array([1, 1, 1], dtype=np.int32)
    a = np.array([2.0, 0.0, 4.0], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    result = alpha * (base ** scal) if beta == 1.0 else 0.0
    np.testing.assert_allclose(result, 0.0, atol=1e-15)


def test_ma01bd_negative_numbers():
    """
    Test product with negative numbers.
    Product: (-2) * 3 * (-4) = 24
    """
    base = 2.0
    lgbas = math.log(base)
    k = 3
    s = np.array([1, 1, 1], dtype=np.int32)
    a = np.array([-2.0, 3.0, -4.0], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 24.0, rtol=1e-14)


def test_ma01bd_inca_stride():
    """
    Test with non-unity stride (INCA=2).
    Uses every second element: a[0], a[2], a[4] = 2, 4, 6
    Product: 2 * 4 * 6 = 48
    """
    base = 2.0
    lgbas = math.log(base)
    k = 3
    s = np.array([1, 1, 1], dtype=np.int32)
    a = np.array([2.0, 100.0, 4.0, 100.0, 6.0], dtype=np.float64)
    inca = 2

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 48.0, rtol=1e-14)


def test_ma01bd_large_numbers():
    """
    Test prevention of overflow with large numbers.
    Product: 1e100 * 1e100 = 1e200 (would overflow without scaling)

    Random seed: 42 (for reproducibility)
    """
    base = 2.0
    lgbas = math.log(base)
    k = 2
    s = np.array([1, 1], dtype=np.int32)
    a = np.array([1e100, 1e100], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    expected = 1e200
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_ma01bd_small_numbers():
    """
    Test prevention of underflow with small numbers.
    Product: 1e-200 * 1e-100 = 1e-300 (would underflow without scaling)
    """
    base = 2.0
    lgbas = math.log(base)
    k = 2
    s = np.array([1, 1], dtype=np.int32)
    a = np.array([1e-200, 1e-100], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    expected = 1e-300
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_ma01bd_mixed_large_small():
    """
    Test product of large and small numbers (cancellation).
    Product: 1e200 * 1e-200 = 1.0
    """
    base = 2.0
    lgbas = math.log(base)
    k = 2
    s = np.array([1, 1], dtype=np.int32)
    a = np.array([1e200, 1e-200], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 1.0, rtol=1e-12)


def test_ma01bd_property_product_correctness():
    """
    Mathematical property: result equals true product for moderate numbers.

    Tests that ALPHA / BETA * BASE^SCAL equals naive product.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    base = 2.0
    lgbas = math.log(base)

    for _ in range(10):
        k = np.random.randint(2, 8)
        s = np.random.choice([1, -1], size=k).astype(np.int32)
        a = np.random.uniform(0.5, 2.0, size=k)
        inca = 1

        alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

        result = alpha * (base ** scal) if beta == 1.0 else 0.0

        expected = 1.0
        for i in range(k):
            if s[i] == 1:
                expected *= a[i]
            else:
                expected /= a[i]

        np.testing.assert_allclose(result, expected, rtol=1e-13)


def test_ma01bd_all_ones():
    """
    Test product of all ones.
    Product: 1 * 1 * 1 = 1
    """
    base = 2.0
    lgbas = math.log(base)
    k = 5
    s = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 1.0, rtol=1e-14)


def test_ma01bd_powers_of_base():
    """
    Test with powers of base (exact in floating point).
    Product: 2 * 4 * 8 = 64 (using base=2)
    """
    base = 2.0
    lgbas = math.log(base)
    k = 3
    s = np.array([1, 1, 1], dtype=np.int32)
    a = np.array([2.0, 4.0, 8.0], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 64.0, rtol=1e-14)


def test_ma01bd_base_10():
    """
    Test with base 10.
    Product: 100 * 1000 = 100000
    """
    base = 10.0
    lgbas = math.log(base)
    k = 2
    s = np.array([1, 1], dtype=np.int32)
    a = np.array([100.0, 1000.0], dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 100000.0, rtol=1e-14)


def test_ma01bd_periodic_rescaling():
    """
    Test with many elements to trigger periodic rescaling (every 10 elements).
    Product: 2^15 = 32768 (15 factors of 2)
    """
    base = 2.0
    lgbas = math.log(base)
    k = 15
    s = np.ones(k, dtype=np.int32)
    a = np.full(k, 2.0, dtype=np.float64)
    inca = 1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 32768.0, rtol=1e-14)


def test_ma01bd_negative_stride():
    """
    Test with negative stride (INCA=-1).

    For negative stride, array must be passed starting from appropriate offset.
    With INCA=-1 and k=4, accesses indices: 0, -1, -2, -3 from starting point.
    We pass a[3:] to start at element 5.0 and go backward: 5, 4, 3, 2.
    Product: 5 * 4 * 3 * 2 = 120
    """
    base = 2.0
    lgbas = math.log(base)
    k = 4
    s = np.array([1, 1, 1, 1], dtype=np.int32)
    a_full = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    a = a_full[3:]  # Start at last element
    inca = -1

    alpha, beta, scal = ma01bd(base, lgbas, k, s, a, inca)

    assert beta == 1.0
    result = alpha * (base ** scal)
    np.testing.assert_allclose(result, 120.0, rtol=1e-14)
