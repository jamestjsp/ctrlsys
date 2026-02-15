"""
Tests for MA01CD - Compute sign of sum of two scaled numbers.

MA01CD computes, without over- or underflow, the sign of the sum of two
real numbers represented using integer powers of a base:
    A * BASE^IA + B * BASE^IB

Returns 1 (positive), 0 (zero), or -1 (negative).

Test data sources:
- Mathematical properties of signed sums
- Edge cases (zeros, equal exponents, opposite signs)
"""

import numpy as np
import pytest

from ctrlsys import ma01cd


def test_ma01cd_both_zero():
    """
    Both A and B are zero - result should be 0.
    """
    result = ma01cd(0.0, 0, 0.0, 0)
    assert result == 0


def test_ma01cd_a_zero():
    """
    A is zero, B positive - result should be sign of B.
    """
    result = ma01cd(0.0, 0, 3.5, 0)
    assert result == 1

    result = ma01cd(0.0, 0, -2.5, 10)
    assert result == -1


def test_ma01cd_b_zero():
    """
    B is zero, A positive/negative - result should be sign of A.
    """
    result = ma01cd(5.0, 0, 0.0, 0)
    assert result == 1

    result = ma01cd(-7.0, 5, 0.0, 0)
    assert result == -1


def test_ma01cd_equal_exponents_positive_sum():
    """
    Equal exponents, sum is positive.
    A * BASE^IA + B * BASE^IB = (A + B) * BASE^IA
    3.0 + 2.0 = 5.0 > 0
    """
    result = ma01cd(3.0, 5, 2.0, 5)
    assert result == 1


def test_ma01cd_equal_exponents_negative_sum():
    """
    Equal exponents, sum is negative.
    -3.0 + (-2.0) = -5.0 < 0
    """
    result = ma01cd(-3.0, 5, -2.0, 5)
    assert result == -1


def test_ma01cd_equal_exponents_zero_sum():
    """
    Equal exponents, sum is zero.
    3.0 + (-3.0) = 0
    """
    result = ma01cd(3.0, 5, -3.0, 5)
    assert result == 0


def test_ma01cd_same_sign_different_exponents():
    """
    Same sign, different exponents - result should match the common sign.
    2.0 * 10^5 + 3.0 * 10^3 > 0
    """
    result = ma01cd(2.0, 5, 3.0, 3)
    assert result == 1

    result = ma01cd(-2.0, 5, -3.0, 3)
    assert result == -1


def test_ma01cd_opposite_signs_larger_first():
    """
    Opposite signs, first term dominates (IA > IB).
    A * BASE^IA + B * BASE^IB where |A * BASE^IA| > |B * BASE^IB|
    Result should be sign of A.

    Using BASE=10: 5.0 * 10^10 - 2.0 * 10^5 > 0
    """
    result = ma01cd(5.0, 10, -2.0, 5)
    assert result == 1

    result = ma01cd(-5.0, 10, 2.0, 5)
    assert result == -1


def test_ma01cd_opposite_signs_larger_second():
    """
    Opposite signs, second term dominates (IB > IA).
    A * BASE^IA + B * BASE^IB where |B * BASE^IB| > |A * BASE^IA|
    Result should be sign of B.
    """
    result = ma01cd(2.0, 5, -5.0, 10)
    assert result == -1

    result = ma01cd(-2.0, 5, 5.0, 10)
    assert result == 1


def test_ma01cd_large_exponent_difference():
    """
    Large exponent difference - avoids overflow by using logarithms.
    A * BASE^IA + B * BASE^IB with |IA - IB| >> 0

    This tests the overflow prevention mechanism.
    """
    result = ma01cd(1.0, 1000, -1.0, 0)
    assert result == 1

    result = ma01cd(-1.0, 1000, 1.0, 0)
    assert result == -1

    result = ma01cd(1.0, 0, -1.0, 1000)
    assert result == -1

    result = ma01cd(-1.0, 0, 1.0, 1000)
    assert result == 1


def test_ma01cd_close_magnitudes_opposite_signs():
    """
    Close magnitudes with opposite signs - tests boundary comparison.

    When IA > IB: log(|A|) + IA - IB >= log(|B|) means A dominates

    Example with BASE=e (natural log):
    A = e^2, B = -e^1, IA = 5, IB = 3
    log(e^2) + 5 - 3 = 2 + 2 = 4 >= log(e) = 1 => A dominates => sign = +1
    """
    import math
    e = math.e
    result = ma01cd(e**2, 5, -e, 3)
    assert result == 1


def test_ma01cd_boundary_case_equal_log_magnitudes():
    """
    Boundary case where log magnitudes are equal.
    log(|A|) + IA - IB = log(|B|) when IA > IB

    This tests the >= comparison in the algorithm.
    A = 2.0, B = -4.0, IA = 1, IB = 0
    log(2) + 1 - 0 = 0.693 + 1 = 1.693 >= log(4) = 1.386 => A dominates
    """
    result = ma01cd(2.0, 1, -4.0, 0)
    assert result == 1


def test_ma01cd_negative_exponents():
    """
    Negative exponents should work correctly.
    A * BASE^(-10) + B * BASE^(-5)
    """
    result = ma01cd(1.0, -10, 1.0, -5)
    assert result == 1

    result = ma01cd(1.0, -10, -1.0, -5)
    assert result == -1


def test_ma01cd_property_symmetry():
    """
    Property test: swapping A,IA with B,IB should give same result
    if signs are equal, otherwise depends on magnitudes.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    for _ in range(20):
        a = np.random.randn() * 10
        b = np.random.randn() * 10
        ia = np.random.randint(-50, 50)
        ib = np.random.randint(-50, 50)

        result1 = ma01cd(a, ia, b, ib)
        result2 = ma01cd(b, ib, a, ia)

        assert result1 == result2, f"Symmetry failed for a={a}, ia={ia}, b={b}, ib={ib}"


def test_ma01cd_property_consistency_with_direct_sum():
    """
    Property test: when exponents equal and sum computable,
    result should match sign of direct sum.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    for _ in range(20):
        a = np.random.randn() * 100
        b = np.random.randn() * 100
        ia = np.random.randint(-10, 10)

        result = ma01cd(a, ia, b, ia)

        direct_sum = a + b
        if direct_sum > 0:
            expected = 1
        elif direct_sum < 0:
            expected = -1
        else:
            expected = 0

        assert result == expected, f"Direct sum check failed: a={a}, b={b}, sum={direct_sum}"


def test_ma01cd_property_sign_of_dominant_term():
    """
    Property test: when one term clearly dominates (large exponent diff),
    result should be sign of dominant term.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    for _ in range(20):
        a = np.random.randn()
        b = np.random.randn()
        if a == 0.0:
            a = 1.0
        if b == 0.0:
            b = 1.0

        ia = 100
        ib = 0

        result = ma01cd(a, ia, b, ib)
        expected = 1 if a > 0 else -1

        assert result == expected, f"Dominant term test failed: a={a}"


def test_ma01cd_unity_values():
    """
    Test with unity values.
    1.0 * BASE^0 + 1.0 * BASE^0 = 2.0 > 0
    """
    result = ma01cd(1.0, 0, 1.0, 0)
    assert result == 1


def test_ma01cd_opposite_unity():
    """
    Test opposite unity values.
    1.0 * BASE^0 + (-1.0) * BASE^0 = 0
    """
    result = ma01cd(1.0, 0, -1.0, 0)
    assert result == 0
