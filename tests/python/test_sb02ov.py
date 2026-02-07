"""
Tests for SB02OV - Select unstable generalized eigenvalues for discrete-time Riccati.

This is a callback function for DGGES eigenvalue selection. Returns True for
generalized eigenvalues with modulus >= 1 (unstable in discrete-time).

For generalized eigenvalue lambda = (ALPHAR + i*ALPHAI)/BETA:
  unstable when |lambda| = sqrt(ALPHAR^2 + ALPHAI^2)/|BETA| >= 1
  i.e., when DLAPY2(ALPHAR, ALPHAI) >= ABS(BETA)
"""

import numpy as np
import pytest


def test_sb02ov_unstable_real_outside_unit_circle():
    """
    Test unstable eigenvalue: |ALPHAR/BETA| >= 1.

    lambda = 2.0/1.0 = 2.0, |lambda| = 2.0 >= 1 -> unstable (should return True)
    """
    from slicot import sb02ov

    result = sb02ov(2.0, 0.0, 1.0)
    assert result is True


def test_sb02ov_stable_real_inside_unit_circle():
    """
    Test stable eigenvalue: |ALPHAR/BETA| < 1.

    lambda = 0.5/1.0 = 0.5, |lambda| = 0.5 < 1 -> stable (should return False)
    """
    from slicot import sb02ov

    result = sb02ov(0.5, 0.0, 1.0)
    assert result is False


def test_sb02ov_on_unit_circle():
    """
    Test eigenvalue on unit circle: |lambda| = 1.

    lambda = 1.0/1.0 = 1.0, |lambda| = 1.0 >= 1 -> unstable (should return True)
    """
    from slicot import sb02ov

    result = sb02ov(1.0, 0.0, 1.0)
    assert result is True


def test_sb02ov_complex_unstable():
    """
    Test complex unstable eigenvalue with |lambda| > 1.

    lambda = (0.8 + 0.8i)/1.0, |lambda| = sqrt(0.64 + 0.64) = sqrt(1.28) > 1 -> unstable
    """
    from slicot import sb02ov

    result = sb02ov(0.8, 0.8, 1.0)
    assert result is True


def test_sb02ov_complex_stable():
    """
    Test complex stable eigenvalue with |lambda| < 1.

    lambda = (0.5 + 0.5i)/1.0, |lambda| = sqrt(0.25 + 0.25) = sqrt(0.5) < 1 -> stable
    """
    from slicot import sb02ov

    result = sb02ov(0.5, 0.5, 1.0)
    assert result is False


def test_sb02ov_negative_beta():
    """
    Test with negative beta (only absolute value matters for modulus).

    lambda = 0.5/(-2.0), |lambda| = 0.5/2.0 = 0.25 < 1 -> stable
    """
    from slicot import sb02ov

    result = sb02ov(0.5, 0.0, -2.0)
    assert result is False


def test_sb02ov_both_negative():
    """
    Test with negative ALPHAR and BETA.

    lambda = (-3.0)/(-2.0), |lambda| = 3.0/2.0 = 1.5 >= 1 -> unstable
    """
    from slicot import sb02ov

    result = sb02ov(-3.0, 0.0, -2.0)
    assert result is True


def test_sb02ov_mathematical_property():
    """
    Validate mathematical property: selection based on modulus.

    For any combination, unstable iff:
      sqrt(ALPHAR^2 + ALPHAI^2) >= abs(BETA)

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    from slicot import sb02ov

    for _ in range(30):
        alphar = np.random.uniform(-5, 5)
        alphai = np.random.uniform(-5, 5)
        beta = np.random.uniform(-5, 5)

        if abs(beta) < 0.01:
            continue

        result = sb02ov(alphar, alphai, beta)

        modulus_numerator = np.sqrt(alphar**2 + alphai**2)
        modulus_denominator = abs(beta)
        expected = modulus_numerator >= modulus_denominator

        assert result == expected, \
            f"sb02ov({alphar}, {alphai}, {beta}) = {result}, expected {expected}"

        assert result == (alphar**2 + alphai**2 >= beta**2), \
            f"modulus squared test failed for ({alphar}, {alphai}, {beta})"


def test_sb02ov_zero_eigenvalue():
    """
    Test zero eigenvalue: ALPHAR = ALPHAI = 0.

    lambda = 0, |lambda| = 0 < 1 -> stable (should return False)
    Except when BETA = 0 (undefined), but we test with BETA != 0.
    """
    from slicot import sb02ov

    result = sb02ov(0.0, 0.0, 1.0)
    assert result is False


def test_sb02ov_infinite_eigenvalue():
    """
    Test infinite eigenvalue: BETA = 0.

    When BETA = 0 and ALPHAR or ALPHAI != 0, eigenvalue is infinite -> unstable.
    sqrt(1^2 + 0^2) = 1 >= abs(0) = 0 -> True
    """
    from slicot import sb02ov

    result = sb02ov(1.0, 0.0, 0.0)
    assert result is True
