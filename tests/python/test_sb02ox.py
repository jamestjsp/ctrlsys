"""
Tests for SB02OX - Select stable eigenvalues for discrete-time Riccati (generalized).

This is a callback function for DGGES eigenvalue selection. Returns True for
generalized eigenvalues with modulus less than 1 (stable in discrete-time).

For generalized eigenvalue lambda = (ALPHAR + i*ALPHAI) / BETA:
  stable when |lambda| = sqrt(ALPHAR^2 + ALPHAI^2) / |BETA| < 1
  i.e., when DLAPY2(ALPHAR, ALPHAI) < |BETA|
"""

import numpy as np


def test_sb02ox_stable_real_eigenvalue():
    """
    Test stable real eigenvalue: |lambda| = 0.5 < 1.

    ALPHAR = 1, ALPHAI = 0, BETA = 2 -> |lambda| = 1/2 = 0.5 < 1 -> stable
    """
    from slicot import sb02ox

    result = sb02ox(1.0, 0.0, 2.0)
    assert result is True


def test_sb02ox_stable_real_negative_beta():
    """
    Test stable real eigenvalue with negative BETA.

    ALPHAR = 1, ALPHAI = 0, BETA = -2 -> |lambda| = 1/2 = 0.5 < 1 -> stable
    """
    from slicot import sb02ox

    result = sb02ox(1.0, 0.0, -2.0)
    assert result is True


def test_sb02ox_unstable_real_eigenvalue():
    """
    Test unstable real eigenvalue: |lambda| = 2 > 1.

    ALPHAR = 2, ALPHAI = 0, BETA = 1 -> |lambda| = 2/1 = 2 > 1 -> unstable
    """
    from slicot import sb02ox

    result = sb02ox(2.0, 0.0, 1.0)
    assert result is False


def test_sb02ox_marginal_eigenvalue():
    """
    Test marginally stable eigenvalue: |lambda| = 1.

    ALPHAR = 1, ALPHAI = 0, BETA = 1 -> |lambda| = 1/1 = 1 -> not strictly stable
    """
    from slicot import sb02ox

    result = sb02ox(1.0, 0.0, 1.0)
    assert result is False


def test_sb02ox_stable_complex_eigenvalue():
    """
    Test stable complex eigenvalue.

    ALPHAR = 0.3, ALPHAI = 0.4, BETA = 1 -> |lambda| = 0.5 < 1 -> stable
    """
    from slicot import sb02ox

    result = sb02ox(0.3, 0.4, 1.0)
    assert result is True


def test_sb02ox_unstable_complex_eigenvalue():
    """
    Test unstable complex eigenvalue.

    ALPHAR = 3, ALPHAI = 4, BETA = 1 -> |lambda| = 5 > 1 -> unstable
    """
    from slicot import sb02ox

    result = sb02ox(3.0, 4.0, 1.0)
    assert result is False


def test_sb02ox_zero_eigenvalue():
    """
    Test zero eigenvalue (origin): |lambda| = 0 < 1 -> stable.

    ALPHAR = 0, ALPHAI = 0, BETA = 1 -> |lambda| = 0 < 1 -> stable
    """
    from slicot import sb02ox

    result = sb02ox(0.0, 0.0, 1.0)
    assert result is True


def test_sb02ox_purely_imaginary_stable():
    """
    Test stable purely imaginary eigenvalue.

    ALPHAR = 0, ALPHAI = 0.5, BETA = 1 -> |lambda| = 0.5 < 1 -> stable
    """
    from slicot import sb02ox

    result = sb02ox(0.0, 0.5, 1.0)
    assert result is True


def test_sb02ox_purely_imaginary_unstable():
    """
    Test unstable purely imaginary eigenvalue.

    ALPHAR = 0, ALPHAI = 2, BETA = 1 -> |lambda| = 2 > 1 -> unstable
    """
    from slicot import sb02ox

    result = sb02ox(0.0, 2.0, 1.0)
    assert result is False


def test_sb02ox_mathematical_property():
    """
    Validate mathematical property: stable iff DLAPY2(ALPHAR, ALPHAI) < |BETA|.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    from slicot import sb02ox

    for _ in range(20):
        alphar = np.random.uniform(-10, 10)
        alphai = np.random.uniform(-10, 10)
        beta = np.random.uniform(-10, 10)

        if abs(beta) < 0.01:
            continue

        result = sb02ox(alphar, alphai, beta)

        modulus = np.hypot(alphar, alphai)
        expected = modulus < abs(beta)

        assert result == expected, \
            f"sb02ox({alphar}, {alphai}, {beta}) = {result}, expected {expected}"

        assert result == (alphar**2 + alphai**2 < beta**2), \
            f"modulus squared test failed for ({alphar}, {alphai}, {beta})"


def test_sb02ox_sb02ov_complement():
    """
    Validate sb02ox and sb02ov are complements.

    For eigenvalues away from the unit circle (|lambda| != 1):
    - sb02ox returns True for stable (|lambda| < 1)
    - sb02ov returns True for unstable (|lambda| >= 1)

    For strictly inside or outside the unit circle, they should be opposite.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    from slicot import sb02ox, sb02ov

    for _ in range(20):
        alphar = np.random.uniform(-10, 10)
        alphai = np.random.uniform(-10, 10)
        beta = np.random.uniform(-10, 10)

        if abs(beta) < 0.01:
            continue

        modulus = np.hypot(alphar, alphai)
        if abs(modulus - abs(beta)) < 0.01:
            continue

        stable = sb02ox(alphar, alphai, beta)
        unstable = sb02ov(alphar, alphai, beta)

        assert stable != unstable, \
            f"For alphar={alphar}, alphai={alphai}, beta={beta}: " \
            f"stable={stable}, unstable={unstable} should be opposite"


def test_sb02ox_boundary_case_just_inside():
    """
    Test eigenvalue just inside unit circle.

    |lambda| = 0.999 < 1 -> stable
    """
    from slicot import sb02ox

    result = sb02ox(0.999, 0.0, 1.0)
    assert result is True


def test_sb02ox_boundary_case_just_outside():
    """
    Test eigenvalue just outside unit circle.

    |lambda| = 1.001 > 1 -> unstable
    """
    from slicot import sb02ox

    result = sb02ox(1.001, 0.0, 1.0)
    assert result is False
