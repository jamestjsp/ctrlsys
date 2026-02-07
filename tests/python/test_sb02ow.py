"""
Tests for SB02OW - Select stable eigenvalues for continuous-time Riccati (generalized).

This is a callback function for DGGES eigenvalue selection. Returns True for
generalized eigenvalues with negative real part (stable in continuous-time).

For generalized eigenvalue lambda = ALPHAR/BETA (ALPHAI is imaginary part):
  stable when Re(lambda) = ALPHAR/BETA < 0
  i.e., when (ALPHAR < 0 AND BETA > 0) OR (ALPHAR > 0 AND BETA < 0)
"""

import numpy as np


def test_sb02ow_negative_real_positive_beta():
    """
    Test stable eigenvalue: ALPHAR < 0, BETA > 0.

    lambda = ALPHAR/BETA = (-1)/2 = -0.5 < 0 -> stable (should return True)
    """
    from slicot import sb02ow

    result = sb02ow(-1.0, 0.0, 2.0)
    assert result is True


def test_sb02ow_positive_real_negative_beta():
    """
    Test stable eigenvalue: ALPHAR > 0, BETA < 0.

    lambda = ALPHAR/BETA = 1/(-2) = -0.5 < 0 -> stable (should return True)
    """
    from slicot import sb02ow

    result = sb02ow(1.0, 0.0, -2.0)
    assert result is True


def test_sb02ow_positive_real_positive_beta():
    """
    Test unstable eigenvalue: ALPHAR > 0, BETA > 0.

    lambda = ALPHAR/BETA = 1/2 = 0.5 > 0 -> unstable (should return False)
    """
    from slicot import sb02ow

    result = sb02ow(1.0, 0.0, 2.0)
    assert result is False


def test_sb02ow_negative_real_negative_beta():
    """
    Test unstable eigenvalue: ALPHAR < 0, BETA < 0.

    lambda = ALPHAR/BETA = (-1)/(-2) = 0.5 > 0 -> unstable (should return False)
    """
    from slicot import sb02ow

    result = sb02ow(-1.0, 0.0, -2.0)
    assert result is False


def test_sb02ow_zero_real_part():
    """
    Test marginally stable: ALPHAR = 0.

    lambda = 0/BETA = 0 -> on imaginary axis (should return False)
    Neither (ALPHAR < 0 AND BETA > 0) nor (ALPHAR > 0 AND BETA < 0) holds.
    """
    from slicot import sb02ow

    result = sb02ow(0.0, 1.0, 2.0)
    assert result is False

    result = sb02ow(0.0, 1.0, -2.0)
    assert result is False


def test_sb02ow_complex_eigenvalue_stable():
    """
    Test complex eigenvalue with negative real part.

    ALPHAR < 0, ALPHAI != 0, BETA > 0 -> Re(lambda) < 0 -> stable
    """
    from slicot import sb02ow

    result = sb02ow(-1.5, 2.0, 1.0)
    assert result is True


def test_sb02ow_complex_eigenvalue_unstable():
    """
    Test complex eigenvalue with positive real part.

    ALPHAR > 0, ALPHAI != 0, BETA > 0 -> Re(lambda) > 0 -> unstable
    """
    from slicot import sb02ow

    result = sb02ow(1.5, 2.0, 1.0)
    assert result is False


def test_sb02ow_mathematical_property():
    """
    Validate mathematical property: selection based on sign of ALPHAR/BETA.

    For any combination, stable iff:
      (ALPHAR < 0 AND BETA > 0) OR (ALPHAR > 0 AND BETA < 0)

    This is equivalent to: ALPHAR * BETA < 0

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    from slicot import sb02ow

    for _ in range(20):
        alphar = np.random.uniform(-10, 10)
        alphai = np.random.uniform(-10, 10)
        beta = np.random.uniform(-10, 10)

        if abs(beta) < 0.01:
            continue

        result = sb02ow(alphar, alphai, beta)

        expected = (alphar < 0.0 and beta > 0.0) or (alphar > 0.0 and beta < 0.0)

        assert result == expected, \
            f"sb02ow({alphar}, {alphai}, {beta}) = {result}, expected {expected}"

        assert result == (alphar * beta < 0), \
            f"product sign test failed for ({alphar}, {beta})"


def test_sb02ow_sb02ou_complement():
    """
    Validate sb02ow and sb02ou are complements (for non-zero real part).

    For eigenvalues with non-zero real part:
    - sb02ow returns True for stable (Re(lambda) < 0)
    - sb02ou returns True for unstable (Re(lambda) > 0)

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    from slicot import sb02ow, sb02ou

    for _ in range(20):
        alphar = np.random.uniform(-10, 10)
        alphai = np.random.uniform(-10, 10)
        beta = np.random.uniform(-10, 10)

        if abs(alphar) < 0.01 or abs(beta) < 0.01:
            continue

        stable = sb02ow(alphar, alphai, beta)
        unstable = sb02ou(alphar, alphai, beta)

        assert stable != unstable, \
            f"For alphar={alphar}, beta={beta}: stable={stable}, unstable={unstable} should be opposite"
