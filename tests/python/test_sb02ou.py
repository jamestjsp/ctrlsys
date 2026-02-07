"""
Tests for SB02OU - Select unstable eigenvalues for continuous-time Riccati (generalized).

This is a callback function for DGGES eigenvalue selection. Returns True for
generalized eigenvalues with positive real part (unstable in continuous-time).

For generalized eigenvalue lambda = ALPHAR/BETA (ALPHAI is imaginary part):
  unstable when Re(lambda) = ALPHAR/BETA > 0
  i.e., when (ALPHAR < 0 AND BETA < 0) OR (ALPHAR > 0 AND BETA > 0)
"""

import numpy as np
import pytest


def test_sb02ou_positive_real_positive_beta():
    """
    Test unstable eigenvalue: ALPHAR > 0, BETA > 0.

    lambda = ALPHAR/BETA > 0 -> unstable (should return True)
    """
    from slicot import sb02ou

    result = sb02ou(1.0, 0.0, 2.0)
    assert result is True


def test_sb02ou_negative_real_negative_beta():
    """
    Test unstable eigenvalue: ALPHAR < 0, BETA < 0.

    lambda = ALPHAR/BETA = (-1)/(-2) = 0.5 > 0 -> unstable (should return True)
    """
    from slicot import sb02ou

    result = sb02ou(-1.0, 0.0, -2.0)
    assert result is True


def test_sb02ou_positive_real_negative_beta():
    """
    Test stable eigenvalue: ALPHAR > 0, BETA < 0.

    lambda = ALPHAR/BETA = 1/(-2) = -0.5 < 0 -> stable (should return False)
    """
    from slicot import sb02ou

    result = sb02ou(1.0, 0.0, -2.0)
    assert result is False


def test_sb02ou_negative_real_positive_beta():
    """
    Test stable eigenvalue: ALPHAR < 0, BETA > 0.

    lambda = ALPHAR/BETA = (-1)/2 = -0.5 < 0 -> stable (should return False)
    """
    from slicot import sb02ou

    result = sb02ou(-1.0, 0.0, 2.0)
    assert result is False


def test_sb02ou_zero_real_part():
    """
    Test marginally stable: ALPHAR = 0.

    lambda = 0/BETA = 0 -> on imaginary axis (should return False)
    Neither (ALPHAR < 0 AND BETA < 0) nor (ALPHAR > 0 AND BETA > 0) holds.
    """
    from slicot import sb02ou

    result = sb02ou(0.0, 1.0, 2.0)
    assert result is False

    result = sb02ou(0.0, 1.0, -2.0)
    assert result is False


def test_sb02ou_complex_eigenvalue():
    """
    Test complex eigenvalue with positive real part.

    ALPHAR > 0, ALPHAI != 0, BETA > 0 -> Re(lambda) > 0 -> unstable
    """
    from slicot import sb02ou

    result = sb02ou(1.5, 2.0, 1.0)
    assert result is True


def test_sb02ou_complex_stable():
    """
    Test complex eigenvalue with negative real part.

    ALPHAR < 0, ALPHAI != 0, BETA > 0 -> Re(lambda) < 0 -> stable
    """
    from slicot import sb02ou

    result = sb02ou(-1.5, 2.0, 1.0)
    assert result is False


def test_sb02ou_mathematical_property():
    """
    Validate mathematical property: selection based on sign of ALPHAR/BETA.

    For any combination, unstable iff:
      (ALPHAR < 0 AND BETA < 0) OR (ALPHAR > 0 AND BETA > 0)

    This is equivalent to: ALPHAR * BETA > 0

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    from slicot import sb02ou

    for _ in range(20):
        alphar = np.random.uniform(-10, 10)
        alphai = np.random.uniform(-10, 10)
        beta = np.random.uniform(-10, 10)

        if abs(beta) < 0.01:
            continue

        result = sb02ou(alphar, alphai, beta)

        expected = (alphar < 0.0 and beta < 0.0) or (alphar > 0.0 and beta > 0.0)

        assert result == expected, \
            f"sb02ou({alphar}, {alphai}, {beta}) = {result}, expected {expected}"

        assert result == (alphar * beta > 0), \
            f"product sign test failed for ({alphar}, {beta})"
