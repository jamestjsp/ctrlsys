"""
Tests for SB02MR: Select unstable eigenvalues for continuous-time Riccati.

SB02MR is a selector function for continuous-time algebraic Riccati equations.
Returns True for unstable eigenvalues (real part >= 0).

Mathematical definition:
- Continuous-time stability: eigenvalue is stable if Re(lambda) < 0
- SB02MR selects UNSTABLE eigenvalues: Re(lambda) >= 0
"""

import numpy as np
from numpy.testing import assert_equal


def test_sb02mr_unstable_positive():
    """Positive real part eigenvalues are unstable (return True)."""
    from slicot import sb02mr

    result = sb02mr(1.0, 0.0)
    assert_equal(result, True)

    result = sb02mr(0.5, 2.0)
    assert_equal(result, True)

    result = sb02mr(100.0, -50.0)
    assert_equal(result, True)


def test_sb02mr_unstable_zero():
    """Zero real part eigenvalues are unstable (return True).

    Purely imaginary eigenvalues on stability boundary are selected.
    """
    from slicot import sb02mr

    result = sb02mr(0.0, 1.0)
    assert_equal(result, True)

    result = sb02mr(0.0, 0.0)
    assert_equal(result, True)

    result = sb02mr(0.0, -5.5)
    assert_equal(result, True)


def test_sb02mr_stable_negative():
    """Negative real part eigenvalues are stable (return False)."""
    from slicot import sb02mr

    result = sb02mr(-1.0, 0.0)
    assert_equal(result, False)

    result = sb02mr(-0.5, 2.0)
    assert_equal(result, False)

    result = sb02mr(-100.0, 50.0)
    assert_equal(result, False)


def test_sb02mr_boundary_small():
    """Test small positive/negative values near zero.

    Ensures strict >= 0 comparison works correctly.
    """
    from slicot import sb02mr

    eps = np.finfo(float).eps

    result = sb02mr(eps, 1.0)
    assert_equal(result, True)

    result = sb02mr(-eps, 1.0)
    assert_equal(result, False)


def test_sb02mr_imaginary_symmetry():
    """
    Mathematical property: sign of imaginary part doesn't affect result.

    sb02mr(reig, ieig) == sb02mr(reig, -ieig) for all values.
    """
    from slicot import sb02mr

    test_cases = [
        (0.0, 1.0),
        (0.5, 2.0),
        (-0.5, 3.0),
        (1e-10, 100.0),
        (-1e-10, 100.0),
    ]

    for reig, ieig in test_cases:
        result_pos = sb02mr(reig, ieig)
        result_neg = sb02mr(reig, -ieig)
        assert_equal(result_pos, result_neg,
                    f"Imaginary symmetry failed for reig={reig}, ieig={ieig}")


def test_sb02mr_complement_sb02mv():
    from slicot import sb02mr, sb02mv

    test_cases = [
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, -2.0),
        (-0.5, 2.0),
        (0.0, 0.0),
        (1e-15, 0.0),
        (-1e-15, 0.0),
    ]

    for reig, ieig in test_cases:
        result_mr = sb02mr(reig, ieig)
        result_mv = sb02mv(reig, ieig)
        assert_equal(result_mr != result_mv, True,
                     f"Complement failed for reig={reig}, ieig={ieig}")


def test_sb02mr_independent_criterion():
    from slicot import sb02mr

    test_cases = [
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, -2.0),
        (-0.5, 2.0),
        (0.0, 0.0),
        (1e-15, 0.0),
        (-1e-15, 0.0),
        (100.0, -50.0),
        (-100.0, 50.0),
    ]

    for reig, ieig in test_cases:
        result = sb02mr(reig, ieig)
        expected = reig >= 0.0
        assert_equal(result, expected,
                     f"Mismatch for reig={reig}: got {result}, expected {expected}")
