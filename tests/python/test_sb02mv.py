"""
Tests for SB02MV: Select stable eigenvalues for continuous-time Riccati.

SB02MV is a selector function for continuous-time algebraic Riccati equations.
Returns True for stable eigenvalues (real part < 0).

Mathematical definition:
- Continuous-time stability: eigenvalue is stable if Re(lambda) < 0
- SB02MV selects STABLE eigenvalues: Re(lambda) < 0
"""

import numpy as np
from numpy.testing import assert_equal


def test_sb02mv_stable_negative():
    """Negative real part eigenvalues are stable (return True)."""
    from ctrlsys import sb02mv

    result = sb02mv(-1.0, 0.0)
    assert_equal(result, True)

    result = sb02mv(-0.5, 2.0)
    assert_equal(result, True)

    result = sb02mv(-100.0, 50.0)
    assert_equal(result, True)


def test_sb02mv_unstable_positive():
    """Positive real part eigenvalues are unstable (return False)."""
    from ctrlsys import sb02mv

    result = sb02mv(1.0, 0.0)
    assert_equal(result, False)

    result = sb02mv(0.5, 2.0)
    assert_equal(result, False)

    result = sb02mv(100.0, -50.0)
    assert_equal(result, False)


def test_sb02mv_unstable_zero():
    """Zero real part eigenvalues are unstable (return False).

    Purely imaginary eigenvalues on stability boundary are NOT selected.
    """
    from ctrlsys import sb02mv

    result = sb02mv(0.0, 1.0)
    assert_equal(result, False)

    result = sb02mv(0.0, 0.0)
    assert_equal(result, False)

    result = sb02mv(0.0, -5.5)
    assert_equal(result, False)


def test_sb02mv_boundary_small():
    """Test small positive/negative values near zero.

    Ensures strict < 0 comparison works correctly.
    """
    from ctrlsys import sb02mv

    eps = np.finfo(float).eps

    result = sb02mv(-eps, 1.0)
    assert_equal(result, True)

    result = sb02mv(eps, 1.0)
    assert_equal(result, False)


def test_sb02mv_imaginary_symmetry():
    """
    Mathematical property: sign of imaginary part doesn't affect result.

    sb02mv(reig, ieig) == sb02mv(reig, -ieig) for all values.
    """
    from ctrlsys import sb02mv

    test_cases = [
        (0.0, 1.0),
        (0.5, 2.0),
        (-0.5, 3.0),
        (1e-10, 100.0),
        (-1e-10, 100.0),
    ]

    for reig, ieig in test_cases:
        result_pos = sb02mv(reig, ieig)
        result_neg = sb02mv(reig, -ieig)
        assert_equal(result_pos, result_neg,
                    f"Imaginary symmetry failed for reig={reig}, ieig={ieig}")


def test_sb02mv_complement_sb02mr():
    from ctrlsys import sb02mv, sb02mr

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
        result_mv = sb02mv(reig, ieig)
        result_mr = sb02mr(reig, ieig)
        assert_equal(result_mv != result_mr, True,
                     f"Complement failed for reig={reig}, ieig={ieig}")


def test_sb02mv_independent_criterion():
    from ctrlsys import sb02mv

    test_cases = [
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, -2.0),
        (-0.5, 2.0),
        (0.0, 0.0),
        (1e-15, 0.0),
        (-1e-15, 0.0),
        (-100.0, 50.0),
        (100.0, -50.0),
    ]

    for reig, ieig in test_cases:
        result = sb02mv(reig, ieig)
        expected = reig < 0.0
        assert_equal(result, expected,
                     f"Mismatch for reig={reig}: got {result}, expected {expected}")
