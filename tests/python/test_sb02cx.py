"""
Tests for SB02CX: Select purely imaginary eigenvalues.

SB02CX is a selector function used in computing the H-infinity norm.
It returns True for purely imaginary eigenvalues (|real part| < 100*eps).

Note: DLAMCH('Epsilon') returns 2^-53 (half of NumPy's eps which is 2^-52).
"""

import numpy as np
from numpy.testing import assert_equal

# LAPACK DLAMCH('Epsilon') = 2^-53, not NumPy's eps = 2^-52
DLAMCH_EPS = 2.0**-53
TOL = 100.0 * DLAMCH_EPS


def test_sb02cx_purely_imaginary():
    """
    Purely imaginary eigenvalue (reig=0) should return True.
    """
    from ctrlsys import sb02cx

    # Exactly zero real part
    result = sb02cx(0.0, 1.0)
    assert_equal(result, True)

    result = sb02cx(0.0, -5.5)
    assert_equal(result, True)

    result = sb02cx(0.0, 0.0)
    assert_equal(result, True)


def test_sb02cx_nearly_imaginary():
    """
    Eigenvalue with real part < 100*eps should return True.

    Uses DLAMCH('Epsilon') = 2^-53 for tolerance calculation.
    """
    from ctrlsys import sb02cx

    # Just below tolerance
    result = sb02cx(TOL * 0.5, 2.0)
    assert_equal(result, True)

    result = sb02cx(-TOL * 0.5, 2.0)
    assert_equal(result, True)


def test_sb02cx_not_imaginary():
    """
    Eigenvalue with |real part| >= 100*eps should return False.

    Uses DLAMCH('Epsilon') = 2^-53 for tolerance calculation.
    """
    from ctrlsys import sb02cx

    # At tolerance boundary (should be False since >= tol)
    result = sb02cx(TOL, 1.0)
    assert_equal(result, False)

    # Above tolerance
    result = sb02cx(1.0, 0.5)
    assert_equal(result, False)

    result = sb02cx(-0.1, 3.0)
    assert_equal(result, False)

    # Large real part
    result = sb02cx(100.0, 0.0)
    assert_equal(result, False)


def test_sb02cx_boundary():
    """
    Test behavior exactly at tolerance boundary.

    tol = 100 * DLAMCH('Epsilon') where DLAMCH('Epsilon') = 2^-53.
    |reig| < tol => True, |reig| >= tol => False
    """
    from ctrlsys import sb02cx

    eps = DLAMCH_EPS

    # Just below tolerance - should be True
    below = TOL * (1.0 - eps)
    result = sb02cx(below, 0.0)
    assert_equal(result, True)

    # Exactly at tolerance - should be False (uses < not <=)
    result = sb02cx(TOL, 0.0)
    assert_equal(result, False)

    # Just above tolerance - should be False
    above = TOL * (1.0 + eps)
    result = sb02cx(above, 0.0)
    assert_equal(result, False)


def test_sb02cx_involution():
    """
    Mathematical property: sign of imaginary part doesn't affect result.

    sb02cx(reig, ieig) == sb02cx(reig, -ieig) for all values.
    """
    from ctrlsys import sb02cx

    np.random.seed(42)

    test_cases = [
        (0.0, 1.0),
        (0.0, -1.0),
        (0.001, 5.0),
        (0.001, -5.0),
        (-0.001, 2.5),
        (-0.001, -2.5),
    ]

    for reig, ieig in test_cases:
        result_pos = sb02cx(reig, ieig)
        result_neg = sb02cx(reig, -ieig)
        assert_equal(result_pos, result_neg,
                    f"Sign symmetry failed for reig={reig}, ieig={ieig}")


def test_sb02cx_symmetry():
    from ctrlsys import sb02cx

    test_cases = [
        (0.001, 1.0),
        (0.1, 2.0),
        (1e-15, 0.5),
    ]

    for reig, ieig in test_cases:
        result_pos = sb02cx(reig, ieig)
        result_neg = sb02cx(-reig, ieig)
        assert_equal(result_pos, result_neg,
                    f"Symmetry failed for reig={reig}")


def test_sb02cx_independent_criterion():
    from ctrlsys import sb02cx

    tol = 100.0 * DLAMCH_EPS
    test_cases = [
        (0.0, 1.0),
        (0.0, 0.0),
        (TOL * 0.5, 2.0),
        (-TOL * 0.5, 2.0),
        (TOL, 1.0),
        (1.0, 0.5),
        (-0.1, 3.0),
        (100.0, 0.0),
        (1e-16, 7.0),
        (-1e-16, -3.0),
        (TOL * 0.99, 0.0),
        (TOL * 1.01, 0.0),
    ]

    for reig, ieig in test_cases:
        result = sb02cx(reig, ieig)
        expected = abs(reig) < tol
        assert_equal(result, expected,
                     f"Mismatch for reig={reig}: got {result}, expected {expected}")
