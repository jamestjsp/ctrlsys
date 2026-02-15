"""
Tests for MC01TD - Polynomial stability test.

Determines whether a polynomial P(x) with real coefficients is stable:
- Continuous-time (DICO='C'): all zeros in left half-plane
- Discrete-time (DICO='D'): all zeros inside unit circle

Uses Routh algorithm (continuous) or Schur-Cohn algorithm (discrete).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mc01td_continuous_unstable():
    """
    Test continuous-time stability using SLICOT HTML doc example.

    Polynomial: P(x) = 2 + 0*x + 1*x^2 - 1*x^3 + 1*x^4
    Expected: unstable, 2 zeros in right half-plane
    """
    from ctrlsys import mc01td

    p = np.array([2.0, 0.0, 1.0, -1.0, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert iwarn == 0
    assert dp_out == 4
    assert stable is False
    assert nz == 2


def test_mc01td_continuous_stable():
    """
    Test a stable continuous-time polynomial.

    P(x) = 1 + 2*x + x^2 = (x + 1)^2
    Roots: x = -1 (double root, left half-plane)
    Expected: stable, 0 zeros in right half-plane

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01td

    p = np.array([1.0, 2.0, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert iwarn == 0
    assert dp_out == 2
    assert stable is True
    assert nz == 0


def test_mc01td_discrete_stable():
    """
    Test a stable discrete-time polynomial.

    P(x) = 1 - 0.5*x = 0 => x = 2 (outside unit circle = stable zero)
    But we test all zeros inside unit circle for stability.

    Actually for discrete-time: stable means all zeros INSIDE unit circle.
    P(x) = 3 - 2*x has zero at x = 1.5 (outside unit circle).
    So P(x) = 3 - 2*x is UNSTABLE in discrete-time (1 zero outside).

    Let's use P(x) = 2 - x with zero at x = 2: unstable (outside circle).

    For stable: P(x) = 1 + 0.5*x has zero at x = -2: unstable.

    P(x) = 4 - x has zero at x = 4: unstable.

    Stable example: P(x) = 2 - x has zero at 2 (|2| > 1) = unstable.
    To be stable: P(x) where zeros have |z| < 1.
    P(x) = (x - 0.5) = -0.5 + x has zero at 0.5 (|0.5| < 1) = stable.

    P(x) = -0.5 + 1*x, coefficients: [-0.5, 1.0]
    """
    from ctrlsys import mc01td

    p = np.array([-0.5, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('D', p)

    assert info == 0
    assert iwarn == 0
    assert dp_out == 1
    assert stable is True
    assert nz == 0


def test_mc01td_discrete_unstable():
    """
    Test an unstable discrete-time polynomial.

    P(x) = (x - 2) = -2 + x has zero at x = 2 (|2| > 1) = unstable.
    Coefficients: [-2.0, 1.0]

    Expected: 1 zero outside unit circle.
    """
    from ctrlsys import mc01td

    p = np.array([-2.0, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('D', p)

    assert info == 0
    assert iwarn == 0
    assert dp_out == 1
    assert stable is False
    assert nz == 1


def test_mc01td_degree_reduction():
    """
    Test polynomial with trailing zeros (degree reduction).

    P(x) = 1 + 2*x + 0*x^2 + 0*x^3 (actual degree 1, leading zeros stripped)
    Should reduce to P(x) = 1 + 2*x with degree 1.
    Zero at x = -0.5 (left half-plane) = stable.
    """
    from ctrlsys import mc01td

    p = np.array([1.0, 2.0, 0.0, 0.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert iwarn == 2
    assert dp_out == 1
    assert stable is True
    assert nz == 0


def test_mc01td_zero_polynomial():
    """
    Test error handling for zero polynomial (all zeros).
    """
    from ctrlsys import mc01td

    p = np.array([0.0, 0.0, 0.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 1


def test_mc01td_constant_polynomial():
    """
    Test constant polynomial (degree 0).

    P(x) = 5 (no zeros) = stable.
    """
    from ctrlsys import mc01td

    p = np.array([5.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert iwarn == 0
    assert dp_out == 0
    assert stable is True
    assert nz == 0


def test_mc01td_higher_degree_continuous():
    """
    Test higher degree polynomial in continuous-time.

    P(x) = (x + 1)*(x + 2)*(x + 3) = 6 + 11*x + 6*x^2 + x^3
    All roots negative real, so stable.

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01td

    p = np.array([6.0, 11.0, 6.0, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert stable is True
    assert nz == 0


def test_mc01td_higher_degree_unstable():
    """
    Test higher degree polynomial with mixed stability.

    P(x) = (x - 1)*(x + 2)*(x + 3) = -6 - x + 4*x^2 + x^3
    Roots: 1, -2, -3. One positive real root = unstable.

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01td

    p = np.array([-6.0, -1.0, 4.0, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert stable is False
    assert nz == 1


def test_mc01td_discrete_higher_degree():
    """
    Test higher degree discrete-time polynomial.

    P(x) = (x - 0.5)*(x - 0.25) = 0.125 - 0.75*x + x^2
    Both roots inside unit circle = stable.

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01td

    p = np.array([0.125, -0.75, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('D', p)

    assert info == 0
    assert stable is True
    assert nz == 0


def test_mc01td_discrete_mixed_stability():
    """
    Test discrete polynomial with roots inside and outside unit circle.

    P(x) = (x - 0.3)(x - 3) = x^2 - 3.3x + 0.9
    Coefficients in increasing powers: [0.9, -3.3, 1.0]
    Roots: 0.3 (inside), 3 (outside) = 1 unstable zero.

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01td

    p = np.array([0.9, -3.3, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('D', p)

    assert info == 0
    assert stable is False
    assert nz == 1


def test_mc01td_complex_roots_continuous():
    """
    Test polynomial with complex conjugate roots.

    P(x) = x^2 + 2*x + 5 has roots at x = -1 +/- 2i (left half-plane).
    Coefficients: [5.0, 2.0, 1.0]
    Expected: stable.

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01td

    p = np.array([5.0, 2.0, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert stable is True
    assert nz == 0


def test_mc01td_complex_roots_unstable():
    """
    Test polynomial with complex conjugate roots in right half-plane.

    P(x) = x^2 - 2*x + 5 has roots at x = 1 +/- 2i (right half-plane).
    Coefficients: [5.0, -2.0, 1.0]
    Expected: unstable, 2 zeros in right half-plane.

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01td

    p = np.array([5.0, -2.0, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert stable is False
    assert nz == 2


def test_mc01td_numpy_roots_validation():
    """
    Validate MC01TD results against NumPy root finding.

    Generate polynomial, find roots with NumPy, verify MC01TD agrees.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc01td

    np.random.seed(42)

    roots = np.array([-1.0, -2.0, -3.0, -4.0])
    p = np.poly(roots)[::-1]
    p = np.asarray(p, order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert stable is True
    assert nz == 0


def test_mc01td_numpy_unstable_validation():
    """
    Validate MC01TD with polynomial having known unstable roots.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mc01td

    np.random.seed(123)

    roots = np.array([-1.0, 2.0, -3.0, 4.0])
    p = np.poly(roots)[::-1]
    p = np.asarray(p, order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 0
    assert stable is False
    assert nz == 2


def test_mc01td_error_invalid_dico():
    """
    Test error handling for invalid DICO parameter.
    """
    from ctrlsys import mc01td

    p = np.array([1.0, 2.0, 1.0], order='F', dtype=float)

    with pytest.raises(ValueError):
        mc01td('X', p)


def test_mc01td_imaginary_axis_root():
    """
    Test polynomial with root on imaginary axis (marginally stable).

    P(x) = x^2 + 1 has roots at x = +/- i (on imaginary axis).
    Coefficients: [1.0, 0.0, 1.0]

    The Routh algorithm should detect this as marginally unstable
    (INFO=2) since coefficients become zero during computation.

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01td

    p = np.array([1.0, 0.0, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('C', p)

    assert info == 2


def test_mc01td_unit_circle_root():
    """
    Test discrete polynomial with root on unit circle (marginally stable).

    P(x) = x - 1 has root at x = 1 (on unit circle).
    Coefficients: [-1.0, 1.0]

    The Schur-Cohn algorithm should flag this as potentially unstable.

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01td

    p = np.array([-1.0, 1.0], order='F', dtype=float)

    stable, nz, dp_out, iwarn, info = mc01td('D', p)

    assert info == 2
