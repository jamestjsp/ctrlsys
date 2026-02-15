"""
Tests for mc01xd - Compute roots of cubic polynomial.

MC01XD computes roots of P(t) = ALPHA + BETA*t + GAMMA*t^2 + DELTA*t^3.
Roots are returned as quotients (EVR + i*EVI) / EVQ where EVQ >= 0.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest


def test_mc01xd_simple_cubic():
    """
    Test with simple cubic x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3).

    Roots should be 1, 2, 3.
    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import mc01xd

    # P(t) = -6 + 11*t - 6*t^2 + 1*t^3 = (t-1)(t-2)(t-3)
    alpha, beta, gamma, delta = -6.0, 11.0, -6.0, 1.0

    evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

    assert info == 0

    # Compute actual roots where EVQ != 0
    roots = []
    for k in range(3):
        if evq[k] != 0:
            assert evq[k] >= 0, "EVQ must be non-negative"
            roots.append(complex(evr[k], evi[k]) / evq[k])

    # All roots should be real
    for r in roots:
        assert abs(r.imag) < 1e-14, f"Expected real root, got {r}"

    real_roots = sorted([r.real for r in roots])
    assert_allclose(real_roots, [1.0, 2.0, 3.0], rtol=1e-13)


def test_mc01xd_complex_roots():
    """
    Test with cubic having one real root and two complex conjugate roots.

    P(t) = (t - 1)(t^2 + 1) = t^3 - t^2 + t - 1
    Roots: 1, i, -i
    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import mc01xd

    # P(t) = -1 + 1*t - 1*t^2 + 1*t^3
    alpha, beta, gamma, delta = -1.0, 1.0, -1.0, 1.0

    evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

    assert info == 0

    # Compute roots
    roots = []
    for k in range(3):
        if evq[k] != 0:
            roots.append(complex(evr[k], evi[k]) / evq[k])

    # Should have root at 1 and complex conjugate pair at +/-i
    real_roots = [r for r in roots if abs(r.imag) < 1e-10]
    complex_roots = [r for r in roots if abs(r.imag) >= 1e-10]

    assert len(real_roots) == 1
    assert_allclose(real_roots[0].real, 1.0, rtol=1e-13)

    assert len(complex_roots) == 2
    # Complex roots should be conjugates
    imag_parts = sorted([r.imag for r in complex_roots])
    assert_allclose(imag_parts, [-1.0, 1.0], rtol=1e-13)
    # Real parts should be 0
    for r in complex_roots:
        assert abs(r.real) < 1e-13


def test_mc01xd_triple_root():
    """
    Test with triple root at t = 2.

    P(t) = (t - 2)^3 = t^3 - 6*t^2 + 12*t - 8
    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import mc01xd

    # P(t) = -8 + 12*t - 6*t^2 + 1*t^3
    alpha, beta, gamma, delta = -8.0, 12.0, -6.0, 1.0

    evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

    assert info == 0

    roots = []
    for k in range(3):
        if evq[k] != 0:
            roots.append(complex(evr[k], evi[k]) / evq[k])

    # All three roots should be real and equal to 2
    # Note: Triple roots are numerically ill-conditioned, so use relaxed tolerance
    for r in roots:
        assert abs(r.imag) < 1e-4, f"Expected real root, got {r}"
        assert_allclose(r.real, 2.0, rtol=1e-4)


def test_mc01xd_quadratic_delta_zero():
    """
    Test quadratic case when delta = 0.

    P(t) = 2 - 3*t + 1*t^2 = (t - 1)(t - 2)
    Roots: 1, 2, and one infinite root
    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import mc01xd

    # P(t) = 2 - 3*t + 1*t^2 + 0*t^3
    alpha, beta, gamma, delta = 2.0, -3.0, 1.0, 0.0

    evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

    assert info == 0

    # Count finite and infinite roots
    finite_roots = []
    infinite_count = 0
    for k in range(3):
        if evq[k] == 0:
            infinite_count += 1
        else:
            finite_roots.append(complex(evr[k], evi[k]) / evq[k])

    assert infinite_count == 1, "Should have one infinite root"
    assert len(finite_roots) == 2

    real_finite = sorted([r.real for r in finite_roots])
    assert_allclose(real_finite, [1.0, 2.0], rtol=1e-13)


def test_mc01xd_polynomial_evaluation():
    """
    Verify roots satisfy P(root) = 0 within numerical tolerance.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc01xd

    np.random.seed(42)
    alpha = np.random.randn() * 10
    beta = np.random.randn() * 10
    gamma = np.random.randn() * 10
    delta = np.random.randn() * 10

    evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

    assert info == 0

    # Evaluate P(t) at each finite root
    for k in range(3):
        if evq[k] != 0:
            t = complex(evr[k], evi[k]) / evq[k]
            p_t = alpha + beta * t + gamma * t**2 + delta * t**3
            assert abs(p_t) < 1e-10, f"P({t}) = {p_t} should be ~0"


def test_mc01xd_vieta_formulas():
    """
    Verify Vieta's formulas for sum and product of roots.

    For P(t) = delta*t^3 + gamma*t^2 + beta*t + alpha:
    - sum of roots = -gamma/delta
    - product of roots = -alpha/delta

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mc01xd

    np.random.seed(123)
    # Ensure delta is not too small to avoid numerical issues
    alpha = np.random.randn() * 5
    beta = np.random.randn() * 5
    gamma = np.random.randn() * 5
    delta = np.random.randn() * 5
    if abs(delta) < 0.5:
        delta = 1.0

    evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

    assert info == 0

    # Check all roots are finite
    for k in range(3):
        assert evq[k] != 0, "Expected all finite roots for non-zero delta"

    roots = [complex(evr[k], evi[k]) / evq[k] for k in range(3)]

    # Sum of roots = -gamma/delta
    expected_sum = -gamma / delta
    actual_sum = sum(roots)
    assert_allclose(actual_sum.real, expected_sum, rtol=1e-12)
    assert abs(actual_sum.imag) < 1e-12

    # Product of roots = -alpha/delta
    expected_prod = -alpha / delta
    actual_prod = roots[0] * roots[1] * roots[2]
    assert_allclose(actual_prod.real, expected_prod, rtol=1e-12)
    assert abs(actual_prod.imag) < 1e-12


def test_mc01xd_evq_nonnegative():
    """
    Verify EVQ values are always non-negative as documented.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mc01xd

    np.random.seed(456)

    for _ in range(10):
        alpha = np.random.randn() * 100
        beta = np.random.randn() * 100
        gamma = np.random.randn() * 100
        delta = np.random.randn() * 100

        evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

        assert info == 0
        for k in range(3):
            assert evq[k] >= 0, f"EVQ[{k}] = {evq[k]} should be >= 0"


def test_mc01xd_large_coefficient_variation():
    """
    Test case with large coefficient variation (VAR > 10).

    This triggers the QR algorithm path when I = 0 or 3.
    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import mc01xd

    # Large alpha, small others: triggers I = 0
    alpha, beta, gamma, delta = 1000.0, 1.0, 1.0, 0.001

    evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

    assert info == 0

    # Verify polynomial evaluation at each finite root
    for k in range(3):
        if evq[k] != 0:
            t = complex(evr[k], evi[k]) / evq[k]
            p_t = alpha + beta * t + gamma * t**2 + delta * t**3
            # Larger tolerance for ill-conditioned case
            assert abs(p_t) / max(abs(alpha), abs(delta) * abs(t)**3) < 1e-8


def test_mc01xd_small_coefficient_variation():
    """
    Test case with small coefficient variation (VAR <= 10).

    This triggers the QZ algorithm path.
    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import mc01xd

    # Similar magnitude coefficients
    alpha, beta, gamma, delta = 1.0, 2.0, 3.0, 4.0

    evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

    assert info == 0

    # Verify polynomial evaluation at each finite root
    for k in range(3):
        if evq[k] != 0:
            t = complex(evr[k], evi[k]) / evq[k]
            p_t = alpha + beta * t + gamma * t**2 + delta * t**3
            assert abs(p_t) < 1e-12


def test_mc01xd_linear_gamma_delta_zero():
    """
    Test linear case when gamma = delta = 0.

    P(t) = alpha + beta*t
    One finite root at -alpha/beta, two infinite roots.
    Random seed: N/A (deterministic test data)
    """
    from ctrlsys import mc01xd

    alpha, beta, gamma, delta = 6.0, 2.0, 0.0, 0.0

    evr, evi, evq, info = mc01xd(alpha, beta, gamma, delta)

    assert info == 0

    finite_roots = []
    infinite_count = 0
    for k in range(3):
        if evq[k] == 0:
            infinite_count += 1
        else:
            finite_roots.append(evr[k] / evq[k])

    assert infinite_count == 2, "Should have two infinite roots"
    assert len(finite_roots) == 1
    assert_allclose(finite_roots[0], -3.0, rtol=1e-13)  # -alpha/beta = -6/2 = -3
