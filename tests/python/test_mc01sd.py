"""
Tests for mc01sd - Scale polynomial coefficients for minimal variation.

MC01SD scales the coefficients of real polynomial P(x) such that the
coefficients of the scaled polynomial Q(x) = s*P(t*x) have minimal variation,
where s = BASE**S and t = BASE**T.
"""

import numpy as np
from ctrlsys import mc01sd


def test_mc01sd_html_example():
    """
    Test using HTML documentation example.

    Input: P(x) = 10 - 40.5x + 159.5x^2 + 0x^3 + 2560x^4 - 10236.5x^5
    Expected: s = BASE**(-3), t = BASE**(-2) where BASE = 2

    Scaled coefficients (from HTML doc):
        q[0] = 1.2500
        q[1] = -1.2656
        q[2] = 1.2461
        q[3] = 0.0000
        q[4] = 1.2500
        q[5] = -1.2496
    """
    p = np.array([10.0, -40.5, 159.5, 0.0, 2560.0, -10236.5], order='F', dtype=float)

    q, s, t, mant, e, info = mc01sd(p)

    assert info == 0

    assert s == -3
    assert t == -2

    q_expected = np.array([1.2500, -1.2656, 1.2461, 0.0000, 1.2500, -1.2496], dtype=float)
    np.testing.assert_allclose(q, q_expected, rtol=1e-3, atol=1e-4)


def test_mc01sd_constant_polynomial():
    """
    Test with constant polynomial P(x) = 5.

    A constant polynomial has no variation to minimize.
    The scaling should set q[0] in [1, BASE).
    """
    p = np.array([5.0], order='F', dtype=float)

    q, s, t, mant, e, info = mc01sd(p)

    assert info == 0

    base = 2.0
    reconstructed = (base ** (-s)) * q[0]
    np.testing.assert_allclose(reconstructed, 5.0, rtol=1e-14)


def test_mc01sd_linear_polynomial():
    """
    Test with linear polynomial P(x) = 1 + 2x.

    Random seed: N/A (deterministic input)
    """
    p = np.array([1.0, 2.0], order='F', dtype=float)

    q, s, t, mant, e, info = mc01sd(p)

    assert info == 0

    scaled_p0 = q[0] * (2.0 ** s)
    np.testing.assert_allclose(scaled_p0, 1.0, rtol=1e-14)

    q0_valid = 1.0 <= abs(q[0]) < 2.0 or q[0] == 0.0
    assert q0_valid, f"q[0]={q[0]} should satisfy 1 <= |q[0]| < 2"


def test_mc01sd_zero_polynomial():
    """
    Test error handling for zero polynomial.

    All coefficients zero should return info = 1.
    """
    p = np.array([0.0, 0.0, 0.0], order='F', dtype=float)

    q, s, t, mant, e, info = mc01sd(p)

    assert info == 1


def test_mc01sd_leading_zeros():
    """
    Test polynomial with leading zero coefficients.

    P(x) = 4 + 0x + 0x^2 should be treated like constant.
    """
    p = np.array([4.0, 0.0, 0.0], order='F', dtype=float)

    q, s, t, mant, e, info = mc01sd(p)

    assert info == 0

    base = 2.0
    reconstructed = (base ** (-s)) * q[0]
    np.testing.assert_allclose(reconstructed, 4.0, rtol=1e-14)


def test_mc01sd_output_arrays():
    """
    Validate output array dimensions and types.
    """
    p = np.array([10.0, -40.5, 159.5], order='F', dtype=float)

    q, s, t, mant, e, info = mc01sd(p)

    assert info == 0
    assert len(q) == 3
    assert len(mant) == 3
    assert len(e) == 3

    assert isinstance(s, int)
    assert isinstance(t, int)


def test_mc01sd_reconstruction_property():
    """
    Validate mathematical property: can recover original p[i] from q[i].

    The scaled polynomial Q(x) = s*P(t*x) means:
      q[i] = BASE^S * (BASE^T)^i * p[i]

    So original: p[i] = q[i] / (BASE^S * (BASE^T)^i)
    """
    p_orig = np.array([1.0, -2.0, 4.0, -8.0], order='F', dtype=float)
    p = p_orig.copy()

    q, s, t, mant, e, info = mc01sd(p)

    assert info == 0

    base = 2.0
    s_factor = base ** s
    t_factor = base ** t

    for i in range(len(p_orig)):
        if p_orig[i] != 0.0:
            recovered = q[i] / (s_factor * (t_factor ** i))
            np.testing.assert_allclose(recovered, p_orig[i], rtol=1e-14,
                err_msg=f"Mismatch at coefficient {i}")


def test_mc01sd_mantissa_exponent_relation():
    """
    Validate mantissa and exponent relation: q[i] = mant[i] * BASE^e[i].

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    p = np.array([10.0, -40.5, 159.5, 0.0, 2560.0, -10236.5], order='F', dtype=float)

    q, s, t, mant, e, info = mc01sd(p)

    assert info == 0

    base = 2.0
    for i in range(len(p)):
        if mant[i] != 0.0:
            reconstructed = mant[i] * (base ** e[i])
            np.testing.assert_allclose(q[i], reconstructed, rtol=1e-14,
                err_msg=f"q[{i}]={q[i]} != mant[{i}]*BASE^e[{i}]={reconstructed}")


def test_mc01sd_minimal_variation():
    """
    Validate that variation is minimized (heuristic check).

    The variation V = max(e[i]) - min(e[i]) for non-zero coefficients
    should be small compared to original coefficient range.
    """
    p = np.array([1e-10, 1.0, 1e10], order='F', dtype=float)

    q, s, t, mant, e, info = mc01sd(p)

    assert info == 0

    nonzero_e = [e[i] for i in range(len(p)) if mant[i] != 0.0]
    if len(nonzero_e) > 1:
        variation = max(nonzero_e) - min(nonzero_e)

        original_range = np.log2(1e20)
        assert variation < original_range, "Variation should be reduced"
