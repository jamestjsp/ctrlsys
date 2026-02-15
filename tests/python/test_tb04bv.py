"""
Tests for TB04BV: Separate strictly proper part from constant part of transfer function matrix.

TB04BV separates a proper transfer function matrix G into:
- G0: strictly proper part (degree(num) < degree(den))
- D: constant part (feedthrough matrix)

Such that G = G0 + D.

The routine handles both increasing and decreasing polynomial orderings.
"""

import numpy as np
import pytest
from ctrlsys import tb04bv


def test_tb04bv_basic_increasing_order():
    """
    Test TB04BV with increasing order polynomial storage.

    SISO transfer function: G(s) = (s + 5) / (s + 3)

    This is a proper transfer function with degree(num) = degree(den) = 1.

    G(s) = (s + 5)/(s + 3) = 1 + 2/(s + 3) = D + G0(s)

    So D = 1 and G0(s) = 2/(s + 3).

    Polynomial storage (increasing order):
    - Numerator: [5, 1] (5 + 1*s)
    - Denominator: [3, 1] (3 + 1*s)
    """
    p, m, md = 1, 1, 2

    ign = np.array([[1]], order='F', dtype=np.int32)
    igd = np.array([[1]], order='F', dtype=np.int32)

    gn = np.array([5.0, 1.0], order='F', dtype=float)
    gd = np.array([3.0, 1.0], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd)

    assert info == 0

    np.testing.assert_allclose(d[0, 0], 1.0, rtol=1e-14)

    assert ign_out[0, 0] == 0
    np.testing.assert_allclose(gn_out[0], 2.0, rtol=1e-14)


def test_tb04bv_basic_decreasing_order():
    """
    Test TB04BV with decreasing order polynomial storage.

    Same SISO transfer function: G(s) = (s + 5) / (s + 3)

    Polynomial storage (decreasing order):
    - Numerator: [1, 5] (1*s + 5)
    - Denominator: [1, 3] (1*s + 3)
    """
    p, m, md = 1, 1, 2

    ign = np.array([[1]], order='F', dtype=np.int32)
    igd = np.array([[1]], order='F', dtype=np.int32)

    gn = np.array([1.0, 5.0], order='F', dtype=float)
    gd = np.array([1.0, 3.0], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('D', p, m, md, ign, igd, gn, gd)

    assert info == 0

    np.testing.assert_allclose(d[0, 0], 1.0, rtol=1e-14)

    assert ign_out[0, 0] == 0
    np.testing.assert_allclose(gn_out[0], 2.0, rtol=1e-14)


def test_tb04bv_strictly_proper():
    """
    Test TB04BV with strictly proper transfer function (deg(num) < deg(den)).

    G(s) = 1 / (s + 2)  ->  D = 0, G0 = G

    Polynomial storage (increasing):
    - Numerator: [1, 0] (constant 1, degree 0)
    - Denominator: [2, 1] (2 + s, degree 1)
    """
    p, m, md = 1, 1, 2

    ign = np.array([[0]], order='F', dtype=np.int32)
    igd = np.array([[1]], order='F', dtype=np.int32)

    gn = np.array([1.0, 0.0], order='F', dtype=float)
    gd = np.array([2.0, 1.0], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd)

    assert info == 0
    np.testing.assert_allclose(d[0, 0], 0.0, atol=1e-14)
    assert ign_out[0, 0] == 0


def test_tb04bv_mimo_2x2():
    """
    Test TB04BV with 2x2 MIMO transfer function matrix.

    G = [G11 G12]  where:
        [G21 G22]

    G11 = (s+1)/(s+2) = 1 + (-1)/(s+2) -> D11=1, strictly proper residue
    G12 = 1/(s+2)                      -> D12=0
    G21 = 2/(s+3)                      -> D21=0
    G22 = (2s+5)/(s+3) = 2 + (-1)/(s+3) -> D22=2, strictly proper residue

    Polynomial storage (increasing order, column-wise):
    Each polynomial uses MD=2 locations.
    Index: ((j-1)*P + i-1)*MD for (i,j)

    (1,1): index 0  -> num=[1,1], den=[2,1]
    (2,1): index 2  -> num=[2,0], den=[3,1]
    (1,2): index 4  -> num=[1,0], den=[2,1]
    (2,2): index 6  -> num=[5,2], den=[3,1]
    """
    p, m, md = 2, 2, 2

    ign = np.array([[1, 0], [0, 1]], order='F', dtype=np.int32)
    igd = np.array([[1, 1], [1, 1]], order='F', dtype=np.int32)

    gn = np.array([
        1.0, 1.0,
        2.0, 0.0,
        1.0, 0.0,
        5.0, 2.0
    ], order='F', dtype=float)

    gd = np.array([
        2.0, 1.0,
        3.0, 1.0,
        2.0, 1.0,
        3.0, 1.0
    ], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd)

    assert info == 0

    np.testing.assert_allclose(d[0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(d[1, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(d[0, 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(d[1, 1], 2.0, rtol=1e-14)


def test_tb04bv_higher_degree():
    """
    Test TB04BV with higher degree polynomials.

    G(s) = (s^2 + 3s + 2) / (s^2 + 5s + 6) = 1 + (-2s - 4)/(s^2 + 5s + 6)

    Polynomial storage (increasing): coeffs of 1, s, s^2
    - Numerator: [2, 3, 1]
    - Denominator: [6, 5, 1]

    Leading coeff ratio: 1/1 = 1
    After subtraction: num - 1*den = [2-6, 3-5, 1-1] = [-4, -2, 0]
    New numerator: [-4, -2] with degree 1
    """
    p, m, md = 1, 1, 3

    ign = np.array([[2]], order='F', dtype=np.int32)
    igd = np.array([[2]], order='F', dtype=np.int32)

    gn = np.array([2.0, 3.0, 1.0], order='F', dtype=float)
    gd = np.array([6.0, 5.0, 1.0], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd)

    assert info == 0
    np.testing.assert_allclose(d[0, 0], 1.0, rtol=1e-14)
    assert ign_out[0, 0] == 1
    np.testing.assert_allclose(gn_out[0], -4.0, rtol=1e-14)
    np.testing.assert_allclose(gn_out[1], -2.0, rtol=1e-14)


def test_tb04bv_improper_error():
    """
    Test TB04BV returns error for improper transfer function.

    Improper: degree(num) > degree(den)
    G(s) = s^2 / (s + 1) is improper
    """
    p, m, md = 1, 1, 3

    ign = np.array([[2]], order='F', dtype=np.int32)
    igd = np.array([[1]], order='F', dtype=np.int32)

    gn = np.array([0.0, 0.0, 1.0], order='F', dtype=float)
    gd = np.array([1.0, 1.0, 0.0], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd)

    assert info == 1


def test_tb04bv_null_denominator_error():
    """
    Test TB04BV returns error for null denominator.

    Leading coefficient of denominator is zero.
    """
    p, m, md = 1, 1, 2

    ign = np.array([[1]], order='F', dtype=np.int32)
    igd = np.array([[1]], order='F', dtype=np.int32)

    gn = np.array([1.0, 1.0], order='F', dtype=float)
    gd = np.array([1.0, 0.0], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd)

    assert info == 2


def test_tb04bv_invalid_order_error():
    """
    Test TB04BV raises ValueError for invalid ORDER parameter.
    """
    p, m, md = 1, 1, 2

    ign = np.array([[1]], order='F', dtype=np.int32)
    igd = np.array([[1]], order='F', dtype=np.int32)

    gn = np.array([1.0, 1.0], order='F', dtype=float)
    gd = np.array([1.0, 1.0], order='F', dtype=float)

    with pytest.raises(ValueError):
        tb04bv('X', p, m, md, ign, igd, gn, gd)


def test_tb04bv_zero_degree_proper():
    """
    Test TB04BV with zero-degree proper (constant over constant).

    G(s) = 5/2 is proper (both degree 0).
    D = 5/2 = 2.5, G0 = 0.
    """
    p, m, md = 1, 1, 1

    ign = np.array([[0]], order='F', dtype=np.int32)
    igd = np.array([[0]], order='F', dtype=np.int32)

    gn = np.array([5.0], order='F', dtype=float)
    gd = np.array([2.0], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd)

    assert info == 0
    np.testing.assert_allclose(d[0, 0], 2.5, rtol=1e-14)


def test_tb04bv_transfer_function_property():
    """
    Mathematical property test: G = G0 + D at complex frequency s.

    For s = j*omega, verify that:
    Original G(s) = (strictly proper G0(s)) + D

    Random seed: 42 (for reproducibility)
    """
    p, m, md = 1, 1, 3

    ign_in = np.array([[2]], order='F', dtype=np.int32)
    igd = np.array([[2]], order='F', dtype=np.int32)

    gn_in = np.array([2.0, 3.0, 1.0], order='F', dtype=float)
    gd_coeff = np.array([6.0, 5.0, 1.0], order='F', dtype=float)

    gn_work = gn_in.copy()
    ign_work = ign_in.copy()

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign_work, igd, gn_work, gd_coeff)

    assert info == 0

    for omega in [0.1, 1.0, 10.0]:
        s = 1j * omega

        g_orig = np.polyval(gn_in[::-1], s) / np.polyval(gd_coeff[::-1], s)

        new_deg = ign_out[0, 0]
        g0 = np.polyval(gn_out[:new_deg+1][::-1], s) / np.polyval(gd_coeff[::-1], s)

        g_reconstructed = g0 + d[0, 0]

        np.testing.assert_allclose(g_reconstructed, g_orig, rtol=1e-10)


def test_tb04bv_quick_return_p_zero():
    """
    Test TB04BV quick return when P=0.
    """
    p, m, md = 0, 2, 2

    ign = np.zeros((1, 2), order='F', dtype=np.int32)
    igd = np.zeros((1, 2), order='F', dtype=np.int32)

    gn = np.zeros(4, order='F', dtype=float)
    gd = np.zeros(4, order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd)

    assert info == 0


def test_tb04bv_quick_return_m_zero():
    """
    Test TB04BV quick return when M=0.
    """
    p, m, md = 2, 0, 2

    ign = np.zeros((2, 1), order='F', dtype=np.int32)
    igd = np.zeros((2, 1), order='F', dtype=np.int32)

    gn = np.zeros(4, order='F', dtype=float)
    gd = np.zeros(4, order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd)

    assert info == 0


def test_tb04bv_tolerance_default():
    """
    Test TB04BV with default tolerance (TOL <= 0).

    Creates a case where the strictly proper numerator has leading
    coefficients that are nearly zero (within machine precision tolerance).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    p, m, md = 1, 1, 3

    ign = np.array([[2]], order='F', dtype=np.int32)
    igd = np.array([[2]], order='F', dtype=np.int32)

    gn = np.array([6.0, 5.0, 1.0], order='F', dtype=float)
    gd = np.array([6.0, 5.0, 1.0], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd, -1.0)

    assert info == 0
    np.testing.assert_allclose(d[0, 0], 1.0, rtol=1e-14)


def test_tb04bv_tolerance_explicit():
    """
    Test TB04BV with explicit tolerance.

    Small leading coefficient should be considered negligible.
    """
    p, m, md = 1, 1, 3

    ign = np.array([[2]], order='F', dtype=np.int32)
    igd = np.array([[2]], order='F', dtype=np.int32)

    gn = np.array([6.0, 5.0 + 1e-10, 1.0], order='F', dtype=float)
    gd = np.array([6.0, 5.0, 1.0], order='F', dtype=float)

    ign_out, gn_out, d, info = tb04bv('I', p, m, md, ign, igd, gn, gd, 1e-8)

    assert info == 0
    np.testing.assert_allclose(d[0, 0], 1.0, rtol=1e-8)
