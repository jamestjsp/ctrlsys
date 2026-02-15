"""
Tests for TD05AD: Evaluation of transfer function G(jW) at specified frequency.

TD05AD evaluates a complex valued rational transfer function G(jW) = B(jW)/A(jW)
for a specified frequency value. It can output either:
- Cartesian coordinates (real and imaginary parts)
- Polar coordinates (magnitude in dB and phase in degrees)
"""

import numpy as np
import pytest
from ctrlsys import td05ad


class TestTD05ADBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_html_example_cartesian(self):
        """
        Test case from TD05AD HTML documentation.

        Transfer function:
        G(jW) = (6 + 2*jW + 3*(jW)^2 + (jW)^3) /
                (1 + jW + 2*(jW)^4 + (jW)^5)

        At W = 1.0 rad/s, OUTPUT='C' (Cartesian):
        Expected: G(j*1.0) = 0.8462 - 0.2308j
        """
        np1 = 6  # Denominator order + 1
        mp1 = 4  # Numerator order + 1
        w = 1.0

        # Denominator coefficients [1, 1, 0, 0, 2, 1] (ascending powers of jW)
        a = np.array([1.0, 1.0, 0.0, 0.0, 2.0, 1.0], dtype=float)

        # Numerator coefficients [6, 2, 3, 1] (ascending powers of jW)
        b = np.array([6.0, 2.0, 3.0, 1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        np.testing.assert_allclose(valr, 0.8462, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(vali, -0.2308, rtol=1e-3, atol=1e-4)

    def test_html_example_polar(self):
        """
        Same transfer function as Cartesian, but output in polar form.

        G = 0.8462 - 0.2308j
        Magnitude = sqrt(0.8462^2 + 0.2308^2) = 0.8771
        Magnitude_dB = 20 * log10(0.8771) = -1.139 dB
        Phase = atan2(-0.2308, 0.8462) = -15.26 degrees
        """
        np1 = 6
        mp1 = 4
        w = 1.0

        a = np.array([1.0, 1.0, 0.0, 0.0, 2.0, 1.0], dtype=float)
        b = np.array([6.0, 2.0, 3.0, 1.0], dtype=float)

        valr, vali, info = td05ad('R', 'P', np1, mp1, w, a, b)

        assert info == 0
        # Expected magnitude in dB
        expected_mag_db = 20 * np.log10(np.sqrt(0.8462**2 + 0.2308**2))
        # Expected phase in degrees
        expected_phase = np.degrees(np.arctan2(-0.2308, 0.8462))

        np.testing.assert_allclose(valr, expected_mag_db, rtol=1e-2, atol=0.1)
        np.testing.assert_allclose(vali, expected_phase, rtol=1e-2, atol=0.5)

    def test_hertz_frequency_unit(self):
        """
        Test frequency input in Hertz instead of radians.

        Using UNITF='H', W is converted to W*2*pi internally.
        """
        np1 = 6
        mp1 = 4
        w_hz = 1.0 / (2 * np.pi)  # 1 rad/s = 1/(2*pi) Hz

        a = np.array([1.0, 1.0, 0.0, 0.0, 2.0, 1.0], dtype=float)
        b = np.array([6.0, 2.0, 3.0, 1.0], dtype=float)

        valr, vali, info = td05ad('H', 'C', np1, mp1, w_hz, a, b)

        assert info == 0
        np.testing.assert_allclose(valr, 0.8462, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(vali, -0.2308, rtol=1e-3, atol=1e-4)


class TestTD05ADMathematical:
    """Mathematical property validation tests."""

    def test_dc_gain(self):
        """
        Test DC gain (W = 0).

        At W = 0, G(0) = B(0)/A(0) = b[0]/a[0]
        This tests the constant term ratio.
        """
        np1 = 3  # A = [2, 1, 1]
        mp1 = 2  # B = [4, 2]
        w = 0.0

        a = np.array([2.0, 1.0, 1.0], dtype=float)
        b = np.array([4.0, 2.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        # At W=0: G(0) = B(0)/A(0) = 4/2 = 2 + 0j
        np.testing.assert_allclose(valr, 2.0, rtol=1e-14)
        np.testing.assert_allclose(vali, 0.0, atol=1e-14)

    def test_simple_first_order(self):
        """
        Test simple first-order transfer function: G(s) = 1/(1+s).

        At s = jW = j*1:
        G(j*1) = 1/(1+j) = (1-j)/(1+1) = 0.5 - 0.5j
        """
        np1 = 2  # A = [1, 1] -> 1 + jW
        mp1 = 1  # B = [1] -> 1
        w = 1.0

        a = np.array([1.0, 1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        np.testing.assert_allclose(valr, 0.5, rtol=1e-14)
        np.testing.assert_allclose(vali, -0.5, rtol=1e-14)

    def test_pure_integrator(self):
        """
        Test pure integrator: G(s) = 1/s.

        At s = jW = j*1:
        G(j*1) = 1/j = -j = 0 - 1j
        """
        np1 = 2  # A = [0, 1] -> jW
        mp1 = 1  # B = [1] -> 1
        w = 1.0

        a = np.array([0.0, 1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        np.testing.assert_allclose(valr, 0.0, atol=1e-14)
        np.testing.assert_allclose(vali, -1.0, rtol=1e-14)

    def test_pure_differentiator(self):
        """
        Test pure differentiator: G(s) = s/1 = s.

        At s = jW = j*2:
        G(j*2) = j*2 = 0 + 2j
        """
        np1 = 1  # A = [1] -> 1
        mp1 = 2  # B = [0, 1] -> jW
        w = 2.0

        a = np.array([1.0], dtype=float)
        b = np.array([0.0, 1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        np.testing.assert_allclose(valr, 0.0, atol=1e-14)
        np.testing.assert_allclose(vali, 2.0, rtol=1e-14)

    def test_second_order_system(self):
        """
        Test second-order system with manual calculation.

        G(s) = 1/(s^2 + 0.5*s + 1)
        A = [1, 0.5, 1] (ascending powers)
        B = [1]

        At s = j*1:
        Denominator = 1 + 0.5*j - 1 = 0.5*j
        G(j*1) = 1/(0.5*j) = -2j

        Random seed: 42 (not used - deterministic test)
        """
        np1 = 3  # A = [1, 0.5, 1]
        mp1 = 1  # B = [1]
        w = 1.0

        a = np.array([1.0, 0.5, 1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        # 1/(0.5j) = -2j = 0 - 2j
        np.testing.assert_allclose(valr, 0.0, atol=1e-14)
        np.testing.assert_allclose(vali, -2.0, rtol=1e-14)

    def test_polar_magnitude_phase_relationship(self):
        """
        Validate polar output matches manual calculation from Cartesian.

        G(jW) in Cartesian: (valr, vali)
        G(jW) in Polar: magnitude_dB = 20*log10(|G|), phase = atan(vali/valr)*180/pi

        Note: SLICOT uses atan(vali/valr) not atan2, so phase is in (-90, 90) range.
        This test uses a case where valr > 0 to avoid quadrant issues.
        """
        np1 = 2
        mp1 = 1
        w = 1.0

        # G(s) = 1/(1+s), at s=j -> G(j) = 1/(1+j) = 0.5 - 0.5j
        # valr > 0, so atan and atan2 agree
        a = np.array([1.0, 1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        # Get Cartesian result
        valr_c, vali_c, info_c = td05ad('R', 'C', np1, mp1, w, a, b)
        assert info_c == 0

        # Get Polar result
        valr_p, vali_p, info_p = td05ad('R', 'P', np1, mp1, w, a, b)
        assert info_p == 0

        # Compute expected polar from Cartesian
        magnitude = np.sqrt(valr_c**2 + vali_c**2)
        expected_db = 20 * np.log10(magnitude)
        # SLICOT uses atan(vali/valr), which equals atan2 when valr > 0
        expected_phase = np.degrees(np.arctan(vali_c / valr_c))

        np.testing.assert_allclose(valr_p, expected_db, rtol=1e-10)
        np.testing.assert_allclose(vali_p, expected_phase, rtol=1e-10, atol=1e-10)


class TestTD05ADEdgeCases:
    """Edge case and boundary condition tests."""

    def test_constant_numerator_denominator(self):
        """
        Test constant transfer function: G = 3/2.

        At any frequency, G(jW) = 3/2 = 1.5 + 0j
        """
        np1 = 1
        mp1 = 1
        w = 10.0

        a = np.array([2.0], dtype=float)
        b = np.array([3.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        np.testing.assert_allclose(valr, 1.5, rtol=1e-14)
        np.testing.assert_allclose(vali, 0.0, atol=1e-14)

    def test_zero_frequency_with_leading_zeros(self):
        """
        Test with numerator and denominator having leading zero coefficients.

        G(s) = s/(s + 1)
        At W=0: G(0) = 0
        """
        np1 = 2  # A = [1, 1]
        mp1 = 2  # B = [0, 1]
        w = 0.0

        a = np.array([1.0, 1.0], dtype=float)
        b = np.array([0.0, 1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        np.testing.assert_allclose(valr, 0.0, atol=1e-14)
        np.testing.assert_allclose(vali, 0.0, atol=1e-14)

    def test_negative_frequency(self):
        """
        Test with negative frequency value.

        For G(s) = 1/(1+s), at s = j*(-1) = -j:
        G(-j) = 1/(1-j) = (1+j)/2 = 0.5 + 0.5j
        """
        np1 = 2
        mp1 = 1
        w = -1.0

        a = np.array([1.0, 1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        np.testing.assert_allclose(valr, 0.5, rtol=1e-14)
        np.testing.assert_allclose(vali, 0.5, rtol=1e-14)

    def test_high_frequency(self):
        """
        Test at high frequency where higher-order terms dominate.

        G(s) = 1/(1 + s + s^2)
        At large W, G(jW) ~ 1/(jW)^2 = -1/W^2

        At W = 100:
        Expected magnitude ~ 1/W^2 = 1e-4
        """
        np1 = 3  # A = [1, 1, 1]
        mp1 = 1  # B = [1]
        w = 100.0

        a = np.array([1.0, 1.0, 1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 0
        # At high frequency, result should be small
        magnitude = np.sqrt(valr**2 + vali**2)
        assert magnitude < 1e-3


class TestTD05ADErrorHandling:
    """Error handling and validation tests."""

    def test_pole_at_frequency(self):
        """
        Test when frequency W is a pole of G(jW).

        G(s) = 1/s has a pole at s = 0.
        At W = 0, this should return INFO = 1.
        """
        np1 = 2  # A = [0, 1] -> jW
        mp1 = 1  # B = [1]
        w = 0.0

        a = np.array([0.0, 1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 1  # Pole at frequency

    def test_all_denominator_coefficients_zero(self):
        """
        Test when all denominator coefficients are zero.

        Should return INFO = 1.
        """
        np1 = 2
        mp1 = 1
        w = 1.0

        a = np.array([0.0, 0.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == 1

    def test_invalid_unitf(self):
        """Test invalid UNITF parameter."""
        np1 = 2
        mp1 = 1
        w = 1.0

        a = np.array([1.0, 1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('X', 'C', np1, mp1, w, a, b)

        assert info == -1

    def test_invalid_output(self):
        """Test invalid OUTPUT parameter."""
        np1 = 2
        mp1 = 1
        w = 1.0

        a = np.array([1.0, 1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'X', np1, mp1, w, a, b)

        assert info == -2

    def test_invalid_np1(self):
        """Test invalid NP1 parameter (< 1)."""
        np1 = 0
        mp1 = 1
        w = 1.0

        a = np.array([1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == -3

    def test_invalid_mp1(self):
        """Test invalid MP1 parameter (< 1)."""
        np1 = 1
        mp1 = 0
        w = 1.0

        a = np.array([1.0], dtype=float)
        b = np.array([1.0], dtype=float)

        valr, vali, info = td05ad('R', 'C', np1, mp1, w, a, b)

        assert info == -4
