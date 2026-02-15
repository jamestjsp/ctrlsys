"""
Tests for SG03BY: Complex Givens rotation in real arithmetic.

SG03BY computes parameters for the complex Givens rotation:

    (  CR-CI*I   SR-SI*I )   ( XR+XI*I )   ( Z )
    (                    ) * (         ) = (   )
    ( -SR-SI*I   CR+CI*I )   ( YR+YI*I )   ( 0 )

The routine avoids overflow using max-norm scaling.
"""

import numpy as np
import pytest
from ctrlsys import sg03by


def apply_givens_rotation(cr, ci, sr, si, xr, xi, yr, yi):
    """Apply complex Givens rotation and return result.

    G = [[ CR-CI*I,  SR-SI*I ],
         [-SR-SI*I,  CR+CI*I ]]

    v = [ XR+XI*I, YR+YI*I ]

    Returns G @ v
    """
    c = complex(cr, -ci)
    s = complex(sr, -si)
    x = complex(xr, xi)
    y = complex(yr, yi)

    z = c * x + s * y
    zero_elem = -s.conjugate() * x + c.conjugate() * y
    return z, zero_elem


class TestSG03BY:
    """Tests for SG03BY complex Givens rotation."""

    def test_basic_real_only(self):
        """Test with purely real input (no imaginary parts).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        xr, xi = 3.0, 0.0
        yr, yi = 4.0, 0.0

        cr, ci, sr, si, z = sg03by(xr, xi, yr, yi)

        # For real input, rotation should give z = sqrt(xr^2 + yr^2)
        expected_z = np.sqrt(xr**2 + yr**2)
        np.testing.assert_allclose(z, expected_z, rtol=1e-14)

        # ci and si should be zero for purely real input
        np.testing.assert_allclose(ci, 0.0, atol=1e-14)
        np.testing.assert_allclose(si, 0.0, atol=1e-14)

        # Apply rotation and verify result
        result_z, result_zero = apply_givens_rotation(cr, ci, sr, si, xr, xi, yr, yi)
        np.testing.assert_allclose(result_z.real, z, rtol=1e-14)
        np.testing.assert_allclose(result_z.imag, 0.0, atol=1e-14)
        np.testing.assert_allclose(abs(result_zero), 0.0, atol=1e-14)

    def test_complex_input(self):
        """Test with full complex input.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        xr, xi = 1.0, 2.0
        yr, yi = 3.0, 4.0

        cr, ci, sr, si, z = sg03by(xr, xi, yr, yi)

        # z should be >= 0
        assert z >= 0

        # z should equal norm: sqrt(|xr+xi*i|^2 + |yr+yi*i|^2)
        expected_z = np.sqrt(xr**2 + xi**2 + yr**2 + yi**2)
        np.testing.assert_allclose(z, expected_z, rtol=1e-14)

        # Apply rotation and verify mathematical property
        result_z, result_zero = apply_givens_rotation(cr, ci, sr, si, xr, xi, yr, yi)
        np.testing.assert_allclose(result_z.real, z, rtol=1e-14)
        np.testing.assert_allclose(result_z.imag, 0.0, atol=1e-14)
        np.testing.assert_allclose(abs(result_zero), 0.0, atol=1e-14)

    def test_zero_input(self):
        """Test with all zeros - edge case."""
        xr, xi = 0.0, 0.0
        yr, yi = 0.0, 0.0

        cr, ci, sr, si, z = sg03by(xr, xi, yr, yi)

        # With all zeros: CR=1, CI=0, SR=0, SI=0, Z=0
        assert z == 0.0
        assert cr == 1.0
        assert ci == 0.0
        assert sr == 0.0
        assert si == 0.0

    def test_imaginary_only(self):
        """Test with purely imaginary input.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        xr, xi = 0.0, 5.0
        yr, yi = 0.0, 12.0

        cr, ci, sr, si, z = sg03by(xr, xi, yr, yi)

        # z should be norm: sqrt(5^2 + 12^2) = 13
        expected_z = np.sqrt(xi**2 + yi**2)
        np.testing.assert_allclose(z, expected_z, rtol=1e-14)

        # cr and sr should be zero for purely imaginary input
        np.testing.assert_allclose(cr, 0.0, atol=1e-14)
        np.testing.assert_allclose(sr, 0.0, atol=1e-14)

        # Apply rotation and verify
        result_z, result_zero = apply_givens_rotation(cr, ci, sr, si, xr, xi, yr, yi)
        np.testing.assert_allclose(result_z.real, z, rtol=1e-14)
        np.testing.assert_allclose(result_z.imag, 0.0, atol=1e-14)
        np.testing.assert_allclose(abs(result_zero), 0.0, atol=1e-14)

    def test_overflow_prevention(self):
        """Test that max-norm scaling prevents overflow.

        Uses large values that would overflow without scaling.
        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        scale = 1e150
        xr, xi = 3.0 * scale, 4.0 * scale
        yr, yi = 5.0 * scale, 12.0 * scale

        cr, ci, sr, si, z = sg03by(xr, xi, yr, yi)

        # Should not overflow - z should be finite
        assert np.isfinite(z)
        assert np.isfinite(cr)
        assert np.isfinite(ci)
        assert np.isfinite(sr)
        assert np.isfinite(si)

        # z should equal norm (scaled)
        expected_z = np.sqrt(xr**2 + xi**2 + yr**2 + yi**2)
        np.testing.assert_allclose(z, expected_z, rtol=1e-10)

        # Verify rotation property
        result_z, result_zero = apply_givens_rotation(cr, ci, sr, si, xr, xi, yr, yi)
        np.testing.assert_allclose(result_z.real, z, rtol=1e-10)
        np.testing.assert_allclose(result_z.imag, 0.0, atol=z * 1e-14)
        np.testing.assert_allclose(abs(result_zero), 0.0, atol=z * 1e-14)

    def test_unitary_property(self):
        """Verify the Givens matrix is unitary (G* @ G = I).

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        xr, xi = 2.5, -1.3
        yr, yi = -0.7, 3.2

        cr, ci, sr, si, z = sg03by(xr, xi, yr, yi)

        # Build complex Givens matrix
        c = complex(cr, -ci)
        s = complex(sr, -si)
        G = np.array([
            [c, s],
            [-np.conj(s), np.conj(c)]
        ])

        # Check unitary: G* @ G = I
        GtG = np.conj(G.T) @ G
        np.testing.assert_allclose(GtG, np.eye(2), rtol=1e-14, atol=1e-14)

    def test_single_nonzero_x(self):
        """Test when only X is nonzero.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        xr, xi = 7.0, 11.0
        yr, yi = 0.0, 0.0

        cr, ci, sr, si, z = sg03by(xr, xi, yr, yi)

        # z should be |X| = sqrt(xr^2 + xi^2)
        expected_z = np.sqrt(xr**2 + xi**2)
        np.testing.assert_allclose(z, expected_z, rtol=1e-14)

        # S should be zero
        np.testing.assert_allclose(sr, 0.0, atol=1e-14)
        np.testing.assert_allclose(si, 0.0, atol=1e-14)

        # Apply rotation
        result_z, result_zero = apply_givens_rotation(cr, ci, sr, si, xr, xi, yr, yi)
        np.testing.assert_allclose(result_z.real, z, rtol=1e-14)
        np.testing.assert_allclose(result_z.imag, 0.0, atol=1e-14)
        np.testing.assert_allclose(abs(result_zero), 0.0, atol=1e-14)

    def test_single_nonzero_y(self):
        """Test when only Y is nonzero.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        xr, xi = 0.0, 0.0
        yr, yi = 8.0, 15.0

        cr, ci, sr, si, z = sg03by(xr, xi, yr, yi)

        # z should be |Y| = sqrt(yr^2 + yi^2)
        expected_z = np.sqrt(yr**2 + yi**2)
        np.testing.assert_allclose(z, expected_z, rtol=1e-14)

        # C should be zero
        np.testing.assert_allclose(cr, 0.0, atol=1e-14)
        np.testing.assert_allclose(ci, 0.0, atol=1e-14)

        # Apply rotation
        result_z, result_zero = apply_givens_rotation(cr, ci, sr, si, xr, xi, yr, yi)
        np.testing.assert_allclose(result_z.real, z, rtol=1e-14)
        np.testing.assert_allclose(result_z.imag, 0.0, atol=1e-14)
        np.testing.assert_allclose(abs(result_zero), 0.0, atol=1e-14)

    def test_negative_values(self):
        """Test with negative input values.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        xr, xi = -2.0, -3.0
        yr, yi = -5.0, -7.0

        cr, ci, sr, si, z = sg03by(xr, xi, yr, yi)

        # z should be non-negative norm
        expected_z = np.sqrt(xr**2 + xi**2 + yr**2 + yi**2)
        assert z >= 0
        np.testing.assert_allclose(z, expected_z, rtol=1e-14)

        # Apply rotation
        result_z, result_zero = apply_givens_rotation(cr, ci, sr, si, xr, xi, yr, yi)
        np.testing.assert_allclose(result_z.real, z, rtol=1e-14)
        np.testing.assert_allclose(result_z.imag, 0.0, atol=1e-14)
        np.testing.assert_allclose(abs(result_zero), 0.0, atol=1e-14)
