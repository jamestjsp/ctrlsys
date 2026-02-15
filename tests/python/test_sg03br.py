import pytest
import numpy as np
from ctrlsys import sg03br


def test_sg03br_basic():
    """Test SG03BR with basic input values.

    SG03BR computes complex Givens rotation parameters such that:
    (    C      SR+SI*I )   ( XR+XI*I )   ( ZR+ZI*I )
    (                   ) * (         ) = (         )
    ( -SR+SI*I     C    )   ( YR+YI*I )   (    0    )

    where C**2 + |SR+SI*I|**2 = 1
    """
    xr, xi = 3.0, 4.0  # X = 3 + 4i
    yr, yi = 1.0, 2.0  # Y = 1 + 2i

    c, sr, si, zr, zi = sg03br(xr, xi, yr, yi)

    # Verify the rotation property: result should zero out second component
    # First row: C*(XR+XI*I) + (SR+SI*I)*(YR+YI*I) = ZR+ZI*I
    zr_computed = c * xr + sr * yr - si * yi
    zi_computed = c * xi + si * yr + sr * yi
    np.testing.assert_allclose(zr, zr_computed, rtol=1e-14)
    np.testing.assert_allclose(zi, zi_computed, rtol=1e-14)

    # Second row: (-SR+SI*I)*(XR+XI*I) + C*(YR+YI*I) should be zero
    result_r = -sr * xr - si * xi + c * yr
    result_i = si * xr - sr * xi + c * yi
    np.testing.assert_allclose(result_r, 0.0, atol=1e-14)
    np.testing.assert_allclose(result_i, 0.0, atol=1e-14)

    # Verify normalization: C**2 + SR**2 + SI**2 = 1
    norm = c**2 + sr**2 + si**2
    np.testing.assert_allclose(norm, 1.0, rtol=1e-14)


def test_sg03br_real_values():
    """Test SG03BR with purely real complex numbers."""
    xr, xi = 5.0, 0.0  # X = 5
    yr, yi = 3.0, 0.0  # Y = 3

    c, sr, si, zr, zi = sg03br(xr, xi, yr, yi)

    # Verify rotation zeroes out second component
    result_r = -sr * xr - si * xi + c * yr
    result_i = si * xr - sr * xi + c * yi
    np.testing.assert_allclose(result_r, 0.0, atol=1e-14)
    np.testing.assert_allclose(result_i, 0.0, atol=1e-14)

    # Verify normalization
    norm = c**2 + sr**2 + si**2
    np.testing.assert_allclose(norm, 1.0, rtol=1e-14)


def test_sg03br_y_zero():
    """Test SG03BR when Y is zero (edge case)."""
    xr, xi = 2.0, 3.0  # X = 2 + 3i
    yr, yi = 0.0, 0.0  # Y = 0

    c, sr, si, zr, zi = sg03br(xr, xi, yr, yi)

    # When Y=0, expect C=1, SR=SI=0, Z=X
    np.testing.assert_allclose(c, 1.0, rtol=1e-14)
    np.testing.assert_allclose(sr, 0.0, atol=1e-14)
    np.testing.assert_allclose(si, 0.0, atol=1e-14)
    np.testing.assert_allclose(zr, xr, rtol=1e-14)
    np.testing.assert_allclose(zi, xi, rtol=1e-14)


def test_sg03br_x_zero():
    """Test SG03BR when X is zero (edge case)."""
    xr, xi = 0.0, 0.0  # X = 0
    yr, yi = 4.0, 3.0  # Y = 4 + 3i

    c, sr, si, zr, zi = sg03br(xr, xi, yr, yi)

    # When X=0, expect C=0, and rotation gives Z = |Y|
    np.testing.assert_allclose(c, 0.0, atol=1e-14)

    # |Z| should equal |Y|
    z_mag = np.sqrt(zr**2 + zi**2)
    y_mag = np.sqrt(yr**2 + yi**2)
    np.testing.assert_allclose(z_mag, y_mag, rtol=1e-14)

    # Verify normalization
    norm = c**2 + sr**2 + si**2
    np.testing.assert_allclose(norm, 1.0, rtol=1e-14)


def test_sg03br_large_values():
    """Test SG03BR with large values (tests overflow handling)."""
    xr, xi = 1e150, 2e150
    yr, yi = 3e150, 4e150

    c, sr, si, zr, zi = sg03br(xr, xi, yr, yi)

    # Should not overflow, verify basic properties
    assert np.isfinite(c) and np.isfinite(sr) and np.isfinite(si)
    assert np.isfinite(zr) and np.isfinite(zi)

    # Verify normalization
    norm = c**2 + sr**2 + si**2
    np.testing.assert_allclose(norm, 1.0, rtol=1e-14)


def test_sg03br_small_values():
    """Test SG03BR with small values (tests underflow handling)."""
    xr, xi = 1e-150, 2e-150
    yr, yi = 3e-150, 4e-150

    c, sr, si, zr, zi = sg03br(xr, xi, yr, yi)

    # Should not underflow inappropriately
    assert np.isfinite(c) and np.isfinite(sr) and np.isfinite(si)

    # Verify normalization
    norm = c**2 + sr**2 + si**2
    np.testing.assert_allclose(norm, 1.0, rtol=1e-13)  # Slightly looser for tiny values
