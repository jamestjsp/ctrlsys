"""
Tests for SB01BX: Select eigenvalue(s) closest to a given value.

Selects a real eigenvalue or complex conjugate pair at minimal distance
to a given real or complex value. Reorders arrays so selected eigenvalue(s)
appear at the end.
"""
import numpy as np
import pytest
from ctrlsys import sb01bx


"""Basic functionality tests."""

def test_real_eigenvalue_selection():
    """
    Select closest real eigenvalue to target.
    """
    reig = True
    n = 5
    xr = 2.5  # Target value
    xi = 0.0  # Not used for real

    wr = np.array([1.0, 4.0, 2.0, 6.0, 3.0], order='F', dtype=float)
    wi = np.zeros(n, order='F', dtype=float)  # Not used

    wr_out, wi_out, s, p = sb01bx(reig, n, xr, xi, wr, wi)

    # Closest to 2.5 is 2.0 (distance 0.5) or 3.0 (distance 0.5)
    # Actually: |1-2.5|=1.5, |4-2.5|=1.5, |2-2.5|=0.5, |6-2.5|=3.5, |3-2.5|=0.5
    # First minimum found is at index 2 (value 2.0)
    assert s == 2.0
    assert p == 2.0  # For real, p = s
    assert wr_out[-1] == 2.0  # Selected moved to end

def test_complex_eigenvalue_selection():
    """
    Select closest complex conjugate pair to target.
    """
    reig = False
    n = 4  # 2 complex pairs
    xr = 1.0
    xi = 2.0  # Target: 1 + 2i

    # Complex pairs: (0, +/-1), (2, +/-3)
    wr = np.array([0.0, 0.0, 2.0, 2.0], order='F', dtype=float)
    wi = np.array([1.0, -1.0, 3.0, -3.0], order='F', dtype=float)

    wr_out, wi_out, s, p = sb01bx(reig, n, xr, xi, wr, wi)

    # Distance to (0, 1): |0-1| + |1-2| = 1 + 1 = 2
    # Distance to (2, 3): |2-1| + |3-2| = 1 + 1 = 2
    # First minimum is (0, 1)
    np.testing.assert_allclose(s, 0.0 + 0.0, rtol=1e-14)  # Sum = 2*real = 0
    np.testing.assert_allclose(p, 0.0**2 + 1.0**2, rtol=1e-14)  # Product = real^2 + imag^2 = 1

    # Selected pair at end
    np.testing.assert_allclose(wr_out[-2:], [0.0, 0.0], rtol=1e-14)
    np.testing.assert_allclose(np.abs(wi_out[-2:]), [1.0, 1.0], rtol=1e-14)

def test_single_real_eigenvalue():
    """
    N=1 with real eigenvalue.
    """
    reig = True
    n = 1
    xr = 5.0
    xi = 0.0

    wr = np.array([3.0], order='F', dtype=float)
    wi = np.zeros(1, order='F', dtype=float)

    wr_out, wi_out, s, p = sb01bx(reig, n, xr, xi, wr, wi)

    assert s == 3.0
    assert p == 3.0
    assert wr_out[0] == 3.0


"""Edge case tests."""

def test_target_equals_eigenvalue():
    """
    Target exactly equals one eigenvalue.
    """
    reig = True
    n = 3
    xr = 2.0
    xi = 0.0

    wr = np.array([1.0, 2.0, 3.0], order='F', dtype=float)
    wi = np.zeros(n, order='F', dtype=float)

    wr_out, wi_out, s, p = sb01bx(reig, n, xr, xi, wr, wi)

    # Distance 0 to eigenvalue 2.0
    assert s == 2.0
    assert p == 2.0

def test_all_eigenvalues_equal():
    """
    All eigenvalues the same value.
    """
    reig = True
    n = 4
    xr = 0.0
    xi = 0.0

    wr = np.array([5.0, 5.0, 5.0, 5.0], order='F', dtype=float)
    wi = np.zeros(n, order='F', dtype=float)

    wr_out, wi_out, s, p = sb01bx(reig, n, xr, xi, wr, wi)

    # First one is selected (all have same distance)
    assert s == 5.0
    assert p == 5.0


"""Mathematical property validation tests."""

def test_sum_product_complex():
    """
    For complex pair a +/- bi: sum=2a, product=a^2+b^2.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    reig = False
    n = 2
    xr = 0.0
    xi = 0.0

    # Complex pair: 3 +/- 4i
    wr = np.array([3.0, 3.0], order='F', dtype=float)
    wi = np.array([4.0, -4.0], order='F', dtype=float)

    wr_out, wi_out, s, p = sb01bx(reig, n, xr, xi, wr, wi)

    # Sum = 2 * 3 = 6
    np.testing.assert_allclose(s, 6.0, rtol=1e-14)
    # Product = 3^2 + 4^2 = 25
    np.testing.assert_allclose(p, 25.0, rtol=1e-14)

def test_reordering_preserves_other_eigenvalues():
    """
    Eigenvalues not selected should still be in the array.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    reig = True
    n = 5
    xr = 10.0
    xi = 0.0

    wr = np.array([1.0, 5.0, 3.0, 7.0, 9.0], order='F', dtype=float)
    wi = np.zeros(n, order='F', dtype=float)

    wr_orig = wr.copy()

    wr_out, wi_out, s, p = sb01bx(reig, n, xr, xi, wr, wi)

    # 9.0 is closest to 10.0
    assert s == 9.0
    assert wr_out[-1] == 9.0

    # All original eigenvalues should be present (permutation)
    np.testing.assert_allclose(sorted(wr_out), sorted(wr_orig), rtol=1e-14)

def test_distance_minimization():
    """
    Verify the selected eigenvalue truly minimizes distance.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    reig = True
    n = 10
    xr = 0.0
    xi = 0.0

    wr = np.random.randn(n).astype(float, order='F')
    wi = np.zeros(n, order='F', dtype=float)

    # Find true minimum manually
    distances = np.abs(wr - xr)
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    min_val = wr[min_idx]

    wr_out, wi_out, s, p = sb01bx(reig, n, xr, xi, wr.copy(), wi.copy())

    # Selected should be at minimum distance
    np.testing.assert_allclose(np.abs(s - xr), min_dist, rtol=1e-14)
