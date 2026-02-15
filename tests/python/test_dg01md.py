"""Tests for DG01MD - Discrete Fourier Transform of complex signal."""

import numpy as np
import pytest
from ctrlsys import dg01md


"""Basic functionality tests from SLICOT HTML documentation."""

def test_forward_fft_doc_example():
    """
    Test forward FFT using HTML doc example.

    Input: N=8, INDI='D'
    XR = [-0.1862, 0.3948, 0.6788, 0.1861, 0.7254, 0.5815, 0.4904, -0.9599]
    XI = [0.1288, 0.0671, -0.2417, 0.8875, 0.9380, -0.2682, 0.9312, -0.3116]

    Expected output:
    XR = [1.9109, -1.9419, -1.4070, 2.2886, 1.5059, -2.2271, 0.1470, -1.7660]
    XI = [2.1311, -2.2867, -1.3728, -0.6883, 1.3815, 0.2915, 2.1274, -0.5533]
    """
    xr = np.array([-0.1862, 0.3948, 0.6788, 0.1861, 0.7254, 0.5815, 0.4904, -0.9599], dtype=float)
    xi = np.array([0.1288, 0.0671, -0.2417, 0.8875, 0.9380, -0.2682, 0.9312, -0.3116], dtype=float)

    xr_expected = np.array([1.9109, -1.9419, -1.4070, 2.2886, 1.5059, -2.2271, 0.1470, -1.7660], dtype=float)
    xi_expected = np.array([2.1311, -2.2867, -1.3728, -0.6883, 1.3815, 0.2915, 2.1274, -0.5533], dtype=float)

    xr_out, xi_out, info = dg01md('D', xr, xi)

    assert info == 0
    np.testing.assert_allclose(xr_out, xr_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(xi_out, xi_expected, rtol=1e-3, atol=1e-4)


"""Test inverse transform property: FFT then IFFT scales by N."""

def test_forward_inverse_scaling():
    """
    Validate FFT followed by IFFT gives N * original signal.

    This is the key mathematical property documented in SLICOT:
    "a discrete Fourier transform, followed by an inverse discrete
    Fourier transform, will result in a signal which is a factor N
    larger than the original input signal."

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 8
    xr_orig = np.random.randn(n)
    xi_orig = np.random.randn(n)
    xr = xr_orig.copy()
    xi = xi_orig.copy()

    xr_fft, xi_fft, info1 = dg01md('D', xr, xi)
    assert info1 == 0

    xr_ifft, xi_ifft, info2 = dg01md('I', xr_fft, xi_fft)
    assert info2 == 0

    np.testing.assert_allclose(xr_ifft, n * xr_orig, rtol=1e-14)
    np.testing.assert_allclose(xi_ifft, n * xi_orig, rtol=1e-14)

def test_inverse_forward_scaling():
    """
    Validate IFFT followed by FFT also gives N * original signal.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 16
    xr_orig = np.random.randn(n)
    xi_orig = np.random.randn(n)
    xr = xr_orig.copy()
    xi = xi_orig.copy()

    xr_ifft, xi_ifft, info1 = dg01md('I', xr, xi)
    assert info1 == 0

    xr_fft, xi_fft, info2 = dg01md('D', xr_ifft, xi_ifft)
    assert info2 == 0

    np.testing.assert_allclose(xr_fft, n * xr_orig, rtol=1e-14)
    np.testing.assert_allclose(xi_fft, n * xi_orig, rtol=1e-14)


"""Mathematical property tests for DFT."""

def test_linearity_forward():
    """
    Validate DFT linearity: DFT(a*x + b*y) = a*DFT(x) + b*DFT(y).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 8
    a, b = 2.5, -1.3

    xr1 = np.random.randn(n)
    xi1 = np.random.randn(n)
    xr2 = np.random.randn(n)
    xi2 = np.random.randn(n)

    xr_sum = a * xr1 + b * xr2
    xi_sum = a * xi1 + b * xi2

    xr1_fft, xi1_fft, _ = dg01md('D', xr1.copy(), xi1.copy())
    xr2_fft, xi2_fft, _ = dg01md('D', xr2.copy(), xi2.copy())
    xr_sum_fft, xi_sum_fft, _ = dg01md('D', xr_sum.copy(), xi_sum.copy())

    np.testing.assert_allclose(xr_sum_fft, a * xr1_fft + b * xr2_fft, rtol=1e-14)
    np.testing.assert_allclose(xi_sum_fft, a * xi1_fft + b * xi2_fft, rtol=1e-14)

def test_parseval_theorem():
    """
    Validate Parseval's theorem: sum|x|^2 = (1/N)*sum|X|^2.

    The energy in time domain equals energy in frequency domain (scaled).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 16

    xr = np.random.randn(n)
    xi = np.random.randn(n)

    energy_time = np.sum(xr**2 + xi**2)

    xr_fft, xi_fft, info = dg01md('D', xr.copy(), xi.copy())
    assert info == 0

    energy_freq = np.sum(xr_fft**2 + xi_fft**2)

    np.testing.assert_allclose(energy_time, energy_freq / n, rtol=1e-14)

def test_dc_component():
    """
    Validate DC component (k=0) is sum of all samples.

    DFT[0] = sum(z[i]) for i=0..N-1

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 8

    xr = np.random.randn(n)
    xi = np.random.randn(n)

    xr_fft, xi_fft, info = dg01md('D', xr.copy(), xi.copy())
    assert info == 0

    np.testing.assert_allclose(xr_fft[0], np.sum(xr), rtol=1e-14)
    np.testing.assert_allclose(xi_fft[0], np.sum(xi), rtol=1e-14)

def test_real_signal_conjugate_symmetry():
    """
    Validate conjugate symmetry for real-valued input.

    For real input: X[k] = conj(X[N-k]) for k=1..N-1

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 8

    xr = np.random.randn(n)
    xi = np.zeros(n)

    xr_fft, xi_fft, info = dg01md('D', xr.copy(), xi.copy())
    assert info == 0

    for k in range(1, n // 2):
        np.testing.assert_allclose(xr_fft[k], xr_fft[n - k], rtol=1e-14)
        np.testing.assert_allclose(xi_fft[k], -xi_fft[n - k], rtol=1e-14)


"""Edge case tests."""

def test_n_equals_2():
    """
    Test minimum valid N=2.

    For N=2: DFT gives X[0]=x[0]+x[1], X[1]=x[0]-x[1]
    """
    xr = np.array([1.0, 2.0], dtype=float)
    xi = np.array([0.5, -0.5], dtype=float)

    xr_fft, xi_fft, info = dg01md('D', xr, xi)

    assert info == 0
    np.testing.assert_allclose(xr_fft[0], 3.0, rtol=1e-14)
    np.testing.assert_allclose(xi_fft[0], 0.0, rtol=1e-14)
    np.testing.assert_allclose(xr_fft[1], -1.0, rtol=1e-14)
    np.testing.assert_allclose(xi_fft[1], 1.0, rtol=1e-14)

def test_n_equals_4():
    """
    Test N=4.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    xr = np.random.randn(4)
    xi = np.random.randn(4)
    xr_orig = xr.copy()
    xi_orig = xi.copy()

    xr_fft, xi_fft, info1 = dg01md('D', xr, xi)
    assert info1 == 0

    xr_back, xi_back, info2 = dg01md('I', xr_fft, xi_fft)
    assert info2 == 0

    np.testing.assert_allclose(xr_back, 4 * xr_orig, rtol=1e-14)
    np.testing.assert_allclose(xi_back, 4 * xi_orig, rtol=1e-14)

def test_large_n_power_of_2():
    """
    Test larger N (128) to verify algorithm scales.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 128
    xr = np.random.randn(n)
    xi = np.random.randn(n)
    xr_orig = xr.copy()
    xi_orig = xi.copy()

    xr_fft, xi_fft, info1 = dg01md('D', xr, xi)
    assert info1 == 0

    xr_back, xi_back, info2 = dg01md('I', xr_fft, xi_fft)
    assert info2 == 0

    np.testing.assert_allclose(xr_back, n * xr_orig, rtol=1e-12)
    np.testing.assert_allclose(xi_back, n * xi_orig, rtol=1e-12)

def test_zero_signal():
    """Test that zero signal gives zero output."""
    n = 8
    xr = np.zeros(n, dtype=float)
    xi = np.zeros(n, dtype=float)

    xr_fft, xi_fft, info = dg01md('D', xr, xi)

    assert info == 0
    np.testing.assert_allclose(xr_fft, np.zeros(n), atol=1e-15)
    np.testing.assert_allclose(xi_fft, np.zeros(n), atol=1e-15)

def test_impulse_response():
    """
    Test impulse at position 0 gives constant spectrum.

    delta[0]=1, delta[k]=0 for k>0 => DFT = [1, 1, 1, ..., 1]
    """
    n = 8
    xr = np.zeros(n, dtype=float)
    xi = np.zeros(n, dtype=float)
    xr[0] = 1.0

    xr_fft, xi_fft, info = dg01md('D', xr, xi)

    assert info == 0
    np.testing.assert_allclose(xr_fft, np.ones(n), rtol=1e-14)
    np.testing.assert_allclose(xi_fft, np.zeros(n), atol=1e-15)


"""Error handling tests."""

def test_invalid_indi():
    """Test with invalid INDI parameter."""
    xr = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    xi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    xr_out, xi_out, info = dg01md('X', xr, xi)
    assert info == -1

def test_n_not_power_of_2():
    """Test with N not a power of 2."""
    xr = np.array([1.0, 2.0, 3.0], dtype=float)
    xi = np.array([0.0, 0.0, 0.0], dtype=float)
    xr_out, xi_out, info = dg01md('D', xr, xi)
    assert info == -2

def test_n_equals_1():
    """Test with N=1 (less than minimum 2)."""
    xr = np.array([1.0], dtype=float)
    xi = np.array([0.0], dtype=float)
    xr_out, xi_out, info = dg01md('D', xr, xi)
    assert info == -2

def test_lowercase_d():
    """Test that lowercase 'd' works for forward FFT."""
    xr = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    xi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    xr_out, xi_out, info = dg01md('d', xr, xi)
    assert info == 0

def test_lowercase_i():
    """Test that lowercase 'i' works for inverse FFT."""
    xr = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    xi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    xr_out, xi_out, info = dg01md('i', xr, xi)
    assert info == 0
