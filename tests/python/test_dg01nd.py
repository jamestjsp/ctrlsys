"""Tests for DG01ND - Discrete Fourier Transform of real signal."""

import numpy as np
import pytest
from ctrlsys import dg01nd


"""Basic functionality tests from SLICOT HTML documentation."""

def test_forward_fft_doc_example():
    """
    Test forward FFT of real signal using HTML doc example.

    Input: N=8, INDI='D'
    Real signal A(1..16) split into odd/even parts:
    XR = odd samples = A(1), A(3), A(5), ..., A(15)
    XI = even samples = A(2), A(4), A(6), ..., A(16)

    From HTML doc:
    A = [-0.1862, 0.1288, 0.3948, 0.0671, 0.6788, -0.2417, 0.1861, 0.8875,
         0.7254, 0.9380, 0.5815, -0.2682, 0.4904, 0.9312, -0.9599, -0.3116]

    XR (odd) = [-0.1862, 0.3948, 0.6788, 0.1861, 0.7254, 0.5815, 0.4904, -0.9599]
    XI (even) = [0.1288, 0.0671, -0.2417, 0.8875, 0.9380, -0.2682, 0.9312, -0.3116]

    Expected output (N+1 = 9 components):
    XR = [4.0420, -3.1322, 0.1862, -2.1312, 1.5059, 2.1927, -1.4462, -0.5757, -0.2202]
    XI = [0.0000, -0.2421, -1.4675, -1.1707, -1.3815, -0.1908, 2.0327, 1.4914, 0.0000]
    """
    xr = np.array([-0.1862, 0.3948, 0.6788, 0.1861, 0.7254, 0.5815, 0.4904, -0.9599], dtype=float)
    xi = np.array([0.1288, 0.0671, -0.2417, 0.8875, 0.9380, -0.2682, 0.9312, -0.3116], dtype=float)

    xr_expected = np.array([4.0420, -3.1322, 0.1862, -2.1312, 1.5059, 2.1927, -1.4462, -0.5757, -0.2202], dtype=float)
    xi_expected = np.array([0.0000, -0.2421, -1.4675, -1.1707, -1.3815, -0.1908, 2.0327, 1.4914, 0.0000], dtype=float)

    xr_out, xi_out, info = dg01nd('D', xr, xi)

    assert info == 0
    np.testing.assert_allclose(xr_out, xr_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(xi_out, xi_expected, rtol=1e-3, atol=1e-4)


"""Test inverse transform property: FFT then IFFT scales by 2*N."""

def test_forward_inverse_scaling():
    """
    Validate FFT followed by IFFT gives 2*N * original signal.

    From HTML doc: "a discrete Fourier transform, followed by an inverse
    transform will result in a signal which is a factor 2*N larger
    than the original input signal."

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 8
    xr_orig = np.random.randn(n)
    xi_orig = np.random.randn(n)
    xr = xr_orig.copy()
    xi = xi_orig.copy()

    xr_fft, xi_fft, info1 = dg01nd('D', xr, xi)
    assert info1 == 0
    assert len(xr_fft) == n + 1
    assert len(xi_fft) == n + 1

    xr_ifft, xi_ifft, info2 = dg01nd('I', xr_fft, xi_fft)
    assert info2 == 0

    np.testing.assert_allclose(xr_ifft[:n], 2 * n * xr_orig, rtol=1e-13)
    np.testing.assert_allclose(xi_ifft[:n], 2 * n * xi_orig, rtol=1e-13)

def test_inverse_forward_scaling():
    """
    Validate IFFT followed by FFT also gives 2*N * original signal.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 16
    xr_orig = np.zeros(n + 1, dtype=float)
    xi_orig = np.zeros(n + 1, dtype=float)
    xr_orig[:n + 1] = np.random.randn(n + 1)
    xi_orig[:n + 1] = np.random.randn(n + 1)
    xr_orig[0] = abs(xr_orig[0])
    xr_orig[n] = abs(xr_orig[n])
    xi_orig[0] = 0.0
    xi_orig[n] = 0.0

    xr = xr_orig.copy()
    xi = xi_orig.copy()

    xr_ifft, xi_ifft, info1 = dg01nd('I', xr, xi)
    assert info1 == 0

    xr_fft, xi_fft, info2 = dg01nd('D', xr_ifft, xi_ifft)
    assert info2 == 0

    np.testing.assert_allclose(xr_fft, 2 * n * xr_orig, rtol=1e-13)
    np.testing.assert_allclose(xi_fft, 2 * n * xi_orig, rtol=1e-13)


"""Mathematical property tests for real signal DFT."""

def test_dc_component_sum():
    """
    Validate DC component (m=1) is sum of all 2*N real samples.

    For real signal A: FA(1) = sum(A(i)) for i=1..2*N

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 8
    xr = np.random.randn(n)
    xi = np.random.randn(n)

    total_sum = np.sum(xr) + np.sum(xi)

    xr_fft, xi_fft, info = dg01nd('D', xr.copy(), xi.copy())
    assert info == 0

    np.testing.assert_allclose(xr_fft[0], total_sum, rtol=1e-14)
    np.testing.assert_allclose(xi_fft[0], 0.0, atol=1e-14)

def test_nyquist_component_real():
    """
    Validate Nyquist frequency (m=N+1) is purely real.

    For real signal, FA(N+1) should have zero imaginary part.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 8
    xr = np.random.randn(n)
    xi = np.random.randn(n)

    xr_fft, xi_fft, info = dg01nd('D', xr.copy(), xi.copy())
    assert info == 0

    np.testing.assert_allclose(xi_fft[n], 0.0, atol=1e-14)

def test_parseval_energy_real_signal():
    """
    Validate Parseval's theorem for real signal FFT.

    Energy in time domain = (1/(2*N)) * energy in frequency domain.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 16

    xr = np.random.randn(n)
    xi = np.random.randn(n)

    time_energy = np.sum(xr**2) + np.sum(xi**2)

    xr_fft, xi_fft, info = dg01nd('D', xr.copy(), xi.copy())
    assert info == 0

    freq_energy = xr_fft[0]**2 + xi_fft[0]**2 + xr_fft[n]**2 + xi_fft[n]**2
    for k in range(1, n):
        freq_energy += 2.0 * (xr_fft[k]**2 + xi_fft[k]**2)

    np.testing.assert_allclose(time_energy, freq_energy / (2 * n), rtol=1e-13)


"""Edge case tests."""

def test_n_equals_2():
    """
    Test minimum valid N=2.

    With N=2, we have a 4-sample real signal split into:
    XR = [A(1), A(3)]
    XI = [A(2), A(4)]

    Output should be N+1=3 frequency components.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    xr = np.random.randn(2)
    xi = np.random.randn(2)
    xr_orig = xr.copy()
    xi_orig = xi.copy()

    xr_fft, xi_fft, info = dg01nd('D', xr, xi)

    assert info == 0
    assert len(xr_fft) == 3
    assert len(xi_fft) == 3

    xr_back, xi_back, info2 = dg01nd('I', xr_fft, xi_fft)
    assert info2 == 0

    np.testing.assert_allclose(xr_back[:2], 4 * xr_orig, rtol=1e-14)
    np.testing.assert_allclose(xi_back[:2], 4 * xi_orig, rtol=1e-14)

def test_n_equals_4():
    """
    Test N=4.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 4
    xr = np.random.randn(n)
    xi = np.random.randn(n)
    xr_orig = xr.copy()
    xi_orig = xi.copy()

    xr_fft, xi_fft, info1 = dg01nd('D', xr, xi)
    assert info1 == 0
    assert len(xr_fft) == n + 1
    assert len(xi_fft) == n + 1

    xr_back, xi_back, info2 = dg01nd('I', xr_fft, xi_fft)
    assert info2 == 0

    np.testing.assert_allclose(xr_back[:n], 2 * n * xr_orig, rtol=1e-14)
    np.testing.assert_allclose(xi_back[:n], 2 * n * xi_orig, rtol=1e-14)

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

    xr_fft, xi_fft, info1 = dg01nd('D', xr, xi)
    assert info1 == 0

    xr_back, xi_back, info2 = dg01nd('I', xr_fft, xi_fft)
    assert info2 == 0

    np.testing.assert_allclose(xr_back[:n], 2 * n * xr_orig, rtol=1e-12)
    np.testing.assert_allclose(xi_back[:n], 2 * n * xi_orig, rtol=1e-12)

def test_zero_signal():
    """Test that zero signal gives zero output."""
    n = 8
    xr = np.zeros(n, dtype=float)
    xi = np.zeros(n, dtype=float)

    xr_fft, xi_fft, info = dg01nd('D', xr, xi)

    assert info == 0
    np.testing.assert_allclose(xr_fft, np.zeros(n + 1), atol=1e-15)
    np.testing.assert_allclose(xi_fft, np.zeros(n + 1), atol=1e-15)

def test_constant_signal():
    """
    Test constant signal (all ones).

    If A(i) = 1 for all i=1..2*N, then:
    - DC component FA(1) = 2*N
    - All other components = 0
    """
    n = 8
    xr = np.ones(n, dtype=float)
    xi = np.ones(n, dtype=float)

    xr_fft, xi_fft, info = dg01nd('D', xr, xi)

    assert info == 0
    np.testing.assert_allclose(xr_fft[0], 2 * n, rtol=1e-14)
    np.testing.assert_allclose(xi_fft[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(xr_fft[1:], 0.0, atol=1e-14)
    np.testing.assert_allclose(xi_fft[1:], 0.0, atol=1e-14)


"""Error handling tests."""

def test_invalid_indi():
    """Test with invalid INDI parameter."""
    xr = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    xi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    xr_out, xi_out, info = dg01nd('X', xr, xi)
    assert info == -1

def test_n_not_power_of_2():
    """Test with N not a power of 2."""
    xr = np.array([1.0, 2.0, 3.0], dtype=float)
    xi = np.array([0.0, 0.0, 0.0], dtype=float)
    xr_out, xi_out, info = dg01nd('D', xr, xi)
    assert info == -2

def test_n_equals_1():
    """Test with N=1 (less than minimum 2)."""
    xr = np.array([1.0], dtype=float)
    xi = np.array([0.0], dtype=float)
    xr_out, xi_out, info = dg01nd('D', xr, xi)
    assert info == -2

def test_lowercase_d():
    """Test that lowercase 'd' works for forward FFT."""
    xr = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    xi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    xr_out, xi_out, info = dg01nd('d', xr, xi)
    assert info == 0

def test_lowercase_i():
    """Test that lowercase 'i' works for inverse FFT."""
    xr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    xr_out, xi_out, info = dg01nd('i', xr, xi)
    assert info == 0
