"""Tests for DE01OD - Convolution or deconvolution of two real signals."""

import numpy as np
import pytest
from ctrlsys import de01od


"""Basic functionality tests from SLICOT HTML documentation."""

def test_convolution_doc_example():
    """
    Test convolution using HTML doc example.

    Input: N=8, CONV='C'
    A = [0.4862, 0.1948, 0.5788, -0.5861, 0.8254, 0.1815, 0.2904, -0.3599]
    B = [0.2288, 0.3671, 0.6417, 0.3875, 0.2380, 0.4682, 0.5312, 0.6116]

    Expected output A:
    [0.5844, 0.5769, 0.6106, 1.0433, 0.6331, 0.4531, 0.7027, 0.9929]
    """
    a = np.array([0.4862, 0.1948, 0.5788, -0.5861, 0.8254, 0.1815, 0.2904, -0.3599], dtype=float)
    b = np.array([0.2288, 0.3671, 0.6417, 0.3875, 0.2380, 0.4682, 0.5312, 0.6116], dtype=float)

    a_expected = np.array([0.5844, 0.5769, 0.6106, 1.0433, 0.6331, 0.4531, 0.7027, 0.9929], dtype=float)

    a_out, info = de01od('C', a, b)

    assert info == 0
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)


"""Test that convolution followed by deconvolution gives original signal."""

def test_convolution_deconvolution_inverse():
    """
    Validate convolution followed by deconvolution recovers original signal.

    conv(a, b) -> c, then deconv(c, b) -> a

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 8
    a_orig = np.random.randn(n)
    b = np.random.randn(n)

    a = a_orig.copy()
    b_conv = b.copy()
    a_conv, info1 = de01od('C', a, b_conv)
    assert info1 == 0

    b_deconv = b.copy()
    a_deconv, info2 = de01od('D', a_conv, b_deconv)
    assert info2 == 0

    np.testing.assert_allclose(a_deconv, a_orig, rtol=1e-12)

def test_deconvolution_convolution_inverse():
    """
    Validate deconvolution followed by convolution recovers original signal.

    deconv(a, b) -> c, then conv(c, b) -> a

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 16
    a_orig = np.random.randn(n)
    b = np.random.randn(n)

    a = a_orig.copy()
    b_deconv = b.copy()
    a_deconv, info1 = de01od('D', a, b_deconv)
    assert info1 == 0

    b_conv = b.copy()
    a_conv, info2 = de01od('C', a_deconv, b_conv)
    assert info2 == 0

    np.testing.assert_allclose(a_conv, a_orig, rtol=1e-12)


"""Mathematical property tests for convolution/deconvolution."""

def test_convolution_commutativity():
    """
    Validate convolution commutativity: conv(a, b) = conv(b, a).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 8
    a = np.random.randn(n)
    b = np.random.randn(n)

    a1, b1 = a.copy(), b.copy()
    c1, info1 = de01od('C', a1, b1)
    assert info1 == 0

    a2, b2 = b.copy(), a.copy()
    c2, info2 = de01od('C', a2, b2)
    assert info2 == 0

    np.testing.assert_allclose(c1, c2, rtol=1e-14)

def test_convolution_with_impulse():
    """
    Validate convolution with impulse gives original signal.

    conv(a, delta) = a, where delta = [1, 0, 0, ..., 0]

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 8
    a_orig = np.random.randn(n)
    delta = np.zeros(n)
    delta[0] = 1.0

    a = a_orig.copy()
    b = delta.copy()
    a_out, info = de01od('C', a, b)

    assert info == 0
    np.testing.assert_allclose(a_out, a_orig, rtol=1e-13)

def test_convolution_linearity():
    """
    Validate convolution linearity: conv(a1 + a2, b) = conv(a1, b) + conv(a2, b).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 8
    a1 = np.random.randn(n)
    a2 = np.random.randn(n)
    b = np.random.randn(n)

    a1_copy, b1 = a1.copy(), b.copy()
    c1, _ = de01od('C', a1_copy, b1)

    a2_copy, b2 = a2.copy(), b.copy()
    c2, _ = de01od('C', a2_copy, b2)

    a_sum, b3 = (a1 + a2).copy(), b.copy()
    c_sum, _ = de01od('C', a_sum, b3)

    np.testing.assert_allclose(c_sum, c1 + c2, rtol=1e-14)

def test_convolution_scaling():
    """
    Validate convolution scaling: conv(k*a, b) = k * conv(a, b).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 8
    k = 3.5
    a = np.random.randn(n)
    b = np.random.randn(n)

    a1, b1 = a.copy(), b.copy()
    c1, _ = de01od('C', a1, b1)

    a2, b2 = (k * a).copy(), b.copy()
    c2, _ = de01od('C', a2, b2)

    np.testing.assert_allclose(c2, k * c1, rtol=1e-14)


"""Edge case tests."""

def test_n_equals_2():
    """Test minimum valid N=2."""
    a = np.array([1.0, 2.0], dtype=float)
    b = np.array([0.5, -0.5], dtype=float)

    a_out, info = de01od('C', a, b)

    assert info == 0
    assert len(a_out) == 2

def test_n_equals_4():
    """
    Test N=4 with roundtrip.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    a_orig = np.random.randn(4)
    b = np.random.randn(4)

    a1, b1 = a_orig.copy(), b.copy()
    a_conv, info1 = de01od('C', a1, b1)
    assert info1 == 0

    b2 = b.copy()
    a_back, info2 = de01od('D', a_conv, b2)
    assert info2 == 0

    np.testing.assert_allclose(a_back, a_orig, rtol=1e-12)

def test_large_n_power_of_2():
    """
    Test larger N (128) to verify algorithm scales.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 128
    a_orig = np.random.randn(n)
    b = np.random.randn(n)

    a1, b1 = a_orig.copy(), b.copy()
    a_conv, info1 = de01od('C', a1, b1)
    assert info1 == 0

    b2 = b.copy()
    a_back, info2 = de01od('D', a_conv, b2)
    assert info2 == 0

    np.testing.assert_allclose(a_back, a_orig, rtol=1e-11)

def test_zero_signal():
    """Test that zero signal A gives zero output."""
    n = 8
    a = np.zeros(n, dtype=float)
    b = np.random.randn(n)
    np.random.seed(555)
    b = np.random.randn(n)

    a_out, info = de01od('C', a, b)

    assert info == 0
    np.testing.assert_allclose(a_out, np.zeros(n), atol=1e-15)


"""Error handling tests."""

def test_invalid_conv():
    """Test with invalid CONV parameter."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    b = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    a_out, info = de01od('X', a, b)
    assert info == -1

def test_n_not_power_of_2():
    """Test with N not a power of 2."""
    a = np.array([1.0, 2.0, 3.0], dtype=float)
    b = np.array([0.0, 0.0, 0.0], dtype=float)
    a_out, info = de01od('C', a, b)
    assert info == -2

def test_n_equals_1():
    """Test with N=1 (less than minimum 2)."""
    a = np.array([1.0], dtype=float)
    b = np.array([0.0], dtype=float)
    a_out, info = de01od('C', a, b)
    assert info == -2

def test_lowercase_c():
    """Test that lowercase 'c' works for convolution."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    b = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    a_out, info = de01od('c', a, b)
    assert info == 0

def test_lowercase_d():
    """Test that lowercase 'd' works for deconvolution."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    b = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    a_out, info = de01od('d', a, b)
    assert info == 0
