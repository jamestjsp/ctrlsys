"""Tests for DE01PD - Convolution or deconvolution using Hartley transform."""

import numpy as np
import pytest
from ctrlsys import de01pd


def test_convolution_doc_example():
    """
    Test convolution using HTML doc example.

    Input: N=8, CONV='C', WGHT='N'
    A = [0.4862, 0.1948, 0.5788, -0.5861, 0.8254, 0.1815, 0.2904, -0.3599]
    B = [0.2288, 0.3671, 0.6417, 0.3875, 0.2380, 0.4682, 0.5312, 0.6116]

    Expected output A (from HTML doc):
    [0.5844, 0.5769, 0.6106, 1.0433, 0.6331, 0.4531, 0.7027, 0.9929]
    """
    a = np.array([0.4862, 0.1948, 0.5788, -0.5861, 0.8254, 0.1815, 0.2904, -0.3599],
                 dtype=float, order='F')
    b = np.array([0.2288, 0.3671, 0.6417, 0.3875, 0.2380, 0.4682, 0.5312, 0.6116],
                 dtype=float, order='F')
    w = np.zeros(8, dtype=float, order='F')

    a_expected = np.array([0.5844, 0.5769, 0.6106, 1.0433, 0.6331, 0.4531, 0.7027, 0.9929],
                          dtype=float)

    a_out, w_out, info = de01pd('C', 'N', a, b, w)

    assert info == 0
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)


def test_convolution_deconvolution_inverse():
    """
    Validate convolution followed by deconvolution recovers original signal.

    conv(a, b) -> c, then deconv(c, b) -> a

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 8
    a_orig = np.random.randn(n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a = a_orig.copy()
    b_conv = b.copy()
    a_conv, w_out, info1 = de01pd('C', 'N', a, b_conv, w)
    assert info1 == 0

    b_deconv = b.copy()
    a_deconv, w_out2, info2 = de01pd('D', 'A', a_conv, b_deconv, w_out)
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
    a_orig = np.random.randn(n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a = a_orig.copy()
    b_deconv = b.copy()
    a_deconv, w_out, info1 = de01pd('D', 'N', a, b_deconv, w)
    assert info1 == 0

    b_conv = b.copy()
    a_conv, w_out2, info2 = de01pd('C', 'A', a_deconv, b_conv, w_out)
    assert info2 == 0

    np.testing.assert_allclose(a_conv, a_orig, rtol=1e-12)


def test_convolution_commutativity():
    """
    Validate convolution commutativity: conv(a, b) = conv(b, a).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 8
    a = np.random.randn(n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a1, b1 = a.copy(), b.copy()
    c1, w1, info1 = de01pd('C', 'N', a1, b1, w.copy())
    assert info1 == 0

    a2, b2 = b.copy(), a.copy()
    c2, w2, info2 = de01pd('C', 'N', a2, b2, w.copy())
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
    a_orig = np.random.randn(n).astype(float, order='F')
    delta = np.zeros(n, dtype=float, order='F')
    delta[0] = 1.0
    w = np.zeros(n, dtype=float, order='F')

    a = a_orig.copy()
    b = delta.copy()
    a_out, w_out, info = de01pd('C', 'N', a, b, w)

    assert info == 0
    np.testing.assert_allclose(a_out, a_orig, rtol=1e-13)


def test_convolution_linearity():
    """
    Validate convolution linearity: conv(a1 + a2, b) = conv(a1, b) + conv(a2, b).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 8
    a1 = np.random.randn(n).astype(float, order='F')
    a2 = np.random.randn(n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a1_copy, b1 = a1.copy(), b.copy()
    c1, _, _ = de01pd('C', 'N', a1_copy, b1, w.copy())

    a2_copy, b2 = a2.copy(), b.copy()
    c2, _, _ = de01pd('C', 'N', a2_copy, b2, w.copy())

    a_sum = (a1 + a2).astype(float, order='F')
    b3 = b.copy()
    c_sum, _, _ = de01pd('C', 'N', a_sum, b3, w.copy())

    np.testing.assert_allclose(c_sum, c1 + c2, rtol=1e-13)


def test_convolution_scaling():
    """
    Validate convolution scaling: conv(k*a, b) = k * conv(a, b).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 8
    k = 3.5
    a = np.random.randn(n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a1, b1 = a.copy(), b.copy()
    c1, _, _ = de01pd('C', 'N', a1, b1, w.copy())

    a2 = (k * a).astype(float, order='F')
    b2 = b.copy()
    c2, _, _ = de01pd('C', 'N', a2, b2, w.copy())

    np.testing.assert_allclose(c2, k * c1, rtol=1e-13)


def test_weight_reuse():
    """
    Validate weight vector can be reused across calls (WGHT='A').

    First call computes weights, subsequent calls reuse them.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n = 8
    a1 = np.random.randn(n).astype(float, order='F')
    a2 = np.random.randn(n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    c1, w_out, info1 = de01pd('C', 'N', a1.copy(), b.copy(), w)
    assert info1 == 0

    c2, w_out2, info2 = de01pd('C', 'A', a2.copy(), b.copy(), w_out)
    assert info2 == 0

    c2_check, _, info3 = de01pd('C', 'N', a2.copy(), b.copy(), np.zeros(n, dtype=float))
    assert info3 == 0

    np.testing.assert_allclose(c2, c2_check, rtol=1e-14)


def test_n_equals_1():
    """
    Test N=1 edge case.

    For N=1: convolution is just a[0] * b[0], deconvolution is a[0] / b[0].
    """
    a = np.array([2.5], dtype=float, order='F')
    b = np.array([0.5], dtype=float, order='F')
    w = np.zeros(1, dtype=float, order='F')

    a_out, w_out, info = de01pd('C', 'N', a, b, w)

    assert info == 0
    np.testing.assert_allclose(a_out, np.array([1.25]), rtol=1e-14)


def test_n_equals_1_deconv():
    """Test N=1 deconvolution."""
    a = np.array([2.5], dtype=float, order='F')
    b = np.array([0.5], dtype=float, order='F')
    w = np.zeros(1, dtype=float, order='F')

    a_out, w_out, info = de01pd('D', 'N', a, b, w)

    assert info == 0
    np.testing.assert_allclose(a_out, np.array([5.0]), rtol=1e-14)


def test_n_equals_2():
    """
    Test N=2 with roundtrip.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    a_orig = np.random.randn(2).astype(float, order='F')
    b = np.random.randn(2).astype(float, order='F')
    w = np.zeros(2, dtype=float, order='F')

    a1 = a_orig.copy()
    b1 = b.copy()
    a_conv, w_out, info1 = de01pd('C', 'N', a1, b1, w)
    assert info1 == 0

    b2 = b.copy()
    a_back, w_out2, info2 = de01pd('D', 'A', a_conv, b2, w_out)
    assert info2 == 0

    np.testing.assert_allclose(a_back, a_orig, rtol=1e-12)


def test_large_n_power_of_2():
    """
    Test larger N (128) to verify algorithm scales.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 128
    a_orig = np.random.randn(n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a1 = a_orig.copy()
    b1 = b.copy()
    a_conv, w_out, info1 = de01pd('C', 'N', a1, b1, w)
    assert info1 == 0

    b2 = b.copy()
    a_back, w_out2, info2 = de01pd('D', 'A', a_conv, b2, w_out)
    assert info2 == 0

    np.testing.assert_allclose(a_back, a_orig, rtol=1e-11)


def test_n_equals_0():
    """Test N=0 returns immediately with no error."""
    a = np.array([], dtype=float, order='F')
    b = np.array([], dtype=float, order='F')
    w = np.array([], dtype=float, order='F')

    a_out, w_out, info = de01pd('C', 'N', a, b, w)

    assert info == 0


def test_invalid_conv():
    """Test with invalid CONV parameter."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    b = np.array([0.0, 0.0, 0.0, 0.0], dtype=float, order='F')
    w = np.zeros(4, dtype=float, order='F')

    a_out, w_out, info = de01pd('X', 'N', a, b, w)
    assert info == -1


def test_invalid_wght():
    """Test with invalid WGHT parameter."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    b = np.array([0.0, 0.0, 0.0, 0.0], dtype=float, order='F')
    w = np.zeros(4, dtype=float, order='F')

    a_out, w_out, info = de01pd('C', 'X', a, b, w)
    assert info == -2


def test_n_not_power_of_2():
    """Test with N not a power of 2."""
    a = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
    b = np.array([0.0, 0.0, 0.0], dtype=float, order='F')
    w = np.zeros(3, dtype=float, order='F')

    a_out, w_out, info = de01pd('C', 'N', a, b, w)
    assert info == -3


def test_lowercase_c():
    """Test that lowercase 'c' works for convolution."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    b = np.array([1.0, 0.0, 0.0, 0.0], dtype=float, order='F')
    w = np.zeros(4, dtype=float, order='F')

    a_out, w_out, info = de01pd('c', 'n', a, b, w)
    assert info == 0


def test_lowercase_d():
    """Test that lowercase 'd' works for deconvolution."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    b = np.array([1.0, 0.0, 0.0, 0.0], dtype=float, order='F')
    w = np.zeros(4, dtype=float, order='F')

    a_out, w_out, info = de01pd('d', 'a', a, b, w)
    assert info == 0
