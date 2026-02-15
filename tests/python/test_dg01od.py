"""Tests for DG01OD - Discrete Hartley Transform of real signal."""

import numpy as np
import pytest
from ctrlsys import dg01od


"""Basic functionality tests from SLICOT HTML documentation."""


def test_hartley_transform_doc_example():
    """
    Test Hartley transform using HTML doc example.

    Input: N=16, SCR='N', WGHT='N'
    A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    Expected output (from HTML doc):
    A = [136.0000, -48.2187, -27.3137, -19.9728, -16.0000, -13.3454,
         -11.3137, -9.5913, -8.0000, -6.4087, -4.6863, -2.6546,
         0.0000, 3.9728, 11.3137, 32.2187]
    """
    a = np.array([1., 2., 3., 4., 5., 6., 7., 8.,
                  9., 10., 11., 12., 13., 14., 15., 16.], dtype=float, order='F')
    n = len(a)
    w = np.zeros(n, dtype=float, order='F')

    a_expected = np.array([136.0000, -48.2187, -27.3137, -19.9728, -16.0000, -13.3454,
                           -11.3137, -9.5913, -8.0000, -6.4087, -4.6863, -2.6546,
                           0.0000, 3.9728, 11.3137, 32.2187], dtype=float)

    a_out, w_out, info = dg01od('N', 'N', a, w)

    assert info == 0
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)


"""Mathematical property tests for Hartley transform."""


def test_hartley_involution():
    """
    Validate Hartley transform involution: H(H(x)) = N*x.

    The Hartley transform is self-inverse up to scaling by N.
    This is a fundamental property: applying twice returns N * original.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 16
    a_orig = np.random.randn(n).astype(float, order='F')
    a = a_orig.copy()
    w = np.zeros(n, dtype=float, order='F')

    a_h1, w_out, info1 = dg01od('N', 'N', a, w)
    assert info1 == 0

    a_h2, w_out2, info2 = dg01od('N', 'A', a_h1.copy(), w_out)
    assert info2 == 0

    np.testing.assert_allclose(a_h2, n * a_orig, rtol=1e-14)


def test_hartley_linearity():
    """
    Validate Hartley transform linearity: H(a*x + b*y) = a*H(x) + b*H(y).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 8
    alpha, beta = 2.5, -1.3

    x = np.random.randn(n).astype(float, order='F')
    y = np.random.randn(n).astype(float, order='F')
    z = alpha * x + beta * y
    w = np.zeros(n, dtype=float, order='F')

    hx, w1, _ = dg01od('N', 'N', x.copy(), w.copy())
    hy, w2, _ = dg01od('N', 'N', y.copy(), w.copy())
    hz, w3, _ = dg01od('N', 'N', z.copy(), w.copy())

    np.testing.assert_allclose(hz, alpha * hx + beta * hy, rtol=1e-14)


def test_hartley_dc_component():
    """
    Validate DC component (index 0) is sum of all samples.

    H[0] = sum(a[i]) for i=0..N-1

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 8
    a = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a_h, w_out, info = dg01od('N', 'N', a.copy(), w)
    assert info == 0

    np.testing.assert_allclose(a_h[0], np.sum(a), rtol=1e-14)


def test_hartley_parseval():
    """
    Validate Parseval's theorem for Hartley: sum|x|^2 = (1/N)*sum|H(x)|^2.

    Energy is preserved (up to N scaling).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 16
    a = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    energy_time = np.sum(a**2)

    a_h, w_out, info = dg01od('N', 'N', a.copy(), w)
    assert info == 0

    energy_freq = np.sum(a_h**2)

    np.testing.assert_allclose(energy_time, energy_freq / n, rtol=1e-14)


def test_hartley_impulse():
    """
    Validate impulse response: delta[0]=1 gives constant spectrum.

    If a = [1, 0, 0, ..., 0], then H(a) = [1, 1, 1, ..., 1].
    """
    n = 8
    a = np.zeros(n, dtype=float, order='F')
    a[0] = 1.0
    w = np.zeros(n, dtype=float, order='F')

    a_h, w_out, info = dg01od('N', 'N', a, w)
    assert info == 0

    np.testing.assert_allclose(a_h, np.ones(n), rtol=1e-14)


def test_hartley_constant_signal():
    """
    Validate constant signal: a = [c, c, ..., c] gives H(a) = [N*c, 0, ..., 0].

    A DC signal transforms to a single spike at index 0.
    """
    n = 8
    c = 3.5
    a = np.full(n, c, dtype=float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a_h, w_out, info = dg01od('N', 'N', a, w)
    assert info == 0

    expected = np.zeros(n, dtype=float)
    expected[0] = n * c

    np.testing.assert_allclose(a_h, expected, rtol=1e-14, atol=1e-14)


"""Tests for different scrambling modes."""


def test_scrambling_mode_n():
    """
    Test SCR='N' (no scrambling) - standard Hartley transform.

    Random seed: 100 (for reproducibility)
    """
    np.random.seed(100)
    n = 8
    a = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a_out, w_out, info = dg01od('N', 'N', a, w)
    assert info == 0


def test_scrambling_mode_i():
    """
    Test SCR='I' (input bit-reversed).

    When input is bit-reversed, output is standard order.
    Applying to bit-reversed input should give same result as standard to normal input.

    Random seed: 101 (for reproducibility)
    """
    np.random.seed(101)
    n = 8
    a = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    def bit_reverse_permutation(n):
        """Get bit-reversal permutation indices for power-of-2 n."""
        bits = int(np.log2(n))
        indices = np.zeros(n, dtype=int)
        for i in range(n):
            rev = 0
            temp = i
            for _ in range(bits):
                rev = (rev << 1) | (temp & 1)
                temp >>= 1
            indices[i] = rev
        return indices

    perm = bit_reverse_permutation(n)
    a_bit_rev = a[perm].copy()

    a_std, w1, info1 = dg01od('N', 'N', a.copy(), w.copy())
    a_br, w2, info2 = dg01od('I', 'N', a_bit_rev, w.copy())

    assert info1 == 0
    assert info2 == 0
    np.testing.assert_allclose(a_br, a_std, rtol=1e-14)


def test_scrambling_mode_o():
    """
    Test SCR='O' (output bit-reversed).

    Output is in bit-reversed order.

    Random seed: 102 (for reproducibility)
    """
    np.random.seed(102)
    n = 8
    a = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    def bit_reverse_permutation(n):
        """Get bit-reversal permutation indices for power-of-2 n."""
        bits = int(np.log2(n))
        indices = np.zeros(n, dtype=int)
        for i in range(n):
            rev = 0
            temp = i
            for _ in range(bits):
                rev = (rev << 1) | (temp & 1)
                temp >>= 1
            indices[i] = rev
        return indices

    a_std, w1, info1 = dg01od('N', 'N', a.copy(), w.copy())
    a_scr, w2, info2 = dg01od('O', 'N', a.copy(), w.copy())

    assert info1 == 0
    assert info2 == 0

    perm = bit_reverse_permutation(n)
    np.testing.assert_allclose(a_scr[perm], a_std, rtol=1e-14)


"""Tests for weight caching."""


def test_weight_caching():
    """
    Test that weights computed once can be reused.

    First call with WGHT='N' computes weights.
    Subsequent calls with WGHT='A' reuse them.

    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n = 16
    a1 = np.random.randn(n).astype(float, order='F')
    a2_orig = np.random.randn(n).astype(float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a1_h, w_computed, info1 = dg01od('N', 'N', a1, w)
    assert info1 == 0

    a2_h, w_reused, info2 = dg01od('N', 'A', a2_orig.copy(), w_computed.copy())
    assert info2 == 0

    a2_h_fresh, w_fresh, info3 = dg01od('N', 'N', a2_orig.copy(), np.zeros(n, dtype=float))
    assert info3 == 0

    np.testing.assert_allclose(a2_h, a2_h_fresh, rtol=1e-14)


"""Edge case tests."""


def test_n_equals_2():
    """Test minimum valid N=2."""
    a = np.array([1.0, 2.0], dtype=float, order='F')
    w = np.zeros(2, dtype=float, order='F')

    a_h, w_out, info = dg01od('N', 'N', a, w)
    assert info == 0

    np.testing.assert_allclose(a_h[0], 3.0, rtol=1e-14)
    np.testing.assert_allclose(a_h[1], -1.0, rtol=1e-14)


def test_n_equals_4():
    """Test N=4."""
    np.random.seed(300)
    a = np.random.randn(4).astype(float, order='F')
    a_orig = a.copy()
    w = np.zeros(4, dtype=float, order='F')

    a_h1, w1, info1 = dg01od('N', 'N', a, w)
    assert info1 == 0

    a_h2, w2, info2 = dg01od('N', 'A', a_h1.copy(), w1)
    assert info2 == 0

    np.testing.assert_allclose(a_h2, 4 * a_orig, rtol=1e-14)


def test_n_equals_1():
    """Test N=1 (edge case, should be identity or return early)."""
    a = np.array([5.0], dtype=float, order='F')
    w = np.zeros(1, dtype=float, order='F')

    a_h, w_out, info = dg01od('N', 'N', a, w)
    assert info == 0
    np.testing.assert_allclose(a_h[0], 5.0, rtol=1e-14)


def test_large_n_power_of_2():
    """
    Test larger N (128) to verify algorithm scales.

    Random seed: 400 (for reproducibility)
    """
    np.random.seed(400)
    n = 128
    a = np.random.randn(n).astype(float, order='F')
    a_orig = a.copy()
    w = np.zeros(n, dtype=float, order='F')

    a_h1, w1, info1 = dg01od('N', 'N', a, w)
    assert info1 == 0

    a_h2, w2, info2 = dg01od('N', 'A', a_h1.copy(), w1)
    assert info2 == 0

    np.testing.assert_allclose(a_h2, n * a_orig, rtol=1e-12)


def test_zero_signal():
    """Test that zero signal gives zero output."""
    n = 8
    a = np.zeros(n, dtype=float, order='F')
    w = np.zeros(n, dtype=float, order='F')

    a_h, w_out, info = dg01od('N', 'N', a, w)

    assert info == 0
    np.testing.assert_allclose(a_h, np.zeros(n), atol=1e-15)


"""Error handling tests."""


def test_invalid_scr():
    """Test with invalid SCR parameter."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    w = np.zeros(4, dtype=float, order='F')

    a_out, w_out, info = dg01od('X', 'N', a, w)
    assert info == -1


def test_invalid_wght():
    """Test with invalid WGHT parameter."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    w = np.zeros(4, dtype=float, order='F')

    a_out, w_out, info = dg01od('N', 'X', a, w)
    assert info == -2


def test_n_not_power_of_2():
    """Test with N not a power of 2."""
    a = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
    w = np.zeros(3, dtype=float, order='F')

    a_out, w_out, info = dg01od('N', 'N', a, w)
    assert info == -3


def test_n_negative():
    """Test with negative N (via empty array or other means)."""
    a = np.array([], dtype=float, order='F')
    w = np.array([], dtype=float, order='F')

    a_out, w_out, info = dg01od('N', 'N', a, w)
    assert info == 0


def test_lowercase_scr():
    """Test that lowercase SCR parameters work."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    w = np.zeros(4, dtype=float, order='F')

    a_out, w_out, info = dg01od('n', 'n', a, w)
    assert info == 0


def test_lowercase_wght():
    """Test that lowercase WGHT parameters work."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    w = np.zeros(4, dtype=float, order='F')

    a_out, w_out, info = dg01od('N', 'a', a.copy(), w.copy())
    assert info == 0
