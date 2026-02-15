"""Tests for DK01MD - Anti-aliasing window applied to a real signal."""

import numpy as np
import pytest
from ctrlsys import dk01md


"""Basic functionality tests from SLICOT HTML documentation."""

def test_hamming_window_doc_example():
    """
    Test Hamming window using HTML doc example.

    Input: N=8, TYPE='M', A=[0.3262, 0.8723, -0.7972, 0.6673, -0.1722, 0.3237, 0.5263, -0.3275]
    Expected: A=[0.3262, 0.8326, -0.6591, 0.4286, -0.0754, 0.0820, 0.0661, -0.0262]
    """
    a = np.array([0.3262, 0.8723, -0.7972, 0.6673, -0.1722, 0.3237, 0.5263, -0.3275], dtype=float)
    a_expected = np.array([0.3262, 0.8326, -0.6591, 0.4286, -0.0754, 0.0820, 0.0661, -0.0262], dtype=float)

    a_out, info = dk01md('M', a)

    assert info == 0
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)


"""Test all window types."""

def test_hamming_window_formula():
    """
    Validate Hamming window formula: A(i) = (0.54 + 0.46*cos(pi*(i-1)/(N-1)))*A(i).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 10
    a_input = np.random.randn(n)
    a = a_input.copy()

    a_out, info = dk01md('M', a)

    assert info == 0
    n1 = n - 1
    for i in range(n):
        window = 0.54 + 0.46 * np.cos(np.pi * i / n1)
        expected = window * a_input[i]
        np.testing.assert_allclose(a_out[i], expected, rtol=1e-14)

def test_hann_window_formula():
    """
    Validate Hann window formula: A(i) = 0.5*(1 + cos(pi*(i-1)/(N-1)))*A(i).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 10
    a_input = np.random.randn(n)
    a = a_input.copy()

    a_out, info = dk01md('N', a)

    assert info == 0
    n1 = n - 1
    for i in range(n):
        window = 0.5 * (1.0 + np.cos(np.pi * i / n1))
        expected = window * a_input[i]
        np.testing.assert_allclose(a_out[i], expected, rtol=1e-14)

def test_quadratic_window_formula():
    """
    Validate quadratic window formula.

    For i = 1,...,(N-1)/2+1:
        A(i) = (1 - 2*((i-1)/(N-1))**2)*(1 - (i-1)/(N-1))*A(i)
    For i = (N-1)/2+2,...,N:
        A(i) = 2*(1 - ((i-1)/(N-1))**3)*A(i)

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 11
    a_input = np.random.randn(n)
    a = a_input.copy()

    a_out, info = dk01md('Q', a)

    assert info == 0
    fn = n - 1
    n1 = (n - 1) // 2 + 1

    for i in range(n):
        buf = i / fn
        temp = buf ** 2
        if i < n1:
            window = (1.0 - 2.0 * temp) * (1.0 - buf)
        else:
            window = 2.0 * (1.0 - buf * temp)
        expected = window * a_input[i]
        np.testing.assert_allclose(a_out[i], expected, rtol=1e-14)


"""Mathematical property tests."""

def test_window_boundary_values_hamming():
    """
    Validate Hamming window boundary values.

    At i=0 (first sample): window = 0.54 + 0.46*cos(0) = 1.0
    At i=N-1 (last sample): window = 0.54 + 0.46*cos(pi) = 0.54 - 0.46 = 0.08
    """
    n = 8
    a = np.ones(n, dtype=float)

    a_out, info = dk01md('M', a)

    assert info == 0
    np.testing.assert_allclose(a_out[0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(a_out[-1], 0.08, rtol=1e-14)

def test_window_boundary_values_hann():
    """
    Validate Hann window boundary values.

    At i=0 (first sample): window = 0.5*(1 + cos(0)) = 1.0
    At i=N-1 (last sample): window = 0.5*(1 + cos(pi)) = 0.0
    """
    n = 8
    a = np.ones(n, dtype=float)

    a_out, info = dk01md('N', a)

    assert info == 0
    np.testing.assert_allclose(a_out[0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(a_out[-1], 0.0, atol=1e-15)

def test_window_coefficients_decrease_hamming():
    """
    Validate Hamming window coefficients decrease from start.

    At i=0: window = 0.54 + 0.46 = 1.0 (maximum)
    At i=N-1: window = 0.54 - 0.46 = 0.08 (minimum)
    """
    n = 9
    a = np.ones(n, dtype=float)

    a_out, info = dk01md('M', a)

    assert info == 0
    # First coefficient should be maximum (1.0)
    assert a_out[0] == max(a_out)
    # Last coefficient should be minimum (0.08)
    np.testing.assert_allclose(a_out[-1], 0.08, rtol=1e-14)

def test_window_coefficients_decrease_hann():
    """
    Validate Hann window coefficients decrease from start.

    At i=0: window = 0.5*(1+1) = 1.0 (maximum)
    At i=N-1: window = 0.5*(1-1) = 0.0 (minimum)
    """
    n = 9
    a = np.ones(n, dtype=float)

    a_out, info = dk01md('N', a)

    assert info == 0
    # First coefficient should be maximum (1.0)
    assert a_out[0] == max(a_out)
    # Last coefficient should be minimum (0.0)
    np.testing.assert_allclose(a_out[-1], 0.0, atol=1e-15)

def test_zero_signal_stays_zero():
    """
    Validate that zero signal stays zero after windowing.

    Random seed: 789 (for reproducibility)
    """
    n = 10
    a = np.zeros(n, dtype=float)

    for win_type in ['M', 'N', 'Q']:
        a_copy = a.copy()
        a_out, info = dk01md(win_type, a_copy)
        assert info == 0
        np.testing.assert_allclose(a_out, np.zeros(n), atol=1e-15)


"""Edge case tests."""

def test_single_sample():
    """
    Test with single sample (N=1).

    For N=1, window coefficient should be 1.0 for all window types
    (since cos(0) = 1 and boundary formulas reduce to 1.0).
    """
    for win_type in ['M', 'N', 'Q']:
        a = np.array([2.5], dtype=float)
        a_out, info = dk01md(win_type, a)
        assert info == 0
        np.testing.assert_allclose(a_out[0], 2.5, rtol=1e-14)

def test_two_samples_hamming():
    """Test Hamming window with N=2."""
    a = np.array([1.0, 1.0], dtype=float)
    a_out, info = dk01md('M', a)
    assert info == 0
    np.testing.assert_allclose(a_out[0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(a_out[1], 0.08, rtol=1e-14)

def test_two_samples_hann():
    """Test Hann window with N=2."""
    a = np.array([1.0, 1.0], dtype=float)
    a_out, info = dk01md('N', a)
    assert info == 0
    np.testing.assert_allclose(a_out[0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(a_out[1], 0.0, atol=1e-15)


"""Error handling tests."""

def test_invalid_type():
    """Test with invalid window type."""
    a = np.array([1.0, 2.0, 3.0], dtype=float)
    a_out, info = dk01md('X', a)
    assert info == -1

def test_lowercase_type_m():
    """Test that lowercase 'm' works for Hamming window."""
    a = np.array([1.0, 1.0, 1.0], dtype=float)
    a_out, info = dk01md('m', a)
    assert info == 0

def test_lowercase_type_n():
    """Test that lowercase 'n' works for Hann window."""
    a = np.array([1.0, 1.0, 1.0], dtype=float)
    a_out, info = dk01md('n', a)
    assert info == 0

def test_lowercase_type_q():
    """Test that lowercase 'q' works for Quadratic window."""
    a = np.array([1.0, 1.0, 1.0], dtype=float)
    a_out, info = dk01md('q', a)
    assert info == 0

def test_empty_array():
    """Test with empty array (N=0) - should return error."""
    a = np.array([], dtype=float)
    a_out, info = dk01md('M', a)
    assert info == -2
