"""Tests for TF01ND - discrete-time state-space output response with Hessenberg A."""

import numpy as np
import pytest
from ctrlsys import tf01nd


"""Basic functionality tests from SLICOT HTML documentation."""

def test_html_doc_example_upper_hessenberg():
    """
    Test case from SLICOT HTML documentation.

    3x2x2 system (N=3 states, M=2 inputs, P=2 outputs), NY=10 time steps.
    A is upper Hessenberg (UPLO='U').

    Data read column-wise from:
    READ ( NIN, FMT = * ) ( ( A(I,J), I = 1,N ), J = 1,N )
    """
    n, m, p, ny = 3, 2, 2, 10

    # A (3x3) upper Hessenberg - read column-wise: ((A(I,J), I=1,N), J=1,N)
    # Data: 0.0000 -0.0700 0.0000 / 1.0000 0.8000 -0.1500 / 0.0000 0.0000 0.5000
    # Column 1: [0.0, -0.07, 0.0], Column 2: [1.0, 0.8, -0.15], Column 3: [0.0, 0.0, 0.5]
    a = np.array([
        [0.0000, 1.0000, 0.0000],
        [-0.0700, 0.8000, 0.0000],
        [0.0000, -0.1500, 0.5000]
    ], order='F', dtype=float)

    # B (3x2) - read column-wise: ((B(I,J), I=1,N), J=1,M)
    # Data: 0.0000 2.0000 1.0000 / -1.0000 -0.1000 1.0000
    b = np.array([
        [0.0000, -1.0000],
        [2.0000, -0.1000],
        [1.0000, 1.0000]
    ], order='F', dtype=float)

    # C (2x3) - read column-wise: ((C(I,J), I=1,P), J=1,N)
    # Data: 0.0000 1.0000 / 0.0000 0.0000 / 1.0000 0.0000
    c = np.array([
        [0.0000, 0.0000, 1.0000],
        [1.0000, 0.0000, 0.0000]
    ], order='F', dtype=float)

    # D (2x2) - read column-wise: ((D(I,J), I=1,P), J=1,M)
    # Data: 1.0000 0.5000 / 0.0000 0.5000
    d = np.array([
        [1.0000, 0.0000],
        [0.5000, 0.5000]
    ], order='F', dtype=float)

    # x0 (3,) - initial state: 1.0000 1.0000 1.0000
    x0 = np.array([1.0000, 1.0000, 1.0000], dtype=float)

    # U (2x10) - input sequence, read column-wise: ((U(I,J), I=1,M), J=1,NY)
    # Data: -0.6922 -1.4934 / 0.3081 -2.7726 / 2.0039 0.2614 / -0.9160 -0.6030 /
    #       1.2556 0.2951 / -1.5734 1.5639 / -0.9942 1.8957 / 0.8988 0.4118 /
    #       -1.4893 -0.9344 / 1.2506 -0.0701
    u_flat = np.array([
        -0.6922, -1.4934, 0.3081, -2.7726, 2.0039,
        0.2614, -0.9160, -0.6030, 1.2556, 0.2951,
        -1.5734, 1.5639, -0.9942, 1.8957, 0.8988,
        0.4118, -1.4893, -0.9344, 1.2506, -0.0701
    ], dtype=float)
    u = u_flat.reshape((m, ny), order='F')

    # Expected outputs from HTML doc (P=2, NY=10)
    # Y( 1): 0.3078, -0.0928
    # Y( 2): -1.5275, 1.2611
    # Y( 3): -1.3026, 3.4002
    # Y( 4): -0.3512, -0.7060
    # Y( 5): -0.5922, 5.4532
    # Y( 6): -1.1693, 1.1846
    # Y( 7): -1.3029, 2.2286
    # Y( 8): 1.7529, -1.9534
    # Y( 9): 0.6793, -4.4965
    # Y(10): -0.0349, 1.1654
    y_expected = np.array([
        [0.3078, -1.5275, -1.3026, -0.3512, -0.5922, -1.1693, -1.3029, 1.7529, 0.6793, -0.0349],
        [-0.0928, 1.2611, 3.4002, -0.7060, 5.4532, 1.1846, 2.2286, -1.9534, -4.4965, 1.1654]
    ], order='F', dtype=float)

    y, x_final, info = tf01nd('U', a, b, c, d, u, x0.copy())

    assert info == 0
    np.testing.assert_allclose(y, y_expected, rtol=1e-3, atol=1e-4)


def test_state_update_property_upper():
    """
    Validate state-space equations hold: x(k+1) = A*x(k) + B*u(k).

    Upper Hessenberg A matrix.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 3, 2, 2
    ny = 5

    # Create upper Hessenberg A (subdiagonal + upper triangle)
    a = np.triu(np.random.randn(n, n), -1).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')
    u = np.random.randn(m, ny).astype(float, order='F')
    x0 = np.random.randn(n).astype(float)

    y, x_final, info = tf01nd('U', a, b, c, d, u, x0.copy())

    assert info == 0

    # Manual simulation to verify
    x = x0.copy()
    for k in range(ny):
        y_k = c @ x + d @ u[:, k]
        x_next = a @ x + b @ u[:, k]
        np.testing.assert_allclose(y[:, k], y_k, rtol=1e-14, atol=1e-15)
        x = x_next

    np.testing.assert_allclose(x_final, x, rtol=1e-14, atol=1e-15)


def test_state_update_property_lower():
    """
    Validate state-space equations hold: x(k+1) = A*x(k) + B*u(k).

    Lower Hessenberg A matrix.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 4, 2, 3
    ny = 8

    # Create lower Hessenberg A (superdiagonal + lower triangle)
    a = np.tril(np.random.randn(n, n), 1).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')
    u = np.random.randn(m, ny).astype(float, order='F')
    x0 = np.random.randn(n).astype(float)

    y, x_final, info = tf01nd('L', a, b, c, d, u, x0.copy())

    assert info == 0

    # Simulate and check each output
    x = x0.copy()
    for k in range(ny):
        y_expected = c @ x + d @ u[:, k]
        np.testing.assert_allclose(y[:, k], y_expected, rtol=1e-14, atol=1e-15)
        x = a @ x + b @ u[:, k]

    np.testing.assert_allclose(x_final, x, rtol=1e-14, atol=1e-15)


def test_output_equation_property():
    """
    Validate output equation: y(k) = C*x(k) + D*u(k).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 4, 2, 3
    ny = 8

    # Upper Hessenberg A
    a = np.triu(np.random.randn(n, n), -1).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')
    u = np.random.randn(m, ny).astype(float, order='F')
    x0 = np.random.randn(n).astype(float)

    y, x_final, info = tf01nd('U', a, b, c, d, u, x0.copy())

    assert info == 0

    # Simulate and check each output
    x = x0.copy()
    for k in range(ny):
        y_expected = c @ x + d @ u[:, k]
        np.testing.assert_allclose(y[:, k], y_expected, rtol=1e-13, atol=1e-14)
        x = a @ x + b @ u[:, k]


"""Edge case tests."""

def test_zero_time_steps():
    """Test with NY=0 (no time steps)."""
    n, m, p = 2, 1, 1
    a = np.triu(np.eye(n), -1).astype(float, order='F')
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)
    u = np.empty((m, 0), order='F', dtype=float)
    x0 = np.ones(n, dtype=float)

    y, x_final, info = tf01nd('U', a, b, c, d, u, x0.copy())

    assert info == 0
    assert y.shape == (p, 0)


def test_zero_states():
    """Test with N=0 (pure feedthrough, no dynamics)."""
    n, m, p = 0, 2, 2
    ny = 5

    a = np.empty((n, n), order='F', dtype=float)
    b = np.empty((n, m), order='F', dtype=float)
    c = np.empty((p, n), order='F', dtype=float)
    d = np.array([[1.0, 0.5], [0.0, 2.0]], order='F', dtype=float)
    x0 = np.empty(n, dtype=float)

    np.random.seed(789)
    u = np.random.randn(m, ny).astype(float, order='F')

    y, x_final, info = tf01nd('U', a, b, c, d, u, x0)

    assert info == 0
    # y = D*u for each time step
    y_expected = d @ u
    np.testing.assert_allclose(y, y_expected, rtol=1e-14)


def test_single_input_single_output():
    """Test SISO system with upper Hessenberg A.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 2, 1, 1
    ny = 10

    # Upper Hessenberg A
    a = np.array([[0.9, 0.1], [-0.1, 0.8]], order='F', dtype=float)
    b = np.array([[1.0], [0.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)
    u = np.random.randn(m, ny).astype(float, order='F')
    x0 = np.zeros(n, dtype=float)

    y, x_final, info = tf01nd('U', a, b, c, d, u, x0.copy())

    assert info == 0
    assert y.shape == (p, ny)

    # Manual simulation
    x = x0.copy()
    for k in range(ny):
        y_k = c @ x + d @ u[:, k]
        np.testing.assert_allclose(y[:, k], y_k.flatten(), rtol=1e-14)
        x = a @ x + b @ u[:, k].reshape(-1)


def test_zero_inputs():
    """Test with M=0 (no inputs)."""
    n, m, p = 3, 0, 2
    ny = 5

    # Upper Hessenberg A
    a = np.array([
        [0.5, 0.2, 0.1],
        [0.3, 0.4, 0.2],
        [0.0, 0.1, 0.3]
    ], order='F', dtype=float)
    b = np.empty((n, m), order='F', dtype=float)
    c = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], order='F', dtype=float)
    d = np.empty((p, m), order='F', dtype=float)
    u = np.empty((m, ny), order='F', dtype=float)
    x0 = np.array([1.0, 0.5, 0.25], dtype=float)

    y, x_final, info = tf01nd('U', a, b, c, d, u, x0.copy())

    assert info == 0
    # Free response: y = C*x, x(k+1) = A*x(k)
    x = x0.copy()
    for k in range(ny):
        y_expected = c @ x
        np.testing.assert_allclose(y[:, k], y_expected, rtol=1e-14)
        x = a @ x


"""Error handling tests."""

def test_invalid_uplo():
    """Test error with invalid UPLO parameter."""
    n, m, p = 3, 2, 2
    ny = 5

    a = np.eye(n, order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)
    c = np.zeros((p, n), order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)
    u = np.zeros((m, ny), order='F', dtype=float)
    x0 = np.zeros(n, dtype=float)

    y, x_final, info = tf01nd('X', a, b, c, d, u, x0)

    assert info == -1


def test_dimension_mismatch_a():
    """Test error with wrong A dimensions."""
    n, m, p = 3, 2, 2
    ny = 5

    a = np.eye(2, order='F', dtype=float)  # Wrong: should be 3x3
    b = np.zeros((n, m), order='F', dtype=float)
    c = np.zeros((p, n), order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)
    u = np.zeros((m, ny), order='F', dtype=float)
    x0 = np.zeros(n, dtype=float)

    with pytest.raises(ValueError):
        tf01nd('U', a, b, c, d, u, x0)
