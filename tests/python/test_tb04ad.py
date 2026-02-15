"""
Tests for TB04AD: State-space to transfer function conversion.

TB04AD computes the transfer function matrix T(s) from state-space
representation (A,B,C,D) as rows or columns over common denominators.

T(s) = C * (sI - A)^(-1) * B + D
"""

import numpy as np
import pytest
from ctrlsys import tb04ad



def test_tb04ad_html_doc_example():
    """
    Test TB04AD using HTML documentation example.

    System: N=3, M=2, P=2
    A = diag(-1, -2, -3)
    Transfer function as rows over common denominators.

    Verifies frequency response T(s) = C * (sI - A)^(-1) * B + D
    at multiple frequencies.
    """
    n, m, p = 3, 2, 2

    a = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0, -2.0,  0.0],
        [ 0.0,  0.0, -3.0]
    ], order='F', dtype=float)

    b = np.array([
        [ 0.0,  1.0],
        [ 1.0,  1.0],
        [-1.0,  0.0]
    ], order='F', dtype=float)

    c = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ], order='F', dtype=float)

    d = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = tb04ad('R', a, b, c, d)

    assert info == 0
    assert nr == 3

    for omega in [0.1, 1.0, 10.0]:
        s = 1j * omega

        eye = np.eye(n)
        t_direct = c @ np.linalg.solve(s * eye - a, b) + d

        t_poly = np.zeros((p, m), dtype=complex)
        for i in range(p):
            deg = index[i]
            den_val = sum(dcoeff[i, k] * s**(deg - k) for k in range(deg + 1))
            for j in range(m):
                num_val = sum(ucoeff[i, j, k] * s**(deg - k) for k in range(deg + 1))
                t_poly[i, j] = num_val / den_val

        np.testing.assert_allclose(t_poly, t_direct, rtol=1e-10)


def test_tb04ad_columns_mode():
    """
    Test TB04AD with ROWCOL='C' (columns over common denominators).

    Uses same system as HTML doc but requests column factorization.
    This internally uses the dual system (A^T, C^T, B^T, D^T).
    """
    n, m, p = 3, 2, 2

    a = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0, -2.0,  0.0],
        [ 0.0,  0.0, -3.0]
    ], order='F', dtype=float)

    b = np.array([
        [ 0.0,  1.0],
        [ 1.0,  1.0],
        [-1.0,  0.0]
    ], order='F', dtype=float)

    c = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ], order='F', dtype=float)

    d = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = tb04ad('C', a, b, c, d)

    assert info == 0
    # For column mode, index has m entries (number of inputs)
    assert len(index) >= m


def test_tb04ad_siso_system():
    """
    Test TB04AD with a simple SISO system.

    System: x' = -2*x + u, y = x
    Transfer function: G(s) = 1/(s+2)

    Random seed: 42 (for reproducibility)
    """
    n, m, p = 1, 1, 1

    a = np.array([[-2.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = tb04ad('R', a, b, c, d)

    assert info == 0
    assert nr == 1

    # Denominator: s + 2 (degree 1)
    assert index[0] == 1
    np.testing.assert_allclose(dcoeff[0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(dcoeff[0, 1], 2.0, rtol=1e-14)

    # Numerator: 1 (degree 0)
    np.testing.assert_allclose(ucoeff[0, 0, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(ucoeff[0, 0, 1], 1.0, rtol=1e-14)


def test_tb04ad_siso_with_feedthrough():
    """
    Test TB04AD with SISO system with feedthrough.

    System: x' = -3*x + 2*u, y = x + u
    Transfer function: G(s) = (s + 5)/(s + 3)

    Derivation: G(s) = C*(sI-A)^(-1)*B + D = 1/(s+3)*2 + 1 = (2 + s + 3)/(s+3) = (s+5)/(s+3)
    """
    n, m, p = 1, 1, 1

    a = np.array([[-3.0]], order='F', dtype=float)
    b = np.array([[2.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[1.0]], order='F', dtype=float)

    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = tb04ad('R', a, b, c, d)

    assert info == 0
    assert nr == 1
    assert index[0] == 1

    # Denominator: s + 3
    np.testing.assert_allclose(dcoeff[0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(dcoeff[0, 1], 3.0, rtol=1e-14)

    # Numerator: s + 5
    np.testing.assert_allclose(ucoeff[0, 0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(ucoeff[0, 0, 1], 5.0, rtol=1e-14)


def test_tb04ad_uncontrollable_mode():
    """
    Test TB04AD with system containing uncontrollable mode.

    System has eigenvalue at -1 that is uncontrollable.
    Transfer function should reflect only controllable part.
    """
    n, m, p = 2, 1, 1

    # A = diag(-1, -2), B = [0; 1] -> first mode uncontrollable
    a = np.array([
        [-1.0,  0.0],
        [ 0.0, -2.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = tb04ad('R', a, b, c, d)

    assert info == 0
    # Only the controllable part (n=1) contributes
    assert nr == 1
    assert index[0] == 1

    # Transfer function: G(s) = 1/(s+2) (only controllable mode)
    np.testing.assert_allclose(dcoeff[0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(dcoeff[0, 1], 2.0, rtol=1e-14)
    np.testing.assert_allclose(ucoeff[0, 0, 1], 1.0, rtol=1e-14)


def test_tb04ad_zero_order_system():
    """
    Test TB04AD with zero-order system (N=0).

    Transfer function: T(s) = D (constant matrix)
    """
    n, m, p = 0, 2, 2

    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, m), order='F', dtype=float)
    c = np.zeros((p, 0), order='F', dtype=float)
    d = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = tb04ad('R', a, b, c, d)

    assert info == 0
    assert nr == 0

    # All indices should be 0 (constant transfer function)
    np.testing.assert_array_equal(index[:p], [0, 0])

    # Denominators are all 1 (monic constant)
    np.testing.assert_allclose(dcoeff[0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(dcoeff[1, 0], 1.0, rtol=1e-14)

    # Numerators equal D matrix elements
    np.testing.assert_allclose(ucoeff[0, 0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(ucoeff[0, 1, 0], 2.0, rtol=1e-14)
    np.testing.assert_allclose(ucoeff[1, 0, 0], 3.0, rtol=1e-14)
    np.testing.assert_allclose(ucoeff[1, 1, 0], 4.0, rtol=1e-14)



def test_tb04ad_transfer_function_evaluation():
    """
    Mathematical property test: verify transfer function values.

    Compute T(s) at specific frequency and compare with direct formula:
    T(s) = C * (sI - A)^(-1) * B + D

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 2, 1, 1

    # Create stable system
    a = np.array([
        [-1.0, 0.5],
        [0.0, -2.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [1.0]
    ], order='F', dtype=float)

    c = np.array([[1.0, 0.0]], order='F', dtype=float)
    d = np.array([[0.5]], order='F', dtype=float)

    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = tb04ad('R', a, b, c, d)

    assert info == 0

    # Evaluate at s = 1j (purely imaginary frequency)
    s = 1.0j

    # Direct computation: T(s) = C * (sI - A)^(-1) * B + D
    eye = np.eye(n)
    t_direct = c @ np.linalg.solve(s * eye - a, b) + d

    # From polynomial coefficients: T(s) = num(s) / den(s)
    # Build denominator polynomial
    deg = index[0]
    den_val = sum(dcoeff[0, k] * s**(deg - k) for k in range(deg + 1))

    # Build numerator polynomial
    num_val = sum(ucoeff[0, 0, k] * s**(deg - k) for k in range(deg + 1))

    t_poly = num_val / den_val

    np.testing.assert_allclose(t_poly, t_direct[0, 0], rtol=1e-10)



def test_tb04ad_mimo_frequency_response():
    """
    Mathematical property test: verify MIMO transfer function at s = j*omega.

    Compare polynomial representation with direct state-space formula.

    Random seed: 456 (for reproducibility)
    """
    n, m, p = 3, 2, 2

    # Use HTML doc example system
    a = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0, -2.0,  0.0],
        [ 0.0,  0.0, -3.0]
    ], order='F', dtype=float)

    b = np.array([
        [ 0.0,  1.0],
        [ 1.0,  1.0],
        [-1.0,  0.0]
    ], order='F', dtype=float)

    c = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ], order='F', dtype=float)

    d = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = tb04ad('R', a, b, c, d)

    assert info == 0

    # Test at multiple frequencies
    for omega in [0.1, 1.0, 10.0]:
        s = 1j * omega

        # Direct: T(s) = C * (sI - A)^(-1) * B + D
        eye = np.eye(n)
        t_direct = c @ np.linalg.solve(s * eye - a, b) + d

        # From polynomials
        kdcoef = max(index[:p]) + 1
        t_poly = np.zeros((p, m), dtype=complex)

        for i in range(p):
            deg = index[i]
            # Denominator
            den_val = sum(dcoeff[i, k] * s**(deg - k) for k in range(deg + 1))

            for j in range(m):
                # Numerator
                num_val = sum(ucoeff[i, j, k] * s**(deg - k) for k in range(deg + 1))
                t_poly[i, j] = num_val / den_val

        np.testing.assert_allclose(t_poly, t_direct, rtol=1e-10)


def test_tb04ad_error_invalid_rowcol():
    """
    Test that invalid ROWCOL parameter returns error.
    """
    n, m, p = 2, 1, 1

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    # Invalid ROWCOL should return info < 0
    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = tb04ad('X', a, b, c, d)

    assert info == -1
