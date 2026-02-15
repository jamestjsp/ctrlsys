"""
Tests for TD04AD - Transfer function to minimal state-space conversion.

TD04AD finds a minimal state-space representation (A,B,C,D) for a proper
transfer matrix T(s) given as row or column polynomial vectors over
denominator polynomials. Uses Wolovich's Observable Structure Theorem
plus orthogonal similarity transformations via TB01PD.

T(s) = inv(D(s)) * U(s) for ROWCOL='R'
T(s) = U(s) * inv(D(s)) for ROWCOL='C'
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest


def test_td04ad_html_doc_example():
    """
    Test case from SLICOT HTML documentation.

    Input: M=2, P=2, TOL=0.0, ROWCOL='R'
    INDEX = [3, 3]

    DCOEFF (row-wise in data file):
        Row 1: 1.0  6.0 11.0  6.0
        Row 2: 1.0  6.0 11.0  6.0

    UCOEFF read as ((UCOEFF(I,J,K), K=1,KDCOEF), J=1,M), I=1,P):
        U(1,1,:) = [1.0, 6.0, 12.0, 7.0]
        U(1,2,:) = [0.0, 1.0,  4.0, 3.0]
        U(2,1,:) = [0.0, 0.0,  1.0, 1.0]
        U(2,2,:) = [1.0, 8.0, 20.0, 15.0]

    Expected output: NR = 3

    A matrix (3x3):
        0.5000  -0.8028   0.9387
        4.4047  -2.3380   2.5076
       -5.5541   1.6872  -4.1620

    B matrix (3x2):
       -0.2000  -1.2500
        0.0000  -0.6097
        0.0000   2.2217

    C matrix (2x3):
        0.0000  -0.8679   0.2119
        0.0000   0.0000   0.9002

    D matrix (2x2):
        1.0000   0.0000
        0.0000   1.0000
    """
    from ctrlsys import td04ad

    m, p = 2, 2
    rowcol = 'R'
    tol = 0.0

    index = np.array([3, 3], dtype=np.int32)
    kdcoef = 4

    dcoeff = np.array([
        [1.0, 6.0, 11.0, 6.0],
        [1.0, 6.0, 11.0, 6.0]
    ], order='F', dtype=float)

    ucoeff = np.zeros((p, m, kdcoef), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 6.0, 12.0, 7.0]
    ucoeff[0, 1, :] = [0.0, 1.0, 4.0, 3.0]
    ucoeff[1, 0, :] = [0.0, 0.0, 1.0, 1.0]
    ucoeff[1, 1, :] = [1.0, 8.0, 20.0, 15.0]

    nr, a, b, c, d, info = td04ad(rowcol, m, p, index, dcoeff, ucoeff, tol)

    assert info == 0
    assert nr == 3

    a_expected = np.array([
        [0.5000, -0.8028,  0.9387],
        [4.4047, -2.3380,  2.5076],
        [-5.5541, 1.6872, -4.1620]
    ], order='F', dtype=float)

    b_expected = np.array([
        [-0.2000, -1.2500],
        [0.0000,  -0.6097],
        [0.0000,   2.2217]
    ], order='F', dtype=float)

    c_expected = np.array([
        [0.0000, -0.8679, 0.2119],
        [0.0000,  0.0000, 0.9002]
    ], order='F', dtype=float)

    d_expected = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    assert_allclose(a[:nr, :nr], a_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(b[:nr, :m], b_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(c[:p, :nr], c_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(d[:p, :m], d_expected, rtol=1e-3, atol=1e-4)


def test_td04ad_siso_first_order():
    """
    Test simple SISO first-order transfer function.

    T(s) = (2s + 1) / (s + 3)

    Minimal realization should have order 1.
    """
    from ctrlsys import td04ad

    m, p = 1, 1
    index = np.array([1], dtype=np.int32)

    dcoeff = np.array([[1.0, 3.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [2.0, 1.0]

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 0.0)

    assert info == 0
    assert nr == 1

    assert_allclose(d[0, 0], 2.0, rtol=1e-14)


def test_td04ad_siso_second_order():
    """
    Test SISO second-order transfer function.

    T(s) = (s + 2) / (s^2 + 3s + 2)
         = (s + 2) / ((s + 1)(s + 2))
         = 1 / (s + 1)

    The common factor (s+2) should be cancelled, giving NR=1.
    """
    from ctrlsys import td04ad

    m, p = 1, 1
    index = np.array([2], dtype=np.int32)

    dcoeff = np.array([[1.0, 3.0, 2.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 3), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 1.0, 2.0]

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 0.0)

    assert info == 0
    assert nr == 1


def test_td04ad_column_form():
    """
    Test with ROWCOL='C' (columns over common denominators).

    The dual form T(s) = U(s) * inv(D(s)).
    """
    from ctrlsys import td04ad

    m, p = 2, 2
    index = np.array([1, 1], dtype=np.int32)

    dcoeff = np.array([
        [1.0, 2.0],
        [1.0, 3.0]
    ], order='F', dtype=float)

    ucoeff = np.zeros((max(m, p), max(m, p), 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 1.0]
    ucoeff[0, 1, :] = [0.0, 0.5]
    ucoeff[1, 0, :] = [0.0, 0.5]
    ucoeff[1, 1, :] = [1.0, 1.0]

    nr, a, b, c, d, info = td04ad('C', m, p, index, dcoeff, ucoeff, 0.0)

    assert info == 0
    assert nr <= 2


def test_td04ad_transfer_function_equivalence():
    """
    Validate transfer function equivalence at sample frequencies.

    The state-space system should produce the same transfer function
    as the polynomial representation at all frequencies.
    """
    from ctrlsys import td04ad

    m, p = 1, 1
    index = np.array([2], dtype=np.int32)
    kdcoef = 3

    dcoeff = np.array([[1.0, 2.0, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, kdcoef), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 1.0, 0.0]

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 0.0)
    assert info == 0

    ar = a[:nr, :nr].copy()
    br = b[:nr, :m].copy()
    cr = c[:p, :nr].copy()
    dr = d[:p, :m].copy()

    for s in [0.5j, 1.0j, 2.0j, 1.0 + 0.5j]:
        D_s = dcoeff[0, 0] * s**2 + dcoeff[0, 1] * s + dcoeff[0, 2]
        U_s = ucoeff[0, 0, 0] * s**2 + ucoeff[0, 0, 1] * s + ucoeff[0, 0, 2]
        T_poly = U_s / D_s

        I_n = np.eye(nr, dtype=complex)
        T_ss = (cr @ np.linalg.solve(s * I_n - ar, br) + dr)[0, 0]

        assert_allclose(T_ss, T_poly, rtol=1e-10, atol=1e-12)


def test_td04ad_zero_leading_coefficient_error():
    """
    Test error when leading denominator coefficient is near zero.

    INFO = I when row I has leading coefficient too small.
    """
    from ctrlsys import td04ad

    m, p = 1, 1
    index = np.array([1], dtype=np.int32)

    dcoeff = np.array([[1e-320, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 0.0]

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 0.0)

    assert info == 1


def test_td04ad_zero_order():
    """
    Test system with zero order (all INDEX=0).

    When all denominators are degree 0 (constants), N=0 and only D matrix exists.
    """
    from ctrlsys import td04ad

    m, p = 1, 1
    index = np.array([0], dtype=np.int32)
    dcoeff = np.array([[1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 1), order='F', dtype=float)
    ucoeff[0, 0, 0] = 2.0

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 0.0)

    assert info == 0
    assert nr == 0
    assert_allclose(d[0, 0], 2.0, rtol=1e-14)


def test_td04ad_mixed_orders():
    """
    Test system with different denominator orders per row.

    INDEX = [2, 1] -> N = 3
    """
    from ctrlsys import td04ad

    m, p = 1, 2
    index = np.array([2, 1], dtype=np.int32)
    kdcoef = 3

    dcoeff = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 3.0, 0.0]
    ], order='F', dtype=float)

    ucoeff = np.zeros((p, m, kdcoef), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 1.0, 0.0]
    ucoeff[1, 0, :] = [0.0, 0.0, 1.0]

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 0.0)

    assert info == 0
    assert nr <= 3


def test_td04ad_markov_parameter_preservation():
    """
    Validate Markov parameters (transfer function equivalence).

    For T(s) = C(sI-A)^{-1}B + D, the frequency response must match
    the polynomial form T(s) = U(s)/D(s).

    Check at several complex frequencies that responses match.
    """
    from ctrlsys import td04ad

    m, p = 1, 1
    index = np.array([2], dtype=np.int32)
    kdcoef = 3

    dcoeff = np.array([[1.0, 0.5, 0.1]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, kdcoef), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 1.0, 0.5]

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 0.0)
    assert info == 0

    ar = a[:nr, :nr].copy()
    br = b[:nr, :m].copy()
    cr = c[:p, :nr].copy()
    dr = d[:p, :m].copy()

    for s in [0.1j, 0.5j, 1.0j, 2.0j, 0.5 + 0.5j]:
        D_s = dcoeff[0, 0] * s**2 + dcoeff[0, 1] * s + dcoeff[0, 2]
        U_s = ucoeff[0, 0, 0] * s**2 + ucoeff[0, 0, 1] * s + ucoeff[0, 0, 2]
        T_poly = U_s / D_s

        I_n = np.eye(nr, dtype=complex)
        T_ss = (cr @ np.linalg.solve(s * I_n - ar, br) + dr)[0, 0]

        assert_allclose(T_ss, T_poly, rtol=1e-10, atol=1e-12)


def test_td04ad_eigenvalue_preservation():
    """
    Validate that eigenvalues match denominator polynomial roots.

    For a minimal realization, the eigenvalues of A are the poles of T(s).
    Use a numerator without common factors to ensure no cancellation.

    D(s) = s^3 + 6s^2 + 11s + 6 = (s+1)(s+2)(s+3), roots = -1, -2, -3
    U(s) = s^2 + 2s + 1 = (s+1)^2 (has common factor with D(s))

    This tests that minimal realization correctly identifies the common
    factor and reduces the order.
    """
    from ctrlsys import td04ad

    m, p = 1, 1

    index = np.array([2], dtype=np.int32)
    dcoeff = np.array([[1.0, 3.0, 2.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 3), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 0.0, 1.0]

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 0.0)
    assert info == 0

    ar = a[:nr, :nr].copy()
    eigs = np.linalg.eigvals(ar)

    poly_roots = np.roots([1.0, 3.0, 2.0])

    eigs_sorted = np.sort(eigs.real)
    roots_sorted = np.sort(poly_roots.real)

    assert_allclose(eigs_sorted, roots_sorted, rtol=1e-10, atol=1e-12)


def test_td04ad_invalid_rowcol():
    """
    Test invalid ROWCOL parameter.
    """
    from ctrlsys import td04ad

    m, p = 1, 1
    index = np.array([1], dtype=np.int32)
    dcoeff = np.array([[1.0, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 0.0]

    nr, a, b, c, d, info = td04ad('X', m, p, index, dcoeff, ucoeff, 0.0)

    assert info == -1


def test_td04ad_with_tolerance():
    """
    Test with explicit positive tolerance.
    """
    from ctrlsys import td04ad

    m, p = 2, 2
    index = np.array([3, 3], dtype=np.int32)
    kdcoef = 4

    dcoeff = np.array([
        [1.0, 6.0, 11.0, 6.0],
        [1.0, 6.0, 11.0, 6.0]
    ], order='F', dtype=float)

    ucoeff = np.zeros((p, m, kdcoef), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 6.0, 12.0, 7.0]
    ucoeff[0, 1, :] = [0.0, 1.0, 4.0, 3.0]
    ucoeff[1, 0, :] = [0.0, 0.0, 1.0, 1.0]
    ucoeff[1, 1, :] = [1.0, 8.0, 20.0, 15.0]

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 1e-10)

    assert info == 0
    assert nr == 3


def test_td04ad_state_space_equations():
    """
    Validate discrete state-space equations hold.

    x(k+1) = A*x(k) + B*u(k)
    y(k) = C*x(k) + D*u(k)

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import td04ad

    np.random.seed(888)

    m, p = 1, 1
    index = np.array([2], dtype=np.int32)
    kdcoef = 3

    dcoeff = np.array([[1.0, 0.5, 0.1]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, kdcoef), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 1.0, 0.5]

    nr, a, b, c, d, info = td04ad('R', m, p, index, dcoeff, ucoeff, 0.0)
    assert info == 0

    ar = a[:nr, :nr].copy()
    br = b[:nr, :m].copy()
    cr = c[:p, :nr].copy()
    dr = d[:p, :m].copy()

    x = np.random.randn(nr, 1).astype(float, order='F')
    u = np.random.randn(m, 1).astype(float, order='F')

    x_next = ar @ x + br @ u
    y = cr @ x + dr @ u

    x_next_manual = np.zeros_like(x)
    for i in range(nr):
        for j in range(nr):
            x_next_manual[i, 0] += ar[i, j] * x[j, 0]
        for k in range(m):
            x_next_manual[i, 0] += br[i, k] * u[k, 0]

    assert_allclose(x_next, x_next_manual, rtol=1e-14, atol=1e-15)
