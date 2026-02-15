"""
Tests for TD03AD - Polynomial matrix representation for proper transfer matrix.

TD03AD finds a relatively prime left or right polynomial matrix representation
for a proper transfer matrix T(s) given as row or column polynomial vectors
over common denominator polynomials.

For LERI='L' (left): T(s) = inv(P(s)) * Q(s)
For LERI='R' (right): T(s) = Q(s) * inv(P(s))

Also computes minimal state-space representation (A,B,C,D) en route.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest


def test_td03ad_html_doc_example():
    """
    Test case from SLICOT HTML documentation.

    Input: M=2, P=2, TOL=0.0, ROWCOL='R', LERI='L', EQUIL='N'
    INDEXD = [3, 3]

    DCOEFF (row-wise):
        Row 1: 1.0  6.0 11.0  6.0
        Row 2: 1.0  6.0 11.0  6.0

    UCOEFF read as ((UCOEFF(I,J,K), K=1,KDCOEF), J=1,M), I=1,P):
        U(1,1,:) = [1.0, 6.0, 12.0, 7.0]
        U(1,2,:) = [0.0, 1.0,  4.0, 3.0]
        U(2,1,:) = [0.0, 0.0,  1.0, 1.0]
        U(2,2,:) = [1.0, 8.0, 20.0, 15.0]

    Expected:
        NR = 3
        A (3x3), B (3x2), C (2x3), D (2x2)
        INDEXP = [2, 1]
        PCOEFF, QCOEFF polynomial matrices
    """
    from ctrlsys import td03ad

    m, p = 2, 2
    rowcol = 'R'
    leri = 'L'
    equil = 'N'
    tol = 0.0

    indexd = np.array([3, 3], dtype=np.int32)
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

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        rowcol, leri, equil, m, p, indexd, dcoeff, ucoeff, tol
    )

    assert info == 0
    assert nr == 3

    a_expected = np.array([
        [0.5000, 0.9478, 10.1036],
        [0.0000, -1.0000, 0.0000],
        [-0.8660, -0.6156, -5.5000]
    ], order='F', dtype=float)

    b_expected = np.array([
        [2.0000, 12.5000],
        [0.0000, -5.6273],
        [0.0000, -2.0207]
    ], order='F', dtype=float)

    c_expected = np.array([
        [0.0000, 0.0296, -0.5774],
        [0.0000, -0.1481, -0.5774]
    ], order='F', dtype=float)

    d_expected = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    indexp_expected = np.array([2, 1], dtype=np.int32)

    assert_allclose(a[:nr, :nr], a_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(b[:nr, :m], b_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(c[:p, :nr], c_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(d[:p, :m], d_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(indexp[:p], indexp_expected)


def test_td03ad_pcoeff_qcoeff_values():
    """
    Verify PCOEFF and QCOEFF from HTML doc example.

    From HTML doc results:
    The denominator matrix P(s):
        P(1,1,:) = [1.6667, 4.3333, 6.6667]
        P(1,2,:) = [0.3333, 5.6667, 5.3333]
        P(2,1,:) = [5.6273, 5.6273, 0.0000]
        P(2,2,:) = [-5.6273, -5.6273, 0.0000]

    The numerator matrix Q(s):
        Q(1,1,:) = [1.6667, 4.3333, 8.6667]
        Q(1,2,:) = [0.3333, 8.0000, 16.6667]
        Q(2,1,:) = [5.6273, 5.6273, 0.0000]
        Q(2,2,:) = [-5.6273, -11.2546, 0.0000]
    """
    from ctrlsys import td03ad

    m, p = 2, 2
    indexd = np.array([3, 3], dtype=np.int32)
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

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'L', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == 0
    assert nr == 3

    kpcoef = int(indexp[0]) + 1

    p_expected = np.zeros((p, p, kpcoef), order='F', dtype=float)
    p_expected[0, 0, :kpcoef] = [1.6667, 4.3333, 6.6667][:kpcoef]
    p_expected[0, 1, :kpcoef] = [0.3333, 5.6667, 5.3333][:kpcoef]
    p_expected[1, 0, :kpcoef] = [5.6273, 5.6273, 0.0000][:kpcoef]
    p_expected[1, 1, :kpcoef] = [-5.6273, -5.6273, 0.0000][:kpcoef]

    q_expected = np.zeros((p, p, kpcoef), order='F', dtype=float)
    q_expected[0, 0, :kpcoef] = [1.6667, 4.3333, 8.6667][:kpcoef]
    q_expected[0, 1, :kpcoef] = [0.3333, 8.0000, 16.6667][:kpcoef]
    q_expected[1, 0, :kpcoef] = [5.6273, 5.6273, 0.0000][:kpcoef]
    q_expected[1, 1, :kpcoef] = [-5.6273, -11.2546, 0.0000][:kpcoef]

    assert_allclose(pcoeff[:p, :p, :kpcoef], p_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(qcoeff[:p, :p, :kpcoef], q_expected, rtol=1e-3, atol=1e-4)


def test_td03ad_siso_first_order():
    """
    Test simple SISO first-order transfer function.

    T(s) = (2s + 1) / (s + 3)

    With left PMR: T(s) = inv(P(s)) * Q(s)
    """
    from ctrlsys import td03ad

    m, p = 1, 1
    indexd = np.array([1], dtype=np.int32)

    dcoeff = np.array([[1.0, 3.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [2.0, 1.0]

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'L', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == 0
    assert nr == 1

    assert_allclose(d[0, 0], 2.0, rtol=1e-14)


def test_td03ad_right_pmr():
    """
    Test right polynomial matrix representation.

    With LERI='R': T(s) = Q(s) * inv(P(s))
    """
    from ctrlsys import td03ad

    m, p = 2, 2
    indexd = np.array([1, 1], dtype=np.int32)

    dcoeff = np.array([
        [1.0, 2.0],
        [1.0, 3.0]
    ], order='F', dtype=float)

    ucoeff = np.zeros((p, m, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 1.0]
    ucoeff[0, 1, :] = [0.0, 0.5]
    ucoeff[1, 0, :] = [0.0, 0.5]
    ucoeff[1, 1, :] = [1.0, 1.0]

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'R', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == 0
    assert nr <= 2


def test_td03ad_column_form():
    """
    Test with ROWCOL='C' (columns over common denominators).
    """
    from ctrlsys import td03ad

    m, p = 2, 2
    maxmp = max(m, p)
    indexd = np.array([1, 1], dtype=np.int32)

    dcoeff = np.array([
        [1.0, 2.0],
        [1.0, 3.0]
    ], order='F', dtype=float)

    ucoeff = np.zeros((maxmp, maxmp, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 1.0]
    ucoeff[0, 1, :] = [0.0, 0.5]
    ucoeff[1, 0, :] = [0.0, 0.5]
    ucoeff[1, 1, :] = [1.0, 1.0]

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'C', 'L', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == 0
    assert nr <= 2


def test_td03ad_with_equil():
    """
    Test with EQUIL='S' (scaling enabled).
    """
    from ctrlsys import td03ad

    m, p = 2, 2
    indexd = np.array([3, 3], dtype=np.int32)
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

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'L', 'S', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == 0
    assert nr == 3


def test_td03ad_transfer_function_equivalence():
    """
    Validate transfer function equivalence at sample frequencies.

    For T(s) = C(sI-A)^{-1}B + D, the frequency response must match
    the polynomial form T(s) = inv(D(s)) * U(s).
    """
    from ctrlsys import td03ad

    m, p = 1, 1
    indexd = np.array([2], dtype=np.int32)
    kdcoef = 3

    dcoeff = np.array([[1.0, 2.0, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, kdcoef), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 1.0, 0.0]

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'L', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

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


def test_td03ad_zero_leading_coefficient_error():
    """
    Test error when leading denominator coefficient is near zero.

    INFO = I when row I has leading coefficient too small.
    """
    from ctrlsys import td03ad

    m, p = 1, 1
    indexd = np.array([1], dtype=np.int32)

    dcoeff = np.array([[1e-320, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 0.0]

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'L', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == 1


def test_td03ad_invalid_rowcol():
    """
    Test invalid ROWCOL parameter.
    """
    from ctrlsys import td03ad

    m, p = 1, 1
    indexd = np.array([1], dtype=np.int32)
    dcoeff = np.array([[1.0, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 0.0]

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'X', 'L', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == -1


def test_td03ad_invalid_leri():
    """
    Test invalid LERI parameter.
    """
    from ctrlsys import td03ad

    m, p = 1, 1
    indexd = np.array([1], dtype=np.int32)
    dcoeff = np.array([[1.0, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 0.0]

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'X', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == -2


def test_td03ad_invalid_equil():
    """
    Test invalid EQUIL parameter.
    """
    from ctrlsys import td03ad

    m, p = 1, 1
    indexd = np.array([1], dtype=np.int32)
    dcoeff = np.array([[1.0, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 0.0]

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'L', 'X', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == -3


def test_td03ad_polynomial_matrix_identity():
    """
    Validate polynomial matrix representation.

    For left PMR: T(s) = inv(P(s)) * Q(s)
    Check at sample frequency that P(s)*T(s) = Q(s).
    """
    from ctrlsys import td03ad

    m, p = 1, 1
    indexd = np.array([2], dtype=np.int32)
    kdcoef = 3

    dcoeff = np.array([[1.0, 0.5, 0.1]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, kdcoef), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 1.0, 0.5]

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'L', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == 0

    kpcoef = int(indexp[0]) + 1 if nr > 0 else 1
    pormp = p

    ar = a[:nr, :nr].copy() if nr > 0 else np.zeros((0, 0))
    br = b[:nr, :m].copy() if nr > 0 else np.zeros((0, m))
    cr = c[:p, :nr].copy() if nr > 0 else np.zeros((p, 0))
    dr = d[:p, :m].copy()

    for s in [0.5j, 1.0j, 2.0j]:
        if nr > 0:
            I_n = np.eye(nr, dtype=complex)
            T_s = cr @ np.linalg.solve(s * I_n - ar, br) + dr
        else:
            T_s = dr.astype(complex)

        P_s = np.zeros((pormp, pormp), dtype=complex)
        Q_s = np.zeros((pormp, m), dtype=complex)

        for i in range(pormp):
            deg = int(indexp[i])
            for j in range(pormp):
                for k in range(kpcoef):
                    power = deg - k
                    if power >= 0:
                        P_s[i, j] += pcoeff[i, j, k] * (s ** power)

            for j in range(m):
                for k in range(kpcoef):
                    power = deg - k
                    if power >= 0:
                        Q_s[i, j] += qcoeff[i, j, k] * (s ** power)

        lhs = P_s @ T_s
        rhs = Q_s

        assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-10)


def test_td03ad_zero_order():
    """
    Test system with zero order (all INDEXD=0).

    When all denominators are degree 0, N=0 and only D matrix exists.
    """
    from ctrlsys import td03ad

    m, p = 1, 1
    indexd = np.array([0], dtype=np.int32)
    dcoeff = np.array([[1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 1), order='F', dtype=float)
    ucoeff[0, 0, 0] = 2.0

    (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info) = td03ad(
        'R', 'L', 'N', m, p, indexd, dcoeff, ucoeff, 0.0
    )

    assert info == 0
    assert nr == 0
    assert_allclose(d[0, 0], 2.0, rtol=1e-14)
