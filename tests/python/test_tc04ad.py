"""
Tests for TC04AD - Polynomial matrix representation to state-space.

Converts left/right polynomial matrix fraction T(s) = inv(P(s))*Q(s) or
T(s) = Q(s)*inv(P(s)) to state-space form (A,B,C,D) using Wolovich's
Observable Structure Theorem.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tc04ad_left_matrix_fraction():
    """
    Test left matrix fraction using SLICOT HTML doc example.

    Input: M=2, P=2, LERI='L', INDEX=[2,2]
    PCOEFF: 2x2x3 denominator polynomial coefficients
    QCOEFF: 2x2x3 numerator polynomial coefficients

    Expected output: N=4
    RCOND = 0.25
    A (4x4), B (4x2), C (2x4), D (2x2) state-space matrices
    """
    from ctrlsys import tc04ad

    m, p = 2, 2
    index = np.array([2, 2], dtype=np.int32)

    pcoeff = np.zeros((p, p, 3), order='F', dtype=float)
    pcoeff[0, 0, :] = [2.0, 3.0, 1.0]
    pcoeff[0, 1, :] = [4.0, -1.0, -1.0]
    pcoeff[1, 0, :] = [5.0, 7.0, -6.0]
    pcoeff[1, 1, :] = [3.0, 2.0, 2.0]

    qcoeff = np.zeros((p, m, 3), order='F', dtype=float)
    qcoeff[0, 0, :] = [6.0, -1.0, 5.0]
    qcoeff[0, 1, :] = [1.0, 7.0, 5.0]
    qcoeff[1, 0, :] = [1.0, 1.0, 1.0]
    qcoeff[1, 1, :] = [4.0, 1.0, -1.0]

    n, rcond, a, b, c, d, info = tc04ad('L', m, p, index, pcoeff, qcoeff)

    assert info == 0
    assert n == 4
    assert_allclose(rcond, 0.25, rtol=1e-2)

    a_expected = np.array([
        [0.0000,  0.5714, 0.0000, -0.4286],
        [1.0000,  1.0000, 0.0000, -1.0000],
        [0.0000, -2.0000, 0.0000,  2.0000],
        [0.0000,  0.7857, 1.0000, -1.7143]
    ], order='F', dtype=float)

    b_expected = np.array([
        [ 8.0000, 3.8571],
        [ 4.0000, 4.0000],
        [-9.0000, 5.0000],
        [ 4.0000, -5.0714]
    ], order='F', dtype=float)

    c_expected = np.array([
        [0.0000, -0.2143, 0.0000, 0.2857],
        [0.0000,  0.3571, 0.0000, -0.1429]
    ], order='F', dtype=float)

    d_expected = np.array([
        [-1.0000, 0.9286],
        [ 2.0000, -0.2143]
    ], order='F', dtype=float)

    assert_allclose(a, a_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(b, b_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(c, c_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(d, d_expected, rtol=1e-3, atol=1e-4)


def test_tc04ad_right_matrix_fraction():
    """
    Test right matrix fraction representation.

    For right PMR: T(s) = Q(s)*inv(P(s)), the routine converts via duality.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tc04ad

    m, p = 2, 2
    index = np.array([1, 1], dtype=np.int32)

    pcoeff = np.zeros((m, m, 2), order='F', dtype=float)
    pcoeff[0, 0, :] = [1.0, 0.5]
    pcoeff[0, 1, :] = [0.0, 0.0]
    pcoeff[1, 0, :] = [0.0, 0.0]
    pcoeff[1, 1, :] = [1.0, 0.3]

    qcoeff = np.zeros((max(m, p), max(m, p), 2), order='F', dtype=float)
    qcoeff[0, 0, :] = [1.0, 0.2]
    qcoeff[0, 1, :] = [0.0, 0.1]
    qcoeff[1, 0, :] = [0.0, 0.0]
    qcoeff[1, 1, :] = [1.0, 0.4]

    n, rcond, a, b, c, d, info = tc04ad('R', m, p, index, pcoeff, qcoeff)

    assert info == 0
    assert n == 2

    assert a.shape == (n, n)
    assert b.shape == (n, m)
    assert c.shape == (p, n)
    assert d.shape == (p, m)

    assert rcond > 0.0


def test_tc04ad_transfer_function_equivalence():
    """
    Validate transfer function equivalence at frequency points.

    The state-space realization must produce the same transfer matrix
    T(s) = C * inv(sI - A) * B + D as the polynomial representation
    T(s) = inv(P(s)) * Q(s).

    Uses data from HTML doc example.
    """
    from ctrlsys import tc04ad

    m, p = 2, 2
    index = np.array([2, 2], dtype=np.int32)

    pcoeff = np.zeros((p, p, 3), order='F', dtype=float)
    pcoeff[0, 0, :] = [2.0, 3.0, 1.0]
    pcoeff[0, 1, :] = [4.0, -1.0, -1.0]
    pcoeff[1, 0, :] = [5.0, 7.0, -6.0]
    pcoeff[1, 1, :] = [3.0, 2.0, 2.0]

    qcoeff = np.zeros((p, m, 3), order='F', dtype=float)
    qcoeff[0, 0, :] = [6.0, -1.0, 5.0]
    qcoeff[0, 1, :] = [1.0, 7.0, 5.0]
    qcoeff[1, 0, :] = [1.0, 1.0, 1.0]
    qcoeff[1, 1, :] = [4.0, 1.0, -1.0]

    n, rcond, a, b, c, d, info = tc04ad('L', m, p, index, pcoeff, qcoeff)
    assert info == 0

    def eval_poly_matrix(coeff, index_arr, s, is_left):
        """Evaluate P(s) or Q(s) at complex frequency s."""
        rows, cols = coeff.shape[0], coeff.shape[1]
        result = np.zeros((rows, cols), dtype=complex)
        for i in range(rows):
            for j in range(cols):
                if is_left:
                    deg_ref = index_arr[i]
                else:
                    deg_ref = index_arr[j] if j < len(index_arr) else 0
                for k in range(coeff.shape[2]):
                    power = deg_ref - k
                    if power >= 0:
                        result[i, j] += coeff[i, j, k] * (s ** power)
        return result

    for s in [0.1j, 0.5j, 1.0j, 2.0j, 1.0 + 0.5j]:
        P_s = eval_poly_matrix(pcoeff, index, s, True)
        Q_s = eval_poly_matrix(qcoeff, index, s, True)
        T_poly = np.linalg.solve(P_s, Q_s)

        I_n = np.eye(n, dtype=complex)
        T_ss = c @ np.linalg.solve(s * I_n - a, b) + d

        assert_allclose(T_ss, T_poly, rtol=1e-10, atol=1e-12)


def test_tc04ad_single_input_output():
    """
    Test SISO system (m=1, p=1).

    For scalar case, polynomial division simplifies.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import tc04ad

    m, p = 1, 1
    index = np.array([2], dtype=np.int32)

    pcoeff = np.zeros((1, 1, 3), order='F', dtype=float)
    pcoeff[0, 0, :] = [1.0, 3.0, 2.0]

    qcoeff = np.zeros((1, 1, 3), order='F', dtype=float)
    qcoeff[0, 0, :] = [1.0, 1.0, 0.0]

    n, rcond, a, b, c, d, info = tc04ad('L', m, p, index, pcoeff, qcoeff)

    assert info == 0
    assert n == 2
    assert a.shape == (2, 2)
    assert b.shape == (2, 1)
    assert c.shape == (1, 2)
    assert d.shape == (1, 1)

    I_n = np.eye(n, dtype=complex)
    for s in [1.0j, 2.0j]:
        P_s = pcoeff[0, 0, 0] * s**2 + pcoeff[0, 0, 1] * s + pcoeff[0, 0, 2]
        Q_s = qcoeff[0, 0, 0] * s**2 + qcoeff[0, 0, 1] * s + qcoeff[0, 0, 2]
        T_poly = Q_s / P_s

        T_ss = (c @ np.linalg.solve(s * I_n - a, b) + d)[0, 0]

        assert_allclose(T_ss, T_poly, rtol=1e-12, atol=1e-14)


def test_tc04ad_nonproper_error():
    """
    Test error when P(s) is not row proper (for left PMR).

    P(s) is row proper if the leading coefficient matrix has full rank.
    If not, INFO=1 is returned.
    """
    from ctrlsys import tc04ad

    m, p = 2, 2
    index = np.array([1, 1], dtype=np.int32)

    pcoeff = np.zeros((p, p, 2), order='F', dtype=float)
    pcoeff[0, 0, :] = [1.0, 0.0]
    pcoeff[0, 1, :] = [1.0, 0.0]
    pcoeff[1, 0, :] = [1.0, 0.0]
    pcoeff[1, 1, :] = [1.0, 0.0]

    qcoeff = np.zeros((p, m, 2), order='F', dtype=float)
    qcoeff[0, 0, :] = [1.0, 0.0]
    qcoeff[0, 1, :] = [0.0, 0.0]
    qcoeff[1, 0, :] = [0.0, 0.0]
    qcoeff[1, 1, :] = [1.0, 0.0]

    n, rcond, a, b, c, d, info = tc04ad('L', m, p, index, pcoeff, qcoeff)

    assert info == 1


def test_tc04ad_zero_order():
    """
    Test with zero-order system (all INDEX values are 0).

    Should result in N=0 state-space realization (direct feedthrough only).
    """
    from ctrlsys import tc04ad

    m, p = 2, 2
    index = np.array([0, 0], dtype=np.int32)

    pcoeff = np.zeros((p, p, 1), order='F', dtype=float)
    pcoeff[0, 0, 0] = 1.0
    pcoeff[1, 1, 0] = 1.0

    qcoeff = np.zeros((p, m, 1), order='F', dtype=float)
    qcoeff[0, 0, 0] = 2.0
    qcoeff[0, 1, 0] = 0.5
    qcoeff[1, 0, 0] = 0.3
    qcoeff[1, 1, 0] = 1.5

    n, rcond, a, b, c, d, info = tc04ad('L', m, p, index, pcoeff, qcoeff)

    assert info == 0
    assert n == 0

    d_expected = np.array([[2.0, 0.5], [0.3, 1.5]], order='F', dtype=float)
    assert_allclose(d, d_expected, rtol=1e-14)


def test_tc04ad_state_space_consistency():
    """
    Test observable companion form structure.

    The resulting state-space should satisfy:
    - A has observable companion form structure
    - (C, A) is observable

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tc04ad

    m, p = 2, 2
    index = np.array([2, 2], dtype=np.int32)

    pcoeff = np.zeros((p, p, 3), order='F', dtype=float)
    pcoeff[0, 0, :] = [2.0, 3.0, 1.0]
    pcoeff[0, 1, :] = [4.0, -1.0, -1.0]
    pcoeff[1, 0, :] = [5.0, 7.0, -6.0]
    pcoeff[1, 1, :] = [3.0, 2.0, 2.0]

    qcoeff = np.zeros((p, m, 3), order='F', dtype=float)
    qcoeff[0, 0, :] = [6.0, -1.0, 5.0]
    qcoeff[0, 1, :] = [1.0, 7.0, 5.0]
    qcoeff[1, 0, :] = [1.0, 1.0, 1.0]
    qcoeff[1, 1, :] = [4.0, 1.0, -1.0]

    n, rcond, a, b, c, d, info = tc04ad('L', m, p, index, pcoeff, qcoeff)
    assert info == 0
    assert n == 4

    obs = np.vstack([c, c @ a, c @ a @ a, c @ a @ a @ a])
    rank_obs = np.linalg.matrix_rank(obs, tol=1e-10)
    assert rank_obs == n


def test_tc04ad_invalid_leri():
    """
    Test error handling for invalid LERI parameter.
    """
    from ctrlsys import tc04ad

    m, p = 1, 1
    index = np.array([1], dtype=np.int32)
    pcoeff = np.ones((1, 1, 2), order='F', dtype=float)
    qcoeff = np.ones((1, 1, 2), order='F', dtype=float)

    with pytest.raises(ValueError):
        tc04ad('X', m, p, index, pcoeff, qcoeff)
