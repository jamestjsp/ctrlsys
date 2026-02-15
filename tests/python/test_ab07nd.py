"""
Tests for AB07ND - Inverse of a given linear system.

Computes the inverse (Ai,Bi,Ci,Di) of a given system (A,B,C,D):
    Ai = A - B*D^-1*C,  Bi = -B*D^-1,  Ci = D^-1*C,  Di = D^-1
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_ab07nd_basic():
    """
    Test basic functionality using SLICOT HTML doc example.

    Input: N=3, M=2
    A = [[1, 2, 0], [4, -1, 0], [0, 0, 1]]
    B = [[1, 0], [0, 1], [1, 0]]
    C = [[0, 1, -1], [0, 0, 1]]
    D = [[4, 0], [0, 1]]

    Expected output:
    Ai = [[1.0, 1.75, 0.25], [4.0, -1.0, -1.0], [0.0, -0.25, 1.25]]
    Bi = [[-0.25, 0], [0, -1], [-0.25, 0]]
    Ci = [[0, 0.25, -0.25], [0, 0, 1]]
    Di = [[0.25, 0], [0, 1]]
    """
    from ctrlsys import ab07nd

    a = np.array([[1.0, 2.0, 0.0],
                  [4.0, -1.0, 0.0],
                  [0.0, 0.0, 1.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0]], order='F', dtype=float)
    c = np.array([[0.0, 1.0, -1.0],
                  [0.0, 0.0, 1.0]], order='F', dtype=float)
    d = np.array([[4.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)

    ai_expected = np.array([[1.0, 1.75, 0.25],
                            [4.0, -1.0, -1.0],
                            [0.0, -0.25, 1.25]], order='F', dtype=float)
    bi_expected = np.array([[-0.25, 0.0],
                            [0.0, -1.0],
                            [-0.25, 0.0]], order='F', dtype=float)
    ci_expected = np.array([[0.0, 0.25, -0.25],
                            [0.0, 0.0, 1.0]], order='F', dtype=float)
    di_expected = np.array([[0.25, 0.0],
                            [0.0, 1.0]], order='F', dtype=float)

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)

    assert info == 0
    assert rcond > 0
    assert_allclose(ai, ai_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(bi, bi_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(ci, ci_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(di, di_expected, rtol=1e-3, atol=1e-4)


def test_ab07nd_double_inverse():
    """
    Test mathematical property: (G^-1)^-1 = G.

    Applying inverse twice should return original system.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab07nd

    np.random.seed(42)
    n, m = 3, 2

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(m, n).astype(float, order='F')
    d_orig = np.eye(m, order='F', dtype=float) + 0.1 * np.random.randn(m, m).astype(float, order='F')

    a1, b1, c1, d1, rcond1, info1 = ab07nd(
        a_orig.copy(), b_orig.copy(), c_orig.copy(), d_orig.copy())
    assert info1 == 0

    a2, b2, c2, d2, rcond2, info2 = ab07nd(a1, b1, c1, d1)
    assert info2 == 0

    assert_allclose(a2, a_orig, rtol=1e-13, atol=1e-14)
    assert_allclose(b2, b_orig, rtol=1e-13, atol=1e-14)
    assert_allclose(c2, c_orig, rtol=1e-13, atol=1e-14)
    assert_allclose(d2, d_orig, rtol=1e-13, atol=1e-14)


def test_ab07nd_transfer_function_inverse():
    """
    Test that transfer function of inverse system is reciprocal.

    H_inv(s) = H(s)^-1 for all s.

    For MIMO: H_inv(s) * H(s) = I

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab07nd

    np.random.seed(123)
    n, m = 2, 2

    a = np.array([[-1.0, 0.0],
                  [0.0, -2.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.5],
                  [0.5, 1.0]], order='F', dtype=float)
    d = np.array([[2.0, 0.0],
                  [0.0, 2.0]], order='F', dtype=float)

    ai, bi, ci, di, rcond, info = ab07nd(
        a.copy(), b.copy(), c.copy(), d.copy())
    assert info == 0

    s_vals = [0.1j, 0.5j, 1.0j, 2.0j, -0.5 + 1.0j]

    for s in s_vals:
        I_n = np.eye(n, dtype=complex)
        I_m = np.eye(m, dtype=complex)

        H = d + c @ np.linalg.solve(s * I_n - a, b)
        H_inv = di + ci @ np.linalg.solve(s * I_n - ai, bi)

        product = H_inv @ H
        assert_allclose(product, I_m, rtol=1e-12, atol=1e-13)


def test_ab07nd_di_equals_d_inverse():
    """
    Test that Di = D^-1 exactly.

    The feedthrough matrix of inverse is just matrix inverse.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import ab07nd

    np.random.seed(456)
    n, m = 4, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    d = np.eye(m, order='F', dtype=float) + 0.5 * np.random.randn(m, m).astype(float, order='F')

    d_orig = d.copy()

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)
    assert info == 0

    d_inv_expected = np.linalg.inv(d_orig)
    assert_allclose(di, d_inv_expected, rtol=1e-14, atol=1e-15)


def test_ab07nd_formulas():
    """
    Test that output matches the explicit formulas.

    Ai = A - B*D^-1*C
    Bi = -B*D^-1
    Ci = D^-1*C
    Di = D^-1

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import ab07nd

    np.random.seed(789)
    n, m = 3, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    d = np.eye(m, order='F', dtype=float) + 0.3 * np.random.randn(m, m).astype(float, order='F')

    a_orig = a.copy()
    b_orig = b.copy()
    c_orig = c.copy()
    d_orig = d.copy()

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)
    assert info == 0

    d_inv = np.linalg.inv(d_orig)
    ai_expected = a_orig - b_orig @ d_inv @ c_orig
    bi_expected = -b_orig @ d_inv
    ci_expected = d_inv @ c_orig
    di_expected = d_inv

    assert_allclose(ai, ai_expected, rtol=1e-14, atol=1e-15)
    assert_allclose(bi, bi_expected, rtol=1e-14, atol=1e-15)
    assert_allclose(ci, ci_expected, rtol=1e-14, atol=1e-15)
    assert_allclose(di, di_expected, rtol=1e-14, atol=1e-15)


def test_ab07nd_identity_d():
    """
    Test with D = I (identity feedthrough).

    When D = I:
    Ai = A - B*C, Bi = -B, Ci = C, Di = I

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import ab07nd

    np.random.seed(111)
    n, m = 3, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    d = np.eye(m, order='F', dtype=float)

    a_orig = a.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)
    assert info == 0

    assert_allclose(ai, a_orig - b_orig @ c_orig, rtol=1e-14, atol=1e-15)
    assert_allclose(bi, -b_orig, rtol=1e-14, atol=1e-15)
    assert_allclose(ci, c_orig, rtol=1e-14, atol=1e-15)
    assert_allclose(di, np.eye(m), rtol=1e-14, atol=1e-15)

    assert rcond > 0.99


def test_ab07nd_siso():
    """
    Test SISO system (M=1).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import ab07nd

    np.random.seed(222)
    n, m = 2, 1

    a = np.array([[-1.0, 0.5],
                  [0.0, -2.0]], order='F', dtype=float)
    b = np.array([[1.0],
                  [0.5]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.array([[2.0]], order='F', dtype=float)

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)
    assert info == 0

    assert_allclose(di, [[0.5]], rtol=1e-14)
    assert rcond > 0


def test_ab07nd_n_zero():
    """
    Test edge case: N=0 (static gain system).

    With N=0, only D matters: Di = D^-1.
    """
    from ctrlsys import ab07nd

    m = 2
    a = np.array([], order='F', dtype=float).reshape(0, 0)
    b = np.array([], order='F', dtype=float).reshape(0, m)
    c = np.array([], order='F', dtype=float).reshape(m, 0)
    d = np.array([[2.0, 0.0],
                  [0.0, 4.0]], order='F', dtype=float)
    d_orig = d.copy()

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)
    assert info == 0

    assert ai.shape == (0, 0)
    assert bi.shape == (0, m)
    assert ci.shape == (m, 0)
    assert_allclose(di, np.linalg.inv(d_orig), rtol=1e-14)


def test_ab07nd_m_zero():
    """
    Test edge case: M=0 (no inputs/outputs).

    With M=0, the system has no I/O, rcond should be 1.
    """
    from ctrlsys import ab07nd

    n = 2
    a = np.array([[1.0, 0.0],
                  [0.0, 2.0]], order='F', dtype=float)
    b = np.array([], order='F', dtype=float).reshape(n, 0)
    c = np.array([], order='F', dtype=float).reshape(0, n)
    d = np.array([], order='F', dtype=float).reshape(0, 0)

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)
    assert info == 0
    assert rcond == 1.0


def test_ab07nd_singular_d():
    """
    Test error handling: singular D matrix.

    Should return info > 0 when D is singular.
    """
    from ctrlsys import ab07nd

    n, m = 2, 2
    a = np.array([[1.0, 0.0],
                  [0.0, 2.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    d = np.array([[1.0, 0.0],
                  [0.0, 0.0]], order='F', dtype=float)

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)
    assert info > 0
    assert rcond == 0.0


def test_ab07nd_nearly_singular_d():
    """
    Test warning: nearly singular D matrix.

    Should return info = M+1 when D is ill-conditioned.
    """
    from ctrlsys import ab07nd

    n, m = 2, 2
    a = np.array([[1.0, 0.0],
                  [0.0, 2.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    eps = np.finfo(float).eps
    d = np.array([[1.0, 0.0],
                  [0.0, eps * 0.1]], order='F', dtype=float)

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)
    assert info == m + 1
    assert rcond < eps


def test_ab07nd_rcond_accuracy():
    """
    Test that rcond reflects actual condition number.

    rcond should be approximately 1/cond(D).
    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import ab07nd

    np.random.seed(333)
    n, m = 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    d = np.diag([1.0, 2.0, 10.0]).astype(float, order='F')

    ai, bi, ci, di, rcond, info = ab07nd(a, b, c, d)
    assert info == 0

    cond_d = np.linalg.cond(d, 1)
    rcond_expected = 1.0 / cond_d

    assert_allclose(rcond, rcond_expected, rtol=0.1)
