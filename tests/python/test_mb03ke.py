"""
Tests for MB03KE - Periodic Sylvester-like equation solver.

MB03KE solves small periodic Sylvester-like equations (PSLE):
  op(A(i))*X(i)   + isgn*X(i+1)*op(B(i)) = -scale*C(i), S(i) =  1
  op(A(i))*X(i+1) + isgn*X(i)  *op(B(i)) = -scale*C(i), S(i) = -1

for i = 1, ..., K, where A, B, C are K-periodic matrix sequences,
A(i) are M-by-M, B(i) are N-by-N, with 1 <= M, N <= 2.
"""

import numpy as np
import pytest
from ctrlsys import mb03ke


def test_mb03ke_basic_k2_m1_n1():
    """
    Basic test: K=2, M=1, N=1 (scalar case).

    Solve: A(i)*X(i) + isgn*X(i+1)*B(i) = -scale*C(i) for S(i)=1
    With simple scalar matrices.

    Random seed: 42 (for reproducibility)
    """
    k, m, n = 2, 1, 1
    trana, tranb = False, False
    isgn = 1

    np.random.seed(42)

    a = np.array([2.0, 3.0], dtype=float, order='F')
    b = np.array([1.5, 0.5], dtype=float, order='F')
    c = np.array([1.0, 2.0], dtype=float, order='F')
    s = np.array([1, 1], dtype=np.int32)

    c_orig = c.copy()
    x, scale, info = mb03ke(trana, tranb, isgn, k, m, n, s, a, b, c)

    assert info == 0
    assert 0.0 < scale <= 1.0

    for i in range(k):
        i_next = (i + 1) % k
        if s[i] == 1:
            lhs = a[i] * x[i] + isgn * x[i_next] * b[i]
        else:
            lhs = a[i] * x[i_next] + isgn * x[i] * b[i]
        rhs = -scale * c_orig[i]
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


def test_mb03ke_k3_m2_n1():
    """
    Test with K=3, M=2, N=1: 2x1 matrices.

    Random seed: 123 (for reproducibility)
    """
    k, m, n = 3, 2, 1
    trana, tranb = False, False
    isgn = 1

    np.random.seed(123)

    a = np.array([
        1.0, 0.5, 0.0, 2.0,
        1.5, 0.3, 0.0, 1.8,
        2.0, 0.4, 0.0, 1.6
    ], dtype=float, order='F')

    b = np.array([1.0, 1.5, 2.0], dtype=float, order='F')

    c = np.array([
        1.0, 0.5,
        2.0, 1.0,
        1.5, 0.75
    ], dtype=float, order='F')

    s = np.array([1, 1, 1], dtype=np.int32)

    x, scale, info = mb03ke(trana, tranb, isgn, k, m, n, s, a, b, c)

    assert info == 0
    assert 0.0 < scale <= 1.0


def test_mb03ke_k2_m2_n2():
    """
    Test with K=2, M=2, N=2: full 2x2 matrices.

    Random seed: 456 (for reproducibility)
    """
    k, m, n = 2, 2, 2
    trana, tranb = False, False
    isgn = 1

    np.random.seed(456)

    a = np.array([
        1.0, 0.2, 0.0, 1.5,
        2.0, 0.3, 0.0, 1.8
    ], dtype=float, order='F')

    b = np.array([
        1.0, 0.1, 0.0, 1.2,
        1.5, 0.2, 0.0, 0.8
    ], dtype=float, order='F')

    c = np.array([
        1.0, 0.5, 0.2, 0.3,
        0.8, 0.4, 0.1, 0.2
    ], dtype=float, order='F')

    s = np.array([1, 1], dtype=np.int32)

    x, scale, info = mb03ke(trana, tranb, isgn, k, m, n, s, a, b, c)

    assert info == 0
    assert 0.0 < scale <= 1.0


def test_mb03ke_negative_isgn():
    """
    Test with isgn = -1.

    Random seed: 789 (for reproducibility)
    """
    k, m, n = 2, 1, 1
    trana, tranb = False, False
    isgn = -1

    np.random.seed(789)

    a = np.array([2.0, 3.0], dtype=float, order='F')
    b = np.array([1.5, 0.5], dtype=float, order='F')
    c = np.array([1.0, 2.0], dtype=float, order='F')
    s = np.array([1, 1], dtype=np.int32)

    x, scale, info = mb03ke(trana, tranb, isgn, k, m, n, s, a, b, c)

    assert info == 0
    assert 0.0 < scale <= 1.0


def test_mb03ke_mixed_signatures():
    """
    Test with mixed signatures S(i) = 1 and S(i) = -1.

    Random seed: 999 (for reproducibility)
    """
    k, m, n = 3, 1, 1
    trana, tranb = False, False
    isgn = 1

    np.random.seed(999)

    a = np.array([2.0, 3.0, 1.5], dtype=float, order='F')
    b = np.array([1.5, 0.5, 2.0], dtype=float, order='F')
    c = np.array([1.0, 2.0, 1.5], dtype=float, order='F')
    s = np.array([1, -1, 1], dtype=np.int32)

    x, scale, info = mb03ke(trana, tranb, isgn, k, m, n, s, a, b, c)

    assert info == 0
    assert 0.0 < scale <= 1.0


def test_mb03ke_transpose_a():
    """
    Test with transposed A matrices (trana=True).

    Random seed: 1111 (for reproducibility)
    """
    k, m, n = 2, 2, 1
    trana, tranb = True, False
    isgn = 1

    np.random.seed(1111)

    a = np.array([
        1.0, 0.2, 0.3, 1.5,
        2.0, 0.3, 0.1, 1.8
    ], dtype=float, order='F')

    b = np.array([1.0, 1.5], dtype=float, order='F')

    c = np.array([
        1.0, 0.5,
        0.8, 0.4
    ], dtype=float, order='F')

    s = np.array([1, 1], dtype=np.int32)

    x, scale, info = mb03ke(trana, tranb, isgn, k, m, n, s, a, b, c)

    assert info == 0
    assert 0.0 < scale <= 1.0


def test_mb03ke_transpose_b():
    """
    Test with transposed B matrices (tranb=True).

    Random seed: 2222 (for reproducibility)
    """
    k, m, n = 2, 1, 2
    trana, tranb = False, True
    isgn = 1

    np.random.seed(2222)

    a = np.array([2.0, 3.0], dtype=float, order='F')

    b = np.array([
        1.0, 0.1, 0.2, 1.2,
        1.5, 0.2, 0.3, 0.8
    ], dtype=float, order='F')

    c = np.array([
        1.0, 0.5,
        0.8, 0.4
    ], dtype=float, order='F')

    s = np.array([1, 1], dtype=np.int32)

    x, scale, info = mb03ke(trana, tranb, isgn, k, m, n, s, a, b, c)

    assert info == 0
    assert 0.0 < scale <= 1.0


def test_mb03ke_workspace_query():
    """
    Test workspace query (ldwork = -1).
    """
    k, m, n = 3, 2, 2
    trana, tranb = False, False
    isgn = 1

    a = np.zeros(m * m * k, dtype=float, order='F')
    b = np.zeros(n * n * k, dtype=float, order='F')
    c = np.zeros(m * n * k, dtype=float, order='F')
    s = np.ones(k, dtype=np.int32)

    result = mb03ke(trana, tranb, isgn, k, m, n, s, a, b, c, ldwork=-1)
    x, scale, info = result[0], result[2], result[3]

    assert info == 0
    assert x is None


def test_mb03ke_residual_verification():
    """
    Verify residual of the periodic Sylvester-like equation solution.

    For S(i) = 1: op(A(i))*X(i) + isgn*X(i+1)*op(B(i)) = -scale*C(i)
    For S(i) = -1: op(A(i))*X(i+1) + isgn*X(i)*op(B(i)) = -scale*C(i)

    Random seed: 3333 (for reproducibility)
    """
    k, m, n = 2, 1, 1
    trana, tranb = False, False
    isgn = 1

    np.random.seed(3333)

    a = np.array([2.5, 1.8], dtype=float, order='F')
    b = np.array([1.2, 0.9], dtype=float, order='F')
    c_orig = np.array([1.0, 2.0], dtype=float, order='F')
    s = np.array([1, 1], dtype=np.int32)

    c = c_orig.copy()
    x, scale, info = mb03ke(trana, tranb, isgn, k, m, n, s, a, b, c)

    assert info == 0

    for i in range(k):
        ip1 = (i + 1) % k

        ai = a[i * m * m:(i + 1) * m * m].reshape((m, m), order='F')
        bi = b[i * n * n:(i + 1) * n * n].reshape((n, n), order='F')
        xi = x[i * m * n:(i + 1) * m * n].reshape((m, n), order='F')
        xip1 = x[ip1 * m * n:(ip1 + 1) * m * n].reshape((m, n), order='F')
        ci = c_orig[i * m * n:(i + 1) * m * n].reshape((m, n), order='F')

        if trana:
            opA = ai.T
        else:
            opA = ai
        if tranb:
            opB = bi.T
        else:
            opB = bi

        if s[i] == 1:
            lhs = opA @ xi + isgn * xip1 @ opB
        else:
            lhs = opA @ xip1 + isgn * xi @ opB

        rhs = -scale * ci

        np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)
