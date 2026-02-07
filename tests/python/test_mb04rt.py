"""
Tests for MB04RT - Blocked generalized real Sylvester equation solver.

Solves using Level 3 BLAS:
    A * R - L * B = scale * C
    D * R - L * E = scale * F

where (A, D) and (B, E) are in generalized Schur form (A, B upper quasi-triangular,
D, E upper triangular).

MB04RT is the blocked version of MB04RS, using optimal block sizes from ILAENV.
Solution is aborted early if any element exceeds PMAX.
"""

import numpy as np
import pytest
from slicot import mb04rt


def test_mb04rt_basic_1x1():
    """
    Test 1x1 system - smallest case, goes through unblocked path.

    Random seed: 42 (for reproducibility)
    """
    m, n = 1, 1
    pmax = 1e10

    a = np.array([[2.0]], order='F', dtype=float)
    b = np.array([[3.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[1.0]], order='F', dtype=float)
    e = np.array([[4.0]], order='F', dtype=float)
    f = np.array([[2.0]], order='F', dtype=float)

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rt(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-14)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-14)


def test_mb04rt_2x2_diagonal():
    """
    Test 2x2 diagonal system.

    Eigenvalues of (A,D): 1/0.5=2, 3/1=3
    Eigenvalues of (B,E): 4/1=4, 5/2=2.5
    Distinct eigenvalues ensure well-posed system.

    Random seed: 123 (for reproducibility)
    """
    m, n = 2, 2
    pmax = 1e10

    a = np.diag([1.0, 3.0]).astype(float, order='F')
    b = np.diag([4.0, 5.0]).astype(float, order='F')
    d = np.diag([0.5, 1.0]).astype(float, order='F')
    e = np.diag([1.0, 2.0]).astype(float, order='F')

    c = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    f = np.array([[0.5, 1.0], [1.5, 2.0]], order='F', dtype=float)

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rt(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-13, atol=1e-14)


def test_mb04rt_2x2_block_a():
    """
    Test with 2x2 block in A (quasi-triangular, complex eigenvalues).

    Random seed: 456 (for reproducibility)
    """
    m, n = 2, 1
    pmax = 1e10

    a = np.array([[1.0, 2.0],
                  [-0.5, 1.0]], order='F', dtype=float)
    b = np.array([[3.0]], order='F', dtype=float)

    d = np.array([[1.0, 0.5],
                  [0.0, 1.0]], order='F', dtype=float)
    e = np.array([[2.0]], order='F', dtype=float)

    c = np.array([[1.0], [2.0]], order='F', dtype=float)
    f = np.array([[0.5], [1.0]], order='F', dtype=float)

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rt(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-13, atol=1e-14)


def test_mb04rt_larger_system_blocked():
    """
    Test larger system that should trigger blocked algorithm.

    Uses 8x6 system with mixed 1x1 and 2x2 blocks to exercise block partitioning.
    Block detection: 2x2 block when a[i, i-1] != 0.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    m, n = 8, 6
    pmax = 1e10

    a = np.zeros((m, m), order='F', dtype=float)
    a[0, 0] = 1.0
    a[0, 1] = 2.0
    a[1, 0] = -0.5
    a[1, 1] = 1.0
    a[2, 2] = 2.5
    a[2, 3] = 0.1
    a[3, 3] = 3.0
    a[4, 4] = 1.5
    a[4, 5] = 1.2
    a[5, 4] = -0.4
    a[5, 5] = 1.5
    a[6, 6] = 4.0
    a[7, 7] = 5.0
    for i in range(m):
        for j in range(i + 2, m):
            a[i, j] = np.random.randn() * 0.1

    b = np.zeros((n, n), order='F', dtype=float)
    b[0, 0] = 2.0
    b[1, 1] = 1.5
    b[1, 2] = 0.8
    b[2, 1] = -0.3
    b[2, 2] = 1.5
    b[3, 3] = 3.5
    b[4, 4] = 2.5
    b[5, 5] = 4.0
    for i in range(n):
        for j in range(i + 2, n):
            b[i, j] = np.random.randn() * 0.1

    d = np.eye(m, order='F', dtype=float)
    for i in range(m):
        d[i, i] = 0.5 + i * 0.2
    for i in range(m):
        for j in range(i + 1, m):
            d[i, j] = np.random.randn() * 0.05

    e = np.eye(n, order='F', dtype=float)
    for i in range(n):
        e[i, i] = 1.0 + i * 0.3
    for i in range(n):
        for j in range(i + 1, n):
            e[i, j] = np.random.randn() * 0.05

    c = np.random.randn(m, n).astype(float, order='F')
    f = np.random.randn(m, n).astype(float, order='F')

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rt(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-12, atol=1e-13)


def test_mb04rt_pmax_exceeded():
    """
    Test that INFO=1 when solution exceeds PMAX.

    Random seed: 111 (for reproducibility)
    """
    m, n = 1, 1
    pmax = 0.001

    a = np.array([[0.001]], order='F', dtype=float)
    b = np.array([[0.001]], order='F', dtype=float)
    c = np.array([[100.0]], order='F', dtype=float)
    d = np.array([[0.001]], order='F', dtype=float)
    e = np.array([[0.002]], order='F', dtype=float)
    f = np.array([[100.0]], order='F', dtype=float)

    r, l_mat, scale, info = mb04rt(m, n, pmax, a, b, c, d, e, f)

    assert info == 1


def test_mb04rt_zero_m():
    """
    Test quick return for M=0.
    """
    pmax = 1e10

    a = np.array([], order='F', dtype=float).reshape(0, 0)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([], order='F', dtype=float).reshape(0, 1)
    d = np.array([], order='F', dtype=float).reshape(0, 0)
    e = np.array([[1.0]], order='F', dtype=float)
    f = np.array([], order='F', dtype=float).reshape(0, 1)

    r, l_mat, scale, info = mb04rt(0, 1, pmax, a, b, c, d, e, f)
    assert info == 0
    assert scale == 1.0


def test_mb04rt_zero_n():
    """
    Test quick return for N=0.
    """
    pmax = 1e10

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([], order='F', dtype=float).reshape(0, 0)
    c = np.array([], order='F', dtype=float).reshape(1, 0)
    d = np.array([[1.0]], order='F', dtype=float)
    e = np.array([], order='F', dtype=float).reshape(0, 0)
    f = np.array([], order='F', dtype=float).reshape(1, 0)

    r, l_mat, scale, info = mb04rt(1, 0, pmax, a, b, c, d, e, f)
    assert info == 0
    assert scale == 1.0


def test_mb04rt_residual_property():
    """
    Mathematical property test: verify residual equations hold.

    For any valid solution:
        A * R - L * B = scale * C
        D * R - L * E = scale * F

    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)
    m, n = 5, 4
    pmax = 1e10

    a = np.zeros((m, m), order='F', dtype=float)
    a[0, 0] = 2.0
    a[1, 1] = 1.5
    a[1, 2] = 0.3
    a[2, 2] = 3.0
    a[3, 3] = 1.0
    a[3, 4] = 0.8
    a[4, 3] = -0.2
    a[4, 4] = 1.0
    for i in range(m):
        for j in range(i + 2, m):
            if not (i == 3 and j == 4):
                a[i, j] = np.random.randn() * 0.1

    b = np.zeros((n, n), order='F', dtype=float)
    b[0, 0] = 1.0
    b[0, 1] = 2.0
    b[1, 0] = -0.3
    b[1, 1] = 1.0
    b[2, 2] = 2.0
    b[3, 3] = 3.0
    for i in range(n):
        for j in range(i + 2, n):
            if not (i == 0 and j == 1):
                b[i, j] = np.random.randn() * 0.1

    d = np.eye(m, order='F', dtype=float)
    for i in range(m):
        d[i, i] = 1.0 + i * 0.2
    for i in range(m):
        for j in range(i + 1, m):
            d[i, j] = np.random.randn() * 0.02

    e = np.eye(n, order='F', dtype=float)
    for i in range(n):
        e[i, i] = 2.0 + i * 0.3
    for i in range(n):
        for j in range(i + 1, n):
            e[i, j] = np.random.randn() * 0.02

    c = np.random.randn(m, n).astype(float, order='F')
    f = np.random.randn(m, n).astype(float, order='F')

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rt(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert 0 < scale <= 1.0

    residual1 = a @ r - l_mat @ b - scale * c_orig
    residual2 = d @ r - l_mat @ e - scale * f_orig

    np.testing.assert_allclose(residual1, 0.0, atol=1e-12)
    np.testing.assert_allclose(residual2, 0.0, atol=1e-12)


def test_mb04rt_matches_mb04rs():
    """
    Verify MB04RT produces same result as MB04RS (unblocked).

    For small systems where blocking shouldn't change result.

    Random seed: 888 (for reproducibility)
    """
    from slicot import mb04rs

    np.random.seed(888)
    m, n = 3, 2
    pmax = 1e10

    a = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 2.0],
        [-0.3, 1.0]
    ], order='F', dtype=float)

    d = np.array([
        [1.0, 0.5, 0.2],
        [0.0, 1.0, 0.1],
        [0.0, 0.0, 1.5]
    ], order='F', dtype=float)

    e = np.array([
        [2.0, 0.3],
        [0.0, 2.0]
    ], order='F', dtype=float)

    c = np.random.randn(m, n).astype(float, order='F')
    f = np.random.randn(m, n).astype(float, order='F')

    c1 = c.copy()
    f1 = f.copy()
    c2 = c.copy()
    f2 = f.copy()

    r1, l1, scale1, info1 = mb04rs(m, n, pmax, a, b, c1, d, e, f1)
    r2, l2, scale2, info2 = mb04rt(m, n, pmax, a, b, c2, d, e, f2)

    assert info1 == 0
    assert info2 == 0

    np.testing.assert_allclose(scale1, scale2, rtol=1e-14)
    np.testing.assert_allclose(r1, r2, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(l1, l2, rtol=1e-13, atol=1e-14)


def test_mb04rt_block_boundary_detection():
    """
    Test that 2x2 blocks are detected at block boundaries.

    Uses a 4x4 system where a 2x2 block spans what would otherwise be
    the block boundary. Must not split 2x2 blocks.

    Random seed: 999 (for reproducibility)
    """
    m, n = 4, 4
    pmax = 1e10

    a = np.array([
        [1.0, 0.5, 0.0, 0.0],
        [0.0, 2.0, 1.0, 0.0],
        [0.0, -0.5, 2.0, 0.0],
        [0.0, 0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.5, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [0.0, -0.3, 1.0, 0.0],
        [0.0, 0.0, 0.0, 2.5]
    ], order='F', dtype=float)

    d = np.diag([1.0, 1.0, 1.0, 1.0]).astype(float, order='F')
    e = np.diag([1.0, 1.0, 1.0, 1.0]).astype(float, order='F')

    c = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], order='F', dtype=float)

    f = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6]
    ], order='F', dtype=float)

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rt(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-12, atol=1e-13)
