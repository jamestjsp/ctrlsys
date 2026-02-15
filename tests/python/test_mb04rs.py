"""
Tests for MB04RS - Generalized real Sylvester equation solver.

Solves:
    A * R - L * B = scale * C
    D * R - L * E = scale * F

where (A, D) and (B, E) are in generalized Schur form (A, B upper quasi-triangular,
D, E upper triangular).
"""

import numpy as np
import pytest
from ctrlsys import mb04rs


def test_mb04rs_basic_1x1():
    """
    Test 1x1 system (simplest case).

    A * R - L * B = scale * C
    D * R - L * E = scale * F

    With scalar matrices:
    2*R - L*3 = 1  =>  2R - 3L = 1
    1*R - L*4 = 2  =>   R - 4L = 2

    Solving: R = 2, L = 1 (verify: 2*2 - 3*1 = 1, 1*2 - 4*1 = -2)
    Wait, need to recalculate properly.

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

    r, l_mat, scale, info = mb04rs(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * np.array([[1.0]]), rtol=1e-14)
    np.testing.assert_allclose(rhs2, scale * np.array([[2.0]]), rtol=1e-14)


def test_mb04rs_2x2_diagonal():
    """
    Test 2x2 diagonal system (no 2x2 blocks).

    Uses diagonal matrices in Schur form with distinct eigenvalues.
    Eigenvalues of (A,D): 1/0.5=2, 3/1=3
    Eigenvalues of (B,E): 4/1=4, 5/2=2.5
    These are all distinct, ensuring a well-posed system.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
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

    r, l_mat, scale, info = mb04rs(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-13, atol=1e-14)


def test_mb04rs_2x2_block_a():
    """
    Test with 2x2 block in A (quasi-triangular, complex eigenvalues).

    A has a 2x2 block representing complex conjugate eigenvalues.

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

    r, l_mat, scale, info = mb04rs(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-13, atol=1e-14)


def test_mb04rs_2x2_block_b():
    """
    Test with 2x2 block in B (quasi-triangular).

    Random seed: 789 (for reproducibility)
    """
    m, n = 1, 2
    pmax = 1e10

    a = np.array([[2.0]], order='F', dtype=float)
    b = np.array([[1.0, 3.0],
                  [-0.25, 1.0]], order='F', dtype=float)

    d = np.array([[1.0]], order='F', dtype=float)
    e = np.array([[2.0, 0.5],
                  [0.0, 2.0]], order='F', dtype=float)

    c = np.array([[1.0, 2.0]], order='F', dtype=float)
    f = np.array([[0.5, 1.0]], order='F', dtype=float)

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rs(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-13, atol=1e-14)


def test_mb04rs_both_2x2_blocks():
    """
    Test with 2x2 blocks in both A and B (8x8 system).

    Random seed: 999 (for reproducibility)
    """
    m, n = 2, 2
    pmax = 1e10

    a = np.array([[1.0, 2.0],
                  [-0.5, 1.0]], order='F', dtype=float)
    b = np.array([[2.0, 3.0],
                  [-0.25, 2.0]], order='F', dtype=float)

    d = np.array([[1.0, 0.5],
                  [0.0, 1.0]], order='F', dtype=float)
    e = np.array([[1.5, 0.5],
                  [0.0, 1.5]], order='F', dtype=float)

    c = np.array([[1.0, 2.0],
                  [3.0, 4.0]], order='F', dtype=float)
    f = np.array([[0.5, 1.0],
                  [1.5, 2.0]], order='F', dtype=float)

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rs(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-12, atol=1e-13)


def test_mb04rs_larger_system():
    """
    Test larger 4x3 system with mixed 1x1 and 2x2 blocks.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    m, n = 4, 3
    pmax = 1e10

    a = np.array([
        [1.0, 2.0, 0.5, 0.3],
        [-0.5, 1.0, 0.2, 0.1],
        [0.0, 0.0, 3.0, 0.4],
        [0.0, 0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    b = np.array([
        [2.0, 0.5, 0.3],
        [0.0, 1.5, 1.0],
        [0.0, -0.5, 1.5]
    ], order='F', dtype=float)

    d = np.array([
        [1.0, 0.1, 0.2, 0.3],
        [0.0, 1.0, 0.1, 0.2],
        [0.0, 0.0, 2.0, 0.1],
        [0.0, 0.0, 0.0, 2.5]
    ], order='F', dtype=float)

    e = np.array([
        [1.5, 0.2, 0.1],
        [0.0, 1.0, 0.2],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    c = np.random.randn(m, n).astype(float, order='F')
    f = np.random.randn(m, n).astype(float, order='F')

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rs(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert scale > 0

    rhs1 = a @ r - l_mat @ b
    rhs2 = d @ r - l_mat @ e

    np.testing.assert_allclose(rhs1, scale * c_orig, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(rhs2, scale * f_orig, rtol=1e-12, atol=1e-13)


def test_mb04rs_pmax_exceeded():
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

    r, l_mat, scale, info = mb04rs(m, n, pmax, a, b, c, d, e, f)

    assert info == 1


def test_mb04rs_zero_dimensions():
    """
    Test quick return for M=0 or N=0.
    """
    pmax = 1e10

    a = np.array([], order='F', dtype=float).reshape(0, 0)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([], order='F', dtype=float).reshape(0, 1)
    d = np.array([], order='F', dtype=float).reshape(0, 0)
    e = np.array([[1.0]], order='F', dtype=float)
    f = np.array([], order='F', dtype=float).reshape(0, 1)

    r, l_mat, scale, info = mb04rs(0, 1, pmax, a, b, c, d, e, f)
    assert info == 0
    assert scale == 1.0

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([], order='F', dtype=float).reshape(0, 0)
    c = np.array([], order='F', dtype=float).reshape(1, 0)
    d = np.array([[1.0]], order='F', dtype=float)
    e = np.array([], order='F', dtype=float).reshape(0, 0)
    f = np.array([], order='F', dtype=float).reshape(1, 0)

    r, l_mat, scale, info = mb04rs(1, 0, pmax, a, b, c, d, e, f)
    assert info == 0
    assert scale == 1.0


def test_mb04rs_residual_property():
    """
    Mathematical property test: verify residual equations hold.

    For any valid solution, we must have:
        A * R - L * B = scale * C
        D * R - L * E = scale * F

    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)
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

    c_orig = c.copy()
    f_orig = f.copy()

    r, l_mat, scale, info = mb04rs(m, n, pmax, a, b, c, d, e, f)

    assert info == 0
    assert 0 < scale <= 1.0

    residual1 = a @ r - l_mat @ b - scale * c_orig
    residual2 = d @ r - l_mat @ e - scale * f_orig

    np.testing.assert_allclose(residual1, 0.0, atol=1e-13)
    np.testing.assert_allclose(residual2, 0.0, atol=1e-13)
