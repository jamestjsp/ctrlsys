"""Tests for MB03RY - Sylvester equation solver with norm bound"""
import numpy as np
import pytest
from ctrlsys import mb03ry


def test_mb03ry_basic_1x1():
    """
    Test MB03RY with 1x1 A and 1x1 B matrices.

    Sylvester: -AX + XB = C
    A = [2], B = [3], C = [5]
    Solution: X = C / (B - A) = 5 / (3 - 2) = 5
    Check: -2*5 + 5*3 = -10 + 15 = 5 = C
    """
    a = np.array([[2.0]], order='F', dtype=float)
    b = np.array([[3.0]], order='F', dtype=float)
    c_orig = np.array([[5.0]], order='F', dtype=float)
    c = c_orig.copy(order='F')
    pmax = 1000.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 0
    x_expected = 5.0
    np.testing.assert_allclose(x[0, 0], x_expected, rtol=1e-14)

    residual = -a @ x + x @ b
    np.testing.assert_allclose(residual, c_orig, rtol=1e-14, atol=1e-15)


def test_mb03ry_basic_2x2():
    """
    Test MB03RY with 2x2 A and 2x2 B in Schur form.

    Random seed: 42 (for reproducibility)
    Uses diagonal matrices (simplest Schur form) to verify basic functionality.
    """
    a = np.array([
        [1.0, 0.5],
        [0.0, 2.0]
    ], order='F', dtype=float)
    b = np.array([
        [3.0, 0.2],
        [0.0, 4.0]
    ], order='F', dtype=float)
    c_orig = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)
    c = c_orig.copy(order='F')
    pmax = 1000.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 0
    residual = -a @ x + x @ b
    np.testing.assert_allclose(residual, c_orig, rtol=1e-13, atol=1e-14)


def test_mb03ry_rectangular():
    """
    Test MB03RY with M != N (rectangular C).

    A: 2x2 Schur, B: 3x3 Schur, C: 2x3
    """
    a = np.array([
        [1.0, 0.3],
        [0.0, 2.0]
    ], order='F', dtype=float)
    b = np.array([
        [3.0, 0.1, 0.0],
        [0.0, 4.0, 0.2],
        [0.0, 0.0, 5.0]
    ], order='F', dtype=float)
    c_orig = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], order='F', dtype=float)
    c = c_orig.copy(order='F')
    pmax = 1000.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 0
    residual = -a @ x + x @ b
    np.testing.assert_allclose(residual, c_orig, rtol=1e-13, atol=1e-14)


def test_mb03ry_2x2_block():
    """
    Test MB03RY with 2x2 blocks from complex eigenvalue pairs.

    A 2x2 real Schur block represents complex conjugate eigenvalues.
    The subdiagonal element is non-zero.
    """
    a = np.array([
        [1.0,  1.0],
        [-0.5, 1.0]
    ], order='F', dtype=float)
    b = np.array([
        [2.0,  0.5],
        [-0.3, 2.0]
    ], order='F', dtype=float)
    c_orig = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ], order='F', dtype=float)
    c = c_orig.copy(order='F')
    pmax = 1000.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 0
    residual = -a @ x + x @ b
    np.testing.assert_allclose(residual, c_orig, rtol=1e-13, atol=1e-14)


def test_mb03ry_pmax_exceeded():
    """
    Test MB03RY when solution norm exceeds PMAX.

    Use eigenvalues that are very close to create near-singular system.
    """
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0001]], order='F', dtype=float)
    c = np.array([[10.0]], order='F', dtype=float)
    pmax = 1.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 1


def test_mb03ry_m_zero():
    """Test MB03RY with M=0 (empty A, C)."""
    a = np.empty((0, 0), order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.empty((0, 1), order='F', dtype=float)
    pmax = 1000.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 0
    assert x.shape == (0, 1)


def test_mb03ry_n_zero():
    """Test MB03RY with N=0 (empty B, C)."""
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.empty((0, 0), order='F', dtype=float)
    c = np.empty((1, 0), order='F', dtype=float)
    pmax = 1000.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 0
    assert x.shape == (1, 0)


def test_mb03ry_mixed_blocks():
    """
    Test MB03RY with mixed 1x1 and 2x2 blocks.

    A: 3x3 with 2x2 block (complex eigenvalues) + 1x1 block
    B: 3x3 with 1x1 block + 2x2 block
    Random seed: 123 (for reproducibility)
    """
    a = np.array([
        [1.0,  0.8, 0.0],
        [-0.5, 1.0, 0.0],
        [0.0,  0.0, 3.0]
    ], order='F', dtype=float)
    b = np.array([
        [4.0, 0.0,  0.0],
        [0.0, 5.0,  0.6],
        [0.0, -0.4, 5.0]
    ], order='F', dtype=float)

    np.random.seed(123)
    c_orig = np.random.randn(3, 3).astype(float, order='F')
    c = c_orig.copy(order='F')
    pmax = 1000.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 0
    residual = -a @ x + x @ b
    np.testing.assert_allclose(residual, c_orig, rtol=1e-13, atol=1e-14)


def test_mb03ry_larger_system():
    """
    Test MB03RY with larger 4x4 system.

    Random seed: 456 (for reproducibility)
    """
    a = np.array([
        [1.0, 0.5, 0.0, 0.0],
        [0.0, 2.0, 0.3, 0.0],
        [0.0, 0.0, 3.0, 0.2],
        [0.0, 0.0, 0.0, 4.0]
    ], order='F', dtype=float)
    b = np.array([
        [5.0, 0.1, 0.0, 0.0],
        [0.0, 6.0, 0.2, 0.0],
        [0.0, 0.0, 7.0, 0.1],
        [0.0, 0.0, 0.0, 8.0]
    ], order='F', dtype=float)

    np.random.seed(456)
    c_orig = np.random.randn(4, 4).astype(float, order='F')
    c = c_orig.copy(order='F')
    pmax = 1000.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 0
    residual = -a @ x + x @ b
    np.testing.assert_allclose(residual, c_orig, rtol=1e-13, atol=1e-14)


def test_mb03ry_sylvester_equation_property():
    """
    Verify mathematical property: -AX + XB = C.

    This is the defining equation for MB03RY.
    Uses upper triangular matrices (simplest Schur form).
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5

    a = np.triu(np.random.randn(n, n)).astype(float, order='F')
    for i in range(n):
        a[i, i] = i + 1.0

    b = np.triu(np.random.randn(n, n)).astype(float, order='F')
    for i in range(n):
        b[i, i] = i + 6.0

    c_orig = np.random.randn(n, n).astype(float, order='F')
    c = c_orig.copy(order='F')
    pmax = 1000.0

    x, info = mb03ry(a, b, c, pmax)

    assert info == 0
    residual = -a @ x + x @ b
    np.testing.assert_allclose(residual, c_orig, rtol=1e-12, atol=1e-13)
