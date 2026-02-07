import pytest
import numpy as np
from slicot import sg03bv


def test_sg03bv_1x1_basic():
    """Test continuous-time case with N=1 (scalar problem)"""
    trans = 'N'
    n = 1

    # A^T * X * E + E^T * X * A = -SCALE^2 * B^T * B
    # For stability: Re(A/E) < 0
    a = np.array([[-1.0]], dtype=np.float64, order='F')
    e = np.array([[1.0]], dtype=np.float64, order='F')
    b = np.array([[1.0]], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bv(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (1, 1)
    assert u[0, 0] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ e + e.T @ x @ a + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bv_2x2_basic():
    """Test continuous-time case with N=2"""
    trans = 'N'
    n = 2

    # Stable c-stable pencil (eigenvalues with negative real parts)
    a = np.array([
        [-1.0, 0.5],
        [0.0, -2.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.3],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    # B upper triangular with non-negative diagonal
    b = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bv(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert u[1, 0] == 0.0
    assert u[0, 0] >= 0.0
    assert u[1, 1] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ e + e.T @ x @ a + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bv_2x2_complex():
    """Test continuous-time with 2x2 block (complex conjugate eigenvalues)"""
    trans = 'N'
    n = 2

    # Quasitriangular A with 2x2 block (complex eigenvalues)
    # Eigenvalues must have negative real parts
    a = np.array([
        [-1.0, 2.0],
        [-2.0, -1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.3],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bv(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert u[1, 0] == 0.0
    assert u[0, 0] >= 0.0
    assert u[1, 1] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ e + e.T @ x @ a + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bv_3x3_mixed():
    """Test continuous-time with N=3 (mixed 1x1 and 2x2 blocks)"""
    trans = 'N'
    n = 3

    # Quasitriangular: 1x1 block at (0,0), 2x2 block at (1:3, 1:3)
    a = np.array([
        [-2.0, 0.5, 0.3],
        [0.0, -1.0, 1.5],
        [0.0, -1.5, -1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2, 0.1],
        [0.0, 1.0, 0.3],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.4, 0.2],
        [0.0, 1.0, 0.3],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bv(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (3, 3)
    assert u[1, 0] == 0.0
    assert u[2, 0] == 0.0
    assert u[2, 1] == 0.0
    for i in range(3):
        assert u[i, i] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ e + e.T @ x @ a + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bv_transposed():
    """Test continuous-time with TRANS='T' (transposed equation)"""
    trans = 'T'
    n = 2

    a = np.array([
        [-1.5, 0.3],
        [0.0, -2.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.8, 0.3],
        [0.0, 0.9]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bv(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert u[1, 0] == 0.0

    x = u @ u.T
    residual = a @ x @ e.T + e @ x @ a.T + scale**2 * b_orig @ b_orig.T
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bv_error_unstable():
    """Test error when pencil is not c-stable (eigenvalues with non-negative real parts)"""
    trans = 'N'
    n = 2

    # Unstable: Re(lambda) >= 0
    a = np.array([
        [1.0, 0.5],
        [0.0, 2.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    u, scale, info = sg03bv(trans, a, e, b)

    assert info == 3  # Not c-stable


def test_sg03bv_error_invalid_trans():
    """Test error with invalid TRANS parameter"""
    trans = 'X'
    n = 2

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    b = np.eye(2, dtype=np.float64, order='F')

    u, scale, info = sg03bv(trans, a, e, b)

    assert info < 0


def test_sg03bv_n0_quick_return():
    """Test quick return with N=0"""
    trans = 'N'
    n = 0

    a = np.empty((1, 0), dtype=np.float64, order='F')
    e = np.empty((1, 0), dtype=np.float64, order='F')
    b = np.empty((1, 0), dtype=np.float64, order='F')

    u, scale, info = sg03bv(trans, a, e, b)

    assert info == 0
    assert scale == 1.0


def test_sg03bv_4x4_recursive():
    """Test recursive application with N=4"""
    trans = 'N'
    n = 4

    # Multiple 2x2 blocks to test recursive algorithm
    a = np.array([
        [-1.0, 0.8, 0.3, 0.1],
        [-0.8, -1.0, 0.2, 0.1],
        [0.0, 0.0, -1.5, 1.2],
        [0.0, 0.0, -1.2, -1.5]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.3, 0.2, 0.1],
        [0.0, 1.0, 0.1, 0.1],
        [0.0, 0.0, 1.0, 0.2],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.3, 0.2, 0.1],
        [0.0, 1.0, 0.2, 0.1],
        [0.0, 0.0, 1.0, 0.2],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bv(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (4, 4)
    for i in range(4):
        for j in range(i):
            assert u[i, j] == 0.0
    for i in range(4):
        assert u[i, i] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ e + e.T @ x @ a + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bv_near_singular():
    """Test case that may trigger INFO=1 (nearly singular Sylvester equation)"""
    trans = 'N'
    n = 2

    # Create scenario that might lead to near singularity
    a = np.array([
        [-1e-2, 1e-3],
        [0.0, -1e-2]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 1e-3],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bv(trans, a, e, b)

    assert info >= 0
    if info == 0:
        assert 0.0 < scale <= 1.0
        x = u.T @ u
        residual = a.T @ x @ e + e.T @ x @ a + scale**2 * b_orig.T @ b_orig
        np.testing.assert_allclose(residual, 0, atol=1e-10)
