import pytest
import numpy as np
from ctrlsys import sg03bu


def test_sg03bu_1x1_basic():
    """Test discrete-time case with N=1 (scalar problem)"""
    trans = 'N'
    n = 1

    # A^T * X * A - E^T * X * E = -SCALE^2 * B^T * B
    # For stability: |A/E| < 1
    a = np.array([[0.5]], dtype=np.float64, order='F')
    e = np.array([[1.0]], dtype=np.float64, order='F')
    b = np.array([[1.0]], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bu(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (1, 1)
    assert u[0, 0] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ a - e.T @ x @ e + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bu_2x2_basic():
    """Test discrete-time case with N=2"""
    trans = 'N'
    n = 2

    # Stable d-stable pencil (eigenvalues inside unit circle)
    a = np.array([
        [0.3, 0.2],
        [0.0, 0.4]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bu(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert u[1, 0] == 0.0
    assert u[0, 0] >= 0.0
    assert u[1, 1] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ a - e.T @ x @ e + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bu_2x2_complex():
    """Test discrete-time with 2x2 block (complex conjugate eigenvalues)"""
    trans = 'N'
    n = 2

    # Quasitriangular A with 2x2 block (complex eigenvalues)
    # Eigenvalues must be inside unit circle
    a = np.array([
        [0.3, 0.4],
        [-0.4, 0.3]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.3],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.8, 0.2],
        [0.0, 0.8]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bu(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert u[1, 0] == 0.0
    assert u[0, 0] >= 0.0
    assert u[1, 1] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ a - e.T @ x @ e + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bu_3x3_mixed():
    """Test discrete-time with N=3 (mixed 1x1 and 2x2 blocks)"""
    trans = 'N'
    n = 3

    # Quasitriangular: 1x1 block at (0,0), 2x2 block at (1:3, 1:3)
    a = np.array([
        [0.5, 0.3, 0.2],
        [0.0, 0.4, 0.3],
        [0.0, -0.3, 0.4]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2, 0.1],
        [0.0, 1.0, 0.2],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.3, 0.2],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 0.9]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bu(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (3, 3)
    assert u[1, 0] == 0.0
    assert u[2, 0] == 0.0
    assert u[2, 1] == 0.0
    for i in range(3):
        assert u[i, i] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ a - e.T @ x @ e + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bu_transposed():
    """Test discrete-time with TRANS='T' (transposed equation)"""
    trans = 'T'
    n = 2

    a = np.array([
        [0.4, 0.1],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.3],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.8, 0.2],
        [0.0, 0.7]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bu(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert u[1, 0] == 0.0

    x = u @ u.T
    residual = a @ x @ a.T - e @ x @ e.T + scale**2 * b_orig @ b_orig.T
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_sg03bu_error_unstable():
    """Test error when pencil is not d-stable (eigenvalues outside unit circle)"""
    trans = 'N'
    n = 2

    # Unstable: |lambda| >= 1
    a = np.array([
        [1.5, 0.2],
        [0.0, 1.2]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    u, scale, info = sg03bu(trans, a, e, b)

    assert info == 3  # Not d-stable


def test_sg03bu_error_invalid_trans():
    """Test error with invalid TRANS parameter"""
    trans = 'X'
    n = 2

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    b = np.eye(2, dtype=np.float64, order='F')

    u, scale, info = sg03bu(trans, a, e, b)

    assert info < 0


def test_sg03bu_n0_quick_return():
    """Test quick return with N=0"""
    trans = 'N'
    n = 0

    a = np.empty((1, 0), dtype=np.float64, order='F')
    e = np.empty((1, 0), dtype=np.float64, order='F')
    b = np.empty((1, 0), dtype=np.float64, order='F')

    u, scale, info = sg03bu(trans, a, e, b)

    assert info == 0
    assert scale == 1.0


def test_sg03bu_4x4_recursive():
    """Test recursive application with N=4"""
    trans = 'N'
    n = 4

    # Multiple 2x2 blocks to test recursive algorithm
    a = np.array([
        [0.3, 0.2, 0.1, 0.0],
        [-0.2, 0.3, 0.0, 0.1],
        [0.0, 0.0, 0.4, 0.15],
        [0.0, 0.0, -0.15, 0.4]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2, 0.1, 0.0],
        [0.0, 1.0, 0.0, 0.1],
        [0.0, 0.0, 1.0, 0.2],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.2, 0.1, 0.05],
        [0.0, 1.0, 0.15, 0.1],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.0, 0.0, 0.9]
    ], dtype=np.float64, order='F')
    b_orig = b.copy()

    u, scale, info = sg03bu(trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (4, 4)
    for i in range(4):
        for j in range(i):
            assert u[i, j] == 0.0
    for i in range(4):
        assert u[i, i] >= 0.0

    x = u.T @ u
    residual = a.T @ x @ a - e.T @ x @ e + scale**2 * b_orig.T @ b_orig
    np.testing.assert_allclose(residual, 0, atol=1e-10)
