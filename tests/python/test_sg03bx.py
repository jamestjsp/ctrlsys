import pytest
import numpy as np
from slicot import sg03bx


def _check_lyapunov_residual(dico, trans, a, e, b, u, scale, m1, m2, atol=1e-8):
    if trans == 'N':
        x = u.T @ u
        e_inv = np.linalg.inv(e)
        if abs(np.linalg.det(u)) > 1e-15:
            u_inv = np.linalg.inv(u)
            m1_expected = u @ a @ e_inv @ u_inv
            m2_expected = b @ e_inv @ u_inv
            np.testing.assert_allclose(m1, m1_expected, atol=atol)
            np.testing.assert_allclose(m2, m2_expected, atol=atol)
        if dico == 'C':
            res = a.T @ x @ e + e.T @ x @ a + scale**2 * b.T @ b
        else:
            res = a.T @ x @ a - e.T @ x @ e + scale**2 * b.T @ b
    else:
        x = u @ u.T
        if dico == 'C':
            res = a @ x @ e.T + e @ x @ a.T + scale**2 * b @ b.T
        else:
            res = a @ x @ a.T - e @ x @ e.T + scale**2 * b @ b.T
    norm_res = np.linalg.norm(res, 'fro')
    norm_scale = max(np.linalg.norm(x, 'fro'), 1.0)
    assert norm_res / norm_scale < atol, f"Lyapunov residual too large: {norm_res / norm_scale}"


def test_sg03bx_continuous_basic():
    dico = 'C'
    trans = 'N'

    a = np.array([
        [-1.0, 2.0],
        [-2.0, -1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert m1.shape == (2, 2)
    assert m2.shape == (2, 2)
    assert u[1, 0] == 0.0
    assert u[0, 0] >= 0.0
    assert u[1, 1] >= 0.0

    _check_lyapunov_residual(dico, trans, a, e, b, u, scale, m1, m2)


def test_sg03bx_discrete_basic():
    dico = 'D'
    trans = 'N'

    a = np.array([
        [0.3, 0.2],
        [-0.2, 0.3]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.5, 0.2],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert u[1, 0] == 0.0
    assert u[0, 0] >= 0.0
    assert u[1, 1] >= 0.0

    _check_lyapunov_residual(dico, trans, a, e, b, u, scale, m1, m2)


def test_sg03bx_transpose():
    dico = 'C'
    trans = 'T'

    a = np.array([
        [-1.0, 2.0],
        [-2.0, -1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0

    _check_lyapunov_residual(dico, trans, a, e, b, u, scale, m1, m2)


def test_sg03bx_error_real_eigenvalues():
    """Test error when pencil has real eigenvalues (not complex conjugate)"""
    dico = 'C'
    trans = 'N'

    # Diagonal A and E -> real eigenvalues
    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 2  # Not complex conjugate


def test_sg03bx_error_unstable():
    """Test error when eigenvalues not in correct half-plane"""
    dico = 'C'
    trans = 'N'

    # Unstable for continuous-time (eigenvalues in left half plane)
    a = np.array([
        [1.0, 2.0],
        [-2.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 3  # Stability error


def test_sg03bx_continuous_nondiag_e():
    dico = 'C'
    trans = 'N'

    a = np.array([
        [-2.0, 3.0],
        [-3.0, -2.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [2.0, 0.8],
        [0.0, 1.5]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.7, 0.3],
        [0.0, 0.4]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    _check_lyapunov_residual(dico, trans, a, e, b, u, scale, m1, m2)


def test_sg03bx_discrete_nondiag_e():
    dico = 'D'
    trans = 'N'

    a = np.array([
        [0.2, 0.4],
        [-0.4, 0.2]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.5, 0.3],
        [0.0, 1.2]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.6, 0.2],
        [0.0, 0.3]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    _check_lyapunov_residual(dico, trans, a, e, b, u, scale, m1, m2)


def test_sg03bx_discrete_transpose():
    dico = 'D'
    trans = 'T'

    a = np.array([
        [0.3, 0.2],
        [-0.2, 0.3]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.5, 0.2],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    _check_lyapunov_residual(dico, trans, a, e, b, u, scale, m1, m2)


def test_sg03bx_continuous_transpose_nondiag_e():
    dico = 'C'
    trans = 'T'

    a = np.array([
        [-2.0, 3.0],
        [-3.0, -2.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [2.0, 0.8],
        [0.0, 1.5]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.7, 0.3],
        [0.0, 0.4]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    _check_lyapunov_residual(dico, trans, a, e, b, u, scale, m1, m2)
