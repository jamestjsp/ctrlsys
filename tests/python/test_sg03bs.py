"""
Tests for SG03BS - Generalized discrete-time Lyapunov equation (complex Hammarling).

Computes Cholesky factor U of X = U^H*U or X = U*U^H for:
    A^H * X * A - E^H * X * E = -SCALE^2 * B^H * B   (TRANS='N')
    A * X * A^H - E * X * E^H = -SCALE^2 * B * B^H   (TRANS='C')

where A, E, B are upper triangular complex matrices.
"""

import numpy as np
import pytest
from ctrlsys import sg03bs


def test_sg03bs_basic_trans_n():
    """
    Test SG03BS with TRANS='N' for small d-stable system.

    Tests eq(1): A^H * X * A - E^H * X * E = -SCALE^2 * B^H * B
    where X = U^H * U.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3

    a = np.array([
        [0.3+0.1j, 0.1+0.05j, 0.05+0.02j],
        [0.0, 0.2-0.1j, 0.08+0.03j],
        [0.0, 0.0, 0.15+0.05j]
    ], dtype=complex, order='F')

    e = np.array([
        [1.0+0j, 0.1+0.02j, 0.05+0.01j],
        [0.0, 1.0+0j, 0.08+0.02j],
        [0.0, 0.0, 1.0+0j]
    ], dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j, 0.1+0.05j],
        [0.0, 0.3, 0.15+0.08j],
        [0.0, 0.0, 0.25]
    ], dtype=complex, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    u, scale, info = sg03bs('N', a, e, b)

    assert info == 0, f"SG03BS failed with INFO={info}"
    assert 0.0 < scale <= 1.0, f"Invalid scale: {scale}"

    np.testing.assert_allclose(np.tril(u, -1), 0, atol=1e-14)

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ a_orig - e_orig.conj().T @ x @ e_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs))
    assert residual < 1e-10, f"Lyapunov residual too large: {residual}"

    assert np.all(np.diag(u).real >= 0), "U diagonal should be non-negative real"


def test_sg03bs_basic_trans_c():
    """
    Test SG03BS with TRANS='C' for small d-stable system.

    Tests eq(2): A * X * A^H - E * X * E^H = -SCALE^2 * B * B^H
    where X = U * U^H.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3

    a = np.array([
        [0.25+0.08j, 0.12+0.04j, 0.06+0.02j],
        [0.0, 0.18-0.06j, 0.09+0.03j],
        [0.0, 0.0, 0.12+0.04j]
    ], dtype=complex, order='F')

    e = np.array([
        [1.0+0j, 0.15+0.03j, 0.08+0.02j],
        [0.0, 1.0+0j, 0.1+0.02j],
        [0.0, 0.0, 1.0+0j]
    ], dtype=complex, order='F')

    b = np.array([
        [0.4, 0.18+0.08j, 0.09+0.04j],
        [0.0, 0.35, 0.12+0.06j],
        [0.0, 0.0, 0.2]
    ], dtype=complex, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    u, scale, info = sg03bs('C', a, e, b)

    assert info == 0, f"SG03BS failed with INFO={info}"
    assert 0.0 < scale <= 1.0, f"Invalid scale: {scale}"

    np.testing.assert_allclose(np.tril(u, -1), 0, atol=1e-14)

    x = u @ u.conj().T
    lhs = a_orig @ x @ a_orig.conj().T - e_orig @ x @ e_orig.conj().T
    rhs = -scale**2 * (b_orig @ b_orig.conj().T)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs))
    assert residual < 1e-10, f"Lyapunov residual too large: {residual}"


def test_sg03bs_n1():
    """
    Test SG03BS with N=1 (scalar case).

    Random seed: 456 (for reproducibility)
    """
    n = 1

    a = np.array([[0.3+0.1j]], dtype=complex, order='F')
    e = np.array([[1.0+0j]], dtype=complex, order='F')
    b = np.array([[0.5]], dtype=complex, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    u, scale, info = sg03bs('N', a, e, b)

    assert info == 0, f"SG03BS failed with INFO={info}"
    assert 0.0 < scale <= 1.0

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ a_orig - e_orig.conj().T @ x @ e_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-14)


def test_sg03bs_n0():
    """Test SG03BS with N=0 (quick return)."""
    a = np.zeros((0, 0), dtype=complex, order='F')
    e = np.zeros((0, 0), dtype=complex, order='F')
    b = np.zeros((0, 0), dtype=complex, order='F')

    u, scale, info = sg03bs('N', a, e, b)

    assert info == 0
    assert scale == 1.0


def test_sg03bs_unstable_eigenvalue():
    """Test SG03BS returns INFO=3 for unstable (non d-stable) pencil."""
    n = 2

    a = np.array([
        [1.5+0j, 0.1+0.05j],
        [0.0, 1.2-0.1j]
    ], dtype=complex, order='F')

    e = np.array([
        [1.0+0j, 0.1+0.02j],
        [0.0, 1.0+0j]
    ], dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j],
        [0.0, 0.3]
    ], dtype=complex, order='F')

    u, scale, info = sg03bs('N', a, e, b)

    assert info == 3, f"Expected INFO=3 for unstable system, got {info}"


def test_sg03bs_invalid_trans():
    """Test SG03BS returns INFO=-1 for invalid TRANS."""
    n = 2

    a = np.array([
        [0.3+0.1j, 0.1+0.05j],
        [0.0, 0.2-0.1j]
    ], dtype=complex, order='F')

    e = np.eye(2, dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j],
        [0.0, 0.3]
    ], dtype=complex, order='F')

    u, scale, info = sg03bs('X', a, e, b)

    assert info == -1, f"Expected INFO=-1 for invalid TRANS, got {info}"


def test_sg03bs_eigenvalue_preservation():
    """
    Validate that the solution preserves mathematical properties.

    For d-stable system, compute Lyapunov solution and verify:
    1. U is upper triangular with non-negative real diagonal
    2. Lyapunov equation is satisfied
    3. X = U^H * U is positive semidefinite

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4

    diag_a = 0.2 + 0.1 * np.random.randn(n) + 0.1j * np.random.randn(n)
    for i in range(n):
        while np.abs(diag_a[i]) >= 0.95:
            diag_a[i] = 0.2 + 0.1 * np.random.randn() + 0.1j * np.random.randn()

    a = np.diag(diag_a).astype(complex, order='F')
    for i in range(n):
        for j in range(i+1, n):
            a[i, j] = 0.05 * (np.random.randn() + 1j * np.random.randn())
    a = np.asfortranarray(a)

    e = np.eye(n, dtype=complex, order='F')
    for i in range(n):
        for j in range(i+1, n):
            e[i, j] = 0.02 * (np.random.randn() + 1j * np.random.randn())
    e = np.asfortranarray(e)

    b = np.zeros((n, n), dtype=complex, order='F')
    for i in range(n):
        b[i, i] = 0.3 + 0.2 * np.random.rand()
        for j in range(i+1, n):
            b[i, j] = 0.1 * (np.random.randn() + 1j * np.random.randn())
    b = np.asfortranarray(b)

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    u, scale, info = sg03bs('N', a, e, b)

    assert info == 0, f"SG03BS failed with INFO={info}"

    np.testing.assert_allclose(np.tril(u, -1), 0, atol=1e-14)
    assert np.all(np.diag(u).real >= -1e-14), "U diagonal should be non-negative real"

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ a_orig - e_orig.conj().T @ x @ e_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    residual = np.linalg.norm(lhs - rhs, 'fro') / max(np.linalg.norm(lhs, 'fro'), 1e-15)
    assert residual < 1e-10, f"Lyapunov residual too large: {residual}"

    eig_x = np.linalg.eigvalsh(x)
    assert np.all(eig_x >= -1e-10), f"X should be positive semidefinite, min eig={eig_x.min()}"


def test_sg03bs_large_system():
    """
    Test SG03BS with larger system to verify recursive algorithm.

    Random seed: 2024 (for reproducibility)
    """
    np.random.seed(2024)
    n = 10

    diag_a = 0.15 + 0.05 * np.random.randn(n) + 0.05j * np.random.randn(n)
    for i in range(n):
        while np.abs(diag_a[i]) >= 0.9:
            diag_a[i] = 0.15 + 0.05 * np.random.randn() + 0.05j * np.random.randn()

    a = np.diag(diag_a).astype(complex, order='F')
    for i in range(n):
        for j in range(i+1, n):
            a[i, j] = 0.02 * (np.random.randn() + 1j * np.random.randn())
    a = np.asfortranarray(a)

    e = np.eye(n, dtype=complex, order='F')
    for i in range(n):
        for j in range(i+1, n):
            e[i, j] = 0.01 * (np.random.randn() + 1j * np.random.randn())
    e = np.asfortranarray(e)

    b = np.zeros((n, n), dtype=complex, order='F')
    for i in range(n):
        b[i, i] = 0.2 + 0.1 * np.random.rand()
        for j in range(i+1, n):
            b[i, j] = 0.05 * (np.random.randn() + 1j * np.random.randn())
    b = np.asfortranarray(b)

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    u, scale, info = sg03bs('N', a, e, b)

    assert info == 0, f"SG03BS failed with INFO={info}"
    assert 0.0 < scale <= 1.0

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ a_orig - e_orig.conj().T @ x @ e_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    residual = np.linalg.norm(lhs - rhs, 'fro') / max(np.linalg.norm(lhs, 'fro'), 1e-15)
    assert residual < 1e-9, f"Lyapunov residual too large: {residual}"
