"""
Tests for MB03GD - Exchange eigenvalues of 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil (factored version).

MB03GD computes orthogonal matrix Q and orthogonal symplectic matrix U for a real
regular 2-by-2 or 4-by-4 skew-Hamiltonian/Hamiltonian pencil a J B' J' B - b D with

    B = [[B11, B12], [0, B22]], D = [[D11, D12], [0, -D11']], J = [[0, I], [-I, 0]]

such that J Q' J' D Q and U' B Q keep block triangular form, but eigenvalues are reordered.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03gd_n2_basic():
    """
    Test N=2 case with simple 2x2 matrices.

    For N=2, Q and U are Givens rotations computed from B and D.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03gd

    n = 2

    b = np.array([
        [2.0, 0.5],
        [0.0, 1.5]
    ], order='F', dtype=float)

    d = np.array([[1.0, 0.3]], order='F', dtype=float)

    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    q, u, info = mb03gd(n, b, d, macpar)

    assert info == 0
    assert q.shape == (2, 2)
    assert u.shape == (2, 2)

    np.testing.assert_allclose(q @ q.T, np.eye(2), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(u @ u.T, np.eye(2), rtol=1e-14, atol=1e-14)


def test_mb03gd_n2_orthogonality():
    """
    Verify Q and U are orthogonal for N=2 case with various matrices.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03gd

    np.random.seed(123)

    n = 2
    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    for _ in range(5):
        b11 = np.random.randn() + 1.0
        b12 = np.random.randn()
        b22 = np.random.randn() + 1.0
        b = np.array([
            [b11, b12],
            [0.0, b22]
        ], order='F', dtype=float)

        d11 = np.random.randn()
        d12 = np.random.randn()
        d = np.array([[d11, d12]], order='F', dtype=float)

        q, u, info = mb03gd(n, b, d, macpar)

        assert info == 0
        np.testing.assert_allclose(q @ q.T, np.eye(2), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(u @ u.T, np.eye(2), rtol=1e-14, atol=1e-14)


def test_mb03gd_n4_basic():
    """
    Test N=4 case with skew-Hamiltonian/Hamiltonian pencil.

    For N=4:
    - B is 4x4 upper block triangular
    - D is 2x4 (first block row of Hamiltonian matrix)

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03gd

    n = 4

    b = np.array([
        [2.0, 0.5, 0.3, 0.1],
        [0.0, 1.5, 0.2, 0.4],
        [0.0, 0.0, 1.8, 0.6],
        [0.0, 0.0, 0.0, 2.2]
    ], order='F', dtype=float)

    d = np.array([
        [1.0, 0.3, 0.5, 0.2],
        [0.0, 0.8, 0.1, 0.7]
    ], order='F', dtype=float)

    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    q, u, info = mb03gd(n, b, d, macpar)

    assert info == 0
    assert q.shape == (4, 4)
    assert u.shape == (4, 4)

    np.testing.assert_allclose(q @ q.T, np.eye(4), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(u @ u.T, np.eye(4), rtol=1e-13, atol=1e-13)


def test_mb03gd_n4_orthogonality():
    """
    Verify Q and U are orthogonal for N=4 case with random matrices.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03gd

    np.random.seed(789)

    n = 4
    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    for _ in range(3):
        b = np.zeros((4, 4), order='F', dtype=float)
        b[0, 0] = np.random.randn() + 2.0
        b[0, 1] = np.random.randn()
        b[1, 1] = np.random.randn() + 2.0
        b[0, 2] = np.random.randn()
        b[0, 3] = np.random.randn()
        b[1, 2] = np.random.randn()
        b[1, 3] = np.random.randn()
        b[2, 2] = np.random.randn() + 2.0
        b[2, 3] = np.random.randn()
        b[3, 3] = np.random.randn() + 2.0

        d = np.zeros((2, 4), order='F', dtype=float)
        d[0, 0] = np.random.randn()
        d[0, 1] = np.random.randn()
        d[1, 1] = np.random.randn()
        d[0, 2] = np.random.randn()
        d[0, 3] = np.random.randn()
        d[1, 2] = np.random.randn()
        d[1, 3] = np.random.randn()

        q, u, info = mb03gd(n, b, d, macpar)

        if info == 0:
            np.testing.assert_allclose(q @ q.T, np.eye(4), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(u @ u.T, np.eye(4), rtol=1e-13, atol=1e-13)


def test_mb03gd_n4_u_symplectic():
    """
    Verify U is orthogonal symplectic for N=4.

    Orthogonal symplectic matrix U satisfies:
    U^T U = I and U^T J U = J where J = [[0, I], [-I, 0]]

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import mb03gd

    np.random.seed(888)

    n = 4
    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    b = np.array([
        [2.5, 0.3, 0.4, 0.2],
        [0.0, 1.8, 0.1, 0.5],
        [0.0, 0.0, 2.0, 0.3],
        [0.0, 0.0, 0.0, 1.5]
    ], order='F', dtype=float)

    d = np.array([
        [1.2, 0.4, 0.6, 0.3],
        [0.0, 0.9, 0.2, 0.8]
    ], order='F', dtype=float)

    q, u, info = mb03gd(n, b, d, macpar)

    assert info == 0

    np.testing.assert_allclose(u @ u.T, np.eye(4), rtol=1e-13, atol=1e-13)

    J = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0]
    ], dtype=float)

    utju = u.T @ J @ u
    np.testing.assert_allclose(utju, J, rtol=1e-13, atol=1e-13)


def test_mb03gd_singular_b11():
    """
    Test INFO=1 when B11 is numerically singular.

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb03gd

    n = 4

    b = np.array([
        [1e-320, 1e-320, 0.3, 0.1],
        [0.0, 1e-320, 0.2, 0.4],
        [0.0, 0.0, 1.8, 0.6],
        [0.0, 0.0, 0.0, 2.2]
    ], order='F', dtype=float)

    d = np.array([
        [1.0, 0.3, 0.5, 0.2],
        [0.0, 0.8, 0.1, 0.7]
    ], order='F', dtype=float)

    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    q, u, info = mb03gd(n, b, d, macpar)

    assert info == 1


def test_mb03gd_n2_determinant():
    """
    Verify det(Q) = +/-1 and det(U) = 1 for N=2.

    Random seed: 1111 (for reproducibility)
    """
    from ctrlsys import mb03gd

    np.random.seed(1111)

    n = 2
    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    b = np.array([
        [1.5, 0.7],
        [0.0, 2.0]
    ], order='F', dtype=float)

    d = np.array([[0.8, 0.4]], order='F', dtype=float)

    q, u, info = mb03gd(n, b, d, macpar)

    assert info == 0

    det_q = np.linalg.det(q)
    det_u = np.linalg.det(u)

    np.testing.assert_allclose(np.abs(det_q), 1.0, rtol=1e-14)
    np.testing.assert_allclose(np.abs(det_u), 1.0, rtol=1e-14)


def test_mb03gd_n4_determinant():
    """
    Verify det(Q) = +/-1 and det(U) = 1 for N=4.

    Random seed: 2222 (for reproducibility)
    """
    from ctrlsys import mb03gd

    np.random.seed(2222)

    n = 4
    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    b = np.array([
        [2.0, 0.5, 0.3, 0.1],
        [0.0, 1.5, 0.2, 0.4],
        [0.0, 0.0, 1.8, 0.6],
        [0.0, 0.0, 0.0, 2.2]
    ], order='F', dtype=float)

    d = np.array([
        [1.0, 0.3, 0.5, 0.2],
        [0.0, 0.8, 0.1, 0.7]
    ], order='F', dtype=float)

    q, u, info = mb03gd(n, b, d, macpar)

    assert info == 0

    det_q = np.linalg.det(q)
    det_u = np.linalg.det(u)

    np.testing.assert_allclose(np.abs(det_q), 1.0, rtol=1e-14)
    np.testing.assert_allclose(np.abs(det_u), 1.0, rtol=1e-14)
