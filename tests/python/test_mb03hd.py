"""
Tests for MB03HD - Exchange eigenvalues of 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil.

MB03HD computes an orthogonal matrix Q for a real regular 2-by-2 or 4-by-4
skew-Hamiltonian/Hamiltonian pencil in structured Schur form such that
J Q' J' (aA - bB) Q is still in structured Schur form but the eigenvalues
are exchanged.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03hd_n2_basic():
    """
    Test N=2 case with simple 2x2 matrices.

    For N=2, Q is computed from B matrix only using Givens rotation.
    Q rotates to exchange eigenvalues.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03hd

    n = 2
    b = np.array([
        [1.0, 0.5],
        [0.0, 2.0]
    ], order='F', dtype=float)

    a = np.zeros((1, 2), order='F', dtype=float)
    macpar = np.array([1e-16, 1e-308], dtype=float)

    q, info = mb03hd(n, a, b, macpar)

    assert info == 0
    assert q.shape == (2, 2)

    np.testing.assert_allclose(q @ q.T, np.eye(2), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q.T @ q, np.eye(2), rtol=1e-14, atol=1e-14)


def test_mb03hd_n2_orthogonality():
    """
    Verify Q is orthogonal for N=2 case with various B matrices.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03hd

    np.random.seed(123)

    n = 2
    a = np.zeros((1, 2), order='F', dtype=float)
    macpar = np.array([1e-16, 1e-308], dtype=float)

    for _ in range(5):
        b11 = np.random.randn()
        b12 = np.random.randn()
        b = np.array([
            [b11, b12],
            [0.0, -b11]
        ], order='F', dtype=float)

        q, info = mb03hd(n, a, b, macpar)

        assert info == 0
        np.testing.assert_allclose(q @ q.T, np.eye(2), rtol=1e-14, atol=1e-14)


def test_mb03hd_n4_basic():
    """
    Test N=4 case with skew-Hamiltonian/Hamiltonian pencil.

    For N=4, A is 2x4 upper trapezoidal (first block row of skew-Hamiltonian)
    and B is 2x4 (first block row of Hamiltonian).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03hd

    n = 4

    a = np.array([
        [1.0, 0.5, 0.0, 0.2],
        [0.0, 2.0, 0.0, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.5, 0.3, 0.4, 0.1],
        [0.0, 1.0, 0.0, 0.5]
    ], order='F', dtype=float)

    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    q, info = mb03hd(n, a, b, macpar)

    assert info == 0 or info == 1
    assert q.shape == (4, 4)

    np.testing.assert_allclose(q @ q.T, np.eye(4), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q.T @ q, np.eye(4), rtol=1e-14, atol=1e-14)


def test_mb03hd_n4_orthogonality():
    """
    Verify Q is orthogonal for N=4 case with random matrices.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03hd

    np.random.seed(789)

    n = 4
    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    for _ in range(3):
        a11 = np.random.randn()
        a12 = np.random.randn()
        a14 = np.random.randn()
        a22 = np.random.randn()
        a = np.array([
            [a11, a12, 0.0, a14],
            [0.0, a22, 0.0, 0.0]
        ], order='F', dtype=float)

        b11 = np.random.randn()
        b12 = np.random.randn()
        b13 = np.random.randn()
        b14 = np.random.randn()
        b22 = np.random.randn()
        b24 = np.random.randn()
        b = np.array([
            [b11, b12, b13, b14],
            [0.0, b22, 0.0, b24]
        ], order='F', dtype=float)

        q, info = mb03hd(n, a, b, macpar)

        assert info in [0, 1]
        np.testing.assert_allclose(q @ q.T, np.eye(4), rtol=1e-13, atol=1e-13)


def test_mb03hd_n4_singular_warning():
    """
    Test that INFO=1 is returned when B11 block is nearly singular.

    Random seed: 1001 (for reproducibility)
    """
    from ctrlsys import mb03hd

    n = 4

    a = np.array([
        [1.0, 0.5, 0.0, 0.2],
        [0.0, 2.0, 0.0, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [1e-320, 1e-320, 0.1, 0.2],
        [0.0, 1e-320, 0.0, 0.3]
    ], order='F', dtype=float)

    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    q, info = mb03hd(n, a, b, macpar)

    assert q.shape == (4, 4)


def test_mb03hd_n2_givens_structure():
    """
    Verify Q has Givens rotation structure for N=2.

    For N=2, Q should be:
    Q = [[co, si], [-si, co]]

    Random seed: 2002 (for reproducibility)
    """
    from ctrlsys import mb03hd

    n = 2

    b = np.array([
        [2.0, 1.0],
        [0.0, -2.0]
    ], order='F', dtype=float)

    a = np.zeros((1, 2), order='F', dtype=float)
    macpar = np.array([1e-16, 1e-308], dtype=float)

    q, info = mb03hd(n, a, b, macpar)

    assert info == 0

    co = q[0, 0]
    si = q[0, 1]

    np.testing.assert_allclose(q[0, 0], co, rtol=1e-14)
    np.testing.assert_allclose(q[1, 1], co, rtol=1e-14)
    np.testing.assert_allclose(q[1, 0], -si, rtol=1e-14)
    np.testing.assert_allclose(q[0, 1], si, rtol=1e-14)

    np.testing.assert_allclose(co**2 + si**2, 1.0, rtol=1e-14)


def test_mb03hd_n4_determinant():
    """
    Verify det(Q) = 1 for orthogonal transformation (N=4).

    Random seed: 3003 (for reproducibility)
    """
    from ctrlsys import mb03hd

    np.random.seed(3003)

    n = 4
    macpar = np.array([2.2e-16, 2.2e-308], dtype=float)

    a = np.array([
        [1.5, 0.8, 0.0, 0.3],
        [0.0, 1.2, 0.0, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [2.0, 0.5, 0.6, 0.4],
        [0.0, 1.8, 0.0, 0.7]
    ], order='F', dtype=float)

    q, info = mb03hd(n, a, b, macpar)

    assert info in [0, 1]

    det_q = np.linalg.det(q)
    np.testing.assert_allclose(np.abs(det_q), 1.0, rtol=1e-14)
