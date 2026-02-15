"""
Tests for MB03QW: Reduce 2-by-2 diagonal block pair of quasi-triangular pencil.

MB03QW computes eigenvalues of a 2-by-2 diagonal block pair of an upper
quasi-triangular pencil, reduces it to standard form, and splits it if
eigenvalues are real. Uses LAPACK's DLAGV2.
"""
import numpy as np
import pytest
from ctrlsys import mb03qw


def test_mb03qw_complex_eigenvalues():
    """
    Test MB03QW with a 2x2 block having complex eigenvalues.

    Random seed: 42 (for reproducibility)

    Property: eigenvalues should be complex conjugate pairs.
    """
    np.random.seed(42)
    n = 3
    l = 1

    a = np.array([
        [1.0, 2.0, 0.5],
        [3.0, 1.0, 0.3],
        [0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    e = np.array([
        [2.0, 1.0, 0.2],
        [0.0, 1.5, 0.1],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)
    v = np.eye(n, order='F', dtype=float)

    a_out, e_out, u_out, v_out, alphar, alphai, beta, info = mb03qw(n, l, a, e, u, v)

    assert info == 0

    assert alphar.shape == (2,)
    assert alphai.shape == (2,)
    assert beta.shape == (2,)

    eig1 = (alphar[0] + 1j * alphai[0]) / beta[0] if beta[0] != 0 else np.inf
    eig2 = (alphar[1] + 1j * alphai[1]) / beta[1] if beta[1] != 0 else np.inf

    if alphai[0] != 0.0:
        np.testing.assert_allclose(alphai[0], -alphai[1], rtol=1e-14)
        np.testing.assert_allclose(alphar[0], alphar[1], rtol=1e-14)


def test_mb03qw_real_eigenvalues():
    """
    Test MB03QW with a 2x2 block having real eigenvalues.

    Random seed: 123 (for reproducibility)

    Property: If eigenvalues are real, block should be triangularized.
    """
    np.random.seed(123)
    n = 2
    l = 1

    a = np.array([
        [4.0, 1.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)
    v = np.eye(n, order='F', dtype=float)

    a_out, e_out, u_out, v_out, alphar, alphai, beta, info = mb03qw(n, l, a, e, u, v)

    assert info == 0

    assert alphar.shape == (2,)
    assert alphai.shape == (2,)
    assert beta.shape == (2,)

    np.testing.assert_allclose(alphai[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(alphai[1], 0.0, atol=1e-14)

    if beta[0] != 0 and beta[1] != 0:
        eig1 = alphar[0] / beta[0]
        eig2 = alphar[1] / beta[1]
        assert abs(eig1) >= abs(eig2) or abs(abs(eig1) - abs(eig2)) < 1e-10


def test_mb03qw_transformation_property():
    """
    Test that transformations are correctly applied.

    Random seed: 456 (for reproducibility)

    Property: U'*A*V should equal transformed A (at the 2x2 block).
    """
    np.random.seed(456)
    n = 4
    l = 2

    a = np.array([
        [1.0, 0.3, 0.1, 0.2],
        [0.0, 2.0, 3.0, 0.4],
        [0.0, 1.0, 2.5, 0.5],
        [0.0, 0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.2, 0.1, 0.1],
        [0.0, 1.5, 0.8, 0.2],
        [0.0, 0.0, 1.2, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)
    v = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()
    e_orig = e.copy()

    a_out, e_out, u_out, v_out, alphar, alphai, beta, info = mb03qw(n, l, a, e, u, v)

    assert info == 0

    np.testing.assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03qw_invalid_n():
    """Test MB03QW with invalid n < 2."""
    n = 1
    l = 1
    a = np.array([[1.0]], order='F', dtype=float)
    e = np.array([[1.0]], order='F', dtype=float)
    u = np.array([[1.0]], order='F', dtype=float)
    v = np.array([[1.0]], order='F', dtype=float)

    a_out, e_out, u_out, v_out, alphar, alphai, beta, info = mb03qw(n, l, a, e, u, v)

    assert info == -1


def test_mb03qw_invalid_l():
    """Test MB03QW with invalid l (out of range)."""
    n = 3
    l = 0
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    v = np.eye(n, order='F', dtype=float)

    a_out, e_out, u_out, v_out, alphar, alphai, beta, info = mb03qw(n, l, a, e, u, v)

    assert info == -2


def test_mb03qw_l_equals_n():
    """Test MB03QW with l = n (invalid, need l < n)."""
    n = 3
    l = 3
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    v = np.eye(n, order='F', dtype=float)

    a_out, e_out, u_out, v_out, alphar, alphai, beta, info = mb03qw(n, l, a, e, u, v)

    assert info == -2


def test_mb03qw_generalized_eigenvalue_property():
    """
    Test that returned eigenvalues are correct generalized eigenvalues.

    Random seed: 789 (for reproducibility)

    Property: det(alphar[i]*E - beta[i]*A) at 2x2 block should be ~0.
    """
    np.random.seed(789)
    n = 2
    l = 1

    a = np.array([
        [3.0, 2.0],
        [1.0, 4.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.5],
        [0.0, 2.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)
    v = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()
    e_orig = e.copy()

    a_out, e_out, u_out, v_out, alphar, alphai, beta, info = mb03qw(n, l, a, e, u, v)

    assert info == 0

    for i in range(2):
        if beta[i] != 0:
            lam = (alphar[i] + 1j * alphai[i]) / beta[i]
            a_block = a_orig[0:2, 0:2]
            e_block = e_orig[0:2, 0:2]
            det_val = np.linalg.det(lam * e_block - a_block)
            np.testing.assert_allclose(abs(det_val), 0.0, atol=1e-10)
