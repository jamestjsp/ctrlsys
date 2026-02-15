"""
Tests for mb03qg - Reorder diagonal blocks of an upper quasi-triangular
matrix pencil A-lambda*E.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03qg_basic_continuous_stable():
    """
    Test basic continuous-time stability ordering from SLICOT HTML doc example.

    Reorders pencil (A,E) to place eigenvalues with Re < 0 first.
    Uses DGGES to compute initial Schur form, then MB03QG to reorder.
    """
    from ctrlsys import mb03qg

    n = 4
    nlow = 1
    nsup = 4
    alpha = 0.0

    a_schur = np.array([
        [-1.4394,  2.5550, -12.5655, -4.0714],
        [ 2.8887, -1.1242,   9.2819, -2.6724],
        [ 0.0000,  0.0000, -19.7785, 36.4447],
        [ 0.0000,  0.0000,   0.0000,  3.5537]
    ], order='F', dtype=float)

    e_schur = np.array([
        [-16.0178,  0.0000,  2.3850,  4.7645],
        [  0.0000,  3.2809, -1.5640,  1.9954],
        [  0.0000,  0.0000, -3.0652,  0.3039],
        [  0.0000,  0.0000,  0.0000,  1.1671]
    ], order='F', dtype=float)

    u_expected = np.array([
        [-0.1518, -0.0737, -0.9856,  0.0140],
        [-0.2865, -0.9466,  0.1136, -0.0947],
        [-0.5442,  0.0924,  0.0887,  0.8292],
        [-0.7738,  0.3000,  0.0890, -0.5508]
    ], order='F', dtype=float)

    v_expected = np.array([
        [ 0.2799,  0.9041,  0.2685,  0.1794],
        [ 0.4009, -0.0714,  0.3780, -0.8315],
        [ 0.7206, -0.4006,  0.2628,  0.5012],
        [ 0.4917,  0.1306, -0.8462, -0.1588]
    ], order='F', dtype=float)

    a_out, e_out, u_out, v_out, ndim, info = mb03qg(
        'C', 'S', 'I', 'I', a_schur.copy(), e_schur.copy(), nlow, nsup, alpha
    )

    assert info == 0
    assert ndim == 2

    assert_allclose(a_out, a_schur, rtol=1e-3, atol=1e-3)
    assert_allclose(e_out, e_schur, rtol=1e-3, atol=1e-3)


def test_mb03qg_discrete_stable():
    """
    Test discrete-time stability ordering.

    Reorders pencil (A,E) to place eigenvalues with |lambda| < alpha first.
    Random seed: 42
    """
    from ctrlsys import mb03qg

    np.random.seed(42)
    n = 3

    a = np.array([
        [0.5, 0.2, 0.1],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 0.8]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.1, 0.2],
        [0.0, 1.0, 0.1],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    a_out, e_out, u_out, v_out, ndim, info = mb03qg(
        'D', 'S', 'I', 'I', a.copy(), e.copy(), 1, n, 1.0
    )

    assert info == 0
    assert ndim == 2

    assert u_out.shape == (n, n)
    assert v_out.shape == (n, n)

    assert_allclose(u_out @ u_out.T, np.eye(n), rtol=1e-10, atol=1e-10)
    assert_allclose(v_out @ v_out.T, np.eye(n), rtol=1e-10, atol=1e-10)


def test_mb03qg_discrete_unstable():
    """
    Test discrete-time instability ordering.

    Reorders pencil (A,E) to place eigenvalues with |lambda| > alpha first.
    Random seed: 123
    """
    from ctrlsys import mb03qg

    np.random.seed(123)
    n = 3

    a = np.array([
        [0.5, 0.2, 0.1],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 0.8]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.1, 0.2],
        [0.0, 1.0, 0.1],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    a_out, e_out, u_out, v_out, ndim, info = mb03qg(
        'D', 'U', 'I', 'I', a.copy(), e.copy(), 1, n, 1.0
    )

    assert info == 0
    assert ndim == 1

    assert_allclose(u_out @ u_out.T, np.eye(n), rtol=1e-10, atol=1e-10)
    assert_allclose(v_out @ v_out.T, np.eye(n), rtol=1e-10, atol=1e-10)


def test_mb03qg_update_transformation():
    """
    Test update mode for transformation matrices (JOBU='U', JOBV='U').

    Verifies that existing U, V matrices are updated correctly.
    Random seed: 456
    """
    from ctrlsys import mb03qg

    np.random.seed(456)
    n = 3

    a = np.array([
        [0.5, 0.2, 0.1],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 0.8]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.1, 0.2],
        [0.0, 1.0, 0.1],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    u_init = np.eye(n, order='F', dtype=float)
    v_init = np.eye(n, order='F', dtype=float)

    a_out, e_out, u_out, v_out, ndim, info = mb03qg(
        'D', 'S', 'U', 'U', a.copy(), e.copy(), 1, n, 1.0,
        u=u_init, v=v_init
    )

    assert info == 0

    assert_allclose(u_out @ u_out.T, np.eye(n), rtol=1e-10, atol=1e-10)
    assert_allclose(v_out @ v_out.T, np.eye(n), rtol=1e-10, atol=1e-10)


def test_mb03qg_eigenvalue_preservation():
    """
    Test that generalized eigenvalues are preserved after reordering.

    The eigenvalues of (A,E) should be preserved, only reordered.
    Random seed: 789
    """
    from ctrlsys import mb03qg

    np.random.seed(789)
    n = 4

    a = np.array([
        [-0.5,  0.3, 0.1, 0.2],
        [ 0.0,  1.2, 0.4, 0.1],
        [ 0.0,  0.0, 0.3, 0.2],
        [ 0.0,  0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.2, 0.1, 0.3],
        [0.0, 1.0, 0.2, 0.1],
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    eig_before = np.linalg.eigvals(np.linalg.solve(e, a))
    eig_before_sorted = sorted(eig_before.real)

    a_out, e_out, u_out, v_out, ndim, info = mb03qg(
        'C', 'S', 'I', 'I', a.copy(), e.copy(), 1, n, 1.0
    )

    assert info == 0

    eig_after = np.linalg.eigvals(np.linalg.solve(e_out, a_out))
    eig_after_sorted = sorted(eig_after.real)

    assert_allclose(eig_before_sorted, eig_after_sorted, rtol=1e-10, atol=1e-10)


def test_mb03qg_subpencil_reorder():
    """
    Test reordering only a subpencil (NLOW > 1).

    Random seed: 999
    """
    from ctrlsys import mb03qg

    np.random.seed(999)
    n = 5

    a = np.array([
        [-0.5, 0.3, 0.1, 0.2, 0.1],
        [ 0.0, 1.2, 0.4, 0.1, 0.2],
        [ 0.0, 0.0,-0.3, 0.2, 0.1],
        [ 0.0, 0.0, 0.0, 0.8, 0.3],
        [ 0.0, 0.0, 0.0, 0.0,-1.5]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    a_out, e_out, u_out, v_out, ndim, info = mb03qg(
        'C', 'S', 'I', 'I', a.copy(), e.copy(), 2, 4, 0.0
    )

    assert info == 0

    assert_allclose(u_out @ u_out.T, np.eye(n), rtol=1e-10, atol=1e-10)
    assert_allclose(v_out @ v_out.T, np.eye(n), rtol=1e-10, atol=1e-10)


def test_mb03qg_quick_return():
    """
    Test quick return when NSUP=0.
    """
    from ctrlsys import mb03qg

    n = 3
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)

    a_out, e_out, u_out, v_out, ndim, info = mb03qg(
        'C', 'S', 'I', 'I', a.copy(), e.copy(), 0, 0, 0.0
    )

    assert info == 0
    assert ndim == 0


def test_mb03qg_invalid_dico():
    """Test error handling for invalid DICO parameter."""
    from ctrlsys import mb03qg

    n = 3
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)

    with pytest.raises(ValueError):
        mb03qg('X', 'S', 'I', 'I', a, e, 1, n, 0.0)


def test_mb03qg_invalid_alpha_discrete():
    """Test error handling for negative alpha in discrete mode."""
    from ctrlsys import mb03qg

    n = 3
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)

    with pytest.raises(ValueError):
        mb03qg('D', 'S', 'I', 'I', a, e, 1, n, -1.0)
