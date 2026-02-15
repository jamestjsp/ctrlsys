"""
Tests for SB10ID - Positive feedback controller for loop shaping design.

SB10ID computes the matrices of the positive feedback controller K = [Ak, Bk; Ck, Dk]
for a shaped plant G = [A, B; C, D] in the McFarlane/Glover Loop Shaping Design Procedure.

The routine implements H-infinity loop shaping formulas from:
    McFarlane, D. and Glover, K.
    "A loop shaping design procedure using H_infinity synthesis."
    IEEE Trans. Automat. Control, vol. AC-37, no. 6, pp. 759-769, 1992.
"""
import numpy as np
import pytest
from ctrlsys import sb10id


def test_sb10id_html_doc_example():
    """
    Test SB10ID using HTML documentation example.

    N=6, M=2, NP=3, FACTOR=1.0 (optimal controller)
    Expected NK=5 (controller order less than plant order).
    """
    n, m, np_ = 6, 2, 3
    factor = 1.0

    a = np.array([
        [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
        [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
        [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
        [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
        [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
        [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
    ], order='F', dtype=np.float64)

    b = np.array([
        [-3.0, -4.0],
        [ 2.0,  0.0],
        [-5.0, -7.0],
        [ 4.0, -6.0],
        [-3.0,  9.0],
        [ 1.0, -2.0]
    ], order='F', dtype=np.float64)

    c = np.array([
        [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
        [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
        [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0]
    ], order='F', dtype=np.float64)

    d = np.array([
        [1.0, -2.0],
        [0.0,  4.0],
        [5.0, -3.0]
    ], order='F', dtype=np.float64)

    ak, bk, ck, dk, nk, rcond, info = sb10id(n, m, np_, a, b, c, d, factor)

    assert info == 0
    assert nk == 5

    ak_expected = np.array([
        [-39.0671,   9.9293,  22.2322, -27.4113,  43.8655],
        [ -6.6117,   3.0006,  11.0878, -11.4130,  15.4269],
        [ 33.6805,  -6.6934, -23.9953,  14.1438, -33.4358],
        [-32.3191,   9.7316,  25.4033, -24.0473,  42.0517],
        [-44.1655,  18.7767,  34.8873, -42.4369,  50.8437]
    ], order='F', dtype=np.float64)

    bk_expected = np.array([
        [-10.2905, -16.5382, -10.9782],
        [ -4.3598,  -8.7525,  -5.1447],
        [  6.5962,   1.8975,   6.2316],
        [ -9.8770, -14.7041, -11.8778],
        [ -9.6726, -22.7309, -18.2692]
    ], order='F', dtype=np.float64)

    ck_expected = np.array([
        [-0.6647, -0.0599, -1.0376,  0.5619,  1.7297],
        [-8.4202,  3.9573,  7.3094, -7.6283, 10.6768]
    ], order='F', dtype=np.float64)

    dk_expected = np.array([
        [ 0.8466,  0.4979, -0.6993],
        [-1.2226, -4.8689, -4.5056]
    ], order='F', dtype=np.float64)

    np.testing.assert_allclose(ak[:nk, :nk], ak_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(bk[:nk, :np_], bk_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ck[:m, :nk], ck_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dk[:m, :np_], dk_expected, rtol=1e-3, atol=1e-3)

    assert 0 < rcond[0] <= 1.0
    assert 0 < rcond[1] <= 1.0


def test_sb10id_suboptimal_controller():
    """
    Test suboptimal controller with FACTOR > 1.

    A factor > 1 should produce a suboptimal controller with
    different (typically simpler or more robust) characteristics.

    Random seed: 42 (for reproducibility)
    """
    n, m, np_ = 6, 2, 3
    factor = 1.1

    a = np.array([
        [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
        [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
        [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
        [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
        [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
        [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
    ], order='F', dtype=np.float64)

    b = np.array([
        [-3.0, -4.0],
        [ 2.0,  0.0],
        [-5.0, -7.0],
        [ 4.0, -6.0],
        [-3.0,  9.0],
        [ 1.0, -2.0]
    ], order='F', dtype=np.float64)

    c = np.array([
        [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
        [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
        [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0]
    ], order='F', dtype=np.float64)

    d = np.array([
        [1.0, -2.0],
        [0.0,  4.0],
        [5.0, -3.0]
    ], order='F', dtype=np.float64)

    ak, bk, ck, dk, nk, rcond, info = sb10id(n, m, np_, a, b, c, d, factor)

    assert info == 0
    assert nk <= n
    assert rcond[0] > 0
    assert rcond[1] > 0


def test_sb10id_closed_loop_stability():
    """
    Test that closed-loop system is stable.

    The positive feedback controller K should stabilize the closed-loop system.
    Closed-loop state matrix eigenvalues must have negative real parts.

    Random seed: 123 (for reproducibility)
    """
    n, m, np_ = 6, 2, 3
    factor = 1.0

    a = np.array([
        [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
        [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
        [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
        [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
        [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
        [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
    ], order='F', dtype=np.float64)

    b = np.array([
        [-3.0, -4.0],
        [ 2.0,  0.0],
        [-5.0, -7.0],
        [ 4.0, -6.0],
        [-3.0,  9.0],
        [ 1.0, -2.0]
    ], order='F', dtype=np.float64)

    c = np.array([
        [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
        [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
        [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0]
    ], order='F', dtype=np.float64)

    d = np.array([
        [1.0, -2.0],
        [0.0,  4.0],
        [5.0, -3.0]
    ], order='F', dtype=np.float64)

    ak, bk, ck, dk, nk, rcond, info = sb10id(n, m, np_, a, b, c, d, factor)

    assert info == 0

    ak_sub = ak[:nk, :nk]
    bk_sub = bk[:nk, :np_]
    ck_sub = ck[:m, :nk]
    dk_sub = dk[:m, :np_]

    iddk = np.eye(np_) - d @ dk_sub
    iddk_inv = np.linalg.inv(iddk)

    imdk = np.eye(m) - dk_sub @ d
    imdk_inv = np.linalg.inv(imdk)

    acl = np.zeros((n + nk, n + nk), dtype=np.float64)
    acl[:n, :n] = a + b @ imdk_inv @ dk_sub @ c
    acl[:n, n:] = b @ imdk_inv @ ck_sub
    acl[n:, :n] = bk_sub @ iddk_inv @ c
    acl[n:, n:] = ak_sub + bk_sub @ iddk_inv @ d @ ck_sub

    eig_cl = np.linalg.eigvals(acl)
    max_real = np.max(eig_cl.real)
    assert max_real < 0, f"Closed-loop unstable: max eigenvalue real part = {max_real}"


def test_sb10id_quick_return_n_zero():
    """Test quick return when N=0."""
    n, m, np_ = 0, 2, 3
    factor = 1.0

    a = np.zeros((1, 1), order='F', dtype=np.float64)
    b = np.zeros((1, m), order='F', dtype=np.float64)
    c = np.zeros((np_, 1), order='F', dtype=np.float64)
    d = np.zeros((np_, m), order='F', dtype=np.float64)

    ak, bk, ck, dk, nk, rcond, info = sb10id(n, m, np_, a, b, c, d, factor)

    assert info == 0
    assert rcond[0] == 1.0
    assert rcond[1] == 1.0


def test_sb10id_quick_return_m_zero():
    """Test quick return when M=0."""
    n, m, np_ = 3, 0, 2
    factor = 1.0

    a = -np.eye(n, order='F', dtype=np.float64)
    b = np.zeros((n, 1), order='F', dtype=np.float64)
    c = np.zeros((np_, n), order='F', dtype=np.float64)
    d = np.zeros((np_, 1), order='F', dtype=np.float64)

    ak, bk, ck, dk, nk, rcond, info = sb10id(n, m, np_, a, b, c, d, factor)

    assert info == 0
    assert rcond[0] == 1.0
    assert rcond[1] == 1.0


def test_sb10id_quick_return_np_zero():
    """Test quick return when NP=0."""
    n, m, np_ = 3, 2, 0
    factor = 1.0

    a = -np.eye(n, order='F', dtype=np.float64)
    b = np.ones((n, m), order='F', dtype=np.float64)
    c = np.zeros((1, n), order='F', dtype=np.float64)
    d = np.zeros((1, m), order='F', dtype=np.float64)

    ak, bk, ck, dk, nk, rcond, info = sb10id(n, m, np_, a, b, c, d, factor)

    assert info == 0
    assert rcond[0] == 1.0
    assert rcond[1] == 1.0


def test_sb10id_invalid_factor():
    """Test error for FACTOR < 1."""
    n, m, np_ = 3, 2, 2
    factor = 0.9

    a = -np.eye(n, order='F', dtype=np.float64)
    b = np.ones((n, m), order='F', dtype=np.float64)
    c = np.ones((np_, n), order='F', dtype=np.float64)
    d = np.zeros((np_, m), order='F', dtype=np.float64)

    ak, bk, ck, dk, nk, rcond, info = sb10id(n, m, np_, a, b, c, d, factor)

    assert info == -12


def test_sb10id_rcond_values():
    """
    Test that RCOND values are reasonable.

    RCOND(1) is reciprocal condition number of X-Riccati equation.
    RCOND(2) is reciprocal condition number of Z-Riccati equation.
    Both should be in (0, 1] for well-conditioned problems.
    """
    n, m, np_ = 6, 2, 3
    factor = 1.0

    a = np.array([
        [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
        [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
        [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
        [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
        [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
        [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
    ], order='F', dtype=np.float64)

    b = np.array([
        [-3.0, -4.0],
        [ 2.0,  0.0],
        [-5.0, -7.0],
        [ 4.0, -6.0],
        [-3.0,  9.0],
        [ 1.0, -2.0]
    ], order='F', dtype=np.float64)

    c = np.array([
        [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
        [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
        [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0]
    ], order='F', dtype=np.float64)

    d = np.array([
        [1.0, -2.0],
        [0.0,  4.0],
        [5.0, -3.0]
    ], order='F', dtype=np.float64)

    ak, bk, ck, dk, nk, rcond, info = sb10id(n, m, np_, a, b, c, d, factor)

    assert info == 0
    assert 0 < rcond[0] <= 1.0
    assert 0 < rcond[1] <= 1.0
