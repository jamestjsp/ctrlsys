"""Tests for AB05RD - Closed-loop system for mixed output and state feedback control law."""

import numpy as np
import pytest


def test_ab05rd_html_doc_example():
    """
    Test AB05RD using data from SLICOT HTML documentation example.

    System: N=3, M=2, P=2, MV=2, PZ=2
    Control law: u = alpha*F*y + beta*K*x + G*v, z = H*y
    With alpha=1.0, beta=1.0, FBTYPE='O', JOBD='D'

    Expected outputs from HTML doc:
        Ac = [[-4.8333,  0.1667, -2.8333],
              [-0.8333,  0.1667,  0.1667],
              [-1.5000,  0.5000,  1.5000]]
        Bc = [[-0.5000, -0.8333],
              [ 0.5000,  0.1667],
              [-0.5000, -0.5000]]
        Cc = [[ 1.1667, -1.8333, -0.8333],
              [ 1.8333, -1.1667, -0.1667]]
        Dc = [[ 0.5000, -0.8333],
              [ 0.5000, -0.1667]]
        RCOND = 0.2000
    """
    from ctrlsys import ab05rd

    n, m, p, mv, pz = 3, 2, 2, 2, 2
    alpha = 1.0
    beta = 1.0

    A = np.array([
        [1.0, 0.0, -1.0],
        [0.0, -1.0, 1.0],
        [1.0, 1.0, 2.0]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 2.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    C = np.array([
        [3.0, -2.0, 1.0],
        [0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    D = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    F = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    K = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ], order='F', dtype=float)

    G = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    H = np.array([
        [4.0, 3.0],
        [2.0, 1.0]
    ], order='F', dtype=float)

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('O', 'D', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F, K, G, H)

    assert info == 0

    Ac_expected = np.array([
        [-4.8333, 0.1667, -2.8333],
        [-0.8333, 0.1667, 0.1667],
        [-1.5000, 0.5000, 1.5000]
    ], order='F', dtype=float)

    Bc_expected = np.array([
        [-0.5000, -0.8333],
        [0.5000, 0.1667],
        [-0.5000, -0.5000]
    ], order='F', dtype=float)

    Cc_expected = np.array([
        [1.1667, -1.8333, -0.8333],
        [1.8333, -1.1667, -0.1667]
    ], order='F', dtype=float)

    Dc_expected = np.array([
        [0.5000, -0.8333],
        [0.5000, -0.1667]
    ], order='F', dtype=float)

    np.testing.assert_allclose(rcond, 0.2, rtol=1e-3)
    np.testing.assert_allclose(Ac, Ac_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Bc, Bc_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Cc, Cc_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Dc, Dc_expected, rtol=1e-3, atol=1e-4)


def test_ab05rd_closed_loop_formulas():
    """
    Validate closed-loop formulas from AB05RD documentation.

    Formulas:
        E = (I - alpha*D*F)^(-1)
        A1 = A + alpha*B*F*E*C,  B1 = B + alpha*B*F*E*D
        C1 = E*C,               D1 = E*D
        Ac = A1 + beta*B1*K,    Bc = B1*G
        Cc = H*(C1 + beta*D1*K), Dc = H*D1*G

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab05rd

    np.random.seed(42)
    n, m, p, mv, pz = 3, 2, 2, 2, 2
    alpha = 0.5
    beta = 0.3

    A = np.array([
        [-1.0, 0.5, 0.0],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0]
    ], order='F', dtype=float)

    D = np.array([
        [0.1, 0.0],
        [0.0, 0.1]
    ], order='F', dtype=float)

    F = np.array([
        [0.2, 0.3],
        [0.1, 0.4]
    ], order='F', dtype=float)

    K = np.array([
        [0.5, 0.2, 0.1],
        [0.1, 0.3, 0.2]
    ], order='F', dtype=float)

    G = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], order='F', dtype=float)

    H = np.array([
        [1.0, 0.0],
        [0.5, 1.0]
    ], order='F', dtype=float)

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()
    D_orig = D.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('O', 'D', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F, K, G, H)

    assert info == 0
    assert rcond > 0

    E = np.linalg.inv(np.eye(p) - alpha * D_orig @ F)
    A1 = A_orig + alpha * B_orig @ F @ E @ C_orig
    B1 = B_orig + alpha * B_orig @ F @ E @ D_orig
    C1 = E @ C_orig
    D1 = E @ D_orig

    Ac_expected = A1 + beta * B1 @ K
    Bc_expected = B1 @ G
    Cc_expected = H @ (C1 + beta * D1 @ K)
    Dc_expected = H @ D1 @ G

    np.testing.assert_allclose(Ac, Ac_expected, rtol=1e-14)
    np.testing.assert_allclose(Bc, Bc_expected, rtol=1e-14)
    np.testing.assert_allclose(Cc, Cc_expected, rtol=1e-14)
    np.testing.assert_allclose(Dc, Dc_expected, rtol=1e-14)


def test_ab05rd_identity_feedback():
    """
    Test AB05RD with identity feedback (FBTYPE='I').

    When F=I, requires M=P.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab05rd

    np.random.seed(123)
    n, m = 2, 2
    p = m
    mv, pz = 2, 2
    alpha = 0.3
    beta = 0.5

    A = np.array([
        [-1.0, 0.2],
        [0.0, -0.5]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], order='F', dtype=float)

    D = np.array([
        [0.2, 0.0],
        [0.0, 0.2]
    ], order='F', dtype=float)

    K = np.array([
        [0.4, 0.1],
        [0.2, 0.3]
    ], order='F', dtype=float)

    G = np.eye(m, order='F', dtype=float)
    H = np.eye(pz, order='F', dtype=float)

    F_dummy = np.zeros((m, p), order='F', dtype=float)

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()
    D_orig = D.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('I', 'D', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F_dummy, K, G, H)

    assert info == 0
    assert rcond > 0

    F_identity = np.eye(m)
    E = np.linalg.inv(np.eye(p) - alpha * D_orig @ F_identity)
    A1 = A_orig + alpha * B_orig @ F_identity @ E @ C_orig
    B1 = B_orig + alpha * B_orig @ F_identity @ E @ D_orig
    C1 = E @ C_orig
    D1 = E @ D_orig

    Ac_expected = A1 + beta * B1 @ K
    Bc_expected = B1 @ G
    Cc_expected = H @ (C1 + beta * D1 @ K)
    Dc_expected = H @ D1 @ G

    np.testing.assert_allclose(Ac, Ac_expected, rtol=1e-14)
    np.testing.assert_allclose(Bc, Bc_expected, rtol=1e-14)
    np.testing.assert_allclose(Cc, Cc_expected, rtol=1e-14)
    np.testing.assert_allclose(Dc, Dc_expected, rtol=1e-14)


def test_ab05rd_no_feedthrough():
    """
    Test AB05RD with D=0 (JOBD='Z').

    When D=0, E=I and formulas simplify.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import ab05rd

    np.random.seed(456)
    n, m, p, mv, pz = 4, 2, 3, 2, 3
    alpha = 1.0
    beta = 0.5

    A = np.array([
        [-2.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
        [0.0, 0.0, -0.5, 1.0],
        [0.0, 0.0, 0.0, -3.0]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0]
    ], order='F', dtype=float)

    D = np.zeros((p, m), order='F', dtype=float)

    F = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ], order='F', dtype=float)

    K = np.array([
        [0.2, 0.1, 0.0, 0.1],
        [0.1, 0.2, 0.1, 0.0]
    ], order='F', dtype=float)

    G = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], order='F', dtype=float)

    H = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ], order='F', dtype=float)

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('O', 'Z', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F, K, G, H)

    assert info == 0
    assert rcond == 1.0

    A1 = A_orig + alpha * B_orig @ F @ C_orig
    B1 = B_orig
    C1 = C_orig

    Ac_expected = A1 + beta * B1 @ K
    Bc_expected = B1 @ G
    Cc_expected = H @ (C1 + beta * np.zeros((p, m)) @ K)

    np.testing.assert_allclose(Ac, Ac_expected, rtol=1e-14)
    np.testing.assert_allclose(Bc, Bc_expected, rtol=1e-14)
    np.testing.assert_allclose(Cc, Cc_expected, rtol=1e-14)


def test_ab05rd_beta_zero():
    """
    Test AB05RD with beta=0 (no state feedback).

    When beta=0:
        Ac = A1, Bc = B1*G
        Cc = H*C1, Dc = H*D1*G

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import ab05rd

    np.random.seed(789)
    n, m, p, mv, pz = 3, 2, 2, 2, 2
    alpha = 0.5
    beta = 0.0

    A = np.array([
        [-1.0, 0.5, 0.0],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0]
    ], order='F', dtype=float)

    D = np.array([
        [0.1, 0.0],
        [0.0, 0.1]
    ], order='F', dtype=float)

    F = np.array([
        [0.2, 0.3],
        [0.1, 0.4]
    ], order='F', dtype=float)

    K = np.zeros((m, n), order='F', dtype=float)

    G = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], order='F', dtype=float)

    H = np.array([
        [1.0, 0.0],
        [0.5, 1.0]
    ], order='F', dtype=float)

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()
    D_orig = D.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('O', 'D', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F, K, G, H)

    assert info == 0

    E = np.linalg.inv(np.eye(p) - alpha * D_orig @ F)
    A1 = A_orig + alpha * B_orig @ F @ E @ C_orig
    B1 = B_orig + alpha * B_orig @ F @ E @ D_orig
    C1 = E @ C_orig
    D1 = E @ D_orig

    Ac_expected = A1
    Bc_expected = B1 @ G
    Cc_expected = H @ C1
    Dc_expected = H @ D1 @ G

    np.testing.assert_allclose(Ac, Ac_expected, rtol=1e-14)
    np.testing.assert_allclose(Bc, Bc_expected, rtol=1e-14)
    np.testing.assert_allclose(Cc, Cc_expected, rtol=1e-14)
    np.testing.assert_allclose(Dc, Dc_expected, rtol=1e-14)


def test_ab05rd_alpha_zero():
    """
    Test AB05RD with alpha=0 (no output feedback, only state feedback).

    When alpha=0, output feedback is disabled:
        A1 = A, B1 = B, C1 = C, D1 = D
        Ac = A + beta*B*K, Bc = B*G
        Cc = H*(C + beta*D*K), Dc = H*D*G

    Random seed: 321 (for reproducibility)
    """
    from ctrlsys import ab05rd

    np.random.seed(321)
    n, m, p, mv, pz = 3, 2, 2, 2, 2
    alpha = 0.0
    beta = 1.0

    A = np.array([
        [-1.0, 0.5, 0.0],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0]
    ], order='F', dtype=float)

    D = np.array([
        [0.1, 0.0],
        [0.0, 0.1]
    ], order='F', dtype=float)

    F = np.zeros((m, p), order='F', dtype=float)

    K = np.array([
        [0.5, 0.2, 0.1],
        [0.1, 0.3, 0.2]
    ], order='F', dtype=float)

    G = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], order='F', dtype=float)

    H = np.array([
        [1.0, 0.0],
        [0.5, 1.0]
    ], order='F', dtype=float)

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()
    D_orig = D.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('O', 'D', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F, K, G, H)

    assert info == 0
    assert rcond == 1.0

    Ac_expected = A_orig + beta * B_orig @ K
    Bc_expected = B_orig @ G
    Cc_expected = H @ (C_orig + beta * D_orig @ K)
    Dc_expected = H @ D_orig @ G

    np.testing.assert_allclose(Ac, Ac_expected, rtol=1e-14)
    np.testing.assert_allclose(Bc, Bc_expected, rtol=1e-14)
    np.testing.assert_allclose(Cc, Cc_expected, rtol=1e-14)
    np.testing.assert_allclose(Dc, Dc_expected, rtol=1e-14)


def test_ab05rd_singular_matrix_error():
    """
    Test that AB05RD returns INFO=1 when I - alpha*D*F is singular.

    Random seed: 654 (for reproducibility)
    """
    from ctrlsys import ab05rd

    np.random.seed(654)
    n, m, p, mv, pz = 2, 2, 2, 2, 2
    alpha = 1.0
    beta = 0.5

    A = np.array([[-1.0, 0.0], [0.0, -1.0]], order='F', dtype=float)
    B = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    C = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    D = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    F = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    K = np.array([[0.1, 0.0], [0.0, 0.1]], order='F', dtype=float)
    G = np.eye(m, order='F', dtype=float)
    H = np.eye(pz, order='F', dtype=float)

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('O', 'D', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F, K, G, H)

    assert info == 1
    assert rcond == 0.0


def test_ab05rd_n_zero():
    """
    Test edge case: n=0 (no state variables).

    Only feedthrough path exists.

    Random seed: 987 (for reproducibility)
    """
    from ctrlsys import ab05rd

    np.random.seed(987)
    n, m, p, mv, pz = 0, 2, 2, 2, 2
    alpha = 0.5
    beta = 1.0

    A = np.zeros((0, 0), order='F', dtype=float)
    B = np.zeros((0, m), order='F', dtype=float)
    C = np.zeros((p, 0), order='F', dtype=float)
    D = np.array([[0.5, 0.1], [0.2, 0.6]], order='F', dtype=float)
    F = np.array([[0.3, 0.1], [0.2, 0.4]], order='F', dtype=float)
    K = np.zeros((m, 0), order='F', dtype=float)
    G = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=float)
    H = np.array([[1.0, 0.0], [0.5, 1.0]], order='F', dtype=float)

    D_orig = D.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('O', 'D', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F, K, G, H)

    assert info == 0

    E = np.linalg.inv(np.eye(p) - alpha * D_orig @ F)
    D1 = E @ D_orig
    Dc_expected = H @ D1 @ G

    np.testing.assert_allclose(Dc, Dc_expected, rtol=1e-14)


def test_ab05rd_eigenvalue_shift():
    """
    Validate that state feedback modifies eigenvalues correctly.

    For stable A, eigenvalues of Ac differ from open-loop.

    Random seed: 135 (for reproducibility)
    """
    from ctrlsys import ab05rd

    np.random.seed(135)
    n, m, p, mv, pz = 3, 2, 2, 2, 2
    alpha = 0.0
    beta = 1.0

    A = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
    B = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], order='F', dtype=float)
    C = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]], order='F', dtype=float)
    D = np.zeros((p, m), order='F', dtype=float)
    F = np.zeros((m, p), order='F', dtype=float)
    K = np.array([[0.5, -0.3, 0.1], [-0.2, 0.4, -0.1]], order='F', dtype=float)
    G = np.eye(m, order='F', dtype=float)
    H = np.eye(pz, order='F', dtype=float)

    eig_open = np.linalg.eigvals(A)

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('O', 'Z', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F, K, G, H)

    assert info == 0

    eig_closed = np.linalg.eigvals(Ac)

    assert not np.allclose(sorted(eig_open.real), sorted(eig_closed.real), rtol=1e-10)


def test_ab05rd_state_space_property():
    """
    Validate state-space equations hold for closed-loop system.

    For closed-loop with u = G*v:
        x(k+1) = Ac*x(k) + Bc*v(k)
        z(k) = Cc*x(k) + Dc*v(k)

    Random seed: 246 (for reproducibility)
    """
    from ctrlsys import ab05rd

    np.random.seed(246)
    n, m, p, mv, pz = 3, 2, 2, 2, 2
    alpha = 0.3
    beta = 0.5

    A = np.array([
        [0.9, 0.1, 0.0],
        [0.0, 0.8, 0.1],
        [0.0, 0.0, 0.7]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0]
    ], order='F', dtype=float)

    D = np.array([
        [0.1, 0.0],
        [0.0, 0.1]
    ], order='F', dtype=float)

    F = np.array([
        [0.1, 0.1],
        [0.1, 0.1]
    ], order='F', dtype=float)

    K = np.array([
        [0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2]
    ], order='F', dtype=float)

    G = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], order='F', dtype=float)

    H = np.array([
        [1.0, 0.0],
        [0.5, 1.0]
    ], order='F', dtype=float)

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()
    D_orig = D.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05rd('O', 'D', n, m, p, mv, pz, alpha, beta,
                                          A, B, C, D, F, K, G, H)

    assert info == 0

    E = np.linalg.inv(np.eye(p) - alpha * D_orig @ F)
    A1 = A_orig + alpha * B_orig @ F @ E @ C_orig
    B1 = B_orig + alpha * B_orig @ F @ E @ D_orig
    C1 = E @ C_orig
    D1 = E @ D_orig

    Ac_expected = A1 + beta * B1 @ K
    Bc_expected = B1 @ G
    Cc_expected = H @ (C1 + beta * D1 @ K)
    Dc_expected = H @ D1 @ G

    x = np.array([[1.0], [0.5], [0.2]], order='F')
    v = np.array([[0.3], [0.4]], order='F')

    x_next = Ac @ x + Bc @ v
    z = Cc @ x + Dc @ v

    x_next_expected = Ac_expected @ x + Bc_expected @ v
    z_expected = Cc_expected @ x + Dc_expected @ v

    np.testing.assert_allclose(x_next, x_next_expected, rtol=1e-14)
    np.testing.assert_allclose(z, z_expected, rtol=1e-14)
