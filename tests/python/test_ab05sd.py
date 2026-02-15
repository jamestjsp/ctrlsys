"""Tests for AB05SD - Output feedback closed-loop system construction."""

import numpy as np
import pytest


def test_ab05sd_general_feedback_with_d():
    """
    Test AB05SD with general output feedback (FBTYPE='O') and D present (JOBD='D').

    System: 3 states, 2 inputs, 2 outputs
    Control law: u = alpha*F*y + v with alpha = 0.5

    Closed-loop formulas:
        E = (I - alpha*D*F)^(-1)
        Ac = A + alpha*B*F*E*C
        Bc = B + alpha*B*F*E*D
        Cc = E*C
        Dc = E*D

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab05sd

    np.random.seed(42)
    n, m, p = 3, 2, 2
    alpha = 0.5

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

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()
    D_orig = D.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05sd('O', 'D', n, m, p, alpha, A, B, C, D, F)

    assert info == 0
    assert rcond > 0

    E = np.linalg.inv(np.eye(p) - alpha * D_orig @ F)
    Ac_expected = A_orig + alpha * B_orig @ F @ E @ C_orig
    Bc_expected = B_orig + alpha * B_orig @ F @ E @ D_orig
    Cc_expected = E @ C_orig
    Dc_expected = E @ D_orig

    np.testing.assert_allclose(Ac, Ac_expected, rtol=1e-14)
    np.testing.assert_allclose(Bc, Bc_expected, rtol=1e-14)
    np.testing.assert_allclose(Cc, Cc_expected, rtol=1e-14)
    np.testing.assert_allclose(Dc, Dc_expected, rtol=1e-14)


def test_ab05sd_identity_feedback():
    """
    Test AB05SD with identity feedback (FBTYPE='I').

    When F = I, the closed-loop system simplifies (requires M = P).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab05sd

    np.random.seed(123)
    n, m = 2, 2
    p = m  # Must equal M for identity feedback
    alpha = 0.3

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

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()
    D_orig = D.copy()

    F_dummy = np.zeros((m, p), order='F', dtype=float)

    Ac, Bc, Cc, Dc, rcond, info = ab05sd('I', 'D', n, m, p, alpha, A, B, C, D, F_dummy)

    assert info == 0
    assert rcond > 0

    F_identity = np.eye(m)
    E = np.linalg.inv(np.eye(p) - alpha * D_orig @ F_identity)
    Ac_expected = A_orig + alpha * B_orig @ F_identity @ E @ C_orig
    Bc_expected = B_orig + alpha * B_orig @ F_identity @ E @ D_orig
    Cc_expected = E @ C_orig
    Dc_expected = E @ D_orig

    np.testing.assert_allclose(Ac, Ac_expected, rtol=1e-14)
    np.testing.assert_allclose(Bc, Bc_expected, rtol=1e-14)
    np.testing.assert_allclose(Cc, Cc_expected, rtol=1e-14)
    np.testing.assert_allclose(Dc, Dc_expected, rtol=1e-14)


def test_ab05sd_no_feedthrough():
    """
    Test AB05SD with D=0 (JOBD='Z').

    When D=0, E=I and formulas simplify:
        Ac = A + alpha*B*F*C
        Bc = B
        Cc = C
        Dc = 0

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import ab05sd

    np.random.seed(456)
    n, m, p = 4, 2, 3
    alpha = 1.0

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

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05sd('O', 'Z', n, m, p, alpha, A, B, C, D, F)

    assert info == 0
    assert rcond == 1.0  # E = I when D = 0

    Ac_expected = A_orig + alpha * B_orig @ F @ C_orig
    Bc_expected = B_orig
    Cc_expected = C_orig

    np.testing.assert_allclose(Ac, Ac_expected, rtol=1e-14)
    np.testing.assert_allclose(Bc, Bc_expected, rtol=1e-14)
    np.testing.assert_allclose(Cc, Cc_expected, rtol=1e-14)


def test_ab05sd_alpha_zero():
    """
    Test AB05SD with alpha=0 (no feedback, quick return).

    When alpha=0, closed-loop = open-loop:
        Ac = A, Bc = B, Cc = C, Dc = D

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import ab05sd

    np.random.seed(789)
    n, m, p = 2, 1, 1
    alpha = 0.0

    A = np.array([
        [-1.0, 0.5],
        [0.2, -0.8]
    ], order='F', dtype=float)

    B = np.array([
        [1.0],
        [0.5]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.0]
    ], order='F', dtype=float)

    D = np.array([
        [0.1]
    ], order='F', dtype=float)

    F = np.array([
        [5.0]
    ], order='F', dtype=float)

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()
    D_orig = D.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05sd('O', 'D', n, m, p, alpha, A, B, C, D, F)

    assert info == 0
    assert rcond == 1.0

    np.testing.assert_allclose(Ac, A_orig, rtol=1e-14)
    np.testing.assert_allclose(Bc, B_orig, rtol=1e-14)
    np.testing.assert_allclose(Cc, C_orig, rtol=1e-14)
    np.testing.assert_allclose(Dc, D_orig, rtol=1e-14)


def test_ab05sd_eigenvalue_shift():
    """
    Validate that output feedback modifies eigenvalues correctly.

    For stable A and proper choice of F, closed-loop poles differ from open-loop.

    Random seed: 321 (for reproducibility)
    """
    from ctrlsys import ab05sd

    np.random.seed(321)
    n, m, p = 3, 2, 2
    alpha = 1.0

    A = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
    B = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], order='F', dtype=float)
    C = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]], order='F', dtype=float)
    D = np.zeros((p, m), order='F', dtype=float)

    F = np.array([[0.5, -0.3], [-0.2, 0.4]], order='F', dtype=float)

    eig_open = np.linalg.eigvals(A)

    Ac, Bc, Cc, Dc, rcond, info = ab05sd('O', 'Z', n, m, p, alpha, A, B, C, D, F)

    assert info == 0

    eig_closed = np.linalg.eigvals(Ac)

    assert not np.allclose(sorted(eig_open.real), sorted(eig_closed.real), rtol=1e-10)


def test_ab05sd_singular_matrix_error():
    """
    Test that AB05SD returns INFO=1 when I - alpha*D*F is singular.

    Random seed: 654 (for reproducibility)
    """
    from ctrlsys import ab05sd

    np.random.seed(654)
    n, m, p = 2, 2, 2
    alpha = 1.0

    A = np.array([[-1.0, 0.0], [0.0, -1.0]], order='F', dtype=float)
    B = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    C = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    D = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
    F = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)

    Ac, Bc, Cc, Dc, rcond, info = ab05sd('O', 'D', n, m, p, alpha, A, B, C, D, F)

    assert info == 1
    assert rcond == 0.0


def test_ab05sd_n_zero():
    """
    Test edge case: n=0 (no state variables).

    Only feedthrough path exists: Dc = E*D.

    Random seed: 987 (for reproducibility)
    """
    from ctrlsys import ab05sd

    np.random.seed(987)
    n, m, p = 0, 2, 2
    alpha = 0.5

    A = np.zeros((0, 0), order='F', dtype=float)
    B = np.zeros((0, m), order='F', dtype=float)
    C = np.zeros((p, 0), order='F', dtype=float)
    D = np.array([[0.5, 0.1], [0.2, 0.6]], order='F', dtype=float)
    F = np.array([[0.3, 0.1], [0.2, 0.4]], order='F', dtype=float)

    D_orig = D.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05sd('O', 'D', n, m, p, alpha, A, B, C, D, F)

    assert info == 0

    E = np.linalg.inv(np.eye(p) - alpha * D_orig @ F)
    Dc_expected = E @ D_orig

    np.testing.assert_allclose(Dc, Dc_expected, rtol=1e-14)


def test_ab05sd_m_or_p_zero():
    """
    Test edge case: m=0 or p=0 (quick return).

    With no inputs or outputs, closed-loop = open-loop.

    Random seed: 246 (for reproducibility)
    """
    from ctrlsys import ab05sd

    np.random.seed(246)
    n, m, p = 2, 0, 1
    alpha = 1.0

    A = np.array([[-1.0, 0.5], [0.0, -2.0]], order='F', dtype=float)
    B = np.zeros((n, m), order='F', dtype=float)
    C = np.array([[1.0, 0.0]], order='F', dtype=float)
    D = np.zeros((p, m), order='F', dtype=float)
    F = np.zeros((m, p), order='F', dtype=float)

    A_orig = A.copy()
    C_orig = C.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05sd('O', 'D', n, m, p, alpha, A, B, C, D, F)

    assert info == 0
    assert rcond == 1.0

    np.testing.assert_allclose(Ac, A_orig, rtol=1e-14)
    np.testing.assert_allclose(Cc, C_orig, rtol=1e-14)


def test_ab05sd_transfer_function_property():
    """
    Validate transfer function relation at DC (s=0).

    For stable systems: G_cl(0) = Dc - Cc*inv(Ac)*Bc
    With D=0: G_cl(0) = (I-alpha*G(0)*F)^(-1)*G(0) where G(0) = -C*inv(A)*B

    Random seed: 135 (for reproducibility)
    """
    from ctrlsys import ab05sd

    np.random.seed(135)
    n, m, p = 3, 2, 2
    alpha = 0.5

    A = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    C = np.random.randn(p, n).astype(float, order='F')
    D = np.zeros((p, m), order='F', dtype=float)
    F = np.random.randn(m, p).astype(float, order='F')

    A_orig = A.copy()
    B_orig = B.copy()
    C_orig = C.copy()

    Ac, Bc, Cc, Dc, rcond, info = ab05sd('O', 'Z', n, m, p, alpha, A, B, C, D, F)

    assert info == 0

    G0 = -C_orig @ np.linalg.solve(A_orig, B_orig)
    G_cl0_expected = np.linalg.solve(np.eye(p) - alpha * G0 @ F, G0)

    G_cl0 = -Cc @ np.linalg.solve(Ac, Bc)

    np.testing.assert_allclose(G_cl0, G_cl0_expected, rtol=1e-13)
