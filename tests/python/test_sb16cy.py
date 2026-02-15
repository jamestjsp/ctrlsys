"""
Tests for SB16CY - Cholesky factors of controllability/observability Grammians
for coprime factors of state-feedback controller.

SB16CY computes Cholesky factors Su and Ru of:
- Controllability Grammian P = Su*Su'
- Observability Grammian Q = Ru'*Ru

For left coprime factorization (JOBCF='L'):
  Continuous: (A+B*F)*P + P*(A+B*F)' + scalec^2*B*B' = 0
              (A+G*C)'*Q + Q*(A+G*C) + scaleo^2*F'*F = 0
  Discrete:   (A+B*F)*P*(A+B*F)' - P + scalec^2*B*B' = 0
              (A+G*C)'*Q*(A+G*C) - Q + scaleo^2*F'*F = 0

For right coprime factorization (JOBCF='R'):
  Continuous: (A+B*F)*P + P*(A+B*F)' + scalec^2*G*G' = 0
              (A+G*C)'*Q + Q*(A+G*C) + scaleo^2*C'*C = 0
  Discrete:   (A+B*F)*P*(A+B*F)' - P + scalec^2*G*G' = 0
              (A+G*C)'*Q*(A+G*C) - Q + scaleo^2*C'*C = 0
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import sb16cy


def test_sb16cy_continuous_left():
    """
    Test continuous-time left coprime factorization.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n, m, p = 3, 2, 2

    A = np.array([
        [-2.0, 0.0, 0.0],
        [0.0, -3.0, 0.0],
        [0.0, 0.0, -4.0]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]
    ], order='F', dtype=float)

    F = np.array([
        [-1.0, 0.5, 0.0],
        [0.0, -0.5, 0.3]
    ], order='F', dtype=float)

    G = np.array([
        [0.5, -0.2],
        [-0.3, 0.4],
        [0.1, -0.1]
    ], order='F', dtype=float)

    Acl_bf = A + B @ F
    Acl_gc = A + G @ C
    eig_bf = np.linalg.eigvals(Acl_bf)
    eig_gc = np.linalg.eigvals(Acl_gc)
    assert all(eig_bf.real < 0), f"A+B*F not stable: {eig_bf}"
    assert all(eig_gc.real < 0), f"A+G*C not stable: {eig_gc}"

    S, R, scalec, scaleo, info = sb16cy('C', 'L', n, m, p, A, B, C, F, G)

    assert info == 0, f"sb16cy failed with info={info}"

    P = S @ S.T
    Q = R.T @ R

    res_P = Acl_bf @ P + P @ Acl_bf.T + scalec**2 * (B @ B.T)
    res_Q = Acl_gc.T @ Q + Q @ Acl_gc + scaleo**2 * (F.T @ F)

    assert_allclose(res_P, np.zeros((n, n)), atol=1e-12)
    assert_allclose(res_Q, np.zeros((n, n)), atol=1e-12)


def test_sb16cy_continuous_right():
    """
    Test continuous-time right coprime factorization.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n, m, p = 4, 2, 2

    A = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')

    B = np.array([
        [1.0, 0.5],
        [0.2, 1.0],
        [0.5, 0.3],
        [0.1, 0.4]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.5, 0.2, 0.1],
        [0.3, 1.0, 0.4, 0.2]
    ], order='F', dtype=float)

    F = np.array([
        [-0.3, 0.1, 0.0, 0.0],
        [0.0, -0.2, 0.1, 0.0]
    ], order='F', dtype=float)

    G = np.array([
        [0.2, -0.1],
        [-0.1, 0.2],
        [0.0, 0.1],
        [0.1, 0.0]
    ], order='F', dtype=float)

    Acl_bf = A + B @ F
    Acl_gc = A + G @ C
    eig_bf = np.linalg.eigvals(Acl_bf)
    eig_gc = np.linalg.eigvals(Acl_gc)
    assert all(eig_bf.real < 0), f"A+B*F not stable: {eig_bf}"
    assert all(eig_gc.real < 0), f"A+G*C not stable: {eig_gc}"

    S, R, scalec, scaleo, info = sb16cy('C', 'R', n, m, p, A, B, C, F, G)

    assert info == 0, f"sb16cy failed with info={info}"

    P = S @ S.T
    Q = R.T @ R

    res_P = Acl_bf @ P + P @ Acl_bf.T + scalec**2 * (G @ G.T)
    res_Q = Acl_gc.T @ Q + Q @ Acl_gc + scaleo**2 * (C.T @ C)

    assert_allclose(res_P, np.zeros((n, n)), atol=1e-12)
    assert_allclose(res_Q, np.zeros((n, n)), atol=1e-12)


def test_sb16cy_discrete_left():
    """
    Test discrete-time left coprime factorization.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n, m, p = 3, 2, 2

    A = np.array([
        [0.5, 0.1, 0.0],
        [0.0, 0.4, 0.1],
        [0.0, 0.0, 0.3]
    ], order='F', dtype=float)

    B = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    F = np.array([
        [-0.1, 0.0, 0.05],
        [0.0, -0.15, 0.0]
    ], order='F', dtype=float)

    G = np.array([
        [0.1, 0.0],
        [0.0, 0.1],
        [0.05, 0.05]
    ], order='F', dtype=float)

    Acl_bf = A + B @ F
    Acl_gc = A + G @ C
    eig_bf = np.linalg.eigvals(Acl_bf)
    eig_gc = np.linalg.eigvals(Acl_gc)
    assert all(np.abs(eig_bf) < 1), f"A+B*F not Schur stable: {eig_bf}"
    assert all(np.abs(eig_gc) < 1), f"A+G*C not Schur stable: {eig_gc}"

    S, R, scalec, scaleo, info = sb16cy('D', 'L', n, m, p, A, B, C, F, G)

    assert info == 0, f"sb16cy failed with info={info}"

    P = S @ S.T
    Q = R.T @ R

    res_P = Acl_bf @ P @ Acl_bf.T - P + scalec**2 * (B @ B.T)
    res_Q = Acl_gc.T @ Q @ Acl_gc - Q + scaleo**2 * (F.T @ F)

    assert_allclose(res_P, np.zeros((n, n)), atol=1e-12)
    assert_allclose(res_Q, np.zeros((n, n)), atol=1e-12)


def test_sb16cy_discrete_right():
    """
    Test discrete-time right coprime factorization.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n, m, p = 4, 2, 3

    A = np.diag([0.3, 0.4, 0.5, 0.6]).astype(float, order='F')

    B = np.array([
        [0.5, 0.0],
        [0.0, 0.5],
        [0.2, 0.2],
        [0.1, 0.3]
    ], order='F', dtype=float)

    C = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    F = np.array([
        [-0.05, 0.0, 0.0, 0.0],
        [0.0, -0.08, 0.0, 0.0]
    ], order='F', dtype=float)

    G = np.array([
        [0.1, 0.0, 0.05],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        [0.05, 0.05, 0.0]
    ], order='F', dtype=float)

    Acl_bf = A + B @ F
    Acl_gc = A + G @ C
    eig_bf = np.linalg.eigvals(Acl_bf)
    eig_gc = np.linalg.eigvals(Acl_gc)
    assert all(np.abs(eig_bf) < 1), f"A+B*F not Schur stable: {eig_bf}"
    assert all(np.abs(eig_gc) < 1), f"A+G*C not Schur stable: {eig_gc}"

    S, R, scalec, scaleo, info = sb16cy('D', 'R', n, m, p, A, B, C, F, G)

    assert info == 0, f"sb16cy failed with info={info}"

    P = S @ S.T
    Q = R.T @ R

    res_P = Acl_bf @ P @ Acl_bf.T - P + scalec**2 * (G @ G.T)
    res_Q = Acl_gc.T @ Q @ Acl_gc - Q + scaleo**2 * (C.T @ C)

    assert_allclose(res_P, np.zeros((n, n)), atol=1e-12)
    assert_allclose(res_Q, np.zeros((n, n)), atol=1e-12)


def test_sb16cy_grammian_symmetry():
    """
    Verify that Grammians P=S*S' and Q=R'*R are symmetric positive semi-definite.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    n, m, p = 3, 2, 2

    A = np.diag([-1.5, -2.5, -3.5]).astype(float, order='F')
    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], order='F', dtype=float)
    C = np.array([
        [1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0]
    ], order='F', dtype=float)
    F = np.array([
        [-0.2, 0.1, 0.0],
        [0.0, -0.2, 0.1]
    ], order='F', dtype=float)
    G = np.array([
        [0.3, -0.1],
        [-0.1, 0.3],
        [0.0, 0.0]
    ], order='F', dtype=float)

    S, R, scalec, scaleo, info = sb16cy('C', 'L', n, m, p, A, B, C, F, G)
    assert info == 0

    P = S @ S.T
    Q = R.T @ R

    assert_allclose(P, P.T, rtol=1e-14)
    assert_allclose(Q, Q.T, rtol=1e-14)

    eigP = np.linalg.eigvalsh(P)
    eigQ = np.linalg.eigvalsh(Q)
    assert all(eigP >= -1e-14), f"P not PSD: {eigP}"
    assert all(eigQ >= -1e-14), f"Q not PSD: {eigQ}"


def test_sb16cy_quick_return():
    """
    Test quick return for n=0 case.
    """
    n, m, p = 0, 2, 2
    A = np.array([], dtype=float, order='F').reshape(0, 0)
    B = np.array([], dtype=float, order='F').reshape(0, 2)
    C = np.array([], dtype=float, order='F').reshape(2, 0)
    F = np.array([], dtype=float, order='F').reshape(2, 0)
    G = np.array([], dtype=float, order='F').reshape(0, 2)

    S, R, scalec, scaleo, info = sb16cy('C', 'L', n, m, p, A, B, C, F, G)

    assert info == 0
    assert scalec == 1.0
    assert scaleo == 1.0


def test_sb16cy_invalid_dico():
    """
    Test error handling for invalid DICO parameter.
    """
    n, m, p = 2, 1, 1
    A = np.eye(n, order='F', dtype=float)
    B = np.ones((n, m), order='F', dtype=float)
    C = np.ones((p, n), order='F', dtype=float)
    F = np.ones((m, n), order='F', dtype=float)
    G = np.ones((n, p), order='F', dtype=float)

    S, R, scalec, scaleo, info = sb16cy('X', 'L', n, m, p, A, B, C, F, G)

    assert info == -1


def test_sb16cy_invalid_jobcf():
    """
    Test error handling for invalid JOBCF parameter.
    """
    n, m, p = 2, 1, 1
    A = np.diag([-1.0, -2.0]).astype(float, order='F')
    B = np.ones((n, m), order='F', dtype=float)
    C = np.ones((p, n), order='F', dtype=float)
    F = np.ones((m, n), order='F', dtype=float)
    G = np.ones((n, p), order='F', dtype=float)

    S, R, scalec, scaleo, info = sb16cy('C', 'X', n, m, p, A, B, C, F, G)

    assert info == -2
