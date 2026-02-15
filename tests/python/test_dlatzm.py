import numpy as np
import pytest


def test_dlatzm_left_side():
    """
    Test DLATZM with SIDE='L' (left multiplication P*C).

    Validates that P = I - tau*u*u^T applied from left.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    m, n = 4, 3
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 0.5
    incv = 1

    c1 = np.random.randn(1, n).astype(float, order='F')
    c2 = np.random.randn(m - 1, n).astype(float, order='F')

    c1_orig = c1.copy()
    c2_orig = c2.copy()

    u = np.vstack([np.ones((1, 1)), v.reshape(-1, 1)])
    P = np.eye(m) - tau * u @ u.T

    C_orig = np.vstack([c1_orig, c2_orig])
    C_expected = P @ C_orig

    from ctrlsys import dlatzm

    c1_out, c2_out = dlatzm('L', m, n, v, incv, tau, c1, c2)

    C_result = np.vstack([c1_out, c2_out])
    np.testing.assert_allclose(C_result, C_expected, rtol=1e-14, atol=1e-15)


def test_dlatzm_right_side():
    """
    Test DLATZM with SIDE='R' (right multiplication C*P).

    Validates that P = I - tau*u*u^T applied from right.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    m, n = 3, 5
    v = np.random.randn(n - 1).astype(float, order='F')
    tau = 0.7
    incv = 1

    c1 = np.random.randn(m, 1).astype(float, order='F')
    c2 = np.random.randn(m, n - 1).astype(float, order='F')

    c1_orig = c1.copy()
    c2_orig = c2.copy()

    u = np.vstack([np.ones((1, 1)), v.reshape(-1, 1)])
    P = np.eye(n) - tau * u @ u.T

    C_orig = np.hstack([c1_orig, c2_orig])
    C_expected = C_orig @ P

    from ctrlsys import dlatzm

    c1_out, c2_out = dlatzm('R', m, n, v, incv, tau, c1, c2)

    C_result = np.hstack([c1_out, c2_out])
    np.testing.assert_allclose(C_result, C_expected, rtol=1e-14, atol=1e-15)


def test_dlatzm_tau_zero():
    """
    Test DLATZM with tau=0 (identity transformation).

    When tau=0, P=I and matrices should be unchanged.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    m, n = 4, 3
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 0.0
    incv = 1

    c1 = np.random.randn(1, n).astype(float, order='F')
    c2 = np.random.randn(m - 1, n).astype(float, order='F')

    c1_orig = c1.copy()
    c2_orig = c2.copy()

    from ctrlsys import dlatzm

    c1_out, c2_out = dlatzm('L', m, n, v, incv, tau, c1, c2)

    np.testing.assert_allclose(c1_out, c1_orig, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(c2_out, c2_orig, rtol=1e-14, atol=1e-15)


def test_dlatzm_orthogonal_property():
    """
    Test DLATZM orthogonality: P^T * P = I.

    The Householder matrix P = I - tau*u*u^T is orthogonal when
    tau = 2/(u^T*u), which is the standard Householder scalar.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    m, n = 5, 5
    v = np.random.randn(m - 1).astype(float, order='F')
    u = np.vstack([np.ones((1, 1)), v.reshape(-1, 1)])
    tau = 2.0 / (u.T @ u)[0, 0]
    incv = 1

    c1 = np.eye(m)[:1, :].astype(float, order='F').copy()
    c2 = np.eye(m)[1:, :].astype(float, order='F').copy()

    from ctrlsys import dlatzm

    c1_out, c2_out = dlatzm('L', m, n, v, incv, tau, c1, c2)
    P = np.vstack([c1_out, c2_out])

    np.testing.assert_allclose(P.T @ P, np.eye(m), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(P @ P.T, np.eye(m), rtol=1e-14, atol=1e-15)


def test_dlatzm_involution():
    """
    Test DLATZM involution: P*P = I (for Householder tau=2/(u^T*u)).

    Standard Householder reflector is an involution.
    Random seed: 101 (for reproducibility)
    """
    np.random.seed(101)

    m, n = 4, 3
    v = np.random.randn(m - 1).astype(float, order='F')
    u = np.vstack([np.ones((1, 1)), v.reshape(-1, 1)])
    tau = 2.0 / (u.T @ u)[0, 0]
    incv = 1

    c1 = np.random.randn(1, n).astype(float, order='F')
    c2 = np.random.randn(m - 1, n).astype(float, order='F')

    c1_orig = c1.copy()
    c2_orig = c2.copy()

    from ctrlsys import dlatzm

    c1_out, c2_out = dlatzm('L', m, n, v, incv, tau, c1, c2)
    c1_out2, c2_out2 = dlatzm('L', m, n, v, incv, tau, c1_out, c2_out)

    np.testing.assert_allclose(c1_out2, c1_orig, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(c2_out2, c2_orig, rtol=1e-14, atol=1e-15)


def test_dlatzm_incv_not_one():
    """
    Test DLATZM with non-unit increment (incv != 1).

    Random seed: 202 (for reproducibility)
    """
    np.random.seed(202)

    m, n = 4, 3
    incv = 2
    v_full = np.random.randn((m - 1) * incv).astype(float, order='F')
    v_extracted = v_full[::incv]
    tau = 0.6

    c1 = np.random.randn(1, n).astype(float, order='F')
    c2 = np.random.randn(m - 1, n).astype(float, order='F')

    u = np.vstack([np.ones((1, 1)), v_extracted.reshape(-1, 1)])
    P = np.eye(m) - tau * u @ u.T

    C_orig = np.vstack([c1.copy(), c2.copy()])
    C_expected = P @ C_orig

    from ctrlsys import dlatzm

    c1_out, c2_out = dlatzm('L', m, n, v_full, incv, tau, c1, c2)

    C_result = np.vstack([c1_out, c2_out])
    np.testing.assert_allclose(C_result, C_expected, rtol=1e-14, atol=1e-15)
