"""
Tests for IB01QD - Estimate initial state and system matrices B, D.

Given (A, C) and input/output trajectories, estimate B, D, and x0 for:
    x(k+1) = A*x(k) + B*u(k)
    y(k)   = C*x(k) + D*u(k)

Matrix A is assumed to be in real Schur form.
"""

import numpy as np
import pytest
from ctrlsys import ib01qd


def create_schur_system(n, m, l, seed=42):
    """
    Create a stable discrete-time system with A in real Schur form.

    Random seed: specified for reproducibility
    """
    np.random.seed(seed)

    A = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        A[i, i] = 0.3 + 0.4 * np.random.rand()
        if i > 0:
            A[i-1, i] = 0.1 * np.random.randn()

    B = np.random.randn(n, m).astype(float, order='F')
    C = np.random.randn(l, n).astype(float, order='F')
    D = np.random.randn(l, m).astype(float, order='F')
    x0 = np.random.randn(n).astype(float, order='F')

    return A, B, C, D, x0


def simulate_system(A, B, C, D, u, x0=None):
    """
    Simulate discrete-time state-space system.

    x(k+1) = A*x(k) + B*u(k)
    y(k)   = C*x(k) + D*u(k)
    """
    nsmp, m = u.shape
    n = A.shape[0]
    l = C.shape[0]

    if x0 is None:
        x0 = np.zeros(n)

    y = np.zeros((nsmp, l), order='F', dtype=float)
    x = x0.copy()

    for k in range(nsmp):
        y[k, :] = C @ x + D @ u[k, :]
        x = A @ x + B @ u[k, :]

    return y


def test_ib01qd_basic():
    """
    Basic test: estimate B and D from input/output data.

    Tests JOBX0='X' (compute initial state), JOB='D' (compute B and D).
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, l = 2, 1, 1
    A, B, C, D, x0_true = create_schur_system(n, m, l, seed=42)

    nsmp = n * m + n + m + 10
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'X'
    job = 'D'
    tol = 0.0

    x0_est, B_est, D_est, rcond_w2, rcond_u, iwarn, info = ib01qd(
        jobx0, job, n, m, l, A, C, u, y, tol
    )

    assert info == 0
    np.testing.assert_allclose(B_est, B, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(D_est, D, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-6, atol=1e-10)


def test_ib01qd_b_only():
    """
    Test computing B only (D is known to be zero).

    Tests JOBX0='X' (compute initial state), JOB='B' (compute B only).
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, l = 3, 2, 1
    A, B, C, D_unused, x0_true = create_schur_system(n, m, l, seed=123)
    D = np.zeros((l, m), order='F', dtype=float)

    nsmp = n * m + n + 5
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'X'
    job = 'B'
    tol = 0.0

    x0_est, B_est, D_est, rcond_w2, rcond_u, iwarn, info = ib01qd(
        jobx0, job, n, m, l, A, C, u, y, tol
    )

    assert info == 0
    np.testing.assert_allclose(B_est, B, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-6, atol=1e-10)


def test_ib01qd_no_initial_state():
    """
    Test with known zero initial state.

    Tests JOBX0='N' (x0 is known to be zero), JOB='D'.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, l = 2, 1, 2
    A, B, C, D, x0_unused = create_schur_system(n, m, l, seed=456)
    x0_true = np.zeros(n)

    nsmp = n * m + m + 2
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'N'
    job = 'D'
    tol = 0.0

    x0_est, B_est, D_est, rcond_w2, rcond_u, iwarn, info = ib01qd(
        jobx0, job, n, m, l, A, C, u, y, tol
    )

    assert info == 0
    np.testing.assert_allclose(B_est, B, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(D_est, D, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(x0_est, np.zeros(n), rtol=1e-14)


def test_ib01qd_state_equation_property():
    """
    Mathematical property: estimated B satisfies state equation.

    Tests: x(k+1) = A*x(k) + B*u(k) holds for estimated B.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, l = 3, 1, 2
    A, B_true, C, D, x0_true = create_schur_system(n, m, l, seed=789)

    nsmp = n * m + n + m + 20
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B_true, C, D, u, x0_true)

    # Save originals - ib01qd modifies u in-place
    u_orig = u.copy()
    y_orig = y.copy()

    jobx0 = 'X'
    job = 'D'
    tol = 0.0

    x0_est, B_est, D_est, rcond_w2, rcond_u, iwarn, info = ib01qd(
        jobx0, job, n, m, l, A, C, u, y, tol
    )

    assert info == 0

    y_reconstructed = simulate_system(A, B_est, C, D_est, u_orig, x0_est)
    np.testing.assert_allclose(y_reconstructed, y_orig, rtol=1e-10, atol=1e-12)


def test_ib01qd_output_equation_property():
    """
    Mathematical property: y(k) = C*x(k) + D*u(k) holds for estimated D.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m, l = 2, 2, 2
    A, B_true, C, D_true, x0_true = create_schur_system(n, m, l, seed=888)

    nsmp = n * m + n + m + 15
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B_true, C, D_true, u, x0_true)

    # Save originals - ib01qd modifies u in-place
    u_orig = u.copy()
    y_orig = y.copy()

    jobx0 = 'X'
    job = 'D'
    tol = 0.0

    x0_est, B_est, D_est, rcond_w2, rcond_u, iwarn, info = ib01qd(
        jobx0, job, n, m, l, A, C, u, y, tol
    )

    assert info == 0

    x = x0_est.copy()
    for k in range(nsmp):
        y_k_expected = C @ x + D_est @ u_orig[k, :]
        np.testing.assert_allclose(y_orig[k, :], y_k_expected, rtol=1e-6, atol=1e-10)
        x = A @ x + B_est @ u_orig[k, :]


def test_ib01qd_larger_system():
    """
    Test with larger system dimensions.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, l = 5, 2, 3
    A, B, C, D, x0_true = create_schur_system(n, m, l, seed=111)

    nsmp = n * m + n + m + 20
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'X'
    job = 'D'
    tol = 0.0

    x0_est, B_est, D_est, rcond_w2, rcond_u, iwarn, info = ib01qd(
        jobx0, job, n, m, l, A, C, u, y, tol
    )

    assert info == 0
    np.testing.assert_allclose(B_est, B, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(D_est, D, rtol=1e-5, atol=1e-8)


def test_ib01qd_n_zero():
    """
    Edge case: N = 0 (no states).

    Should return quickly with minimal outputs.
    """
    n, m, l = 0, 2, 1
    nsmp = m + 2

    A = np.zeros((0, 0), order='F', dtype=float)
    C = np.zeros((l, 0), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    jobx0 = 'X'
    job = 'D'
    tol = 0.0

    x0_est, B_est, D_est, rcond_w2, rcond_u, iwarn, info = ib01qd(
        jobx0, job, n, m, l, A, C, u, y, tol
    )

    assert info == 0


def test_ib01qd_m_zero():
    """
    Edge case: M = 0 (no inputs).

    System reduces to: x(k+1) = A*x(k), y(k) = C*x(k)
    """
    np.random.seed(222)
    n, m, l = 2, 0, 1
    nsmp = n + 5

    A = np.array([[0.5, 0.1], [0.0, 0.4]], order='F', dtype=float)
    C = np.array([[1.0, 0.5]], order='F', dtype=float)
    x0_true = np.array([1.0, 0.5])

    u = np.zeros((nsmp, 0), order='F', dtype=float)

    y = np.zeros((nsmp, l), order='F', dtype=float)
    x = x0_true.copy()
    for k in range(nsmp):
        y[k, :] = C @ x
        x = A @ x

    jobx0 = 'X'
    job = 'B'
    tol = 0.0

    x0_est, B_est, D_est, rcond_w2, rcond_u, iwarn, info = ib01qd(
        jobx0, job, n, m, l, A, C, u, y, tol
    )

    assert info == 0
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-10, atol=1e-12)


def test_ib01qd_error_invalid_jobx0():
    """
    Error handling: invalid JOBX0 parameter.
    """
    n, m, l = 2, 1, 1
    nsmp = 10
    A = np.eye(n, order='F', dtype=float) * 0.5
    C = np.ones((l, n), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="JOBX0"):
        ib01qd('Z', 'D', n, m, l, A, C, u, y, 0.0)


def test_ib01qd_error_invalid_job():
    """
    Error handling: invalid JOB parameter.
    """
    n, m, l = 2, 1, 1
    nsmp = 10
    A = np.eye(n, order='F', dtype=float) * 0.5
    C = np.ones((l, n), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="JOB"):
        ib01qd('X', 'Z', n, m, l, A, C, u, y, 0.0)


def test_ib01qd_error_n_negative():
    """
    Error handling: N < 0.
    """
    n, m, l = -1, 1, 1
    nsmp = 10
    A = np.zeros((0, 0), order='F', dtype=float)
    C = np.zeros((l, 0), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="N"):
        ib01qd('X', 'D', n, m, l, A, C, u, y, 0.0)


def test_ib01qd_error_l_nonpositive():
    """
    Error handling: L <= 0.
    """
    n, m, l = 2, 1, 0
    nsmp = 10
    A = np.eye(n, order='F', dtype=float) * 0.5
    C = np.zeros((0, n), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, 0).astype(float, order='F')

    with pytest.raises(ValueError, match="L"):
        ib01qd('X', 'D', n, m, l, A, C, u, y, 0.0)


def test_ib01qd_error_nsmp_too_small():
    """
    Error handling: NSMP too small for the problem.
    """
    n, m, l = 3, 2, 1
    nsmp = n * m  # Too small - needs at least n*m + n + m for 'D'
    A = np.eye(n, order='F', dtype=float) * 0.5
    C = np.ones((l, n), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="NSMP"):
        ib01qd('X', 'D', n, m, l, A, C, u, y, 0.0)


def test_ib01qd_rcond_positive():
    """
    Test that reciprocal condition numbers are returned positive.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, l = 2, 1, 1
    A, B, C, D, x0_true = create_schur_system(n, m, l, seed=333)

    nsmp = n * m + n + m + 10
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'X'
    job = 'D'
    tol = 0.0

    x0_est, B_est, D_est, rcond_w2, rcond_u, iwarn, info = ib01qd(
        jobx0, job, n, m, l, A, C, u, y, tol
    )

    assert info == 0
    assert rcond_w2 > 0
    assert rcond_u > 0


def test_ib01qd_tol_effect():
    """
    Test tolerance parameter effect on estimation.

    When TOL <= 0, machine epsilon is used; when TOL > 0, the given value
    is used as a lower bound for the reciprocal condition number.

    This test verifies:
    1. tol=0 (use machine epsilon) produces correct results
    2. tol=-1 (also uses machine epsilon) produces identical results to tol=0
    3. Both tolerance variants produce valid condition number estimates

    Note: ib01qd modifies u in-place, so fresh copies must be used for each call.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n, m, l = 2, 1, 1
    A, B_true, C, D_true, x0_true = create_schur_system(n, m, l, seed=444)

    nsmp = n * m + n + m + 15
    u_orig = np.random.randn(nsmp, m).astype(float, order='F')
    y_orig = simulate_system(A, B_true, C, D_true, u_orig, x0_true)

    jobx0 = 'X'
    job = 'D'

    x0_1, B_1, D_1, rcond_1, rcond_u_1, iwarn_1, info_1 = ib01qd(
        jobx0, job, n, m, l, A.copy(order='F'), C.copy(order='F'),
        u_orig.copy(order='F'), y_orig.copy(order='F'), 0.0
    )
    x0_2, B_2, D_2, rcond_2, rcond_u_2, iwarn_2, info_2 = ib01qd(
        jobx0, job, n, m, l, A.copy(order='F'), C.copy(order='F'),
        u_orig.copy(order='F'), y_orig.copy(order='F'), -1.0
    )

    assert info_1 == 0
    assert info_2 == 0

    np.testing.assert_allclose(B_1, B_true, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(D_1, D_true, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(x0_1, x0_true, rtol=1e-6, atol=1e-10)

    np.testing.assert_allclose(B_1, B_2, rtol=1e-12)
    np.testing.assert_allclose(D_1, D_2, rtol=1e-12)
    np.testing.assert_allclose(x0_1, x0_2, rtol=1e-12)

    assert rcond_1 > 0
    assert rcond_2 > 0
    assert rcond_u_1 > 0
    assert rcond_u_2 > 0
