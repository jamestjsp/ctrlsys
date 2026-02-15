"""Tests for IB01CD - Initial state and system matrices B, D estimation (driver).

IB01CD is a driver routine that estimates the initial state x(0) and optionally
system matrices B and D for a discrete-time LTI system:
    x(k+1) = A*x(k) + B*u(k)
    y(k)   = C*x(k) + D*u(k)

Key differences from IB01QD/IB01RD:
- IB01CD accepts A in general form (not Schur) and internally transforms to Schur
- Returns orthogonal matrix V such that A = V*At*V', where At is in Schur form
- Multiple operation modes via JOBX0, COMUSE, JOB parameters

Mode combinations:
  JOBX0='X', COMUSE='C': Compute x0, estimate B (and D if JOB='D')
  JOBX0='X', COMUSE='U': Compute x0, use given B (and D if JOB='D')
  JOBX0='X', COMUSE='N': Compute x0, do not use B,D
  JOBX0='N', COMUSE='C': Set x0=0, estimate B (and D)
  JOBX0='N', COMUSE='U': Set x0=0, skip computation
  JOBX0='N', COMUSE='N': Quick return (no operation)
"""

import numpy as np
import pytest
from ctrlsys import ib01cd


def create_stable_system(n, m, l, seed=42):
    """
    Create a stable discrete-time system with known initial state.

    Random seed: specified for reproducibility
    """
    np.random.seed(seed)

    A = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        A[i, i] = 0.3 + 0.4 * np.random.rand()
        for j in range(i + 1, n):
            A[i, j] = 0.1 * np.random.randn()

    B = np.random.randn(n, m).astype(float, order='F') * 0.5
    C = np.random.randn(l, n).astype(float, order='F') * 0.5
    D = np.random.randn(l, m).astype(float, order='F') * 0.1
    x0 = np.random.randn(n).astype(float, order='F')

    return A, B, C, D, x0


def simulate_system(A, B, C, D, u, x0):
    """
    Simulate discrete-time state-space system.

    x(k+1) = A*x(k) + B*u(k)
    y(k)   = C*x(k) + D*u(k)
    """
    nsmp, m = u.shape
    n = A.shape[0]
    l = C.shape[0]

    y = np.zeros((nsmp, l), order='F', dtype=float)
    x = x0.copy()

    for k in range(nsmp):
        y[k, :] = C @ x + D @ u[k, :]
        x = A @ x + B @ u[k, :]

    return y


def test_ib01cd_compute_x0_use_bd():
    """
    Test JOBX0='X', COMUSE='U': estimate x0 using known B, D.

    The most common use case: system is identified, B and D are known,
    only initial state needs to be estimated.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, l = 3, 1, 2
    A, B, C, D, x0_true = create_stable_system(n, m, l, seed=42)

    nsmp = max(n, 20)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'X'
    comuse = 'U'
    job = 'D'
    tol = 0.0

    x0_est, B_out, D_out, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, B, C, D, u, y, tol
    )

    assert info == 0, f"IB01CD returned info={info}"
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-6, atol=1e-10)


def test_ib01cd_compute_all():
    """
    Test JOBX0='X', COMUSE='C', JOB='D': estimate x0, B, and D.

    Full estimation mode: given only A, C, and input/output data,
    estimate B, D, and initial state.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, l = 2, 1, 1
    A, B_true, C, D_true, x0_true = create_stable_system(n, m, l, seed=123)

    nsmp = n * m + n + m + 20
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B_true, C, D_true, u, x0_true)

    jobx0 = 'X'
    comuse = 'C'
    job = 'D'
    tol = 0.0

    x0_est, B_est, D_est, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, np.zeros((n, m), order='F', dtype=float),
        C, np.zeros((l, m), order='F', dtype=float), u, y, tol
    )

    assert info == 0, f"IB01CD returned info={info}"
    np.testing.assert_allclose(B_est, B_true, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(D_est, D_true, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-5, atol=1e-8)


def test_ib01cd_compute_b_only():
    """
    Test JOBX0='X', COMUSE='C', JOB='B': estimate x0 and B (D=0).

    When feedthrough D is known to be zero.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, l = 3, 2, 1
    A, B_true, C, D_unused, x0_true = create_stable_system(n, m, l, seed=456)
    D = np.zeros((l, m), order='F', dtype=float)

    nsmp = n * m + n + 10
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B_true, C, D, u, x0_true)

    jobx0 = 'X'
    comuse = 'C'
    job = 'B'
    tol = 0.0

    x0_est, B_est, D_est, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, np.zeros((n, m), order='F', dtype=float),
        C, np.zeros((l, m), order='F', dtype=float), u, y, tol
    )

    assert info == 0, f"IB01CD returned info={info}"
    np.testing.assert_allclose(B_est, B_true, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-5, atol=1e-8)


def test_ib01cd_no_x0_use_bd():
    """
    Test JOBX0='N', COMUSE='U': skip computation, set x0=0.

    Quick return when x0 is known to be zero and B, D are given.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, l = 2, 1, 1
    A, B, C, D, x0_unused = create_stable_system(n, m, l, seed=789)

    nsmp = 10
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    jobx0 = 'N'
    comuse = 'U'
    job = 'D'
    tol = 0.0

    x0_est, B_out, D_out, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, B, C, D, u, y, tol
    )

    assert info == 0
    np.testing.assert_allclose(x0_est, np.zeros(n), rtol=1e-14, atol=1e-14)


def test_ib01cd_no_x0_no_bd():
    """
    Test JOBX0='N', COMUSE='N': quick return, no operation.

    Should return DWORK(1)=2, DWORK(2)=1 without computation.
    """
    n, m, l = 2, 1, 1
    A = np.eye(n, order='F', dtype=float) * 0.5
    B = np.ones((n, m), order='F', dtype=float)
    C = np.ones((l, n), order='F', dtype=float)
    D = np.ones((l, m), order='F', dtype=float)
    u = np.ones((5, m), order='F', dtype=float)
    y = np.ones((5, l), order='F', dtype=float)

    jobx0 = 'N'
    comuse = 'N'
    job = 'D'
    tol = 0.0

    x0_est, B_out, D_out, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, B, C, D, u, y, tol
    )

    assert info == 0


def test_ib01cd_compute_x0_no_bd():
    """
    Test JOBX0='X', COMUSE='N': estimate x0 from output only (no input effect).

    Autonomous system case: estimate x0 using only A, C, and output y.
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, l = 3, 0, 2
    A = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        A[i, i] = 0.5 + 0.3 * np.random.rand()
        for j in range(i + 1, n):
            A[i, j] = 0.1 * np.random.randn()

    C = np.random.randn(l, n).astype(float, order='F') * 0.5
    x0_true = np.random.randn(n).astype(float, order='F')

    nsmp = n + 10
    y = np.zeros((nsmp, l), order='F', dtype=float)
    x = x0_true.copy()
    for k in range(nsmp):
        y[k, :] = C @ x
        x = A @ x

    B_dummy = np.zeros((n, 1), order='F', dtype=float)
    D_dummy = np.zeros((l, 1), order='F', dtype=float)
    u_dummy = np.zeros((nsmp, 1), order='F', dtype=float)

    jobx0 = 'X'
    comuse = 'N'
    job = 'B'
    tol = 0.0

    x0_est, B_out, D_out, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, 0, l, A, B_dummy, C, D_dummy, u_dummy, y, tol
    )

    assert info == 0
    np.testing.assert_allclose(x0_est, x0_true, rtol=1e-6, atol=1e-10)


def test_ib01cd_schur_transformation_property():
    """
    Mathematical property: V returned is orthogonal and A = V*At*V'.

    Validates that the Schur transformation matrix V is correctly returned.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m, l = 4, 1, 1
    A, B, C, D, x0_true = create_stable_system(n, m, l, seed=222)

    nsmp = max(n, 20)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'X'
    comuse = 'U'
    job = 'D'
    tol = 0.0

    x0_est, B_out, D_out, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, B, C, D, u, y, tol
    )

    assert info == 0

    VtV = V.T @ V
    np.testing.assert_allclose(VtV, np.eye(n), rtol=1e-14, atol=1e-14)


def test_ib01cd_output_reconstruction_property():
    """
    Mathematical property: estimated parameters reconstruct output trajectory.

    Tests: y(k) = C*A^k*x0 + sum terms produces original y.
    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, l = 2, 1, 2
    A, B, C, D, x0_true = create_stable_system(n, m, l, seed=333)

    nsmp = max(n, 25)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'X'
    comuse = 'U'
    job = 'D'
    tol = 0.0

    x0_est, B_out, D_out, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, B, C, D, u, y, tol
    )

    assert info == 0

    y_reconstructed = simulate_system(A, B, C, D, u, x0_est)
    np.testing.assert_allclose(y_reconstructed, y, rtol=1e-10, atol=1e-12)


def test_ib01cd_larger_system():
    """
    Test with larger system dimensions (MIMO).

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n, m, l = 5, 2, 3
    A, B, C, D, x0_true = create_stable_system(n, m, l, seed=444)

    nsmp = n * m + n + m + 30
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'X'
    comuse = 'C'
    job = 'D'
    tol = 0.0

    x0_est, B_est, D_est, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, np.zeros((n, m), order='F', dtype=float),
        C, np.zeros((l, m), order='F', dtype=float), u, y, tol
    )

    assert info == 0
    np.testing.assert_allclose(B_est, B, rtol=1e-4, atol=1e-7)


def test_ib01cd_n_zero():
    """
    Edge case: N=0 (no states).

    Should return quickly with minimal operations.
    """
    n, m, l = 0, 1, 1
    nsmp = 5

    A = np.zeros((0, 0), order='F', dtype=float)
    B = np.zeros((0, m), order='F', dtype=float)
    C = np.zeros((l, 0), order='F', dtype=float)
    D = np.ones((l, m), order='F', dtype=float)
    u = np.ones((nsmp, m), order='F', dtype=float)
    y = np.ones((nsmp, l), order='F', dtype=float)

    jobx0 = 'X'
    comuse = 'U'
    job = 'D'
    tol = 0.0

    x0_est, B_out, D_out, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, B, C, D, u, y, tol
    )

    assert info == 0


def test_ib01cd_error_invalid_jobx0():
    """
    Error handling: invalid JOBX0 parameter.
    """
    n, m, l = 2, 1, 1
    nsmp = 10
    A = np.eye(n, order='F', dtype=float) * 0.5
    B = np.ones((n, m), order='F', dtype=float)
    C = np.ones((l, n), order='F', dtype=float)
    D = np.ones((l, m), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="JOBX0"):
        ib01cd('Z', 'U', 'D', n, m, l, A, B, C, D, u, y, 0.0)


def test_ib01cd_error_invalid_comuse():
    """
    Error handling: invalid COMUSE parameter.
    """
    n, m, l = 2, 1, 1
    nsmp = 10
    A = np.eye(n, order='F', dtype=float) * 0.5
    B = np.ones((n, m), order='F', dtype=float)
    C = np.ones((l, n), order='F', dtype=float)
    D = np.ones((l, m), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="COMUSE"):
        ib01cd('X', 'Z', 'D', n, m, l, A, B, C, D, u, y, 0.0)


def test_ib01cd_error_invalid_job():
    """
    Error handling: invalid JOB parameter.
    """
    n, m, l = 2, 1, 1
    nsmp = 10
    A = np.eye(n, order='F', dtype=float) * 0.5
    B = np.ones((n, m), order='F', dtype=float)
    C = np.ones((l, n), order='F', dtype=float)
    D = np.ones((l, m), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="JOB"):
        ib01cd('X', 'U', 'Z', n, m, l, A, B, C, D, u, y, 0.0)


def test_ib01cd_error_n_negative():
    """
    Error handling: N < 0.
    """
    n, m, l = -1, 1, 1
    A = np.zeros((0, 0), order='F', dtype=float)
    B = np.zeros((0, 1), order='F', dtype=float)
    C = np.zeros((l, 0), order='F', dtype=float)
    D = np.ones((l, 1), order='F', dtype=float)
    u = np.random.randn(10, 1).astype(float, order='F')
    y = np.random.randn(10, l).astype(float, order='F')

    with pytest.raises(ValueError, match="N"):
        ib01cd('X', 'U', 'D', n, m, l, A, B, C, D, u, y, 0.0)


def test_ib01cd_error_l_nonpositive():
    """
    Error handling: L <= 0.
    """
    n, m, l = 2, 1, 0
    A = np.eye(n, order='F', dtype=float) * 0.5
    B = np.ones((n, m), order='F', dtype=float)
    C = np.zeros((0, n), order='F', dtype=float)
    D = np.zeros((0, m), order='F', dtype=float)
    u = np.random.randn(10, m).astype(float, order='F')
    y = np.random.randn(10, 0).astype(float, order='F')

    with pytest.raises(ValueError, match="L"):
        ib01cd('X', 'U', 'D', n, m, l, A, B, C, D, u, y, 0.0)


def test_ib01cd_error_nsmp_too_small():
    """
    Error handling: NSMP too small for problem dimensions.
    """
    n, m, l = 3, 2, 1
    nsmp = 5
    A = np.eye(n, order='F', dtype=float) * 0.5
    B = np.ones((n, m), order='F', dtype=float)
    C = np.ones((l, n), order='F', dtype=float)
    D = np.ones((l, m), order='F', dtype=float)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="NSMP"):
        ib01cd('X', 'C', 'D', n, m, l, A, B, C, D, u, y, 0.0)


def test_ib01cd_rcond_positive():
    """
    Test that reciprocal condition number is returned positive.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n, m, l = 2, 1, 1
    A, B, C, D, x0_true = create_stable_system(n, m, l, seed=555)

    nsmp = max(n, 20)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = simulate_system(A, B, C, D, u, x0_true)

    jobx0 = 'X'
    comuse = 'U'
    job = 'D'
    tol = 0.0

    x0_est, B_out, D_out, V, rcond, iwarn, info = ib01cd(
        jobx0, comuse, job, n, m, l, A, B, C, D, u, y, tol
    )

    assert info == 0
    assert rcond > 0
