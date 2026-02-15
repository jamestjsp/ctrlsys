"""
Tests for SB03QX - Estimate forward error bound for continuous-time Lyapunov equation.

Tests forward error estimation for: op(A)'*X + X*op(A) = C
where op(A) = A or A' and C is symmetric.

The routine estimates ||X - X_true||_max / ||X||_max given:
- T: Schur form of A (upper quasi-triangular)
- U: Orthogonal transformation (A = U*T*U')
- R: Absolute residual matrix with rounding error bounds
- XANORM: ||X||_max
"""

import numpy as np
import pytest


def solve_diagonal_lyapunov(t_diag, c):
    """
    Solve T'*X + X*T = C for diagonal T.

    For diagonal T with entries t_i, the solution is:
    X[i,j] = C[i,j] / (t_i + t_j)
    """
    n = len(t_diag)
    x = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            x[i, j] = c[i, j] / (t_diag[i] + t_diag[j])
    return x


def test_sb03qx_import():
    """Verify sb03qx can be imported."""
    from ctrlsys import sb03qx
    assert callable(sb03qx)


def test_sb03qx_small_system_reduced():
    """
    Test forward error bound estimation for 3x3 system in reduced form (LYAPUN='R').

    Uses a stable diagonal matrix T so we have exact control over eigenvalues.
    The Lyapunov equation T'*X + X*T = C has known solution properties.
    For diagonal T: X[i,j] = C[i,j] / (T[i,i] + T[j,j])
    """
    from ctrlsys import sb03qx

    n = 3

    t_diag = np.array([-1.0, -2.0, -3.0])
    t = np.diag(t_diag)
    t = np.asfortranarray(t)

    c = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.4],
        [0.3, 0.4, 4.0]
    ], dtype=float, order='F')

    x_sol = solve_diagonal_lyapunov(t_diag, c)

    xanorm = np.abs(x_sol).max()

    residual = t.T @ x_sol + x_sol @ t - c
    r = np.abs(residual) + 1e-15 * (np.abs(t.T @ x_sol) + np.abs(x_sol @ t) + np.abs(c))
    r = np.asfortranarray(r, dtype=float)

    u = np.eye(n, dtype=float, order='F')

    ferr, r_out, info = sb03qx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info == 0, f"Expected info=0, got {info}"
    assert ferr >= 0.0, f"Forward error bound must be non-negative, got {ferr}"
    assert ferr < 1.0, f"Expected small forward error bound, got {ferr}"


def test_sb03qx_small_system_original():
    """
    Test forward error bound estimation with orthogonal transformation (LYAPUN='O').

    Uses upper triangular T (already in Schur form) with U=I.
    For TRANA='N': T^T * X + X * T = C
    """
    from ctrlsys import sb03qx

    n = 3

    t = np.array([
        [-2.0, 1.0, 0.5],
        [0.0, -3.0, 1.0],
        [0.0, 0.0, -4.0]
    ], dtype=float, order='F')

    u = np.eye(n, dtype=float, order='F')

    c = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 3.0]
    ], dtype=float, order='F')

    x_sol = np.array([
        [0.25, 0.0, 0.0],
        [0.0, 0.333, 0.0],
        [0.0, 0.0, 0.375]
    ], dtype=float, order='F')
    xanorm = 1.0

    r = 1e-14 * np.ones((n, n), dtype=float, order='F')

    ferr, r_out, info = sb03qx('N', 'U', 'O', n, xanorm, t, u, r)

    assert info == 0, f"Expected info=0, got {info}"
    assert ferr >= 0.0, f"Forward error bound must be non-negative, got {ferr}"


def test_sb03qx_transpose_mode():
    """
    Test forward error bound with TRANA='T' (op(A) = A^T).

    For TRANA='T' and diagonal T: T * X + X * T = C
    Solution: X[i,j] = C[i,j] / (T[i,i] + T[j,j])
    """
    from ctrlsys import sb03qx

    n = 3

    t_diag = np.array([-1.5, -2.5, -3.5])
    t = np.diag(t_diag)
    t = np.asfortranarray(t)

    c = np.array([
        [2.0, 0.4, 0.2],
        [0.4, 3.0, 0.5],
        [0.2, 0.5, 4.0]
    ], dtype=float, order='F')

    x_sol = solve_diagonal_lyapunov(t_diag, c)
    xanorm = np.abs(x_sol).max()

    residual = t @ x_sol + x_sol @ t - c
    r = np.abs(residual) + 1e-15 * (np.abs(t @ x_sol) + np.abs(x_sol @ t) + np.abs(c))
    r = np.asfortranarray(r, dtype=float)

    u = np.eye(n, dtype=float, order='F')

    ferr, r_out, info = sb03qx('T', 'U', 'R', n, xanorm, t, u, r)

    assert info == 0, f"Expected info=0, got {info}"
    assert ferr >= 0.0, "Forward error bound must be non-negative"


def test_sb03qx_lower_triangle():
    """
    Test with UPLO='L' (lower triangle of R).

    For TRANA='N': T^T * X + X * T = C
    """
    from ctrlsys import sb03qx

    n = 3

    t_diag = np.array([-1.0, -2.0, -3.0])
    t = np.diag(t_diag)
    t = np.asfortranarray(t)

    c = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.4],
        [0.3, 0.4, 4.0]
    ], dtype=float, order='F')

    x_sol = solve_diagonal_lyapunov(t_diag, c)
    xanorm = np.abs(x_sol).max()

    residual = t.T @ x_sol + x_sol @ t - c
    r = np.abs(residual) + 1e-15 * (np.abs(t.T @ x_sol) + np.abs(x_sol @ t) + np.abs(c))
    r = np.asfortranarray(r, dtype=float)

    u = np.eye(n, dtype=float, order='F')

    ferr, r_out, info = sb03qx('N', 'L', 'R', n, xanorm, t, u, r)

    assert info == 0, f"Expected info=0, got {info}"
    assert ferr >= 0.0, "Forward error bound must be non-negative"


def test_sb03qx_zero_dimension():
    """Test quick return for n=0."""
    from ctrlsys import sb03qx

    n = 0
    t = np.array([[]], dtype=float, order='F').reshape(0, 0)
    u = np.array([[]], dtype=float, order='F').reshape(0, 0)
    r = np.array([[]], dtype=float, order='F').reshape(0, 0)
    xanorm = 1.0

    ferr, r_out, info = sb03qx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info == 0
    assert ferr == 0.0


def test_sb03qx_zero_xanorm():
    """Test quick return for xanorm=0 (zero solution)."""
    from ctrlsys import sb03qx

    n = 2
    t = np.diag(np.array([-1.0, -2.0], dtype=float, order='F'))
    t = np.asfortranarray(t)
    u = np.eye(n, dtype=float, order='F')
    r = np.zeros((n, n), dtype=float, order='F')
    xanorm = 0.0

    ferr, r_out, info = sb03qx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info == 0
    assert ferr == 0.0


def test_sb03qx_2x2_block():
    """
    Test with 2x2 block in Schur form (complex eigenvalue pair).

    A 2x2 block in real Schur form represents a complex conjugate pair.
    For TRANA='N': T^T * X + X * T = C

    T = [[-1, 2], [-2, -1]] has eigenvalues -1 +/- 2i
    """
    from ctrlsys import sb03qx

    n = 2

    t = np.array([
        [-1.0, 2.0],
        [-2.0, -1.0]
    ], dtype=float, order='F')

    c = np.array([
        [3.0, 0.5],
        [0.5, 4.0]
    ], dtype=float, order='F')

    xanorm = 1.0
    r = 1e-14 * np.ones((n, n), dtype=float, order='F')

    u = np.eye(n, dtype=float, order='F')

    ferr, r_out, info = sb03qx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info == 0, f"Expected info=0, got {info}"
    assert ferr >= 0.0, "Forward error bound must be non-negative"


def test_sb03qx_r_symmetrized():
    """
    Test that R matrix is symmetrized on output.

    The routine fills in the remaining triangle of R.
    For TRANA='N': T^T * X + X * T = C
    """
    from ctrlsys import sb03qx

    n = 3

    t_diag = np.array([-1.0, -2.0, -3.0])
    t = np.diag(t_diag)
    t = np.asfortranarray(t)

    c = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.4],
        [0.3, 0.4, 4.0]
    ], dtype=float, order='F')

    x_sol = solve_diagonal_lyapunov(t_diag, c)
    xanorm = np.abs(x_sol).max()

    residual = t.T @ x_sol + x_sol @ t - c
    r = np.abs(residual) + 1e-15 * (np.abs(t.T @ x_sol) + np.abs(x_sol @ t) + np.abs(c))
    r = np.asfortranarray(r, dtype=float)

    u = np.eye(n, dtype=float, order='F')

    ferr, r_out, info = sb03qx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info == 0
    np.testing.assert_allclose(r_out, r_out.T, rtol=1e-14)


def test_sb03qx_ill_conditioned():
    """
    Test forward error bound for ill-conditioned problem.

    When T and -T' have close eigenvalues, INFO = N+1 may be returned.
    For TRANA='N': T^T * X + X * T = C
    """
    from ctrlsys import sb03qx

    n = 2

    eps = 1e-10
    t = np.array([
        [-eps, 0.0],
        [0.0, -1.0]
    ], dtype=float, order='F')

    c = np.array([
        [1.0, 0.2],
        [0.2, 2.0]
    ], dtype=float, order='F')

    xanorm = 1.0 / (2 * eps)
    r = 1e-12 * np.ones((n, n), dtype=float, order='F')

    u = np.eye(n, dtype=float, order='F')

    ferr, r_out, info = sb03qx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info in [0, n + 1], f"Expected info=0 or {n+1}, got {info}"
    assert ferr >= 0.0, "Forward error bound must be non-negative"
