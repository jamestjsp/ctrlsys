"""
Tests for SB03SD: Conditioning and forward error bound for discrete-time
Lyapunov equation.

Solves: op(A)'*X*op(A) - X = scale*C

Tests extracted from SLICOT HTML documentation example.
"""

import numpy as np
import pytest

from slicot import sb03sd


def solve_diagonal_discrete_lyapunov(a_diag, c):
    """
    Solve A'*X*A - X = C for diagonal A analytically.

    For diagonal A with entries a_i, the equation becomes:
    a_i * a_j * X[i,j] - X[i,j] = C[i,j]
    X[i,j] = C[i,j] / (a_i * a_j - 1)
    """
    n = len(a_diag)
    x = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            x[i, j] = c[i, j] / (a_diag[i] * a_diag[j] - 1.0)
    return x


class TestSB03SDBasic:
    """Basic functionality tests from HTML doc example."""

    def test_html_example_both(self):
        """
        Test from SB03SD HTML documentation example.

        System: 3x3 discrete-time Lyapunov equation
        JOB='B' (compute both condition number and error bound)
        FACT='F' (Schur factorization already provided)
        TRANA='N' (op(A) = A)
        UPLO='U' (upper triangular C)
        LYAPUN='O' (solve original equations)

        Uses upper triangular A which is already in Schur form.
        """
        n = 3

        a = np.array([
            [0.5, 0.1, 0.05],
            [0.0, 0.6, 0.1],
            [0.0, 0.0, 0.7]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 2.0, 0.5],
            [0.2, 0.5, 1.5]
        ], order='F', dtype=float)

        t = a.copy()
        u = np.eye(n, order='F', dtype=float)

        from slicot import sb03md
        a_work = a.copy(order='F')
        c_work = c.copy(order='F')
        md_result = sb03md('D', 'X', 'N', 'N', n, a_work, c_work)
        x = md_result[0]
        scale = md_result[5]
        info = md_result[-1]
        assert info == 0 or info == n + 1

        residual = a.T @ x @ a - x - scale * c
        np.testing.assert_allclose(residual, 0, atol=1e-10)

        result = sb03sd('B', 'F', 'N', 'U', 'O', n, scale,
                        a, t, u, c, x)

        sepd, rcond, ferr, info = result

        assert info == 0 or info == n + 1, f"SB03SD failed with info={info}"
        assert sepd > 0, f"SEPD={sepd} should be positive"
        assert 0 <= rcond <= 1, f"RCOND={rcond} should be in [0,1]"
        assert ferr >= 0, f"FERR={ferr} should be non-negative"

    def test_condition_only(self):
        """
        Test JOB='C' - compute only reciprocal condition number.

        Uses diagonal matrix for analytical Lyapunov solution.
        """
        n = 3

        a_diag = np.array([0.5, 0.6, 0.7])
        a = np.diag(a_diag)
        a = np.asfortranarray(a)

        c = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 2.0, 0.5],
            [0.2, 0.5, 1.5]
        ], order='F', dtype=float)

        x = solve_diagonal_discrete_lyapunov(a_diag, c)
        x = np.asfortranarray(x)

        residual = a.T @ x @ a - x - c
        np.testing.assert_allclose(residual, 0, atol=1e-10)

        t = a.copy()
        u = np.eye(n, order='F', dtype=float)

        scale = 1.0

        result = sb03sd('C', 'F', 'N', 'U', 'O', n, scale, a, t, u, c, x)

        sepd, rcond, ferr, info = result

        assert info == 0 or info == n + 1, f"Unexpected info={info}"
        assert sepd > 0, f"SEPD={sepd} should be positive"
        assert 0 <= rcond <= 1, f"RCOND={rcond} should be in [0,1]"

    def test_error_only(self):
        """
        Test JOB='E' - compute only error bound.

        Uses upper triangular matrix (already in Schur form).
        """
        n = 3

        a = np.array([
            [0.5, 0.1, 0.0],
            [0.0, 0.6, 0.1],
            [0.0, 0.0, 0.7]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.0],
            [0.5, 2.0, 0.5],
            [0.0, 0.5, 1.5]
        ], order='F', dtype=float)

        from slicot import sb03md
        a_work = a.copy(order='F')
        c_work = c.copy(order='F')
        md_result = sb03md('D', 'X', 'N', 'N', n, a_work, c_work)
        x = md_result[0]
        scale = md_result[5]
        info = md_result[-1]
        assert info == 0 or info == n + 1

        residual = a.T @ x @ a - x - scale * c
        np.testing.assert_allclose(residual, 0, atol=1e-10)

        t = a.copy()
        u = np.eye(n, order='F', dtype=float)

        result = sb03sd('E', 'F', 'N', 'U', 'O', n, scale, a, t, u, c, x)

        sepd, rcond, ferr, info = result

        assert info == 0 or info == n + 1
        assert ferr >= 0, f"FERR={ferr} should be non-negative"


class TestSB03SDEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with n=0 (quick return)."""
        n = 0
        a = np.array([], dtype=float).reshape(0, 0)
        t = np.array([], dtype=float).reshape(0, 0)
        u = np.array([], dtype=float).reshape(0, 0)
        c = np.array([], dtype=float).reshape(0, 0)
        x = np.array([], dtype=float).reshape(0, 0)

        result = sb03sd('B', 'F', 'N', 'U', 'O', n, 1.0, a, t, u, c, x)
        sepd, rcond, ferr, info = result

        assert info == 0
        assert rcond == 1.0
        assert ferr == 0.0

    def test_x_zero(self):
        """
        Test with X=0 (solution is zero matrix).

        For X=0, RCOND should be 0 and FERR should be 0.
        """
        n = 3
        a = np.eye(3, order='F', dtype=float) * 0.5
        t = a.copy()
        u = np.eye(3, order='F', dtype=float)
        c = np.zeros((3, 3), order='F', dtype=float)
        x = np.zeros((3, 3), order='F', dtype=float)

        result = sb03sd('B', 'F', 'N', 'U', 'O', n, 1.0, a, t, u, c, x)
        sepd, rcond, ferr, info = result

        assert info == 0
        assert rcond == 0.0
        assert ferr == 0.0


class TestSB03SDTranspose:
    """Test different TRANA options."""

    def test_transpose(self):
        """
        Test TRANA='T' (op(A) = A').

        Uses diagonal matrix for analytical solution.
        For TRANA='T', equation is A*X*A' - X = scale*C
        For diagonal A: a_i * a_j * X[i,j] - X[i,j] = C[i,j]
        """
        n = 3

        a_diag = np.array([0.4, 0.5, 0.6])
        a = np.diag(a_diag)
        a = np.asfortranarray(a)

        c = np.array([
            [2.0, 1.0, 0.5],
            [1.0, 3.0, 1.0],
            [0.5, 1.0, 2.0]
        ], order='F', dtype=float)

        x = solve_diagonal_discrete_lyapunov(a_diag, c)
        x = np.asfortranarray(x)

        residual = a.T @ x @ a - x - c
        np.testing.assert_allclose(residual, 0, atol=1e-10)

        t = a.copy()
        u = np.eye(n, order='F', dtype=float)

        result = sb03sd('B', 'F', 'T', 'U', 'O', n, 1.0, a, t, u, c, x)
        sepd, rcond, ferr, info = result

        assert info == 0 or info == n + 1
        assert sepd > 0
        assert 0 <= rcond <= 1


class TestSB03SDNoFact:
    """Test FACT='N' (compute Schur factorization internally)."""

    def test_compute_schur(self):
        """
        Test FACT='N' - routine computes Schur factorization.

        Uses upper triangular A (already in Schur form).
        """
        n = 3

        a = np.array([
            [0.5, 0.3, 0.1],
            [0.0, 0.6, 0.2],
            [0.0, 0.0, 0.4]
        ], order='F', dtype=float)

        c = np.array([
            [1.5, 0.8, 0.3],
            [0.8, 2.0, 0.6],
            [0.3, 0.6, 1.2]
        ], order='F', dtype=float)

        from slicot import sb03md
        a_work = a.copy(order='F')
        c_work = c.copy(order='F')
        md_result = sb03md('D', 'X', 'N', 'N', n, a_work, c_work)
        x = md_result[0]
        scale = md_result[5]
        info = md_result[-1]
        assert info == 0 or info == n + 1

        residual = a.T @ x @ a - x - scale * c
        np.testing.assert_allclose(residual, 0, atol=1e-10)

        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        result = sb03sd('B', 'N', 'N', 'U', 'O', n, scale, a.copy(order='F'), t, u, c, x)
        sepd, rcond, ferr, info = result

        assert info == 0 or info == n + 1
        assert sepd > 0
        assert 0 <= rcond <= 1


class TestSB03SDErrors:
    """Error handling tests."""

    def test_invalid_job(self):
        """Test invalid JOB parameter."""
        n = 2
        a = np.eye(2, order='F', dtype=float)
        t = np.eye(2, order='F', dtype=float)
        u = np.eye(2, order='F', dtype=float)
        c = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            sb03sd('X', 'F', 'N', 'U', 'O', n, 1.0, a, t, u, c, x)

    def test_invalid_n(self):
        """Test negative N."""
        a = np.eye(2, order='F', dtype=float)
        t = np.eye(2, order='F', dtype=float)
        u = np.eye(2, order='F', dtype=float)
        c = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            sb03sd('B', 'F', 'N', 'U', 'O', -1, 1.0, a, t, u, c, x)

    def test_invalid_scale(self):
        """Test scale out of range."""
        n = 2
        a = np.eye(2, order='F', dtype=float)
        t = np.eye(2, order='F', dtype=float)
        u = np.eye(2, order='F', dtype=float)
        c = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            sb03sd('B', 'F', 'N', 'U', 'O', n, 2.0, a, t, u, c, x)


class TestSB03SDLyapun:
    """Test LYAPUN='R' option (reduced equations only)."""

    def test_reduced_equations(self):
        """
        Test LYAPUN='R' - solve reduced Lyapunov equations only.

        In this mode, the routine works with T directly without transforming
        with U.
        """
        n = 3

        a = np.array([
            [0.5, 0.2, 0.1],
            [0.0, 0.4, 0.3],
            [0.0, 0.0, 0.6]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 1.5, 0.3],
            [0.2, 0.3, 2.0]
        ], order='F', dtype=float)

        from slicot import sb03md
        a_work = a.copy(order='F')
        c_work = c.copy(order='F')
        md_result = sb03md('D', 'X', 'N', 'N', n, a_work, c_work)
        x = md_result[0]
        scale = md_result[5]
        info = md_result[-1]
        assert info == 0 or info == n + 1

        residual = a.T @ x @ a - x - scale * c
        np.testing.assert_allclose(residual, 0, atol=1e-10)

        t = a.copy()
        u = np.eye(n, order='F', dtype=float)

        result = sb03sd('B', 'F', 'N', 'U', 'R', n, scale, a, t, u, c, x)
        sepd, rcond, ferr, info = result

        assert info == 0 or info == n + 1
        assert sepd > 0
        assert 0 <= rcond <= 1
