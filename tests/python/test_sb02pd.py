"""
Tests for SB02PD: Continuous-time algebraic Riccati equation solver
using matrix sign function method with error bounds and condition estimates.

Solves: op(A)'*X + X*op(A) + Q - X*G*X = 0
"""

import numpy as np
import pytest
from ctrlsys import sb02pd


class TestSB02PDBasic:
    """Basic functionality tests from SLICOT HTML documentation."""

    def test_html_example(self):
        """
        Test case from SLICOT HTML documentation.
        N=2, JOB='A', TRANA='N', UPLO='U'

        Riccati: A'X + XA + Q - XGX = 0
        """
        n = 2

        a = np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ], dtype=float, order='F')

        q = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], dtype=float, order='F')

        g = np.array([
            [0.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        x_expected = np.array([
            [2.0, 1.0],
            [1.0, 2.0]
        ], dtype=float, order='F')

        x, rcond, ferr, wr, wi, info = sb02pd(a, g, q, job='A', trana='N', uplo='U')

        assert info == 0
        np.testing.assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(rcond, 0.1333, rtol=0.1)
        assert ferr < 1e-10

    def test_solution_only(self):
        """
        Test JOB='X' mode - compute solution only.
        Uses same data as HTML example.
        """
        a = np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ], dtype=float, order='F')

        q = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], dtype=float, order='F')

        g = np.array([
            [0.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        x_expected = np.array([
            [2.0, 1.0],
            [1.0, 2.0]
        ], dtype=float, order='F')

        x, rcond, ferr, wr, wi, info = sb02pd(a, g, q, job='X', trana='N', uplo='U')

        assert info == 0
        np.testing.assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)

    def test_transpose_mode(self):
        """
        Test TRANA='T' mode - solve A*X + X*A' + Q - X*G*X = 0.

        Use a stable system for transpose mode.
        """
        a = np.array([
            [-1.0, 0.5],
            [0.0, -2.0]
        ], dtype=float, order='F')

        q = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        g = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ], dtype=float, order='F')

        x, rcond, ferr, wr, wi, info = sb02pd(a, g, q, job='A', trana='T', uplo='U')

        assert info == 0 or info == 2
        assert x.shape == (2, 2)
        np.testing.assert_allclose(x, x.T, rtol=1e-12)


class TestSB02PDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_riccati_residual(self):
        """
        Validate Riccati equation residual: A'X + XA + Q - XGX = 0

        Random seed: 42
        """
        a = np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ], dtype=float, order='F')

        q = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], dtype=float, order='F')

        g = np.array([
            [0.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        x, rcond, ferr, wr, wi, info = sb02pd(a, g, q, job='A', trana='N', uplo='U')

        assert info == 0

        q_full = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], dtype=float, order='F')

        g_full = np.array([
            [0.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        residual = a.T @ x + x @ a + q_full - x @ g_full @ x
        np.testing.assert_allclose(residual, np.zeros((2, 2)), atol=1e-10)

    def test_solution_symmetry(self):
        """
        Validate solution X is symmetric: X = X'.

        Random seed: 123
        """
        np.random.seed(123)
        n = 4

        a = -0.5 * np.eye(n, dtype=float, order='F') + 0.1 * np.random.randn(n, n)
        a = np.asfortranarray(a)

        tmp = 0.1 * np.random.randn(n, n)
        q = tmp @ tmp.T
        q = np.asfortranarray(q)

        tmp = 0.1 * np.random.randn(n, n)
        g = tmp @ tmp.T
        g = np.asfortranarray(g)

        x, rcond, ferr, wr, wi, info = sb02pd(a, g, q, job='X', trana='N', uplo='U')

        assert info == 0 or info == 2
        np.testing.assert_allclose(x, x.T, rtol=1e-13, atol=1e-14)

    def test_closed_loop_stability(self):
        """
        Validate closed-loop eigenvalues are stable (negative real parts).

        For Riccati solution, A - G*X should have all eigenvalues in open left half-plane.
        """
        a = np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ], dtype=float, order='F')

        q = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], dtype=float, order='F')

        g = np.array([
            [0.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        x, rcond, ferr, wr, wi, info = sb02pd(a, g, q, job='A', trana='N', uplo='U')

        assert info == 0

        assert len(wr) == 2
        assert len(wi) == 2

        ac = a - g @ x
        eigs = np.linalg.eigvals(ac)

        for eig in eigs:
            assert eig.real < 0, f"Closed-loop eigenvalue {eig} has non-negative real part"

        eigs_from_wr_wi = wr + 1j * wi
        np.testing.assert_allclose(
            sorted(eigs.real),
            sorted(eigs_from_wr_wi.real),
            rtol=1e-10
        )


class TestSB02PDEdgeCases:
    """Edge case and boundary condition tests."""

    def test_zero_dimension(self):
        """Test n=0 case - should return immediately with success."""
        a = np.array([], dtype=float, order='F').reshape(0, 0)
        g = np.array([], dtype=float, order='F').reshape(0, 0)
        q = np.array([], dtype=float, order='F').reshape(0, 0)

        x, rcond, ferr, wr, wi, info = sb02pd(a, g, q, job='A', trana='N', uplo='U')

        assert info == 0
        assert x.shape == (0, 0)
        assert rcond == 1.0
        assert ferr == 0.0

    def test_lower_triangular_storage(self):
        """Test UPLO='L' - lower triangular storage of G and Q.

        Use stable system with lower triangular storage.
        """
        a = np.array([
            [-1.0, 0.0],
            [0.5, -2.0]
        ], dtype=float, order='F')

        q = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        g = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ], dtype=float, order='F')

        x, rcond, ferr, wr, wi, info = sb02pd(a, g, q, job='A', trana='N', uplo='L')

        assert info == 0 or info == 2
        np.testing.assert_allclose(x, x.T, rtol=1e-12)


class TestSB02PDLargerSystem:
    """Test with larger system to validate scaling."""

    def test_4x4_system(self):
        """
        Test 4x4 Riccati equation - double integrator cascade.

        Random seed: 456
        """
        n = 4

        a = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=float, order='F')

        q = np.eye(n, dtype=float, order='F')

        g = np.zeros((n, n), dtype=float, order='F')
        g[1, 1] = 1.0
        g[3, 3] = 1.0

        x, rcond, ferr, wr, wi, info = sb02pd(a, g, q, job='A', trana='N', uplo='U')

        assert info == 0 or info == 2
        assert x.shape == (n, n)

        np.testing.assert_allclose(x, x.T, rtol=1e-13, atol=1e-14)

        q_full = np.eye(n, dtype=float)
        g_full = np.zeros((n, n), dtype=float)
        g_full[1, 1] = 1.0
        g_full[3, 3] = 1.0

        residual = a.T @ x + x @ a + q_full - x @ g_full @ x
        np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-8)
