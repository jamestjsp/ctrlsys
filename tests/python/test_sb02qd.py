"""
Tests for SB02QD: Estimating conditioning and forward error bound for the
solution of continuous-time Riccati equation.

op(A)'*X + X*op(A) + Q - X*G*X = 0
"""

import numpy as np
import pytest
from ctrlsys import sb02qd


class TestSB02QDBasic:
    """Basic functionality tests from HTML documentation."""

    def test_html_example(self):
        """
        Test from SLICOT HTML documentation.

        N=2, JOB='B' (both condition and error), FACT='N' (compute Schur),
        TRANA='N' (no transpose), UPLO='U' (upper), LYAPUN='O' (original).

        System matrices:
        A = [[0, 1], [0, 0]]
        Q = [[1, 0], [0, 2]]
        G = [[0, 0], [0, 1]]
        X (solution) = [[2, 1], [1, 2]]

        Expected: SEP=0.4, RCOND=0.1333, FERR=0.0
        """
        n = 2
        a = np.array([[0.0, 1.0],
                      [0.0, 0.0]], order='F', dtype=float)
        q = np.array([[1.0, 0.0],
                      [0.0, 2.0]], order='F', dtype=float)
        g = np.array([[0.0, 0.0],
                      [0.0, 1.0]], order='F', dtype=float)
        x = np.array([[2.0, 1.0],
                      [1.0, 2.0]], order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'B', 'N', 'N', 'U', 'O', a, t, u, g, q, x
        )

        assert info == 0
        np.testing.assert_allclose(sep, 0.4, rtol=1e-3)
        np.testing.assert_allclose(rcond, 0.1333, rtol=1e-3)
        np.testing.assert_allclose(ferr, 0.0, atol=1e-10)

    def test_condition_only(self):
        """
        Test JOB='C' - compute only reciprocal condition number.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3

        a = np.array([[-1.0, 0.5, 0.0],
                      [0.0, -2.0, 0.3],
                      [0.0, 0.0, -3.0]], order='F', dtype=float)
        q = np.eye(n, order='F', dtype=float)
        g = 0.1 * np.eye(n, order='F', dtype=float)
        x = np.array([[1.05361, 0.27132, 0.01398],
                      [0.27132, 0.53096, 0.03986],
                      [0.01398, 0.03986, 0.17275]], order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'C', 'N', 'N', 'U', 'O', a, t, u, g, q, x
        )

        assert info == 0 or info == n + 1
        assert sep > 0
        assert 0 < rcond <= 1

    def test_error_bound_only(self):
        """
        Test JOB='E' - compute only forward error bound.

        Uses same system as test_condition_only.
        """
        n = 3

        a = np.array([[-1.0, 0.5, 0.0],
                      [0.0, -2.0, 0.3],
                      [0.0, 0.0, -3.0]], order='F', dtype=float)
        q = np.eye(n, order='F', dtype=float)
        g = 0.1 * np.eye(n, order='F', dtype=float)
        x = np.array([[1.05361, 0.27132, 0.01398],
                      [0.27132, 0.53096, 0.03986],
                      [0.01398, 0.03986, 0.17275]], order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'E', 'N', 'N', 'U', 'O', a, t, u, g, q, x
        )

        assert info == 0 or info == n + 1
        assert ferr >= 0


class TestSB02QDTranspose:
    """Test transpose mode (TRANA='T')."""

    def test_transpose_mode(self):
        """
        Test with TRANA='T' - transpose of A.

        For transpose mode: op(A) = A^T, so Riccati equation becomes
        A*X + X*A^T + Q - X*G*X = 0
        """
        n = 2
        a = np.array([[0.0, 0.0],
                      [1.0, 0.0]], order='F', dtype=float)
        q = np.array([[1.0, 0.0],
                      [0.0, 2.0]], order='F', dtype=float)
        g = np.array([[0.0, 0.0],
                      [0.0, 1.0]], order='F', dtype=float)
        x = np.array([[2.0, 1.0],
                      [1.0, 2.0]], order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'B', 'N', 'T', 'U', 'O', a, t, u, g, q, x
        )

        assert info == 0 or info == n + 1
        assert sep > 0
        assert 0 < rcond <= 1
        assert ferr >= 0


class TestSB02QDLowerTriangular:
    """Test lower triangular storage (UPLO='L')."""

    def test_lower_triangular(self):
        """
        Test with UPLO='L' - lower triangular storage for Q and G.
        """
        n = 2
        a = np.array([[0.0, 1.0],
                      [0.0, 0.0]], order='F', dtype=float)
        q = np.array([[1.0, 0.0],
                      [0.0, 2.0]], order='F', dtype=float)
        g = np.array([[0.0, 0.0],
                      [0.0, 1.0]], order='F', dtype=float)
        x = np.array([[2.0, 1.0],
                      [1.0, 2.0]], order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'B', 'N', 'N', 'L', 'O', a, t, u, g, q, x
        )

        assert info == 0
        np.testing.assert_allclose(sep, 0.4, rtol=1e-3)
        np.testing.assert_allclose(rcond, 0.1333, rtol=1e-3)


class TestSB02QDReducedLyapunov:
    """Test reduced Lyapunov mode (LYAPUN='R')."""

    def test_reduced_lyapunov(self):
        """
        Test with LYAPUN='R' - reduced Lyapunov equations.

        Must provide pre-computed Schur factorization in T.
        """
        n = 2
        a = np.array([[0.0, 1.0],
                      [0.0, 0.0]], order='F', dtype=float)
        q = np.array([[1.0, 0.0],
                      [0.0, 2.0]], order='F', dtype=float)
        g = np.array([[0.0, 0.0],
                      [0.0, 1.0]], order='F', dtype=float)
        x = np.array([[2.0, 1.0],
                      [1.0, 2.0]], order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'B', 'N', 'N', 'U', 'R', a, t, u, g, q, x
        )

        assert info == 0 or info == n + 1
        assert sep > 0
        assert 0 < rcond <= 1


class TestSB02QDFactorizedInput:
    """Test with pre-factorized Schur form (FACT='F')."""

    def test_factorized_input(self):
        """
        Test with FACT='F' - Schur factorization already provided.

        When FACT='F', T and U must contain the Schur factorization of
        Ac = A - G*X (for TRANA='N') or Ac = A - X*G (for TRANA='T').
        """
        n = 2
        a = np.array([[0.0, 1.0],
                      [0.0, 0.0]], order='F', dtype=float)
        q = np.array([[1.0, 0.0],
                      [0.0, 2.0]], order='F', dtype=float)
        g = np.array([[0.0, 0.0],
                      [0.0, 1.0]], order='F', dtype=float)
        x = np.array([[2.0, 1.0],
                      [1.0, 2.0]], order='F', dtype=float)
        t_init = np.zeros((n, n), order='F', dtype=float)
        u_init = np.zeros((n, n), order='F', dtype=float)

        # First call with FACT='N' to compute Schur factorization
        _, _, _, t, u, info0 = sb02qd(
            'B', 'N', 'N', 'U', 'O', a, t_init, u_init, g, q, x
        )
        assert info0 == 0

        # Now test with FACT='F' using pre-computed Schur form
        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'B', 'F', 'N', 'U', 'O', a, t, u, g, q, x
        )

        assert info == 0 or info == n + 1
        np.testing.assert_allclose(sep, 0.4, rtol=1e-2)
        np.testing.assert_allclose(rcond, 0.1333, rtol=1e-2)


class TestSB02QDEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with N=0 - quick return case."""
        n = 0
        a = np.zeros((0, 0), order='F', dtype=float)
        q = np.zeros((0, 0), order='F', dtype=float)
        g = np.zeros((0, 0), order='F', dtype=float)
        x = np.zeros((0, 0), order='F', dtype=float)
        t = np.zeros((0, 0), order='F', dtype=float)
        u = np.zeros((0, 0), order='F', dtype=float)

        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'B', 'N', 'N', 'U', 'O', a, t, u, g, q, x
        )

        assert info == 0
        assert rcond == 1.0
        assert ferr == 0.0

    def test_zero_solution(self):
        """Test when solution X is zero matrix."""
        n = 2
        a = np.array([[0.0, 1.0],
                      [0.0, 0.0]], order='F', dtype=float)
        q = np.array([[0.0, 0.0],
                      [0.0, 0.0]], order='F', dtype=float)
        g = np.array([[0.0, 0.0],
                      [0.0, 1.0]], order='F', dtype=float)
        x = np.zeros((n, n), order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'B', 'N', 'N', 'U', 'O', a, t, u, g, q, x
        )

        assert info == 0
        assert rcond == 0.0
        assert ferr == 0.0


class TestSB02QDRiccatiResidual:
    """Mathematical property tests: verify Riccati residual."""

    def test_residual_small(self):
        """
        Verify that X satisfies the Riccati equation:
        A'*X + X*A + Q - X*G*X = 0

        The residual should be small when X is the true solution.
        """
        n = 2
        a = np.array([[0.0, 1.0],
                      [0.0, 0.0]], order='F', dtype=float)
        q = np.array([[1.0, 0.0],
                      [0.0, 2.0]], order='F', dtype=float)
        g = np.array([[0.0, 0.0],
                      [0.0, 1.0]], order='F', dtype=float)
        x = np.array([[2.0, 1.0],
                      [1.0, 2.0]], order='F', dtype=float)

        residual = a.T @ x + x @ a + q - x @ g @ x
        np.testing.assert_allclose(residual, np.zeros((n, n)),
                                   atol=1e-14)

    def test_schur_form_output(self):
        """
        Verify Schur factorization output: Ac = U*T*U'

        When FACT='N', the routine computes Schur form of Ac = A - G*X.
        """
        n = 2
        a = np.array([[0.0, 1.0],
                      [0.0, 0.0]], order='F', dtype=float)
        q = np.array([[1.0, 0.0],
                      [0.0, 2.0]], order='F', dtype=float)
        g = np.array([[0.0, 0.0],
                      [0.0, 1.0]], order='F', dtype=float)
        x = np.array([[2.0, 1.0],
                      [1.0, 2.0]], order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        sep, rcond, ferr, t_out, u_out, info = sb02qd(
            'B', 'N', 'N', 'U', 'O', a, t, u, g, q, x
        )

        assert info == 0

        ac = a - g @ x
        ac_reconstructed = u_out @ t_out @ u_out.T
        np.testing.assert_allclose(ac_reconstructed, ac, rtol=1e-14, atol=1e-15)

        u_orthogonal = u_out @ u_out.T
        np.testing.assert_allclose(u_orthogonal, np.eye(n), rtol=1e-14, atol=1e-15)


class TestSB02QDErrors:
    """Error handling tests."""

    def test_invalid_job(self):
        """Test invalid JOB parameter."""
        n = 2
        a = np.zeros((n, n), order='F', dtype=float)
        q = np.zeros((n, n), order='F', dtype=float)
        g = np.zeros((n, n), order='F', dtype=float)
        x = np.zeros((n, n), order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            sb02qd('X', 'N', 'N', 'U', 'O', a, t, u, g, q, x)

    def test_invalid_fact(self):
        """Test invalid FACT parameter."""
        n = 2
        a = np.zeros((n, n), order='F', dtype=float)
        q = np.zeros((n, n), order='F', dtype=float)
        g = np.zeros((n, n), order='F', dtype=float)
        x = np.zeros((n, n), order='F', dtype=float)
        t = np.zeros((n, n), order='F', dtype=float)
        u = np.zeros((n, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            sb02qd('B', 'X', 'N', 'U', 'O', a, t, u, g, q, x)
