"""
Tests for MB01WD - Residuals of Lyapunov or Stein equations for Cholesky factored solutions.

Computes:
  R = alpha*(op(A)'*op(T)'*op(T) + op(T)'*op(T)*op(A)) + beta*R  (continuous, DICO='C')
  R = alpha*(op(A)'*op(T)'*op(T)*op(A) - op(T)'*op(T)) + beta*R  (discrete, DICO='D')

where R and result are symmetric, T is triangular, A is general or Hessenberg.
"""

import numpy as np
import pytest
from ctrlsys import mb01wd


class TestMB01WDContinuous:
    """Test continuous-time formula (DICO='C')."""

    def test_basic_upper_notrans(self):
        """
        Basic test: DICO='C', UPLO='U', TRANS='N', HESS='F'.
        Formula: R = alpha*(A'*T'*T + T'*T*A) + beta*R

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3
        alpha = 1.0
        beta = 0.5

        A = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], order='F', dtype=float)

        T = np.array([
            [2.0, 1.0, 0.5],
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 4.0]
        ], order='F', dtype=float)

        R_init = np.array([
            [1.0, 0.5, 0.25],
            [0.5, 2.0, 0.5],
            [0.25, 0.5, 3.0]
        ], order='F', dtype=float)

        TT = T.T @ T
        R_expected_upper = alpha * (A.T @ TT + TT @ A) + beta * R_init
        R_expected_upper = np.triu(R_expected_upper)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('C', 'U', 'N', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.triu(R_out), R_expected_upper, rtol=1e-14)

    def test_basic_lower_notrans(self):
        """
        Test: DICO='C', UPLO='L', TRANS='N', HESS='F'.
        Formula: R = alpha*(A'*T'*T + T'*T*A) + beta*R (lower triangle)

        Random seed: 43 (for reproducibility)
        """
        np.random.seed(43)
        n = 3
        alpha = 2.0
        beta = 0.0

        A = np.random.randn(n, n).astype(float, order='F')
        T = np.tril(np.random.randn(n, n)).astype(float, order='F')
        R_init = np.zeros((n, n), order='F', dtype=float)

        TT = T.T @ T
        R_expected = alpha * (A.T @ TT + TT @ A)
        R_expected_lower = np.tril(R_expected)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('C', 'L', 'N', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.tril(R_out), R_expected_lower, rtol=1e-14)

    def test_transpose(self):
        """
        Test: DICO='C', UPLO='U', TRANS='T', HESS='F'.
        Formula: R = alpha*(A*T*T' + T*T'*A') + beta*R

        Random seed: 44 (for reproducibility)
        """
        np.random.seed(44)
        n = 4
        alpha = 0.5
        beta = 1.0

        A = np.random.randn(n, n).astype(float, order='F')
        T = np.triu(np.random.randn(n, n)).astype(float, order='F')
        R_init = np.eye(n, order='F', dtype=float)

        TT = T @ T.T
        R_expected = alpha * (A @ TT + TT @ A.T) + beta * R_init
        R_expected_upper = np.triu(R_expected)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('C', 'U', 'T', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.triu(R_out), R_expected_upper, rtol=1e-14)


class TestMB01WDDiscrete:
    """Test discrete-time formula (DICO='D')."""

    def test_basic_upper_notrans(self):
        """
        Test: DICO='D', UPLO='U', TRANS='N', HESS='F'.
        Formula: R = alpha*(A'*T'*T*A - T'*T) + beta*R

        Random seed: 45 (for reproducibility)
        """
        np.random.seed(45)
        n = 3
        alpha = 1.0
        beta = 0.0

        A = np.array([
            [0.5, 0.1, 0.0],
            [0.2, 0.6, 0.1],
            [0.0, 0.2, 0.7]
        ], order='F', dtype=float)

        T = np.array([
            [1.0, 0.5, 0.2],
            [0.0, 1.5, 0.3],
            [0.0, 0.0, 2.0]
        ], order='F', dtype=float)

        R_init = np.zeros((n, n), order='F', dtype=float)

        TT = T.T @ T
        R_expected = alpha * (A.T @ TT @ A - TT) + beta * R_init
        R_expected_upper = np.triu(R_expected)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('D', 'U', 'N', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.triu(R_out), R_expected_upper, rtol=1e-14)

    def test_transpose_lower(self):
        """
        Test: DICO='D', UPLO='L', TRANS='T', HESS='F'.
        Formula: R = alpha*(A*T*T'*A' - T*T') + beta*R

        Random seed: 46 (for reproducibility)
        """
        np.random.seed(46)
        n = 4
        alpha = 2.0
        beta = 0.5

        A = np.random.randn(n, n).astype(float, order='F')
        T = np.tril(np.random.randn(n, n)).astype(float, order='F')
        R_init = np.eye(n, order='F', dtype=float)

        TT = T @ T.T
        R_expected = alpha * (A @ TT @ A.T - TT) + beta * R_init
        R_expected_lower = np.tril(R_expected)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('D', 'L', 'T', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.tril(R_out), R_expected_lower, rtol=1e-14)


class TestMB01WDEdgeCases:
    """Test edge cases and special values."""

    def test_alpha_zero(self):
        """When alpha=0, result is beta*R (T and A not referenced)."""
        n = 3
        alpha = 0.0
        beta = 2.0

        R_init = np.eye(n, order='F', dtype=float)
        A = np.zeros((n, n), order='F', dtype=float)
        T = np.zeros((n, n), order='F', dtype=float)

        R_expected_upper = beta * np.triu(R_init)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('C', 'U', 'N', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.triu(R_out), R_expected_upper, rtol=1e-14)

    def test_alpha_zero_beta_zero(self):
        """When alpha=0 and beta=0, result is zero."""
        n = 3
        alpha = 0.0
        beta = 0.0

        R_init = np.ones((n, n), order='F', dtype=float)
        A = np.ones((n, n), order='F', dtype=float)
        T = np.ones((n, n), order='F', dtype=float)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('C', 'U', 'N', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.triu(R_out), 0.0, atol=1e-15)

    def test_n_zero(self):
        """Quick return for n=0."""
        n = 0
        R = np.array([], dtype=float).reshape(0, 0, order='F')
        A = np.array([], dtype=float).reshape(0, 0, order='F')
        T = np.array([], dtype=float).reshape(0, 0, order='F')

        A_out, R_out, info = mb01wd('C', 'U', 'N', 'F', n, 1.0, 0.0, R, A, T)

        assert info == 0
        assert R_out.shape == (0, 0)

    def test_beta_one_alpha_zero(self):
        """When alpha=0 and beta=1, R unchanged."""
        n = 2
        alpha = 0.0
        beta = 1.0

        R_init = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
        A = np.zeros((n, n), order='F', dtype=float)
        T = np.zeros((n, n), order='F', dtype=float)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('C', 'U', 'N', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.triu(R_out), np.triu(R_init), rtol=1e-14)


class TestMB01WDHessenberg:
    """Test Hessenberg form (HESS='H')."""

    def test_hessenberg_continuous(self):
        """
        Test: DICO='C', UPLO='U', TRANS='N', HESS='H'.
        A is upper Hessenberg (one subdiagonal).

        Random seed: 47 (for reproducibility)
        """
        np.random.seed(47)
        n = 4
        alpha = 1.0
        beta = 0.0

        A = np.triu(np.random.randn(n, n), -1).astype(float, order='F')
        T = np.triu(np.random.randn(n, n)).astype(float, order='F')
        R_init = np.zeros((n, n), order='F', dtype=float)

        TT = T.T @ T
        R_expected = alpha * (A.T @ TT + TT @ A)
        R_expected_upper = np.triu(R_expected)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('C', 'U', 'N', 'H', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.triu(R_out), R_expected_upper, rtol=1e-14)

    def test_hessenberg_discrete(self):
        """
        Test: DICO='D', UPLO='U', TRANS='N', HESS='H'.
        A is upper Hessenberg.

        Random seed: 48 (for reproducibility)
        """
        np.random.seed(48)
        n = 4
        alpha = 1.0
        beta = 0.5

        A = np.triu(np.random.randn(n, n), -1).astype(float, order='F')
        T = np.triu(np.random.randn(n, n)).astype(float, order='F')
        R_init = np.eye(n, order='F', dtype=float)

        TT = T.T @ T
        R_expected = alpha * (A.T @ TT @ A - TT) + beta * R_init
        R_expected_upper = np.triu(R_expected)

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('D', 'U', 'N', 'H', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(np.triu(R_out), R_expected_upper, rtol=1e-14)


class TestMB01WDErrors:
    """Test error handling."""

    def test_invalid_dico(self):
        """Invalid DICO parameter."""
        n = 2
        R = np.zeros((n, n), order='F', dtype=float)
        A = np.zeros((n, n), order='F', dtype=float)
        T = np.zeros((n, n), order='F', dtype=float)

        A_out, R_out, info = mb01wd('X', 'U', 'N', 'F', n, 1.0, 0.0, R, A, T)
        assert info == -1

    def test_invalid_uplo(self):
        """Invalid UPLO parameter."""
        n = 2
        R = np.zeros((n, n), order='F', dtype=float)
        A = np.zeros((n, n), order='F', dtype=float)
        T = np.zeros((n, n), order='F', dtype=float)

        A_out, R_out, info = mb01wd('C', 'X', 'N', 'F', n, 1.0, 0.0, R, A, T)
        assert info == -2

    def test_invalid_trans(self):
        """Invalid TRANS parameter."""
        n = 2
        R = np.zeros((n, n), order='F', dtype=float)
        A = np.zeros((n, n), order='F', dtype=float)
        T = np.zeros((n, n), order='F', dtype=float)

        A_out, R_out, info = mb01wd('C', 'U', 'X', 'F', n, 1.0, 0.0, R, A, T)
        assert info == -3

    def test_invalid_hess(self):
        """Invalid HESS parameter."""
        n = 2
        R = np.zeros((n, n), order='F', dtype=float)
        A = np.zeros((n, n), order='F', dtype=float)
        T = np.zeros((n, n), order='F', dtype=float)

        A_out, R_out, info = mb01wd('C', 'U', 'N', 'X', n, 1.0, 0.0, R, A, T)
        assert info == -4

    def test_negative_n(self):
        """Negative N parameter."""
        R = np.zeros((1, 1), order='F', dtype=float)
        A = np.zeros((1, 1), order='F', dtype=float)
        T = np.zeros((1, 1), order='F', dtype=float)

        A_out, R_out, info = mb01wd('C', 'U', 'N', 'F', -1, 1.0, 0.0, R, A, T)
        assert info == -5


class TestMB01WDMathProperties:
    """Mathematical property tests."""

    def test_result_symmetry_continuous(self):
        """
        Continuous formula produces symmetric result.
        We verify upper and lower triangles match on full computation.

        Random seed: 100 (for reproducibility)
        """
        np.random.seed(100)
        n = 4
        alpha = 1.5
        beta = 0.0

        A = np.random.randn(n, n).astype(float, order='F')
        T = np.triu(np.random.randn(n, n)).astype(float, order='F')
        R_init = np.zeros((n, n), order='F', dtype=float)

        TT = T.T @ T
        R_full = alpha * (A.T @ TT + TT @ A)

        np.testing.assert_allclose(R_full, R_full.T, rtol=1e-14)

    def test_result_symmetry_discrete(self):
        """
        Discrete formula produces symmetric result.
        We verify upper and lower triangles match on full computation.

        Random seed: 101 (for reproducibility)
        """
        np.random.seed(101)
        n = 4
        alpha = 1.0
        beta = 0.5

        A = np.random.randn(n, n).astype(float, order='F')
        T = np.triu(np.random.randn(n, n)).astype(float, order='F')
        R_init = np.eye(n, order='F', dtype=float)

        TT = T.T @ T
        R_full = alpha * (A.T @ TT @ A - TT) + beta * R_init

        np.testing.assert_allclose(R_full, R_full.T, rtol=1e-13)

    def test_a_output_continuous_notrans(self):
        """
        Verify A output: For DICO='C', TRANS='N', A := alpha*T'*T*A.

        Random seed: 102 (for reproducibility)
        """
        np.random.seed(102)
        n = 3
        alpha = 2.0
        beta = 0.0

        A = np.random.randn(n, n).astype(float, order='F')
        T = np.triu(np.random.randn(n, n)).astype(float, order='F')
        R_init = np.zeros((n, n), order='F', dtype=float)

        A_expected = alpha * T.T @ T @ A

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('C', 'U', 'N', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(A_out, A_expected, rtol=1e-14)

    def test_a_output_discrete_notrans(self):
        """
        Verify A output: For DICO='D', TRANS='N', A := T*A.

        Random seed: 103 (for reproducibility)
        """
        np.random.seed(103)
        n = 3
        alpha = 1.0
        beta = 0.0

        A = np.random.randn(n, n).astype(float, order='F')
        T = np.triu(np.random.randn(n, n)).astype(float, order='F')
        R_init = np.zeros((n, n), order='F', dtype=float)

        A_expected = T @ A

        A_work = A.copy(order='F')
        R_work = R_init.copy(order='F')

        A_out, R_out, info = mb01wd('D', 'U', 'N', 'F', n, alpha, beta, R_work, A_work, T)

        assert info == 0
        np.testing.assert_allclose(A_out, A_expected, rtol=1e-14)
