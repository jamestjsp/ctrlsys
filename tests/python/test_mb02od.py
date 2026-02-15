"""
Tests for MB02OD - Triangular matrix equation solver with condition estimation.

Solves op(A)*X = alpha*B or X*op(A) = alpha*B where A is triangular.
Only solves if reciprocal condition number RCOND > TOL.
"""

import numpy as np
import pytest


class TestMb02od:
    """Test suite for mb02od triangular system solver."""

    def test_upper_triangular_left_no_trans(self):
        """
        Test solving A*X = alpha*B with upper triangular A.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb02od

        np.random.seed(42)

        m, n = 3, 2
        alpha = 1.0

        a = np.triu(np.random.rand(m, m).astype(float, order='F')) + np.eye(m) * 2.0
        b = np.random.rand(m, n).astype(float, order='F')
        b_orig = b.copy()

        x, rcond, info = mb02od('L', 'U', 'N', 'N', '1', alpha, a, b)

        assert info == 0
        assert rcond > 0.0
        np.testing.assert_allclose(a @ x, alpha * b_orig, rtol=1e-13, atol=1e-14)

    def test_lower_triangular_right_trans(self):
        """
        Test solving X*A' = alpha*B with lower triangular A.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb02od

        np.random.seed(123)

        m, n = 3, 4
        alpha = 2.5

        a = np.tril(np.random.rand(n, n).astype(float, order='F')) + np.eye(n) * 3.0
        b = np.random.rand(m, n).astype(float, order='F')
        b_orig = b.copy()

        x, rcond, info = mb02od('R', 'L', 'T', 'N', 'I', alpha, a, b)

        assert info == 0
        assert rcond > 0.0
        np.testing.assert_allclose(x @ a.T, alpha * b_orig, rtol=1e-13, atol=1e-14)

    def test_unit_triangular(self):
        """
        Test solving with unit triangular matrix (diagonal assumed to be 1).

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import mb02od

        np.random.seed(456)

        n = 4
        alpha = 1.0

        a = np.triu(np.random.rand(n, n).astype(float, order='F'))
        np.fill_diagonal(a, 1.0)
        b = np.random.rand(n, 2).astype(float, order='F')
        b_orig = b.copy()

        x, rcond, info = mb02od('L', 'U', 'N', 'U', '1', alpha, a, b)

        assert info == 0
        assert rcond > 0.0
        np.testing.assert_allclose(a @ x, alpha * b_orig, rtol=1e-13, atol=1e-14)

    def test_singular_matrix_returns_info_1(self):
        """
        Test that singular matrix returns info=1 due to poor condition number.
        """
        from ctrlsys import mb02od

        n = 3
        alpha = 1.0

        a = np.array([
            [1e-16, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)
        b = np.array([[1.0], [2.0], [3.0]], order='F', dtype=float)

        x, rcond, info = mb02od('L', 'U', 'N', 'N', '1', alpha, a, b)

        assert info == 1
        assert rcond < 1e-10

    def test_alpha_zero_returns_zero_solution(self):
        """
        Test that alpha=0 returns zero solution without referencing A.
        """
        from ctrlsys import mb02od

        n = 2
        alpha = 0.0

        a = np.array([[1.0, 2.0], [0.0, 3.0]], order='F', dtype=float)
        b = np.array([[10.0], [20.0]], order='F', dtype=float)

        x, rcond, info = mb02od('L', 'U', 'N', 'N', '1', alpha, a, b)

        assert info == 0
        np.testing.assert_allclose(x, np.zeros_like(b), rtol=1e-14, atol=1e-14)

    def test_empty_nrowa_quick_return(self):
        """
        Test quick return when nrowa=0 (m=0, n>0 but side='L' so k=m=0).

        For side='R', we use n=0 to get nrowa=0 and test quick return.
        We use a valid 1x1 dummy A matrix since lda >= max(1,nrowa) = 1.
        """
        from ctrlsys import mb02od

        m, n = 3, 0
        alpha = 1.0

        a = np.array([[1.0]], order='F', dtype=float)
        b = np.empty((m, 0), order='F', dtype=float)

        x, rcond, info = mb02od('R', 'U', 'N', 'N', '1', alpha, a, b)

        assert info == 0
        assert rcond == 1.0

    def test_user_specified_tolerance(self):
        """
        Test that user-specified TOL is respected.

        With tight tolerance, a moderately-conditioned matrix should fail.
        """
        from ctrlsys import mb02od

        np.random.seed(789)

        n = 3
        alpha = 1.0

        a = np.triu(np.random.rand(n, n).astype(float, order='F'))
        a[0, 0] = 1e-4
        a[1, 1] = 1.0
        a[2, 2] = 1.0
        b = np.ones((n, 1), order='F', dtype=float)

        x, rcond, info = mb02od('L', 'U', 'N', 'N', '1', alpha, a, b, tol=0.5)

        assert info == 1
        assert rcond < 0.5

    def test_infinity_norm_condition(self):
        """
        Test condition number computed using infinity-norm.

        Random seed: 321 (for reproducibility)
        """
        from ctrlsys import mb02od

        np.random.seed(321)

        n = 3
        alpha = 1.0

        a = np.triu(np.random.rand(n, n).astype(float, order='F')) + np.eye(n) * 5.0
        b = np.random.rand(n, 2).astype(float, order='F')
        b_orig = b.copy()

        x, rcond, info = mb02od('L', 'U', 'N', 'N', 'I', alpha, a, b)

        assert info == 0
        assert rcond > 0.0
        np.testing.assert_allclose(a @ x, alpha * b_orig, rtol=1e-13, atol=1e-14)

    def test_solve_equation_invariant(self):
        """
        Test mathematical invariant: solution X satisfies op(A)*X = alpha*B.

        This validates the equation is correctly solved, not just shapes.
        Random seed: 999 (for reproducibility)
        """
        from ctrlsys import mb02od

        np.random.seed(999)

        m, n = 5, 3
        alpha = 3.7

        a = np.triu(np.random.rand(m, m).astype(float, order='F')) + np.eye(m) * 4.0
        b = np.random.rand(m, n).astype(float, order='F')
        b_orig = b.copy()

        x, rcond, info = mb02od('L', 'U', 'N', 'N', '1', alpha, a, b)

        assert info == 0

        reconstructed = a @ x
        expected = alpha * b_orig

        np.testing.assert_allclose(reconstructed, expected, rtol=1e-13, atol=1e-14)


class TestMb02odParameterValidation:
    """Test parameter validation for mb02od."""

    def test_invalid_side(self):
        """Test invalid SIDE parameter."""
        from ctrlsys import mb02od

        a = np.eye(2, order='F', dtype=float)
        b = np.ones((2, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02od('X', 'U', 'N', 'N', '1', 1.0, a, b)

    def test_invalid_uplo(self):
        """Test invalid UPLO parameter."""
        from ctrlsys import mb02od

        a = np.eye(2, order='F', dtype=float)
        b = np.ones((2, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02od('L', 'X', 'N', 'N', '1', 1.0, a, b)

    def test_invalid_trans(self):
        """Test invalid TRANS parameter."""
        from ctrlsys import mb02od

        a = np.eye(2, order='F', dtype=float)
        b = np.ones((2, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02od('L', 'U', 'X', 'N', '1', 1.0, a, b)

    def test_invalid_diag(self):
        """Test invalid DIAG parameter."""
        from ctrlsys import mb02od

        a = np.eye(2, order='F', dtype=float)
        b = np.ones((2, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02od('L', 'U', 'N', 'X', '1', 1.0, a, b)

    def test_invalid_norm(self):
        """Test invalid NORM parameter."""
        from ctrlsys import mb02od

        a = np.eye(2, order='F', dtype=float)
        b = np.ones((2, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02od('L', 'U', 'N', 'N', 'X', 1.0, a, b)
