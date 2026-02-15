"""
Tests for SB03UD - Discrete-time Lyapunov equation solver with conditioning.

Solves: op(A)'*X*op(A) - X = scale*C

where op(A) = A or A', C and X are symmetric.
"""

import numpy as np
import pytest
from ctrlsys import sb03ud


class TestSB03UDBasic:
    """Basic functionality tests from HTML documentation."""

    def test_html_example_job_all(self):
        """
        Test HTML doc example with JOB='A' (compute all).

        N=3, computes solution, separation, rcond, and error bound.
        """
        n = 3

        a = np.array([
            [3.0, 1.0, 1.0],
            [1.0, 3.0, 0.0],
            [0.0, 0.0, 3.0]
        ], dtype=float, order='F')

        c = np.array([
            [25.0, 24.0, 15.0],
            [24.0, 32.0,  8.0],
            [15.0,  8.0, 40.0]
        ], dtype=float, order='F')

        x_expected = np.array([
            [2.0, 1.0, 1.0],
            [1.0, 3.0, 0.0],
            [1.0, 0.0, 4.0]
        ], dtype=float, order='F')

        x, scale, sepd, rcond, ferr, wr, wi, info = sb03ud(
            job='A', fact='N', trana='N', uplo='U', lyapun='O',
            a=a, c=c
        )

        assert info == 0
        np.testing.assert_allclose(x, x_expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(scale, 1.0, rtol=1e-4)
        np.testing.assert_allclose(sepd, 5.2302, rtol=1e-3)
        np.testing.assert_allclose(rcond, 0.1832, rtol=1e-2)
        np.testing.assert_allclose(ferr, 0.0, atol=1e-4)

    def test_solution_only(self):
        """
        Test JOB='X' (solution only).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3

        a = np.array([
            [3.0, 1.0, 1.0],
            [1.0, 3.0, 0.0],
            [0.0, 0.0, 3.0]
        ], dtype=float, order='F')

        c = np.array([
            [25.0, 24.0, 15.0],
            [24.0, 32.0,  8.0],
            [15.0,  8.0, 40.0]
        ], dtype=float, order='F')

        x_expected = np.array([
            [2.0, 1.0, 1.0],
            [1.0, 3.0, 0.0],
            [1.0, 0.0, 4.0]
        ], dtype=float, order='F')

        x, scale, sepd, rcond, ferr, wr, wi, info = sb03ud(
            job='X', fact='N', trana='N', uplo='U', lyapun='O',
            a=a, c=c
        )

        assert info == 0
        np.testing.assert_allclose(x, x_expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(scale, 1.0, rtol=1e-4)


class TestSB03UDMathematicalProperties:
    """Tests validating mathematical properties of the solution."""

    def test_lyapunov_residual(self):
        """
        Validate discrete-time Lyapunov equation: op(A)'*X*op(A) - X = scale*C.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4

        rho = 0.8
        a = rho * np.eye(n, dtype=float, order='F')
        a[0, 1] = 0.1
        a[1, 0] = 0.1
        a = np.asfortranarray(a)

        c = np.eye(n, dtype=float, order='F')
        c[0, 1] = 0.5
        c[1, 0] = 0.5
        c = np.asfortranarray(c)

        x, scale, sepd, rcond, ferr, wr, wi, info = sb03ud(
            job='X', fact='N', trana='N', uplo='U', lyapun='O',
            a=a, c=c
        )

        assert info == 0

        residual = a.T @ x @ a - x - scale * c
        np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-12)

    def test_transpose_form(self):
        """
        Validate TRANA='T': op(A) = A'.

        Equation: A*X*A' - X = scale*C

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 3

        rho = 0.7
        a = rho * np.eye(n, dtype=float, order='F')
        a[0, 1] = 0.2
        a = np.asfortranarray(a)

        c = np.eye(n, dtype=float, order='F')
        c = np.asfortranarray(c)

        x, scale, sepd, rcond, ferr, wr, wi, info = sb03ud(
            job='X', fact='N', trana='T', uplo='U', lyapun='O',
            a=a, c=c
        )

        assert info == 0

        residual = a @ x @ a.T - x - scale * c
        np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-12)

    def test_solution_symmetry(self):
        """
        Validate that solution X is symmetric.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 5

        rho = 0.6
        a = rho * np.eye(n, dtype=float, order='F')
        for i in range(n - 1):
            a[i, i + 1] = 0.1
        a = np.asfortranarray(a)

        c_base = np.random.randn(n, n)
        c = c_base @ c_base.T
        c = np.asfortranarray(c)

        x, scale, sepd, rcond, ferr, wr, wi, info = sb03ud(
            job='X', fact='N', trana='N', uplo='U', lyapun='O',
            a=a, c=c
        )

        assert info == 0
        np.testing.assert_allclose(x, x.T, rtol=1e-14)


class TestSB03UDReducedForm:
    """Tests for reduced Lyapunov equations (LYAPUN='R')."""

    def test_reduced_form_with_schur(self):
        """
        Test LYAPUN='R' with FACT='F' (Schur form provided).

        Uses quasi-triangular T directly, no U transformation.
        Random seed: 321 (for reproducibility)
        """
        np.random.seed(321)
        n = 3

        t = np.array([
            [0.5,  0.2,  0.1],
            [0.0, -0.3,  0.1],
            [0.0,  0.0,  0.4]
        ], dtype=float, order='F')

        c = np.eye(n, dtype=float, order='F')

        x, scale, sepd, rcond, ferr, wr, wi, info = sb03ud(
            job='X', fact='F', trana='N', uplo='U', lyapun='R',
            t=t, c=c
        )

        assert info == 0

        residual = t.T @ x @ t - x - scale * c
        np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-12)


class TestSB03UDEdgeCases:
    """Edge case tests."""

    def test_n_equals_zero(self):
        """Test N=0 quick return."""
        a = np.zeros((0, 0), dtype=float, order='F')
        c = np.zeros((0, 0), dtype=float, order='F')

        x, scale, sepd, rcond, ferr, wr, wi, info = sb03ud(
            job='A', fact='N', trana='N', uplo='U', lyapun='O',
            a=a, c=c
        )

        assert info == 0
        assert scale == 1.0
        assert rcond == 1.0
        assert ferr == 0.0

    def test_n_equals_one(self):
        """Test N=1 case."""
        a = np.array([[0.5]], dtype=float, order='F')
        c = np.array([[3.0]], dtype=float, order='F')

        x, scale, sepd, rcond, ferr, wr, wi, info = sb03ud(
            job='X', fact='N', trana='N', uplo='U', lyapun='O',
            a=a, c=c
        )

        assert info == 0
        x_expected = c / (a[0, 0]**2 - 1.0)
        np.testing.assert_allclose(x, x_expected, rtol=1e-14)

    def test_lower_triangular_c(self):
        """Test UPLO='L' (lower triangular C)."""
        n = 3

        a = np.array([
            [0.5, 0.1, 0.0],
            [0.0, 0.4, 0.1],
            [0.0, 0.0, 0.3]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 2.0, 0.0],
            [0.3, 0.2, 1.5]
        ], dtype=float, order='F')

        x, scale, sepd, rcond, ferr, wr, wi, info = sb03ud(
            job='X', fact='N', trana='N', uplo='L', lyapun='O',
            a=a, c=c
        )

        assert info == 0

        c_full = np.tril(c) + np.tril(c, -1).T
        residual = a.T @ x @ a - x - scale * c_full
        np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-12)


class TestSB03UDErrorHandling:
    """Error handling tests."""

    def test_invalid_job(self):
        """Test invalid JOB parameter raises ValueError."""
        a = np.eye(2, dtype=float, order='F')
        c = np.eye(2, dtype=float, order='F')

        with pytest.raises(ValueError, match="JOB must be"):
            sb03ud(job='Z', fact='N', trana='N', uplo='U', lyapun='O', a=a, c=c)

    def test_invalid_fact(self):
        """Test invalid FACT parameter raises ValueError."""
        a = np.eye(2, dtype=float, order='F')
        c = np.eye(2, dtype=float, order='F')

        with pytest.raises(ValueError, match="FACT must be"):
            sb03ud(job='X', fact='Z', trana='N', uplo='U', lyapun='O', a=a, c=c)
