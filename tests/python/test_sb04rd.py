"""
Tests for SB04RD: Discrete-time Sylvester equation solver.

Solves: X + A*X*B = C

where A is in Schur or Hessenberg form and B is in Schur or Hessenberg form
(at least one must be in Schur form).
"""

import numpy as np
import pytest
from ctrlsys import sb04rd


class TestSB04RDBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_html_doc_example(self):
        """
        Test from SLICOT HTML documentation.

        A is 5x5 upper Hessenberg, B is 5x5 upper Schur.
        ABSCHU='B' (B in Schur form), ULA='U', ULB='U'.
        Expected: X = I (identity matrix).
        """
        n = 5
        m = 5

        # A - upper Hessenberg (read row by row from HTML)
        a = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 1.0],
            [0.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 0.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 0.0, 9.0, 1.0]
        ], dtype=float, order='F')

        # B - upper quasi-triangular Schur form (2x2 block at end)
        b = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, -5.0],
            [0.0, 0.0, 0.0, 4.0, 1.0]
        ], dtype=float, order='F')

        # C - RHS matrix (read row by row from HTML)
        c = np.array([
            [2.0, 4.0, 10.0, 40.0, 7.0],
            [6.0, 20.0, 40.0, 74.0, 38.0],
            [0.0, 2.0, 8.0, 36.0, 2.0],
            [0.0, 0.0, 6.0, 52.0, -9.0],
            [0.0, 0.0, 0.0, 13.0, -43.0]
        ], dtype=float, order='F')

        x, info = sb04rd('B', 'U', 'U', a, b, c)

        assert info == 0

        # Expected result is identity matrix
        x_expected = np.eye(5, dtype=float, order='F')
        np.testing.assert_allclose(x, x_expected, rtol=1e-10, atol=1e-10)


class TestSB04RDSchurSchur:
    """Tests when both A and B are in Schur form."""

    def test_both_schur_upper(self):
        """
        Test with both matrices in upper Schur form.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4
        m = 3

        # Create upper triangular (Schur) matrices
        a = np.triu(np.random.randn(n, n).astype(float, order='F'))
        b = np.triu(np.random.randn(m, m).astype(float, order='F'))

        # Generate random C and solve
        c_orig = np.random.randn(n, m).astype(float, order='F')
        c = c_orig.copy()

        x, info = sb04rd('S', 'U', 'U', a, b, c)

        assert info == 0

        # Verify: X + A @ X @ B = C
        residual = x + a @ x @ b - c_orig
        np.testing.assert_allclose(residual, np.zeros((n, m)), atol=1e-13)


class TestSB04RDSylvesterEquation:
    """Mathematical property tests for Sylvester equation."""

    def test_equation_residual_identity_solution(self):
        """
        Validate Sylvester equation X + A*X*B = C holds when X = I.

        Construct C = I + A*I*B = I + A*B so solution is identity.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4

        # Upper triangular A (Schur form)
        a = np.triu(np.random.randn(n, n).astype(float, order='F'))
        # Upper triangular B (Schur form)
        b = np.triu(np.random.randn(n, n).astype(float, order='F'))

        # X = I, so C = I + A @ I @ B = I + A @ B
        x_expected = np.eye(n, dtype=float, order='F')
        c = x_expected + a @ x_expected @ b
        c = np.asfortranarray(c)

        x, info = sb04rd('S', 'U', 'U', a, b, c)

        assert info == 0
        np.testing.assert_allclose(x, x_expected, rtol=1e-12, atol=1e-13)

    def test_equation_residual_random_solution(self):
        """
        Verify residual X + A*X*B - C = 0 for random data.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 5
        m = 4

        # Upper triangular matrices (Schur form)
        a = np.triu(np.random.randn(n, n).astype(float, order='F'))
        b = np.triu(np.random.randn(m, m).astype(float, order='F'))
        c_orig = np.random.randn(n, m).astype(float, order='F')
        c = c_orig.copy()

        x, info = sb04rd('S', 'U', 'U', a, b, c)

        assert info == 0

        # Compute residual
        residual = x + a @ x @ b - c_orig
        residual_norm = np.linalg.norm(residual, 'fro')
        c_norm = np.linalg.norm(c_orig, 'fro')

        # Relative residual should be small
        assert residual_norm / c_norm < 1e-13


class TestSB04RDHessenberg:
    """Tests with Hessenberg matrices."""

    def test_a_schur_b_hessenberg(self):
        """
        A in Schur form, B in upper Hessenberg form.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4
        m = 3

        # A - upper triangular (Schur form)
        a = np.triu(np.random.randn(n, n).astype(float, order='F'))

        # B - upper Hessenberg (one subdiagonal)
        b = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)

        c_orig = np.random.randn(n, m).astype(float, order='F')
        c = c_orig.copy()

        x, info = sb04rd('A', 'U', 'U', a, b, c)

        assert info == 0

        # Verify equation
        residual = x + a @ x @ b - c_orig
        np.testing.assert_allclose(residual, np.zeros((n, m)), atol=1e-12)

    def test_a_hessenberg_b_schur(self):
        """
        A in upper Hessenberg form, B in Schur form.

        Random seed: 321 (for reproducibility)
        """
        np.random.seed(321)
        n = 4
        m = 3

        # A - upper Hessenberg
        a = np.triu(np.random.randn(n, n).astype(float, order='F'), k=-1)

        # B - upper triangular (Schur form)
        b = np.triu(np.random.randn(m, m).astype(float, order='F'))

        c_orig = np.random.randn(n, m).astype(float, order='F')
        c = c_orig.copy()

        x, info = sb04rd('B', 'U', 'U', a, b, c)

        assert info == 0

        # Verify equation
        residual = x + a @ x @ b - c_orig
        np.testing.assert_allclose(residual, np.zeros((n, m)), atol=1e-12)


class TestSB04RDEdgeCases:
    """Edge case tests."""

    def test_dimension_1x1(self):
        """Test with 1x1 matrices."""
        a = np.array([[2.0]], dtype=float, order='F')
        b = np.array([[3.0]], dtype=float, order='F')
        c = np.array([[10.0]], dtype=float, order='F')

        # Equation: x + 2*x*3 = 10 => x(1 + 6) = 10 => x = 10/7
        x_expected = 10.0 / 7.0

        x, info = sb04rd('S', 'U', 'U', a, b, c)

        assert info == 0
        np.testing.assert_allclose(x[0, 0], x_expected, rtol=1e-14)

    def test_zero_dimensions(self):
        """Test with n=0 or m=0 (quick return)."""
        a = np.zeros((0, 0), dtype=float, order='F')
        b = np.zeros((0, 0), dtype=float, order='F')
        c = np.zeros((0, 0), dtype=float, order='F')

        x, info = sb04rd('S', 'U', 'U', a, b, c)

        assert info == 0
        assert x.shape == (0, 0)


class TestSB04RDErrorHandling:
    """Error handling tests."""

    def test_invalid_abschu(self):
        """Test with invalid ABSCHU parameter."""
        a = np.eye(2, dtype=float, order='F')
        b = np.eye(2, dtype=float, order='F')
        c = np.ones((2, 2), dtype=float, order='F')

        with pytest.raises(ValueError):
            sb04rd('X', 'U', 'U', a, b, c)

    def test_invalid_ula(self):
        """Test with invalid ULA parameter."""
        a = np.eye(2, dtype=float, order='F')
        b = np.eye(2, dtype=float, order='F')
        c = np.ones((2, 2), dtype=float, order='F')

        with pytest.raises(ValueError):
            sb04rd('S', 'X', 'U', a, b, c)

    def test_invalid_ulb(self):
        """Test with invalid ULB parameter."""
        a = np.eye(2, dtype=float, order='F')
        b = np.eye(2, dtype=float, order='F')
        c = np.ones((2, 2), dtype=float, order='F')

        with pytest.raises(ValueError):
            sb04rd('S', 'U', 'X', a, b, c)


class TestSB04RD2x2Blocks:
    """Tests with 2x2 diagonal blocks (complex conjugate eigenvalues)."""

    def test_schur_with_2x2_blocks(self):
        """
        Test with quasi-triangular Schur form containing 2x2 blocks.

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        n = 4

        # Create A with 2x2 block representing complex eigenvalues
        a = np.array([
            [1.0, 2.0, 0.5, 0.3],
            [-0.5, 1.0, 0.2, 0.1],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, -0.3, 2.0]
        ], dtype=float, order='F')

        # Create B with 2x2 block
        b = np.array([
            [0.5, 1.0, 0.1, 0.2],
            [-0.2, 0.5, 0.3, 0.1],
            [0.0, 0.0, 0.3, 0.5],
            [0.0, 0.0, -0.1, 0.3]
        ], dtype=float, order='F')

        c_orig = np.random.randn(n, n).astype(float, order='F')
        c = c_orig.copy()

        x, info = sb04rd('S', 'U', 'U', a, b, c)

        assert info == 0

        # Verify equation
        residual = x + a @ x @ b - c_orig
        np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-12)
