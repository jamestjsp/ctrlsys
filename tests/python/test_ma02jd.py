"""
Tests for MA02JD: Test if a matrix is an orthogonal symplectic matrix.

Computes || Q^T Q - I ||_F for a matrix of the form:
    Q = [  op(Q1)   op(Q2) ]
        [ -op(Q2)   op(Q1) ]

where Q1 and Q2 are N-by-N matrices and op(X) = X or X^T.
"""
import numpy as np
import pytest
from ctrlsys import ma02jd


class TestMA02JDBasic:
    """Basic functionality tests for ma02jd."""

    def test_identity_residual_zero(self):
        """
        For Q1=I, Q2=0 with ltran1=False, ltran2=False:
        Q = [I, 0; 0, I] which is orthogonal, residual should be zero.

        Random seed: N/A (deterministic input)
        """
        n = 3
        q1 = np.eye(n, order='F', dtype=float)
        q2 = np.zeros((n, n), order='F', dtype=float)

        residual = ma02jd(False, False, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_orthogonal_symplectic_residual_zero(self):
        """
        Test with a proper orthogonal symplectic matrix.

        For orthogonal symplectic Q, we have Q^T Q = I, so residual = 0.
        Use Givens rotation structure: Q1 = c*I, Q2 = s*I with c^2 + s^2 = 1.

        Random seed: N/A (deterministic input)
        """
        n = 2
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)

        q1 = c * np.eye(n, order='F', dtype=float)
        q2 = s * np.eye(n, order='F', dtype=float)

        residual = ma02jd(False, False, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_non_orthogonal_has_residual(self):
        """
        For non-orthogonal Q, residual should be nonzero.

        Q1 = 2*I, Q2 = 0 gives Q = [2I, 0; 0, 2I], so Q^T Q = 4I != I.

        Random seed: N/A (deterministic input)
        """
        n = 2
        q1 = 2.0 * np.eye(n, order='F', dtype=float)
        q2 = np.zeros((n, n), order='F', dtype=float)

        residual = ma02jd(False, False, q1, q2)

        assert residual > 0.0

    def test_n_equals_zero(self):
        """Test edge case with n=0."""
        q1 = np.array([], dtype=float, order='F').reshape((0, 0))
        q2 = np.array([], dtype=float, order='F').reshape((0, 0))

        residual = ma02jd(False, False, q1, q2)

        assert residual == 0.0


class TestMA02JDTranspose:
    """Test transpose options (ltran1, ltran2)."""

    def test_ltran1_true(self):
        """Test with ltran1=True (op(Q1) = Q1^T)."""
        n = 2
        c = np.cos(np.pi / 6)
        s = np.sin(np.pi / 6)

        q1 = c * np.eye(n, order='F', dtype=float)
        q2 = s * np.eye(n, order='F', dtype=float)

        residual = ma02jd(True, False, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_ltran2_true(self):
        """Test with ltran2=True (op(Q2) = Q2^T)."""
        n = 2
        c = np.cos(np.pi / 3)
        s = np.sin(np.pi / 3)

        q1 = c * np.eye(n, order='F', dtype=float)
        q2 = s * np.eye(n, order='F', dtype=float)

        residual = ma02jd(False, True, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_both_transpose_true(self):
        """Test with both ltran1=True and ltran2=True."""
        n = 3
        q1 = np.eye(n, order='F', dtype=float)
        q2 = np.zeros((n, n), order='F', dtype=float)

        residual = ma02jd(True, True, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)


class TestMA02JDMathematicalProperty:
    """Mathematical property tests for orthogonal symplectic residual."""

    def test_residual_formula_verification(self):
        """
        Verify residual computation matches manual calculation.

        For Q = [op(Q1), op(Q2); -op(Q2), op(Q1)]:
        Q^T Q = [op(Q1)^T, -op(Q2)^T] [op(Q1),  op(Q2) ]
                [op(Q2)^T,  op(Q1)^T] [-op(Q2), op(Q1)]

              = [op(Q1)^T op(Q1) + op(Q2)^T op(Q2),  op(Q1)^T op(Q2) - op(Q2)^T op(Q1)]
                [op(Q2)^T op(Q1) - op(Q1)^T op(Q2),  op(Q2)^T op(Q2) + op(Q1)^T op(Q1)]

        For Q orthogonal: both diagonal blocks = I, off-diag = 0.

        Random seed: 42
        """
        np.random.seed(42)
        n = 3

        q1 = np.random.randn(n, n).astype(float, order='F')
        q2 = np.random.randn(n, n).astype(float, order='F')

        residual = ma02jd(False, False, q1, q2)

        blk1 = q1.T @ q1 + q2.T @ q2 - np.eye(n)
        blk2 = q1.T @ q2 - q2.T @ q1
        expected = np.sqrt(2) * np.sqrt(np.linalg.norm(blk1, 'fro')**2 +
                                         np.linalg.norm(blk2, 'fro')**2)

        np.testing.assert_allclose(residual, expected, rtol=1e-14)

    def test_residual_formula_with_ltran1(self):
        """
        Verify residual formula with ltran1=True.

        Random seed: 123
        """
        np.random.seed(123)
        n = 2

        q1_stored = np.random.randn(n, n).astype(float, order='F')
        q2_stored = np.random.randn(n, n).astype(float, order='F')

        q1 = q1_stored.T
        q2 = q2_stored

        residual = ma02jd(True, False, q1_stored, q2_stored)

        blk1 = q1.T @ q1 + q2.T @ q2 - np.eye(n)
        blk2 = q1.T @ q2 - q2.T @ q1
        expected = np.sqrt(2) * np.sqrt(np.linalg.norm(blk1, 'fro')**2 +
                                         np.linalg.norm(blk2, 'fro')**2)

        np.testing.assert_allclose(residual, expected, rtol=1e-14)

    def test_residual_formula_with_ltran2(self):
        """
        Verify residual formula with ltran2=True.

        Random seed: 456
        """
        np.random.seed(456)
        n = 2

        q1_stored = np.random.randn(n, n).astype(float, order='F')
        q2_stored = np.random.randn(n, n).astype(float, order='F')

        q1 = q1_stored
        q2 = q2_stored.T

        residual = ma02jd(False, True, q1_stored, q2_stored)

        blk1 = q1.T @ q1 + q2.T @ q2 - np.eye(n)
        blk2 = q1.T @ q2 - q2.T @ q1
        expected = np.sqrt(2) * np.sqrt(np.linalg.norm(blk1, 'fro')**2 +
                                         np.linalg.norm(blk2, 'fro')**2)

        np.testing.assert_allclose(residual, expected, rtol=1e-14)

    def test_residual_formula_with_both_transpose(self):
        """
        Verify residual formula with both transposes.

        Random seed: 789
        """
        np.random.seed(789)
        n = 3

        q1_stored = np.random.randn(n, n).astype(float, order='F')
        q2_stored = np.random.randn(n, n).astype(float, order='F')

        q1 = q1_stored.T
        q2 = q2_stored.T

        residual = ma02jd(True, True, q1_stored, q2_stored)

        blk1 = q1.T @ q1 + q2.T @ q2 - np.eye(n)
        blk2 = q1.T @ q2 - q2.T @ q1
        expected = np.sqrt(2) * np.sqrt(np.linalg.norm(blk1, 'fro')**2 +
                                         np.linalg.norm(blk2, 'fro')**2)

        np.testing.assert_allclose(residual, expected, rtol=1e-14)

    def test_residual_nonnegative(self):
        """
        Mathematical property: residual must be non-negative.

        Random seed: 888
        """
        np.random.seed(888)
        n = 4

        for _ in range(10):
            q1 = np.random.randn(n, n).astype(float, order='F')
            q2 = np.random.randn(n, n).astype(float, order='F')

            residual = ma02jd(False, False, q1, q2)

            assert residual >= 0.0
