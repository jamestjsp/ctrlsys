"""
Tests for MA02JZ: Test if a complex matrix is a unitary symplectic matrix.

Computes || Q^H Q - I ||_F for a complex matrix of the form:
    Q = [  op(Q1)   op(Q2) ]
        [ -op(Q2)   op(Q1) ]

where Q1 and Q2 are N-by-N complex matrices and op(X) = X or X'.
This is the complex version of MA02JD.
"""
import numpy as np
import pytest
from ctrlsys import ma02jz


class TestMA02JZBasic:
    """Basic functionality tests for ma02jz."""

    def test_identity_residual_zero(self):
        """
        For Q1=I, Q2=0 with ltran1=False, ltran2=False:
        Q = [I, 0; 0, I] which is unitary, residual should be zero.

        Random seed: N/A (deterministic input)
        """
        n = 3
        q1 = np.eye(n, order='F', dtype=complex)
        q2 = np.zeros((n, n), order='F', dtype=complex)

        residual = ma02jz(False, False, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_unitary_symplectic_residual_zero(self):
        """
        Test with a proper unitary symplectic matrix.

        For unitary symplectic Q, we have Q^H Q = I, so residual = 0.
        Use Givens rotation structure: Q1 = c*I, Q2 = s*I with |c|^2 + |s|^2 = 1.

        Random seed: N/A (deterministic input)
        """
        n = 2
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)

        q1 = c * np.eye(n, order='F', dtype=complex)
        q2 = s * np.eye(n, order='F', dtype=complex)

        residual = ma02jz(False, False, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_complex_unitary_symplectic(self):
        """
        Test with complex entries that form a unitary symplectic matrix.

        For unitary symplectic: Q1^H Q1 + Q2^H Q2 = I and Q1^H Q2 = Q2^H Q1.
        Use real matrices scaled appropriately: Q1 = c*I, Q2 = s*I with c^2+s^2=1.
        Works because for real diagonal: Q1^H = Q1^T = Q1.

        Random seed: N/A (deterministic input)
        """
        n = 2
        c = 0.6
        s = 0.8

        q1 = c * np.eye(n, order='F', dtype=complex)
        q2 = s * np.eye(n, order='F', dtype=complex)

        residual = ma02jz(False, False, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_non_unitary_has_residual(self):
        """
        For non-unitary Q, residual should be nonzero.

        Q1 = 2*I, Q2 = 0 gives Q = [2I, 0; 0, 2I], so Q^H Q = 4I != I.

        Random seed: N/A (deterministic input)
        """
        n = 2
        q1 = 2.0 * np.eye(n, order='F', dtype=complex)
        q2 = np.zeros((n, n), order='F', dtype=complex)

        residual = ma02jz(False, False, q1, q2)

        assert residual > 0.0

    def test_n_equals_zero(self):
        """Test edge case with n=0."""
        q1 = np.array([], dtype=complex, order='F').reshape((0, 0))
        q2 = np.array([], dtype=complex, order='F').reshape((0, 0))

        residual = ma02jz(False, False, q1, q2)

        assert residual == 0.0


class TestMA02JZTranspose:
    """Test transpose options (ltran1, ltran2)."""

    def test_ltran1_true(self):
        """Test with ltran1=True (op(Q1) = Q1')."""
        n = 2
        c = np.cos(np.pi / 6)
        s = np.sin(np.pi / 6)

        q1 = c * np.eye(n, order='F', dtype=complex)
        q2 = s * np.eye(n, order='F', dtype=complex)

        residual = ma02jz(True, False, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_ltran2_true(self):
        """Test with ltran2=True (op(Q2) = Q2')."""
        n = 2
        c = np.cos(np.pi / 3)
        s = np.sin(np.pi / 3)

        q1 = c * np.eye(n, order='F', dtype=complex)
        q2 = s * np.eye(n, order='F', dtype=complex)

        residual = ma02jz(False, True, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_both_transpose_true(self):
        """Test with both ltran1=True and ltran2=True."""
        n = 3
        q1 = np.eye(n, order='F', dtype=complex)
        q2 = np.zeros((n, n), order='F', dtype=complex)

        residual = ma02jz(True, True, q1, q2)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)


class TestMA02JZMathematicalProperty:
    """Mathematical property tests for unitary symplectic residual."""

    def test_residual_formula_verification(self):
        """
        Verify residual computation matches manual calculation.

        For Q = [op(Q1), op(Q2); -op(Q2), op(Q1)]:
        Q^H Q = [op(Q1)^H, -op(Q2)^H] [op(Q1),  op(Q2) ]
                [op(Q2)^H,  op(Q1)^H] [-op(Q2), op(Q1)]

              = [op(Q1)^H op(Q1) + op(Q2)^H op(Q2),  op(Q1)^H op(Q2) - op(Q2)^H op(Q1)]
                [op(Q2)^H op(Q1) - op(Q1)^H op(Q2),  op(Q2)^H op(Q2) + op(Q1)^H op(Q1)]

        For Q unitary: both diagonal blocks = I, off-diag = 0.

        Random seed: 42
        """
        np.random.seed(42)
        n = 3

        q1 = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        q2 = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')

        residual = ma02jz(False, False, q1, q2)

        blk1 = q1.conj().T @ q1 + q2.conj().T @ q2 - np.eye(n)
        blk2 = q1.conj().T @ q2 - q2.conj().T @ q1
        expected = np.sqrt(2) * np.sqrt(np.linalg.norm(blk1, 'fro')**2 +
                                         np.linalg.norm(blk2, 'fro')**2)

        np.testing.assert_allclose(residual, expected, rtol=1e-14)

    def test_residual_formula_with_ltran1(self):
        """
        Verify residual formula with ltran1=True.

        When ltran1=True, op(Q1) = Q1_stored' (conjugate transpose).
        So Q has Q1_stored' in position (1,1) and (2,2).

        Random seed: 123
        """
        np.random.seed(123)
        n = 2

        q1_stored = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        q2_stored = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')

        q1 = q1_stored.conj().T
        q2 = q2_stored

        residual = ma02jz(True, False, q1_stored, q2_stored)

        blk1 = q1.conj().T @ q1 + q2.conj().T @ q2 - np.eye(n)
        blk2 = q1.conj().T @ q2 - q2.conj().T @ q1
        expected = np.sqrt(2) * np.sqrt(np.linalg.norm(blk1, 'fro')**2 +
                                         np.linalg.norm(blk2, 'fro')**2)

        np.testing.assert_allclose(residual, expected, rtol=1e-14)

    def test_residual_formula_with_ltran2(self):
        """
        Verify residual formula with ltran2=True.

        When ltran2=True, op(Q2) = Q2_stored' (conjugate transpose).

        Random seed: 456
        """
        np.random.seed(456)
        n = 2

        q1_stored = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        q2_stored = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')

        q1 = q1_stored
        q2 = q2_stored.conj().T

        residual = ma02jz(False, True, q1_stored, q2_stored)

        blk1 = q1.conj().T @ q1 + q2.conj().T @ q2 - np.eye(n)
        blk2 = q1.conj().T @ q2 - q2.conj().T @ q1
        expected = np.sqrt(2) * np.sqrt(np.linalg.norm(blk1, 'fro')**2 +
                                         np.linalg.norm(blk2, 'fro')**2)

        np.testing.assert_allclose(residual, expected, rtol=1e-14)

    def test_residual_formula_with_both_transpose(self):
        """
        Verify residual formula with both transposes.

        When ltran1=True and ltran2=True, op(Q1) = Q1_stored' and op(Q2) = Q2_stored'.

        Random seed: 789
        """
        np.random.seed(789)
        n = 3

        q1_stored = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        q2_stored = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')

        q1 = q1_stored.conj().T
        q2 = q2_stored.conj().T

        residual = ma02jz(True, True, q1_stored, q2_stored)

        blk1 = q1.conj().T @ q1 + q2.conj().T @ q2 - np.eye(n)
        blk2 = q1.conj().T @ q2 - q2.conj().T @ q1
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
            q1 = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
            q2 = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')

            residual = ma02jz(False, False, q1, q2)

            assert residual >= 0.0

    def test_real_inputs_match_ma02jd_behavior(self):
        """
        For real inputs, MA02JZ should compute the same residual as MA02JD.

        Random seed: 999
        """
        np.random.seed(999)
        n = 3

        q1_real = np.random.randn(n, n).astype(float, order='F')
        q2_real = np.random.randn(n, n).astype(float, order='F')

        q1_complex = q1_real.astype(complex, order='F')
        q2_complex = q2_real.astype(complex, order='F')

        from ctrlsys import ma02jd
        residual_real = ma02jd(False, False, q1_real, q2_real)
        residual_complex = ma02jz(False, False, q1_complex, q2_complex)

        np.testing.assert_allclose(residual_complex, residual_real, rtol=1e-14)
