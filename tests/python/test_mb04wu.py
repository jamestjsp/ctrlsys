"""
Tests for MB04WU: Generate matrix Q with orthogonal columns from symplectic reflectors and Givens rotations.

MB04WU is the inverse of MB04SU. It generates the orthogonal symplectic matrix Q
from the factored representation produced by MB04SU.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import mb04su, mb04wu


class TestMB04WUBasic:
    """Basic functionality tests for mb04wu."""

    def test_roundtrip_square_no_transpose(self):
        """
        Test mb04su -> mb04wu roundtrip with square M=N=3, no transpose.

        Verifies that applying mb04su then mb04wu recovers orthogonal columns.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m, n = 3, 3

        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0]
        ], dtype=float, order='F')

        b = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], dtype=float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in = a_qr.copy()
        q2_in = b_qr.copy()

        q1, q2, info = mb04wu(False, False, m, n, k, q1_in, q2_in, cs, tau)

        assert info == 0, f"Expected info=0, got {info}"

        q1_q1t = q1.T @ q1
        q2_q2t = q2.T @ q2
        assert_allclose(q1_q1t + q2_q2t, np.eye(n), rtol=1e-13, atol=1e-14,
                       err_msg="Q1^T Q1 + Q2^T Q2 should equal I")

    def test_roundtrip_with_transpose_q1(self):
        """
        Test mb04su -> mb04wu roundtrip with tranq1=True.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m, n = 4, 3

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in = a_qr.T.copy(order='F')
        q2_in = b_qr.copy()

        q1, q2, info = mb04wu(True, False, m, n, k, q1_in, q2_in, cs, tau)

        assert info == 0, f"Expected info=0, got {info}"

    def test_roundtrip_with_transpose_q2(self):
        """
        Test mb04su -> mb04wu roundtrip with tranq2=True.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m, n = 4, 3

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in = a_qr.copy()
        q2_in = b_qr.T.copy(order='F')

        q1, q2, info = mb04wu(False, True, m, n, k, q1_in, q2_in, cs, tau)

        assert info == 0, f"Expected info=0, got {info}"

    def test_roundtrip_both_transpose(self):
        """
        Test mb04su -> mb04wu roundtrip with tranq1=True, tranq2=True.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m, n = 4, 3

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in = a_qr.T.copy(order='F')
        q2_in = b_qr.T.copy(order='F')

        q1, q2, info = mb04wu(True, True, m, n, k, q1_in, q2_in, cs, tau)

        assert info == 0, f"Expected info=0, got {info}"


class TestMB04WUEdgeCases:
    """Edge case tests for mb04wu."""

    def test_n_equals_0(self):
        """Test with N=0 (quick return)."""
        m, n, k = 3, 0, 0
        q1 = np.zeros((3, 0), dtype=float, order='F')
        q2 = np.zeros((3, 0), dtype=float, order='F')
        cs = np.zeros(0, dtype=float)
        tau = np.zeros(0, dtype=float)

        q1_out, q2_out, info = mb04wu(False, False, m, n, k, q1, q2, cs, tau)

        assert info == 0

    def test_k_equals_0(self):
        """Test with K=0 (no reflectors to apply, just initialize to identity)."""
        m, n, k = 3, 3, 0
        q1 = np.zeros((m, n), dtype=float, order='F')
        q2 = np.zeros((m, n), dtype=float, order='F')
        cs = np.zeros(0, dtype=float)
        tau = np.zeros(0, dtype=float)

        q1_out, q2_out, info = mb04wu(False, False, m, n, k, q1, q2, cs, tau)

        assert info == 0

        assert_allclose(q1_out, np.eye(n, dtype=float), rtol=1e-14,
                       err_msg="With K=0, Q1 should be identity")
        assert_allclose(q2_out, np.zeros((m, n), dtype=float), rtol=1e-14,
                       err_msg="With K=0, Q2 should be zero")

    def test_m_equals_n_equals_k_equals_1(self):
        """Test with M=N=K=1 (single element case)."""
        m, n, k = 1, 1, 1

        q1 = np.array([[1.0]], dtype=float, order='F')
        q2 = np.array([[0.5]], dtype=float, order='F')
        cs = np.array([0.8, 0.6], dtype=float)
        tau = np.array([0.5], dtype=float)

        q1_out, q2_out, info = mb04wu(False, False, m, n, k, q1, q2, cs, tau)

        assert info == 0


class TestMB04WUMathematicalProperties:
    """Mathematical property tests for mb04wu."""

    def test_orthogonality_full_symplectic(self):
        """
        Verify Q^T Q = I for the full symplectic matrix.

        The full Q = [[Q1, Q2], [-Q2, Q1]] should satisfy Q^T Q = I.
        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        m, n = 4, 4

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in = a_qr.copy()
        q2_in = b_qr.copy()

        q1, q2, info = mb04wu(False, False, m, n, k, q1_in, q2_in, cs, tau)
        assert info == 0

        q_full_top = np.hstack([q1, q2])
        q_full_bot = np.hstack([-q2, q1])
        q_full = np.vstack([q_full_top, q_full_bot])

        qtq = q_full.T @ q_full
        assert_allclose(qtq, np.eye(2*n), rtol=1e-13, atol=1e-14,
                       err_msg="Full symplectic Q should be orthogonal")

    def test_symplectic_structure(self):
        """
        Verify Q preserves symplectic structure: Q^T J Q = J.

        J = [[0, I], [-I, 0]] is the standard symplectic form.
        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        m, n = 3, 3

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in = a_qr.copy()
        q2_in = b_qr.copy()

        q1, q2, info = mb04wu(False, False, m, n, k, q1_in, q2_in, cs, tau)
        assert info == 0

        q_full_top = np.hstack([q1, q2])
        q_full_bot = np.hstack([-q2, q1])
        q_full = np.vstack([q_full_top, q_full_bot])

        j_top = np.hstack([np.zeros((n, n)), np.eye(n)])
        j_bot = np.hstack([-np.eye(n), np.zeros((n, n))])
        j_mat = np.vstack([j_top, j_bot])

        result = q_full.T @ j_mat @ q_full
        assert_allclose(result, j_mat, rtol=1e-13, atol=1e-14,
                       err_msg="Q should preserve symplectic structure")

    def test_isotropic_columns(self):
        """
        Verify that columns of [Q1; Q2] are isotropic.

        Isotropic means: for columns q_i, q_j, we have q_i^T J q_j = 0.
        This is equivalent to Q1^T Q2 = Q2^T Q1.
        Random seed: 1234 (for reproducibility)
        """
        np.random.seed(1234)
        m, n = 4, 3

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in = a_qr.copy()
        q2_in = b_qr.copy()

        q1, q2, info = mb04wu(False, False, m, n, k, q1_in, q2_in, cs, tau)
        assert info == 0

        q1_t_q2 = q1.T @ q2
        q2_t_q1 = q2.T @ q1
        assert_allclose(q1_t_q2, q2_t_q1, rtol=1e-13, atol=1e-14,
                       err_msg="Columns should be isotropic: Q1^T Q2 = Q2^T Q1")


class TestMB04WUErrors:
    """Error handling tests for mb04wu."""

    def test_negative_m(self):
        """Test that negative M returns error."""
        m, n, k = -1, 3, 2
        q1 = np.zeros((1, 3), dtype=float, order='F')
        q2 = np.zeros((1, 3), dtype=float, order='F')
        cs = np.zeros(4, dtype=float)
        tau = np.zeros(2, dtype=float)

        with pytest.raises(ValueError):
            mb04wu(False, False, m, n, k, q1, q2, cs, tau)

    def test_n_greater_than_m(self):
        """Test that N > M returns error."""
        m, n, k = 3, 5, 2
        q1 = np.zeros((3, 5), dtype=float, order='F')
        q2 = np.zeros((3, 5), dtype=float, order='F')
        cs = np.zeros(4, dtype=float)
        tau = np.zeros(2, dtype=float)

        with pytest.raises(ValueError):
            mb04wu(False, False, m, n, k, q1, q2, cs, tau)

    def test_k_greater_than_n(self):
        """Test that K > N returns error."""
        m, n, k = 5, 3, 5
        q1 = np.zeros((5, 3), dtype=float, order='F')
        q2 = np.zeros((5, 3), dtype=float, order='F')
        cs = np.zeros(10, dtype=float)
        tau = np.zeros(5, dtype=float)

        with pytest.raises(ValueError):
            mb04wu(False, False, m, n, k, q1, q2, cs, tau)
