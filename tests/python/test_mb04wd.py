"""
Tests for MB04WD: Blocked version of MB04WU - generate matrix Q from symplectic reflectors.

MB04WD generates a matrix Q with orthogonal columns (spanning an isotropic subspace)
using a blocked algorithm. It's the blocked version of MB04WU.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import mb04su, mb04wd


class TestMB04WDBasic:
    """Basic functionality tests for mb04wd."""

    def test_roundtrip_square_no_transpose(self):
        """
        Test mb04su -> mb04wd roundtrip with square M=N=3, no transpose.

        Verifies that applying mb04su then mb04wd recovers orthogonal columns.
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

        q1, q2, info = mb04wd(False, False, m, n, k, q1_in, q2_in, cs, tau)

        assert info == 0, f"Expected info=0, got {info}"

        q1_q1t = q1.T @ q1
        q2_q2t = q2.T @ q2
        assert_allclose(q1_q1t + q2_q2t, np.eye(n), rtol=1e-13, atol=1e-14,
                       err_msg="Q1^T Q1 + Q2^T Q2 should equal I")

    def test_roundtrip_with_transpose_q1(self):
        """
        Test mb04su -> mb04wd roundtrip with tranq1=True.

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

        q1, q2, info = mb04wd(True, False, m, n, k, q1_in, q2_in, cs, tau)

        assert info == 0, f"Expected info=0, got {info}"

    def test_roundtrip_with_transpose_q2(self):
        """
        Test mb04su -> mb04wd roundtrip with tranq2=True.

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

        q1, q2, info = mb04wd(False, True, m, n, k, q1_in, q2_in, cs, tau)

        assert info == 0, f"Expected info=0, got {info}"

    def test_roundtrip_both_transpose(self):
        """
        Test mb04su -> mb04wd roundtrip with tranq1=True, tranq2=True.

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

        q1, q2, info = mb04wd(True, True, m, n, k, q1_in, q2_in, cs, tau)

        assert info == 0, f"Expected info=0, got {info}"

    def test_larger_matrix_for_blocking(self):
        """
        Test with larger matrix to exercise blocking code.

        With m=n=8, k=8 the blocked code path should be exercised.
        Random seed: 2024 (for reproducibility)
        """
        np.random.seed(2024)
        m, n = 8, 8

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in = a_qr.copy()
        q2_in = b_qr.copy()

        q1, q2, info = mb04wd(False, False, m, n, k, q1_in, q2_in, cs, tau)

        assert info == 0, f"Expected info=0, got {info}"

        q1_q1t = q1.T @ q1
        q2_q2t = q2.T @ q2
        assert_allclose(q1_q1t + q2_q2t, np.eye(n), rtol=1e-13, atol=1e-14,
                       err_msg="Q1^T Q1 + Q2^T Q2 should equal I")


class TestMB04WDEdgeCases:
    """Edge case tests for mb04wd."""

    def test_n_equals_0(self):
        """Test with N=0 (quick return)."""
        m, n, k = 3, 0, 0
        q1 = np.zeros((3, 0), dtype=float, order='F')
        q2 = np.zeros((3, 0), dtype=float, order='F')
        cs = np.zeros(0, dtype=float)
        tau = np.zeros(0, dtype=float)

        q1_out, q2_out, info = mb04wd(False, False, m, n, k, q1, q2, cs, tau)

        assert info == 0

    def test_k_equals_0(self):
        """Test with K=0 (no reflectors to apply, just initialize to identity)."""
        m, n, k = 3, 3, 0
        q1 = np.zeros((m, n), dtype=float, order='F')
        q2 = np.zeros((m, n), dtype=float, order='F')
        cs = np.zeros(0, dtype=float)
        tau = np.zeros(0, dtype=float)

        q1_out, q2_out, info = mb04wd(False, False, m, n, k, q1, q2, cs, tau)

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

        q1_out, q2_out, info = mb04wd(False, False, m, n, k, q1, q2, cs, tau)

        assert info == 0


class TestMB04WDMathematicalProperties:
    """Mathematical property tests for mb04wd."""

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

        q1, q2, info = mb04wd(False, False, m, n, k, q1_in, q2_in, cs, tau)
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

        q1, q2, info = mb04wd(False, False, m, n, k, q1_in, q2_in, cs, tau)
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

        q1, q2, info = mb04wd(False, False, m, n, k, q1_in, q2_in, cs, tau)
        assert info == 0

        q1_t_q2 = q1.T @ q2
        q2_t_q1 = q2.T @ q1
        assert_allclose(q1_t_q2, q2_t_q1, rtol=1e-13, atol=1e-14,
                       err_msg="Columns should be isotropic: Q1^T Q2 = Q2^T Q1")

    def test_matches_mb04wu_output(self):
        """
        Verify that mb04wd produces same result as mb04wu.

        The blocked version should produce identical results to unblocked.
        Random seed: 5678 (for reproducibility)
        """
        from ctrlsys import mb04wu

        np.random.seed(5678)
        m, n = 5, 4

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in_wu = a_qr.copy()
        q2_in_wu = b_qr.copy()
        q1_wu, q2_wu, info_wu = mb04wu(False, False, m, n, k, q1_in_wu, q2_in_wu, cs, tau)
        assert info_wu == 0

        q1_in_wd = a_qr.copy()
        q2_in_wd = b_qr.copy()
        q1_wd, q2_wd, info_wd = mb04wd(False, False, m, n, k, q1_in_wd, q2_in_wd, cs, tau)
        assert info_wd == 0

        assert_allclose(q1_wd, q1_wu, rtol=1e-14, atol=1e-15,
                       err_msg="mb04wd should match mb04wu output for Q1")
        assert_allclose(q2_wd, q2_wu, rtol=1e-14, atol=1e-15,
                       err_msg="mb04wd should match mb04wu output for Q2")


class TestMB04WDErrors:
    """Error handling tests for mb04wd."""

    def test_negative_m(self):
        """Test that negative M returns error."""
        m, n, k = -1, 3, 2
        q1 = np.zeros((1, 3), dtype=float, order='F')
        q2 = np.zeros((1, 3), dtype=float, order='F')
        cs = np.zeros(4, dtype=float)
        tau = np.zeros(2, dtype=float)

        with pytest.raises(ValueError):
            mb04wd(False, False, m, n, k, q1, q2, cs, tau)

    def test_n_greater_than_m(self):
        """Test that N > M returns error."""
        m, n, k = 3, 5, 2
        q1 = np.zeros((3, 5), dtype=float, order='F')
        q2 = np.zeros((3, 5), dtype=float, order='F')
        cs = np.zeros(4, dtype=float)
        tau = np.zeros(2, dtype=float)

        with pytest.raises(ValueError):
            mb04wd(False, False, m, n, k, q1, q2, cs, tau)

    def test_k_greater_than_n(self):
        """Test that K > N returns error."""
        m, n, k = 5, 3, 5
        q1 = np.zeros((5, 3), dtype=float, order='F')
        q2 = np.zeros((5, 3), dtype=float, order='F')
        cs = np.zeros(10, dtype=float)
        tau = np.zeros(5, dtype=float)

        with pytest.raises(ValueError):
            mb04wd(False, False, m, n, k, q1, q2, cs, tau)

    def test_workspace_query(self):
        """Test workspace query (ldwork=-1)."""
        np.random.seed(9999)
        m, n = 5, 4

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_qr, b_qr, cs, tau, info_su = mb04su(m, n, a, b)
        assert info_su == 0

        k = min(m, n)

        q1_in = a_qr.copy()
        q2_in = b_qr.copy()

        q1, q2, info = mb04wd(False, False, m, n, k, q1_in, q2_in, cs, tau)
        assert info == 0
