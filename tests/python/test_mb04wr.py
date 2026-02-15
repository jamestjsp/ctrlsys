"""
Tests for MB04WR: Generate orthogonal symplectic matrices U or V from symplectic reflectors.

MB04WR generates orthogonal symplectic matrices U or V, defined as products of
symplectic reflectors and Givens rotations, as returned by MB04TS or MB04TB.

The matrices U and V are returned in terms of their first N/2 rows:
    U = [[U1, U2], [-U2, U1]]
    V = [[V1, V2], [-V2, V1]]
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import mb04ts, mb04wr


class TestMB04WRBasic:
    """Basic functionality tests for mb04wr."""

    def test_job_u_trans_n_basic(self):
        """
        Test JOB='U', TRANS='N' case with data from MB04TS.

        For JOB='U', TRANS='N':
        - Q1 input: A_out contains FU reflector vectors (in columns)
        - Q2 input: Q_out contains HU reflector vectors (in columns) with tau on diagonal

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4
        ilo = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, n).astype(float, order='F')
        G = np.random.randn(n, n)
        G = (G + G.T).astype(float, order='F')
        Q = np.random.randn(n, n).astype(float, order='F')

        A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
            'N', 'N', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
        )
        assert info == 0

        q1 = A_out.copy()
        q2 = Q_out.copy()

        q1_out, q2_out, info = mb04wr('U', 'N', n, ilo, q1, q2, csl, taul)

        assert info == 0, f"Expected info=0, got {info}"
        assert q1_out.shape == (n, n), f"Expected Q1 shape {(n, n)}, got {q1_out.shape}"
        assert q2_out.shape == (n, n), f"Expected Q2 shape {(n, n)}, got {q2_out.shape}"

    def test_job_u_trans_t_basic(self):
        """
        Test JOB='U', TRANS='T' case.

        For JOB='U', TRANS='T':
        - Q1 input: A_out contains FU reflector vectors (in rows when TRANA='T')
        - Q2 input: Q_out contains HU reflector vectors (in columns) with tau on diagonal

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4
        ilo = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, n).astype(float, order='F')
        G = np.random.randn(n, n)
        G = (G + G.T).astype(float, order='F')
        Q = np.random.randn(n, n).astype(float, order='F')

        A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
            'T', 'N', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
        )
        assert info == 0

        q1 = A_out.copy()
        q2 = Q_out.copy()

        q1_out, q2_out, info = mb04wr('U', 'T', n, ilo, q1, q2, csl, taul)

        assert info == 0, f"Expected info=0, got {info}"
        assert q1_out.shape == (n, n)
        assert q2_out.shape == (n, n)

    def test_job_v_trans_n_basic(self):
        """
        Test JOB='V', TRANS='N' case.

        For JOB='V', TRANS='N':
        - Q1 input: B_out contains FV reflector vectors (in rows when TRANB='N')
        - Q2 input: Q_out contains HV reflector vectors (in rows)

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 4
        ilo = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, n).astype(float, order='F')
        G = np.random.randn(n, n)
        G = (G + G.T).astype(float, order='F')
        Q = np.random.randn(n, n).astype(float, order='F')

        A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
            'N', 'N', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
        )
        assert info == 0

        q1 = B_out.copy()
        q2 = Q_out.copy()

        q1_out, q2_out, info = mb04wr('V', 'N', n, ilo, q1, q2, csr, taur)

        assert info == 0, f"Expected info=0, got {info}"
        assert q1_out.shape == (n, n)
        assert q2_out.shape == (n, n)

    def test_job_v_trans_t_basic(self):
        """
        Test JOB='V', TRANS='T' case.

        For JOB='V', TRANS='T':
        - Q1 input: B_out contains FV reflector vectors (in columns when TRANB='T')
        - Q2 input: Q_out contains HV reflector vectors (in rows)

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4
        ilo = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, n).astype(float, order='F')
        G = np.random.randn(n, n)
        G = (G + G.T).astype(float, order='F')
        Q = np.random.randn(n, n).astype(float, order='F')

        A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
            'N', 'T', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
        )
        assert info == 0

        q1 = B_out.copy()
        q2 = Q_out.copy()

        q1_out, q2_out, info = mb04wr('V', 'T', n, ilo, q1, q2, csr, taur)

        assert info == 0, f"Expected info=0, got {info}"
        assert q1_out.shape == (n, n)
        assert q2_out.shape == (n, n)


class TestMB04WREdgeCases:
    """Edge case tests for mb04wr."""

    def test_n_equals_0(self):
        """Test with N=0 (quick return)."""
        n = 0
        ilo = 1
        q1 = np.zeros((0, 0), dtype=float, order='F')
        q2 = np.zeros((0, 0), dtype=float, order='F')
        cs = np.zeros(0, dtype=float)
        tau = np.zeros(0, dtype=float)

        q1_out, q2_out, info = mb04wr('U', 'N', n, ilo, q1, q2, cs, tau)

        assert info == 0

    def test_n_equals_1(self):
        """Test with N=1 (minimal case)."""
        np.random.seed(111)
        n = 1
        ilo = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, n).astype(float, order='F')
        G = np.random.randn(n, n)
        G = (G + G.T).astype(float, order='F')
        Q = np.random.randn(n, n).astype(float, order='F')

        A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
            'N', 'N', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
        )
        assert info == 0

        q1 = A_out.copy()
        q2 = Q_out.copy()

        q1_out, q2_out, info = mb04wr('U', 'N', n, ilo, q1, q2, csl, taul)

        assert info == 0


class TestMB04WRMathematicalProperties:
    """Mathematical property tests for mb04wr."""

    def test_orthogonality_job_u(self):
        """
        Verify orthogonality property U1^T U1 + U2^T U2 = I for JOB='U'.

        Random seed: 1234 (for reproducibility)
        """
        np.random.seed(1234)
        n = 4
        ilo = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, n).astype(float, order='F')
        G = np.random.randn(n, n)
        G = (G + G.T).astype(float, order='F')
        Q = np.random.randn(n, n).astype(float, order='F')

        A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
            'N', 'N', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
        )
        assert info == 0

        q1 = A_out.copy()
        q2 = Q_out.copy()

        u1, u2, info = mb04wr('U', 'N', n, ilo, q1, q2, csl, taul)
        assert info == 0

        result = u1.T @ u1 + u2.T @ u2
        assert_allclose(result, np.eye(n), rtol=1e-13, atol=1e-14,
                       err_msg="U1^T U1 + U2^T U2 should equal I")

    def test_symplectic_structure_job_u(self):
        """
        Verify full U preserves symplectic structure: U^T J U = J for JOB='U'.

        J = [[0, I], [-I, 0]] is the standard symplectic form.
        Random seed: 5678 (for reproducibility)
        """
        np.random.seed(5678)
        n = 3
        ilo = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, n).astype(float, order='F')
        G = np.random.randn(n, n)
        G = (G + G.T).astype(float, order='F')
        Q = np.random.randn(n, n).astype(float, order='F')

        A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
            'N', 'N', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
        )
        assert info == 0

        q1 = A_out.copy()
        q2 = Q_out.copy()

        u1, u2, info = mb04wr('U', 'N', n, ilo, q1, q2, csl, taul)
        assert info == 0

        u_full = np.block([
            [u1, u2],
            [-u2, u1]
        ])

        j_mat = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-np.eye(n), np.zeros((n, n))]
        ])

        result = u_full.T @ j_mat @ u_full
        assert_allclose(result, j_mat, rtol=1e-13, atol=1e-14,
                       err_msg="U should preserve symplectic structure")

    def test_isotropic_columns(self):
        """
        Verify that columns satisfy isotropic condition: U1^T U2 = U2^T U1.

        Random seed: 9012 (for reproducibility)
        """
        np.random.seed(9012)
        n = 4
        ilo = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, n).astype(float, order='F')
        G = np.random.randn(n, n)
        G = (G + G.T).astype(float, order='F')
        Q = np.random.randn(n, n).astype(float, order='F')

        A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
            'N', 'N', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
        )
        assert info == 0

        q1 = A_out.copy()
        q2 = Q_out.copy()

        u1, u2, info = mb04wr('U', 'N', n, ilo, q1, q2, csl, taul)
        assert info == 0

        u1_t_u2 = u1.T @ u2
        u2_t_u1 = u2.T @ u1
        assert_allclose(u1_t_u2, u2_t_u1, rtol=1e-13, atol=1e-14,
                       err_msg="Columns should be isotropic: U1^T U2 = U2^T U1")

    def test_full_orthogonality(self):
        """
        Verify full U matrix is orthogonal: U^T U = I.

        Random seed: 3456 (for reproducibility)
        """
        np.random.seed(3456)
        n = 4
        ilo = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, n).astype(float, order='F')
        G = np.random.randn(n, n)
        G = (G + G.T).astype(float, order='F')
        Q = np.random.randn(n, n).astype(float, order='F')

        A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
            'N', 'N', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
        )
        assert info == 0

        q1 = A_out.copy()
        q2 = Q_out.copy()

        u1, u2, info = mb04wr('U', 'N', n, ilo, q1, q2, csl, taul)
        assert info == 0

        u_full = np.block([
            [u1, u2],
            [-u2, u1]
        ])

        utu = u_full.T @ u_full
        assert_allclose(utu, np.eye(2 * n), rtol=1e-13, atol=1e-14,
                       err_msg="Full U should be orthogonal")


class TestMB04WRErrors:
    """Error handling tests for mb04wr."""

    def test_invalid_job(self):
        """Test that invalid JOB returns error."""
        n = 3
        ilo = 1
        q1 = np.zeros((n, n), dtype=float, order='F')
        q2 = np.zeros((n, n), dtype=float, order='F')
        cs = np.zeros(2 * n, dtype=float)
        tau = np.zeros(n, dtype=float)

        with pytest.raises(ValueError):
            mb04wr('X', 'N', n, ilo, q1, q2, cs, tau)

    def test_invalid_trans(self):
        """Test that invalid TRANS returns error."""
        n = 3
        ilo = 1
        q1 = np.zeros((n, n), dtype=float, order='F')
        q2 = np.zeros((n, n), dtype=float, order='F')
        cs = np.zeros(2 * n, dtype=float)
        tau = np.zeros(n, dtype=float)

        with pytest.raises(ValueError):
            mb04wr('U', 'X', n, ilo, q1, q2, cs, tau)

    def test_negative_n(self):
        """Test that negative N returns error."""
        n = -1
        ilo = 1
        q1 = np.zeros((1, 1), dtype=float, order='F')
        q2 = np.zeros((1, 1), dtype=float, order='F')
        cs = np.zeros(2, dtype=float)
        tau = np.zeros(1, dtype=float)

        with pytest.raises(ValueError):
            mb04wr('U', 'N', n, ilo, q1, q2, cs, tau)

    def test_ilo_out_of_range_low(self):
        """Test that ILO < 1 returns error."""
        n = 3
        ilo = 0
        q1 = np.zeros((n, n), dtype=float, order='F')
        q2 = np.zeros((n, n), dtype=float, order='F')
        cs = np.zeros(2 * n, dtype=float)
        tau = np.zeros(n, dtype=float)

        with pytest.raises(ValueError):
            mb04wr('U', 'N', n, ilo, q1, q2, cs, tau)

    def test_ilo_out_of_range_high(self):
        """Test that ILO > N returns error."""
        n = 3
        ilo = 5
        q1 = np.zeros((n, n), dtype=float, order='F')
        q2 = np.zeros((n, n), dtype=float, order='F')
        cs = np.zeros(2 * n, dtype=float)
        tau = np.zeros(n, dtype=float)

        with pytest.raises(ValueError):
            mb04wr('U', 'N', n, ilo, q1, q2, cs, tau)
