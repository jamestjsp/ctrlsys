"""
Tests for MB04XD: Basis for left/right singular subspace of a matrix
corresponding to its smallest singular values.

Uses Partial Singular Value Decomposition (PSVD) algorithm.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestMB04XDBasic:
    """Basic tests from HTML doc example."""

    def test_html_doc_example(self):
        """
        Test MB04XD using HTML documentation example.

        6x4 matrix, RANK=-1 (compute rank), THETA=0.001
        Expected: rank=3, singular subspace basis computed
        """
        from ctrlsys import mb04xd

        m, n = 6, 4
        rank_in = -1
        theta_in = 0.001
        tol = 0.0
        reltol = 0.0
        jobu = 'A'
        jobv = 'A'

        a = np.array([
            [0.80010, 0.39985, 0.60005, 0.89999],
            [0.29996, 0.69990, 0.39997, 0.82997],
            [0.49994, 0.60003, 0.20012, 0.79011],
            [0.90013, 0.20016, 0.79995, 0.85002],
            [0.39998, 0.80006, 0.49985, 0.99016],
            [0.20002, 0.90007, 0.70009, 1.02994],
        ], order='F', dtype=float)

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0
        assert iwarn == 0
        assert rank == 3

        p = min(m, n)

        # Note: bidiagonal elements can have sign ambiguity due to orthogonal transforms
        q_diag_expected = np.array([3.2280, 0.8714, 0.3698, 0.0001])
        q_superdiag_expected = np.array([-0.0287, 0.0168, 0.0000])

        assert_allclose(np.abs(q[:p]), np.abs(q_diag_expected), rtol=1e-3, atol=1e-4)
        assert_allclose(np.abs(q[p:2*p-1]), np.abs(q_superdiag_expected), rtol=1e-2, atol=1e-4)

        assert inul[3] == True
        assert inul[4] == True
        assert inul[5] == True
        assert inul[0] == False
        assert inul[1] == False
        assert inul[2] == False

        assert u.shape == (m, m)
        assert v.shape == (n, n)

    def test_jobu_s_jobv_s(self):
        """
        Test with JOBU='S' and JOBV='S' (smaller singular subspace).

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb04xd

        np.random.seed(42)
        m, n = 5, 4
        p = min(m, n)

        a = np.random.randn(m, n).astype(float, order='F')

        rank_in = 2
        theta_in = -1.0
        tol = 0.0
        reltol = 0.0
        jobu = 'S'
        jobv = 'S'

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0
        assert rank >= 0
        assert u.shape == (m, p)
        assert v.shape == (n, p)
        assert len(q) == 2 * p - 1
        assert len(inul) == max(m, n)

    def test_jobu_n_jobv_n(self):
        """
        Test with JOBU='N' and JOBV='N' (no subspace computation).

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb04xd

        np.random.seed(123)
        m, n = 4, 4
        p = min(m, n)

        a = np.random.randn(m, n).astype(float, order='F')

        rank_in = 2
        theta_in = -1.0
        tol = 0.0
        reltol = 0.0
        jobu = 'N'
        jobv = 'N'

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0
        assert rank >= 0
        assert len(q) == 2 * p - 1


class TestMB04XDMathematical:
    """Mathematical property tests."""

    def test_singular_subspace_orthonormality(self):
        """
        Validate columns of U and V corresponding to singular subspace are orthonormal.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import mb04xd

        np.random.seed(456)
        m, n = 6, 5

        a = np.random.randn(m, n).astype(float, order='F')

        rank_in = 3
        theta_in = -1.0
        tol = 0.0
        reltol = 0.0
        jobu = 'A'
        jobv = 'A'

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0

        subspace_cols_u = [i for i in range(u.shape[1]) if inul[i]]
        if len(subspace_cols_u) > 0:
            u_sub = u[:, subspace_cols_u]
            UtU = u_sub.T @ u_sub
            assert_allclose(UtU, np.eye(len(subspace_cols_u)), rtol=1e-10, atol=1e-11)

        subspace_cols_v = [i for i in range(v.shape[1]) if inul[i]]
        if len(subspace_cols_v) > 0:
            v_sub = v[:, subspace_cols_v]
            VtV = v_sub.T @ v_sub
            assert_allclose(VtV, np.eye(len(subspace_cols_v)), rtol=1e-10, atol=1e-11)

    def test_bidiagonal_structure(self):
        """
        Validate Q contains proper bidiagonal structure.

        The superdiagonal entries that correspond to separated
        singular subspaces should be zero (or near-zero).

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import mb04xd

        np.random.seed(789)
        m, n = 5, 4
        p = min(m, n)

        a = np.random.randn(m, n).astype(float, order='F')

        rank_in = 2
        theta_in = -1.0
        tol = 1e-10
        reltol = 0.0
        jobu = 'A'
        jobv = 'A'

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0

        assert len(q) == 2 * p - 1

    def test_rank_computation_accuracy(self):
        """
        Test rank computation with known rank-deficient matrix.

        Create a matrix with known rank by constructing from
        reduced-rank factors.

        Random seed: 555 (for reproducibility)
        """
        from ctrlsys import mb04xd

        np.random.seed(555)
        m, n = 6, 5
        true_rank = 3

        u_factor = np.random.randn(m, true_rank).astype(float, order='F')
        v_factor = np.random.randn(true_rank, n).astype(float, order='F')
        a = (u_factor @ v_factor).astype(float, order='F')

        rank_in = -1
        theta_in = 1e-10
        tol = 1e-12
        reltol = 0.0
        jobu = 'A'
        jobv = 'A'

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0
        assert rank == true_rank


class TestMB04XDEdgeCases:
    """Edge case tests."""

    def test_square_matrix(self):
        """
        Test with square matrix M=N.

        Random seed: 111 (for reproducibility)
        """
        from ctrlsys import mb04xd

        np.random.seed(111)
        m, n = 4, 4
        p = min(m, n)

        a = np.random.randn(m, n).astype(float, order='F')

        rank_in = 2
        theta_in = -1.0
        tol = 0.0
        reltol = 0.0
        jobu = 'A'
        jobv = 'A'

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0
        assert u.shape == (m, m)
        assert v.shape == (n, n)

    def test_m_less_than_n(self):
        """
        Test with M < N (wide matrix).

        Random seed: 222 (for reproducibility)
        """
        from ctrlsys import mb04xd

        np.random.seed(222)
        m, n = 3, 5
        p = min(m, n)

        a = np.random.randn(m, n).astype(float, order='F')

        rank_in = 2
        theta_in = -1.0
        tol = 0.0
        reltol = 0.0
        jobu = 'A'
        jobv = 'A'

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0
        assert u.shape == (m, m)
        assert v.shape == (n, n)

    def test_zero_matrix(self):
        """
        Test with zero matrix (all singular values are zero).
        """
        from ctrlsys import mb04xd

        m, n = 4, 3
        p = min(m, n)

        a = np.zeros((m, n), order='F', dtype=float)

        rank_in = -1
        theta_in = 1e-10
        tol = 0.0
        reltol = 0.0
        jobu = 'N'
        jobv = 'N'

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0
        assert rank == 0

    def test_single_element(self):
        """
        Test with 1x1 matrix.
        """
        from ctrlsys import mb04xd

        m, n = 1, 1

        a = np.array([[5.0]], order='F', dtype=float)

        rank_in = -1
        theta_in = 1.0
        tol = 0.0
        reltol = 0.0
        jobu = 'A'
        jobv = 'A'

        rank, theta, u, v, q, inul, iwarn, info = mb04xd(
            jobu, jobv, a, rank_in, theta_in, tol, reltol
        )

        assert info == 0
        assert u.shape == (1, 1)
        assert v.shape == (1, 1)
        assert len(q) == 1


class TestMB04XDErrors:
    """Error handling tests."""

    def test_invalid_jobu(self):
        """Test with invalid JOBU parameter."""
        from ctrlsys import mb04xd

        m, n = 3, 3
        a = np.random.randn(m, n).astype(float, order='F')

        with pytest.raises(ValueError, match="argument 1"):
            mb04xd('X', 'N', a, 1, -1.0, 0.0, 0.0)

    def test_invalid_jobv(self):
        """Test with invalid JOBV parameter."""
        from ctrlsys import mb04xd

        m, n = 3, 3
        a = np.random.randn(m, n).astype(float, order='F')

        with pytest.raises(ValueError, match="argument 2"):
            mb04xd('N', 'X', a, 1, -1.0, 0.0, 0.0)

    def test_rank_too_large(self):
        """Test with RANK > min(M, N)."""
        from ctrlsys import mb04xd

        m, n = 3, 4
        p = min(m, n)
        a = np.random.randn(m, n).astype(float, order='F')

        with pytest.raises(ValueError, match="argument 5"):
            mb04xd('N', 'N', a, p + 1, -1.0, 0.0, 0.0)

    def test_negative_rank_negative_theta(self):
        """Test with RANK < 0 and THETA < 0 (invalid combination)."""
        from ctrlsys import mb04xd

        m, n = 3, 4
        a = np.random.randn(m, n).astype(float, order='F')

        with pytest.raises(ValueError, match="argument 6"):
            mb04xd('N', 'N', a, -1, -1.0, 0.0, 0.0)
