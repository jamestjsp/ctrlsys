"""
Tests for MB04YD: Partial diagonalization of a bidiagonal matrix.

MB04YD partially diagonalizes a bidiagonal matrix using QR or QL iterations
in such a way that it is split into unreduced bidiagonal submatrices whose
singular values are either all larger than a given bound or all smaller than
(or equal to) this bound.
"""

import numpy as np
import pytest

from ctrlsys import mb04yd


class TestMB04YDBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_html_doc_example_compute_rank(self):
        """
        Test from SLICOT HTML documentation example.

        Given bidiagonal matrix with:
        - Q = [1, 2, 3, 4, 5] (diagonal)
        - E = [2, 3, 4, 5] (superdiagonal)
        - THETA = 2.0
        - RANK = -1 (compute rank)

        Expected output:
        - Q_out = [0.4045, 1.9839, 3.4815, 5.3723, 7.9948]
        - E_out = [0.0, 0.0, 0.0128, 0.0273]
        - RANK_out = 3 (3 singular values > 2.0)
        """
        m = 5
        n = 5
        p = min(m, n)

        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
        e = np.array([2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
        theta = 2.0
        rank = -1
        tol = 0.0
        reltol = 0.0

        q_out, e_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'N', m, n, rank, theta, q, e, tol, reltol
        )

        assert info == 0
        assert iwarn == 0
        assert rank_out == 3

        # Note: bidiagonal elements can have sign ambiguity due to orthogonal transforms
        q_expected = np.array([0.4045, 1.9839, 3.4815, 5.3723, 7.9948])
        e_expected = np.array([0.0, 0.0, 0.0128, 0.0273])

        np.testing.assert_allclose(np.abs(q_out), np.abs(q_expected), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(np.abs(e_out), np.abs(e_expected), rtol=1e-3, atol=1e-4)

    def test_with_u_identity(self):
        """
        Test with U initialized to identity.

        Random seed: 42 (for reproducibility)
        Validates that U matrix is orthogonal after transformation.
        """
        np.random.seed(42)
        m = 4
        n = 4
        p = min(m, n)

        q = np.array([3.0, 2.0, 1.5, 0.5], dtype=float, order='F')
        e = np.array([0.5, 0.3, 0.2], dtype=float, order='F')
        theta = 1.0
        rank = -1
        tol = 0.0
        reltol = 0.0

        q_out, e_out, u, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'I', 'N', m, n, rank, theta, q, e, tol, reltol
        )

        assert info == 0

        utu = u.T @ u
        np.testing.assert_allclose(utu, np.eye(p), rtol=1e-14, atol=1e-14)

    def test_with_v_identity(self):
        """
        Test with V initialized to identity.

        Validates that V matrix is orthogonal after transformation.
        """
        m = 4
        n = 4
        p = min(m, n)

        q = np.array([3.0, 2.0, 1.5, 0.5], dtype=float, order='F')
        e = np.array([0.5, 0.3, 0.2], dtype=float, order='F')
        theta = 1.0
        rank = -1
        tol = 0.0
        reltol = 0.0

        q_out, e_out, v, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'I', m, n, rank, theta, q, e, tol, reltol
        )

        assert info == 0

        vtv = v.T @ v
        np.testing.assert_allclose(vtv, np.eye(p), rtol=1e-14, atol=1e-14)

    def test_with_both_uv_identity(self):
        """
        Test with both U and V initialized to identity.

        Validates orthogonality of both transformation matrices.
        """
        m = 5
        n = 5
        p = min(m, n)

        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
        e = np.array([2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
        theta = 2.0
        rank = -1
        tol = 0.0
        reltol = 0.0

        q_out, e_out, u, v, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'I', 'I', m, n, rank, theta, q, e, tol, reltol
        )

        assert info == 0
        assert rank_out == 3

        utu = u.T @ u
        np.testing.assert_allclose(utu, np.eye(p), rtol=1e-14, atol=1e-14)

        vtv = v.T @ v
        np.testing.assert_allclose(vtv, np.eye(p), rtol=1e-14, atol=1e-14)


class TestMB04YDRankSpecified:
    """Tests with user-specified rank."""

    def test_specified_rank(self):
        """
        Test with rank specified by user.

        Verifies that the routine computes appropriate THETA value.
        """
        m = 5
        n = 5

        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
        e = np.array([2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
        theta = -1.0
        rank = 3
        tol = 0.0
        reltol = 0.0

        q_out, e_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'N', m, n, rank, theta, q, e, tol, reltol
        )

        assert info == 0
        assert rank_out == 3
        assert theta_out > 0.0


class TestMB04YDEdgeCases:
    """Edge case tests."""

    def test_p_equals_zero(self):
        """Test with P = min(M,N) = 0 (quick return)."""
        m = 0
        n = 5

        q = np.array([], dtype=float, order='F')
        e = np.array([], dtype=float, order='F')
        theta = 1.0
        rank = -1
        tol = 0.0
        reltol = 0.0

        q_out, e_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'N', m, n, rank, theta, q, e, tol, reltol
        )

        assert info == 0
        assert rank_out == 0

    def test_p_equals_one(self):
        """Test with P = min(M,N) = 1 (single element)."""
        m = 1
        n = 3

        q = np.array([5.0], dtype=float, order='F')
        e = np.array([], dtype=float, order='F')
        theta = 3.0
        rank = -1
        tol = 0.0
        reltol = 0.0

        q_out, e_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'N', m, n, rank, theta, q, e, tol, reltol
        )

        assert info == 0
        assert rank_out == 1

    def test_rectangular_m_less_than_n(self):
        """Test with M < N (rectangular case)."""
        m = 3
        n = 5
        p = min(m, n)

        q = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
        e = np.array([0.5, 0.3], dtype=float, order='F')
        theta = 1.5
        rank = -1
        tol = 0.0
        reltol = 0.0

        q_out, e_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'N', m, n, rank, theta, q, e, tol, reltol
        )

        assert info == 0
        assert rank_out >= 0
        assert len(q_out) == p

    def test_rectangular_m_greater_than_n(self):
        """Test with M > N (rectangular case)."""
        m = 5
        n = 3
        p = min(m, n)

        q = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
        e = np.array([0.5, 0.3], dtype=float, order='F')
        theta = 1.5
        rank = -1
        tol = 0.0
        reltol = 0.0

        q_out, e_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'N', m, n, rank, theta, q, e, tol, reltol
        )

        assert info == 0
        assert rank_out >= 0
        assert len(q_out) == p


class TestMB04YDErrorHandling:
    """Error handling tests."""

    def test_invalid_jobu(self):
        """Test invalid JOBU parameter."""
        m = 3
        n = 3

        q = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
        e = np.array([0.5, 0.3], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb04yd('X', 'N', m, n, -1, 1.0, q, e, 0.0, 0.0)

    def test_invalid_jobv(self):
        """Test invalid JOBV parameter."""
        m = 3
        n = 3

        q = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
        e = np.array([0.5, 0.3], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb04yd('N', 'X', m, n, -1, 1.0, q, e, 0.0, 0.0)

    def test_rank_greater_than_p(self):
        """Test RANK > min(M,N)."""
        m = 3
        n = 3

        q = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
        e = np.array([0.5, 0.3], dtype=float, order='F')

        q_out, e_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'N', m, n, 5, 1.0, q, e, 0.0, 0.0
        )

        assert info == -5

    def test_negative_theta_with_negative_rank(self):
        """Test THETA < 0 when RANK < 0 (invalid combination)."""
        m = 3
        n = 3

        q = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
        e = np.array([0.5, 0.3], dtype=float, order='F')

        q_out, e_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'N', m, n, -1, -1.0, q, e, 0.0, 0.0
        )

        assert info == -6


class TestMB04YDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_singular_value_classification(self):
        """
        Test that INUL correctly identifies singular values <= THETA.

        After transformation, INUL[i] = True means the i-th diagonal
        element belongs to a submatrix with all singular values <= THETA.
        """
        m = 5
        n = 5

        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
        e = np.array([2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
        theta = 2.0
        rank = -1

        q_out, e_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'N', m, n, rank, theta, q, e, 0.0, 0.0
        )

        assert info == 0
        num_small = np.sum(inul_out)
        num_large = m - num_small
        assert num_large == rank_out

    def test_bidiagonal_transformation_preserves_singular_values(self):
        """
        Test that U^T * J * V preserves singular values.

        The transformation J_new = U^T * J_orig * V should preserve
        singular values (up to numerical precision).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m = 4
        n = 4
        p = min(m, n)

        q_orig = np.array([3.0, 2.5, 2.0, 1.0], dtype=float, order='F')
        e_orig = np.array([0.1, 0.2, 0.1], dtype=float, order='F')

        j_orig = np.diag(q_orig) + np.diag(e_orig, k=1)
        sv_orig = np.linalg.svd(j_orig, compute_uv=False)

        q_new, e_new, u, v, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'I', 'I', m, n, -1, 1.5, q_orig.copy(), e_orig.copy(), 0.0, 0.0
        )

        assert info == 0

        j_new = np.diag(q_new) + np.diag(e_new, k=1)
        sv_new = np.linalg.svd(j_new, compute_uv=False)

        np.testing.assert_allclose(
            sorted(sv_orig), sorted(sv_new), rtol=1e-10, atol=1e-12
        )

    def test_orthogonality_of_transformations(self):
        """
        Test orthogonality: U^T U = I and V^T V = I.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m = 6
        n = 6
        p = min(m, n)

        q = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5], dtype=float, order='F')
        e = np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=float, order='F')
        theta = 2.5
        rank = -1

        q_out, e_out, u, v, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'I', 'I', m, n, rank, theta, q, e, 0.0, 0.0
        )

        assert info == 0

        utu = u.T @ u
        np.testing.assert_allclose(utu, np.eye(p), rtol=1e-14, atol=1e-14)

        vtv = v.T @ v
        np.testing.assert_allclose(vtv, np.eye(p), rtol=1e-14, atol=1e-14)


class TestMB04YDUpdateMode:
    """Tests for update mode (JOBU='U', JOBV='U')."""

    def test_update_u_matrix(self):
        """
        Test updating an existing U matrix.

        When JOBU='U', the given matrix U is updated by the left-hand
        Givens rotations used in the calculation.
        """
        m = 4
        n = 4
        p = min(m, n)

        q = np.array([3.0, 2.0, 1.5, 0.5], dtype=float, order='F')
        e = np.array([0.5, 0.3, 0.2], dtype=float, order='F')
        theta = 1.0
        rank = -1

        u_init = np.eye(m, p, dtype=float, order='F')

        q_out, e_out, u_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'U', 'N', m, n, rank, theta, q, e, 0.0, 0.0, u=u_init
        )

        assert info == 0

        utu = u_out.T @ u_out
        np.testing.assert_allclose(utu, np.eye(p), rtol=1e-14, atol=1e-14)

    def test_update_v_matrix(self):
        """
        Test updating an existing V matrix.

        When JOBV='U', the given matrix V is updated by the right-hand
        Givens rotations used in the calculation.
        """
        m = 4
        n = 4
        p = min(m, n)

        q = np.array([3.0, 2.0, 1.5, 0.5], dtype=float, order='F')
        e = np.array([0.5, 0.3, 0.2], dtype=float, order='F')
        theta = 1.0
        rank = -1

        v_init = np.eye(n, p, dtype=float, order='F')

        q_out, e_out, v_out, theta_out, rank_out, inul_out, iwarn, info = mb04yd(
            'N', 'U', m, n, rank, theta, q, e, 0.0, 0.0, v=v_init
        )

        assert info == 0

        vtv = v_out.T @ v_out
        np.testing.assert_allclose(vtv, np.eye(p), rtol=1e-14, atol=1e-14)
