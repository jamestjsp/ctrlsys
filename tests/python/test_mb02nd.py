"""
Tests for MB02ND - Total Least Squares solution using Partial SVD.

MB02ND solves the Total Least Squares (TLS) problem using a Partial
Singular Value Decomposition (PSVD) approach.
"""

import numpy as np
import pytest

from ctrlsys import mb02nd


class TestMB02NDBasic:
    """Basic functionality tests from SLICOT HTML documentation example."""

    def test_tls_from_html_doc(self):
        """
        Test TLS solution using example from SLICOT HTML documentation.

        Input: 6x3 data matrix A, 6x1 observation vector B
        RANK = -1 (compute rank automatically)
        THETA = 0.001 (threshold for singular values)

        Expected output:
        - RANK = 3
        - X = [0.5003, 0.8003, 0.2995]^T
        """
        m, n, l = 6, 3, 1

        c = np.array([
            [0.80010, 0.39985, 0.60005, 0.89999],
            [0.29996, 0.69990, 0.39997, 0.82997],
            [0.49994, 0.60003, 0.20012, 0.79011],
            [0.90013, 0.20016, 0.79995, 0.85002],
            [0.39998, 0.80006, 0.49985, 0.99016],
            [0.20002, 0.90007, 0.70009, 1.02994],
        ], dtype=float, order='F')

        rank = -1
        theta = 0.001
        tol = 0.0
        reltol = 0.0

        x, rank_out, theta_out, q, inul, iwarn, info = mb02nd(
            m, n, l, rank, theta, c, tol, reltol
        )

        assert info == 0
        assert rank_out == 3

        x_expected = np.array([[0.5003], [0.8003], [0.2995]], dtype=float)
        np.testing.assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)

        assert len(q) == 2 * min(m, n + l) - 1
        assert len(inul) == n + l
        assert inul[-1] == True


class TestMB02NDRankSpecified:
    """Tests with user-specified rank."""

    def test_tls_with_specified_rank(self):
        """
        Test TLS with user-specified rank.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m, n, l = 8, 4, 2

        A = np.random.randn(m, n)
        true_X = np.array([[1.0, 0.5], [0.5, 1.0], [-0.5, 0.5], [0.2, -0.3]])
        B = A @ true_X + 0.01 * np.random.randn(m, l)

        c = np.asfortranarray(np.hstack([A, B]))

        rank = 4
        theta = -1.0
        tol = 1e-10
        reltol = 1e-10

        x, rank_out, theta_out, q, inul, iwarn, info = mb02nd(
            m, n, l, rank, theta, c, tol, reltol
        )

        assert info == 0
        assert rank_out <= min(m, n)

        np.testing.assert_allclose(x, true_X, rtol=0.1, atol=0.1)


class TestMB02NDEdgeCases:
    """Edge case tests."""

    def test_empty_m_zero(self):
        """Test with M=0 (quick return case)."""
        m, n, l = 0, 3, 1

        c = np.zeros((n + l, n + l), dtype=float, order='F')

        rank = -1
        theta = 0.001
        tol = 0.0
        reltol = 0.0

        x, rank_out, theta_out, q, inul, iwarn, info = mb02nd(
            m, n, l, rank, theta, c, tol, reltol
        )

        assert info == 0
        assert rank_out == 0


    def test_l_zero(self):
        """Test with L=0 (no right-hand side)."""
        m, n, l = 4, 3, 0

        c = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ], dtype=float, order='F')

        rank = -1
        theta = 0.001
        tol = 0.0
        reltol = 0.0

        x, rank_out, theta_out, q, inul, iwarn, info = mb02nd(
            m, n, l, rank, theta, c, tol, reltol
        )

        assert info == 0


class TestMB02NDErrorHandling:
    """Error handling tests."""

    def test_invalid_m_negative(self):
        """Test error for negative M."""
        m, n, l = -1, 3, 1
        c = np.zeros((4, 4), dtype=float, order='F')

        with pytest.raises(ValueError):
            mb02nd(m, n, l, -1, 0.001, c, 0.0, 0.0)

    def test_invalid_rank_too_large(self):
        """Test error for RANK > min(M,N)."""
        m, n, l = 4, 3, 1
        c = np.zeros((4, 4), dtype=float, order='F')

        with pytest.raises(ValueError):
            mb02nd(m, n, l, 5, 0.001, c, 0.0, 0.0)


class TestMB02NDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_residual_minimization(self):
        """
        Validate TLS residual is near-optimal.

        For TLS solution X, verify ||[A|B] - [A+DA|B+DB]||_F is small
        where (A+DA)X = B+DB.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m, n, l = 10, 3, 1

        A = np.random.randn(m, n)
        true_X = np.array([[1.0], [2.0], [3.0]])
        B = A @ true_X + 0.01 * np.random.randn(m, l)

        c = np.asfortranarray(np.hstack([A, B]))
        c_orig = c.copy()

        rank = n
        theta = -1.0
        tol = 1e-12
        reltol = 1e-12

        x, rank_out, theta_out, q, inul, iwarn, info = mb02nd(
            m, n, l, rank, theta, c, tol, reltol
        )

        assert info == 0

        residual = A @ x - B
        residual_norm = np.linalg.norm(residual, 'fro')
        data_norm = np.linalg.norm(c_orig, 'fro')

        assert residual_norm / data_norm < 0.1

    def test_solution_dimension(self):
        """
        Validate output X has correct dimensions (N x L).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m, n, l = 8, 4, 2

        A = np.random.randn(m, n)
        true_X = np.array([[1.0, 0.5], [0.5, 1.0], [-0.5, 0.5], [0.2, -0.3]])
        B = A @ true_X + 0.01 * np.random.randn(m, l)
        c = np.asfortranarray(np.hstack([A, B]))

        rank = n
        theta = -1.0
        tol = 0.0
        reltol = 0.0

        x, rank_out, theta_out, q, inul, iwarn, info = mb02nd(
            m, n, l, rank, theta, c, tol, reltol
        )

        assert info == 0
        assert x.shape == (n, l)

    def test_bidiagonal_structure(self):
        """
        Validate Q array contains bidiagonal matrix elements.

        Q[0:p] contains diagonal elements q(1),...,q(p)
        Q[p:2p-1] contains superdiagonal elements e(1),...,e(p-1)
        where p = min(M, N+L).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m, n, l = 6, 3, 1
        p = min(m, n + l)

        A = np.random.randn(m, n)
        true_X = np.array([[1.0], [2.0], [0.5]])
        B = A @ true_X + 0.01 * np.random.randn(m, l)
        c = np.asfortranarray(np.hstack([A, B]))

        rank = n
        theta = -1.0
        tol = 0.0
        reltol = 0.0

        x, rank_out, theta_out, q, inul, iwarn, info = mb02nd(
            m, n, l, rank, theta, c, tol, reltol
        )

        assert info == 0
        assert len(q) == 2 * p - 1

        diagonal = q[:p]
        superdiagonal = q[p:]

        assert len(diagonal) == p
        assert len(superdiagonal) == p - 1
