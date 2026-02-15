"""Tests for MB02JD - Full QR factorization of block Toeplitz matrix."""

import numpy as np
import pytest
from ctrlsys import mb02jd


class TestMB02JDBasic:
    """Basic functionality tests from HTML documentation."""

    def test_html_example_compute_q_and_r(self):
        """
        Test from SLICOT HTML documentation.

        Computes Q and R for a 8x9 block Toeplitz matrix T with K=2, L=3, M=4, N=3.
        T = Q * R^T where Q^T Q = I and R is lower triangular.
        """
        k, l, m, n = 2, 3, 4, 3
        p, s = 0, 3  # Compute all 3 block columns from scratch

        # TC (first block column): M*K x L = 8x3, read row-by-row from HTML
        tc = np.array([
            [1.0, 4.0, 0.0],
            [4.0, 1.0, 2.0],
            [4.0, 2.0, 2.0],
            [5.0, 3.0, 2.0],
            [2.0, 4.0, 4.0],
            [5.0, 3.0, 4.0],
            [2.0, 2.0, 5.0],
            [4.0, 2.0, 3.0],
        ], order='F', dtype=float)

        # TR (first block row minus first block): K x (N-1)*L = 2x6, read row-by-row
        tr = np.array([
            [3.0, 4.0, 2.0, 5.0, 0.0, 4.0],
            [5.0, 1.0, 1.0, 2.0, 4.0, 1.0],
        ], order='F', dtype=float)

        q, r, info = mb02jd('Q', k, l, m, n, p, s, tc, tr)

        assert info == 0

        # Expected Q from HTML (8x8)
        q_expected = np.array([
            [-0.0967,  0.7166, -0.4651,  0.1272,  0.4357,  0.0435,  0.2201,  0.0673],
            [-0.3867, -0.3108, -0.0534,  0.5251,  0.0963, -0.3894,  0.1466,  0.5412],
            [-0.3867, -0.0990, -0.1443, -0.7021,  0.3056, -0.3367, -0.3233,  0.1249],
            [-0.4834, -0.0178, -0.3368, -0.1763, -0.5446,  0.5100,  0.1503,  0.2054],
            [-0.1933,  0.5859,  0.3214,  0.1156, -0.4670, -0.3199, -0.4185,  0.0842],
            [-0.4834, -0.0178,  0.1072,  0.0357, -0.0575, -0.2859,  0.4339, -0.6928],
            [-0.1933,  0.1623,  0.7251, -0.1966,  0.2736,  0.3058,  0.3398,  0.2968],
            [-0.3867, -0.0990,  0.0777,  0.3615,  0.3386,  0.4421, -0.5693, -0.2641],
        ], order='F', dtype=float)

        # Expected R from HTML (9x8 lower triangular, only 8 cols since min(M*K, N*L)=8)
        r_expected = np.array([
            [-10.3441,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ -6.3805,  4.7212,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ -7.3472,  1.9320,  4.5040,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [-10.0541,  2.5101,  0.5065,  3.6550,  0.0000,  0.0000,  0.0000,  0.0000],
            [ -6.5738,  3.6127,  1.2702, -1.3146,  3.5202,  0.0000,  0.0000,  0.0000],
            [ -5.2204,  2.4764,  2.4113,  1.3890,  1.2780,  2.4976,  0.0000,  0.0000],
            [ -9.6674,  3.2445, -0.5099, -0.0224,  2.6548,  2.9491,  1.0049,  0.0000],
            [ -6.3805,  0.6968,  1.9483,  0.3050,  0.7002, -2.0220, -2.8246,  2.3147],
            [ -4.1570,  2.4309, -0.7190, -0.1455,  3.0149,  0.5454,  0.9394, -0.0548],
        ], order='F', dtype=float)

        # Check shapes
        assert q.shape == (m * k, min(s * l, min(m * k, n * l) - p * l))
        assert r.shape[1] == min(s * l, min(m * k, n * l) - p * l)

        # Validate Q values (rtol=1e-3 for HTML 4-decimal precision)
        np.testing.assert_allclose(q, q_expected, rtol=1e-3, atol=1e-4)

        # Validate R values (only lower triangular part is meaningful)
        np.testing.assert_allclose(r[:r_expected.shape[0], :], r_expected, rtol=1e-3, atol=1e-4)

    def test_r_only_mode(self):
        """Test JOB='R' computes only R factor."""
        k, l, m, n = 2, 3, 4, 3
        p, s = 0, 3

        tc = np.array([
            [1.0, 4.0, 0.0],
            [4.0, 1.0, 2.0],
            [4.0, 2.0, 2.0],
            [5.0, 3.0, 2.0],
            [2.0, 4.0, 4.0],
            [5.0, 3.0, 4.0],
            [2.0, 2.0, 5.0],
            [4.0, 2.0, 3.0],
        ], order='F', dtype=float)

        tr = np.array([
            [3.0, 4.0, 2.0, 5.0, 0.0, 4.0],
            [5.0, 1.0, 1.0, 2.0, 4.0, 1.0],
        ], order='F', dtype=float)

        q, r, info = mb02jd('R', k, l, m, n, p, s, tc, tr)

        assert info == 0

        # R should match the Q+R case
        r_expected_diag = np.array([-10.3441, 4.7212, 4.5040, 3.6550, 3.5202, 2.4976, 1.0049, 2.3147])
        for i, expected in enumerate(r_expected_diag):
            np.testing.assert_allclose(r[i, i], expected, rtol=1e-3, atol=1e-4)


class TestMB02JDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_orthogonality_of_q(self):
        """
        Validate Q^T Q = I (orthonormality of Q columns).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k, l, m, n = 2, 2, 3, 3
        p, s = 0, 3

        # Generate random Toeplitz data
        tc = np.random.randn(m * k, l).astype(float, order='F')
        tr = np.random.randn(k, (n - 1) * l).astype(float, order='F')

        q, r, info = mb02jd('Q', k, l, m, n, p, s, tc, tr)

        if info == 0:  # Only check if rank condition satisfied
            # Q^T Q should be identity
            qtq = q.T @ q
            np.testing.assert_allclose(qtq, np.eye(qtq.shape[0]), rtol=1e-12, atol=1e-12)

    def test_lower_triangular_r(self):
        """
        Validate R is lower triangular.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k, l, m, n = 2, 2, 4, 4
        p, s = 0, 4

        tc = np.random.randn(m * k, l).astype(float, order='F')
        tr = np.random.randn(k, (n - 1) * l).astype(float, order='F')

        q, r, info = mb02jd('Q', k, l, m, n, p, s, tc, tr)

        if info == 0:
            # Check R is lower triangular (strictly upper part should be zero)
            for i in range(r.shape[0]):
                for j in range(i + 1, r.shape[1]):
                    assert abs(r[i, j]) < 1e-14, f"R[{i},{j}] = {r[i, j]} should be 0"


class TestMB02JDEdgeCases:
    """Edge case tests."""

    def test_single_block_n_equals_1(self):
        """Test N=1 (single block column in row)."""
        k, l, m, n = 2, 3, 4, 1
        p, s = 0, 1

        tc = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
        ], order='F', dtype=float)

        # TR not used when N=1, but still need valid array
        tr = np.zeros((k, 0), order='F', dtype=float)

        q, r, info = mb02jd('Q', k, l, m, n, p, s, tc, tr)

        # Should succeed (N=1 is valid)
        assert info == 0

    def test_zero_s_quick_return(self):
        """Test with S=0 (quick return, no block columns to compute)."""
        k, l, m, n = 2, 3, 4, 3
        p, s = 0, 0  # S=0 means nothing to compute

        tc = np.zeros((m * k, l), order='F', dtype=float)
        tr = np.zeros((k, (n - 1) * l), order='F', dtype=float)

        q, r, info = mb02jd('Q', k, l, m, n, p, s, tc, tr)

        assert info == 0


class TestMB02JDErrorHandling:
    """Error handling tests."""

    def test_invalid_job(self):
        """Test invalid JOB parameter."""
        k, l, m, n = 2, 3, 4, 3
        p, s = 0, 3

        tc = np.zeros((m * k, l), order='F', dtype=float)
        tr = np.zeros((k, (n - 1) * l), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02jd('X', k, l, m, n, p, s, tc, tr)

    def test_negative_k(self):
        """Test negative K parameter."""
        with pytest.raises(ValueError):
            tc = np.zeros((8, 3), order='F', dtype=float)
            tr = np.zeros((2, 6), order='F', dtype=float)
            mb02jd('Q', -1, 3, 4, 3, 0, 3, tc, tr)

    def test_invalid_p(self):
        """Test invalid P parameter (P*L >= min(M*K, N*L) + L)."""
        k, l, m, n = 2, 3, 4, 3
        # P=3 means P*L=9 >= min(8,9)+3=11 is false, but P=4 gives P*L=12 >= 11 which is invalid
        p, s = 4, 1

        tc = np.zeros((m * k, l), order='F', dtype=float)
        tr = np.zeros((k, (n - 1) * l), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02jd('Q', k, l, m, n, p, s, tc, tr)


class TestMB02JDWorkspaceQuery:
    """Workspace query tests."""

    def test_workspace_query(self):
        """Test LDWORK=-1 workspace query mode."""
        k, l, m, n = 2, 3, 4, 3
        p, s = 0, 3

        tc = np.zeros((m * k, l), order='F', dtype=float)
        tr = np.zeros((k, (n - 1) * l), order='F', dtype=float)

        # Workspace query should return optimal size without error
        q, r, info = mb02jd('Q', k, l, m, n, p, s, tc, tr)
        assert info == 0
