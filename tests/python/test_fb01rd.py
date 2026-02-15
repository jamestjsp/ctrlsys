"""
FB01RD - Time-invariant square root covariance Kalman filter (observer Hessenberg form).

Combined measurement and time update of one iteration of the time-invariant
Kalman filter using square root covariance filter with condensed observer
Hessenberg form.

Test data extracted from SLICOT HTML documentation example.
"""

import numpy as np
import pytest
from ctrlsys import fb01rd


class TestFB01RDBasic:
    """Basic functionality test with HTML documentation example data."""

    def test_basic_jobk_k_multbq_n(self):
        """
        Test basic FB01RD functionality with JOBK='K', MULTBQ='N'.

        Test data from SLICOT FB01RD.html documentation example.
        N=4, M=2, P=2, three iterations of Kalman filter.
        """
        n, m, p = 4, 2, 2

        # Input: S (state covariance square root) - N×N lower triangular
        # Read row-wise from HTML doc: ((S(I,J), J=1,N), I=1,N)
        # Initial S is all zeros
        s = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=float, order='F')

        # Input: A (state transition matrix in lower observer Hessenberg form)
        # Read row-wise: ((A(I,J), J=1,N), I=1,N)
        a = np.array([
            [0.2113, 0.8497, 0.7263, 0.0000],
            [0.7560, 0.6857, 0.1985, 0.6525],
            [0.0002, 0.8782, 0.5442, 0.3076],
            [0.3303, 0.0683, 0.2320, 0.9329]
        ], dtype=float, order='F')

        # Input: B (input weight matrix) - N×M
        # Read row-wise: ((B(I,J), J=1,M), I=1,N)
        b = np.array([
            [0.5618, 0.5042],
            [0.5896, 0.3493],
            [0.6853, 0.3873],
            [0.8906, 0.9222]
        ], dtype=float, order='F')

        # Input: Q (process noise covariance sqrt) - M×M lower triangular
        # Read row-wise: ((Q(I,J), J=1,M), I=1,M)
        q = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        # Input: C (output matrix in lower observer Hessenberg form) - P×N
        # Read row-wise: ((C(I,J), J=1,N), I=1,P)
        c = np.array([
            [0.3616, 0.0000, 0.0000, 0.0000],
            [0.2922, 0.4826, 0.0000, 0.0000]
        ], dtype=float, order='F')

        # Input: R (measurement noise covariance sqrt) - P×P lower triangular
        # Read row-wise: ((R(I,J), J=1,P), I=1,P)
        r_orig = np.array([
            [0.9488, 0.0000],
            [0.3760, 0.7340]
        ], dtype=float, order='F')

        # Run 3 iterations as in the example
        s_work = s.copy(order='F')
        for _ in range(3):
            r_work = r_orig.copy(order='F')
            s_work, r_out, k_out, info = fb01rd(
                'K',       # jobk
                'N',       # multbq
                s_work, a, b, q, c, r_work,
                0.0        # tol
            )
            assert info == 0, f"fb01rd failed with info={info}"

        # Expected output: S after 3 iterations (from HTML Program Results)
        s_expected = np.array([
            [-1.7223, 0.0000, 0.0000, 0.0000],
            [-2.1073, 0.5467, 0.0000, 0.0000],
            [-1.7649, 0.1412, -0.1710, 0.0000],
            [-1.8291, 0.2058, -0.1497, 0.7760]
        ], dtype=float, order='F')

        # Expected K after 3 iterations (from HTML Program Results)
        k_expected = np.array([
            [-0.2135, 1.6649],
            [-0.2345, 2.1442],
            [-0.2147, 1.7069],
            [-0.1345, 1.4777]
        ], dtype=float, order='F')

        # Validate S output (lower triangular part, allowing sign differences)
        np.testing.assert_allclose(np.abs(s_work), np.abs(s_expected), rtol=1e-3, atol=1e-4)

        # Validate K output (allowing sign differences due to different factorizations)
        np.testing.assert_allclose(np.abs(k_out), np.abs(k_expected), rtol=1e-3, atol=1e-4)

    def test_basic_jobk_n_multbq_n(self):
        """
        Test FB01RD with JOBK='N' (K not computed).

        Same data, but Kalman gain K is not computed.
        """
        n, m, p = 4, 2, 2

        s = np.zeros((n, n), dtype=float, order='F')
        a = np.array([
            [0.2113, 0.8497, 0.7263, 0.0000],
            [0.7560, 0.6857, 0.1985, 0.6525],
            [0.0002, 0.8782, 0.5442, 0.3076],
            [0.3303, 0.0683, 0.2320, 0.9329]
        ], dtype=float, order='F')
        b = np.array([
            [0.5618, 0.5042],
            [0.5896, 0.3493],
            [0.6853, 0.3873],
            [0.8906, 0.9222]
        ], dtype=float, order='F')
        q = np.eye(m, dtype=float, order='F')
        c = np.array([
            [0.3616, 0.0000, 0.0000, 0.0000],
            [0.2922, 0.4826, 0.0000, 0.0000]
        ], dtype=float, order='F')
        r = np.array([
            [0.9488, 0.0000],
            [0.3760, 0.7340]
        ], dtype=float, order='F')

        s_out, r_out, k_out, info = fb01rd(
            'N',       # jobk - K not computed
            'N',       # multbq
            s, a, b, q, c, r,
            0.0
        )

        assert info == 0

    def test_single_iteration(self):
        """
        Test single iteration of FB01RD.

        Validates state covariance update for one iteration.
        """
        n, m, p = 4, 2, 2

        s = np.zeros((n, n), dtype=float, order='F')
        a = np.array([
            [0.2113, 0.8497, 0.7263, 0.0000],
            [0.7560, 0.6857, 0.1985, 0.6525],
            [0.0002, 0.8782, 0.5442, 0.3076],
            [0.3303, 0.0683, 0.2320, 0.9329]
        ], dtype=float, order='F')
        b = np.array([
            [0.5618, 0.5042],
            [0.5896, 0.3493],
            [0.6853, 0.3873],
            [0.8906, 0.9222]
        ], dtype=float, order='F')
        q = np.eye(m, dtype=float, order='F')
        c = np.array([
            [0.3616, 0.0000, 0.0000, 0.0000],
            [0.2922, 0.4826, 0.0000, 0.0000]
        ], dtype=float, order='F')
        r = np.array([
            [0.9488, 0.0000],
            [0.3760, 0.7340]
        ], dtype=float, order='F')

        s_out, r_out, k_out, info = fb01rd(
            'K',
            'N',
            s, a, b, q, c, r,
            0.0
        )

        assert info == 0
        assert s_out.shape == (n, n)
        assert k_out.shape == (n, p)


class TestFB01RDEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_n_equals_zero(self):
        """Test with N=0 (quick return case)."""
        n, m, p = 0, 2, 2

        s = np.zeros((1, 1), dtype=float, order='F')
        a = np.zeros((1, 1), dtype=float, order='F')
        b = np.zeros((1, m), dtype=float, order='F')
        q = np.eye(m, dtype=float, order='F')
        c = np.zeros((p, 1), dtype=float, order='F')
        r = np.eye(p, dtype=float, order='F')

        # Use positional arg to override n to 0
        s_out, r_out, k_out, info = fb01rd(
            'K', 'N',
            s, a, b, q, c, r,
            0.0, 0  # tol=0.0, n_override=0
        )

        assert info == 0

    def test_small_system_n1(self):
        """Test with minimal N=1 system."""
        n, m, p = 1, 1, 1

        np.random.seed(42)

        s = np.array([[0.5]], dtype=float, order='F')
        a = np.array([[0.9]], dtype=float, order='F')
        b = np.array([[0.3]], dtype=float, order='F')
        q = np.array([[1.0]], dtype=float, order='F')
        c = np.array([[0.7]], dtype=float, order='F')
        r = np.array([[1.0]], dtype=float, order='F')

        s_out, r_out, k_out, info = fb01rd(
            'K', 'N',
            s, a, b, q, c, r,
            0.0
        )

        assert info == 0
        assert np.all(np.isfinite(s_out))
        assert np.all(np.isfinite(k_out))


class TestFB01RDMultbqVariants:
    """Test MULTBQ parameter variants (P vs N)."""

    def test_multbq_p_variant(self):
        """
        Test with MULTBQ='P' (B already contains B*Q^{1/2} product).

        Q is not used when MULTBQ='P'.
        """
        n, m, p = 3, 2, 2

        np.random.seed(123)

        s = np.zeros((n, n), dtype=float, order='F')
        a = np.array([
            [0.5, 0.2, 0.0],
            [0.1, 0.6, 0.3],
            [0.0, 0.1, 0.7]
        ], dtype=float, order='F')

        # B already contains B*Q^{1/2} product
        b_times_q = np.array([
            [0.3, 0.2],
            [0.1, 0.4],
            [0.2, 0.1]
        ], dtype=float, order='F')

        # Q not used with MULTBQ='P'
        q_dummy = np.zeros((1, 1), dtype=float, order='F')

        c = np.array([
            [0.5, 0.0, 0.0],
            [0.2, 0.4, 0.0]
        ], dtype=float, order='F')

        r = np.eye(p, dtype=float, order='F')

        s_out, r_out, k_out, info = fb01rd(
            'K', 'P',
            s, a, b_times_q, q_dummy, c, r,
            0.0
        )

        assert info == 0
        assert np.all(np.isfinite(s_out))


class TestFB01RDParameters:
    """Test parameter validation and error handling."""

    def test_invalid_jobk(self):
        """Test with invalid JOBK parameter."""
        n, m, p = 2, 1, 1

        s = np.eye(n, dtype=float, order='F')
        a = np.eye(n, dtype=float, order='F')
        b = np.ones((n, m), dtype=float, order='F')
        q = np.eye(m, dtype=float, order='F')
        c = np.ones((p, n), dtype=float, order='F')
        r = np.eye(p, dtype=float, order='F')

        with pytest.raises((ValueError, RuntimeError)):
            fb01rd(
                'Z',  # Invalid JOBK
                'N',
                s, a, b, q, c, r,
                0.0
            )

    def test_invalid_multbq(self):
        """Test with invalid MULTBQ parameter."""
        n, m, p = 2, 1, 1

        s = np.eye(n, dtype=float, order='F')
        a = np.eye(n, dtype=float, order='F')
        b = np.ones((n, m), dtype=float, order='F')
        q = np.eye(m, dtype=float, order='F')
        c = np.ones((p, n), dtype=float, order='F')
        r = np.eye(p, dtype=float, order='F')

        with pytest.raises((ValueError, RuntimeError)):
            fb01rd(
                'K',
                'Z',  # Invalid MULTBQ
                s, a, b, q, c, r,
                0.0
            )


class TestFB01RDNumericalProperties:
    """Test mathematical properties of the Kalman filter update."""

    def test_s_lower_triangular_preservation(self):
        """
        Verify S remains lower triangular after update.

        By design, FB01RD preserves the lower triangular structure of S.
        """
        n, m, p = 4, 2, 2

        np.random.seed(789)

        s = np.tril(np.random.randn(n, n).astype(float))
        s = np.asfortranarray(s)

        a = np.array([
            [0.2113, 0.8497, 0.7263, 0.0000],
            [0.7560, 0.6857, 0.1985, 0.6525],
            [0.0002, 0.8782, 0.5442, 0.3076],
            [0.3303, 0.0683, 0.2320, 0.9329]
        ], dtype=float, order='F')

        b = np.array([
            [0.5618, 0.5042],
            [0.5896, 0.3493],
            [0.6853, 0.3873],
            [0.8906, 0.9222]
        ], dtype=float, order='F')

        q = np.eye(m, dtype=float, order='F')

        c = np.array([
            [0.3616, 0.0000, 0.0000, 0.0000],
            [0.2922, 0.4826, 0.0000, 0.0000]
        ], dtype=float, order='F')

        r = np.array([
            [0.9488, 0.0000],
            [0.3760, 0.7340]
        ], dtype=float, order='F')

        s_out, r_out, k_out, info = fb01rd(
            'K', 'N',
            s, a, b, q, c, r,
            0.0
        )

        assert info == 0

        # Verify S_out is lower triangular (strict upper = 0)
        for i in range(n):
            for j in range(i + 1, n):
                assert abs(s_out[i, j]) < 1e-10, f"S({i},{j})={s_out[i,j]} should be ~0"

    def test_r_lower_triangular_output(self):
        """
        Verify R output (RINOV^{1/2}) is lower triangular.
        """
        n, m, p = 4, 2, 2

        s = np.zeros((n, n), dtype=float, order='F')
        a = np.array([
            [0.2113, 0.8497, 0.7263, 0.0000],
            [0.7560, 0.6857, 0.1985, 0.6525],
            [0.0002, 0.8782, 0.5442, 0.3076],
            [0.3303, 0.0683, 0.2320, 0.9329]
        ], dtype=float, order='F')

        b = np.array([
            [0.5618, 0.5042],
            [0.5896, 0.3493],
            [0.6853, 0.3873],
            [0.8906, 0.9222]
        ], dtype=float, order='F')

        q = np.eye(m, dtype=float, order='F')

        c = np.array([
            [0.3616, 0.0000, 0.0000, 0.0000],
            [0.2922, 0.4826, 0.0000, 0.0000]
        ], dtype=float, order='F')

        r = np.array([
            [0.9488, 0.0000],
            [0.3760, 0.7340]
        ], dtype=float, order='F')

        s_out, r_out, k_out, info = fb01rd(
            'K', 'N',
            s, a, b, q, c, r,
            0.0
        )

        assert info == 0

        # Verify R_out is lower triangular
        for i in range(p):
            for j in range(i + 1, p):
                assert abs(r_out[i, j]) < 1e-10, f"R({i},{j})={r_out[i,j]} should be ~0"

    def test_covariance_positive_semi_definite(self):
        """
        Verify P = S * S' is positive semi-definite.

        The state covariance matrix must have non-negative eigenvalues.
        Random seed: 456 (for reproducibility)
        """
        n, m, p = 4, 2, 2

        np.random.seed(456)

        s = np.tril(np.random.randn(n, n).astype(float))
        s = np.asfortranarray(s)

        a = np.array([
            [0.2113, 0.8497, 0.7263, 0.0000],
            [0.7560, 0.6857, 0.1985, 0.6525],
            [0.0002, 0.8782, 0.5442, 0.3076],
            [0.3303, 0.0683, 0.2320, 0.9329]
        ], dtype=float, order='F')

        b = np.array([
            [0.5618, 0.5042],
            [0.5896, 0.3493],
            [0.6853, 0.3873],
            [0.8906, 0.9222]
        ], dtype=float, order='F')

        q = np.eye(m, dtype=float, order='F')

        c = np.array([
            [0.3616, 0.0000, 0.0000, 0.0000],
            [0.2922, 0.4826, 0.0000, 0.0000]
        ], dtype=float, order='F')

        r = np.array([
            [0.9488, 0.0000],
            [0.3760, 0.7340]
        ], dtype=float, order='F')

        s_out, r_out, k_out, info = fb01rd(
            'K', 'N',
            s, a, b, q, c, r,
            0.0
        )

        assert info == 0

        # Compute P = S * S'
        p_cov = s_out @ s_out.T

        # Check positive semi-definite (all eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(p_cov)
        assert np.all(eigvals >= -1e-10), f"Covariance has negative eigenvalues: {eigvals}"
