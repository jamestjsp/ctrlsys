"""Tests for FB01QD - Time-varying square root covariance Kalman filter."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestFB01QDBasic:
    """Basic functionality tests from HTML documentation."""

    def test_html_example_3_iterations(self):
        """
        Validate FB01QD using HTML doc example.

        The example runs 3 Kalman filter iterations and returns
        the state covariance square root S and gain K.

        From SLICOT HTML documentation - program runs 3 iterations.
        """
        from ctrlsys import fb01qd

        n, m, p = 4, 2, 2

        # Initial state covariance sqrt S_{i-1} = zeros (row-wise in HTML)
        s = np.zeros((n, n), order='F', dtype=float)

        # State transition matrix A (row-wise in HTML)
        a = np.array([
            [0.2113, 0.8497, 0.7263, 0.8833],
            [0.7560, 0.6857, 0.1985, 0.6525],
            [0.0002, 0.8782, 0.5442, 0.3076],
            [0.3303, 0.0683, 0.2320, 0.9329],
        ], order='F', dtype=float)

        # Input weight matrix B (row-wise in HTML)
        b = np.array([
            [0.5618, 0.5042],
            [0.5896, 0.3493],
            [0.6853, 0.3873],
            [0.8906, 0.9222],
        ], order='F', dtype=float)

        # Process noise sqrt Q (lower triangular, row-wise)
        q = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], order='F', dtype=float)

        # Output matrix C (row-wise in HTML)
        c = np.array([
            [0.3616, 0.5664, 0.5015, 0.2693],
            [0.2922, 0.4826, 0.4368, 0.6325],
        ], order='F', dtype=float)

        # Measurement noise sqrt R (lower triangular, row-wise)
        r_orig = np.array([
            [0.9488, 0.0],
            [0.3760, 0.7340],
        ], order='F', dtype=float)

        # Run 3 iterations
        for _ in range(3):
            r = r_orig.copy(order='F')
            s, k, r_out, rcond, info = fb01qd(
                'K', 'N', s, a, b, q, c, r, 0.0
            )
            assert info == 0, f"fb01qd returned info={info}"

        # Expected S after 3 iterations (from HTML doc, lower triangular)
        s_expected = np.array([
            [-1.2936,  0.0000,  0.0000,  0.0000],
            [-1.1382, -0.2579,  0.0000,  0.0000],
            [-0.9622, -0.1529,  0.2974,  0.0000],
            [-1.3076,  0.0936,  0.4508, -0.4897],
        ], order='F', dtype=float)

        # Expected K after 3 iterations (from HTML doc)
        k_expected = np.array([
            [0.3638, 0.9469],
            [0.3532, 0.8179],
            [0.2471, 0.5542],
            [0.1982, 0.6471],
        ], order='F', dtype=float)

        # Validate with HTML doc precision (4 decimal places)
        assert_allclose(s, s_expected, rtol=1e-3, atol=1e-3)
        assert_allclose(k, k_expected, rtol=1e-3, atol=1e-3)

    def test_multbq_product_mode(self):
        """
        Test MULTBQ='P' mode where B already contains B*Q^{1/2}.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import fb01qd

        np.random.seed(42)
        n, m, p = 3, 2, 2

        # Start with identity-like S
        s = np.eye(n, order='F', dtype=float) * 0.1

        # Random stable A
        a = np.random.randn(n, n).astype(float, order='F') * 0.3

        # B and Q
        b = np.random.randn(n, m).astype(float, order='F') * 0.5
        q_sqrt = np.tril(np.random.randn(m, m).astype(float, order='F')) * 0.3
        np.fill_diagonal(q_sqrt, np.abs(np.diag(q_sqrt)) + 0.1)

        # Pre-compute B * Q^{1/2}
        bq = b @ q_sqrt

        c = np.random.randn(p, n).astype(float, order='F') * 0.5

        # R sqrt (lower triangular)
        r = np.tril(np.random.randn(p, p).astype(float, order='F'))
        np.fill_diagonal(r, np.abs(np.diag(r)) + 0.5)

        # Run with MULTBQ='N' (separate B and Q)
        s1 = s.copy(order='F')
        r1 = r.copy(order='F')
        q1 = q_sqrt.copy(order='F')
        s1, k1, _, _, info1 = fb01qd('K', 'N', s1, a, b, q1, c, r1, 0.0)
        assert info1 == 0

        # Run with MULTBQ='P' (B already contains B*Q^{1/2})
        s2 = s.copy(order='F')
        r2 = r.copy(order='F')
        q_dummy = np.zeros((1, 1), order='F', dtype=float)
        s2, k2, _, _, info2 = fb01qd('K', 'P', s2, a, bq, q_dummy, c, r2, 0.0)
        assert info2 == 0

        # Results should match
        assert_allclose(s1, s2, rtol=1e-13, atol=1e-14)
        assert_allclose(k1, k2, rtol=1e-13, atol=1e-14)


class TestFB01QDEdgeCases:
    """Edge case tests."""

    def test_n_equals_zero(self):
        """Test quick return when n=0."""
        from ctrlsys import fb01qd

        n, m, p = 0, 2, 2

        s = np.zeros((0, 0), order='F', dtype=float)
        a = np.zeros((0, 0), order='F', dtype=float)
        b = np.zeros((0, m), order='F', dtype=float)
        q = np.eye(m, order='F', dtype=float)
        c = np.zeros((p, 0), order='F', dtype=float)
        r = np.eye(p, order='F', dtype=float)

        s, k, r_out, rcond, info = fb01qd('K', 'N', s, a, b, q, c, r, 0.0)

        assert info == 0
        assert s.shape == (0, 0)
        assert k.shape == (0, p)

    def test_jobk_no_gain(self):
        """
        Test JOBK='N' mode where gain K is not computed as final Kalman gain.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import fb01qd

        np.random.seed(123)
        n, m, p = 3, 2, 2

        s = np.eye(n, order='F', dtype=float) * 0.2

        a = np.random.randn(n, n).astype(float, order='F') * 0.3
        b = np.random.randn(n, m).astype(float, order='F') * 0.5
        q = np.tril(np.random.randn(m, m).astype(float, order='F'))
        np.fill_diagonal(q, np.abs(np.diag(q)) + 0.1)
        c = np.random.randn(p, n).astype(float, order='F') * 0.5
        r = np.tril(np.random.randn(p, p).astype(float, order='F'))
        np.fill_diagonal(r, np.abs(np.diag(r)) + 0.5)

        s, k, r_out, rcond, info = fb01qd('N', 'N', s, a, b, q, c, r, 0.0)

        assert info == 0
        # S should be updated
        assert s.shape == (n, n)
        # K contains AK_i (intermediate matrix, not final gain)
        assert k.shape == (n, p)


class TestFB01QDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_covariance_positivity(self):
        """
        Validate state covariance S*S' is positive semi-definite.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import fb01qd

        np.random.seed(456)
        n, m, p = 4, 2, 2

        # Start with small positive definite S
        s = np.tril(np.random.randn(n, n).astype(float, order='F')) * 0.1
        np.fill_diagonal(s, np.abs(np.diag(s)) + 0.1)

        # Stable A (small spectral radius)
        a = np.random.randn(n, n).astype(float, order='F') * 0.3

        b = np.random.randn(n, m).astype(float, order='F') * 0.5
        q = np.tril(np.random.randn(m, m).astype(float, order='F')) * 0.2
        np.fill_diagonal(q, np.abs(np.diag(q)) + 0.1)

        c = np.random.randn(p, n).astype(float, order='F') * 0.5
        r = np.tril(np.random.randn(p, p).astype(float, order='F'))
        np.fill_diagonal(r, np.abs(np.diag(r)) + 0.5)

        # Run several iterations
        for i in range(5):
            r_copy = r.copy(order='F')
            s, k, _, _, info = fb01qd('K', 'N', s, a, b, q, c, r_copy, 0.0)
            assert info == 0, f"Iteration {i}: info={info}"

            # P = S * S' should be positive semi-definite
            p_cov = s @ s.T
            eigvals = np.linalg.eigvalsh(p_cov)
            # Allow small negative due to numerical precision
            assert np.all(eigvals >= -1e-10), \
                f"Iteration {i}: P has negative eigenvalue {eigvals.min()}"

    def test_innovations_covariance_update(self):
        """
        Validate that R output contains (RINOV_i)^{1/2}, the innovations covariance sqrt.

        The innovations covariance is: RINOV = R + C*P*C'
        where P = S*S' is the prior covariance.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import fb01qd

        np.random.seed(789)
        n, m, p = 3, 2, 2

        # Initial S (lower triangular)
        s = np.tril(np.random.randn(n, n).astype(float, order='F')) * 0.5
        np.fill_diagonal(s, np.abs(np.diag(s)) + 0.2)

        a = np.random.randn(n, n).astype(float, order='F') * 0.3
        b = np.random.randn(n, m).astype(float, order='F') * 0.5
        q = np.tril(np.random.randn(m, m).astype(float, order='F'))
        np.fill_diagonal(q, np.abs(np.diag(q)) + 0.1)
        c = np.random.randn(p, n).astype(float, order='F') * 0.5
        r = np.tril(np.random.randn(p, p).astype(float, order='F'))
        np.fill_diagonal(r, np.abs(np.diag(r)) + 0.5)

        # Compute expected innovations covariance
        # P_prior = S * S'
        p_prior = s @ s.T
        # R_full = R * R'
        r_full = r @ r.T
        # RINOV = R_full + C * P_prior * C'
        rinov_expected = r_full + c @ p_prior @ c.T

        # Call FB01QD
        s_out, k, r_out, rcond, info = fb01qd('K', 'N', s.copy(order='F'), a, b, q, c, r.copy(order='F'), 0.0)
        assert info == 0

        # R_out is lower triangular RINOV^{1/2}
        # Reconstruct RINOV = R_out * R_out'
        rinov_computed = r_out @ r_out.T

        # Validate
        assert_allclose(rinov_computed, rinov_expected, rtol=1e-12, atol=1e-13)


class TestFB01QDErrorHandling:
    """Error handling tests."""

    def test_invalid_jobk(self):
        """Test invalid JOBK parameter."""
        from ctrlsys import fb01qd

        n, m, p = 2, 2, 2
        s = np.eye(n, order='F', dtype=float)
        a = np.eye(n, order='F', dtype=float)
        b = np.eye(n, m, order='F', dtype=float)
        q = np.eye(m, order='F', dtype=float)
        c = np.eye(p, n, order='F', dtype=float)
        r = np.eye(p, order='F', dtype=float)

        with pytest.raises(ValueError, match="JOBK"):
            fb01qd('X', 'N', s, a, b, q, c, r, 0.0)

    def test_invalid_multbq(self):
        """Test invalid MULTBQ parameter."""
        from ctrlsys import fb01qd

        n, m, p = 2, 2, 2
        s = np.eye(n, order='F', dtype=float)
        a = np.eye(n, order='F', dtype=float)
        b = np.eye(n, m, order='F', dtype=float)
        q = np.eye(m, order='F', dtype=float)
        c = np.eye(p, n, order='F', dtype=float)
        r = np.eye(p, order='F', dtype=float)

        with pytest.raises(ValueError, match="MULTBQ"):
            fb01qd('K', 'X', s, a, b, q, c, r, 0.0)
