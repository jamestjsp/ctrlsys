"""
Tests for FB01VD - One recursion of the conventional Kalman filter.

FB01VD computes one update of the Riccati difference equation and
the Kalman filter gain using the conventional Kalman filter approach.

References:
    - SLICOT Library Routine Documentation (FB01VD.html)
"""
import numpy as np
import pytest
from ctrlsys import fb01vd


class TestFB01VDHtmlDocExample:
    """Test FB01VD using the HTML documentation example."""

    def test_basic_functionality(self):
        """
        Validate FB01VD using the HTML doc example.

        Example parameters: n=4, m=3, l=2
        Tests one recursion of Kalman filter.
        """
        n, m, l = 4, 3, 2

        P = np.array([
            [0.5015, 0.4368, 0.2693, 0.6325],
            [0.4368, 0.4818, 0.2639, 0.4148],
            [0.2693, 0.2639, 0.1121, 0.6856],
            [0.6325, 0.4148, 0.6856, 0.8906]
        ], order='F', dtype=float)

        A = np.array([
            [0.2113, 0.8497, 0.7263, 0.8833],
            [0.7560, 0.6857, 0.1985, 0.6525],
            [0.0002, 0.8782, 0.5442, 0.3076],
            [0.3303, 0.0683, 0.2320, 0.9329]
        ], order='F', dtype=float)

        B = np.array([
            [0.0437, 0.7783, 0.5618],
            [0.4818, 0.2119, 0.5896],
            [0.2639, 0.1121, 0.6853],
            [0.4148, 0.6856, 0.8906]
        ], order='F', dtype=float)

        Q = np.array([
            [0.9329, 0.2146, 0.3126],
            [0.2146, 0.2922, 0.5664],
            [0.3126, 0.5664, 0.5935]
        ], order='F', dtype=float)

        C = np.array([
            [0.3873, 0.9488, 0.3760, 0.0881],
            [0.9222, 0.3435, 0.7340, 0.4498]
        ], order='F', dtype=float)

        R = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        tol = 0.0

        P_updated, K, Rinov_sqrt, rcond, info = fb01vd(n, m, l, P, A, B, C, Q, R, tol)

        assert info == 0, f"fb01vd failed with info={info}"

        P_expected = np.array([
            [1.6007, 1.3283, 1.1153, 1.7177],
            [1.3283, 1.2763, 1.0132, 1.5137],
            [1.1153, 1.0132, 0.8222, 1.2722],
            [1.7177, 1.5137, 1.2722, 2.1562]
        ], order='F', dtype=float)

        K_expected = np.array([
            [0.1648, 0.2241],
            [0.2115, 0.1610],
            [0.0728, 0.1673],
            [0.1304, 0.3892]
        ], order='F', dtype=float)

        Rinov_sqrt_expected = np.array([
            [1.5091, 1.1543],
            [0.0000, 1.5072]
        ], order='F', dtype=float)

        np.testing.assert_allclose(
            np.triu(P_updated), np.triu(P_expected), rtol=1e-3, atol=1e-4,
            err_msg="P_updated does not match expected"
        )
        np.testing.assert_allclose(
            K, K_expected, rtol=1e-3, atol=1e-4,
            err_msg="K does not match expected"
        )
        np.testing.assert_allclose(
            np.triu(Rinov_sqrt), np.triu(Rinov_sqrt_expected), rtol=1e-3, atol=1e-4,
            err_msg="Rinov_sqrt does not match expected"
        )


class TestFB01VDMathematicalProperties:
    """Test mathematical properties of the Kalman filter."""

    def test_riccati_equation_consistency(self):
        """
        Validate the discrete-time Riccati equation:
        P_next = A * (P - K*C*P) * A' + B * Q * B'

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, l = 3, 2, 2

        P = np.eye(n, order='F', dtype=float) * 0.5
        A = np.eye(n, order='F', dtype=float) * 0.9
        B = np.random.randn(n, m).astype(float, order='F') * 0.1
        Q = np.eye(m, order='F', dtype=float) * 0.01
        C = np.random.randn(l, n).astype(float, order='F')
        R = np.eye(l, order='F', dtype=float) * 0.1

        P_copy = P.copy(order='F')
        Q_copy = Q.copy(order='F')
        R_copy = R.copy(order='F')

        P_updated, K, Rinov_sqrt, rcond, info = fb01vd(n, m, l, P_copy, A, B, C, Q_copy, R_copy, 0.0)

        assert info == 0, f"fb01vd failed with info={info}"

        P_sym = np.triu(P)
        P_sym = P_sym + P_sym.T - np.diag(np.diag(P_sym))

        RINOV = C @ P_sym @ C.T + R
        K_manual = P_sym @ C.T @ np.linalg.inv(RINOV)

        np.testing.assert_allclose(
            K, K_manual, rtol=1e-10, atol=1e-12,
            err_msg="Kalman gain K does not match K = P*C'*inv(RINOV)"
        )

        P_next_manual = A @ (P_sym - K @ C @ P_sym) @ A.T + B @ Q @ B.T

        P_updated_sym = np.triu(P_updated)
        P_updated_sym = P_updated_sym + P_updated_sym.T - np.diag(np.diag(P_updated_sym))

        np.testing.assert_allclose(
            np.triu(P_updated_sym), np.triu(P_next_manual), rtol=1e-10, atol=1e-12,
            err_msg="Updated P does not satisfy Riccati equation"
        )

    def test_innovation_covariance_cholesky(self):
        """
        Validate that R output contains Cholesky factor of RINOV.
        RINOV = C * P * C' + R
        R_out = chol(RINOV) such that R_out' * R_out = RINOV

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, l = 4, 2, 3

        P = np.eye(n, order='F', dtype=float)
        A = np.eye(n, order='F', dtype=float)
        B = np.zeros((n, m), order='F', dtype=float)
        B[0, 0] = 0.1
        Q = np.eye(m, order='F', dtype=float)
        C = np.random.randn(l, n).astype(float, order='F') * 0.5
        R = np.eye(l, order='F', dtype=float)

        P_copy = P.copy(order='F')
        Q_copy = Q.copy(order='F')
        R_copy = R.copy(order='F')

        P_updated, K, Rinov_sqrt, rcond, info = fb01vd(n, m, l, P_copy, A, B, C, Q_copy, R_copy, 0.0)

        assert info == 0, f"fb01vd failed with info={info}"

        Rinov_sqrt_upper = np.triu(Rinov_sqrt)

        P_sym = np.triu(P)
        P_sym = P_sym + P_sym.T - np.diag(np.diag(P_sym))
        RINOV_expected = C @ P_sym @ C.T + R

        RINOV_reconstructed = Rinov_sqrt_upper.T @ Rinov_sqrt_upper

        np.testing.assert_allclose(
            RINOV_reconstructed, RINOV_expected, rtol=1e-10, atol=1e-12,
            err_msg="R_out is not the Cholesky factor of RINOV"
        )

    def test_covariance_symmetry_preservation(self):
        """
        Validate that the upper triangular part of P_updated is consistent
        (i.e., P_updated is symmetric when completed).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, l = 5, 3, 2

        tmp = np.random.randn(n, n).astype(float, order='F')
        P = tmp @ tmp.T
        P = np.asfortranarray(P)

        A = np.random.randn(n, n).astype(float, order='F') * 0.5
        B = np.random.randn(n, m).astype(float, order='F') * 0.3

        tmp = np.random.randn(m, m).astype(float, order='F')
        Q = tmp @ tmp.T * 0.1
        Q = np.asfortranarray(Q)

        C = np.random.randn(l, n).astype(float, order='F')

        tmp = np.random.randn(l, l).astype(float, order='F')
        R = tmp @ tmp.T + np.eye(l) * 0.5
        R = np.asfortranarray(R)

        P_copy = P.copy(order='F')
        Q_copy = Q.copy(order='F')
        R_copy = R.copy(order='F')

        P_updated, K, Rinov_sqrt, rcond, info = fb01vd(n, m, l, P_copy, A, B, C, Q_copy, R_copy, 0.0)

        assert info == 0, f"fb01vd failed with info={info}"

        P_updated_sym = np.triu(P_updated)
        P_updated_full = P_updated_sym + P_updated_sym.T - np.diag(np.diag(P_updated_sym))

        eigvals = np.linalg.eigvalsh(P_updated_full)
        assert np.all(eigvals >= -1e-10), "P_updated should be positive semi-definite"


class TestFB01VDEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_dimensions(self):
        """Test with n=0, l=0 (quick return)."""
        n, m, l = 0, 2, 0

        P = np.zeros((0, 0), order='F', dtype=float)
        A = np.zeros((0, 0), order='F', dtype=float)
        B = np.zeros((0, m), order='F', dtype=float)
        Q = np.eye(m, order='F', dtype=float)
        C = np.zeros((0, 0), order='F', dtype=float)
        R = np.zeros((0, 0), order='F', dtype=float)

        P_updated, K, Rinov_sqrt, rcond, info = fb01vd(n, m, l, P, A, B, C, Q, R, 0.0)

        assert info == 0, f"Quick return failed with info={info}"

    def test_scalar_system(self):
        """Test with n=m=l=1 (scalar system)."""
        n, m, l = 1, 1, 1

        P = np.array([[1.0]], order='F', dtype=float)
        A = np.array([[0.9]], order='F', dtype=float)
        B = np.array([[0.5]], order='F', dtype=float)
        Q = np.array([[0.1]], order='F', dtype=float)
        C = np.array([[1.0]], order='F', dtype=float)
        R = np.array([[0.2]], order='F', dtype=float)

        P_copy = P.copy(order='F')
        Q_copy = Q.copy(order='F')
        R_copy = R.copy(order='F')

        P_updated, K, Rinov_sqrt, rcond, info = fb01vd(n, m, l, P_copy, A, B, C, Q_copy, R_copy, 0.0)

        assert info == 0, f"Scalar system failed with info={info}"

        p = P[0, 0]
        a = A[0, 0]
        b = B[0, 0]
        c = C[0, 0]
        q = Q[0, 0]
        r = R[0, 0]

        rinov = c * p * c + r
        k = p * c / rinov
        p_next = a * (p - k * c * p) * a + b * q * b

        np.testing.assert_allclose(K[0, 0], k, rtol=1e-12)
        np.testing.assert_allclose(P_updated[0, 0], p_next, rtol=1e-12)


class TestFB01VDErrorHandling:
    """Test error conditions."""

    def test_singular_innovation_covariance(self):
        """
        Test when innovation covariance RINOV is singular.
        Should return info = l+1.
        """
        n, m, l = 2, 1, 2

        P = np.eye(n, order='F', dtype=float) * 1000
        A = np.eye(n, order='F', dtype=float)
        B = np.zeros((n, m), order='F', dtype=float)
        Q = np.eye(m, order='F', dtype=float)
        C = np.array([[1.0, 0.0], [1.0, 0.0]], order='F', dtype=float)
        R = np.zeros((l, l), order='F', dtype=float)

        P_copy = P.copy(order='F')
        Q_copy = Q.copy(order='F')
        R_copy = R.copy(order='F')

        _, _, _, _, info = fb01vd(n, m, l, P_copy, A, B, C, Q_copy, R_copy, 0.0)

        assert info > 0, "Expected error for singular RINOV"

    def test_non_positive_definite_rinov(self):
        """
        Test when RINOV is not positive definite (Cholesky fails).
        Should return info in [1, l].
        """
        n, m, l = 2, 1, 2

        P = np.zeros((n, n), order='F', dtype=float)
        A = np.eye(n, order='F', dtype=float)
        B = np.zeros((n, m), order='F', dtype=float)
        Q = np.eye(m, order='F', dtype=float)
        C = np.eye(l, n, order='F', dtype=float)
        R = np.array([[-1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)

        P_copy = P.copy(order='F')
        Q_copy = Q.copy(order='F')
        R_copy = R.copy(order='F')

        _, _, _, _, info = fb01vd(n, m, l, P_copy, A, B, C, Q_copy, R_copy, 0.0)

        assert 1 <= info <= l, f"Expected Cholesky failure, got info={info}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
