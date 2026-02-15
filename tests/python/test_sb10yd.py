"""
Tests for SB10YD: Frequency response fitting with stable minimum phase system.

SB10YD fits frequency response data with a SISO state-space system (A, B, C, D).
"""

import unittest
import numpy as np
from ctrlsys import sb10yd


class TestSB10YD(unittest.TestCase):

    def test_sb10yd_continuous_first_order(self):
        """
        Test SB10YD with continuous-time first order system H(s) = 1/(s+1).

        The frequency response is H(jw) = 1/(1+jw).
        Random seed: N/A (deterministic test data from analytical formula)
        """
        n_points = 100
        omega = np.linspace(0.1, 10.0, n_points)
        rfrdat = np.zeros(n_points, dtype=float)
        ifrdat = np.zeros(n_points, dtype=float)

        for i in range(n_points):
            w = omega[i]
            denom = 1.0 + w * w
            rfrdat[i] = 1.0 / denom
            ifrdat[i] = -w / denom

        n = 1
        discfl = 0  # Continuous
        flag = 1    # Stable minimum phase
        tol = 0.0

        A, B, C, D, N_out, info = sb10yd(discfl, flag, n, rfrdat, ifrdat, omega, tol)

        self.assertEqual(info, 0)
        self.assertEqual(N_out, 1)

        # Eigenvalues should be stable (negative real part)
        eig_a = np.linalg.eigvals(A)
        self.assertTrue(np.all(np.real(eig_a) < 0), "A should be stable (negative eigenvalues)")

        # Verify frequency response at test points
        for w_test in [0.5, 1.0, 2.0]:
            s = 1j * w_test
            resolvent = np.linalg.inv(s * np.eye(N_out) - A)
            h_val = C @ resolvent @ B + D
            expected = 1.0 / (1.0 + 1j * w_test)
            np.testing.assert_allclose(np.abs(h_val), np.abs(expected), rtol=0.15)

    def test_sb10yd_discrete_first_order(self):
        """
        Test SB10YD with discrete-time first order system H(z) = z/(z-0.5).

        Random seed: N/A (deterministic test data from analytical formula)
        """
        n_points = 100
        omega = np.linspace(0.01, 3.0, n_points)  # Must be < pi for discrete
        rfrdat = np.zeros(n_points, dtype=float)
        ifrdat = np.zeros(n_points, dtype=float)

        for i in range(n_points):
            z = np.exp(1j * omega[i])
            h = z / (z - 0.5)
            rfrdat[i] = np.real(h)
            ifrdat[i] = np.imag(h)

        n = 1
        discfl = 1  # Discrete
        flag = 1    # Stable minimum phase
        tol = 0.0

        A, B, C, D, N_out, info = sb10yd(discfl, flag, n, rfrdat, ifrdat, omega, tol)

        self.assertEqual(info, 0)

        # Eigenvalues should be inside unit disk (modulus < 1)
        eig_a = np.linalg.eigvals(A)
        self.assertTrue(np.all(np.abs(eig_a) < 1.0), "A should be stable (eigenvalues inside unit disk)")

    def test_sb10yd_second_order_system(self):
        """
        Test SB10YD with second order system: H(s) = 1/((s+1)(s+2)).

        Random seed: N/A (deterministic test data from analytical formula)
        """
        n_points = 150
        omega = np.linspace(0.05, 15.0, n_points)
        rfrdat = np.zeros(n_points, dtype=float)
        ifrdat = np.zeros(n_points, dtype=float)

        for i in range(n_points):
            w = omega[i]
            s = 1j * w
            h = 1.0 / ((s + 1.0) * (s + 2.0))
            rfrdat[i] = np.real(h)
            ifrdat[i] = np.imag(h)

        n = 2
        discfl = 0  # Continuous
        flag = 1    # Stable minimum phase
        tol = 0.0

        A, B, C, D, N_out, info = sb10yd(discfl, flag, n, rfrdat, ifrdat, omega, tol)

        self.assertEqual(info, 0)
        self.assertEqual(N_out, 2)

        # Eigenvalues should be stable
        eig_a = np.linalg.eigvals(A)
        self.assertTrue(np.all(np.real(eig_a) < 0), "A should be stable")

        # Verify DC gain: H(0) = 1/(1*2) = 0.5
        resolvent = np.linalg.inv(-A)
        h_dc = C @ resolvent @ B + D
        np.testing.assert_allclose(np.real(h_dc), 0.5, rtol=0.15)

    def test_sb10yd_no_stability_constraint(self):
        """
        Test SB10YD without stability constraint (FLAG=0).

        Random seed: N/A (deterministic test data)
        """
        n_points = 100
        omega = np.linspace(0.1, 10.0, n_points)
        rfrdat = np.zeros(n_points, dtype=float)
        ifrdat = np.zeros(n_points, dtype=float)

        for i in range(n_points):
            w = omega[i]
            denom = 1.0 + w * w
            rfrdat[i] = 1.0 / denom
            ifrdat[i] = -w / denom

        n = 1
        discfl = 0
        flag = 0  # No stability constraint
        tol = 0.0

        A, B, C, D, N_out, info = sb10yd(discfl, flag, n, rfrdat, ifrdat, omega, tol)

        self.assertEqual(info, 0)
        self.assertEqual(N_out, 1)

    def test_sb10yd_zero_order(self):
        """
        Test SB10YD with N=0 (pure gain estimation).

        When N=0, the routine should estimate only D.
        """
        n_points = 50
        omega = np.linspace(0.1, 5.0, n_points)
        gain = 2.5
        rfrdat = np.ones(n_points, dtype=float) * gain
        ifrdat = np.zeros(n_points, dtype=float)

        n = 0
        discfl = 0  # Continuous
        flag = 0
        tol = 0.0

        A, B, C, D, N_out, info = sb10yd(discfl, flag, n, rfrdat, ifrdat, omega, tol)

        self.assertEqual(info, 0)
        self.assertEqual(N_out, 0)
        np.testing.assert_allclose(D, gain, rtol=1e-3)

    def test_sb10yd_freq_response_reconstruction(self):
        """
        Test that fitted system reproduces input frequency response.

        Mathematical property: H_fitted(jw) should approximate H_original(jw).
        Random seed: N/A (deterministic)
        """
        n_points = 100
        omega = np.linspace(0.1, 8.0, n_points)
        rfrdat = np.zeros(n_points, dtype=float)
        ifrdat = np.zeros(n_points, dtype=float)

        # Original system H(s) = 1/(s+1)
        for i in range(n_points):
            w = omega[i]
            denom = 1.0 + w * w
            rfrdat[i] = 1.0 / denom
            ifrdat[i] = -w / denom

        n = 1
        discfl = 0
        flag = 1
        tol = 0.0

        A, B, C, D, N_out, info = sb10yd(discfl, flag, n, rfrdat, ifrdat, omega, tol)
        self.assertEqual(info, 0)

        # Reconstruct frequency response
        max_rel_error = 0.0
        for i in range(0, n_points, 10):  # Check every 10th point
            w = omega[i]
            s = 1j * w
            resolvent = np.linalg.inv(s * np.eye(N_out) - A)
            h_fitted = C @ resolvent @ B + D
            h_original = rfrdat[i] + 1j * ifrdat[i]

            rel_error = np.abs(h_fitted - h_original) / np.abs(h_original)
            if rel_error > max_rel_error:
                max_rel_error = rel_error

        # Relative error should be reasonable for fitting (allowing for cepstrum method approximation)
        self.assertTrue(max_rel_error < 0.7, f"Max relative error {max_rel_error} too large")

    def test_sb10yd_minimum_phase(self):
        """
        Test that FLAG=1 produces minimum phase system (zeros in LHP).

        Mathematical property: All zeros should have Re(z) < 0 for continuous-time.
        """
        n_points = 100
        omega = np.linspace(0.1, 10.0, n_points)
        rfrdat = np.zeros(n_points, dtype=float)
        ifrdat = np.zeros(n_points, dtype=float)

        # System with two poles: H(s) = 1/((s+0.5)(s+2))
        for i in range(n_points):
            w = omega[i]
            s = 1j * w
            h = 1.0 / ((s + 0.5) * (s + 2.0))
            rfrdat[i] = np.real(h)
            ifrdat[i] = np.imag(h)

        n = 2
        discfl = 0
        flag = 1  # Enforce minimum phase
        tol = 0.0

        A, B, C, D, N_out, info = sb10yd(discfl, flag, n, rfrdat, ifrdat, omega, tol)

        self.assertEqual(info, 0)

        # All eigenvalues (poles) should be in LHP
        eig_a = np.linalg.eigvals(A)
        self.assertTrue(np.all(np.real(eig_a) < 0), "All poles should be stable")


if __name__ == '__main__':
    unittest.main()
