"""
Tests for SB16CD: Frequency-weighted coprime controller reduction.

SB16CD computes reduced order controller using coprime factorization
with frequency-weighted B&T model reduction.
"""

import numpy as np
import pytest


class TestSB16CDBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_html_example_continuous_right_coprime(self):
        """
        Test from SB16CD HTML documentation example.

        Continuous-time system with right coprime factorization,
        balancing-free B&T, fixed order NCR=2.
        """
        from ctrlsys import sb16cd

        n, m, p = 8, 1, 1
        ncr_in = 2
        tol = 0.1

        # System matrices A (8x8) - read row-wise from HTML
        a = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.015, 0.765, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.765, -0.015, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.028, 1.41, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.41, -0.028, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04, 1.85],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.85, -0.04]
        ], order='F', dtype=float)

        # B (8x1) - column vector
        b = np.array([
            [0.026],
            [-0.251],
            [0.033],
            [-0.886],
            [-4.017],
            [0.145],
            [3.604],
            [0.280]
        ], order='F', dtype=float)

        # C (1x8) - row vector
        c = np.array([
            [-0.996, -0.105, 0.261, 0.009, -0.001, -0.043, 0.002, -0.026]
        ], order='F', dtype=float)

        # D (1x1)
        d = np.array([[0.0]], order='F', dtype=float)

        # F (1x8) - state feedback gain
        f = np.array([
            [4.472135954999638e-002, 6.610515358414598e-001,
             4.698598960657579e-003, 3.601363251422058e-001,
             1.032530880771415e-001, -3.754055214487997e-002,
             -4.268536964759344e-002, 3.287284547842979e-002]
        ], order='F', dtype=float)

        # G (8x1) - observer gain
        g = np.array([
            [4.108939884667451e-001],
            [8.684600000000012e-002],
            [3.852317308197148e-004],
            [-3.619366874815911e-003],
            [-8.803722876359955e-003],
            [8.420521094001852e-003],
            [1.234944428038507e-003],
            [4.263205617645322e-003]
        ], order='F', dtype=float)

        # Call routine with right coprime factorization
        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'C', 'D', 'F', 'R', 'F',
            a, b, c, d, f, g, ncr_in, tol
        )

        assert info == 0, f"sb16cd failed with info={info}"
        assert ncr == 2, f"Expected NCR=2, got {ncr}"

        # Expected Hankel singular values from HTML
        hsv_expected = np.array([3.3073, 0.7274, 0.1124, 0.0784,
                                 0.0242, 0.0182, 0.0101, 0.0094])
        np.testing.assert_allclose(hsv[:n], hsv_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced controller matrices from HTML (NCR=2)
        ac_expected = np.array([
            [-0.4334, 0.4884],
            [-0.1950, -0.1093]
        ], order='F', dtype=float)

        bc_expected = np.array([
            [-0.4231],
            [-0.1785]
        ], order='F', dtype=float)

        cc_expected = np.array([
            [-0.0326, -0.2307]
        ], order='F', dtype=float)

        np.testing.assert_allclose(ac[:ncr, :ncr], ac_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(bc[:ncr, :p], bc_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(cc[:m, :ncr], cc_expected, rtol=1e-3, atol=1e-4)


class TestSB16CDAutomaticOrder:
    """Tests for automatic order selection."""

    def test_automatic_order_selection(self):
        """
        Test automatic order selection (ORDSEL='A').

        Uses same system but lets algorithm choose order based on tolerance.
        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import sb16cd

        n, m, p = 8, 1, 1
        ncr_in = 0  # Will be determined automatically
        tol = 0.1   # Should give NCR=2 based on HSV threshold

        # Same system matrices as basic test
        a = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.015, 0.765, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.765, -0.015, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.028, 1.41, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.41, -0.028, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04, 1.85],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.85, -0.04]
        ], order='F', dtype=float)

        b = np.array([
            [0.026], [-0.251], [0.033], [-0.886],
            [-4.017], [0.145], [3.604], [0.280]
        ], order='F', dtype=float)

        c = np.array([
            [-0.996, -0.105, 0.261, 0.009, -0.001, -0.043, 0.002, -0.026]
        ], order='F', dtype=float)

        d = np.array([[0.0]], order='F', dtype=float)

        f = np.array([
            [4.472135954999638e-002, 6.610515358414598e-001,
             4.698598960657579e-003, 3.601363251422058e-001,
             1.032530880771415e-001, -3.754055214487997e-002,
             -4.268536964759344e-002, 3.287284547842979e-002]
        ], order='F', dtype=float)

        g = np.array([
            [4.108939884667451e-001], [8.684600000000012e-002],
            [3.852317308197148e-004], [-3.619366874815911e-003],
            [-8.803722876359955e-003], [8.420521094001852e-003],
            [1.234944428038507e-003], [4.263205617645322e-003]
        ], order='F', dtype=float)

        # Automatic order selection
        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'C', 'D', 'F', 'R', 'A',
            a, b, c, d, f, g, ncr_in, tol
        )

        assert info == 0, f"sb16cd failed with info={info}"
        # With TOL=0.1, HSV[2]=0.1124 > TOL, HSV[3]=0.0784 < TOL
        # So NCR should be 3 or possibly 2
        assert 2 <= ncr <= 3, f"Expected NCR in [2,3], got {ncr}"


class TestSB16CDLeftCoprime:
    """Tests for left coprime factorization."""

    def test_left_coprime_factorization(self):
        """
        Test left coprime factorization (JOBCF='L').

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import sb16cd

        n, m, p = 8, 1, 1
        ncr_in = 2
        tol = 0.1

        # Same stable system
        a = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.015, 0.765, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.765, -0.015, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.028, 1.41, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.41, -0.028, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04, 1.85],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.85, -0.04]
        ], order='F', dtype=float)

        b = np.array([
            [0.026], [-0.251], [0.033], [-0.886],
            [-4.017], [0.145], [3.604], [0.280]
        ], order='F', dtype=float)

        c = np.array([
            [-0.996, -0.105, 0.261, 0.009, -0.001, -0.043, 0.002, -0.026]
        ], order='F', dtype=float)

        d = np.array([[0.0]], order='F', dtype=float)

        f = np.array([
            [4.472135954999638e-002, 6.610515358414598e-001,
             4.698598960657579e-003, 3.601363251422058e-001,
             1.032530880771415e-001, -3.754055214487997e-002,
             -4.268536964759344e-002, 3.287284547842979e-002]
        ], order='F', dtype=float)

        g = np.array([
            [4.108939884667451e-001], [8.684600000000012e-002],
            [3.852317308197148e-004], [-3.619366874815911e-003],
            [-8.803722876359955e-003], [8.420521094001852e-003],
            [1.234944428038507e-003], [4.263205617645322e-003]
        ], order='F', dtype=float)

        # Left coprime factorization
        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'C', 'D', 'F', 'L', 'F',
            a, b, c, d, f, g, ncr_in, tol
        )

        assert info == 0, f"sb16cd failed with info={info}"
        assert ncr == 2, f"Expected NCR=2, got {ncr}"

        # Verify HSV are positive and decreasing
        assert hsv[0] > 0, "First HSV should be positive"
        for i in range(ncr - 1):
            assert hsv[i] >= hsv[i + 1], "HSV should be decreasing"


class TestSB16CDDiscreteTime:
    """Tests for discrete-time systems."""

    def test_discrete_time_system(self):
        """
        Test discrete-time system (DICO='D').

        Creates a stable discrete-time system and verifies controller reduction.
        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import sb16cd

        np.random.seed(456)
        n, m, p = 4, 2, 2
        ncr_in = 2
        tol = 0.01

        # Create a stable discrete-time A matrix (eigenvalues inside unit circle)
        # Use a diagonal matrix with eigenvalues < 1
        a = np.diag([0.5, 0.6, 0.7, 0.8]).astype(float, order='F')

        # Random B, C matrices
        b = np.array([
            [0.1, 0.2],
            [0.3, 0.1],
            [0.2, 0.4],
            [0.1, 0.3]
        ], order='F', dtype=float)

        c = np.array([
            [0.1, 0.2, 0.3, 0.1],
            [0.2, 0.1, 0.2, 0.3]
        ], order='F', dtype=float)

        d = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ], order='F', dtype=float)

        # Stabilizing F (state feedback) - make A+B*F stable (discrete)
        # For discrete, need eigenvalues of A+B*F inside unit circle
        f = np.array([
            [-0.1, -0.1, -0.1, -0.1],
            [-0.1, -0.1, -0.1, -0.1]
        ], order='F', dtype=float)

        # Stabilizing G (observer gain) - make A+G*C stable (discrete)
        g = np.array([
            [-0.1, -0.1],
            [-0.1, -0.1],
            [-0.1, -0.1],
            [-0.1, -0.1]
        ], order='F', dtype=float)

        # Call routine
        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'D', 'Z', 'B', 'R', 'F',
            a, b, c, d, f, g, ncr_in, tol
        )

        assert info == 0, f"sb16cd failed with info={info}"
        assert ncr <= n, f"NCR should be <= N, got {ncr}"

        # Verify HSV are positive and decreasing
        for i in range(n - 1):
            if hsv[i + 1] > 0:
                assert hsv[i] >= hsv[i + 1] - 1e-10, "HSV should be decreasing"


class TestSB16CDBalancingMethods:
    """Tests for different balancing methods."""

    def test_square_root_bt_method(self):
        """
        Test square-root B&T method (JOBMR='B').

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import sb16cd

        n, m, p = 8, 1, 1
        ncr_in = 2
        tol = 0.1

        a = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.015, 0.765, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.765, -0.015, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.028, 1.41, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.41, -0.028, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04, 1.85],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.85, -0.04]
        ], order='F', dtype=float)

        b = np.array([
            [0.026], [-0.251], [0.033], [-0.886],
            [-4.017], [0.145], [3.604], [0.280]
        ], order='F', dtype=float)

        c = np.array([
            [-0.996, -0.105, 0.261, 0.009, -0.001, -0.043, 0.002, -0.026]
        ], order='F', dtype=float)

        d = np.array([[0.0]], order='F', dtype=float)

        f = np.array([
            [4.472135954999638e-002, 6.610515358414598e-001,
             4.698598960657579e-003, 3.601363251422058e-001,
             1.032530880771415e-001, -3.754055214487997e-002,
             -4.268536964759344e-002, 3.287284547842979e-002]
        ], order='F', dtype=float)

        g = np.array([
            [4.108939884667451e-001], [8.684600000000012e-002],
            [3.852317308197148e-004], [-3.619366874815911e-003],
            [-8.803722876359955e-003], [8.420521094001852e-003],
            [1.234944428038507e-003], [4.263205617645322e-003]
        ], order='F', dtype=float)

        # Square-root B&T method
        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'C', 'D', 'B', 'R', 'F',
            a, b, c, d, f, g, ncr_in, tol
        )

        assert info == 0, f"sb16cd failed with info={info}"
        assert ncr == 2, f"Expected NCR=2, got {ncr}"

        # Expected HSV should be similar (same system)
        hsv_expected = np.array([3.3073, 0.7274, 0.1124, 0.0784,
                                 0.0242, 0.0182, 0.0101, 0.0094])
        np.testing.assert_allclose(hsv[:n], hsv_expected, rtol=1e-3, atol=1e-4)


class TestSB16CDEdgeCases:
    """Edge case tests."""

    def test_zero_dimension(self):
        """Test with N=0 (quick return)."""
        from ctrlsys import sb16cd

        n, m, p = 0, 1, 1
        ncr_in = 0
        tol = 0.1

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 1), order='F', dtype=float)
        c = np.zeros((1, 1), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        f = np.zeros((1, 1), order='F', dtype=float)
        g = np.zeros((1, 1), order='F', dtype=float)

        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'C', 'Z', 'B', 'R', 'F',
            a, b, c, d, f, g, ncr_in, tol
        )

        assert info == 0
        assert ncr == 0

    def test_ncr_zero_fixed(self):
        """Test with NCR=0 and ORDSEL='F' (quick return)."""
        from ctrlsys import sb16cd

        n, m, p = 4, 1, 1
        ncr_in = 0  # Request zero order
        tol = 0.1

        a = np.eye(4, order='F', dtype=float) * (-0.5)
        b = np.ones((4, 1), order='F', dtype=float)
        c = np.ones((1, 4), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        f = np.ones((1, 4), order='F', dtype=float) * (-0.1)
        g = np.ones((4, 1), order='F', dtype=float) * (-0.1)

        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'C', 'Z', 'B', 'R', 'F',
            a, b, c, d, f, g, ncr_in, tol
        )

        assert info == 0
        assert ncr == 0


class TestSB16CDErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test with invalid DICO parameter."""
        from ctrlsys import sb16cd

        n, m, p = 4, 1, 1
        a = np.eye(4, order='F', dtype=float) * (-0.5)
        b = np.ones((4, 1), order='F', dtype=float)
        c = np.ones((1, 4), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        f = np.ones((1, 4), order='F', dtype=float) * (-0.1)
        g = np.ones((4, 1), order='F', dtype=float) * (-0.1)

        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'X', 'Z', 'B', 'R', 'F',
            a, b, c, d, f, g, 2, 0.1
        )

        assert info == -1, f"Expected info=-1 for invalid DICO, got {info}"

    def test_unstable_a_gc(self):
        """
        Test error when A+G*C is not stable.

        Should return INFO=2.
        """
        from ctrlsys import sb16cd

        n, m, p = 4, 1, 1
        # Continuous: need A+G*C with negative real eigenvalues
        # Create A that's already unstable, G*C can't fix it
        a = np.eye(4, order='F', dtype=float) * 2.0  # Positive eigenvalues
        b = np.ones((4, 1), order='F', dtype=float)
        c = np.ones((1, 4), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        f = np.zeros((1, 4), order='F', dtype=float)  # Zero feedback
        g = np.zeros((4, 1), order='F', dtype=float)  # Zero observer gain

        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'C', 'Z', 'B', 'R', 'F',
            a, b, c, d, f, g, 2, 0.1
        )

        # Should fail with unstable A+G*C (info=2) or A+B*F (info=3)
        assert info in [2, 3], f"Expected info in [2,3] for unstable system, got {info}"


class TestSB16CDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_controller_stability(self):
        """
        Verify reduced controller is stable (Ac has stable eigenvalues).

        For continuous-time: Re(eigenvalues) < 0
        For discrete-time: |eigenvalues| < 1

        Random seed: 888 (for reproducibility)
        """
        from ctrlsys import sb16cd

        n, m, p = 8, 1, 1
        ncr_in = 3
        tol = 0.01

        a = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.015, 0.765, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.765, -0.015, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.028, 1.41, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.41, -0.028, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04, 1.85],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.85, -0.04]
        ], order='F', dtype=float)

        b = np.array([
            [0.026], [-0.251], [0.033], [-0.886],
            [-4.017], [0.145], [3.604], [0.280]
        ], order='F', dtype=float)

        c = np.array([
            [-0.996, -0.105, 0.261, 0.009, -0.001, -0.043, 0.002, -0.026]
        ], order='F', dtype=float)

        d = np.array([[0.0]], order='F', dtype=float)

        f = np.array([
            [4.472135954999638e-002, 6.610515358414598e-001,
             4.698598960657579e-003, 3.601363251422058e-001,
             1.032530880771415e-001, -3.754055214487997e-002,
             -4.268536964759344e-002, 3.287284547842979e-002]
        ], order='F', dtype=float)

        g = np.array([
            [4.108939884667451e-001], [8.684600000000012e-002],
            [3.852317308197148e-004], [-3.619366874815911e-003],
            [-8.803722876359955e-003], [8.420521094001852e-003],
            [1.234944428038507e-003], [4.263205617645322e-003]
        ], order='F', dtype=float)

        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'C', 'D', 'F', 'R', 'F',
            a, b, c, d, f, g, ncr_in, tol
        )

        assert info == 0, f"sb16cd failed with info={info}"

        if ncr > 0:
            # Extract reduced Ac and check stability
            ac_reduced = ac[:ncr, :ncr]
            eigenvalues = np.linalg.eigvals(ac_reduced)

            # For continuous-time, all eigenvalues should have negative real part
            for eig in eigenvalues:
                assert eig.real < 1e-10, f"Reduced controller unstable: eigenvalue {eig}"

    def test_hsv_positive_decreasing(self):
        """
        Verify Hankel singular values are positive and decreasing.

        Random seed: 999 (for reproducibility)
        """
        from ctrlsys import sb16cd

        n, m, p = 8, 1, 1
        ncr_in = 4
        tol = 0.001

        a = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.015, 0.765, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.765, -0.015, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.028, 1.41, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.41, -0.028, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04, 1.85],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.85, -0.04]
        ], order='F', dtype=float)

        b = np.array([
            [0.026], [-0.251], [0.033], [-0.886],
            [-4.017], [0.145], [3.604], [0.280]
        ], order='F', dtype=float)

        c = np.array([
            [-0.996, -0.105, 0.261, 0.009, -0.001, -0.043, 0.002, -0.026]
        ], order='F', dtype=float)

        d = np.array([[0.0]], order='F', dtype=float)

        f = np.array([
            [4.472135954999638e-002, 6.610515358414598e-001,
             4.698598960657579e-003, 3.601363251422058e-001,
             1.032530880771415e-001, -3.754055214487997e-002,
             -4.268536964759344e-002, 3.287284547842979e-002]
        ], order='F', dtype=float)

        g = np.array([
            [4.108939884667451e-001], [8.684600000000012e-002],
            [3.852317308197148e-004], [-3.619366874815911e-003],
            [-8.803722876359955e-003], [8.420521094001852e-003],
            [1.234944428038507e-003], [4.263205617645322e-003]
        ], order='F', dtype=float)

        ac, bc, cc, hsv, ncr, iwarn, info = sb16cd(
            'C', 'D', 'F', 'R', 'F',
            a, b, c, d, f, g, ncr_in, tol
        )

        assert info == 0

        # Check HSV are positive
        for i in range(n):
            assert hsv[i] >= 0, f"HSV[{i}] = {hsv[i]} should be non-negative"

        # Check HSV are decreasing
        for i in range(n - 1):
            assert hsv[i] >= hsv[i + 1] - 1e-10, \
                f"HSV not decreasing: HSV[{i}]={hsv[i]} < HSV[{i+1}]={hsv[i+1]}"
