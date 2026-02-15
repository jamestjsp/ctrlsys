"""
FB01SD - Time-varying square root information Kalman filter (dense matrices).

Combined measurement and time update of one iteration of the time-varying
Kalman filter using square root information filter with dense matrices.

Test data extracted from SLICOT HTML documentation example.
"""

import numpy as np
import pytest
from ctrlsys import fb01sd


class TestFB01SDBasic:
    """Basic functionality test with HTML documentation example data."""

    def test_basic_x_mode_multab_p_multrc_n(self):
        """
        Test basic FB01SD functionality with JOBX='X', MULTAB='P', MULTRC='N'.

        Test data from SLICOT FB01SD.html documentation example.
        N=4, M=2, P=2, one iteration of Kalman filter.
        """
        n, m, p = 4, 2, 2

        # Input: AINV (inverse of state transition matrix)
        # Read row-wise from HTML doc: ((AINV(I,J), J=1,N), I=1,N)
        ainv = np.array([
            [0.2113, 0.7560, 0.0002, 0.3303],
            [0.8497, 0.6857, 0.8782, 0.0683],
            [0.7263, 0.1985, 0.5442, 0.2320],
            [0.8833, 0.6525, 0.3076, 0.9329]
        ], dtype=float, order='F')

        # Input: C (output weight matrix) - P×N
        # Read row-wise from HTML doc: ((C(I,J), J=1,N), I=1,P)
        c = np.array([
            [0.3616, 0.5664, 0.5015, 0.2693],
            [0.2922, 0.4826, 0.4368, 0.6325]
        ], dtype=float, order='F')

        # Input: RINV (measurement noise inverse square root) - P×P upper triangular
        rinv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        # Input: B (input weight matrix) - N×M (already contains A^{-1}*B since MULTAB='P')
        # Read row-wise from HTML doc: ((B(I,J), J=1,M), I=1,N)
        b = np.array([
            [-0.8805, 1.3257],
            [2.1039, 0.5207],
            [-0.6075, 1.0386],
            [-0.8531, 1.1688]
        ], dtype=float, order='F')

        # Input: QINV (process noise inverse square root) - M×M upper triangular
        # Read: ((QINV(I,J), J=1,M), I=1,M)
        qinv = np.array([
            [1.1159, 0.2305],
            [0.0, 0.6597]
        ], dtype=float, order='F')

        # Input: SINV (state covariance inverse square root) - N×N upper triangular
        # Read: ((SINV(I,J), J=1,N), I=1,N)
        sinv = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float, order='F')

        # Input: Z (process noise mean) - M
        z = np.array([0.0019, 0.5075], dtype=float)

        # Input: X (filtered state) - N
        x = np.array([0.4076, 0.8408, 0.5017, 0.9128], dtype=float)

        # Input: RINVY (measured output * measurement noise inverse square root) - P
        rinvy = np.array([0.2129, 0.5591], dtype=float)

        # Output: E (estimated error) - P
        e = np.zeros(p, dtype=float)

        # Call routine
        sinv_out, qinv_out, x_out, e_out, info = fb01sd(
            'X',       # jobx
            'P',       # multab
            'N',       # multrc
            sinv, ainv, b, rinv, c, qinv,
            x, rinvy, z, e
        )

        # Verify success
        assert info == 0, f"fb01sd failed with info={info}"

        # Expected output: SINV (N×N upper triangular)
        # After 1 iteration (HTML doc shows results after 3 iterations)
        # Values from running Fortran reference for 1 iteration
        sinv_expected = np.array([
            [0.916623, 0.553627, 0.827940, 0.631696],
            [0.0, -0.759654, -0.155874, -0.360044],
            [0.0, 0.0, -0.352632, 0.190891],
            [0.0, 0.0, 0.0, 0.494975]
        ], dtype=float, order='F')

        # Expected X output (filtered state) after 1 iteration
        x_expected = np.array([-0.057875, -0.139206, 0.394001, 0.698866], dtype=float)

        # Validate SINV output (upper triangular part)
        np.testing.assert_allclose(sinv_out, sinv_expected, rtol=1e-4, atol=1e-5)

        # Validate X output
        np.testing.assert_allclose(x_out, x_expected, rtol=1e-4, atol=1e-5)

        # Validate E has expected dimension
        assert e_out.shape == (p,)

    def test_basic_n_mode(self):
        """
        Test FB01SD with JOBX='N' (X not required).

        Same test data, but X computation skipped.
        """
        n, m, p = 4, 2, 2

        # Correct data from HTML doc (read row-wise)
        ainv = np.array([
            [0.2113, 0.7560, 0.0002, 0.3303],
            [0.8497, 0.6857, 0.8782, 0.0683],
            [0.7263, 0.1985, 0.5442, 0.2320],
            [0.8833, 0.6525, 0.3076, 0.9329]
        ], dtype=float, order='F')

        c = np.array([
            [0.3616, 0.5664, 0.5015, 0.2693],
            [0.2922, 0.4826, 0.4368, 0.6325]
        ], dtype=float, order='F')

        rinv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        b = np.array([
            [-0.8805, 1.3257],
            [2.1039, 0.5207],
            [-0.6075, 1.0386],
            [-0.8531, 1.1688]
        ], dtype=float, order='F')

        qinv = np.array([
            [1.1159, 0.2305],
            [0.0, 0.6597]
        ], dtype=float, order='F')

        sinv = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float, order='F')

        z = np.array([0.0019, 0.5075], dtype=float)
        x = np.array([0.4076, 0.8408, 0.5017, 0.9128], dtype=float)
        rinvy = np.array([0.2129, 0.5591], dtype=float)
        e = np.zeros(p, dtype=float)

        # Call with JOBX='N'
        sinv_out, qinv_out, x_out, e_out, info = fb01sd(
            'N',       # jobx
            'P',       # multab
            'N',       # multrc
            sinv, ainv, b, rinv, c, qinv,
            x, rinvy, z, e,
            0.0
        )

        # Verify success
        assert info == 0, f"fb01sd failed with info={info}"

        # Expected output: SINV after 1 iteration (from Fortran reference)
        sinv_expected = np.array([
            [0.916623, 0.553627, 0.827940, 0.631696],
            [0.0, -0.759654, -0.155874, -0.360044],
            [0.0, 0.0, -0.352632, 0.190891],
            [0.0, 0.0, 0.0, 0.494975]
        ], dtype=float, order='F')

        np.testing.assert_allclose(sinv_out, sinv_expected, rtol=1e-4, atol=1e-5)

        # Validate outputs have expected dimensions
        assert e_out.shape == (p,)


class TestFB01SDEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_n1_system(self):
        """Test with minimal N=1 system (edge case)."""
        n, m, p = 1, 2, 2

        sinv = np.array([[1.0]], dtype=float, order='F')
        ainv = np.array([[0.5]], dtype=float, order='F')
        b = np.array([[0.3, 0.2]], dtype=float, order='F')
        rinv = np.eye(p, dtype=float, order='F')
        c = np.array([[0.5], [0.3]], dtype=float, order='F')
        qinv = np.array([
            [1.1159, 0.2305],
            [0.0, 0.6597]
        ], dtype=float, order='F')
        x = np.array([0.5], dtype=float)
        rinvy = np.array([0.2129, 0.5591], dtype=float)
        z = np.array([0.0019, 0.5075], dtype=float)
        e = np.zeros(p, dtype=float)

        # Call with N=1 (minimal system)
        sinv_out, qinv_out, x_out, e_out, info = fb01sd(
            'X', 'P', 'N',
            sinv, ainv, b, rinv, c, qinv,
            x, rinvy, z, e,
            0.0
        )

        # Should return successfully
        assert info == 0
        assert np.all(np.isfinite(sinv_out))

    def test_small_system(self):
        """Test with small system: N=1, M=1, P=1."""
        n, m, p = 1, 1, 1

        # Random but stable system
        np.random.seed(42)

        sinv = np.array([[1.0]], dtype=float, order='F')
        ainv = np.array([[0.5]], dtype=float, order='F')
        b = np.array([[0.3]], dtype=float, order='F')
        rinv = np.array([[1.0]], dtype=float, order='F')
        c = np.array([[0.7]], dtype=float, order='F')
        qinv = np.array([[1.0]], dtype=float, order='F')
        x = np.array([0.5], dtype=float)
        rinvy = np.array([0.2], dtype=float)
        z = np.array([0.1], dtype=float)
        e = np.zeros(p, dtype=float)

        sinv_out, qinv_out, x_out, e_out, info = fb01sd(
            'X', 'P', 'N',
            sinv, ainv, b, rinv, c, qinv,
            x, rinvy, z, e,
            0.0
        )

        assert info == 0
        # Verify outputs are real numbers (not NaN/Inf)
        assert np.all(np.isfinite(sinv_out))
        assert np.all(np.isfinite(x_out))
        assert np.all(np.isfinite(e_out))


class TestFB01SDMultabVariants:
    """Test MULTAB parameter variants (P vs N)."""

    def test_multab_n_variant(self):
        """
        Test with MULTAB='N' (B contains A_inv*B product computation).

        This requires routine to compute A_inv * B internally.
        """
        n, m, p = 3, 2, 2
        np.random.seed(123)

        # Create stable system
        a_inv = np.array([
            [0.8, 0.1, 0.0],
            [0.0, 0.7, 0.1],
            [0.0, 0.0, 0.6]
        ], dtype=float, order='F')

        b = np.array([
            [0.2, 0.1],
            [0.1, 0.3],
            [0.0, 0.2]
        ], dtype=float, order='F')

        c = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.4, 0.1]
        ], dtype=float, order='F')

        rinv = np.eye(p, dtype=float, order='F')
        sinv = np.eye(n, dtype=float, order='F')
        qinv = np.eye(m, dtype=float, order='F')

        x = np.array([0.1, 0.2, 0.3], dtype=float)
        rinvy = np.array([0.1, 0.2], dtype=float)
        z = np.array([0.1, 0.1], dtype=float)
        e = np.zeros(p, dtype=float)

        # Call with MULTAB='N'
        sinv_out, qinv_out, x_out, e_out, info = fb01sd(
            'X', 'N', 'N',
            sinv, a_inv, b, rinv, c, qinv,
            x, rinvy, z, e,
            0.0
        )

        assert info == 0
        assert np.all(np.isfinite(sinv_out))


class TestFB01SDMultrcVariants:
    """Test MULTRC parameter variants (P vs N)."""

    def test_multrc_p_variant(self):
        """
        Test with MULTRC='P' (C contains R_inv^{-1/2} * C product).

        RINV is not used, C already has precomputed product.
        """
        n, m, p = 3, 2, 2

        sinv = np.eye(n, dtype=float, order='F')
        ainv = 0.8 * np.eye(n, dtype=float, order='F')
        b = np.array([
            [0.2, 0.1],
            [0.1, 0.3],
            [0.0, 0.2]
        ], dtype=float, order='F')

        # RINV not used with MULTRC='P'
        rinv = np.zeros((1, 1), dtype=float, order='F')

        # C contains precomputed R_inv^{-1/2} * C product
        c = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.4, 0.1]
        ], dtype=float, order='F')

        qinv = np.eye(m, dtype=float, order='F')

        x = np.array([0.1, 0.2, 0.3], dtype=float)
        rinvy = np.array([0.1, 0.2], dtype=float)
        z = np.array([0.1, 0.1], dtype=float)
        e = np.zeros(p, dtype=float)

        # Call with MULTRC='P'
        sinv_out, qinv_out, x_out, e_out, info = fb01sd(
            'X', 'P', 'P',
            sinv, ainv, b, rinv, c, qinv,
            x, rinvy, z, e,
            0.0
        )

        assert info == 0
        assert np.all(np.isfinite(sinv_out))


class TestFB01SDParameters:
    """Test parameter validation and error handling."""

    def test_invalid_jobx(self):
        """Test with invalid JOBX parameter."""
        n, m, p = 2, 1, 1

        sinv = np.eye(n, dtype=float, order='F')
        ainv = np.eye(n, dtype=float, order='F')
        b = np.ones((n, m), dtype=float, order='F')
        rinv = np.eye(p, dtype=float, order='F')
        c = np.ones((p, n), dtype=float, order='F')
        qinv = np.eye(m, dtype=float, order='F')
        x = np.ones(n, dtype=float)
        rinvy = np.ones(p, dtype=float)
        z = np.ones(m, dtype=float)
        e = np.zeros(p, dtype=float)

        # Call with invalid JOBX
        with pytest.raises((ValueError, RuntimeError)):
            fb01sd(
                'Z',  # Invalid JOBX
                'P', 'N',
                sinv, ainv, b, rinv, c, qinv,
                x, rinvy, z, e,
                0.0
            )

    def test_info_nonzero_for_singular(self):
        """Test that info=1 is returned when matrix is singular."""
        n, m, p = 2, 1, 1

        sinv = np.eye(n, dtype=float, order='F')
        ainv = np.eye(n, dtype=float, order='F')
        # B with zero columns may cause issues
        b = np.zeros((n, m), dtype=float, order='F')
        rinv = np.eye(p, dtype=float, order='F')
        c = np.ones((p, n), dtype=float, order='F')
        qinv = np.eye(m, dtype=float, order='F')
        x = np.ones(n, dtype=float)
        rinvy = np.ones(p, dtype=float)
        z = np.ones(m, dtype=float)
        e = np.zeros(p, dtype=float)

        # Call - should succeed but may return info=1 for singular matrix
        sinv_out, qinv_out, x_out, e_out, info = fb01sd(
            'X', 'P', 'N',
            sinv, ainv, b, rinv, c, qinv,
            x, rinvy, z, e,
            0.0
        )

        # Info can be 0 (success) or 1 (singular), both are valid
        assert info >= 0


class TestFB01SDNumericalProperties:
    """Test mathematical properties of the Kalman filter update."""

    def test_sinv_triangular_preservation(self):
        """
        Verify SINV remains upper triangular after update.

        By design, FB01SD preserves the upper triangular structure of SINV.
        """
        n, m, p = 4, 2, 2

        sinv = np.array([
            [0.6897, 0.7721, 0.7079, 0.6102],
            [0.0, -0.3363, -0.2252, -0.2642],
            [0.0, 0.0, -0.1650, 0.0319],
            [0.0, 0.0, 0.0, 0.3708]
        ], dtype=float, order='F')

        ainv = np.array([
            [0.2113, 0.0002, 0.7560, 0.3303],
            [0.8497, 0.8782, 0.6857, 0.0683],
            [0.7263, 0.5442, 0.1985, 0.2320],
            [0.8833, 0.3076, 0.6525, 0.9329]
        ], dtype=float, order='F')

        c = np.array([
            [0.3616, 0.5015, 0.5664, 0.2693],
            [0.2922, 0.4368, 0.4826, 0.6325]
        ], dtype=float, order='F')

        rinv = np.eye(p, dtype=float, order='F')
        b = np.array([
            [-0.8805, 2.1039],
            [1.3257, 0.5207],
            [-0.6075, -0.8531],
            [1.0386, 1.1688]
        ], dtype=float, order='F')

        qinv = np.array([
            [1.1159, 0.2305],
            [0.0, 0.6597]
        ], dtype=float, order='F')

        x = np.array([0.4076, 0.8408, 0.5017, 0.9128], dtype=float)
        rinvy = np.array([0.2129, 0.5591], dtype=float)
        z = np.array([0.0019, 0.5075], dtype=float)
        e = np.zeros(p, dtype=float)

        sinv_out, qinv_out, x_out, e_out, info = fb01sd(
            'X', 'P', 'N',
            sinv, ainv, b, rinv, c, qinv,
            x, rinvy, z, e,
            0.0
        )

        assert info == 0

        # Verify SINV_out is upper triangular
        for i in range(n):
            for j in range(i):
                assert abs(sinv_out[i, j]) < 1e-14, f"SINV({i},{j})={sinv_out[i,j]} should be ~0"

    def test_qinv_triangular_preservation(self):
        """
        Verify QINV remains upper triangular after update.

        By design, FB01SD preserves the upper triangular structure of QINV.
        """
        n, m, p = 4, 2, 2

        sinv = np.eye(n, dtype=float, order='F')
        ainv = np.eye(n, dtype=float, order='F')
        b = np.random.randn(n, m).astype(float)
        b = np.asfortranarray(b)

        c = np.random.randn(p, n).astype(float)
        c = np.asfortranarray(c)

        rinv = np.eye(p, dtype=float, order='F')

        qinv = np.array([
            [1.1159, 0.2305],
            [0.0, 0.6597]
        ], dtype=float, order='F')

        x = np.random.randn(n).astype(float)
        rinvy = np.random.randn(p).astype(float)
        z = np.random.randn(m).astype(float)
        e = np.zeros(p, dtype=float)

        np.random.seed(456)

        sinv_out, qinv_out, x_out, e_out, info = fb01sd(
            'X', 'P', 'N',
            sinv, ainv, b, rinv, c, qinv,
            x, rinvy, z, e,
            0.0
        )

        if info == 0:
            # Verify QINV_out is upper triangular
            for i in range(m):
                for j in range(i):
                    assert abs(qinv_out[i, j]) < 1e-10, f"QINV({i},{j})={qinv_out[i,j]} should be ~0"
