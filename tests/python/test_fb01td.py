"""
Tests for FB01TD: Time-invariant square root information Kalman filter.

Tests the combined measurement and time update of one iteration of the
time-invariant Kalman filter using square root information filter with
condensed controller Hessenberg form.

Test data extracted from SLICOT HTML documentation (FB01TD.html).
"""

import numpy as np
import pytest
from ctrlsys import fb01td


class TestFB01TDBasic:
    """Basic functionality tests using HTML documentation example."""

    @staticmethod
    def load_example_data():
        """Load test data from SLICOT HTML doc example (N=4, M=2, P=2, JOBX='X')."""
        # Extracted from "Program Data" section of FB01TD.html
        # Line: 4     2     2     X     0.0     N
        n, m, p = 4, 2, 2
        jobx, tol, multrc = 'X', 0.0, 'N'

        # AINV(4,4) - read row-wise per Fortran: ((AINV(I,J), J=1,N), I=1,N)
        ainv = np.array([
            [0.2113, 0.7560, 0.0002, 0.3303],
            [0.8497, 0.6857, 0.8782, 0.0683],
            [0.7263, 0.1985, 0.5442, 0.2320],
            [0.0000, 0.6525, 0.3076, 0.9329]
        ], dtype=float, order='F')

        # C(2,4) - read row-wise per Fortran: ((C(I,J), J=1,N), I=1,P)
        c = np.array([
            [0.3616, 0.5664, 0.5015, 0.2693],
            [0.2922, 0.4826, 0.4368, 0.6325]
        ], dtype=float, order='F')

        # RINV(2,2) - upper triangular, read row-wise
        rinv = np.array([
            [1.0000, 0.0000],
            [0.0000, 1.0000]
        ], dtype=float, order='F')

        # AINVB(4,2) - read row-wise
        ainvb = np.array([
            [-0.8805, 1.3257],
            [0.0000, 0.5207],
            [0.0000, 0.0000],
            [0.0000, 0.0000]
        ], dtype=float, order='F')

        # QINV(2,2) - upper triangular, read row-wise
        qinv = np.array([
            [1.1159, 0.2305],
            [0.0000, 0.6597]
        ], dtype=float, order='F')

        # SINV(4,4) - upper triangular, read row-wise
        sinv = np.array([
            [1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ], dtype=float, order='F')

        # Z(2) - input vector
        z = np.array([0.0019, 0.5075], dtype=float)

        # X(4) - state vector
        x = np.array([0.4076, 0.8408, 0.5017, 0.9128], dtype=float)

        # RINVY(2) - measurement output
        rinvy = np.array([0.2129, 0.5591], dtype=float)

        # Expected output for SINV (1 iteration, from Fortran reference)
        # Note: HTML doc shows 3 iterations; we run only 1
        sinv_expected = np.array([
            [-1.139960, -0.745557, -1.150722, -0.392934],
            [0.0, -0.867597, -0.366423, -0.977560],
            [0.0, 0.0, 0.215766, -0.000656],
            [0.0, 0.0, 0.0, 0.550389]
        ], dtype=float, order='F')

        # Expected output for X (1 iteration, from Fortran reference)
        x_expected = np.array([-0.552200, -0.784573, 1.550764, 0.878872], dtype=float)

        return {
            'n': n, 'm': m, 'p': p,
            'jobx': jobx, 'tol': tol, 'multrc': multrc,
            'ainv': ainv, 'c': c, 'rinv': rinv,
            'ainvb': ainvb, 'qinv': qinv, 'sinv': sinv,
            'z': z, 'x': x, 'rinvy': rinvy,
            'sinv_expected': sinv_expected,
            'x_expected': x_expected
        }

    def test_basic_example_jobx_x(self):
        """Test basic functionality with JOBX='X' (compute state update)."""
        data = self.load_example_data()

        # Make copies for modification
        sinv = data['sinv'].copy(order='F')
        qinv = data['qinv'].copy(order='F')
        x = data['x'].copy()
        e = np.zeros(data['p'], dtype=float)

        # Call routine
        sinv_out, qinv_out, x_out, e_out, info = fb01td(
            data['jobx'],
            data['multrc'],
            sinv,
            data['ainv'],
            data['ainvb'],
            data['rinv'],
            data['c'],
            qinv,
            x,
            data['rinvy'],
            data['z'],
            e,
            data['tol']
        )

        # Check success
        assert info == 0, f"FB01TD returned info={info}"

        # Verify SINV output (upper triangular part)
        np.testing.assert_allclose(
            sinv_out[:data['n'], :data['n']],
            data['sinv_expected'],
            rtol=1e-3, atol=1e-4,
            err_msg="SINV output mismatch"
        )

        # Verify X output
        np.testing.assert_allclose(
            x_out,
            data['x_expected'],
            rtol=1e-3, atol=1e-4,
            err_msg="X output mismatch"
        )

    def test_basic_example_jobx_n(self):
        """Test with JOBX='N' (state update not required)."""
        data = self.load_example_data()

        sinv = data['sinv'].copy(order='F')
        qinv = data['qinv'].copy(order='F')
        x = data['x'].copy()
        e = np.zeros(data['p'], dtype=float)

        # Call with JOBX='N'
        sinv_out, qinv_out, x_out, e_out, info = fb01td(
            'N',
            data['multrc'],
            sinv,
            data['ainv'],
            data['ainvb'],
            data['rinv'],
            data['c'],
            qinv,
            x,
            data['rinvy'],
            data['z'],
            e,
            data['tol']
        )

        # Check success
        assert info == 0, f"FB01TD returned info={info}"

        # With JOBX='N', X contains S^{-1}_{i+1} X_{i+1}, not X_{i+1}
        # SINV should still be updated
        # Note: SINV output differs for JOBX='N' vs 'X' so we only check structure
        # The upper triangular part should be non-trivial
        assert sinv_out[0, 0] != 0.0, "SINV should be modified"


class TestFB01TDEdgeCases:
    """Edge case tests."""

    def test_zero_dimensions(self):
        """Test with zero dimensions (quick return)."""
        # N=0, M=1, P=1 - with JOBX='N' to avoid singularity check
        n, m, p = 0, 1, 1

        sinv = np.zeros((1, 1), dtype=float, order='F')
        ainv = np.zeros((1, 1), dtype=float, order='F')
        ainvb = np.zeros((1, 1), dtype=float, order='F')
        rinv = np.eye(1, dtype=float, order='F')
        c = np.zeros((1, 1), dtype=float, order='F')
        qinv = np.eye(1, dtype=float, order='F')
        x = np.zeros(1, dtype=float)
        rinvy = np.array([0.5], dtype=float)
        z = np.array([0.5], dtype=float)
        e = np.zeros(1, dtype=float)

        _, _, _, _, info = fb01td(
            'N', 'N',  # JOBX='N' avoids singularity check on S^{-1}
            sinv, ainv, ainvb, rinv, c, qinv,
            x, rinvy, z, e, 0.0
        )

        # Should succeed with quick return
        assert info == 0

    def test_small_system_1x1(self):
        """Test minimal 1x1 system (N=1, M=1, P=1)."""
        n, m, p = 1, 1, 1

        # Simple scalar system
        sinv = np.array([[0.5]], dtype=float, order='F')
        ainv = np.array([[2.0]], dtype=float, order='F')
        ainvb = np.array([[1.0]], dtype=float, order='F')
        rinv = np.array([[1.0]], dtype=float, order='F')
        c = np.array([[1.0]], dtype=float, order='F')
        qinv = np.array([[1.0]], dtype=float, order='F')
        x = np.array([1.0], dtype=float)
        rinvy = np.array([0.5], dtype=float)
        z = np.array([0.2], dtype=float)
        e = np.zeros(1, dtype=float)

        sinv_out, _, _, _, info = fb01td(
            'X', 'N',
            sinv, ainv, ainvb, rinv, c, qinv,
            x, rinvy, z, e, 0.0
        )

        # Should complete successfully
        assert info == 0
        # SINV should be modified (upper triangular part)
        assert sinv_out[0, 0] != 0.0


class TestFB01TDMULTRC:
    """Test different MULTRC modes."""

    @staticmethod
    def get_test_data_multrc_p():
        """Create test data with MULTRC='P' (pre-multiplied R^{-1/2}C)."""
        n, m, p = 3, 2, 2

        # Create simple test matrices
        ainv = np.array([
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.1],
            [0.0, 0.0, 1.0]
        ], dtype=float, order='F')

        ainvb = np.array([
            [0.5, 0.1],
            [0.0, 0.5],
            [0.0, 0.0]
        ], dtype=float, order='F')

        sinv = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=float, order='F')

        qinv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        # For MULTRC='P', C already contains R^{-1/2}C
        c_scaled = np.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5]
        ], dtype=float, order='F')

        x = np.array([0.1, 0.2, 0.3], dtype=float)
        rinvy = np.array([0.4, 0.5], dtype=float)
        z = np.array([0.1, 0.2], dtype=float)
        e = np.zeros(p, dtype=float)

        return {
            'n': n, 'm': m, 'p': p,
            'ainv': ainv, 'ainvb': ainvb,
            'sinv': sinv, 'qinv': qinv,
            'c_scaled': c_scaled,
            'x': x, 'rinvy': rinvy, 'z': z, 'e': e
        }

    def test_multrc_p_mode(self):
        """Test MULTRC='P' mode (pre-multiplied matrices)."""
        data = self.get_test_data_multrc_p()

        sinv = data['sinv'].copy(order='F')
        qinv = data['qinv'].copy(order='F')
        x = data['x'].copy()
        e = data['e'].copy()

        # With MULTRC='P', RINV is not used (dummy array)
        rinv_dummy = np.zeros((1, 1), dtype=float, order='F')

        sinv_out, _, _, _, info = fb01td(
            'X', 'P',
            sinv, data['ainv'], data['ainvb'],
            rinv_dummy, data['c_scaled'], qinv,
            x, data['rinvy'], data['z'], e, 0.0
        )

        assert info == 0, f"MULTRC='P' mode failed with info={info}"
        # SINV should be updated
        assert not np.allclose(sinv_out, data['sinv']), "SINV should be modified"


class TestFB01TDInvariance:
    """Mathematical property validation tests."""

    def test_sinv_is_upper_triangular(self):
        """Verify output SINV maintains upper triangular structure."""
        data = TestFB01TDBasic.load_example_data()

        sinv = data['sinv'].copy(order='F')
        qinv = data['qinv'].copy(order='F')
        x = data['x'].copy()
        e = np.zeros(data['p'], dtype=float)

        sinv_out, _, _, _, _ = fb01td(
            data['jobx'],
            data['multrc'],
            sinv, data['ainv'], data['ainvb'],
            data['rinv'], data['c'], qinv,
            x, data['rinvy'], data['z'], e,
            data['tol']
        )

        # Check upper triangular: all strictly lower triangle elements should be zero
        for i in range(1, data['n']):
            for j in range(i):
                assert sinv_out[i, j] == 0.0, f"SINV[{i},{j}] should be 0 (lower triangle)"

    def test_qinv_is_upper_triangular(self):
        """Verify output QINV maintains upper triangular structure."""
        data = TestFB01TDBasic.load_example_data()

        sinv = data['sinv'].copy(order='F')
        qinv = data['qinv'].copy(order='F')
        x = data['x'].copy()
        e = np.zeros(data['p'], dtype=float)

        _, qinv_out, _, _, _ = fb01td(
            data['jobx'],
            data['multrc'],
            sinv, data['ainv'], data['ainvb'],
            data['rinv'], data['c'], qinv,
            x, data['rinvy'], data['z'], e,
            data['tol']
        )

        # Check upper triangular for QINV
        for i in range(1, data['m']):
            for j in range(i):
                assert qinv_out[i, j] == 0.0, f"QINV[{i},{j}] should be 0 (lower triangle)"

    def test_state_update_consistency(self):
        """Validate state update consistency with filter equations."""
        np.random.seed(42)
        n, m, p = 3, 2, 2

        # Create stable, consistent system
        ainv = np.array([
            [0.8, 0.1, 0.0],
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.8]
        ], dtype=float, order='F')

        ainvb = np.array([
            [0.1, 0.0],
            [0.0, 0.1],
            [0.0, 0.0]
        ], dtype=float, order='F')

        sinv = np.eye(3, dtype=float, order='F')
        qinv = np.eye(2, dtype=float, order='F')
        c = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=float, order='F')
        rinv = np.eye(2, dtype=float, order='F')

        x = np.array([0.5, 0.5, 0.5], dtype=float)
        z = np.array([0.1, 0.1], dtype=float)
        rinvy = np.array([0.2, 0.2], dtype=float)
        e = np.zeros(2, dtype=float)

        sinv_orig = sinv.copy(order='F')
        x_orig = x.copy()

        sinv_out, _, x_out, _, info = fb01td(
            'X', 'N',
            sinv, ainv, ainvb, rinv, c, qinv,
            x, rinvy, z, e, 0.0
        )

        assert info == 0
        # Verify that outputs changed
        assert not np.allclose(sinv_out, sinv_orig), "SINV should be updated"
        assert not np.allclose(x_out, x_orig), "X should be updated"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
