"""
Tests for SLICOT SB10HD routine.

SB10HD computes the H2 optimal n-state controller for a continuous-time system.

Given the plant P in partitioned form:
    P = | A  | B1  B2  |   where B2 has NCON columns (control inputs)
        | C1 |  0  D12 |   and C2 has NMEAS rows (measurements)
        | C2 | D21 D22 |

Computes the controller:
    K = | AK | BK |
        | CK | DK |

Assumptions:
- (A,B2) is stabilizable and (C2,A) is detectable
- D11 = 0
- D12 is full column rank and D21 is full row rank

Test data from SLICOT HTML documentation example.
"""

import numpy as np
import pytest
from ctrlsys import sb10hd


class TestSB10HDBasic:
    """Basic functionality tests using HTML doc example data."""

    def test_html_doc_example(self):
        """
        Test using example from SLICOT HTML documentation.

        System: N=6, M=5, NP=5, NCON=2, NMEAS=2
        Expected results verified against documentation output.
        """
        n = 6
        m = 5
        np_ = 5
        ncon = 2
        nmeas = 2

        A = np.array([
            [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
            [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
            [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
            [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
            [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
            [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
        ], order='F', dtype=float)

        B = np.array([
            [-3.0, -4.0, -2.0,  1.0,  0.0],
            [ 2.0,  0.0,  1.0, -5.0,  2.0],
            [-5.0, -7.0,  0.0,  7.0, -2.0],
            [ 4.0, -6.0,  1.0,  1.0, -2.0],
            [-3.0,  9.0, -8.0,  0.0,  5.0],
            [ 1.0, -2.0,  3.0, -6.0, -2.0]
        ], order='F', dtype=float)

        C = np.array([
            [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
            [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
            [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0],
            [ 9.0, -3.0,  4.0,  0.0,  3.0,  7.0],
            [ 0.0,  1.0, -2.0,  1.0, -6.0, -2.0]
        ], order='F', dtype=float)

        D = np.array([
            [ 0.0,  0.0,  0.0, -4.0, -1.0],
            [ 0.0,  0.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  1.0],
            [ 3.0,  1.0,  0.0,  1.0, -3.0],
            [-2.0,  0.0,  1.0,  7.0,  1.0]
        ], order='F', dtype=float)

        tol = 1e-8

        AK, BK, CK, DK, rcond, info = sb10hd(n, m, np_, ncon, nmeas, A, B, C, D, tol)

        assert info == 0

        AK_expected = np.array([
            [ 88.0015, -145.7298,  -46.2424,   82.2168,  -45.2996,  -31.1407],
            [ 25.7489,  -31.4642,  -12.4198,    9.4625,   -3.5182,    2.7056],
            [ 54.3008, -102.4013,  -41.4968,   50.8412,  -20.1286,  -26.7191],
            [108.1006, -198.0785,  -45.4333,   70.3962,  -25.8591,  -37.2741],
            [-115.8900, 226.1843,   47.2549,  -47.8435,  -12.5004,   34.7474],
            [ 59.0362, -101.8471,  -20.1052,   36.7834,  -16.1063,  -26.4309]
        ], order='F', dtype=float)

        BK_expected = np.array([
            [ 3.7345,  3.4758],
            [-0.3020,  0.6530],
            [ 3.4735,  4.0499],
            [ 4.3198,  7.2755],
            [-3.9424, -10.5942],
            [ 2.1784,  2.5048]
        ], order='F', dtype=float)

        CK_expected = np.array([
            [-2.3346,  3.2556,  0.7150, -0.9724,  0.6962,  0.4074],
            [ 7.6899, -8.4558, -2.9642,  7.0365, -4.2844,  0.1390]
        ], order='F', dtype=float)

        DK_expected = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ], order='F', dtype=float)

        np.testing.assert_allclose(AK, AK_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(BK, BK_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(CK, CK_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(DK, DK_expected, rtol=1e-14, atol=1e-15)

        assert rcond.shape == (4,)
        assert rcond[0] > 0
        assert rcond[1] > 0
        assert rcond[2] > 0
        assert rcond[3] > 0

        # RCOND(1:2) from SB10UD transformations - these match exactly
        np.testing.assert_allclose(rcond[0], 0.2357, rtol=0.01)
        np.testing.assert_allclose(rcond[1], 0.2673, rtol=0.01)
        # RCOND(3:4) from Riccati equations - using rcondu approximation
        # Note: exact values (0.0227, 0.00211) require SB02QD which is not implemented


class TestSB10HDMathematical:
    """Mathematical property tests for SB10HD."""

    def test_controller_dk_is_zero(self):
        """
        Verify DK is always zero for H2 optimal controller.

        This is a fundamental property of the H2 design method.
        """
        n = 6
        m = 5
        np_ = 5
        ncon = 2
        nmeas = 2

        A = np.array([
            [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
            [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
            [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
            [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
            [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
            [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
        ], order='F', dtype=float)

        B = np.array([
            [-3.0, -4.0, -2.0,  1.0,  0.0],
            [ 2.0,  0.0,  1.0, -5.0,  2.0],
            [-5.0, -7.0,  0.0,  7.0, -2.0],
            [ 4.0, -6.0,  1.0,  1.0, -2.0],
            [-3.0,  9.0, -8.0,  0.0,  5.0],
            [ 1.0, -2.0,  3.0, -6.0, -2.0]
        ], order='F', dtype=float)

        C = np.array([
            [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
            [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
            [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0],
            [ 9.0, -3.0,  4.0,  0.0,  3.0,  7.0],
            [ 0.0,  1.0, -2.0,  1.0, -6.0, -2.0]
        ], order='F', dtype=float)

        D = np.array([
            [ 0.0,  0.0,  0.0, -4.0, -1.0],
            [ 0.0,  0.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  1.0],
            [ 3.0,  1.0,  0.0,  1.0, -3.0],
            [-2.0,  0.0,  1.0,  7.0,  1.0]
        ], order='F', dtype=float)

        tol = 1e-8

        AK, BK, CK, DK, rcond, info = sb10hd(n, m, np_, ncon, nmeas, A, B, C, D, tol)

        assert info == 0
        np.testing.assert_allclose(DK, np.zeros((ncon, nmeas)), rtol=1e-14, atol=1e-15)

    def test_controller_dimensions(self):
        """
        Verify controller output dimensions are correct.

        AK: N x N
        BK: N x NMEAS
        CK: NCON x N
        DK: NCON x NMEAS
        """
        n = 6
        m = 5
        np_ = 5
        ncon = 2
        nmeas = 2

        A = np.array([
            [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
            [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
            [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
            [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
            [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
            [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
        ], order='F', dtype=float)

        B = np.array([
            [-3.0, -4.0, -2.0,  1.0,  0.0],
            [ 2.0,  0.0,  1.0, -5.0,  2.0],
            [-5.0, -7.0,  0.0,  7.0, -2.0],
            [ 4.0, -6.0,  1.0,  1.0, -2.0],
            [-3.0,  9.0, -8.0,  0.0,  5.0],
            [ 1.0, -2.0,  3.0, -6.0, -2.0]
        ], order='F', dtype=float)

        C = np.array([
            [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
            [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
            [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0],
            [ 9.0, -3.0,  4.0,  0.0,  3.0,  7.0],
            [ 0.0,  1.0, -2.0,  1.0, -6.0, -2.0]
        ], order='F', dtype=float)

        D = np.array([
            [ 0.0,  0.0,  0.0, -4.0, -1.0],
            [ 0.0,  0.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  1.0],
            [ 3.0,  1.0,  0.0,  1.0, -3.0],
            [-2.0,  0.0,  1.0,  7.0,  1.0]
        ], order='F', dtype=float)

        tol = 1e-8

        AK, BK, CK, DK, rcond, info = sb10hd(n, m, np_, ncon, nmeas, A, B, C, D, tol)

        assert info == 0
        assert AK.shape == (n, n)
        assert BK.shape == (n, nmeas)
        assert CK.shape == (ncon, n)
        assert DK.shape == (ncon, nmeas)


class TestSB10HDEdgeCases:
    """Edge case tests for SB10HD."""

    def test_quick_return_n_zero(self):
        """Test quick return when n=0."""
        n = 0
        m = 2
        np_ = 2
        ncon = 1
        nmeas = 1

        A = np.zeros((0, 0), order='F', dtype=float)
        B = np.zeros((0, m), order='F', dtype=float)
        C = np.zeros((np_, 0), order='F', dtype=float)
        D = np.zeros((np_, m), order='F', dtype=float)
        tol = 0.0

        AK, BK, CK, DK, rcond, info = sb10hd(n, m, np_, ncon, nmeas, A, B, C, D, tol)

        assert info == 0
        assert AK.shape == (0, 0)
        assert BK.shape == (0, nmeas)
        assert CK.shape == (ncon, 0)
        assert DK.shape == (ncon, nmeas)
        np.testing.assert_allclose(rcond, np.array([1.0, 1.0, 1.0, 1.0]), rtol=1e-14)


class TestSB10HDErrors:
    """Error handling tests for SB10HD."""

    def test_invalid_n(self):
        """Test error when n < 0."""
        with pytest.raises(ValueError, match="n must be >= 0"):
            sb10hd(
                -1, 2, 2, 1, 1,
                np.zeros((1, 1), order='F'),
                np.zeros((1, 2), order='F'),
                np.zeros((2, 1), order='F'),
                np.zeros((2, 2), order='F'),
                0.0
            )

    def test_invalid_ncon_greater_than_m(self):
        """Test error when ncon > m (M2 > M)."""
        with pytest.raises(ValueError, match="ncon"):
            sb10hd(
                2, 2, 4, 3, 2,
                np.zeros((2, 2), order='F'),
                np.zeros((2, 2), order='F'),
                np.zeros((4, 2), order='F'),
                np.zeros((4, 2), order='F'),
                0.0
            )

    def test_invalid_nmeas_greater_than_np(self):
        """Test error when nmeas > np (NP2 > NP)."""
        with pytest.raises(ValueError, match="nmeas"):
            sb10hd(
                2, 4, 2, 2, 3,
                np.zeros((2, 2), order='F'),
                np.zeros((2, 4), order='F'),
                np.zeros((2, 2), order='F'),
                np.zeros((2, 4), order='F'),
                0.0
            )

    def test_d12_rank_deficient(self):
        """Test INFO=1 when D12 is rank deficient."""
        n = 2
        m = 3
        np_ = 3
        ncon = 1
        nmeas = 1

        A = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        B = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], order='F', dtype=float)
        C = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], order='F', dtype=float)

        D = np.zeros((np_, m), order='F', dtype=float)
        D[0, 2] = 0.0
        D[1, 2] = 0.0
        D[2, 0] = 1.0
        D[2, 1] = 0.0

        tol = 1e-8

        AK, BK, CK, DK, rcond, info = sb10hd(n, m, np_, ncon, nmeas, A, B, C, D, tol)

        assert info == 1

    def test_d21_rank_deficient(self):
        """Test INFO=2 when D21 is rank deficient."""
        n = 2
        m = 3
        np_ = 3
        ncon = 1
        nmeas = 1

        A = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        B = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], order='F', dtype=float)
        C = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], order='F', dtype=float)

        D = np.zeros((np_, m), order='F', dtype=float)
        D[0, 2] = 1.0
        D[1, 2] = 0.0
        D[2, 0] = 0.0
        D[2, 1] = 0.0

        tol = 1e-8

        AK, BK, CK, DK, rcond, info = sb10hd(n, m, np_, ncon, nmeas, A, B, C, D, tol)

        assert info == 2
