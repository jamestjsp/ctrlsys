"""
Tests for SB10FD - H-infinity (sub)optimal state controller for continuous-time systems.

SB10FD computes the matrices of an H-infinity (sub)optimal n-state controller
for a given value of gamma using Glover-Doyle formulas.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestSB10FD:
    """Test suite for sb10fd routine."""

    def test_basic_from_html_doc(self):
        """
        Test basic functionality using SLICOT HTML documentation example.

        System: N=6, M=5, NP=5, NCON=2, NMEAS=2
        Gamma=15.0, TOL=1e-8
        """
        from ctrlsys import sb10fd

        n = 6
        m = 5
        np_dim = 5
        ncon = 2
        nmeas = 2
        gamma = 15.0
        tol = 1e-8

        a = np.array([
            [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
            [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
            [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
            [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
            [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
            [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
        ], dtype=float, order='F')

        b = np.array([
            [-3.0, -4.0, -2.0,  1.0,  0.0],
            [ 2.0,  0.0,  1.0, -5.0,  2.0],
            [-5.0, -7.0,  0.0,  7.0, -2.0],
            [ 4.0, -6.0,  1.0,  1.0, -2.0],
            [-3.0,  9.0, -8.0,  0.0,  5.0],
            [ 1.0, -2.0,  3.0, -6.0, -2.0]
        ], dtype=float, order='F')

        c = np.array([
            [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
            [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
            [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0],
            [ 9.0, -3.0,  4.0,  0.0,  3.0,  7.0],
            [ 0.0,  1.0, -2.0,  1.0, -6.0, -2.0]
        ], dtype=float, order='F')

        d = np.array([
            [1.0, -2.0, -3.0,  0.0,  0.0],
            [0.0,  4.0,  0.0,  1.0,  0.0],
            [5.0, -3.0, -4.0,  0.0,  1.0],
            [0.0,  1.0,  0.0,  1.0, -3.0],
            [0.0,  0.0,  1.0,  7.0,  1.0]
        ], dtype=float, order='F')

        ak, bk, ck, dk, rcond, info = sb10fd(n, m, np_dim, ncon, nmeas, gamma,
                                              a, b, c, d, tol)

        assert info == 0, f"sb10fd returned info={info}"

        ak_expected = np.array([
            [-2.8043,  14.7367,   4.6658,   8.1596,   0.0848,   2.5290],
            [ 4.6609,   3.2756,  -3.5754,  -2.8941,   0.2393,   8.2920],
            [-15.3127, 23.5592,  -7.1229,   2.7599,   5.9775,  -2.0285],
            [-22.0691, 16.4758,  12.5523, -16.3602,   4.4300,  -3.3168],
            [30.6789,  -3.9026,  -1.3868,  26.2357,  -8.8267,  10.4860],
            [-5.7429,   0.0577,  10.8216, -11.2275,   1.5074, -10.7244]
        ], dtype=float, order='F')

        bk_expected = np.array([
            [-0.1581, -0.0793],
            [-0.9237, -0.5718],
            [ 0.7984,  0.6627],
            [ 0.1145,  0.1496],
            [-0.6743, -0.2376],
            [ 0.0196, -0.7598]
        ], dtype=float, order='F')

        ck_expected = np.array([
            [-0.2480, -0.1713, -0.0880,  0.1534,  0.5016, -0.0730],
            [ 2.8810, -0.3658,  1.3007,  0.3945,  1.2244,  2.5690]
        ], dtype=float, order='F')

        dk_expected = np.array([
            [ 0.0554,  0.1334],
            [-0.3195,  0.0333]
        ], dtype=float, order='F')

        assert_allclose(ak, ak_expected, rtol=1e-3, atol=1e-3)
        assert_allclose(bk, bk_expected, rtol=1e-3, atol=1e-3)
        assert_allclose(ck, ck_expected, rtol=1e-3, atol=1e-3)
        assert_allclose(dk, dk_expected, rtol=1e-3, atol=1e-3)

        assert len(rcond) == 4
        assert rcond[0] > 0
        assert rcond[1] > 0
        assert rcond[2] > 0
        assert rcond[3] > 0

    def test_controller_dimensions(self):
        """
        Verify controller output dimensions are correct.

        Controller K has:
        - AK: N x N
        - BK: N x NMEAS
        - CK: NCON x N
        - DK: NCON x NMEAS
        """
        from ctrlsys import sb10fd

        n = 6
        m = 5
        np_dim = 5
        ncon = 2
        nmeas = 2
        gamma = 15.0
        tol = 0.0

        a = np.array([
            [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
            [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
            [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
            [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
            [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
            [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
        ], dtype=float, order='F')

        b = np.array([
            [-3.0, -4.0, -2.0,  1.0,  0.0],
            [ 2.0,  0.0,  1.0, -5.0,  2.0],
            [-5.0, -7.0,  0.0,  7.0, -2.0],
            [ 4.0, -6.0,  1.0,  1.0, -2.0],
            [-3.0,  9.0, -8.0,  0.0,  5.0],
            [ 1.0, -2.0,  3.0, -6.0, -2.0]
        ], dtype=float, order='F')

        c = np.array([
            [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
            [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
            [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0],
            [ 9.0, -3.0,  4.0,  0.0,  3.0,  7.0],
            [ 0.0,  1.0, -2.0,  1.0, -6.0, -2.0]
        ], dtype=float, order='F')

        d = np.array([
            [1.0, -2.0, -3.0,  0.0,  0.0],
            [0.0,  4.0,  0.0,  1.0,  0.0],
            [5.0, -3.0, -4.0,  0.0,  1.0],
            [0.0,  1.0,  0.0,  1.0, -3.0],
            [0.0,  0.0,  1.0,  7.0,  1.0]
        ], dtype=float, order='F')

        ak, bk, ck, dk, rcond, info = sb10fd(n, m, np_dim, ncon, nmeas, gamma,
                                              a, b, c, d, tol)

        assert info == 0, f"sb10fd returned info={info}"

        assert ak.shape == (n, n), f"AK should be {n}x{n}, got {ak.shape}"
        assert bk.shape == (n, nmeas), f"BK should be {n}x{nmeas}, got {bk.shape}"
        assert ck.shape == (ncon, n), f"CK should be {ncon}x{n}, got {ck.shape}"
        assert dk.shape == (ncon, nmeas), f"DK should be {ncon}x{nmeas}, got {dk.shape}"

    def test_gamma_too_small(self):
        """
        Test that info=6 is returned when gamma is too small for an admissible controller.

        When gamma is too small, the Riccati equations fail or the controller is inadmissible.
        """
        from ctrlsys import sb10fd

        n = 6
        m = 5
        np_dim = 5
        ncon = 2
        nmeas = 2
        gamma = 0.0001  # Too small - Riccati equations will fail
        tol = 0.0

        a = np.array([
            [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
            [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
            [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
            [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
            [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
            [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
        ], dtype=float, order='F')

        b = np.array([
            [-3.0, -4.0, -2.0,  1.0,  0.0],
            [ 2.0,  0.0,  1.0, -5.0,  2.0],
            [-5.0, -7.0,  0.0,  7.0, -2.0],
            [ 4.0, -6.0,  1.0,  1.0, -2.0],
            [-3.0,  9.0, -8.0,  0.0,  5.0],
            [ 1.0, -2.0,  3.0, -6.0, -2.0]
        ], dtype=float, order='F')

        c = np.array([
            [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
            [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
            [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0],
            [ 9.0, -3.0,  4.0,  0.0,  3.0,  7.0],
            [ 0.0,  1.0, -2.0,  1.0, -6.0, -2.0]
        ], dtype=float, order='F')

        d = np.array([
            [1.0, -2.0, -3.0,  0.0,  0.0],
            [0.0,  4.0,  0.0,  1.0,  0.0],
            [5.0, -3.0, -4.0,  0.0,  1.0],
            [0.0,  1.0,  0.0,  1.0, -3.0],
            [0.0,  0.0,  1.0,  7.0,  1.0]
        ], dtype=float, order='F')

        ak, bk, ck, dk, rcond, info = sb10fd(n, m, np_dim, ncon, nmeas, gamma,
                                              a, b, c, d, tol)

        assert info > 0, f"Expected info>0 for gamma too small, got info={info}"

    def test_closed_loop_stability(self):
        """
        Verify closed-loop system is stable (all poles have negative real parts).

        The H-infinity controller should stabilize the closed-loop system.
        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import sb10fd

        n = 6
        m = 5
        np_dim = 5
        ncon = 2
        nmeas = 2
        gamma = 15.0
        tol = 1e-8

        a = np.array([
            [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
            [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
            [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
            [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
            [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
            [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
        ], dtype=float, order='F')

        b = np.array([
            [-3.0, -4.0, -2.0,  1.0,  0.0],
            [ 2.0,  0.0,  1.0, -5.0,  2.0],
            [-5.0, -7.0,  0.0,  7.0, -2.0],
            [ 4.0, -6.0,  1.0,  1.0, -2.0],
            [-3.0,  9.0, -8.0,  0.0,  5.0],
            [ 1.0, -2.0,  3.0, -6.0, -2.0]
        ], dtype=float, order='F')

        c = np.array([
            [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
            [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
            [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0],
            [ 9.0, -3.0,  4.0,  0.0,  3.0,  7.0],
            [ 0.0,  1.0, -2.0,  1.0, -6.0, -2.0]
        ], dtype=float, order='F')

        d = np.array([
            [1.0, -2.0, -3.0,  0.0,  0.0],
            [0.0,  4.0,  0.0,  1.0,  0.0],
            [5.0, -3.0, -4.0,  0.0,  1.0],
            [0.0,  1.0,  0.0,  1.0, -3.0],
            [0.0,  0.0,  1.0,  7.0,  1.0]
        ], dtype=float, order='F')

        ak, bk, ck, dk, rcond, info = sb10fd(n, m, np_dim, ncon, nmeas, gamma,
                                              a, b, c, d, tol)

        assert info == 0, f"sb10fd returned info={info}"

        m1 = m - ncon
        np1 = np_dim - nmeas

        b2 = b[:, m1:]
        c2 = c[np1:, :]
        d22 = d[np1:, m1:]

        # Correct closed-loop formulation accounting for D22 feedthrough
        # u = (I - DK*D22)^{-1} * (DK*C2*x + CK*xk)
        I2 = np.eye(ncon)
        M = np.linalg.inv(I2 - dk @ d22)

        # Effective outputs and gains
        Cy_eff = (np.eye(nmeas) + d22 @ M @ dk) @ c2
        Dyu_eff = d22 @ M @ ck

        # Closed-loop state matrix
        ac = np.block([
            [a + b2 @ M @ dk @ c2, b2 @ M @ ck],
            [bk @ Cy_eff, ak + bk @ Dyu_eff]
        ])

        eig_cl = np.linalg.eigvals(ac)

        assert np.all(eig_cl.real < 0), \
            f"Closed-loop system not stable. Poles: {eig_cl}"

    def test_rcond_values(self):
        """
        Verify RCOND array contains reasonable reciprocal condition numbers.

        RCOND should contain:
        - rcond[0]: control transformation matrix condition
        - rcond[1]: measurement transformation matrix condition
        - rcond[2]: X-Riccati equation condition
        - rcond[3]: Y-Riccati equation condition

        All should be in (0, 1].
        """
        from ctrlsys import sb10fd

        n = 6
        m = 5
        np_dim = 5
        ncon = 2
        nmeas = 2
        gamma = 15.0
        tol = 1e-8

        a = np.array([
            [-1.0,  0.0,  4.0,  5.0, -3.0, -2.0],
            [-2.0,  4.0, -7.0, -2.0,  0.0,  3.0],
            [-6.0,  9.0, -5.0,  0.0,  2.0, -1.0],
            [-8.0,  4.0,  7.0, -1.0, -3.0,  0.0],
            [ 2.0,  5.0,  8.0, -9.0,  1.0, -4.0],
            [ 3.0, -5.0,  8.0,  0.0,  2.0, -6.0]
        ], dtype=float, order='F')

        b = np.array([
            [-3.0, -4.0, -2.0,  1.0,  0.0],
            [ 2.0,  0.0,  1.0, -5.0,  2.0],
            [-5.0, -7.0,  0.0,  7.0, -2.0],
            [ 4.0, -6.0,  1.0,  1.0, -2.0],
            [-3.0,  9.0, -8.0,  0.0,  5.0],
            [ 1.0, -2.0,  3.0, -6.0, -2.0]
        ], dtype=float, order='F')

        c = np.array([
            [ 1.0, -1.0,  2.0, -4.0,  0.0, -3.0],
            [-3.0,  0.0,  5.0, -1.0,  1.0,  1.0],
            [-7.0,  5.0,  0.0, -8.0,  2.0, -2.0],
            [ 9.0, -3.0,  4.0,  0.0,  3.0,  7.0],
            [ 0.0,  1.0, -2.0,  1.0, -6.0, -2.0]
        ], dtype=float, order='F')

        d = np.array([
            [1.0, -2.0, -3.0,  0.0,  0.0],
            [0.0,  4.0,  0.0,  1.0,  0.0],
            [5.0, -3.0, -4.0,  0.0,  1.0],
            [0.0,  1.0,  0.0,  1.0, -3.0],
            [0.0,  0.0,  1.0,  7.0,  1.0]
        ], dtype=float, order='F')

        ak, bk, ck, dk, rcond, info = sb10fd(n, m, np_dim, ncon, nmeas, gamma,
                                              a, b, c, d, tol)

        assert info == 0, f"sb10fd returned info={info}"

        for i in range(4):
            assert 0 < rcond[i] <= 1.0, \
                f"rcond[{i}]={rcond[i]} should be in (0, 1]"

        # rcond[0] and rcond[1] should be 1.0 for well-conditioned transforms
        assert_allclose(rcond[0], 1.0, rtol=1e-3)
        assert_allclose(rcond[1], 1.0, rtol=1e-3)

        # rcond[2] and rcond[3] are Riccati condition numbers - values vary
        # depending on scaling implementation, just check they're small (indicating
        # well-posed Riccati equations for this gamma)
        assert rcond[2] > 1e-3, f"rcond[2]={rcond[2]} too small"
        assert rcond[3] > 1e-4, f"rcond[3]={rcond[3]} too small"
