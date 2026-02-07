"""
Tests for SB16BD: Coprime factorization based controller reduction.

Computes a reduced order controller for a given open-loop model (A,B,C,D)
with state feedback gain F and observer gain G using coprime factorization
based model reduction methods (B&T or SPA).
"""

import numpy as np
import pytest
from slicot import sb16bd


class TestSB16BDBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_html_doc_example_left_coprime_bt(self):
        """
        Test from SLICOT HTML documentation.

        Continuous-time system with left coprime factorization,
        balancing-free B&T method, fixed order reduction to NCR=4.
        """
        n, m, p = 8, 1, 1
        ncr_desired = 4

        a = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.015, 0.765, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.765, -0.015, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.028, 1.41, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.41, -0.028, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04, 1.85],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.85, -0.04]
        ], dtype=float, order='F')

        b = np.array([
            [0.026],
            [-0.251],
            [0.033],
            [-0.886],
            [-4.017],
            [0.145],
            [3.604],
            [0.28]
        ], dtype=float, order='F')

        c = np.array([
            [-0.996, -0.105, 0.261, 0.009, -0.001, -0.043, 0.002, -0.026]
        ], dtype=float, order='F')

        d = np.array([[0.0]], dtype=float, order='F')

        f = np.array([
            [4.4721e-002, 6.6105e-001, 4.6986e-003, 3.6014e-001,
             1.0325e-001, -3.7541e-002, -4.2685e-002, 3.2873e-002]
        ], dtype=float, order='F')

        g = np.array([
            [4.1089e-001],
            [8.6846e-002],
            [3.8523e-004],
            [-3.6194e-003],
            [-8.8037e-003],
            [8.4205e-003],
            [1.2349e-003],
            [4.2632e-003]
        ], dtype=float, order='F')

        # sb16bd(dico, jobd, jobmr, jobcf, equil, ordsel, n, m, p, ncr,
        #        a, b, c, d, f, g, tol1, tol2)
        ac, bc, cc, dc, hsv, ncr_out, iwarn, info = sb16bd(
            'C', 'D', 'F', 'L', 'S', 'F', n, m, p, ncr_desired,
            a, b, c, d, f, g, 0.1, 0.0
        )

        assert info == 0
        assert iwarn == 0
        assert ncr_out == 4

        hsv_expected = np.array([
            4.9078, 4.8745, 3.8455, 3.7811, 1.2289, 1.1785, 0.5176, 0.1148
        ])
        np.testing.assert_allclose(hsv, hsv_expected, rtol=1e-3)

        ac_expected = np.array([
            [0.5946, -0.7336, 0.1914, -0.3368],
            [0.5960, -0.0184, -0.1088, 0.0207],
            [1.2253, 0.2043, 0.1009, -1.4948],
            [-0.0330, -0.0243, 1.3440, 0.0035]
        ], dtype=float, order='F')

        bc_expected = np.array([
            [0.0015],
            [-0.0202],
            [0.0159],
            [-0.0544]
        ], dtype=float, order='F')

        cc_expected = np.array([
            [0.3534, 0.0274, 0.0337, -0.0320]
        ], dtype=float, order='F')

        dc_expected = np.array([[0.0]], dtype=float, order='F')

        eig_actual = np.sort(np.linalg.eigvals(ac))
        eig_expected = np.sort(np.linalg.eigvals(ac_expected))
        np.testing.assert_allclose(eig_actual, eig_expected, rtol=1e-3)

        def transfer_function(ac_mat, bc_mat, cc_mat, dc_mat, s):
            n = ac_mat.shape[0]
            return cc_mat @ np.linalg.solve(s * np.eye(n) - ac_mat, bc_mat) + dc_mat

        for omega in [0.1, 1.0, 10.0]:
            s = 1j * omega
            tf_actual = transfer_function(ac, bc, cc, dc, s)
            tf_expected = transfer_function(ac_expected, bc_expected, cc_expected, dc_expected, s)
            np.testing.assert_allclose(np.abs(tf_actual), np.abs(tf_expected), rtol=2e-2)


class TestSB16BDRightCoprime:
    """Tests for right coprime factorization."""

    def test_right_coprime_factorization(self):
        """
        Test right coprime factorization with same system.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 1, 1

        a = np.array([
            [-0.04, 0.01, 0.0, 0.0],
            [-0.05, -0.03, 0.01, 0.0],
            [0.0, -0.06, -0.05, 0.02],
            [0.0, 0.0, -0.04, -0.02]
        ], dtype=float, order='F')

        b = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=float, order='F')
        c = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float, order='F')
        d = np.array([[0.0]], dtype=float, order='F')

        f = -0.1 * np.array([[0.5, 0.3, 0.2, 0.1]], dtype=float, order='F')
        g = -0.5 * np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float, order='F')

        ac, bc, cc, dc, hsv, ncr_out, iwarn, info = sb16bd(
            'C', 'Z', 'B', 'R', 'N', 'A', n, m, p, 0,
            a, b, c, d, f, g, 0.001, 0.0
        )

        assert info == 0
        assert ncr_out <= n
        assert len(hsv) == n
        assert ac.shape == (ncr_out, ncr_out)
        assert bc.shape == (ncr_out, p)
        assert cc.shape == (m, ncr_out)
        assert dc.shape == (m, p)


class TestSB16BDDiscreteTime:
    """Tests for discrete-time systems."""

    def test_discrete_time_system(self):
        """
        Test discrete-time system reduction.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 3, 1, 1

        a = np.array([
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.7]
        ], dtype=float, order='F')

        b = np.array([[1.0], [0.0], [0.0]], dtype=float, order='F')
        c = np.array([[0.0, 0.0, 1.0]], dtype=float, order='F')
        d = np.array([[0.0]], dtype=float, order='F')

        f = -0.1 * np.array([[0.3, 0.2, 0.1]], dtype=float, order='F')
        g = -0.2 * np.array([[0.1], [0.2], [0.3]], dtype=float, order='F')

        ac, bc, cc, dc, hsv, ncr_out, iwarn, info = sb16bd(
            'D', 'Z', 'B', 'L', 'N', 'A', n, m, p, 0,
            a, b, c, d, f, g, 0.001, 0.0
        )

        assert info == 0
        assert ncr_out <= n


class TestSB16BDModelReductionMethods:
    """Tests for different model reduction methods."""

    def test_spa_method(self):
        """
        Test Singular Perturbation Approximation (SPA) method.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 4, 1, 1

        a = np.array([
            [-0.5, 0.1, 0.0, 0.0],
            [0.0, -0.3, 0.1, 0.0],
            [0.0, 0.0, -0.2, 0.1],
            [0.0, 0.0, 0.0, -0.1]
        ], dtype=float, order='F')

        b = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=float, order='F')
        c = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float, order='F')
        d = np.array([[0.0]], dtype=float, order='F')

        f = -0.1 * np.array([[0.4, 0.3, 0.2, 0.1]], dtype=float, order='F')
        g = -0.3 * np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float, order='F')

        ac, bc, cc, dc, hsv, ncr_out, iwarn, info = sb16bd(
            'C', 'Z', 'S', 'L', 'N', 'A', n, m, p, 0,
            a, b, c, d, f, g, 0.01, 0.0
        )

        assert info == 0
        assert ncr_out <= n

    def test_balancing_free_spa(self):
        """
        Test balancing-free SPA method (JOBMR='P').

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 4, 1, 1

        a = np.array([
            [-0.5, 0.1, 0.0, 0.0],
            [0.0, -0.3, 0.1, 0.0],
            [0.0, 0.0, -0.2, 0.1],
            [0.0, 0.0, 0.0, -0.1]
        ], dtype=float, order='F')

        b = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=float, order='F')
        c = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float, order='F')
        d = np.array([[0.0]], dtype=float, order='F')

        f = -0.1 * np.array([[0.4, 0.3, 0.2, 0.1]], dtype=float, order='F')
        g = -0.3 * np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float, order='F')

        ac, bc, cc, dc, hsv, ncr_out, iwarn, info = sb16bd(
            'C', 'Z', 'P', 'L', 'N', 'A', n, m, p, 0,
            a, b, c, d, f, g, 0.01, 0.0
        )

        assert info == 0
        assert ncr_out <= n


class TestSB16BDEdgeCases:
    """Edge case tests."""

    def test_zero_order_system(self):
        """Test with N=0."""
        n, m, p = 0, 1, 1

        a = np.array([], dtype=float, order='F').reshape(0, 0)
        b = np.array([], dtype=float, order='F').reshape(0, 1)
        c = np.array([], dtype=float, order='F').reshape(1, 0)
        d = np.array([[0.0]], dtype=float, order='F')
        f = np.array([], dtype=float, order='F').reshape(1, 0)
        g = np.array([], dtype=float, order='F').reshape(0, 1)

        ac, bc, cc, dc, hsv, ncr_out, iwarn, info = sb16bd(
            'C', 'Z', 'B', 'L', 'N', 'F', n, m, p, 0,
            a, b, c, d, f, g, 0.0, 0.0
        )

        assert info == 0
        assert ncr_out == 0

    def test_full_order_ncr_equals_n(self):
        """
        Test with NCR=N (no reduction).

        When NCR=N, routine only forms the controller matrices.
        """
        n, m, p = 3, 1, 1
        ncr_in = n

        a = np.array([
            [-0.5, 0.1, 0.0],
            [0.0, -0.3, 0.1],
            [0.0, 0.0, -0.2]
        ], dtype=float, order='F')

        b = np.array([[1.0], [0.0], [0.0]], dtype=float, order='F')
        c = np.array([[0.0, 0.0, 1.0]], dtype=float, order='F')
        d = np.array([[0.0]], dtype=float, order='F')

        f = np.array([[-0.1, -0.05, -0.02]], dtype=float, order='F')
        g = np.array([[-0.05], [-0.1], [-0.15]], dtype=float, order='F')

        ac, bc, cc, dc, hsv, ncr_out, iwarn, info = sb16bd(
            'C', 'Z', 'B', 'L', 'N', 'F', n, m, p, ncr_in,
            a, b, c, d, f, g, 0.0, 0.0
        )

        assert info == 0
        assert ncr_out == n
        assert ac.shape == (n, n)
        assert bc.shape == (n, p)
        assert cc.shape == (m, n)
        assert dc.shape == (m, p)


class TestSB16BDMIMO:
    """MIMO controller reduction tests."""

    def test_mimo_left_coprime_reduction(self):
        """
        MIMO system: 4-state, 2 inputs, 2 outputs.
        Left coprime factorization with balancing-free B&T.
        """
        n, m, p = 4, 2, 2

        a = np.array([
            [-1.0,  0.5,  0.0,  0.0],
            [ 0.0, -2.0,  0.3,  0.0],
            [ 0.0,  0.0, -3.0,  0.1],
            [ 0.0,  0.0,  0.0, -4.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [0.2, 0.3]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.5, 0.2]
        ], dtype=float, order='F')

        d = np.zeros((p, m), dtype=float, order='F')

        f = np.array([
            [-0.5, -0.3, -0.1,  0.0],
            [ 0.0, -0.4, -0.2, -0.1]
        ], dtype=float, order='F')

        g = np.array([
            [-0.3,  0.0],
            [-0.2, -0.3],
            [-0.1, -0.2],
            [ 0.0, -0.1]
        ], dtype=float, order='F')

        ac, bc, cc, dc, hsv, ncr_out, iwarn, info = sb16bd(
            'C', 'Z', 'F', 'L', 'N', 'A', n, m, p, 0,
            a, b, c, d, f, g, 0.01, 0.0
        )

        assert info == 0
        assert ncr_out <= n
        assert ac.shape == (ncr_out, ncr_out)
        assert bc.shape == (ncr_out, p)
        assert cc.shape == (m, ncr_out)
        assert dc.shape == (m, p)

        for i in range(len(hsv) - 1):
            assert hsv[i] >= hsv[i + 1] - 1e-14, "HSV not decreasing"


class TestSB16BDErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test with invalid DICO parameter."""
        n, m, p = 2, 1, 1

        a = np.array([[-0.5, 0.1], [0.0, -0.3]], dtype=float, order='F')
        b = np.array([[1.0], [0.0]], dtype=float, order='F')
        c = np.array([[0.0, 1.0]], dtype=float, order='F')
        d = np.array([[0.0]], dtype=float, order='F')
        f = np.array([[-0.1, -0.05]], dtype=float, order='F')
        g = np.array([[-0.05], [-0.1]], dtype=float, order='F')

        with pytest.raises(ValueError):
            sb16bd(
                'X', 'Z', 'B', 'L', 'N', 'F', n, m, p, 1,
                a, b, c, d, f, g, 0.0, 0.0
            )

    def test_invalid_ncr(self):
        """Test with invalid NCR (out of range)."""
        n, m, p = 2, 1, 1

        a = np.array([[-0.5, 0.1], [0.0, -0.3]], dtype=float, order='F')
        b = np.array([[1.0], [0.0]], dtype=float, order='F')
        c = np.array([[0.0, 1.0]], dtype=float, order='F')
        d = np.array([[0.0]], dtype=float, order='F')
        f = np.array([[-0.1, -0.05]], dtype=float, order='F')
        g = np.array([[-0.05], [-0.1]], dtype=float, order='F')

        with pytest.raises(ValueError):
            sb16bd(
                'C', 'Z', 'B', 'L', 'N', 'F', n, m, p, 5,
                a, b, c, d, f, g, 0.0, 0.0
            )


class TestSB16BDControllerReduction:
    """Test controller reduction property."""

    def test_hankel_singular_values_decreasing(self):
        """
        Verify Hankel singular values are in decreasing order.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        n, m, p = 6, 1, 1

        a = np.diag([-0.1 * (i + 1) for i in range(n)]).astype(float, order='F')
        a[0, 1] = 0.05
        a = np.asfortranarray(a)

        b = np.array([[1.0], [0.5], [0.25], [0.125], [0.0625], [0.03125]],
                     dtype=float, order='F')
        c = np.array([[0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]],
                     dtype=float, order='F')
        d = np.array([[0.0]], dtype=float, order='F')

        f = -0.01 * np.arange(1, n + 1).reshape(1, n).astype(float, order='F')
        g = -0.01 * np.arange(1, n + 1).reshape(n, 1).astype(float, order='F')

        ac, bc, cc, dc, hsv, ncr_out, iwarn, info = sb16bd(
            'C', 'Z', 'B', 'L', 'N', 'A', n, m, p, 0,
            a, b, c, d, f, g, 1e-6, 0.0
        )

        assert info == 0

        for i in range(len(hsv) - 1):
            assert hsv[i] >= hsv[i + 1], "HSV not in decreasing order"
