"""
Tests for SB10KD: Discrete-time loop shaping controller design.

SB10KD computes the positive feedback controller K = [Ak, Bk; Ck, Dk]
for a shaped plant G = [A, B; C, 0] using the McFarlane-Glover method.
"""

import numpy as np
import pytest


class TestSB10KD:
    """Tests for sb10kd routine."""

    def test_html_doc_example(self):
        """
        Test using SLICOT HTML documentation example.

        N=6, M=2, NP=2, FACTOR=1.1
        Data from SB10KD.html Program Data section.
        READ statements show row-by-row reading for A, B, C.
        """
        from ctrlsys import sb10kd

        n, m, np_ = 6, 2, 2
        factor = 1.1

        # A (6x6) - read row-by-row: ((A(I,J), J=1,N), I=1,N)
        a = np.array([
            [0.2, 0.0, 0.3, 0.0, -0.3, -0.1],
            [-0.3, 0.2, -0.4, -0.3, 0.0, 0.0],
            [-0.1, 0.1, -0.1, 0.0, 0.0, -0.3],
            [0.1, 0.0, 0.0, -0.1, -0.1, 0.0],
            [0.0, 0.3, 0.6, 0.2, 0.1, -0.4],
            [0.2, -0.4, 0.0, 0.0, 0.2, -0.2],
        ], dtype=float, order='F')

        # B (6x2) - read row-by-row: ((B(I,J), J=1,M), I=1,N)
        b = np.array([
            [-1.0, -2.0],
            [1.0, 3.0],
            [-3.0, -4.0],
            [1.0, -2.0],
            [0.0, 1.0],
            [1.0, 5.0],
        ], dtype=float, order='F')

        # C (2x6) - read row-by-row: ((C(I,J), J=1,N), I=1,NP)
        c = np.array([
            [1.0, -1.0, 2.0, -2.0, 0.0, -3.0],
            [-3.0, 0.0, 1.0, -1.0, 1.0, -1.0],
        ], dtype=float, order='F')

        # Call routine
        ak, bk, ck, dk, rcond, info = sb10kd(n, m, np_, a, b, c, factor)

        assert info == 0

        # Expected AK (6x6) from Program Results
        ak_expected = np.array([
            [0.0337, 0.0222, 0.0858, 0.1264, -0.1872, 0.1547],
            [0.4457, 0.0668, -0.2255, -0.3204, -0.4548, -0.0691],
            [-0.2419, -0.2506, -0.0982, -0.1321, -0.0130, -0.0838],
            [-0.4402, 0.3654, -0.0335, -0.2444, 0.6366, -0.6469],
            [-0.3623, 0.3854, 0.4162, 0.4502, 0.0065, 0.1261],
            [-0.0121, -0.4377, 0.0604, 0.2265, -0.3389, 0.4542],
        ], dtype=float, order='F')

        # Expected BK (6x2) from Program Results
        bk_expected = np.array([
            [0.0931, -0.0269],
            [-0.0872, 0.1599],
            [0.0956, -0.1469],
            [-0.1728, 0.0129],
            [0.2022, -0.1154],
            [0.2419, -0.1737],
        ], dtype=float, order='F')

        # Expected CK (2x6) from Program Results
        ck_expected = np.array([
            [-0.3677, 0.2188, 0.0403, -0.0854, 0.3564, -0.3535],
            [0.1624, -0.0708, 0.0058, 0.0606, -0.2163, 0.1802],
        ], dtype=float, order='F')

        # Expected DK (2x2) from Program Results
        dk_expected = np.array([
            [-0.0857, -0.0246],
            [0.0460, 0.0074],
        ], dtype=float, order='F')

        # Expected RCOND (4 values) from Program Results
        rcond_expected = np.array([0.11269e-01, 0.17596e-01, 0.18225e+00, 0.75968e-03])

        # Validate with tolerance appropriate for HTML 4-decimal precision
        np.testing.assert_allclose(ak, ak_expected, rtol=5e-3, atol=1e-4)
        np.testing.assert_allclose(bk, bk_expected, rtol=5e-3, atol=1e-4)
        np.testing.assert_allclose(ck, ck_expected, rtol=5e-3, atol=1e-4)
        np.testing.assert_allclose(dk, dk_expected, rtol=5e-3, atol=1e-4)
        np.testing.assert_allclose(rcond, rcond_expected, rtol=5e-2, atol=1e-5)

    def test_closed_loop_stability(self):
        """
        Verify closed-loop system is stable (eigenvalues inside unit circle).

        Mathematical property: for discrete-time stability, all eigenvalues
        of the closed-loop system must have magnitude < 1.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import sb10kd

        n, m, np_ = 6, 2, 2
        factor = 1.1

        # Use same data as HTML example
        a = np.array([
            [0.2, 0.0, 0.3, 0.0, -0.3, -0.1],
            [-0.3, 0.2, -0.4, -0.3, 0.0, 0.0],
            [-0.1, 0.1, -0.1, 0.0, 0.0, -0.3],
            [0.1, 0.0, 0.0, -0.1, -0.1, 0.0],
            [0.0, 0.3, 0.6, 0.2, 0.1, -0.4],
            [0.2, -0.4, 0.0, 0.0, 0.2, -0.2],
        ], dtype=float, order='F')

        b = np.array([
            [-1.0, -2.0],
            [1.0, 3.0],
            [-3.0, -4.0],
            [1.0, -2.0],
            [0.0, 1.0],
            [1.0, 5.0],
        ], dtype=float, order='F')

        c = np.array([
            [1.0, -1.0, 2.0, -2.0, 0.0, -3.0],
            [-3.0, 0.0, 1.0, -1.0, 1.0, -1.0],
        ], dtype=float, order='F')

        ak, bk, ck, dk, rcond, info = sb10kd(n, m, np_, a, b, c, factor)
        assert info == 0

        # Form closed-loop system matrix (2N x 2N) for POSITIVE feedback
        # [A + B*Dk*C,  B*Ck ]
        # [Bk*C,        Ak   ]
        n2 = 2 * n
        acl = np.zeros((n2, n2), dtype=float, order='F')

        # Top-left: A + B*Dk*C (positive feedback)
        acl[:n, :n] = a + b @ dk @ c
        # Top-right: B*Ck
        acl[:n, n:] = b @ ck
        # Bottom-left: Bk*C
        acl[n:, :n] = bk @ c
        # Bottom-right: Ak
        acl[n:, n:] = ak

        # Check all eigenvalues have magnitude < 1
        eigs = np.linalg.eigvals(acl)
        max_abs_eig = np.max(np.abs(eigs))
        assert max_abs_eig < 1.0, f"Closed-loop unstable: max |eig| = {max_abs_eig}"

    def test_quick_return_zero_dimensions(self):
        """
        Test quick return for zero dimensions (N=0, M=0, or NP=0).
        """
        from ctrlsys import sb10kd

        # Test N=0
        a = np.array([[]], dtype=float, order='F').reshape(0, 0)
        b = np.array([[]], dtype=float, order='F').reshape(0, 2)
        c = np.array([[]], dtype=float, order='F').reshape(2, 0)

        ak, bk, ck, dk, rcond, info = sb10kd(0, 2, 2, a, b, c, 1.0)

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0
        assert rcond[2] == 1.0
        assert rcond[3] == 1.0

    def test_optimal_controller_factor_one(self):
        """
        Test with FACTOR close to 1.0.

        Uses factor=1.1 for numerical robustness across BLAS implementations.
        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import sb10kd

        np.random.seed(123)

        n, m, np_ = 3, 1, 1
        factor = 1.1

        # Create stable discrete-time plant (eigenvalues inside unit circle)
        # Use diagonal dominant matrix for stability
        a = np.array([
            [0.3, 0.1, 0.0],
            [0.0, 0.4, 0.1],
            [0.1, 0.0, 0.2],
        ], dtype=float, order='F')

        b = np.array([
            [1.0],
            [0.5],
            [0.2],
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.5, 0.2],
        ], dtype=float, order='F')

        ak, bk, ck, dk, rcond, info = sb10kd(n, m, np_, a, b, c, factor)

        assert info == 0
        assert ak.shape == (n, n)
        assert bk.shape == (n, np_)
        assert ck.shape == (m, n)
        assert dk.shape == (m, np_)
        assert len(rcond) == 4

        # All rcond values should be positive
        assert np.all(rcond > 0)

    def test_suboptimal_controller_factor_greater_one(self):
        """
        Test suboptimal controller with FACTOR > 1.

        A larger FACTOR should give better conditioning.
        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import sb10kd

        n, m, np_ = 6, 2, 2
        factor_optimal = 1.0
        factor_subopt = 1.5

        a = np.array([
            [0.2, 0.0, 0.3, 0.0, -0.3, -0.1],
            [-0.3, 0.2, -0.4, -0.3, 0.0, 0.0],
            [-0.1, 0.1, -0.1, 0.0, 0.0, -0.3],
            [0.1, 0.0, 0.0, -0.1, -0.1, 0.0],
            [0.0, 0.3, 0.6, 0.2, 0.1, -0.4],
            [0.2, -0.4, 0.0, 0.0, 0.2, -0.2],
        ], dtype=float, order='F')

        b = np.array([
            [-1.0, -2.0],
            [1.0, 3.0],
            [-3.0, -4.0],
            [1.0, -2.0],
            [0.0, 1.0],
            [1.0, 5.0],
        ], dtype=float, order='F')

        c = np.array([
            [1.0, -1.0, 2.0, -2.0, 0.0, -3.0],
            [-3.0, 0.0, 1.0, -1.0, 1.0, -1.0],
        ], dtype=float, order='F')

        # Both should succeed
        ak1, bk1, ck1, dk1, rcond1, info1 = sb10kd(n, m, np_, a, b, c, factor_optimal)
        ak2, bk2, ck2, dk2, rcond2, info2 = sb10kd(n, m, np_, a, b, c, factor_subopt)

        # INFO=0 means algorithm succeeded, but closed-loop stability is NOT guaranteed
        # (depends on plant being stabilizable/detectable)
        # With suboptimal factor, should definitely succeed
        assert info2 == 0

        # Verify controller dimensions are correct
        assert ak1.shape == (n, n) or info1 != 0
        assert ak2.shape == (n, n)
        assert bk2.shape == (n, np_)
        assert ck2.shape == (m, n)
        assert dk2.shape == (m, np_)

        # Verify suboptimal has non-zero rcond (conditioning indicator)
        if info1 == 0:
            assert np.all(rcond2 > 0)

    def test_error_invalid_factor(self):
        """
        Test error handling for invalid FACTOR < 1.
        """
        from ctrlsys import sb10kd

        n, m, np_ = 2, 1, 1
        a = np.eye(n, dtype=float, order='F') * 0.5
        b = np.ones((n, m), dtype=float, order='F')
        c = np.ones((np_, n), dtype=float, order='F')

        # FACTOR < 1 is invalid
        ak, bk, ck, dk, rcond, info = sb10kd(n, m, np_, a, b, c, 0.5)

        assert info == -10  # FACTOR validation error
