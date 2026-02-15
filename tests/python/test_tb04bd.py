"""
Tests for TB04BD: Transfer function matrix via pole-zero method

Tests extracted from SLICOT-Reference/doc/TB04BD.html example.
"""

import numpy as np
import pytest
from ctrlsys import tb04bd


class TestTB04BDBasic:
    """Basic functionality tests from HTML doc example."""

    def test_html_example(self):
        """
        Test TB04BD with data from SLICOT HTML documentation.

        System: N=3, M=2, P=2 with non-zero D matrix.
        ORDER='I' (increasing powers), EQUIL='N' (no equilibration).

        Expected transfer function matrix G(s):
          G(1,1) = (7 + 5s + s^2) / (6 + 5s + s^2)
          G(2,1) = 1 / (6 + 5s + s^2)
          G(1,2) = 1 / (2 + s)
          G(2,2) = (5 + 5s + s^2) / (2 + 3s + s^2)
        """
        n, m, p = 3, 2, 2
        md = n + 1

        # A matrix (3x3) - read row-wise in Fortran: ((A(I,J), J=1,N), I=1,N)
        a = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.0],
            [ 0.0,  0.0, -3.0]
        ], dtype=float, order='F')

        # B matrix (3x2) - read column-wise: ((B(I,J), I=1,N), J=1,M)
        # Column 1: 0.0, 1.0, -1.0
        # Column 2: 1.0, 1.0, 0.0
        b = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0]
        ], dtype=float, order='F')

        # C matrix (2x3) - read row-wise: ((C(I,J), J=1,N), I=1,P)
        # Row 1: 0.0, 1.0, 1.0
        # Row 2: 1.0, 1.0, 1.0
        c = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=float, order='F')

        # D matrix (2x2) - read row-wise: ((D(I,J), J=1,M), I=1,P)
        # Row 1: 1.0, 0.0
        # Row 2: 0.0, 1.0
        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        # Call routine with JOBD='D', ORDER='I', EQUIL='N'
        # Note: Use tol=1e-10 to ensure numerical roundoff doesn't affect
        # controllability/observability determination. Default tol=0 can
        # sometimes fail with borderline numerical noise.
        ign, igd, gn, gd, info = tb04bd('D', 'I', 'N', n, m, p, md, a, b, c, d, tol=1e-10)

        assert info == 0, f"Expected info=0, got {info}"

        # Expected results from HTML documentation
        # Element (1,1): Num = 7 + 5s + s^2, Den = 6 + 5s + s^2 (degrees 2, 2)
        assert ign[0, 0] == 2
        assert igd[0, 0] == 2
        idx = 0  # (i=0, j=0) -> index = ((0)*p + 0)*md = 0
        np.testing.assert_allclose(gn[idx:idx+3], [7.0, 5.0, 1.0], rtol=1e-10)
        np.testing.assert_allclose(gd[idx:idx+3], [6.0, 5.0, 1.0], rtol=1e-10)

        # Element (2,1): Num = 1, Den = 6 + 5s + s^2 (degrees 0, 2)
        assert ign[1, 0] == 0
        assert igd[1, 0] == 2
        idx = 1 * md  # (i=1, j=0) -> index = ((0)*p + 1)*md = md
        np.testing.assert_allclose(gn[idx:idx+1], [1.0], rtol=1e-10)
        np.testing.assert_allclose(gd[idx:idx+3], [6.0, 5.0, 1.0], rtol=1e-10)

        # Element (1,2): Num = 1, Den = 2 + s (degrees 0, 1)
        assert ign[0, 1] == 0
        assert igd[0, 1] == 1
        idx = p * md  # (i=0, j=1) -> index = ((1)*p + 0)*md = p*md
        np.testing.assert_allclose(gn[idx:idx+1], [1.0], rtol=1e-10)
        np.testing.assert_allclose(gd[idx:idx+2], [2.0, 1.0], rtol=1e-10)

        # Element (2,2): Num = 5 + 5s + s^2, Den = 2 + 3s + s^2 (degrees 2, 2)
        assert ign[1, 1] == 2
        assert igd[1, 1] == 2
        idx = (p + 1) * md  # (i=1, j=1) -> index = ((1)*p + 1)*md = (p+1)*md
        np.testing.assert_allclose(gn[idx:idx+3], [5.0, 5.0, 1.0], rtol=1e-10)
        np.testing.assert_allclose(gd[idx:idx+3], [2.0, 3.0, 1.0], rtol=1e-10)


class TestTB04BDDecreasingOrder:
    """Test with decreasing polynomial order."""

    def test_decreasing_order(self):
        """
        Same system as HTML example but with ORDER='D' (decreasing powers).

        For polynomial p(s) = a + bs + cs^2:
        - ORDER='I' stores [a, b, c]
        - ORDER='D' stores [c, b, a]
        """
        n, m, p = 3, 2, 2
        md = n + 1

        a = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.0],
            [ 0.0,  0.0, -3.0]
        ], dtype=float, order='F')

        b = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0]
        ], dtype=float, order='F')

        c = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=float, order='F')

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        ign, igd, gn, gd, info = tb04bd('D', 'D', 'N', n, m, p, md, a, b, c, d, tol=1e-10)

        assert info == 0

        # Element (1,1): Num = s^2 + 5s + 7 (stored as [1, 5, 7] in decreasing)
        assert ign[0, 0] == 2
        assert igd[0, 0] == 2
        idx = 0
        np.testing.assert_allclose(gn[idx:idx+3], [1.0, 5.0, 7.0], rtol=1e-10)
        np.testing.assert_allclose(gd[idx:idx+3], [1.0, 5.0, 6.0], rtol=1e-10)


class TestTB04BDZeroD:
    """Test with zero D matrix."""

    def test_zero_d_matrix(self):
        """
        Test JOBD='Z' (D is zero matrix).
        The transfer function is strictly proper.
        """
        n, m, p = 2, 1, 1
        md = n + 1

        # Simple system with single pole at -1 and single zero at -2
        # A = [[-1, 1], [0, -2]], B = [[1], [0]], C = [[1, 1]]
        a = np.array([
            [-1.0,  1.0],
            [ 0.0, -2.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0],
            [0.0]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 1.0]
        ], dtype=float, order='F')

        # D not referenced when JOBD='Z'
        d = np.zeros((1, 1), dtype=float, order='F')

        ign, igd, gn, gd, info = tb04bd('Z', 'I', 'N', n, m, p, md, a, b, c, d, tol=1e-10)

        assert info == 0
        # Should compute transfer function of system with D=0


class TestTB04BDEquilibration:
    """Test with equilibration enabled."""

    def test_equilibration(self):
        """
        Test EQUIL='S' (scaling enabled).
        Should produce same transfer function as without scaling.
        """
        n, m, p = 3, 2, 2
        md = n + 1

        a = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.0],
            [ 0.0,  0.0, -3.0]
        ], dtype=float, order='F')

        b = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0]
        ], dtype=float, order='F')

        c = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=float, order='F')

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        ign, igd, gn, gd, info = tb04bd('D', 'I', 'S', n, m, p, md, a, b, c, d, tol=1e-10)

        assert info == 0

        # Same expected results - transfer function is invariant under scaling
        assert ign[0, 0] == 2
        assert igd[0, 0] == 2
        idx = 0
        np.testing.assert_allclose(gn[idx:idx+3], [7.0, 5.0, 1.0], rtol=1e-10)
        np.testing.assert_allclose(gd[idx:idx+3], [6.0, 5.0, 1.0], rtol=1e-10)


class TestTB04BDEdgeCases:
    """Edge case tests."""

    def test_siso_system(self):
        """Test single-input single-output system."""
        n, m, p = 2, 1, 1
        md = n + 1

        # Stable first-order system with gain
        a = np.array([
            [-1.0,  0.0],
            [ 0.0, -2.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0],
            [1.0]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 1.0]
        ], dtype=float, order='F')

        d = np.array([[0.0]], dtype=float, order='F')

        ign, igd, gn, gd, info = tb04bd('D', 'I', 'N', n, m, p, md, a, b, c, d)

        assert info == 0
        # Verify shapes
        assert ign.shape == (p, m)
        assert igd.shape == (p, m)
        assert len(gn) == p * m * md
        assert len(gd) == p * m * md

    def test_n_zero(self):
        """Test N=0 (static system D only)."""
        n, m, p = 0, 2, 2
        md = 1

        a = np.zeros((1, 1), dtype=float, order='F')
        b = np.zeros((1, 2), dtype=float, order='F')
        c = np.zeros((2, 1), dtype=float, order='F')
        d = np.array([
            [2.0, 3.0],
            [4.0, 5.0]
        ], dtype=float, order='F')

        ign, igd, gn, gd, info = tb04bd('D', 'I', 'N', n, m, p, md, a, b, c, d)

        assert info == 0
        # All transfer functions are constants from D
        for i in range(p):
            for j in range(m):
                assert ign[i, j] == 0
                assert igd[i, j] == 0
                idx = (j * p + i) * md
                np.testing.assert_allclose(gn[idx], d[i, j], rtol=1e-10)
                np.testing.assert_allclose(gd[idx], 1.0, rtol=1e-10)


class TestTB04BDPropertyValidation:
    """Mathematical property validation tests."""

    def test_transfer_function_evaluation(self):
        """
        Validate transfer function: G(s) = C(sI-A)^(-1)B + D

        For a specific s value, compute G(s) two ways:
        1. State-space formula
        2. From numerator/denominator polynomials

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 3, 2, 2
        md = n + 1

        a = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.0],
            [ 0.0,  0.0, -3.0]
        ], dtype=float, order='F')

        b = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0]
        ], dtype=float, order='F')

        c = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=float, order='F')

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float, order='F')

        # Save copies for state-space calculation (tb04bd modifies A, B, C)
        a_orig = a.copy()
        b_orig = b.copy()
        c_orig = c.copy()

        ign, igd, gn, gd, info = tb04bd('D', 'I', 'N', n, m, p, md, a, b, c, d, tol=1e-10)
        assert info == 0

        # Evaluate at s = 1.5 (away from poles)
        s_val = 1.5

        # Method 1: State-space formula G(s) = C(sI-A)^(-1)B + D
        sI_minus_A = s_val * np.eye(n) - a_orig
        G_ss = c_orig @ np.linalg.solve(sI_minus_A, b_orig) + d

        # Method 2: From polynomials
        G_poly = np.zeros((p, m), dtype=float)
        for i in range(p):
            for j in range(m):
                idx = (j * p + i) * md
                deg_num = ign[i, j]
                deg_den = igd[i, j]

                # Evaluate numerator (increasing powers)
                num_val = 0.0
                for k in range(deg_num + 1):
                    num_val += gn[idx + k] * (s_val ** k)

                # Evaluate denominator
                den_val = 0.0
                for k in range(deg_den + 1):
                    den_val += gd[idx + k] * (s_val ** k)

                G_poly[i, j] = num_val / den_val

        # Both methods should give same result
        np.testing.assert_allclose(G_poly, G_ss, rtol=1e-12)
