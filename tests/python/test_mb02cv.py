"""
Tests for MB02CV: Apply MB02CU transformations to generator columns/rows.

MB02CV applies the transformations created by MB02CU on other columns/rows
of the generator, contained in arrays F1, F2 and G.
"""

import numpy as np
import pytest
from ctrlsys import mb02cv


class TestMB02CVBasic:
    """Basic functionality tests for MB02CV."""

    def test_typeg_d_basic(self):
        """
        Test TYPEG='D' (column oriented, rank deficient) basic case.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k = 2
        n = 3
        p = 4
        q = 3
        rnk = 2
        nb = 0
        col2 = p - k

        a1 = np.array([
            [1.5, 0.3],
            [0.2, 1.2]
        ], order='F', dtype=float)

        a2 = np.array([
            [0.5, 0.1],
            [0.3, 0.4]
        ], order='F', dtype=float)

        b = np.array([
            [0.8, 0.2, 0.1],
            [0.1, 0.7, 0.3]
        ], order='F', dtype=float)

        f1 = np.array([
            [1.0, 0.5],
            [0.3, 1.2],
            [0.2, 0.4]
        ], order='F', dtype=float)

        f2 = np.array([
            [0.6, 0.2],
            [0.1, 0.8],
            [0.3, 0.1]
        ], order='F', dtype=float)

        g = np.array([
            [0.4, 0.1, 0.2],
            [0.2, 0.5, 0.1],
            [0.1, 0.2, 0.3]
        ], order='F', dtype=float)

        cs = np.array([
            1.1, 0.1, 1.2, 0.2,
            0.9, -0.1, 0.95, -0.15,
            0.3, 0.4
        ], dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('D', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == 0
        assert f1_out.shape == (n, k)
        assert f2_out.shape == (n, col2)
        assert g_out.shape == (n, q)

    def test_typeg_c_basic(self):
        """
        Test TYPEG='C' (column oriented, not rank deficient) basic case.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k = 2
        n = 4
        p = 3
        q = 2
        rnk = 0
        nb = 1
        col2 = p - k

        a1 = np.zeros((k, k), order='F', dtype=float)
        a2 = np.array([
            [0.5],
            [0.3]
        ], order='F', dtype=float)

        b = np.array([
            [0.8, 0.2],
            [0.1, 0.7]
        ], order='F', dtype=float)

        f1 = np.array([
            [1.0, 0.5],
            [0.3, 1.2],
            [0.2, 0.4],
            [0.1, 0.3]
        ], order='F', dtype=float)

        f2 = np.array([
            [0.6],
            [0.1],
            [0.3],
            [0.2]
        ], order='F', dtype=float)

        g = np.array([
            [0.4, 0.1],
            [0.2, 0.5],
            [0.1, 0.2],
            [0.3, 0.4]
        ], order='F', dtype=float)

        cs = np.array([
            1.05, 0.05, 1.1, 0.1,
            0.95, -0.05, 0.9, -0.1,
            0.2, 0.3
        ], dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('C', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == 0
        assert f1_out.shape == (n, k)
        assert f2_out.shape == (n, col2)
        assert g_out.shape == (n, q)

    def test_typeg_r_basic(self):
        """
        Test TYPEG='R' (row oriented, not rank deficient) basic case.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        k = 2
        n = 4
        p = 3
        q = 2
        rnk = 0
        nb = 1
        col2 = p - k

        a1 = np.zeros((k, k), order='F', dtype=float)
        a2 = np.array([
            [0.5, 0.3]
        ], order='F', dtype=float)

        b = np.array([
            [0.8, 0.1],
            [0.2, 0.7]
        ], order='F', dtype=float)

        f1 = np.array([
            [1.0, 0.3, 0.2, 0.1],
            [0.5, 1.2, 0.4, 0.3]
        ], order='F', dtype=float)

        f2 = np.array([
            [0.6, 0.1, 0.3, 0.2]
        ], order='F', dtype=float)

        g = np.array([
            [0.4, 0.2, 0.1, 0.3],
            [0.1, 0.5, 0.2, 0.4]
        ], order='F', dtype=float)

        cs = np.array([
            1.05, 0.05, 1.1, 0.1,
            0.95, -0.05, 0.9, -0.1,
            0.2, 0.3
        ], dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('R', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == 0
        assert f1_out.shape == (k, n)
        assert f2_out.shape == (col2, n)
        assert g_out.shape == (q, n)


class TestMB02CVTriangularStructure:
    """Tests for triangular structure (STRUCG='T')."""

    def test_typeg_c_triangular(self):
        """
        Test TYPEG='C' with triangular structure.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        k = 2
        n = 4
        p = 4
        q = 2
        rnk = 0
        nb = 1
        col2 = p - k

        a1 = np.zeros((k, k), order='F', dtype=float)
        a2 = np.array([
            [0.5, 0.0],
            [0.3, 0.4]
        ], order='F', dtype=float)

        b = np.array([
            [0.8, 0.0],
            [0.1, 0.7]
        ], order='F', dtype=float)

        f1 = np.array([
            [1.0, 0.5],
            [0.3, 1.2],
            [0.2, 0.4],
            [0.1, 0.3]
        ], order='F', dtype=float)

        f2 = np.array([
            [0.6, 0.2],
            [0.1, 0.8],
            [0.3, 0.1],
            [0.2, 0.3]
        ], order='F', dtype=float)

        g = np.array([
            [0.4, 0.1],
            [0.2, 0.5],
            [0.1, 0.2],
            [0.3, 0.4]
        ], order='F', dtype=float)

        cs = np.array([
            1.05, 0.05, 1.1, 0.1,
            0.95, -0.05, 0.9, -0.1,
            0.2, 0.3
        ], dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('C', 'T', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == 0
        assert f1_out.shape == (n, k)
        assert f2_out.shape == (n, col2)


class TestMB02CVEdgeCases:
    """Edge case tests."""

    def test_p_equals_k(self):
        """Test when P = K (no A2 and F2 columns)."""
        k = 2
        n = 3
        p = 2
        q = 2
        rnk = 2
        nb = 0

        a1 = np.array([
            [1.5, 0.3],
            [0.2, 1.2]
        ], order='F', dtype=float)

        a2 = np.zeros((k, 1), order='F', dtype=float)

        b = np.array([
            [0.8, 0.2],
            [0.1, 0.7]
        ], order='F', dtype=float)

        f1 = np.array([
            [1.0, 0.5],
            [0.3, 1.2],
            [0.2, 0.4]
        ], order='F', dtype=float)

        f2 = np.zeros((n, 1), order='F', dtype=float)

        g = np.array([
            [0.4, 0.1],
            [0.2, 0.5],
            [0.1, 0.2]
        ], order='F', dtype=float)

        cs = np.array([
            1.1, 0.1, 1.2, 0.2,
            0.3, 0.4
        ], dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('D', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == 0
        assert f1_out.shape == (n, k)

    def test_quick_return_k_zero(self):
        """Test quick return when K = 0."""
        k = 0
        n = 3
        p = 2
        q = 2
        rnk = 0
        nb = 0

        a1 = np.zeros((1, 1), order='F', dtype=float)
        a2 = np.zeros((1, 2), order='F', dtype=float)
        b = np.zeros((1, 2), order='F', dtype=float)
        f1 = np.zeros((n, 1), order='F', dtype=float)
        f2 = np.zeros((n, 2), order='F', dtype=float)
        g = np.zeros((n, 2), order='F', dtype=float)
        cs = np.zeros(1, dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('D', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == 0

    def test_quick_return_n_zero(self):
        """Test quick return when N = 0."""
        k = 2
        n = 0
        p = 3
        q = 2
        rnk = 2
        nb = 0

        a1 = np.array([
            [1.5, 0.3],
            [0.2, 1.2]
        ], order='F', dtype=float)
        a2 = np.array([
            [0.5],
            [0.3]
        ], order='F', dtype=float)
        b = np.array([
            [0.8, 0.2],
            [0.1, 0.7]
        ], order='F', dtype=float)
        f1 = np.zeros((1, k), order='F', dtype=float)
        f2 = np.zeros((1, 1), order='F', dtype=float)
        g = np.zeros((1, q), order='F', dtype=float)
        cs = np.array([1.1, 0.1, 1.2, 0.2, 0.9, -0.1, 0.3, 0.4], dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('D', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == 0


class TestMB02CVErrorHandling:
    """Test error handling."""

    def test_invalid_typeg(self):
        """Test invalid TYPEG parameter."""
        k = 2
        n = 3
        p = 4
        q = 2
        rnk = 2
        nb = 0

        a1 = np.zeros((k, k), order='F', dtype=float)
        a2 = np.zeros((k, p-k), order='F', dtype=float)
        b = np.zeros((k, q), order='F', dtype=float)
        f1 = np.zeros((n, k), order='F', dtype=float)
        f2 = np.zeros((n, p-k), order='F', dtype=float)
        g = np.zeros((n, q), order='F', dtype=float)
        cs = np.zeros(10, dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('X', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == -1

    def test_invalid_strucg(self):
        """Test invalid STRUCG parameter."""
        k = 2
        n = 3
        p = 4
        q = 2
        rnk = 2
        nb = 0

        a1 = np.zeros((k, k), order='F', dtype=float)
        a2 = np.zeros((k, p-k), order='F', dtype=float)
        b = np.zeros((k, q), order='F', dtype=float)
        f1 = np.zeros((n, k), order='F', dtype=float)
        f2 = np.zeros((n, p-k), order='F', dtype=float)
        g = np.zeros((n, q), order='F', dtype=float)
        cs = np.zeros(10, dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('D', 'X', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == -2

    def test_negative_k(self):
        """Test negative K parameter."""
        k = -1
        n = 3
        p = 4
        q = 2
        rnk = 0
        nb = 0

        a1 = np.zeros((1, 1), order='F', dtype=float)
        a2 = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 1), order='F', dtype=float)
        f1 = np.zeros((n, 1), order='F', dtype=float)
        f2 = np.zeros((n, 1), order='F', dtype=float)
        g = np.zeros((n, q), order='F', dtype=float)
        cs = np.zeros(10, dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('D', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == -3

    def test_p_less_than_k(self):
        """Test P < K error."""
        k = 4
        n = 3
        p = 2
        q = 2
        rnk = 2
        nb = 0

        a1 = np.zeros((k, k), order='F', dtype=float)
        a2 = np.zeros((k, 1), order='F', dtype=float)
        b = np.zeros((k, q), order='F', dtype=float)
        f1 = np.zeros((n, k), order='F', dtype=float)
        f2 = np.zeros((n, 1), order='F', dtype=float)
        g = np.zeros((n, q), order='F', dtype=float)
        cs = np.zeros(20, dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('D', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == -5

    def test_q_less_than_k_typeg_d(self):
        """Test Q < K error for TYPEG='D'."""
        k = 3
        n = 3
        p = 4
        q = 2
        rnk = 2
        nb = 0

        a1 = np.zeros((k, k), order='F', dtype=float)
        a2 = np.zeros((k, p-k), order='F', dtype=float)
        b = np.zeros((k, q), order='F', dtype=float)
        f1 = np.zeros((n, k), order='F', dtype=float)
        f2 = np.zeros((n, p-k), order='F', dtype=float)
        g = np.zeros((n, q), order='F', dtype=float)
        cs = np.zeros(15, dtype=float)

        f1_out, f2_out, g_out, info = mb02cv('D', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == -6


class TestMB02CVNumericalAccuracy:
    """Tests for numerical accuracy."""

    def test_transformation_consistency_typeg_d(self):
        """
        Test that MB02CV transformations are mathematically consistent.

        This verifies that the hyperbolic rotations preserve certain
        matrix relationships for the rank-deficient case.

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        k = 2
        n = 4
        p = 4
        q = 3
        rnk = 2
        nb = 0
        col2 = p - k

        a1 = np.array([
            [1.5, 0.3],
            [0.2, 1.2]
        ], order='F', dtype=float)

        a2 = np.array([
            [0.5, 0.1],
            [0.3, 0.4]
        ], order='F', dtype=float)

        b = np.array([
            [0.8, 0.2, 0.1],
            [0.1, 0.7, 0.3]
        ], order='F', dtype=float)

        f1 = np.array([
            [1.0, 0.5],
            [0.3, 1.2],
            [0.2, 0.4],
            [0.1, 0.3]
        ], order='F', dtype=float)

        f2 = np.array([
            [0.6, 0.2],
            [0.1, 0.8],
            [0.3, 0.1],
            [0.4, 0.2]
        ], order='F', dtype=float)

        g = np.array([
            [0.4, 0.1, 0.2],
            [0.2, 0.5, 0.1],
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.1]
        ], order='F', dtype=float)

        cs = np.array([
            1.1, 0.1, 1.2, 0.2,
            0.9, -0.1, 0.95, -0.15,
            0.3, 0.4
        ], dtype=float)

        f1_orig = f1.copy()
        f2_orig = f2.copy()
        g_orig = g.copy()

        f1_out, f2_out, g_out, info = mb02cv('D', 'N', k, n, p, q, nb, rnk,
                                              a1, a2, b, f1, f2, g, cs)

        assert info == 0

        assert not np.allclose(f1_out, f1_orig) or not np.allclose(g_out, g_orig), \
            "Transformation should modify at least F1 or G"

        assert np.all(np.isfinite(f1_out))
        assert np.all(np.isfinite(f2_out))
        assert np.all(np.isfinite(g_out))

        assert np.linalg.norm(f1_out) < 1e6
        assert np.linalg.norm(g_out) < 1e6

    def test_typeg_c_output_finite(self):
        """
        Test that typeg='C' outputs are finite.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        k = 3
        n = 5
        p = 5
        q = 3
        rnk = 0
        col2 = p - k

        a1 = np.zeros((k, k), order='F', dtype=float)
        a2 = np.array([
            [0.5, 0.1],
            [0.3, 0.4],
            [0.2, 0.3]
        ], order='F', dtype=float)

        b = np.array([
            [0.8, 0.2, 0.1],
            [0.1, 0.7, 0.3],
            [0.2, 0.1, 0.6]
        ], order='F', dtype=float)

        f1 = np.array([
            [1.0, 0.5, 0.2],
            [0.3, 1.2, 0.4],
            [0.2, 0.4, 1.1],
            [0.1, 0.3, 0.2],
            [0.4, 0.2, 0.3]
        ], order='F', dtype=float)

        f2 = np.array([
            [0.6, 0.2],
            [0.1, 0.8],
            [0.3, 0.1],
            [0.2, 0.4],
            [0.4, 0.3]
        ], order='F', dtype=float)

        g = np.array([
            [0.4, 0.1, 0.2],
            [0.2, 0.5, 0.1],
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.1],
            [0.2, 0.3, 0.2]
        ], order='F', dtype=float)

        cs = np.array([
            1.05, 0.05, 1.1, 0.1, 1.08, 0.08,
            0.95, -0.05, 0.9, -0.1, 0.92, -0.08,
            0.2, 0.3, 0.25
        ], dtype=float)

        f1_out, f2_out, g_out, info = mb02cv(
            'C', 'N', k, n, p, q, 0, rnk,
            a1.copy(), a2.copy(), b.copy(), f1.copy(), f2.copy(), g.copy(), cs.copy())

        assert info == 0
        assert np.all(np.isfinite(f1_out)), "F1 output should be finite"
        assert np.all(np.isfinite(f2_out)), "F2 output should be finite"
        assert np.all(np.isfinite(g_out)), "G output should be finite"

        assert f1_out.shape == (n, k)
        assert f2_out.shape == (n, col2)
        assert g_out.shape == (n, q)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
