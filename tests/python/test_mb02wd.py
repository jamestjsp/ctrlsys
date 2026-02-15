"""
Tests for MB02WD - Conjugate gradient solver for SPD systems.
"""
import numpy as np
import pytest
import ctrlsys


class TestMB02WDBasic:
    """Basic functionality tests for MB02WD."""

    def test_upper_triangle(self):
        """Test with SPD matrix using upper triangle."""
        a = np.array([
            [4.0, 2.0, 1.0],
            [0.0, 5.0, 2.0],
            [0.0, 0.0, 6.0]
        ], dtype=np.float64, order='F')

        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x = np.zeros(3, dtype=np.float64)

        x_expected = np.array([0.03614457831325301, 0.21686746987951808, 0.42168674698795183])

        x_out, iterations, residual, iwarn, info = ctrlsys.mb02wd('U', 100, a, b, x, 1e-12)

        assert info == 0, f"MB02WD returned info = {info}"
        np.testing.assert_allclose(x_out, x_expected, rtol=1e-10)

    def test_lower_triangle(self):
        """Test with SPD matrix using lower triangle."""
        a = np.array([
            [4.0, 0.0, 0.0],
            [2.0, 5.0, 0.0],
            [1.0, 2.0, 6.0]
        ], dtype=np.float64, order='F')

        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x = np.zeros(3, dtype=np.float64)

        x_expected = np.array([0.03614457831325301, 0.21686746987951808, 0.42168674698795183])

        x_out, iterations, residual, iwarn, info = ctrlsys.mb02wd('L', 100, a, b, x, 1e-12)

        assert info == 0, f"MB02WD returned info = {info}"
        np.testing.assert_allclose(x_out, x_expected, rtol=1e-10)

    def test_identity_matrix(self):
        """Test with identity matrix: Ix = b => x = b."""
        n = 4
        a = np.eye(n, dtype=np.float64, order='F')
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        x = np.zeros(n, dtype=np.float64)

        x_out, iterations, residual, iwarn, info = ctrlsys.mb02wd('U', 100, a, b, x, 1e-14)

        assert info == 0, f"MB02WD returned info = {info}"
        np.testing.assert_allclose(x_out, b, rtol=1e-12)


class TestMB02WDEdgeCases:
    """Edge case tests for MB02WD."""

    def test_zero_iterations(self):
        """Test with zero maximum iterations."""
        a = np.array([
            [4.0, 2.0, 1.0],
            [0.0, 5.0, 2.0],
            [0.0, 0.0, 6.0]
        ], dtype=np.float64, order='F')

        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x = np.zeros(3, dtype=np.float64)

        x_out, iterations, residual, iwarn, info = ctrlsys.mb02wd('U', 0, a, b, x, 1e-12)

        assert info == 0
        assert iwarn == 2

    def test_larger_system(self):
        """Test with a larger SPD system."""
        n = 10
        a = np.eye(n, dtype=np.float64, order='F') * 10
        for i in range(n - 1):
            a[i, i + 1] = 1.0

        b = np.ones(n, dtype=np.float64)
        x = np.zeros(n, dtype=np.float64)

        x_out, iterations, residual, iwarn, info = ctrlsys.mb02wd('U', 200, a, b, x, 1e-12)

        assert info == 0
        ax = np.dot(a + a.T - np.diag(np.diag(a)), x_out)
        np.testing.assert_allclose(ax, b, rtol=1e-8)
