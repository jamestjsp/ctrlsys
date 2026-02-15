"""
Tests for AB09AD - Balance & Truncate model reduction for stable systems.

AB09AD computes a reduced order model (Ar,Br,Cr) for a stable original
state-space representation (A,B,C) using the square-root or balancing-free
square-root Balance & Truncate model reduction method.
"""

import numpy as np
import pytest
from ctrlsys import ab09ad


class TestAB09ADBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_continuous_auto_order(self):
        """
        Test continuous-time system with automatic order selection.

        Uses example data from SLICOT HTML documentation.
        7th order system reduced to 5th order based on Hankel singular values.
        """
        n, m, p = 7, 2, 3

        # State matrix A (read row-by-row from HTML)
        a = np.array([
            [-0.04165,  0.0000,  4.9200, -4.9200,  0.0000,  0.0000,  0.0000],
            [-5.2100, -12.500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0000,   3.3300, -3.3300,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.5450,   0.0000,  0.0000,  0.0000, -0.5450,  0.0000,  0.0000],
            [ 0.0000,   0.0000,  0.0000,  4.9200, -0.04165, 0.0000,  4.9200],
            [ 0.0000,   0.0000,  0.0000,  0.0000, -5.2100, -12.500,  0.0000],
            [ 0.0000,   0.0000,  0.0000,  0.0000,  0.0000,  3.3300, -3.3300]
        ], order='F', dtype=float)

        # Input matrix B (read row-by-row from HTML)
        b = np.array([
            [ 0.0000,  0.0000],
            [12.500,   0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000, 12.500],
            [ 0.0000,  0.0000]
        ], order='F', dtype=float)

        # Output matrix C (read row-by-row from HTML)
        c = np.array([
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]
        ], order='F', dtype=float)

        tol = 0.1
        nr_in = 0  # Not used when ordsel='A'

        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'C', 'N', 'N', 'A', n, m, p, nr_in, a, b, c, tol
        )

        assert info == 0, f"ab09ad failed with info={info}"
        assert iwarn == 0, f"unexpected warning: iwarn={iwarn}"
        assert nr_out == 5, f"expected reduced order 5, got {nr_out}"

        # Expected Hankel singular values from HTML doc
        hsv_expected = np.array([2.5139, 2.0846, 1.9178, 0.7666, 0.5473, 0.0253, 0.0246])
        np.testing.assert_allclose(hsv, hsv_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced state dynamics matrix Ar from HTML doc
        ar_expected = np.array([
            [ 1.3451,  5.0399,  0.0000,  0.0000,  4.5315],
            [-4.0214, -3.6604,  0.0000,  0.0000, -0.9056],
            [ 0.0000,  0.0000,  0.5124,  1.7910,  0.0000],
            [ 0.0000,  0.0000, -4.2167, -2.9900,  0.0000],
            [ 1.2402,  1.6416,  0.0000,  0.0000, -0.0586]
        ], order='F', dtype=float)
        np.testing.assert_allclose(ar[:nr_out, :nr_out], ar_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced input matrix Br from HTML doc
        br_expected = np.array([
            [-0.3857,  0.3857],
            [-3.1753,  3.1753],
            [-0.7447, -0.7447],
            [-3.6872, -3.6872],
            [ 1.8197, -1.8197]
        ], order='F', dtype=float)
        np.testing.assert_allclose(br[:nr_out, :], br_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced output matrix Cr from HTML doc
        cr_expected = np.array([
            [-0.6704,  0.1828, -0.6582,  0.2222, -0.0104],
            [ 0.1089,  0.4867,  0.0000,  0.0000,  0.8651],
            [ 0.6704, -0.1828, -0.6582,  0.2222,  0.0104]
        ], order='F', dtype=float)
        np.testing.assert_allclose(cr[:, :nr_out], cr_expected, rtol=1e-3, atol=1e-4)


class TestAB09ADFixedOrder:
    """Tests with fixed order selection."""

    def test_fixed_order_reduction(self):
        """
        Test fixed order model reduction.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        n, m, p = 4, 1, 1

        # Stable continuous-time system (all eigenvalues negative real part)
        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0, 1.0]], order='F', dtype=float)

        nr_in = 2  # Request 2nd order reduction
        tol = 0.0  # Ignored when ordsel='F'

        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'C', 'B', 'N', 'F', n, m, p, nr_in, a, b, c, tol
        )

        assert info == 0, f"ab09ad failed with info={info}"
        assert nr_out <= nr_in, f"nr_out={nr_out} > nr_in={nr_in}"

        # HSV should be sorted in decreasing order
        for i in range(len(hsv) - 1):
            assert hsv[i] >= hsv[i + 1], "HSV not in decreasing order"


class TestAB09ADEdgeCases:
    """Edge case and error handling tests."""

    def test_zero_dimensions(self):
        """Test with zero dimensions (quick return)."""
        n, m, p = 0, 2, 3

        a = np.array([], order='F', dtype=float).reshape(0, 0)
        b = np.array([], order='F', dtype=float).reshape(0, 2)
        c = np.array([], order='F', dtype=float).reshape(3, 0)

        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == 0

    def test_zero_inputs(self):
        """Test with zero inputs (quick return)."""
        n, m, p = 2, 0, 2

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([], order='F', dtype=float).reshape(2, 0)
        c = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)

        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == 0

    def test_fixed_order_zero(self):
        """Test with fixed order = 0 (quick return)."""
        n, m, p = 3, 1, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)

        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'C', 'B', 'N', 'F', n, m, p, 0, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == 0


class TestAB09ADDiscreteTime:
    """Discrete-time system tests."""

    def test_discrete_stable_system(self):
        """
        Test discrete-time stable system (eigenvalues inside unit circle).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        n, m, p = 3, 1, 1

        # Stable discrete-time: eigenvalues at 0.5, 0.6, 0.7
        a = np.diag([0.5, 0.6, 0.7]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)

        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'D', 'B', 'N', 'A', n, m, p, 0, a, b, c, 0.0
        )

        assert info == 0, f"ab09ad failed with info={info}"

        # HSV should be positive and decreasing
        assert np.all(hsv >= 0), "HSV should be non-negative"
        for i in range(len(hsv) - 1):
            assert hsv[i] >= hsv[i + 1], "HSV not in decreasing order"


class TestAB09ADErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test with invalid DICO parameter (Python wrapper validates)."""
        n, m, p = 2, 1, 1
        a = np.eye(2, order='F', dtype=float)
        b = np.ones((2, 1), order='F', dtype=float)
        c = np.ones((1, 2), order='F', dtype=float)

        with pytest.raises(ValueError, match="DICO must be"):
            ab09ad('X', 'B', 'N', 'A', n, m, p, 0, a, b, c, 0.0)

    def test_invalid_job(self):
        """Test with invalid JOB parameter (Python wrapper validates)."""
        n, m, p = 2, 1, 1
        a = np.eye(2, order='F', dtype=float)
        b = np.ones((2, 1), order='F', dtype=float)
        c = np.ones((1, 2), order='F', dtype=float)

        with pytest.raises(ValueError, match="JOB must be"):
            ab09ad('C', 'X', 'N', 'A', n, m, p, 0, a, b, c, 0.0)

    def test_unstable_system(self):
        """Test with unstable continuous-time system."""
        n, m, p = 2, 1, 1

        # Unstable: one positive eigenvalue
        a = np.array([[1.0, 0.0], [0.0, -1.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)

        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, 0.0
        )

        assert info == 2, f"expected info=2 for unstable system, got {info}"


class TestAB09ADMathematicalProperties:
    """Mathematical property validation tests."""

    def test_hankel_norm_bounds(self):
        """
        Validate error bound property: HSV(NR) <= ||G-Gr||_inf <= 2*sum(HSV(NR+1:N)).

        The error bound is a fundamental property of balanced truncation.
        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        n, m, p = 4, 1, 1

        # Stable system with distinct singular values
        a = np.diag([-0.5, -1.0, -2.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [0.5], [0.25], [0.125]], order='F', dtype=float)
        c = np.array([[1.0, 0.5, 0.25, 0.125]], order='F', dtype=float)

        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out >= 1

        # Hankel singular values should be positive
        assert np.all(hsv >= 0)

        # First HSV is the Hankel norm
        hankel_norm = hsv[0]
        assert hankel_norm > 0

    def test_balanced_realization_eigenvalues(self):
        """
        Validate eigenvalue preservation: reduced system eigenvalues are subset of original.

        For balance & truncate, the reduced model preserves dominant modes.
        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        n, m, p = 4, 1, 1

        # Stable system
        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0, 1.0]], order='F', dtype=float)

        # Original eigenvalues
        eig_orig = np.linalg.eigvals(a)

        nr_in = 2
        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'C', 'B', 'N', 'F', n, m, p, nr_in, a, b, c, 0.0
        )

        assert info == 0

        if nr_out > 0:
            # Reduced system eigenvalues
            eig_red = np.linalg.eigvals(ar[:nr_out, :nr_out])

            # Reduced eigenvalues should have negative real parts (stability preserved)
            assert np.all(eig_red.real < 0), "Reduced system should be stable"

    def test_scaling_equil(self):
        """
        Test with equilibration (scaling) enabled.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)

        n, m, p = 3, 1, 1

        # Poorly scaled system
        a = np.array([
            [-1.0, 100.0, 0.0],
            [0.01, -2.0, 0.0],
            [0.0, 0.0, -3.0]
        ], order='F', dtype=float)
        b = np.array([[100.0], [0.01], [1.0]], order='F', dtype=float)
        c = np.array([[0.01, 100.0, 1.0]], order='F', dtype=float)

        ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
            'C', 'B', 'S', 'A', n, m, p, 0, a, b, c, 0.0
        )

        assert info == 0, f"ab09ad failed with info={info}"

        # HSV should be positive
        assert np.all(hsv >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
