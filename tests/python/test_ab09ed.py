"""
Tests for AB09ED: Optimal Hankel-norm approximation based model reduction
for unstable systems.

AB09ED computes a reduced order model (Ar,Br,Cr,Dr) for an original
state-space representation (A,B,C,D) by using the optimal Hankel-norm
approximation method in conjunction with square-root balancing for the
ALPHA-stable part of the system.
"""

import numpy as np
import pytest
from ctrlsys import ab09ed


class TestAB09EDBasic:
    """Test basic functionality using HTML doc example."""

    def test_continuous_time_example(self):
        """
        Test AB09ED with continuous-time example from HTML documentation.

        System has N=7 states, M=2 inputs, P=3 outputs.
        ALPHA=-0.6 boundary, TOL1=0.1 for order selection.
        """
        n, m, p = 7, 2, 3
        nr_in = 0
        alpha = -0.6
        tol1 = 0.1
        tol2 = 1e-14

        A = np.array([
            [-0.04165,  0.0000,  4.9200, -4.9200,  0.0000,  0.0000,  0.0000],
            [-5.2100, -12.500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  3.3300, -3.3300,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.5450,  0.0000,  0.0000,  0.0000, -0.5450,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  4.9200, -0.04165, 0.0000,  4.9200],
            [ 0.0000,  0.0000,  0.0000,  0.0000, -5.2100, -12.500,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  3.3300, -3.3300],
        ], order='F', dtype=float)

        B = np.array([
            [ 0.0000,  0.0000],
            [12.500,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000, 12.500],
            [ 0.0000,  0.0000],
        ], order='F', dtype=float)

        C = np.array([
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
        ], order='F', dtype=float)

        D = np.array([
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
        ], order='F', dtype=float)

        Ar, Br, Cr, Dr, nr, ns, hsv, nmin, iwarn, info = ab09ed(
            'C', 'N', 'A', n, m, p, nr_in, alpha,
            A, B, C, D, tol1, tol2
        )

        assert info == 0, f"AB09ED failed with info={info}"
        assert nr == 5, f"Expected nr=5, got nr={nr}"

        hsv_expected = np.array([1.9178, 0.8621, 0.7666, 0.0336, 0.0246])
        np.testing.assert_allclose(hsv[:ns], hsv_expected, rtol=1e-3, atol=1e-4)

        Ar_expected = np.array([
            [-0.5181, -1.1084,  0.0000,  0.0000,  0.0000],
            [ 8.8157, -0.5181,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000, -1.2769,  7.3264,  0.0000],
            [ 0.0000,  0.0000, -0.6203, -1.2769,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000, -1.5496],
        ], order='F', dtype=float)
        # Compare using absolute values for off-diagonal 2x2 blocks (sign ambiguity)
        Ar_actual = Ar[:nr, :nr].copy()
        np.testing.assert_allclose(np.abs(Ar_actual), np.abs(Ar_expected), rtol=1e-3, atol=1e-4)

        Br_expected = np.array([
            [-1.2837,  1.2837],
            [-0.7522,  0.7522],
            [ 3.2016,  3.2016],
            [-0.7640, -0.7640],
            [ 1.3415, -1.3415],
        ], order='F', dtype=float)

        Cr_expected = np.array([
            [-0.1380, -0.6445, -0.6247, -2.0857, -0.8964],
            [ 0.6246,  0.0196,  0.0000,  0.0000,  0.6131],
            [ 0.1380,  0.6445, -0.6247, -2.0857,  0.8964],
        ], order='F', dtype=float)

        # Fix sign ambiguity: individual states can have sign flips
        # Detect by comparing Br row signs and flip if needed
        Br_actual = Br[:nr, :m].copy()
        Cr_actual = Cr[:p, :nr].copy()
        for i in range(nr):
            if np.abs(Br_expected[i, 0]) > 1e-6:
                if np.sign(Br_actual[i, 0]) != np.sign(Br_expected[i, 0]):
                    Br_actual[i, :] *= -1
                    Cr_actual[:, i] *= -1

        np.testing.assert_allclose(Br_actual, Br_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(Cr_actual, Cr_expected, rtol=1e-3, atol=1e-4)

        Dr_expected = np.array([
            [ 0.0168, -0.0168],
            [ 0.0008, -0.0008],
            [-0.0168,  0.0168],
        ], order='F', dtype=float)
        np.testing.assert_allclose(Dr[:p, :m], Dr_expected, rtol=1e-3, atol=1e-4)


class TestAB09EDEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_dimensions(self):
        """Test with n=0 (empty system)."""
        n, m, p = 0, 1, 1
        nr_in = 0
        alpha = 0.0
        tol1, tol2 = 0.0, 0.0

        A = np.zeros((1, 1), order='F', dtype=float)
        B = np.zeros((1, 1), order='F', dtype=float)
        C = np.zeros((1, 1), order='F', dtype=float)
        D = np.zeros((1, 1), order='F', dtype=float)

        Ar, Br, Cr, Dr, nr, ns, hsv, nmin, iwarn, info = ab09ed(
            'C', 'N', 'A', n, m, p, nr_in, alpha,
            A, B, C, D, tol1, tol2
        )

        assert info == 0
        assert nr == 0
        assert ns == 0

    def test_fixed_order_selection(self):
        """
        Test fixed order selection (ORDSEL='F').

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 1, 1
        nr_in = 2
        alpha = -0.1
        tol1, tol2 = 0.0, 0.0

        A = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0, -4.0],
        ], order='F', dtype=float)

        B = np.array([[1.0], [1.0], [1.0], [1.0]], order='F', dtype=float)
        C = np.array([[1.0, 1.0, 1.0, 1.0]], order='F', dtype=float)
        D = np.zeros((1, 1), order='F', dtype=float)

        Ar, Br, Cr, Dr, nr, ns, hsv, nmin, iwarn, info = ab09ed(
            'C', 'N', 'F', n, m, p, nr_in, alpha,
            A, B, C, D, tol1, tol2
        )

        assert info == 0
        assert nr <= n


class TestAB09EDDiscreteTime:
    """Test discrete-time systems."""

    def test_discrete_time_stable(self):
        """
        Test discrete-time system with stable eigenvalues.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 3, 1, 1
        nr_in = 0
        alpha = 0.9
        tol1, tol2 = 0.0, 0.0

        A = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.6, 0.0],
            [0.0, 0.0, 0.7],
        ], order='F', dtype=float)

        B = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        C = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        D = np.zeros((1, 1), order='F', dtype=float)

        Ar, Br, Cr, Dr, nr, ns, hsv, nmin, iwarn, info = ab09ed(
            'D', 'N', 'A', n, m, p, nr_in, alpha,
            A, B, C, D, tol1, tol2
        )

        assert info == 0
        assert ns == n


class TestAB09EDErrorHandling:
    """Test error handling."""

    def test_invalid_dico(self):
        """Test invalid DICO parameter."""
        n, m, p = 2, 1, 1
        A = np.eye(n, order='F', dtype=float)
        B = np.ones((n, m), order='F', dtype=float)
        C = np.ones((p, n), order='F', dtype=float)
        D = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="DICO"):
            ab09ed('X', 'N', 'A', n, m, p, 0, 0.0, A, B, C, D, 0.0, 0.0)

    def test_invalid_equil(self):
        """Test invalid EQUIL parameter."""
        n, m, p = 2, 1, 1
        A = np.eye(n, order='F', dtype=float)
        B = np.ones((n, m), order='F', dtype=float)
        C = np.ones((p, n), order='F', dtype=float)
        D = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="EQUIL"):
            ab09ed('C', 'X', 'A', n, m, p, 0, 0.0, A, B, C, D, 0.0, 0.0)

    def test_invalid_ordsel(self):
        """Test invalid ORDSEL parameter."""
        n, m, p = 2, 1, 1
        A = np.eye(n, order='F', dtype=float)
        B = np.ones((n, m), order='F', dtype=float)
        C = np.ones((p, n), order='F', dtype=float)
        D = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="ORDSEL"):
            ab09ed('C', 'N', 'X', n, m, p, 0, 0.0, A, B, C, D, 0.0, 0.0)

    def test_invalid_alpha_continuous(self):
        """Test invalid ALPHA for continuous-time (must be <= 0)."""
        n, m, p = 2, 1, 1
        A = np.eye(n, order='F', dtype=float)
        B = np.ones((n, m), order='F', dtype=float)
        C = np.ones((p, n), order='F', dtype=float)
        D = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="ALPHA"):
            ab09ed('C', 'N', 'A', n, m, p, 0, 0.5, A, B, C, D, 0.0, 0.0)

    def test_invalid_alpha_discrete(self):
        """Test invalid ALPHA for discrete-time (must be in [0,1])."""
        n, m, p = 2, 1, 1
        A = np.eye(n, order='F', dtype=float) * 0.5
        B = np.ones((n, m), order='F', dtype=float)
        C = np.ones((p, n), order='F', dtype=float)
        D = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="ALPHA"):
            ab09ed('D', 'N', 'A', n, m, p, 0, 1.5, A, B, C, D, 0.0, 0.0)

    def test_tol2_greater_than_tol1(self):
        """Test error when TOL2 > TOL1 and TOL2 > 0."""
        n, m, p = 2, 1, 1
        A = -np.eye(n, order='F', dtype=float)
        B = np.ones((n, m), order='F', dtype=float)
        C = np.ones((p, n), order='F', dtype=float)
        D = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="TOL2"):
            ab09ed('C', 'N', 'A', n, m, p, 0, 0.0, A, B, C, D, 0.01, 0.1)


class TestAB09EDEquilibration:
    """Test equilibration option."""

    def test_with_equilibration(self):
        """
        Test with equilibration (EQUIL='S').

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 4, 2, 2
        nr_in = 0
        alpha = 0.0
        tol1, tol2 = 0.0, 0.0

        A = np.array([
            [-1.0, 0.1, 0.0, 0.0],
            [0.0, -2.0, 0.2, 0.0],
            [0.0, 0.0, -3.0, 0.3],
            [0.0, 0.0, 0.0, -4.0],
        ], order='F', dtype=float)

        B = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], order='F', dtype=float)

        C = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ], order='F', dtype=float)

        D = np.zeros((p, m), order='F', dtype=float)

        Ar, Br, Cr, Dr, nr, ns, hsv, nmin, iwarn, info = ab09ed(
            'C', 'S', 'A', n, m, p, nr_in, alpha,
            A, B, C, D, tol1, tol2
        )

        assert info == 0
        assert nr >= 0
        assert ns >= 0


class TestAB09EDMathematicalProperties:
    """Test mathematical properties of model reduction."""

    def test_hankel_singular_values_decreasing(self):
        """
        Validate that Hankel singular values are in decreasing order.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 5, 1, 1
        nr_in = 0
        alpha = 0.0
        tol1, tol2 = 0.0, 0.0

        A = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        B = np.ones((n, m), order='F', dtype=float)
        C = np.ones((p, n), order='F', dtype=float)
        D = np.zeros((p, m), order='F', dtype=float)

        Ar, Br, Cr, Dr, nr, ns, hsv, nmin, iwarn, info = ab09ed(
            'C', 'N', 'A', n, m, p, nr_in, alpha,
            A, B, C, D, tol1, tol2
        )

        assert info == 0
        if ns > 1:
            for i in range(ns - 1):
                assert hsv[i] >= hsv[i + 1], f"HSV not decreasing at index {i}"

    def test_reduced_system_stability(self):
        """
        Validate reduced system preserves stability.

        For a stable original system, the reduced system should also be stable.

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        n, m, p = 4, 1, 1
        nr_in = 2
        alpha = 0.0
        tol1, tol2 = 0.0, 0.0

        A = np.array([
            [-1.0, 0.5, 0.0, 0.0],
            [0.0, -2.0, 0.5, 0.0],
            [0.0, 0.0, -3.0, 0.5],
            [0.0, 0.0, 0.0, -4.0],
        ], order='F', dtype=float)

        B = np.ones((n, m), order='F', dtype=float)
        C = np.ones((p, n), order='F', dtype=float)
        D = np.zeros((p, m), order='F', dtype=float)

        Ar, Br, Cr, Dr, nr, ns, hsv, nmin, iwarn, info = ab09ed(
            'C', 'N', 'F', n, m, p, nr_in, alpha,
            A, B, C, D, tol1, tol2
        )

        assert info == 0

        if nr > 0:
            eigenvalues = np.linalg.eigvals(Ar[:nr, :nr])
            for ev in eigenvalues:
                assert ev.real < 0, f"Reduced system unstable: eigenvalue {ev}"
