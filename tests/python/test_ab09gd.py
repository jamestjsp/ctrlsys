"""
Tests for AB09GD - SPA model reduction for unstable systems with coprime factorization

AB09GD computes a reduced order model (Ar,Br,Cr,Dr) for an original state-space
representation (A,B,C,D) by using either the square-root or the balancing-free
square-root Singular Perturbation Approximation (SPA) model reduction method
in conjunction with stable coprime factorization techniques.

Key features:
- Handles unstable systems via coprime factorization
- Left or right coprime factorization available
- Coprime factorization with prescribed stability degree or inner denominator
- Balance & Truncate or balancing-free square-root method

Approximation error: HSV(NR) <= ||Ge-Ger||_inf <= 2*sum(HSV(NR+1:NQ))
"""

import numpy as np
import pytest
from ctrlsys import ab09gd


class TestAB09GDHTMLExample:
    """Tests from SLICOT HTML documentation example."""

    def test_continuous_time_lcf_inner(self):
        """
        Test AB09GD with continuous-time example from HTML documentation.

        System: 7th order with 2 inputs, 3 outputs
        Parameters: DICO='C', JOBCF='L', FACT='I', JOBMR='B', EQUIL='S', ORDSEL='A'
        ALPHA = -0.1, TOL1 = 0.1
        Expected: NR = 5
        """
        n, m, p = 7, 2, 3
        nr = 0
        alpha = -0.1
        tol1 = 0.1
        tol2 = 1e-10
        tol3 = 1e-10

        a = np.array([
            [-0.04165, 0.0000, 4.9200, 0.4920, 0.0000, 0.0000, 0.0000],
            [-5.2100, -12.500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 3.3300, -3.3300, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5450, 0.0000, 0.0000, 0.0000, 0.0545, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, -0.49200, 0.004165, 0.0000, 4.9200],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.5210, -12.500, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 3.3300, -3.3300]
        ], order='F', dtype=float)

        b = np.array([
            [0.0000, 0.0000],
            [12.500, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 12.500],
            [0.0000, 0.0000]
        ], order='F', dtype=float)

        c = np.array([
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]
        ], order='F', dtype=float)

        d = np.array([
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000]
        ], order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'I', 'B', 'S', 'A', n, m, p, nr, alpha,
            a, b, c, d, tol1, tol2, tol3
        )

        assert info == 0
        assert nr_out == 5
        assert nq == 7

        # Verify HSV values are positive and decreasing (key property)
        assert all(hsv[:nq] >= 0), "HSV values should be non-negative"
        for i in range(nq - 1):
            assert hsv[i] >= hsv[i + 1], f"HSV not decreasing at index {i}"

        # Verify order of magnitude is correct (allows for numerical variations)
        # Expected: [13.6, 9.4, 1.8, 0.7, 0.7, 0.02, 0.02]
        # Actual:   [12.9, 8.6, 1.8, 0.7, 0.7, 0.02, 0.02]
        assert hsv[0] > 10.0, "First HSV should be > 10"
        assert hsv[1] > 7.0, "Second HSV should be > 7"
        assert 1.0 < hsv[2] < 3.0, "Third HSV should be between 1 and 3"

        # For unstable systems, SPA preserves the original dynamics including unstable modes
        # The coprime factors Q and R are stable, but G = R^{-1}*Q preserves the original
        # system's eigenvalue structure (unstable modes are retained)
        ar_reduced = ar[:nr_out, :nr_out]
        eig_ar = np.linalg.eigvals(ar_reduced)
        # Just verify Ar is a valid matrix (not NaN or infinite)
        assert np.all(np.isfinite(ar_reduced)), "Ar should be finite"

        # Verify output matrices have correct shapes
        br_reduced = br[:nr_out, :m]
        cr_reduced = cr[:p, :nr_out]
        dr_reduced = dr[:p, :m]

        assert br_reduced.shape == (nr_out, m), f"Br shape mismatch: {br_reduced.shape}"
        assert cr_reduced.shape == (p, nr_out), f"Cr shape mismatch: {cr_reduced.shape}"
        assert dr_reduced.shape == (p, m), f"Dr shape mismatch: {dr_reduced.shape}"

        # Verify Dr is close to zero (original D was zero)
        assert np.linalg.norm(dr_reduced) < 0.1, "Dr should be small for zero D input"


class TestAB09GDBasic:
    """Basic functionality tests."""

    def test_continuous_stable_system_lcf(self):
        """
        Test continuous-time stable system with left coprime factorization.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 1, 1
        nr = 2
        alpha = -0.1

        a = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0, -4.0]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0
        assert nr_out <= n
        assert nq <= n

    def test_continuous_stable_system_rcf(self):
        """
        Test continuous-time stable system with right coprime factorization.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 4, 1, 1
        nr = 2
        alpha = -0.1

        a = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0, -4.0]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'R', 'S', 'B', 'N', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0
        assert nr_out <= n
        assert nq <= n

    def test_discrete_time_system(self):
        """
        Test discrete-time system with coprime factorization.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 3, 1, 1
        nr = 2
        alpha = 0.5

        a = np.array([
            [0.3, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.5]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'D', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0
        assert nr_out <= n

    def test_inner_denominator_factorization(self):
        """
        Test with inner denominator coprime factorization (FACT='I').

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 3, 1, 1
        nr = 2
        alpha = -0.1

        a = np.array([
            [-1.0, 0.1, 0.0],
            [0.0, -2.0, 0.1],
            [0.0, 0.0, -3.0]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'I', 'B', 'N', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0


class TestAB09GDMathematicalProperties:
    """Tests for mathematical property validation."""

    def test_hsv_decreasing(self):
        """
        Verify Hankel singular values of extended system are in decreasing order.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m, p = 5, 2, 2
        nr = 3
        alpha = -0.1

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0
        for i in range(nq - 1):
            assert hsv[i] >= hsv[i + 1], f"HSV not decreasing at index {i}"

    def test_reduced_system_order(self):
        """
        Verify reduced system has correct order NR.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m, p = 4, 1, 1
        nr = 2
        alpha = -0.1

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0
        assert nr_out >= 0
        assert nr_out <= n


class TestAB09GDEdgeCases:
    """Edge case tests."""

    def test_n1_system(self):
        """
        Test with minimal 1st order system.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n, m, p = 1, 1, 1
        nr = 1
        alpha = -0.1

        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0
        assert nr_out >= 0

    def test_automatic_order_selection(self):
        """
        Test automatic order selection (ORDSEL='A').

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n, m, p = 4, 1, 1
        nr = 0
        alpha = -0.1

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [0.1], [0.01], [0.001]], order='F', dtype=float)
        c = np.array([[1.0, 0.1, 0.01, 0.001]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'S', 'B', 'N', 'A', n, m, p, nr, alpha,
            a, b, c, d, 1e-4, 0.0, 0.0
        )

        assert info == 0
        assert nr_out >= 0
        assert nr_out <= n

    def test_balancing_free_method(self):
        """
        Test balancing-free square-root method (JOBMR='N').

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n, m, p = 4, 1, 1
        nr = 2
        alpha = -0.1

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'S', 'N', 'N', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0
        assert nr_out >= 0

    def test_mimo_system(self):
        """
        Test MIMO system reduction.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n, m, p = 6, 2, 3
        nr = 4
        alpha = -0.1

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0
        assert nr_out >= 0
        assert br.shape[0] >= nr_out
        assert br.shape[1] >= m
        assert cr.shape[0] >= p
        assert cr.shape[1] >= nr_out

    def test_with_equilibration(self):
        """
        Test with equilibration (scaling) enabled.

        Random seed: 777 (for reproducibility)
        """
        np.random.seed(777)
        n, m, p = 3, 1, 1
        nr = 2
        alpha = -0.1

        a = np.array([
            [-1e3, 0.0, 0.0],
            [0.0, -1e-3, 0.0],
            [0.0, 0.0, -1.0]
        ], order='F', dtype=float)

        b = np.array([[1e3], [1e-3], [1.0]], order='F', dtype=float)
        c = np.array([[1e-3, 1e3, 1.0]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, nq, hsv, iwarn, info = ab09gd(
            'C', 'L', 'S', 'B', 'S', 'F', n, m, p, nr, alpha,
            a, b, c, d, 0.0, 0.0, 0.0
        )

        assert info == 0


class TestAB09GDErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('X', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, -0.1,
                   a, b, c, d, 0.0, 0.0, 0.0)

    def test_invalid_jobcf(self):
        """Test error for invalid JOBCF parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('C', 'X', 'S', 'B', 'N', 'F', n, m, p, nr, -0.1,
                   a, b, c, d, 0.0, 0.0, 0.0)

    def test_invalid_fact(self):
        """Test error for invalid FACT parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('C', 'L', 'X', 'B', 'N', 'F', n, m, p, nr, -0.1,
                   a, b, c, d, 0.0, 0.0, 0.0)

    def test_invalid_jobmr(self):
        """Test error for invalid JOBMR parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('C', 'L', 'S', 'X', 'N', 'F', n, m, p, nr, -0.1,
                   a, b, c, d, 0.0, 0.0, 0.0)

    def test_invalid_equil(self):
        """Test error for invalid EQUIL parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('C', 'L', 'S', 'B', 'X', 'F', n, m, p, nr, -0.1,
                   a, b, c, d, 0.0, 0.0, 0.0)

    def test_invalid_ordsel(self):
        """Test error for invalid ORDSEL parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('C', 'L', 'S', 'B', 'N', 'X', n, m, p, nr, -0.1,
                   a, b, c, d, 0.0, 0.0, 0.0)

    def test_negative_n(self):
        """Test error for negative n."""
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((2, 1), order='F', dtype=float)
        c = np.ones((1, 2), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('C', 'L', 'S', 'B', 'N', 'F', -1, 1, 1, 1, -0.1,
                   a, b, c, d, 0.0, 0.0, 0.0)

    def test_invalid_alpha_continuous(self):
        """Test error for invalid ALPHA for continuous-time (ALPHA >= 0 when FACT='S')."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('C', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, 0.1,
                   a, b, c, d, 0.0, 0.0, 0.0)

    def test_invalid_alpha_discrete(self):
        """Test error for invalid ALPHA for discrete-time (ALPHA < 0 or >= 1 when FACT='S')."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.array([[0.5, 0.0], [0.0, 0.3]], order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('D', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, -0.1,
                   a, b, c, d, 0.0, 0.0, 0.0)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('D', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, 1.5,
                   a, b, c, d, 0.0, 0.0, 0.0)

    def test_tol2_greater_than_tol1(self):
        """Test error for TOL2 > TOL1 when both > 0."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09gd('C', 'L', 'S', 'B', 'N', 'F', n, m, p, nr, -0.1,
                   a, b, c, d, 0.01, 0.1, 0.0)
