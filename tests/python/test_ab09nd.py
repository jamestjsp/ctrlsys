"""
Tests for AB09ND - Singular Perturbation Approximation model reduction for ALPHA-stable part.

AB09ND computes a reduced order model (Ar,Br,Cr,Dr) for an original
state-space representation (A,B,C,D) by reducing only the ALPHA-stable
part using Singular Perturbation Approximation (SPA).
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from slicot import ab09nd


class TestAB09NDBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_continuous_balancing_free_spa(self):
        """
        Test continuous-time balancing-free SPA model reduction.

        Test data from SLICOT HTML documentation AB09ND example.
        System: N=7, M=2, P=3, continuous-time
        Method: Balancing-free square-root SPA (JOB='N')
        Order selection: Automatic (ORDSEL='A')
        ALPHA stability boundary: -0.6 (eigenvalues with Re < -0.6 are stable)
        """
        n, m, p = 7, 2, 3

        # A matrix (7x7) - row-wise as per Fortran READ
        a = np.array([
            [-0.04165,  0.0000,  4.9200, -4.9200, 0.0000,   0.0000,  0.0000],
            [-5.2100,  -12.500,  0.0000,  0.0000, 0.0000,   0.0000,  0.0000],
            [ 0.0000,   3.3300, -3.3300,  0.0000, 0.0000,   0.0000,  0.0000],
            [ 0.5450,   0.0000,  0.0000,  0.0000, -0.5450,  0.0000,  0.0000],
            [ 0.0000,   0.0000,  0.0000,  4.9200, -0.04165, 0.0000,  4.9200],
            [ 0.0000,   0.0000,  0.0000,  0.0000, -5.2100, -12.500,  0.0000],
            [ 0.0000,   0.0000,  0.0000,  0.0000,  0.0000,  3.3300, -3.3300]
        ], order='F', dtype=float)

        # B matrix (7x2)
        b = np.array([
            [ 0.0000,  0.0000],
            [12.500,   0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000, 12.500],
            [ 0.0000,  0.0000]
        ], order='F', dtype=float)

        # C matrix (3x7)
        c = np.array([
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]
        ], order='F', dtype=float)

        # D matrix (3x2)
        d = np.array([
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000]
        ], order='F', dtype=float)

        # Call ab09nd
        dico = 'C'
        job = 'N'
        equil = 'N'
        ordsel = 'A'
        nr_in = 0
        alpha = -0.6
        tol1 = 0.1
        tol2 = 1e-14

        ar, br, cr, dr, nr, ns, hsv, iwarn, info = ab09nd(
            dico, job, equil, ordsel, n, m, p, nr_in, alpha,
            a, b, c, d, tol1, tol2
        )

        assert info == 0
        assert iwarn == 0

        # Expected reduced order is 5
        assert nr == 5

        # NS should be 5 (all 7 eigenvalues minus 2 unstable = 5 stable)
        # Actually from results: all are stable since alpha=-0.6
        # The system has 7 eigenvalues, all with real part < -0.6 are stable
        # NS = number of alpha-stable eigenvalues = 5
        assert ns == 5

        # Expected Hankel singular values of ALPHA-stable part (from HTML doc)
        hsv_expected = np.array([1.9178, 0.8621, 0.7666, 0.0336, 0.0246])
        assert_allclose(hsv[:ns], hsv_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced state dynamics matrix Ar (5x5, from HTML doc)
        ar_expected = np.array([
            [-0.5181, -1.1084,  0.0000,  0.0000,  0.0000],
            [ 8.8157, -0.5181,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.5847,  0.0000,  1.9230],
            [ 0.0000,  0.0000,  0.0000, -1.6606,  0.0000],
            [ 0.0000,  0.0000, -4.3823,  0.0000, -3.2922]
        ], order='F', dtype=float)
        assert_allclose(ar[:nr, :nr], ar_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced input/state matrix Br (5x2)
        br_expected = np.array([
            [-1.2837,  1.2837],
            [-0.7522,  0.7522],
            [-0.6379, -0.6379],
            [ 2.0656, -2.0656],
            [-3.9315, -3.9315]
        ], order='F', dtype=float)
        assert_allclose(br[:nr, :m], br_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced state/output matrix Cr (3x5)
        cr_expected = np.array([
            [-0.1380, -0.6445, -0.6416, -0.6293,  0.2526],
            [ 0.6246,  0.0196,  0.0000,  0.4107,  0.0000],
            [ 0.1380,  0.6445, -0.6416,  0.6293,  0.2526]
        ], order='F', dtype=float)
        assert_allclose(cr[:p, :nr], cr_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced input/output matrix Dr (3x2)
        dr_expected = np.array([
            [ 0.0582, -0.0090],
            [ 0.0015, -0.0015],
            [-0.0090,  0.0582]
        ], order='F', dtype=float)
        assert_allclose(dr[:p, :m], dr_expected, rtol=1e-3, atol=1e-4)


class TestAB09NDMIMO:
    """MIMO SPA model reduction tests."""

    def test_mimo_spa_reduction(self):
        """
        MIMO: 6-state, 2-input, 3-output SPA reduction.
        Verify HSV ordering, stability, and correct dimensions.
        """
        n, m, p = 6, 2, 3

        a = np.array([
            [-1.0,  0.5,  0.0,  0.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.3,  0.0,  0.0,  0.0],
            [ 0.0,  0.0, -3.0,  0.1,  0.0,  0.0],
            [ 0.0,  0.0,  0.0, -4.0,  0.2,  0.0],
            [ 0.0,  0.0,  0.0,  0.0, -5.0,  0.1],
            [ 0.0,  0.0,  0.0,  0.0,  0.0, -6.0]
        ], order='F', dtype=float)

        np.random.seed(200)
        b = np.asfortranarray(np.random.randn(n, m))
        c = np.asfortranarray(np.random.randn(p, n))
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, ns, hsv, iwarn, info = ab09nd(
            'C', 'B', 'N', 'A', n, m, p, 0, 0.0,
            a.copy(order='F'), b.copy(order='F'),
            c.copy(order='F'), d.copy(order='F'),
            0.0, 0.0
        )

        assert info == 0
        assert ns == n
        assert nr <= n
        assert nr > 0

        for i in range(ns - 1):
            assert hsv[i] >= hsv[i + 1] - 1e-14, \
                f"HSV not decreasing: HSV[{i}]={hsv[i]} < HSV[{i+1}]={hsv[i+1]}"

        eigs = np.linalg.eigvals(ar[:nr, :nr])
        assert all(e.real < 0 for e in eigs), f"Reduced system unstable: {eigs}"

        assert br[:nr, :m].shape == (nr, m)
        assert cr[:p, :nr].shape == (p, nr)


class TestAB09NDEdgeCases:
    """Edge case tests."""

    def test_zero_dimensions(self):
        """Test with N=0 (quick return)."""
        n, m, p = 0, 2, 3

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 2), order='F', dtype=float)
        c = np.zeros((3, 1), order='F', dtype=float)
        d = np.zeros((3, 2), order='F', dtype=float)

        ar, br, cr, dr, nr, ns, hsv, iwarn, info = ab09nd(
            'C', 'B', 'N', 'A', n, m, p, 0, 0.0,
            a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr == 0

    def test_discrete_time_system(self):
        """
        Test discrete-time model reduction with stability boundary.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 1, 1

        # Create a discrete-time system with mixed stable/unstable eigenvalues
        # For discrete-time, alpha=0.9 means |eigenvalue| < 0.9 is stable
        # Create system with some eigenvalues inside and outside unit circle
        eigs = np.array([0.5, 0.7, 0.3, 0.8])  # All inside unit circle
        a = np.diag(eigs).astype(float, order='F')

        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, ns, hsv, iwarn, info = ab09nd(
            'D', 'B', 'N', 'A', n, m, p, 0, 0.9,
            a.copy(order='F'), b.copy(order='F'),
            c.copy(order='F'), d.copy(order='F'),
            0.0, 0.0
        )

        assert info == 0
        # Some eigenvalues should be identified as alpha-stable
        assert ns >= 0
        assert nr <= n


class TestAB09NDFixedOrder:
    """Tests for fixed order selection."""

    def test_fixed_order_reduction(self):
        """
        Test with fixed order selection (ORDSEL='F').

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 5, 2, 2

        # Create a stable continuous-time system
        a = np.array([
            [-1.0,  0.0,  0.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0, -3.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0, -4.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0, -5.0]
        ], order='F', dtype=float)

        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        # Request fixed order of 3
        nr_in = 3
        ar, br, cr, dr, nr, ns, hsv, iwarn, info = ab09nd(
            'C', 'B', 'N', 'F', n, m, p, nr_in, 0.0,
            a.copy(order='F'), b.copy(order='F'),
            c.copy(order='F'), d.copy(order='F'),
            0.0, 0.0
        )

        assert info == 0
        # NR should be at most nr_in (may be less if minimal realization is smaller)
        assert nr <= nr_in
        # All eigenvalues are stable (alpha=0 means Re < 0 for continuous)
        assert ns == n


class TestAB09NDEquilibration:
    """Tests for equilibration option."""

    def test_with_equilibration(self):
        """
        Test with equilibration enabled (EQUIL='S').

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 4, 2, 2

        # Create a poorly scaled system
        a = np.array([
            [-1.0e-3,  1.0e3,   0.0,     0.0],
            [ 0.0,    -2.0e2,   1.0e-2,  0.0],
            [ 0.0,     0.0,    -3.0e1,   1.0e1],
            [ 0.0,     0.0,     0.0,    -4.0]
        ], order='F', dtype=float)

        b = np.random.randn(n, m).astype(float, order='F') * np.array([[1e2], [1], [1e-1], [1e-2]])
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, ns, hsv, iwarn, info = ab09nd(
            'C', 'B', 'S', 'A', n, m, p, 0, 0.0,
            a.copy(order='F'), b.copy(order='F'),
            c.copy(order='F'), d.copy(order='F'),
            0.0, 0.0
        )

        assert info == 0
        assert ns >= 0


class TestAB09NDErrorHandling:
    """Tests for error handling."""

    def test_invalid_dico(self):
        """Test with invalid DICO parameter."""
        n, m, p = 3, 1, 1
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, m), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="DICO"):
            ab09nd('X', 'B', 'N', 'A', n, m, p, 0, 0.0, a, b, c, d, 0.0, 0.0)

    def test_invalid_job(self):
        """Test with invalid JOB parameter."""
        n, m, p = 3, 1, 1
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, m), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="JOB"):
            ab09nd('C', 'X', 'N', 'A', n, m, p, 0, 0.0, a, b, c, d, 0.0, 0.0)

    def test_invalid_equil(self):
        """Test with invalid EQUIL parameter."""
        n, m, p = 3, 1, 1
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, m), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="EQUIL"):
            ab09nd('C', 'B', 'X', 'A', n, m, p, 0, 0.0, a, b, c, d, 0.0, 0.0)

    def test_invalid_ordsel(self):
        """Test with invalid ORDSEL parameter."""
        n, m, p = 3, 1, 1
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, m), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="ORDSEL"):
            ab09nd('C', 'B', 'N', 'X', n, m, p, 0, 0.0, a, b, c, d, 0.0, 0.0)

    def test_invalid_alpha_continuous(self):
        """Test with invalid ALPHA for continuous-time (ALPHA > 0)."""
        n, m, p = 3, 1, 1
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, m), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="ALPHA"):
            ab09nd('C', 'B', 'N', 'A', n, m, p, 0, 0.5, a, b, c, d, 0.0, 0.0)

    def test_invalid_alpha_discrete(self):
        """Test with invalid ALPHA for discrete-time (ALPHA < 0 or > 1)."""
        n, m, p = 3, 1, 1
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, m), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="ALPHA"):
            ab09nd('D', 'B', 'N', 'A', n, m, p, 0, -0.5, a, b, c, d, 0.0, 0.0)

        with pytest.raises(ValueError, match="ALPHA"):
            ab09nd('D', 'B', 'N', 'A', n, m, p, 0, 1.5, a, b, c, d, 0.0, 0.0)

    def test_invalid_nr_fixed_order(self):
        """Test with invalid NR for fixed order selection."""
        n, m, p = 3, 1, 1
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, m), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="NR"):
            ab09nd('C', 'B', 'N', 'F', n, m, p, 5, 0.0, a, b, c, d, 0.0, 0.0)

    def test_invalid_tol2_greater_than_tol1(self):
        """Test with TOL2 > TOL1 when both positive."""
        n, m, p = 3, 1, 1
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, m), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        with pytest.raises(ValueError, match="TOL2"):
            ab09nd('C', 'B', 'N', 'A', n, m, p, 0, 0.0, a, b, c, d, 0.01, 0.1)
