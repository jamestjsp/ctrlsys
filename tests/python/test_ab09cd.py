"""
Tests for AB09CD - Optimal Hankel-norm approximation based model reduction

AB09CD computes a reduced order model (Ar,Br,Cr,Dr) for a stable original
state-space representation (A,B,C,D) by using the optimal Hankel-norm
approximation method in conjunction with square-root balancing.

Unlike AB09CX, AB09CD accepts a general state matrix A (not necessarily
in Schur form) and optionally performs equilibration (scaling) of (A,B,C).

Method:
    The optimal Hankel-norm approximation method, based on square-root
    balancing projection formulas, is employed to produce a reduced system
    such that: HSV(NR) <= ||G-Gr||_inf <= 2*sum(HSV(NR+1:N))

Parameters:
    DICO: 'C' continuous-time, 'D' discrete-time
    EQUIL: 'S' scale, 'N' no scaling
    ORDSEL: 'F' fixed order, 'A' automatic order selection

Error Codes:
    info=0: success
    info=1: reduction of A to real Schur form failed
    info=2: A is not stable (continuous) or not convergent (discrete)
    info=3: computation of Hankel singular values failed
    info=4: computation of stable projection failed
    info=5: order of computed stable projection differs from Hankel-norm approx
"""

import numpy as np
import pytest
from ctrlsys import ab09cd


class TestAB09CDBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_html_doc_example(self):
        """
        Test using exact data from SLICOT HTML documentation.

        This is the AB09CD example program data:
        7th order continuous-time system reduced with TOL1=0.1

        Expected results from HTML doc:
        - NR = 5
        - HSV = [2.5139, 2.0846, 1.9178, 0.7666, 0.5473, 0.0253, 0.0246]
        """
        n, m, p = 7, 2, 3
        nr_input = 0  # Will be determined by automatic order selection

        # Input data from HTML doc (read row-by-row as per Fortran READ)
        a = np.array([
            [-0.04165,  0.0000,  4.9200, -4.9200,  0.0000,  0.0000,  0.0000],
            [-5.2100, -12.500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  3.3300, -3.3300,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.5450,  0.0000,  0.0000,  0.0000, -0.5450,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  4.9200, -0.04165, 0.0000,  4.9200],
            [ 0.0000,  0.0000,  0.0000,  0.0000, -5.2100, -12.500,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  3.3300, -3.3300]
        ], order='F', dtype=float)

        b = np.array([
            [ 0.0000,  0.0000],
            [12.500,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000, 12.500],
            [ 0.0000,  0.0000]
        ], order='F', dtype=float)

        c = np.array([
            [1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000],
            [0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000]
        ], order='F', dtype=float)

        d = np.zeros((p, m), order='F', dtype=float)

        # Expected HSV from HTML doc
        hsv_expected = np.array([2.5139, 2.0846, 1.9178, 0.7666, 0.5473, 0.0253, 0.0246])

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'A', n, m, p, nr_input, a, b, c, d, 0.1, 1e-14
        )

        assert info == 0, f"AB09CD failed with info={info}"
        assert nr == 5, f"Expected NR=5, got NR={nr}"

        # Validate HSV values (4 decimal places from HTML)
        np.testing.assert_allclose(hsv, hsv_expected, rtol=1e-3, atol=1e-4)

    def test_continuous_time_fixed_order(self):
        """
        Test continuous-time Hankel-norm approximation with fixed order.

        Uses a stable 4th order system reduced to 2nd order.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 2, 2
        nr_desired = 2

        a = np.array([
            [-2.0,  1.0,  0.0,  0.0],
            [ 0.0, -3.0,  0.0,  0.0],
            [ 0.0,  0.0, -1.0,  0.5],
            [ 0.0,  0.0, -0.5, -1.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ], order='F', dtype=float)

        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert iwarn == 0 or iwarn == 1
        assert nr <= nr_desired
        assert nr >= 0

        # Validate HSV ordering
        assert hsv.shape[0] == n
        for i in range(n - 1):
            if hsv[i+1] > 1e-15:
                assert hsv[i] >= hsv[i+1] - 1e-14

        # Validate reduced system stability
        if nr > 0:
            ar_reduced = ar[:nr, :nr]
            eigs = np.linalg.eigvals(ar_reduced)
            assert all(np.real(eigs) < 0), "Reduced system must be stable"

    def test_discrete_time_fixed_order(self):
        """
        Test discrete-time Hankel-norm approximation with fixed order.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 4, 1, 1
        nr_desired = 2

        # Stable discrete system (eigenvalues inside unit circle)
        a = np.array([
            [ 0.5,  0.2,  0.0,  0.0],
            [ 0.0,  0.3,  0.0,  0.0],
            [ 0.0,  0.0,  0.4,  0.1],
            [ 0.0,  0.0, -0.1,  0.4]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5],
            [0.3],
            [0.1]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.3, 0.1]
        ], order='F', dtype=float)

        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'D', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr <= nr_desired
        assert nr >= 0

        # Validate reduced system convergence
        if nr > 0:
            ar_reduced = ar[:nr, :nr]
            eigs = np.linalg.eigvals(ar_reduced)
            assert all(np.abs(eigs) < 1.0), "Reduced discrete system must be convergent"


class TestAB09CDMathematicalProperties:
    """Tests for mathematical property validation."""

    def test_hsv_ordering(self):
        """
        Verify Hankel singular values are returned in decreasing order.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 5, 2, 2
        nr_desired = 3

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        for i in range(n - 1):
            if hsv[i+1] > 1e-15:
                assert hsv[i] >= hsv[i+1] - 1e-14, \
                    f"HSV not decreasing: hsv[{i}]={hsv[i]} < hsv[{i+1}]={hsv[i+1]}"

    def test_automatic_order_selection(self):
        """
        Test automatic order selection based on tolerance.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m, p = 5, 1, 1

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        b = np.array([[1.0], [0.5], [0.1], [0.01], [0.001]], order='F', dtype=float)
        c = np.array([[1.0, 0.5, 0.1, 0.01, 0.001]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'A', n, m, p, 0, a, b, c, d, 0.01, 0.0
        )

        assert info == 0
        assert nr <= n
        assert nr >= 0

    def test_reduced_system_stability_continuous(self):
        """
        Verify reduced system preserves stability for continuous-time.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n, m, p = 6, 2, 2
        nr_desired = 3

        a = np.array([
            [-1.0,  0.5,  0.2,  0.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.3,  0.0,  0.0,  0.0],
            [ 0.0,  0.0, -3.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0, -4.0,  0.1,  0.0],
            [ 0.0,  0.0,  0.0,  0.0, -5.0,  0.2],
            [ 0.0,  0.0,  0.0,  0.0,  0.0, -6.0]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float) * 0.5
        c = np.ones((p, n), order='F', dtype=float) * 0.3
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        if nr > 0:
            ar_reduced = ar[:nr, :nr]
            eigs = np.linalg.eigvals(ar_reduced)
            assert all(np.real(eigs) < 1e-10), \
                f"Reduced system not stable: eigenvalues = {eigs}"

    def test_equilibration_effect(self):
        """
        Test that equilibration (scaling) is applied when EQUIL='S'.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n, m, p = 4, 2, 2
        nr_desired = 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([
            [1.0, 0.5],
            [0.8, 0.3],
            [0.2, 0.6],
            [0.1, 0.4]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.5, 0.3, 0.1],
            [0.6, 0.4, 0.2, 0.3]
        ], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        # With scaling
        ar1, br1, cr1, dr1, nr1, hsv1, nmin1, iwarn1, info1 = ab09cd(
            'C', 'S', 'F', n, m, p, nr_desired, a.copy(), b.copy(), c.copy(), d.copy(), 0.0, 0.0
        )

        # Without scaling
        ar2, br2, cr2, dr2, nr2, hsv2, nmin2, iwarn2, info2 = ab09cd(
            'C', 'N', 'F', n, m, p, nr_desired, a.copy(), b.copy(), c.copy(), d.copy(), 0.0, 0.0
        )

        assert info1 == 0
        assert info2 == 0


class TestAB09CDEdgeCases:
    """Edge case tests."""

    def test_minimal_system(self):
        """
        Test with a minimal 2x2 system.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n, m, p = 2, 1, 1
        nr_desired = 1

        a = np.array([
            [-1.0,  0.2],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr >= 0
        assert nr <= nr_desired

    def test_siso_system(self):
        """
        Test single-input single-output system.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n, m, p = 4, 1, 1
        nr_desired = 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [0.5], [0.25], [0.125]], order='F', dtype=float)
        c = np.array([[1.0, 0.5, 0.25, 0.125]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert ar.shape[0] >= nr
        assert br.shape == (n, m)
        assert cr.shape == (p, n)
        assert dr.shape == (p, m)

    def test_mimo_system(self):
        """
        Test multi-input multi-output system.

        Random seed: 777 (for reproducibility)
        """
        np.random.seed(777)
        n, m, p = 4, 3, 2
        nr_desired = 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float) * 0.5
        c = np.ones((p, n), order='F', dtype=float) * 0.3
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr <= nr_desired

    def test_zero_dimensions(self):
        """
        Test quick return for zero dimensions.
        """
        a = np.zeros((0, 0), order='F', dtype=float)
        b = np.zeros((0, 0), order='F', dtype=float)
        c = np.zeros((0, 0), order='F', dtype=float)
        d = np.zeros((0, 0), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'F', 0, 0, 0, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr == 0


class TestAB09CDErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test error handling for invalid DICO parameter."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09cd('X', 'N', 'F', n, m, p, 1, a, b, c, d, 0.0, 0.0)

    def test_invalid_equil(self):
        """Test error handling for invalid EQUIL parameter."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09cd('C', 'X', 'F', n, m, p, 1, a, b, c, d, 0.0, 0.0)

    def test_invalid_ordsel(self):
        """Test error handling for invalid ORDSEL parameter."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09cd('C', 'N', 'X', n, m, p, 1, a, b, c, d, 0.0, 0.0)

    def test_unstable_continuous_system(self):
        """
        Test error when continuous-time system is unstable.

        info should be 2 for unstable system (Schur reduction succeeds, then stability check fails).
        """
        n, m, p = 2, 1, 1
        nr_desired = 1

        a = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'C', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 2, "Expected info=2 for unstable continuous-time system"

    def test_divergent_discrete_system(self):
        """
        Test error when discrete-time system is not convergent.

        info should be 2 for non-convergent system.
        """
        n, m, p = 2, 1, 1
        nr_desired = 1

        a = np.array([
            [2.0, 0.0],
            [0.0, 1.5]
        ], order='F', dtype=float)

        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, nmin, iwarn, info = ab09cd(
            'D', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 2, "Expected info=2 for non-convergent discrete-time system"

    def test_negative_n(self):
        """Test error handling for negative n."""
        a = np.array([[1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09cd('C', 'N', 'F', -1, 1, 1, 0, a, b, c, d, 0.0, 0.0)

    def test_nr_greater_than_n(self):
        """Test error handling when nr > n with fixed order."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09cd('C', 'N', 'F', n, m, p, 5, a, b, c, d, 0.0, 0.0)

    def test_tol2_greater_than_tol1(self):
        """Test error handling when TOL2 > TOL1."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09cd('C', 'N', 'A', n, m, p, 0, a, b, c, d, 0.01, 0.1)
