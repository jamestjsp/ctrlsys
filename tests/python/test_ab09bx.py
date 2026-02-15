"""
Tests for AB09BX - Singular Perturbation Approximation model reduction

AB09BX computes a reduced order model (Ar,Br,Cr,Dr) for a stable original
state-space representation (A,B,C,D) using either the square-root or
balancing-free square-root Singular Perturbation Approximation (SPA) method.

The state dynamics matrix A must be in real Schur canonical form.

Method:
    Am = TI * A * T,  Bm = TI * B,  Cm = C * T
    where T and TI are truncation matrices computed from Hankel singular values.

Parameters:
    DICO: 'C' continuous-time, 'D' discrete-time
    JOB: 'B' square-root SPA (balanced), 'N' balancing-free SPA
    ORDSEL: 'F' fixed order, 'A' automatic order selection

Error Codes:
    info=0: success
    info=1: A is not stable (continuous) or not convergent (discrete)
    info=2: computation of Hankel singular values failed
"""

import numpy as np
import pytest
from ctrlsys import ab09bx


class TestAB09BXBasic:
    """Basic functionality tests."""

    def test_continuous_time_balanced_spa(self):
        """
        Test continuous-time square-root SPA model reduction.

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

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert iwarn == 0 or iwarn == 1
        assert nr <= nr_desired
        assert nr >= 0

        assert hsv.shape[0] == n
        assert all(hsv[i] >= hsv[i+1] for i in range(n-1) if hsv[i+1] > 0)

        if nr > 0:
            ar_reduced = ar[:nr, :nr]
            eigs = np.linalg.eigvals(ar_reduced)
            assert all(np.real(eigs) < 0), "Reduced system must be stable"

    def test_continuous_time_balancing_free_spa(self):
        """
        Test continuous-time balancing-free SPA model reduction.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 4, 1, 1
        nr_desired = 2

        a = np.array([
            [-1.0,  0.5,  0.0,  0.0],
            [ 0.0, -2.0,  0.0,  0.0],
            [ 0.0,  0.0, -3.0,  0.1],
            [ 0.0,  0.0,  0.0, -4.0]
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

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr <= nr_desired
        assert nr >= 0

        assert len(hsv) == n

    def test_discrete_time_balanced_spa(self):
        """
        Test discrete-time square-root SPA model reduction.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 4, 1, 1
        nr_desired = 2

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

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'D', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr <= nr_desired
        assert nr >= 0

        if nr > 0:
            ar_reduced = ar[:nr, :nr]
            eigs = np.linalg.eigvals(ar_reduced)
            assert all(np.abs(eigs) < 1.0), "Reduced discrete system must be convergent"


class TestAB09BXMathematicalProperties:
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

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        for i in range(n - 1):
            if hsv[i+1] > 1e-15:
                assert hsv[i] >= hsv[i+1] - 1e-14, \
                    f"HSV not decreasing: hsv[{i}]={hsv[i]} < hsv[{i+1}]={hsv[i+1]}"

    def test_truncation_matrix_properties(self):
        """
        Verify truncation matrices T and TI satisfy TI * A * T relationship.

        For balanced reduction (JOB='B'), T and TI should be biorthogonal.
        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m, p = 4, 1, 1
        nr_desired = 2

        a = np.array([
            [-1.0,  0.3,  0.0,  0.0],
            [ 0.0, -2.0,  0.0,  0.0],
            [ 0.0,  0.0, -3.0,  0.2],
            [ 0.0,  0.0,  0.0, -4.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5], [0.3], [0.1]], order='F', dtype=float)
        c = np.array([[1.0, 0.5, 0.3, 0.1]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        if nr > 0:
            t_nr = t[:n, :nr]
            ti_nr = ti[:nr, :n]

            ti_t = ti_nr @ t_nr
            expected = np.eye(nr)
            np.testing.assert_allclose(ti_t, expected, rtol=1e-10, atol=1e-10)

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

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.01, 0.0
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

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        if nr > 0:
            ar_reduced = ar[:nr, :nr]
            eigs = np.linalg.eigvals(ar_reduced)
            assert all(np.real(eigs) < 1e-10), \
                f"Reduced system not stable: eigenvalues = {eigs}"


class TestAB09BXEdgeCases:
    """Edge case tests."""

    def test_minimal_system(self):
        """
        Test with a minimal 2x2 system.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n, m, p = 2, 1, 1
        nr_desired = 1

        a = np.array([
            [-1.0,  0.2],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr >= 0
        assert nr <= nr_desired

    def test_siso_system(self):
        """
        Test single-input single-output system.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n, m, p = 4, 1, 1
        nr_desired = 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [0.5], [0.25], [0.125]], order='F', dtype=float)
        c = np.array([[1.0, 0.5, 0.25, 0.125]], order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert ar.shape[0] >= nr
        assert br.shape == (n, m)
        assert cr.shape == (p, n)
        assert dr.shape == (p, m)

    def test_mimo_system(self):
        """
        Test multi-input multi-output system.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n, m, p = 4, 3, 2
        nr_desired = 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float) * 0.5
        c = np.ones((p, n), order='F', dtype=float) * 0.3
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr <= nr_desired


class TestAB09BXErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test error handling for invalid DICO parameter."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09bx('X', 'B', 'F', n, m, p, 1, a, b, c, d, 0.0, 0.0)

    def test_invalid_job(self):
        """Test error handling for invalid JOB parameter."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09bx('C', 'X', 'F', n, m, p, 1, a, b, c, d, 0.0, 0.0)

    def test_invalid_ordsel(self):
        """Test error handling for invalid ORDSEL parameter."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09bx('C', 'B', 'X', n, m, p, 1, a, b, c, d, 0.0, 0.0)

    def test_unstable_continuous_system(self):
        """
        Test error when continuous-time system is unstable.

        info should be 1 for unstable system.
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

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'C', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 1, "Expected info=1 for unstable continuous-time system"

    def test_divergent_discrete_system(self):
        """
        Test error when discrete-time system is not convergent.

        info should be 1 for non-convergent system.
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

        ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info = ab09bx(
            'D', 'B', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 1, "Expected info=1 for non-convergent discrete-time system"

    def test_negative_n(self):
        """Test error handling for negative n."""
        a = np.array([[1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09bx('C', 'B', 'F', -1, 1, 1, 0, a, b, c, d, 0.0, 0.0)

    def test_nr_greater_than_n(self):
        """Test error handling when nr > n with fixed order."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09bx('C', 'B', 'F', n, m, p, 5, a, b, c, d, 0.0, 0.0)

    def test_tol2_greater_than_tol1(self):
        """Test error handling when TOL2 > TOL1."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09bx('C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.01, 0.1)
