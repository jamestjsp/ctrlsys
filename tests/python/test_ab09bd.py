"""
Tests for AB09BD - Singular Perturbation Approximation model reduction.
"""

import numpy as np
import pytest
from slicot import ab09bd


class TestAB09BDBasic:
    """Tests based on SLICOT HTML documentation example."""

    def test_continuous_time_spa_reduction(self):
        """
        Test SPA model reduction for continuous-time system from HTML doc.

        System: 7th order, 2 inputs, 3 outputs
        Method: Balancing-free SPA (JOB='N')
        Expected reduction: Order 7 -> Order 5
        """
        n, m, p = 7, 2, 3

        # Input matrices from HTML doc (read row-wise per Fortran READ pattern)
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
            [12.500,   0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000, 12.500],
            [ 0.0000,  0.0000]
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

        # Call AB09BD with balancing-free SPA, auto order selection
        # DICO='C', JOB='N', EQUIL='N', ORDSEL='A'
        tol1 = 0.1
        tol2 = 1e-14
        nr_in = 0  # ignored for ORDSEL='A'

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'N', 'N', 'A', n, m, p, nr_in, a, b, c, d, tol1, tol2
        )

        assert info == 0, f"AB09BD failed with info={info}"
        assert iwarn == 0, f"Unexpected warning: iwarn={iwarn}"

        # Expected reduced order from HTML doc
        assert nr == 5, f"Expected reduced order 5, got {nr}"

        # Expected HSV from HTML doc
        hsv_expected = np.array([2.5139, 2.0846, 1.9178, 0.7666, 0.5473, 0.0253, 0.0246])
        np.testing.assert_allclose(hsv, hsv_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced state matrix Ar (5x5) from HTML doc
        ar_expected = np.array([
            [ 1.3960,  5.1248,  0.0000,  0.0000,  4.4331],
            [-4.1411, -3.8605,  0.0000,  0.0000, -0.6738],
            [ 0.0000,  0.0000,  0.5847,  1.9230,  0.0000],
            [ 0.0000,  0.0000, -4.3823, -3.2922,  0.0000],
            [ 1.3261,  1.7851,  0.0000,  0.0000, -0.2249]
        ], order='F', dtype=float)
        np.testing.assert_allclose(ar[:nr, :nr], ar_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced input matrix Br (5x2) from HTML doc
        br_expected = np.array([
            [-0.2901,  0.2901],
            [-3.4004,  3.4004],
            [-0.6379, -0.6379],
            [-3.9315, -3.9315],
            [ 1.9813, -1.9813]
        ], order='F', dtype=float)
        np.testing.assert_allclose(br[:nr, :], br_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced output matrix Cr (3x5) from HTML doc
        cr_expected = np.array([
            [-0.6570,  0.2053, -0.6416,  0.2526, -0.0364],
            [ 0.1094,  0.4875,  0.0000,  0.0000,  0.8641],
            [ 0.6570, -0.2053, -0.6416,  0.2526,  0.0364]
        ], order='F', dtype=float)
        np.testing.assert_allclose(cr[:, :nr], cr_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced feedthrough matrix Dr (3x2) from HTML doc
        dr_expected = np.array([
            [ 0.0498, -0.0007],
            [ 0.0010, -0.0010],
            [-0.0007,  0.0498]
        ], order='F', dtype=float)
        np.testing.assert_allclose(dr, dr_expected, rtol=1e-3, atol=1e-4)


class TestAB09BDFixedOrder:
    """Tests for fixed order reduction."""

    def test_fixed_order_reduction(self):
        """
        Test fixed order reduction (ORDSEL='F').

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 1, 1
        nr_desired = 2

        # Create a stable continuous-time system
        # Use diagonal A with negative eigenvalues for stability
        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0, f"AB09BD failed with info={info}"
        # nr may be less than nr_desired if minimal realization is smaller
        assert 0 <= nr <= nr_desired, f"Unexpected nr={nr}"
        # HSV should be positive and decreasing
        assert all(hsv[i] >= hsv[i+1] for i in range(n-1)), "HSV not decreasing"
        assert all(h >= 0 for h in hsv), "HSV has negative values"


class TestAB09BDDiscreteTime:
    """Tests for discrete-time systems."""

    def test_discrete_time_system(self):
        """
        Test SPA reduction for discrete-time system.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 4, 1, 1

        # Create a convergent discrete-time system (spectral radius < 1)
        # Use diagonal A with eigenvalues inside unit circle
        a = np.diag([0.5, 0.3, -0.2, 0.1]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'D', 'B', 'N', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0, f"AB09BD failed with info={info}"
        assert nr >= 0, f"Invalid reduced order nr={nr}"
        # HSV should be positive and decreasing
        assert all(hsv[i] >= hsv[i+1] for i in range(n-1)), "HSV not decreasing"


class TestAB09BDEquilibration:
    """Tests for equilibration option."""

    def test_with_equilibration(self):
        """
        Test SPA reduction with system equilibration (EQUIL='S').

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 4, 2, 2

        # Create a stable system with poor scaling
        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        # Scale B badly
        b = np.random.randn(n, m).astype(float, order='F') * 1000.0
        c = np.random.randn(p, n).astype(float, order='F') * 0.001
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'S', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0, f"AB09BD failed with info={info}"
        assert nr >= 0, f"Invalid reduced order nr={nr}"


class TestAB09BDEdgeCases:
    """Edge case tests."""

    def test_zero_dimensions(self):
        """Test with n=0 (quick return case)."""
        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 1), order='F', dtype=float)
        c = np.zeros((1, 1), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'N', 'A', 0, 1, 1, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr == 0

    def test_zero_inputs(self):
        """Test with m=0 (no inputs)."""
        n, m, p = 2, 0, 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.zeros((n, 1), order='F', dtype=float)  # placeholder
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, 1), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr == 0

    def test_zero_outputs(self):
        """Test with p=0 (no outputs)."""
        n, m, p = 2, 1, 0
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.zeros((1, n), order='F', dtype=float)  # placeholder
        d = np.zeros((1, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr == 0


class TestAB09BDErrorHandling:
    """Error handling tests."""

    def test_unstable_continuous_system(self):
        """Test that unstable continuous system returns info=2."""
        n, m, p = 2, 1, 1

        # Create unstable system (eigenvalue > 0)
        a = np.diag([1.0, -1.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 2, f"Expected info=2 for unstable system, got {info}"

    def test_nonconvergent_discrete_system(self):
        """Test that non-convergent discrete system returns info=2."""
        n, m, p = 2, 1, 1

        # Create non-convergent discrete system (eigenvalue > 1)
        a = np.diag([2.0, 0.5]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'D', 'B', 'N', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 2, f"Expected info=2 for non-convergent system, got {info}"


class TestAB09BDWarnings:
    """Warning indicator tests."""

    def test_fixed_order_exceeds_minimal(self):
        """Test warning when requested order exceeds minimal realization."""
        np.random.seed(789)
        n, m, p = 4, 1, 1
        nr_desired = 4  # Request full order

        # Create system where some states may be uncontrollable/unobservable
        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [0.0], [0.0]], order='F', dtype=float)  # last 2 uncontrollable
        c = np.array([[1.0, 1.0, 0.0, 0.0]], order='F', dtype=float)  # last 2 unobservable
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'N', 'F', n, m, p, nr_desired, a, b, c, d, 0.0, 0.0
        )

        assert info == 0, f"AB09BD failed with info={info}"
        # If minimal realization is smaller than requested, iwarn=1
        if nr < nr_desired:
            assert iwarn == 1, f"Expected iwarn=1 when nr < nr_desired, got {iwarn}"


class TestAB09BDMIMO:
    """MIMO model reduction tests."""

    def test_mimo_reduction_with_hsv_ordering(self):
        """
        MIMO: 6-state, 2-input, 3-output system reduction.
        Verify HSV ordering and reduced system stability.
        """
        n, m, p = 6, 2, 3

        a = np.array([
            [-1.0,  0.5,  0.0,  0.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.3,  0.0,  0.0,  0.0],
            [ 0.0,  0.0, -3.0,  0.1,  0.0,  0.0],
            [ 0.0,  0.0,  0.0, -4.0,  0.2,  0.0],
            [ 0.0,  0.0,  0.0,  0.0, -5.0,  0.1],
            [ 0.0,  0.0,  0.0,  0.0,  0.0, -6.0]
        ], dtype=float, order='F')

        np.random.seed(100)
        b = np.asfortranarray(np.random.randn(n, m))
        c = np.asfortranarray(np.random.randn(p, n))
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr <= n
        assert nr > 0

        for i in range(n - 1):
            assert hsv[i] >= hsv[i + 1] - 1e-14, \
                f"HSV not decreasing: HSV[{i}]={hsv[i]} < HSV[{i+1}]={hsv[i+1]}"

        eigs = np.linalg.eigvals(ar[:nr, :nr])
        assert all(e.real < 0 for e in eigs), f"Reduced system unstable: {eigs}"

        assert br[:nr, :].shape == (nr, m)
        assert cr[:, :nr].shape == (p, nr)


class TestAB09BDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_hsv_ordering(self):
        """
        Verify Hankel singular values are ordered decreasingly.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        n, m, p = 6, 2, 2

        a = np.diag([-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        # HSV must be non-negative and decreasing
        for i in range(n):
            assert hsv[i] >= 0, f"HSV[{i}] = {hsv[i]} is negative"
        for i in range(n - 1):
            assert hsv[i] >= hsv[i + 1], f"HSV not decreasing: HSV[{i}]={hsv[i]} < HSV[{i+1}]={hsv[i+1]}"

    def test_reduced_system_stability_continuous(self):
        """
        Verify reduced continuous-time system is stable.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m, p = 5, 1, 1

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'C', 'B', 'N', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        if nr > 0:
            # All eigenvalues of reduced A must have negative real part
            eigs = np.linalg.eigvals(ar[:nr, :nr])
            assert all(e.real < 0 for e in eigs), f"Reduced system unstable: eigenvalues={eigs}"

    def test_reduced_system_stability_discrete(self):
        """
        Verify reduced discrete-time system is convergent.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m, p = 5, 1, 1

        a = np.diag([0.5, 0.3, -0.2, 0.1, -0.4]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ar, br, cr, dr, nr, hsv, iwarn, info = ab09bd(
            'D', 'B', 'N', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        if nr > 0:
            # All eigenvalues of reduced A must have magnitude < 1
            eigs = np.linalg.eigvals(ar[:nr, :nr])
            assert all(abs(e) < 1.0 for e in eigs), f"Reduced system not convergent: |eigenvalues|={[abs(e) for e in eigs]}"
