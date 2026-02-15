"""
Tests for AB09MD - Balance & Truncate model reduction for ALPHA-stable part

AB09MD computes a reduced order model (Ar,Br,Cr) for a possibly unstable
state-space representation (A,B,C) by using either the square-root or the
balancing-free square-root Balance & Truncate (B&T) model reduction method
for the ALPHA-stable part of the system.

Key features:
- Decomposes system into ALPHA-stable and ALPHA-unstable parts
- Reduces only the ALPHA-stable part
- Preserves the ALPHA-unstable part unchanged
- ALPHA boundary: Re(lambda) < ALPHA for continuous, |lambda| < ALPHA for discrete

Error bound: HSV(NR+NS-N) <= ||G-Gr||_inf <= 2*sum(HSV(NR+NS-N+1:NS))
"""

import numpy as np
import pytest
from ctrlsys import ab09md


class TestAB09MDHTMLExample:
    """Tests from SLICOT HTML documentation example."""

    def test_continuous_time_example(self):
        """
        Test AB09MD with continuous-time example from HTML documentation.

        System: 7th order with 2 inputs, 3 outputs
        ALPHA = -0.6 (stability boundary for continuous-time)
        Expected: NR = 5
        """
        n, m, p = 7, 2, 3
        nr = 0
        alpha = -0.6
        tol = 0.1

        a = np.array([
            [-0.04165, 0.0000, 4.9200, -4.9200, 0.0000, 0.0000, 0.0000],
            [-5.2100, -12.500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 3.3300, -3.3300, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5450, 0.0000, 0.0000, 0.0000, -0.5450, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 4.9200, -0.04165, 0.0000, 4.9200],
            [0.0000, 0.0000, 0.0000, 0.0000, -5.2100, -12.500, 0.0000],
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

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'N', 'N', 'A', n, m, p, nr, alpha, a, b, c, tol
        )

        assert info == 0
        assert nr_out == 5

        hsv_expected = np.array([1.9178, 0.8621, 0.7666, 0.0336, 0.0246])
        np.testing.assert_allclose(hsv[:ns], hsv_expected, rtol=1e-3, atol=1e-4)

        ar_expected = np.array([
            [-0.5181, -1.1084, 0.0000, 0.0000, 0.0000],
            [8.8157, -0.5181, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.5124, 0.0000, 1.7910],
            [0.0000, 0.0000, 0.0000, -1.4460, 0.0000],
            [0.0000, 0.0000, -4.2167, 0.0000, -2.9900]
        ], order='F', dtype=float)
        np.testing.assert_allclose(ar[:nr_out, :nr_out], ar_expected, rtol=1e-3, atol=1e-4)

        br_expected = np.array([
            [-1.2837, 1.2837],
            [-0.7522, 0.7522],
            [-0.7447, -0.7447],
            [1.9275, -1.9275],
            [-3.6872, -3.6872]
        ], order='F', dtype=float)
        np.testing.assert_allclose(br[:nr_out, :m], br_expected, rtol=1e-3, atol=1e-4)

        cr_expected = np.array([
            [-0.1380, -0.6445, -0.6582, -0.5771, 0.2222],
            [0.6246, 0.0196, 0.0000, 0.4131, 0.0000],
            [0.1380, 0.6445, -0.6582, 0.5771, 0.2222]
        ], order='F', dtype=float)
        np.testing.assert_allclose(cr[:p, :nr_out], cr_expected, rtol=1e-3, atol=1e-4)


class TestAB09MDBasic:
    """Basic functionality tests."""

    def test_continuous_stable_system(self):
        """
        Test continuous-time system that is fully ALPHA-stable.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 1, 1
        nr = 2
        alpha = 0.0

        a = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0, -4.0]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'N', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        assert ns == n
        assert nr_out == nr
        assert len(hsv) >= ns
        for i in range(ns - 1):
            assert hsv[i] >= hsv[i + 1]

    def test_discrete_stable_system(self):
        """
        Test discrete-time system that is fully ALPHA-stable.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 3, 1, 1
        nr = 2
        alpha = 1.0

        a = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.2]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'D', 'B', 'N', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        assert ns == n
        assert nr_out == nr

    def test_with_equilibration(self):
        """
        Test with equilibration (scaling) enabled.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 3, 1, 1
        nr = 2
        alpha = 0.0

        a = np.array([
            [-1e3, 0.0, 0.0],
            [0.0, -1e-3, 0.0],
            [0.0, 0.0, -1.0]
        ], order='F', dtype=float)

        b = np.array([[1e3], [1e-3], [1.0]], order='F', dtype=float)
        c = np.array([[1e-3, 1e3, 1.0]], order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'S', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0


class TestAB09MDMathematicalProperties:
    """Tests for mathematical property validation."""

    def test_hsv_decreasing(self):
        """
        Verify Hankel singular values are in decreasing order.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 5, 2, 2
        nr = 3
        alpha = 0.0

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'N', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        for i in range(ns - 1):
            assert hsv[i] >= hsv[i + 1], f"HSV not decreasing at index {i}"

    def test_unstable_part_preserved(self):
        """
        Verify that ALPHA-unstable part eigenvalues are preserved.

        System with one unstable eigenvalue (Re > alpha) should preserve it.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m, p = 4, 1, 1
        nr = 3
        alpha = -0.5

        a = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, -3.0]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'N', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        nu = n - ns
        assert nu >= 1

        eigs = np.linalg.eigvals(ar[:nr_out, :nr_out])
        unstable_eigs = [e.real for e in eigs if e.real >= alpha]
        assert len(unstable_eigs) >= 1


class TestAB09MDEdgeCases:
    """Edge case tests."""

    def test_n1_system(self):
        """
        Test with minimal 1st order system.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m, p = 1, 1, 1
        nr = 1
        alpha = 0.0

        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'N', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        assert ns == 1
        assert nr_out == 1

    def test_automatic_order_selection(self):
        """
        Test automatic order selection (ORDSEL='A').

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n, m, p = 4, 1, 1
        nr = 0
        alpha = 0.0

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [0.1], [0.01], [0.001]], order='F', dtype=float)
        c = np.array([[1.0, 0.1, 0.01, 0.001]], order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'N', 'A', n, m, p, nr, alpha, a, b, c, 1e-4
        )

        assert info == 0
        assert nr_out >= 0
        assert nr_out <= n

    def test_balancing_free_method(self):
        """
        Test balancing-free square-root method (JOB='N').

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n, m, p = 4, 1, 1
        nr = 2
        alpha = 0.0

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'N', 'N', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == nr

    def test_fully_unstable_system(self):
        """
        Test system that is fully ALPHA-unstable.

        All eigenvalues are above ALPHA boundary, so NR = NU.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n, m, p = 3, 1, 1
        nr = 0
        alpha = -10.0

        a = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, -3.0]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'N', 'A', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        assert ns == 0
        assert nr_out == n

    def test_mimo_system(self):
        """
        Test MIMO system reduction.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n, m, p = 6, 2, 3
        nr = 4
        alpha = 0.0

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'N', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == nr
        assert br.shape[0] >= nr_out
        assert br.shape[1] >= m
        assert cr.shape[0] >= p
        assert cr.shape[1] >= nr_out


class TestAB09MDErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09md('X', 'B', 'N', 'F', n, m, p, nr, 0.0, a, b, c, 0.0)

    def test_invalid_job(self):
        """Test error for invalid JOB parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09md('C', 'X', 'N', 'F', n, m, p, nr, 0.0, a, b, c, 0.0)

    def test_invalid_equil(self):
        """Test error for invalid EQUIL parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09md('C', 'B', 'X', 'F', n, m, p, nr, 0.0, a, b, c, 0.0)

    def test_invalid_ordsel(self):
        """Test error for invalid ORDSEL parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09md('C', 'B', 'N', 'X', n, m, p, nr, 0.0, a, b, c, 0.0)

    def test_negative_n(self):
        """Test error for negative n."""
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((2, 1), order='F', dtype=float)
        c = np.ones((1, 2), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09md('C', 'B', 'N', 'F', -1, 1, 1, 1, 0.0, a, b, c, 0.0)

    def test_invalid_alpha_continuous(self):
        """Test error for invalid ALPHA for continuous-time (ALPHA > 0)."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09md('C', 'B', 'N', 'F', n, m, p, nr, 1.0, a, b, c, 0.0)

    def test_invalid_alpha_discrete(self):
        """Test error for invalid ALPHA for discrete-time (ALPHA < 0 or > 1)."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.array([[0.5, 0.0], [0.0, 0.3]], order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09md('D', 'B', 'N', 'F', n, m, p, nr, -0.1, a, b, c, 0.0)

        with pytest.raises((ValueError, RuntimeError)):
            ab09md('D', 'B', 'N', 'F', n, m, p, nr, 1.5, a, b, c, 0.0)


class TestAB09MDWarnings:
    """Warning indicator tests."""

    def test_iwarn_nr_greater_than_minimal(self):
        """
        Test iwarn=1 when requested NR > NSMIN.

        NSMIN = NU + minimal realization order of stable part.
        """
        n, m, p = 4, 1, 1
        nr = 4
        alpha = 0.0

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [1e-16], [1e-16], [1e-16]], order='F', dtype=float)
        c = np.array([[1.0, 1e-16, 1e-16, 1e-16]], order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'N', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        if nr_out < nr:
            assert iwarn == 1

    def test_iwarn_nr_less_than_unstable_part(self):
        """
        Test iwarn=2 when requested NR < NU (unstable part order).

        In this case NR is set to NU.
        """
        n, m, p = 4, 1, 1
        nr = 0
        alpha = -0.5

        a = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -2.0]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(
            'C', 'B', 'N', 'F', n, m, p, nr, alpha, a, b, c, 0.0
        )

        assert info == 0
        nu = n - ns
        if nr < nu and nr_out == nu:
            assert iwarn == 2
