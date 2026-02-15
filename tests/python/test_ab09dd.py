"""
Tests for AB09DD - Feedthrough update for Singular Perturbation Approximation

AB09DD computes a reduced order model using singular perturbation approximation
formulas. Given system (A, B, C, D) and reduced order nr, it computes the
residualized reduced order system (Ar, Br, Cr, Dr).

Method:
    Ar = A11 + A12 * (g*I - A22)^{-1} * A21
    Br = B1  + A12 * (g*I - A22)^{-1} * B2
    Cr = C1  + C2  * (g*I - A22)^{-1} * A21
    Dr = D   + C2  * (g*I - A22)^{-1} * B2

where g = 0 for continuous-time (DICO='C') and g = 1 for discrete-time (DICO='D').
"""

import numpy as np
import pytest
from ctrlsys import ab09dd


class TestAB09DDBasic:
    """Basic functionality tests from SLICOT HTML documentation example."""

    def test_continuous_time_reduction(self):
        """
        Test continuous-time model reduction from SLICOT documentation.

        Example: 7th order system reduced to 5th order.
        """
        n, m, p, nr = 7, 2, 3, 5

        a = np.array([
            [-0.04165, 4.9200, -4.9200,  0.0,     0.0,      0.0,      0.0],
            [ 0.0,    -3.3300,  0.0,     0.0,     0.0,      3.3300,   0.0],
            [ 0.5450,  0.0,     0.0,    -0.5450,  0.0,      0.0,      0.0],
            [ 0.0,     0.0,     4.9200, -0.04165, 4.9200,   0.0,      0.0],
            [ 0.0,     0.0,     0.0,     0.0,    -3.3300,   0.0,      3.3300],
            [-5.2100,  0.0,     0.0,     0.0,     0.0,    -12.5000,   0.0],
            [ 0.0,     0.0,     0.0,    -5.2100,  0.0,      0.0,    -12.5000]
        ], order='F', dtype=float)

        b = np.array([
            [ 0.0,     0.0],
            [ 0.0,     0.0],
            [ 0.0,     0.0],
            [ 0.0,     0.0],
            [ 0.0,     0.0],
            [12.5000,  0.0],
            [ 0.0,    12.5000]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ], order='F', dtype=float)

        ar, br, cr, dr, rcond, info = ab09dd('C', n, m, p, nr, a, b, c, d)

        assert info == 0
        assert rcond == pytest.approx(1.0, rel=1e-3)

        ar_expected = np.array([
            [-0.0416,  4.9200, -4.9200,  0.0000,  0.0000],
            [-1.3879, -3.3300,  0.0000,  0.0000,  0.0000],
            [ 0.5450,  0.0000,  0.0000, -0.5450,  0.0000],
            [ 0.0000,  0.0000,  4.9200, -0.0416,  4.9200],
            [ 0.0000,  0.0000,  0.0000, -1.3879, -3.3300]
        ], order='F', dtype=float)

        br_expected = np.array([
            [0.0000,  0.0000],
            [3.3300,  0.0000],
            [0.0000,  0.0000],
            [0.0000,  0.0000],
            [0.0000,  3.3300]
        ], order='F', dtype=float)

        cr_expected = np.array([
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000]
        ], order='F', dtype=float)

        dr_expected = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ], order='F', dtype=float)

        np.testing.assert_allclose(ar[:nr, :nr], ar_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(br[:nr, :m], br_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(cr[:p, :nr], cr_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(dr, dr_expected, rtol=1e-3, atol=1e-4)


class TestAB09DDMathematicalProperties:
    """Tests for mathematical property validation."""

    def test_no_reduction_identity(self):
        """
        When nr == n (no reduction), output equals input.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p, nr = 3, 2, 2, 3

        a = np.array([
            [-1.0, 0.2, 0.1],
            [ 0.3,-2.0, 0.4],
            [ 0.1, 0.2,-3.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.3]
        ], order='F', dtype=float)

        d = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ], order='F', dtype=float)

        a_orig = a.copy()
        b_orig = b.copy()
        c_orig = c.copy()
        d_orig = d.copy()

        ar, br, cr, dr, rcond, info = ab09dd('C', n, m, p, nr, a, b, c, d)

        assert info == 0
        assert rcond == pytest.approx(1.0, rel=1e-10)

        np.testing.assert_allclose(ar[:n, :n], a_orig, rtol=1e-14)
        np.testing.assert_allclose(br[:n, :m], b_orig, rtol=1e-14)
        np.testing.assert_allclose(cr[:p, :n], c_orig, rtol=1e-14)
        np.testing.assert_allclose(dr, d_orig, rtol=1e-14)

    def test_discrete_time_reduction(self):
        """
        Test discrete-time model reduction.

        For discrete-time, g = 1, so (g*I - A22) = (I - A22).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p, nr = 4, 1, 1, 2

        a = np.array([
            [ 0.5,  0.1,  0.1, 0.0],
            [ 0.2,  0.6,  0.0, 0.1],
            [ 0.1,  0.0,  0.3, 0.1],
            [ 0.0,  0.1,  0.1, 0.2]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.0],
            [0.5],
            [0.2]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.0, 0.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        ar, br, cr, dr, rcond, info = ab09dd('D', n, m, p, nr, a, b, c, d)

        assert info == 0
        assert rcond > 0

        assert ar.shape[0] >= nr
        assert ar.shape[1] >= nr
        assert br.shape[0] >= nr
        assert cr.shape[1] >= nr

    def test_residualization_formula_continuous(self):
        """
        Verify residualization formula directly for continuous-time.

        Ar = A11 + A12 * (-A22)^{-1} * A21
        Br = B1  + A12 * (-A22)^{-1} * B2
        Cr = C1  + C2  * (-A22)^{-1} * A21
        Dr = D   + C2  * (-A22)^{-1} * B2

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p, nr = 4, 2, 2, 2
        ns = n - nr

        a11 = np.array([[-1.0, 0.1], [0.2, -2.0]], order='F', dtype=float)
        a12 = np.array([[0.3, 0.4], [0.5, 0.6]], order='F', dtype=float)
        a21 = np.array([[0.1, 0.2], [0.3, 0.4]], order='F', dtype=float)
        a22 = np.array([[-3.0, 0.1], [0.2, -4.0]], order='F', dtype=float)

        a = np.block([[a11, a12], [a21, a22]]).astype(float, order='F')

        b1 = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        b2 = np.array([[0.5, 0.3], [0.2, 0.4]], order='F', dtype=float)
        b = np.vstack([b1, b2]).astype(float, order='F')

        c1 = np.array([[1.0, 0.5], [0.3, 1.0]], order='F', dtype=float)
        c2 = np.array([[0.2, 0.1], [0.1, 0.2]], order='F', dtype=float)
        c = np.hstack([c1, c2]).astype(float, order='F')

        d = np.array([[0.0, 0.0], [0.0, 0.0]], order='F', dtype=float)

        neg_a22_inv = np.linalg.inv(-a22)
        ar_expected = a11 + a12 @ neg_a22_inv @ a21
        br_expected = b1 + a12 @ neg_a22_inv @ b2
        cr_expected = c1 + c2 @ neg_a22_inv @ a21
        dr_expected = d + c2 @ neg_a22_inv @ b2

        ar, br, cr, dr, rcond, info = ab09dd('C', n, m, p, nr, a, b, c, d)

        assert info == 0

        np.testing.assert_allclose(ar[:nr, :nr], ar_expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(br[:nr, :m], br_expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(cr[:p, :nr], cr_expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(dr, dr_expected, rtol=1e-13, atol=1e-14)


class TestAB09DDEdgeCases:
    """Edge case tests."""

    def test_full_reduction_to_zero(self):
        """
        Test reduction to nr=0 (steady-state gain).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p, nr = 2, 1, 1, 0

        a = np.array([
            [-1.0, 0.0],
            [ 0.0,-2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        a_orig = a.copy()
        b_orig = b.copy()
        c_orig = c.copy()
        d_orig = d.copy()

        ar, br, cr, dr, rcond, info = ab09dd('C', n, m, p, nr, a, b, c, d)

        assert info == 0

        neg_a22_inv = np.linalg.inv(-a_orig)
        dc_gain = d_orig + c_orig @ neg_a22_inv @ b_orig

        np.testing.assert_allclose(dr, dc_gain, rtol=1e-13, atol=1e-14)

    def test_single_input_single_output(self):
        """
        Test SISO system reduction.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m, p, nr = 3, 1, 1, 2

        a = np.array([
            [-1.0, 0.5, 0.2],
            [ 0.3,-2.0, 0.1],
            [ 0.1, 0.2,-3.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5],
            [0.2]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.3, 0.1]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        ar, br, cr, dr, rcond, info = ab09dd('C', n, m, p, nr, a, b, c, d)

        assert info == 0
        assert ar.shape[0] >= nr
        assert ar.shape[1] >= nr
        assert br.shape == (n, m)
        assert cr.shape == (p, n)
        assert dr.shape == (p, m)


class TestAB09DDErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test error handling for invalid DICO parameter."""
        n, m, p, nr = 2, 1, 1, 1
        a = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09dd('X', n, m, p, nr, a, b, c, d)

    def test_singular_matrix(self):
        """
        Test error when A22 - g*I is singular.

        For continuous-time (g=0), A22 is singular, so -A22 is singular.
        """
        n, m, p, nr = 3, 1, 1, 1

        a = np.array([
            [-1.0, 0.5, 0.5],
            [ 0.5, 0.0, 0.0],
            [ 0.5, 0.0, 0.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        ar, br, cr, dr, rcond, info = ab09dd('C', n, m, p, nr, a, b, c, d)

        assert info == 1 or rcond < 1e-10

    def test_negative_dimensions(self):
        """Test error handling for negative dimensions."""
        a = np.array([[1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09dd('C', -1, 1, 1, 0, a, b, c, d)

    def test_nr_greater_than_n(self):
        """Test error handling when nr > n."""
        n, m, p, nr = 2, 1, 1, 3
        a = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09dd('C', n, m, p, nr, a, b, c, d)
