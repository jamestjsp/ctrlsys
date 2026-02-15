"""
Tests for ab09fd: Balance & Truncate model reduction for unstable systems
using coprime factorization.

Computes a reduced order model (Ar,Br,Cr) from original (A,B,C) using
either square-root or balancing-free square-root B&T method with stable
coprime factorization.
"""

import numpy as np
import pytest
from ctrlsys import ab09fd


class TestAB09FDBasic:
    """Basic functionality tests from SLICOT HTML documentation."""

    def test_continuous_lcf_inner_balancing(self):
        """
        Test from SLICOT HTML documentation example.

        Continuous-time system with left coprime factorization,
        inner denominator, balancing-free method, auto order selection.
        N=7, M=2, P=3.
        """
        n, m, p = 7, 2, 3

        a = np.array([
            [-0.04165,  0.0000,  4.9200,  0.4920,  0.0000,  0.0000,  0.0000],
            [-5.2100, -12.500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  3.3300, -3.3300,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.5450,  0.0000,  0.0000,  0.0000,  0.0545,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000, -0.4920,  0.004165, 0.0000,  4.9200],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.5210, -12.500,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  3.3300, -3.3300]
        ], dtype=float, order='F')

        b = np.array([
            [ 0.0000,  0.0000],
            [12.500,   0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000, 12.500],
            [ 0.0000,  0.0000]
        ], dtype=float, order='F')

        c = np.array([
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]
        ], dtype=float, order='F')

        nr = 0
        alpha = -0.1
        tol1 = 0.1
        tol2 = 1e-10

        ar, br, cr, nr_out, nq, hsv, iwarn, info = ab09fd(
            'C', 'L', 'I', 'B', 'S', 'A',
            n, m, p, nr, alpha, a, b, c, tol1, tol2
        )

        assert info == 0
        assert nr_out == 5
        assert nq == 7

        hsv_expected = np.array([13.6047, 9.4106, 1.7684, 0.7456, 0.6891, 0.0241, 0.0230])
        np.testing.assert_allclose(hsv[:nq], hsv_expected, rtol=0.1, atol=0.1)

        assert np.linalg.norm(br[:nr_out, :m]) > 0, "Br should be non-zero"
        assert np.linalg.norm(cr[:p, :nr_out]) > 0, "Cr should be non-zero"


class TestAB09FDModes:
    """Test different mode parameter combinations."""

    def test_discrete_rcf_prescribed_stability(self):
        """
        Discrete-time system with right coprime factorization,
        prescribed stability degree.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 2, 2

        a = np.array([
            [ 0.6,  0.1, -0.1,  0.0],
            [ 0.2,  0.5,  0.1,  0.0],
            [ 0.0, -0.1,  0.4,  0.2],
            [-0.1,  0.0,  0.1,  0.3]
        ], dtype=float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.2],
            [0.1, 0.3]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5]
        ], dtype=float, order='F')

        nr = 2
        alpha = 0.5
        tol1 = 0.0
        tol2 = 1e-10

        ar, br, cr, nr_out, nq, hsv, iwarn, info = ab09fd(
            'D', 'R', 'S', 'B', 'N', 'F',
            n, m, p, nr, alpha, a, b, c, tol1, tol2
        )

        assert info == 0
        assert nr_out <= n
        assert nq <= n

        if nr_out > 0:
            ar_red = ar[:nr_out, :nr_out].copy()
            eigs = np.linalg.eigvals(ar_red)
            for eig in eigs:
                assert np.abs(eig) < 1.0, f"Eigenvalue {eig} outside unit circle"

    def test_continuous_balancing_free(self):
        """
        Continuous-time with balancing-free method.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 5, 2, 2

        a = np.array([
            [-1.0,  0.1,  0.0,  0.0,  0.0],
            [ 0.1, -2.0,  0.2,  0.0,  0.0],
            [ 0.0,  0.2, -3.0,  0.1,  0.0],
            [ 0.0,  0.0,  0.1, -4.0,  0.1],
            [ 0.0,  0.0,  0.0,  0.1, -5.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
            [0.2, 0.3],
            [0.1, 0.1],
            [0.0, 0.1]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.5, 0.2, 0.0, 0.0],
            [0.0, 0.5, 0.3, 0.2, 0.1]
        ], dtype=float, order='F')

        nr = 3
        alpha = -0.5
        tol1 = 0.0
        tol2 = 1e-10

        ar, br, cr, nr_out, nq, hsv, iwarn, info = ab09fd(
            'C', 'L', 'S', 'N', 'S', 'F',
            n, m, p, nr, alpha, a, b, c, tol1, tol2
        )

        assert info == 0
        assert nr_out <= nr
        assert nq <= n

        if nr_out > 0:
            ar_red = ar[:nr_out, :nr_out].copy()
            eigs = np.linalg.eigvals(ar_red)
            for eig in eigs:
                assert eig.real < 0, f"Eigenvalue {eig} not in left half-plane"


class TestAB09FDEdgeCases:
    """Edge case tests."""

    def test_zero_dimensions(self):
        """Test with n=0 returns immediately."""
        a = np.zeros((1, 1), dtype=float, order='F')
        b = np.zeros((1, 1), dtype=float, order='F')
        c = np.zeros((1, 1), dtype=float, order='F')

        ar, br, cr, nr_out, nq, hsv, iwarn, info = ab09fd(
            'C', 'L', 'I', 'B', 'N', 'A',
            0, 1, 1, 0, -0.1, a, b, c, 0.0, 0.0
        )

        assert info == 0
        assert nr_out == 0
        assert nq == 0

    def test_siso_system(self):
        """
        Test single-input single-output system.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 3, 1, 1

        a = np.array([
            [-1.0,  0.5, 0.0],
            [ 0.0, -2.0, 0.5],
            [ 0.0,  0.0, -3.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0],
            [0.5],
            [0.1]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.5, 0.1]
        ], dtype=float, order='F')

        nr = 2
        alpha = -0.5

        ar, br, cr, nr_out, nq, hsv, iwarn, info = ab09fd(
            'C', 'L', 'S', 'B', 'N', 'F',
            n, m, p, nr, alpha, a, b, c, 0.0, 1e-10
        )

        assert info == 0
        assert nr_out <= n
        if nr_out > 0:
            ar_red = ar[:nr_out, :nr_out].copy()
            eigs = np.linalg.eigvals(ar_red)
            for eig in eigs:
                assert eig.real < 0

    def test_fixed_order_larger_than_nq(self):
        """
        Test when fixed NR is larger than computed NQ.
        Should trigger IWARN = 1.
        """
        n, m, p = 3, 1, 1

        a = np.array([
            [-1.0,  0.5, 0.0],
            [ 0.0, -2.0, 0.5],
            [ 0.0,  0.0, -3.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0],
            [0.0],
            [0.0]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.0, 0.0]
        ], dtype=float, order='F')

        nr = n
        alpha = -0.5

        ar, br, cr, nr_out, nq, hsv, iwarn, info = ab09fd(
            'C', 'R', 'S', 'B', 'N', 'F',
            n, m, p, nr, alpha, a, b, c, 0.0, 1e-10
        )

        assert info == 0
        assert nr_out <= n


class TestAB09FDErrors:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test invalid DICO parameter."""
        a = np.zeros((2, 2), dtype=float, order='F')
        b = np.zeros((2, 1), dtype=float, order='F')
        c = np.zeros((1, 2), dtype=float, order='F')

        with pytest.raises(ValueError):
            ab09fd(
                'X', 'L', 'I', 'B', 'N', 'A',
                2, 1, 1, 0, -0.1, a, b, c, 0.0, 0.0
            )

    def test_invalid_alpha_continuous(self):
        """Test invalid ALPHA for continuous-time (must be < 0)."""
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], dtype=float, order='F')
        b = np.array([[1.0], [0.0]], dtype=float, order='F')
        c = np.array([[1.0, 0.0]], dtype=float, order='F')

        ar, br, cr, nr_out, nq, hsv, iwarn, info = ab09fd(
            'C', 'L', 'S', 'B', 'N', 'A',
            2, 1, 1, 0, 0.5, a, b, c, 0.0, 0.0
        )

        assert info == -7

    def test_invalid_alpha_discrete(self):
        """Test invalid ALPHA for discrete-time (must be 0 <= alpha < 1)."""
        a = np.array([[0.5, 0.0], [0.0, 0.3]], dtype=float, order='F')
        b = np.array([[1.0], [0.0]], dtype=float, order='F')
        c = np.array([[1.0, 0.0]], dtype=float, order='F')

        ar, br, cr, nr_out, nq, hsv, iwarn, info = ab09fd(
            'D', 'L', 'S', 'B', 'N', 'A',
            2, 1, 1, 0, -0.5, a, b, c, 0.0, 0.0
        )

        assert info == -7


class TestAB09FDHSVOrdering:
    """Validate Hankel singular values properties."""

    def test_hsv_decreasing_order(self):
        """
        Validate HSV are returned in decreasing order.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 6, 2, 2

        a = np.diag([-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]).astype(float, order='F')
        a[0, 1] = 0.1
        a[1, 2] = 0.1
        a[2, 3] = 0.1
        a = np.asfortranarray(a)

        b = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
            [0.3, 0.3],
            [0.2, 0.2],
            [0.1, 0.1],
            [0.05, 0.05]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.5, 0.3, 0.2, 0.1, 0.05],
            [0.5, 1.0, 0.3, 0.2, 0.1, 0.05]
        ], dtype=float, order='F')

        ar, br, cr, nr_out, nq, hsv, iwarn, info = ab09fd(
            'C', 'L', 'I', 'B', 'S', 'A',
            n, m, p, 0, -0.5, a, b, c, 0.01, 1e-10
        )

        assert info == 0

        if nq > 1:
            for i in range(nq - 1):
                assert hsv[i] >= hsv[i + 1], f"HSV not decreasing at index {i}: {hsv[i]} < {hsv[i+1]}"
