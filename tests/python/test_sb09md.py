"""
Tests for SB09MD: Evaluation of closeness of two multivariable sequences.

SB09MD compares two multivariable sequences M1(k) and M2(k) for k=1,2,...,N
and evaluates their closeness through:
- SS: sum-of-squares matrix of M1
- SE: quadratic error matrix
- PRE: percentage relative error matrix
"""

import numpy as np
import pytest
from ctrlsys import sb09md


class TestSB09MDBasic:
    """Basic functionality tests from HTML documentation."""

    def test_html_doc_example(self):
        """
        Validate basic functionality using SLICOT HTML doc example.

        N=2, NC=2, NB=2, TOL=0.0
        Tests numerical correctness of SS, SE, PRE matrices.
        """
        n = 2
        nc = 2
        nb = 2
        tol = 0.0

        # H1 from HTML doc (column-wise storage)
        # READ ( NIN, FMT = * ) ( ( H1(I,J), I = 1,NC ), J = 1,N*NB )
        # Data stream: 1.3373, 0.1205, 0.6618, -0.3372, -0.4062, 1.6120, 0.9299, 0.7429
        # Reading column-wise with I varying fastest:
        # J=1: H1(1,1)=1.3373, H1(2,1)=0.1205
        # J=2: H1(1,2)=0.6618, H1(2,2)=-0.3372
        # J=3: H1(1,3)=-0.4062, H1(2,3)=1.6120
        # J=4: H1(1,4)=0.9299, H1(2,4)=0.7429
        h1 = np.array([
            [1.3373, 0.6618, -0.4062, 0.9299],
            [0.1205, -0.3372, 1.6120, 0.7429]
        ], order='F', dtype=float)

        # H2 from HTML doc (column-wise storage)
        # Data stream: 1.1480, -0.1837, 0.8843, -0.4947, -0.4616, 1.4674, 0.6028, 0.9524
        # J=1: H2(1,1)=1.1480, H2(2,1)=-0.1837
        # J=2: H2(1,2)=0.8843, H2(2,2)=-0.4947
        # J=3: H2(1,3)=-0.4616, H2(2,3)=1.4674
        # J=4: H2(1,4)=0.6028, H2(2,4)=0.9524
        h2 = np.array([
            [1.1480, 0.8843, -0.4616, 0.6028],
            [-0.1837, -0.4947, 1.4674, 0.9524]
        ], order='F', dtype=float)

        # Expected outputs from HTML doc
        ss_expected = np.array([
            [1.9534, 1.3027],
            [2.6131, 0.6656]
        ], order='F', dtype=float)

        se_expected = np.array([
            [0.0389, 0.1565],
            [0.1134, 0.0687]
        ], order='F', dtype=float)

        pre_expected = np.array([
            [14.1125, 34.6607],
            [20.8363, 32.1262]
        ], order='F', dtype=float)

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == 0
        # HTML shows 4 decimal places
        np.testing.assert_allclose(ss, ss_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(se, se_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(pre, pre_expected, rtol=1e-3, atol=1e-4)


class TestSB09MDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_identical_sequences_zero_error(self):
        """
        When M1 == M2, SE should be zero and PRE should be 0%.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3
        nc = 2
        nb = 2
        tol = 0.0

        h1 = np.random.randn(nc, n * nb).astype(float, order='F')
        h2 = h1.copy()

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == 0
        # SE should be zero (identical sequences)
        np.testing.assert_allclose(se, 0.0, atol=1e-14)
        # PRE should be 0% (no relative error)
        np.testing.assert_allclose(pre, 0.0, atol=1e-14)
        # SS should be sum of squares of h1
        for i in range(nc):
            for j in range(nb):
                expected_ss = sum(h1[i, k * nb + j] ** 2 for k in range(n))
                np.testing.assert_allclose(ss[i, j], expected_ss, rtol=1e-14)

    def test_sum_of_squares_formula(self):
        """
        Validate SS(i,j) = sum_{k=1}^{N} M1(i,j,k)^2.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4
        nc = 3
        nb = 2
        tol = 0.0

        h1 = np.random.randn(nc, n * nb).astype(float, order='F')
        h2 = np.random.randn(nc, n * nb).astype(float, order='F')

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == 0

        # Verify SS formula: SS(i,j) = sum_{k=0}^{N-1} H1(i, k*NB+j)^2
        for i in range(nc):
            for j in range(nb):
                expected_ss = sum(h1[i, k * nb + j] ** 2 for k in range(n))
                np.testing.assert_allclose(ss[i, j], expected_ss, rtol=1e-14)

    def test_quadratic_error_formula(self):
        """
        Validate SE(i,j) = sum_{k=1}^{N} (M1(i,j,k) - M2(i,j,k))^2.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 5
        nc = 2
        nb = 3
        tol = 0.0

        h1 = np.random.randn(nc, n * nb).astype(float, order='F')
        h2 = np.random.randn(nc, n * nb).astype(float, order='F')

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == 0

        # Verify SE formula: SE(i,j) = sum_{k=0}^{N-1} (H1(i, k*NB+j) - H2(i, k*NB+j))^2
        for i in range(nc):
            for j in range(nb):
                expected_se = sum(
                    (h1[i, k * nb + j] - h2[i, k * nb + j]) ** 2 for k in range(n)
                )
                np.testing.assert_allclose(se[i, j], expected_se, rtol=1e-14)

    def test_percentage_relative_error_formula(self):
        """
        Validate PRE(i,j) = 100 * sqrt(SE(i,j) / SS(i,j)).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4
        nc = 2
        nb = 2
        tol = 0.0

        # Use moderately sized values to avoid overflow/underflow edge cases
        h1 = np.random.randn(nc, n * nb).astype(float, order='F')
        h2 = h1 + 0.1 * np.random.randn(nc, n * nb)
        h2 = h2.astype(float, order='F')

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == 0

        # Verify PRE formula: PRE(i,j) = 100 * sqrt(SE(i,j) / SS(i,j))
        for i in range(nc):
            for j in range(nb):
                if ss[i, j] > 1e-15:
                    expected_pre = 100.0 * np.sqrt(se[i, j] / ss[i, j])
                    np.testing.assert_allclose(pre[i, j], expected_pre, rtol=1e-14)


class TestSB09MDEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """N=0 should return quickly with no computation."""
        n = 0
        nc = 2
        nb = 2
        tol = 0.0

        h1 = np.zeros((nc, 1), order='F', dtype=float)
        h2 = np.zeros((nc, 1), order='F', dtype=float)

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == 0

    def test_nc_zero(self):
        """NC=0 should return quickly with no computation."""
        n = 2
        nc = 0
        nb = 2
        tol = 0.0

        h1 = np.zeros((1, n * nb), order='F', dtype=float)
        h2 = np.zeros((1, n * nb), order='F', dtype=float)

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == 0

    def test_nb_zero(self):
        """NB=0 should return quickly with no computation."""
        n = 2
        nc = 2
        nb = 0
        tol = 0.0

        h1 = np.zeros((nc, 1), order='F', dtype=float)
        h2 = np.zeros((nc, 1), order='F', dtype=float)

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == 0

    def test_single_parameter(self):
        """
        Test with N=1 (single parameter case).

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        n = 1
        nc = 3
        nb = 2
        tol = 0.0

        h1 = np.random.randn(nc, n * nb).astype(float, order='F')
        h2 = np.random.randn(nc, n * nb).astype(float, order='F')

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == 0

        # With N=1, SS(i,j) = H1(i,j)^2 and SE(i,j) = (H1(i,j) - H2(i,j))^2
        for i in range(nc):
            for j in range(nb):
                expected_ss = h1[i, j] ** 2
                expected_se = (h1[i, j] - h2[i, j]) ** 2
                np.testing.assert_allclose(ss[i, j], expected_ss, rtol=1e-14)
                np.testing.assert_allclose(se[i, j], expected_se, rtol=1e-14)


class TestSB09MDErrorHandling:
    """Error handling tests."""

    def test_negative_n(self):
        """Negative N should return info=-1."""
        n = -1
        nc = 2
        nb = 2
        tol = 0.0

        h1 = np.zeros((nc, 1), order='F', dtype=float)
        h2 = np.zeros((nc, 1), order='F', dtype=float)

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == -1

    def test_negative_nc(self):
        """Negative NC should return info=-2."""
        n = 2
        nc = -1
        nb = 2
        tol = 0.0

        h1 = np.zeros((1, 1), order='F', dtype=float)
        h2 = np.zeros((1, 1), order='F', dtype=float)

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == -2

    def test_negative_nb(self):
        """Negative NB should return info=-3."""
        n = 2
        nc = 2
        nb = -1
        tol = 0.0

        h1 = np.zeros((nc, 1), order='F', dtype=float)
        h2 = np.zeros((nc, 1), order='F', dtype=float)

        ss, se, pre, info = sb09md(n, nc, nb, h1, h2, tol)

        assert info == -3
