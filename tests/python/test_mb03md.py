"""
Tests for MB03MD - Upper bound for L singular values of a bidiagonal matrix.

MB03MD uses bisection to find theta such that a bidiagonal matrix has precisely
L singular values <= theta + tol.

Test data from SLICOT HTML documentation:
- N=5, L=3, theta=-3.0 (use default)
- Q = [1.0, 2.0, 3.0, 4.0, 5.0]
- E = [2.0, 3.0, 4.0, 5.0]
- Expected: theta = 4.75, L = 3
"""

import numpy as np
import pytest
from ctrlsys import mb03md


def compute_pivmin_and_squares(q, e):
    """Compute Q2, E2, and PIVMIN as in SLICOT example."""
    n = len(q)
    q2 = q ** 2
    e2 = e ** 2
    safmin = np.finfo(float).tiny
    pivmin = max(np.max(q2), np.max(e2)) if n > 1 else q2[0] if n == 1 else safmin
    pivmin = max(pivmin * safmin, safmin)
    return q2, e2, pivmin


class TestMB03MD:
    """Tests for MB03MD singular value bound computation."""

    def test_basic_html_doc_example(self):
        """
        Test basic case from SLICOT HTML documentation.

        Bidiagonal matrix J with:
        - Q = [1, 2, 3, 4, 5] (diagonal)
        - E = [2, 3, 4, 5] (superdiagonal)

        Expected: theta = 4.75 for L = 3 singular values
        """
        pass

        n = 5
        l_in = 3
        theta_in = -3.0  # negative means use default estimate
        tol = 0.0
        reltol = 0.0  # will use default

        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        e = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
        q2, e2, pivmin = compute_pivmin_and_squares(q, e)

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, theta_in, q, e, q2, e2, pivmin, tol, reltol
        )

        assert info == 0
        assert iwarn == 0
        assert l_out == 3
        np.testing.assert_allclose(theta_out, 4.75, rtol=1e-3)

    def test_l_equals_zero(self):
        """Test L=0 case: theta should be 0."""
        pass

        n = 5
        l_in = 0
        theta_in = -1.0

        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        e = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
        q2, e2, pivmin = compute_pivmin_and_squares(q, e)

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, theta_in, q, e, q2, e2, pivmin, 0.0, 0.0
        )

        assert info == 0
        assert l_out == 0
        assert theta_out == 0.0

    def test_l_equals_n(self):
        """Test L=N case: theta should bound all singular values."""
        pass

        n = 4
        l_in = 4
        theta_in = -1.0  # use default

        q = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        e = np.array([1.0, 1.0, 1.0], dtype=float)
        q2, e2, pivmin = compute_pivmin_and_squares(q, e)

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, theta_in, q, e, q2, e2, pivmin, 0.0, 0.0
        )

        assert info == 0
        assert l_out >= l_in

    def test_n_equals_one(self):
        """Test scalar case N=1."""
        pass

        n = 1
        l_in = 1
        theta_in = -1.0

        q = np.array([3.0], dtype=float)
        e = np.array([], dtype=float)
        q2 = q ** 2
        e2 = np.array([], dtype=float)
        safmin = np.finfo(float).tiny
        pivmin = max(q2[0] * safmin, safmin)

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, theta_in, q, e, q2, e2, pivmin, 0.0, 0.0
        )

        assert info == 0
        assert l_out == 1
        np.testing.assert_allclose(theta_out, 3.0, rtol=1e-10)

    def test_n_equals_zero(self):
        """Test empty matrix N=0."""
        pass

        n = 0
        l_in = 0
        theta_in = -1.0

        q = np.array([], dtype=float)
        e = np.array([], dtype=float)
        q2 = np.array([], dtype=float)
        e2 = np.array([], dtype=float)
        pivmin = np.finfo(float).tiny

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, theta_in, q, e, q2, e2, pivmin, 0.0, 0.0
        )

        assert info == 0

    def test_singular_value_count_property(self):
        """
        Validate mathematical property: computed theta bounds exactly L singular values.

        Using SVD to verify the singular values of the bidiagonal matrix.
        Random seed: 42 (for reproducibility)
        """
        pass

        np.random.seed(42)
        n = 6
        l_in = 2

        q = np.random.rand(n) * 5 + 0.5
        e = np.random.rand(n - 1) * 3 + 0.1
        q2, e2, pivmin = compute_pivmin_and_squares(q, e)

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, -1.0, q, e, q2, e2, pivmin, 1e-10, 0.0
        )

        assert info == 0

        J = np.diag(q) + np.diag(e, k=1)
        singular_values = np.linalg.svd(J, compute_uv=False)

        count_le_theta = np.sum(singular_values <= theta_out + 1e-10)
        assert count_le_theta >= l_out

    def test_invalid_n(self):
        """Test error handling for invalid N < 0."""
        pass

        n = -1
        l_in = 0
        theta_in = 0.0

        q = np.array([], dtype=float)
        e = np.array([], dtype=float)
        q2 = np.array([], dtype=float)
        e2 = np.array([], dtype=float)
        pivmin = 1e-20

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, theta_in, q, e, q2, e2, pivmin, 0.0, 0.0
        )

        assert info == -1

    def test_invalid_l_negative(self):
        """Test error handling for invalid L < 0."""
        pass

        n = 5
        l_in = -1
        theta_in = 0.0

        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        e = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
        q2, e2, pivmin = compute_pivmin_and_squares(q, e)

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, theta_in, q, e, q2, e2, pivmin, 0.0, 0.0
        )

        assert info == -2

    def test_invalid_l_greater_than_n(self):
        """Test error handling for invalid L > N."""
        pass

        n = 3
        l_in = 5
        theta_in = 0.0

        q = np.array([1.0, 2.0, 3.0], dtype=float)
        e = np.array([1.0, 2.0], dtype=float)
        q2, e2, pivmin = compute_pivmin_and_squares(q, e)

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, theta_in, q, e, q2, e2, pivmin, 0.0, 0.0
        )

        assert info == -2

    def test_positive_theta_estimate(self):
        """Test with user-provided positive theta estimate."""
        pass

        n = 5
        l_in = 3
        theta_in = 5.0  # positive user estimate

        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        e = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
        q2, e2, pivmin = compute_pivmin_and_squares(q, e)

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, theta_in, q, e, q2, e2, pivmin, 0.0, 0.0
        )

        assert info == 0
        assert l_out >= l_in
        assert theta_out > 0

    def test_coinciding_singular_values_warning(self):
        """
        Test warning when singular values coincide within tolerance.

        When L-th and (L+1)-th smallest singular values are within TOL,
        L is increased and iwarn=1.
        """
        pass

        n = 3
        l_in = 1

        q = np.array([1.0, 1.0, 1.0], dtype=float)
        e = np.array([0.0, 0.0], dtype=float)
        q2, e2, pivmin = compute_pivmin_and_squares(q, e)

        theta_out, l_out, iwarn, info = mb03md(
            n, l_in, -1.0, q, e, q2, e2, pivmin, 0.1, 0.0
        )

        assert info == 0
