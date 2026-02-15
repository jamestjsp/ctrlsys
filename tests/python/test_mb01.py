"""
Tests for MB01 family routines.

SPDX-License-Identifier: BSD-3-Clause
"""
import numpy as np
import pytest
from ctrlsys import (
    mb01pd, mb01qd, mb01rb, mb01rd, mb01ru, mb01rw, mb01rx, mb01ry,
    mb01sd, mb01td, mb01ud, mb01uy, mb01kd, mb01ld, mb01md, mb01nd
)


# =============================================================================
# MB01KD Tests - Skew-symmetric rank-2k update
# =============================================================================

class TestMB01KD:
    """Tests for mb01kd - skew-symmetric rank-2k operations."""

    def test_mb01kd_upper_notrans_basic(self):
        """
        Test C := alpha*A*B' - alpha*B*A' + beta*C with upper triangle.

        Validates skew-symmetric property: C = -C^T
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, k = 4, 3
        alpha, beta = 2.0, 0.5

        a = np.random.randn(n, k).astype(float, order='F')
        b = np.random.randn(n, k).astype(float, order='F')
        c = np.random.randn(n, n).astype(float, order='F')
        c_orig = c.copy()

        c_result, info = mb01kd('U', 'N', n, k, alpha, a, b, beta, c)

        assert info == 0

        expected = alpha * (a @ b.T) - alpha * (b @ a.T)
        for j in range(1, n):
            for i in range(j):
                expected_val = beta * c_orig[i, j] + expected[i, j]
                np.testing.assert_allclose(c_result[i, j], expected_val, rtol=1e-14)

    def test_mb01kd_lower_notrans_basic(self):
        """
        Test C := alpha*A*B' - alpha*B*A' + beta*C with lower triangle.

        Validates skew-symmetric property: C = -C^T
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, k = 4, 3
        alpha, beta = 1.5, 0.0

        a = np.random.randn(n, k).astype(float, order='F')
        b = np.random.randn(n, k).astype(float, order='F')
        c = np.zeros((n, n), dtype=float, order='F')

        c_result, info = mb01kd('L', 'N', n, k, alpha, a, b, beta, c)

        assert info == 0

        expected = alpha * (a @ b.T) - alpha * (b @ a.T)
        for j in range(n - 1):
            for i in range(j + 1, n):
                np.testing.assert_allclose(c_result[i, j], expected[i, j], rtol=1e-14)

    def test_mb01kd_upper_trans_basic(self):
        """
        Test C := alpha*A'*B - alpha*B'*A + beta*C with upper triangle.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, k = 4, 5
        alpha, beta = 1.0, 0.5

        a = np.random.randn(k, n).astype(float, order='F')
        b = np.random.randn(k, n).astype(float, order='F')
        c = np.random.randn(n, n).astype(float, order='F')
        c_orig = c.copy()

        c_result, info = mb01kd('U', 'T', n, k, alpha, a, b, beta, c)

        assert info == 0

        expected = alpha * (a.T @ b) - alpha * (b.T @ a)
        for j in range(1, n):
            for i in range(j):
                expected_val = beta * c_orig[i, j] + expected[i, j]
                np.testing.assert_allclose(c_result[i, j], expected_val, rtol=1e-13)

    def test_mb01kd_skew_symmetric_property(self):
        """
        Validate mathematical property: result is skew-symmetric.

        For any A, B: A*B' - B*A' is skew-symmetric.
        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, k = 5, 3
        alpha, beta = 1.0, 0.0

        a = np.random.randn(n, k).astype(float, order='F')
        b = np.random.randn(n, k).astype(float, order='F')
        c = np.zeros((n, n), dtype=float, order='F')

        c_upper, info = mb01kd('U', 'N', n, k, alpha, a, b, beta, c.copy())
        assert info == 0

        c_lower, info = mb01kd('L', 'N', n, k, alpha, a, b, beta, c.copy())
        assert info == 0

        for i in range(n):
            for j in range(i + 1, n):
                np.testing.assert_allclose(c_upper[i, j], -c_lower[j, i], rtol=1e-14)

    def test_mb01kd_alpha_zero(self):
        """Test quick return when alpha = 0."""
        np.random.seed(100)
        n, k = 3, 2
        alpha, beta = 0.0, 2.0

        a = np.random.randn(n, k).astype(float, order='F')
        b = np.random.randn(n, k).astype(float, order='F')
        c = np.random.randn(n, n).astype(float, order='F')
        c_orig = c.copy()

        c_result, info = mb01kd('U', 'N', n, k, alpha, a, b, beta, c)

        assert info == 0
        for j in range(1, n):
            for i in range(j):
                np.testing.assert_allclose(c_result[i, j], beta * c_orig[i, j], rtol=1e-14)

    def test_mb01kd_beta_zero(self):
        """Test when beta = 0 (C not needed)."""
        np.random.seed(101)
        n, k = 3, 2
        alpha, beta = 1.0, 0.0

        a = np.random.randn(n, k).astype(float, order='F')
        b = np.random.randn(n, k).astype(float, order='F')
        c = np.random.randn(n, n).astype(float, order='F')

        c_result, info = mb01kd('U', 'N', n, k, alpha, a, b, beta, c)

        assert info == 0
        expected = alpha * (a @ b.T) - alpha * (b @ a.T)
        for j in range(1, n):
            for i in range(j):
                np.testing.assert_allclose(c_result[i, j], expected[i, j], rtol=1e-14)

    def test_mb01kd_n_le_1(self):
        """Test quick return for n <= 1."""
        a = np.array([[1.0]], order='F')
        b = np.array([[2.0]], order='F')
        c = np.array([[0.0]], order='F')

        c_result, info = mb01kd('U', 'N', 1, 1, 1.0, a, b, 0.5, c)
        assert info == 0

    def test_mb01kd_invalid_uplo(self):
        """Test error for invalid UPLO parameter."""
        n, k = 3, 2
        a = np.zeros((n, k), dtype=float, order='F')
        b = np.zeros((n, k), dtype=float, order='F')
        c = np.zeros((n, n), dtype=float, order='F')

        c_result, info = mb01kd('X', 'N', n, k, 1.0, a, b, 0.0, c)
        assert info == -1

    def test_mb01kd_invalid_trans(self):
        """Test error for invalid TRANS parameter."""
        n, k = 3, 2
        a = np.zeros((n, k), dtype=float, order='F')
        b = np.zeros((n, k), dtype=float, order='F')
        c = np.zeros((n, n), dtype=float, order='F')

        c_result, info = mb01kd('U', 'X', n, k, 1.0, a, b, 0.0, c)
        assert info == -2


# =============================================================================
# MB01LD Tests - Skew-symmetric matrix formula R = alpha*R + beta*op(A)*X*op(A)'
# =============================================================================

class TestMB01LD:
    """Tests for mb01ld - skew-symmetric matrix formula."""

    def test_mb01ld_upper_notrans_basic(self):
        """
        Test R := alpha*R + beta*A*X*A' with upper triangle, TRANS='N'.

        Validates the skew-symmetric transformation.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m, n = 4, 3
        alpha, beta = 0.5, 2.0

        r_init = np.random.randn(m, m).astype(float, order='F')
        r_init = np.triu(r_init, k=1)
        r = r_init.copy()

        a = np.random.randn(m, n).astype(float, order='F')

        x_skew = np.random.randn(n, n).astype(float, order='F')
        x_skew = np.triu(x_skew, k=1)
        x = x_skew.copy()

        r_result, info = mb01ld('U', 'N', m, n, alpha, beta, r, a, x)

        assert info == 0

        x_full = x_skew - x_skew.T
        axa = a @ x_full @ a.T
        expected = alpha * r_init + beta * axa

        for j in range(1, m):
            for i in range(j):
                np.testing.assert_allclose(r_result[i, j], expected[i, j], rtol=1e-13)

    def test_mb01ld_lower_notrans_basic(self):
        """
        Test R := alpha*R + beta*A*X*A' with lower triangle, TRANS='N'.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m, n = 4, 3
        alpha, beta = 1.0, 1.0

        r_init = np.random.randn(m, m).astype(float, order='F')
        r_init = np.tril(r_init, k=-1)
        r = r_init.copy()

        a = np.random.randn(m, n).astype(float, order='F')

        x_skew = np.random.randn(n, n).astype(float, order='F')
        x_skew = np.tril(x_skew, k=-1)
        x = x_skew.copy()

        r_result, info = mb01ld('L', 'N', m, n, alpha, beta, r, a, x)

        assert info == 0

        x_full = x_skew - x_skew.T
        axa = a @ x_full @ a.T
        expected = alpha * r_init + beta * axa

        for j in range(m - 1):
            for i in range(j + 1, m):
                np.testing.assert_allclose(r_result[i, j], expected[i, j], rtol=1e-13)

    def test_mb01ld_upper_trans_basic(self):
        """
        Test R := alpha*R + beta*A'*X*A with upper triangle, TRANS='T'.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m, n = 3, 5
        alpha, beta = 0.5, 1.5

        r_init = np.random.randn(m, m).astype(float, order='F')
        r_init = np.triu(r_init, k=1)
        r = r_init.copy()

        a = np.random.randn(n, m).astype(float, order='F')

        x_skew = np.random.randn(n, n).astype(float, order='F')
        x_skew = np.triu(x_skew, k=1)
        x = x_skew.copy()

        r_result, info = mb01ld('U', 'T', m, n, alpha, beta, r, a, x)

        assert info == 0

        x_full = x_skew - x_skew.T
        axa = a.T @ x_full @ a
        expected = alpha * r_init + beta * axa

        for j in range(1, m):
            for i in range(j):
                np.testing.assert_allclose(r_result[i, j], expected[i, j], rtol=1e-13)

    def test_mb01ld_skew_symmetric_preservation(self):
        """
        Validate mathematical property: result preserves skew-symmetry.

        If R and X are skew-symmetric, then alpha*R + beta*A*X*A' is also skew-symmetric.
        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m, n = 4, 3
        alpha, beta = 1.0, 1.0

        r_init = np.zeros((m, m), dtype=float, order='F')
        r_init = np.triu(r_init, k=1)
        r_upper = r_init.copy()
        r_lower = np.tril(r_init.T, k=-1).copy()

        a = np.random.randn(m, n).astype(float, order='F')

        x_raw = np.random.randn(n, n)
        x_skew = (x_raw - x_raw.T) / 2
        x_upper = np.triu(x_skew, k=1).astype(float, order='F')
        x_lower = np.tril(x_skew, k=-1).astype(float, order='F')

        r_u, info_u = mb01ld('U', 'N', m, n, alpha, beta, r_upper.copy(), a, x_upper.copy())
        assert info_u == 0

        r_l, info_l = mb01ld('L', 'N', m, n, alpha, beta, r_lower.copy(), a, x_lower.copy())
        assert info_l == 0

        for i in range(m):
            for j in range(i + 1, m):
                np.testing.assert_allclose(r_u[i, j], -r_l[j, i], rtol=1e-13)

    def test_mb01ld_alpha_zero(self):
        """Test when alpha = 0."""
        np.random.seed(200)
        m, n = 3, 2
        alpha, beta = 0.0, 1.0

        r = np.random.randn(m, m).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')
        x = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')

        r_result, info = mb01ld('U', 'N', m, n, alpha, beta, r, a, x)

        assert info == 0

        x_full = x - x.T
        expected = beta * (a @ x_full @ a.T)

        for j in range(1, m):
            for i in range(j):
                np.testing.assert_allclose(r_result[i, j], expected[i, j], rtol=1e-13)

    def test_mb01ld_beta_zero(self):
        """Test when beta = 0 (only scale R)."""
        np.random.seed(201)
        m, n = 3, 2
        alpha, beta = 2.0, 0.0

        r_init = np.random.randn(m, m).astype(float, order='F')
        r_init = np.triu(r_init, k=1)
        r = r_init.copy()

        a = np.random.randn(m, n).astype(float, order='F')
        x = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')

        r_result, info = mb01ld('U', 'N', m, n, alpha, beta, r, a, x)

        assert info == 0

        for j in range(1, m):
            for i in range(j):
                np.testing.assert_allclose(r_result[i, j], alpha * r_init[i, j], rtol=1e-14)

    def test_mb01ld_m_zero(self):
        """Test quick return for m = 0."""
        r = np.zeros((1, 1), dtype=float, order='F')
        a = np.zeros((1, 2), dtype=float, order='F')
        x = np.zeros((2, 2), dtype=float, order='F')

        r_result, info = mb01ld('U', 'N', 0, 2, 1.0, 1.0, r, a, x)
        assert info == 0

    def test_mb01ld_n_le_1(self):
        """Test quick return for n <= 1."""
        np.random.seed(300)
        m, n = 3, 1
        alpha = 2.0

        r_init = np.random.randn(m, m).astype(float, order='F')
        r_init = np.triu(r_init, k=1)
        r = r_init.copy()

        a = np.random.randn(m, n).astype(float, order='F')
        x = np.zeros((n, n), dtype=float, order='F')

        r_result, info = mb01ld('U', 'N', m, n, alpha, 1.0, r, a, x)

        assert info == 0
        for j in range(1, m):
            for i in range(j):
                np.testing.assert_allclose(r_result[i, j], alpha * r_init[i, j], rtol=1e-14)

    def test_mb01ld_invalid_uplo(self):
        """Test error for invalid UPLO parameter."""
        m, n = 3, 2
        r = np.zeros((m, m), dtype=float, order='F')
        a = np.zeros((m, n), dtype=float, order='F')
        x = np.zeros((n, n), dtype=float, order='F')

        r_result, info = mb01ld('X', 'N', m, n, 1.0, 1.0, r, a, x)
        assert info == -1

    def test_mb01ld_invalid_trans(self):
        """Test error for invalid TRANS parameter."""
        m, n = 3, 2
        r = np.zeros((m, m), dtype=float, order='F')
        a = np.zeros((m, n), dtype=float, order='F')
        x = np.zeros((n, n), dtype=float, order='F')

        r_result, info = mb01ld('U', 'X', m, n, 1.0, 1.0, r, a, x)
        assert info == -2

    def test_mb01ld_blas3_path(self):
        """
        Test with sufficient workspace for BLAS 3 path (ldwork >= m*(n-1)).

        Random seed: 500 (for reproducibility)
        """
        np.random.seed(500)
        m, n = 5, 4
        alpha, beta = 0.5, 1.5

        r_init = np.random.randn(m, m).astype(float, order='F')
        r_init = np.triu(r_init, k=1)
        r = r_init.copy()

        a = np.random.randn(m, n).astype(float, order='F')

        x_skew = np.random.randn(n, n).astype(float, order='F')
        x_skew = np.triu(x_skew, k=1)
        x = x_skew.copy()

        r_result, info = mb01ld('U', 'N', m, n, alpha, beta, r, a, x)

        assert info == 0

        x_full = x_skew - x_skew.T
        axa = a @ x_full @ a.T
        expected = alpha * r_init + beta * axa

        for j in range(1, m):
            for i in range(j):
                np.testing.assert_allclose(r_result[i, j], expected[i, j], rtol=1e-13)


# =============================================================================
# Existing MB01 Tests (keep for reference)
# =============================================================================

class TestMB01PD:
    """Tests for mb01pd - matrix scaling."""

    def test_mb01pd_basic(self):
        """Test basic scaling functionality with normal-sized matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float, order='F')
        anrm = np.max(np.abs(a))
        a_scaled, info = mb01pd('S', 'G', 2, 2, 0, 0, anrm, 0, None, a)
        assert info == 0
        np.testing.assert_allclose(a_scaled, a, rtol=1e-14)


class TestMB01RU:
    """Tests for mb01ru."""

    def test_mb01ru_basic(self):
        """Test basic functionality."""
        np.random.seed(42)
        m, n = 3, 2
        r = np.zeros((m, m), dtype=float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')
        x = np.eye(n, dtype=float, order='F')

        r_result, info = mb01ru('U', 'N', m, n, 0.0, 1.0, r, a, x)
        assert info == 0


# =============================================================================
# MB01MD Tests - Skew-symmetric matrix-vector multiply
# =============================================================================

class TestMB01MD:
    """Tests for mb01md - y := alpha*A*x + beta*y with A skew-symmetric."""

    def test_mb01md_upper_basic(self):
        """
        Test basic functionality with upper triangular storage.

        Computes y := alpha*A*x + beta*y where A is skew-symmetric.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4
        alpha, beta = 2.0, 0.5

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_full = a_upper - a_upper.T

        x = np.random.randn(n).astype(float)
        y_init = np.random.randn(n).astype(float)
        y = y_init.copy()

        y_result, info = mb01md('U', n, alpha, a_upper, x, 1, beta, y, 1)

        assert info == 0
        y_expected = alpha * (a_full @ x) + beta * y_init
        np.testing.assert_allclose(y_result, y_expected, rtol=1e-14)

    def test_mb01md_lower_basic(self):
        """
        Test basic functionality with lower triangular storage.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4
        alpha, beta = 1.5, 0.0

        a_lower = np.tril(np.random.randn(n, n), k=-1).astype(float, order='F')
        a_full = -a_lower.T + a_lower

        x = np.random.randn(n).astype(float)
        y = np.zeros(n, dtype=float)

        y_result, info = mb01md('L', n, alpha, a_lower, x, 1, beta, y, 1)

        assert info == 0
        y_expected = alpha * (a_full @ x)
        np.testing.assert_allclose(y_result, y_expected, rtol=1e-14)

    def test_mb01md_skew_symmetric_property(self):
        """
        Validate mathematical property: A = -A' for skew-symmetric matrices.

        The result y = A*x should satisfy y = -A'*x when using both upper
        and lower storage with same skew-symmetric matrix.
        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 5
        alpha, beta = 1.0, 0.0

        raw = np.random.randn(n, n)
        a_skew = (raw - raw.T) / 2
        a_upper = np.triu(a_skew, k=1).astype(float, order='F')
        a_lower = np.tril(a_skew, k=-1).astype(float, order='F')

        x = np.random.randn(n).astype(float)

        y_u = np.zeros(n, dtype=float)
        y_u_result, info_u = mb01md('U', n, alpha, a_upper, x, 1, beta, y_u, 1)
        assert info_u == 0

        y_l = np.zeros(n, dtype=float)
        y_l_result, info_l = mb01md('L', n, alpha, a_lower, x, 1, beta, y_l, 1)
        assert info_l == 0

        np.testing.assert_allclose(y_u_result, y_l_result, rtol=1e-14)

    def test_mb01md_negative_incx(self):
        """
        Test with negative increment for x (reversed order access).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4
        alpha, beta = 1.0, 0.0

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_full = a_upper - a_upper.T

        x = np.random.randn(n).astype(float)
        y = np.zeros(n, dtype=float)

        y_result, info = mb01md('U', n, alpha, a_upper, x, -1, beta, y, 1)

        assert info == 0
        x_reversed = x[::-1]
        y_expected = alpha * (a_full @ x_reversed)
        np.testing.assert_allclose(y_result, y_expected, rtol=1e-14)

    def test_mb01md_negative_incy(self):
        """
        Test with negative increment for y (reversed order access).

        Random seed: 101 (for reproducibility)
        """
        np.random.seed(101)
        n = 4
        alpha, beta = 1.0, 0.5

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_full = a_upper - a_upper.T

        x = np.random.randn(n).astype(float)
        y_init = np.random.randn(n).astype(float)
        y = y_init.copy()

        y_result, info = mb01md('U', n, alpha, a_upper, x, 1, beta, y, -1)

        assert info == 0
        y_init_reversed = y_init[::-1]
        y_expected_reversed = alpha * (a_full @ x) + beta * y_init_reversed
        y_expected = y_expected_reversed[::-1]
        np.testing.assert_allclose(y_result, y_expected, rtol=1e-14)

    def test_mb01md_alpha_zero(self):
        """Test quick return when alpha = 0 (just scale y by beta)."""
        np.random.seed(200)
        n = 3
        alpha, beta = 0.0, 2.0

        a = np.random.randn(n, n).astype(float, order='F')
        x = np.random.randn(n).astype(float)
        y_init = np.random.randn(n).astype(float)
        y = y_init.copy()

        y_result, info = mb01md('U', n, alpha, a, x, 1, beta, y, 1)

        assert info == 0
        np.testing.assert_allclose(y_result, beta * y_init, rtol=1e-14)

    def test_mb01md_beta_one_alpha_zero(self):
        """Test quick return when alpha=0 and beta=1 (y unchanged)."""
        np.random.seed(201)
        n = 3

        a = np.random.randn(n, n).astype(float, order='F')
        x = np.random.randn(n).astype(float)
        y_init = np.random.randn(n).astype(float)
        y = y_init.copy()

        y_result, info = mb01md('U', n, 0.0, a, x, 1, 1.0, y, 1)

        assert info == 0
        np.testing.assert_allclose(y_result, y_init, rtol=1e-14)

    def test_mb01md_beta_zero(self):
        """Test when beta=0 (y not needed on input)."""
        np.random.seed(202)
        n = 4
        alpha = 2.0

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_full = a_upper - a_upper.T

        x = np.random.randn(n).astype(float)
        y = np.array([999.0] * n, dtype=float)

        y_result, info = mb01md('U', n, alpha, a_upper, x, 1, 0.0, y, 1)

        assert info == 0
        y_expected = alpha * (a_full @ x)
        np.testing.assert_allclose(y_result, y_expected, rtol=1e-14)

    def test_mb01md_n_zero(self):
        """Test quick return for n=0."""
        a = np.zeros((1, 1), dtype=float, order='F')
        x = np.array([], dtype=float)
        y = np.array([], dtype=float)

        y_result, info = mb01md('U', 0, 1.0, a, x, 1, 1.0, y, 1)
        assert info == 0

    def test_mb01md_invalid_uplo(self):
        """Test error for invalid UPLO parameter."""
        n = 3
        a = np.zeros((n, n), dtype=float, order='F')
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)

        y_result, info = mb01md('X', n, 1.0, a, x, 1, 0.0, y, 1)
        assert info == 1

    def test_mb01md_invalid_n(self):
        """Test error for invalid N parameter (N < 0)."""
        a = np.zeros((3, 3), dtype=float, order='F')
        x = np.zeros(3, dtype=float)
        y = np.zeros(3, dtype=float)

        y_result, info = mb01md('U', -1, 1.0, a, x, 1, 0.0, y, 1)
        assert info == 2

    def test_mb01md_invalid_incx(self):
        """Test error for invalid INCX parameter (INCX = 0)."""
        n = 3
        a = np.zeros((n, n), dtype=float, order='F')
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)

        y_result, info = mb01md('U', n, 1.0, a, x, 0, 0.0, y, 1)
        assert info == 7

    def test_mb01md_invalid_incy(self):
        """Test error for invalid INCY parameter (INCY = 0)."""
        n = 3
        a = np.zeros((n, n), dtype=float, order='F')
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)

        y_result, info = mb01md('U', n, 1.0, a, x, 1, 0.0, y, 0)
        assert info == 10

    def test_mb01md_stride_incx(self):
        """
        Test with non-unit increment for x (incx=2).

        Random seed: 300 (for reproducibility)
        """
        np.random.seed(300)
        n = 3

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_full = a_upper - a_upper.T

        x_full = np.random.randn(2 * n - 1).astype(float)
        x_strided = x_full[::2]
        y = np.zeros(n, dtype=float)

        y_result, info = mb01md('U', n, 1.0, a_upper, x_full, 2, 0.0, y, 1)

        assert info == 0
        y_expected = a_full @ x_strided
        np.testing.assert_allclose(y_result, y_expected, rtol=1e-14)

    def test_mb01md_stride_incy(self):
        """
        Test with non-unit increment for y (incy=2).

        Random seed: 301 (for reproducibility)
        """
        np.random.seed(301)
        n = 3

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_full = a_upper - a_upper.T

        x = np.random.randn(n).astype(float)
        y_full = np.zeros(2 * n - 1, dtype=float)
        y_full_init = y_full.copy()

        y_result, info = mb01md('U', n, 1.0, a_upper, x, 1, 0.0, y_full, 2)

        assert info == 0
        y_expected = a_full @ x
        np.testing.assert_allclose(y_result[::2], y_expected, rtol=1e-14)


# =============================================================================
# MB01ND Tests - Skew-symmetric rank-2 operation
# =============================================================================

class TestMB01ND:
    """Tests for mb01nd - skew-symmetric rank 2 operation A := alpha*x*y' - alpha*y*x' + A."""

    def test_mb01nd_upper_basic(self):
        """
        Test basic functionality with upper triangular storage.

        Computes A := alpha*x*y' - alpha*y*x' + A where A is skew-symmetric.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4
        alpha = 2.0

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_orig = a_upper.copy()

        x = np.random.randn(n).astype(float)
        y = np.random.randn(n).astype(float)

        a_result, info = mb01nd('U', n, alpha, x, 1, y, 1, a_upper)

        assert info == 0

        xy_t = np.outer(x, y)
        yx_t = np.outer(y, x)
        update = alpha * xy_t - alpha * yx_t

        for j in range(1, n):
            for i in range(j):
                expected = a_orig[i, j] + update[i, j]
                np.testing.assert_allclose(a_result[i, j], expected, rtol=1e-14)

    def test_mb01nd_lower_basic(self):
        """
        Test basic functionality with lower triangular storage.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4
        alpha = 1.5

        a_lower = np.tril(np.random.randn(n, n), k=-1).astype(float, order='F')
        a_orig = a_lower.copy()

        x = np.random.randn(n).astype(float)
        y = np.random.randn(n).astype(float)

        a_result, info = mb01nd('L', n, alpha, x, 1, y, 1, a_lower)

        assert info == 0

        xy_t = np.outer(x, y)
        yx_t = np.outer(y, x)
        update = alpha * xy_t - alpha * yx_t

        for j in range(n - 1):
            for i in range(j + 1, n):
                expected = a_orig[i, j] + update[i, j]
                np.testing.assert_allclose(a_result[i, j], expected, rtol=1e-14)

    def test_mb01nd_skew_symmetric_property(self):
        """
        Validate mathematical property: x*y' - y*x' is skew-symmetric.

        For any vectors x, y: the outer product difference x*y' - y*x' is skew-symmetric.
        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 5
        alpha = 1.0

        a_upper = np.zeros((n, n), dtype=float, order='F')
        a_lower = np.zeros((n, n), dtype=float, order='F')

        x = np.random.randn(n).astype(float)
        y = np.random.randn(n).astype(float)

        a_u, info_u = mb01nd('U', n, alpha, x, 1, y, 1, a_upper.copy())
        assert info_u == 0

        a_l, info_l = mb01nd('L', n, alpha, x, 1, y, 1, a_lower.copy())
        assert info_l == 0

        for i in range(n):
            for j in range(i + 1, n):
                np.testing.assert_allclose(a_u[i, j], -a_l[j, i], rtol=1e-13, atol=1e-15)

    def test_mb01nd_negative_incx(self):
        """
        Test with negative increment for x (reversed order access).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4
        alpha = 1.0

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_orig = a_upper.copy()

        x = np.random.randn(n).astype(float)
        y = np.random.randn(n).astype(float)

        a_result, info = mb01nd('U', n, alpha, x, -1, y, 1, a_upper)

        assert info == 0

        x_reversed = x[::-1]
        xy_t = np.outer(x_reversed, y)
        yx_t = np.outer(y, x_reversed)
        update = alpha * xy_t - alpha * yx_t

        for j in range(1, n):
            for i in range(j):
                expected = a_orig[i, j] + update[i, j]
                np.testing.assert_allclose(a_result[i, j], expected, rtol=1e-14)

    def test_mb01nd_negative_incy(self):
        """
        Test with negative increment for y (reversed order access).

        Random seed: 101 (for reproducibility)
        """
        np.random.seed(101)
        n = 4
        alpha = 1.0

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_orig = a_upper.copy()

        x = np.random.randn(n).astype(float)
        y = np.random.randn(n).astype(float)

        a_result, info = mb01nd('U', n, alpha, x, 1, y, -1, a_upper)

        assert info == 0

        y_reversed = y[::-1]
        xy_t = np.outer(x, y_reversed)
        yx_t = np.outer(y_reversed, x)
        update = alpha * xy_t - alpha * yx_t

        for j in range(1, n):
            for i in range(j):
                expected = a_orig[i, j] + update[i, j]
                np.testing.assert_allclose(a_result[i, j], expected, rtol=1e-14)

    def test_mb01nd_stride_incx(self):
        """
        Test with non-unit increment for x (incx=2).

        Random seed: 300 (for reproducibility)
        """
        np.random.seed(300)
        n = 3
        alpha = 1.5

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_orig = a_upper.copy()

        x_full = np.random.randn(2 * n - 1).astype(float)
        x_strided = x_full[::2]
        y = np.random.randn(n).astype(float)

        a_result, info = mb01nd('U', n, alpha, x_full, 2, y, 1, a_upper)

        assert info == 0

        xy_t = np.outer(x_strided, y)
        yx_t = np.outer(y, x_strided)
        update = alpha * xy_t - alpha * yx_t

        for j in range(1, n):
            for i in range(j):
                expected = a_orig[i, j] + update[i, j]
                np.testing.assert_allclose(a_result[i, j], expected, rtol=1e-14)

    def test_mb01nd_stride_incy(self):
        """
        Test with non-unit increment for y (incy=2).

        Random seed: 301 (for reproducibility)
        """
        np.random.seed(301)
        n = 3
        alpha = 1.5

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_orig = a_upper.copy()

        x = np.random.randn(n).astype(float)
        y_full = np.random.randn(2 * n - 1).astype(float)
        y_strided = y_full[::2]

        a_result, info = mb01nd('U', n, alpha, x, 1, y_full, 2, a_upper)

        assert info == 0

        xy_t = np.outer(x, y_strided)
        yx_t = np.outer(y_strided, x)
        update = alpha * xy_t - alpha * yx_t

        for j in range(1, n):
            for i in range(j):
                expected = a_orig[i, j] + update[i, j]
                np.testing.assert_allclose(a_result[i, j], expected, rtol=1e-14)

    def test_mb01nd_alpha_zero(self):
        """Test quick return when alpha = 0."""
        np.random.seed(200)
        n = 4

        a_upper = np.triu(np.random.randn(n, n), k=1).astype(float, order='F')
        a_orig = a_upper.copy()

        x = np.random.randn(n).astype(float)
        y = np.random.randn(n).astype(float)

        a_result, info = mb01nd('U', n, 0.0, x, 1, y, 1, a_upper)

        assert info == 0
        for j in range(1, n):
            for i in range(j):
                np.testing.assert_allclose(a_result[i, j], a_orig[i, j], rtol=1e-14)

    def test_mb01nd_n_zero(self):
        """Test quick return for n=0."""
        a = np.zeros((1, 1), dtype=float, order='F')
        x = np.array([], dtype=float)
        y = np.array([], dtype=float)

        a_result, info = mb01nd('U', 0, 1.0, x, 1, y, 1, a)
        assert info == 0

    def test_mb01nd_invalid_uplo(self):
        """Test error for invalid UPLO parameter."""
        n = 3
        a = np.zeros((n, n), dtype=float, order='F')
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)

        a_result, info = mb01nd('X', n, 1.0, x, 1, y, 1, a)
        assert info == 1

    def test_mb01nd_invalid_n(self):
        """Test error for invalid N parameter (N < 0)."""
        a = np.zeros((3, 3), dtype=float, order='F')
        x = np.zeros(3, dtype=float)
        y = np.zeros(3, dtype=float)

        a_result, info = mb01nd('U', -1, 1.0, x, 1, y, 1, a)
        assert info == 2

    def test_mb01nd_invalid_incx(self):
        """Test error for invalid INCX parameter (INCX = 0)."""
        n = 3
        a = np.zeros((n, n), dtype=float, order='F')
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)

        a_result, info = mb01nd('U', n, 1.0, x, 0, y, 1, a)
        assert info == 5

    def test_mb01nd_invalid_incy(self):
        """Test error for invalid INCY parameter (INCY = 0)."""
        n = 3
        a = np.zeros((n, n), dtype=float, order='F')
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)

        a_result, info = mb01nd('U', n, 1.0, x, 1, y, 0, a)
        assert info == 7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
