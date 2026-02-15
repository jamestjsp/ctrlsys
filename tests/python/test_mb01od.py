"""
Tests for mb01od: Compute matrix formula R := alpha*R + beta*(op(H)*X*op(E)' + op(E)*X*op(H)')
with R, X symmetric, H upper Hessenberg, and E upper triangular.
"""
import numpy as np
import pytest
from ctrlsys import mb01od


def make_symmetric(a, uplo='U'):
    """Make a matrix symmetric by copying the specified triangle."""
    n = a.shape[0]
    result = a.copy()
    if uplo == 'U':
        for i in range(n):
            for j in range(i):
                result[i, j] = result[j, i]
    else:
        for i in range(n):
            for j in range(i + 1, n):
                result[i, j] = result[j, i]
    return result


def make_upper_hessenberg(a):
    """Zero out elements below the first subdiagonal."""
    n = a.shape[0]
    result = a.copy()
    for i in range(2, n):
        for j in range(i - 1):
            result[i, j] = 0.0
    return result


def make_upper_triangular(a):
    """Zero out elements below the diagonal."""
    return np.triu(a)


class TestMB01ODBasic:
    """Basic functionality tests."""

    def test_identity_case_n1(self):
        """
        Test n=1 case with simple scalars.

        For n=1, H, E, X are scalars, formula simplifies to:
        R = alpha*R + beta*(H*X*E + E*X*H) = alpha*R + 2*beta*H*X*E
        """
        n = 1
        alpha = 1.0
        beta = 1.0

        r = np.array([[2.0]], order='F', dtype=float)
        h = np.array([[3.0]], order='F', dtype=float)
        x = np.array([[4.0]], order='F', dtype=float)
        e = np.array([[5.0]], order='F', dtype=float)

        r_out, info = mb01od('U', 'N', n, alpha, beta, r, h, x, e)

        assert info == 0
        expected = alpha * 2.0 + 2.0 * beta * 3.0 * 4.0 * 5.0
        np.testing.assert_allclose(r_out[0, 0], expected, rtol=1e-14)

    def test_upper_notrans_n3(self):
        """
        Test UPLO='U', TRANS='N' with 3x3 matrices.

        R := alpha*R + beta*(H*X*E' + E*X*H')

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3
        alpha = 0.5
        beta = 1.5

        r_sym = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0]
        ], order='F', dtype=float)

        h_full = np.random.randn(n, n).astype(float, order='F')
        h = make_upper_hessenberg(h_full)

        x_sym = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 2.0, 0.7],
            [0.3, 0.7, 3.0]
        ], order='F', dtype=float)

        e = make_upper_triangular(np.random.randn(n, n)).astype(float, order='F')

        r_in = r_sym.copy(order='F')
        x_in = x_sym.copy(order='F')
        h_in = h.copy(order='F')

        r_out, info = mb01od('U', 'N', n, alpha, beta, r_in, h_in, x_in, e)

        assert info == 0

        x_full = make_symmetric(x_sym, 'U')
        h_full_preserved = h.copy()
        product = h @ x_full @ e.T + e @ x_full @ h.T
        r_expected = alpha * r_sym + beta * product

        np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-13)

    def test_upper_trans_n3(self):
        """
        Test UPLO='U', TRANS='T' with 3x3 matrices.

        R := alpha*R + beta*(H'*X*E + E'*X*H)

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 3
        alpha = 2.0
        beta = 0.5

        r_sym = np.eye(n, order='F', dtype=float)

        h_full = np.random.randn(n, n).astype(float, order='F')
        h = make_upper_hessenberg(h_full)

        x_sym = np.eye(n, order='F', dtype=float) * 2.0

        e = make_upper_triangular(np.random.randn(n, n) + np.eye(n)).astype(float, order='F')

        r_in = r_sym.copy(order='F')
        x_in = x_sym.copy(order='F')
        h_in = h.copy(order='F')

        r_out, info = mb01od('U', 'T', n, alpha, beta, r_in, h_in, x_in, e)

        assert info == 0

        x_full = make_symmetric(x_sym, 'U')
        product = h.T @ x_full @ e + e.T @ x_full @ h
        r_expected = alpha * r_sym + beta * product

        np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-13)

    def test_lower_notrans_n3(self):
        """
        Test UPLO='L', TRANS='N' with 3x3 matrices.

        R := alpha*R + beta*(H*X*E' + E*X*H')

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 3
        alpha = 1.0
        beta = 1.0

        r_base = np.random.randn(n, n)
        r_sym = (r_base + r_base.T) / 2.0
        r_sym = np.asfortranarray(r_sym, dtype=float)

        h_full = np.random.randn(n, n).astype(float, order='F')
        h = make_upper_hessenberg(h_full)

        x_base = np.random.randn(n, n)
        x_sym = (x_base + x_base.T) / 2.0
        x_sym = np.asfortranarray(x_sym, dtype=float)

        e = make_upper_triangular(np.random.randn(n, n)).astype(float, order='F')

        r_in = r_sym.copy(order='F')
        x_in = x_sym.copy(order='F')
        h_in = h.copy(order='F')

        r_out, info = mb01od('L', 'N', n, alpha, beta, r_in, h_in, x_in, e)

        assert info == 0

        x_full = make_symmetric(x_sym, 'L')
        product = h @ x_full @ e.T + e @ x_full @ h.T
        r_expected = alpha * r_sym + beta * product

        np.testing.assert_allclose(np.tril(r_out), np.tril(r_expected), rtol=1e-13)

    def test_lower_trans_n3(self):
        """
        Test UPLO='L', TRANS='T' with 3x3 matrices.

        R := alpha*R + beta*(H'*X*E + E'*X*H)

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 3
        alpha = 0.0
        beta = 2.0

        r_sym = np.zeros((n, n), order='F', dtype=float)

        h_full = np.random.randn(n, n).astype(float, order='F')
        h = make_upper_hessenberg(h_full)

        x_base = np.random.randn(n, n)
        x_sym = (x_base + x_base.T) / 2.0
        x_sym = np.asfortranarray(x_sym, dtype=float)

        e = make_upper_triangular(np.random.randn(n, n)).astype(float, order='F')

        r_in = r_sym.copy(order='F')
        x_in = x_sym.copy(order='F')
        h_in = h.copy(order='F')

        r_out, info = mb01od('L', 'T', n, alpha, beta, r_in, h_in, x_in, e)

        assert info == 0

        x_full = make_symmetric(x_sym, 'L')
        product = h.T @ x_full @ e + e.T @ x_full @ h
        r_expected = alpha * r_sym + beta * product

        np.testing.assert_allclose(np.tril(r_out), np.tril(r_expected), rtol=1e-13)


class TestMB01ODSpecialCases:
    """Test special cases (alpha=0, beta=0)."""

    def test_alpha_zero(self):
        """
        Test alpha=0 case.

        R := 0*R + beta*(H*X*E' + E*X*H') = beta*(H*X*E' + E*X*H')

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n = 3
        alpha = 0.0
        beta = 1.0

        r_in = np.array([
            [100.0, 200.0, 300.0],
            [0.0, 400.0, 500.0],
            [0.0, 0.0, 600.0]
        ], order='F', dtype=float)

        h = make_upper_hessenberg(np.random.randn(n, n)).astype(float, order='F')
        x = np.eye(n, order='F', dtype=float)
        e = make_upper_triangular(np.random.randn(n, n) + np.eye(n)).astype(float, order='F')

        r_out, info = mb01od('U', 'N', n, alpha, beta, r_in.copy(), h.copy(), x.copy(), e)

        assert info == 0

        product = h @ x @ e.T + e @ x @ h.T
        r_expected = beta * product

        np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-13)

    def test_beta_zero(self):
        """
        Test beta=0 case.

        R := alpha*R + 0*(...) = alpha*R
        """
        n = 3
        alpha = 2.5
        beta = 0.0

        r_in = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 6.0]
        ], order='F', dtype=float)

        h = np.ones((n, n), order='F', dtype=float)
        x = np.ones((n, n), order='F', dtype=float)
        e = np.ones((n, n), order='F', dtype=float)

        r_out, info = mb01od('U', 'N', n, alpha, beta, r_in.copy(), h, x, e)

        assert info == 0

        r_expected = alpha * r_in
        np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)

    def test_alpha_and_beta_zero(self):
        """
        Test alpha=0 and beta=0 case.

        R := 0*R + 0*(...) = 0
        """
        n = 3
        alpha = 0.0
        beta = 0.0

        r_in = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 6.0]
        ], order='F', dtype=float)

        h = np.ones((n, n), order='F', dtype=float)
        x = np.ones((n, n), order='F', dtype=float)
        e = np.ones((n, n), order='F', dtype=float)

        r_out, info = mb01od('U', 'N', n, alpha, beta, r_in.copy(), h, x, e)

        assert info == 0

        np.testing.assert_allclose(np.triu(r_out), 0.0, atol=1e-15)

    def test_n_zero(self):
        """Test n=0 case - should return immediately."""
        n = 0
        alpha = 1.0
        beta = 1.0

        r = np.array([], order='F', dtype=float).reshape(0, 0)
        h = np.array([], order='F', dtype=float).reshape(0, 0)
        x = np.array([], order='F', dtype=float).reshape(0, 0)
        e = np.array([], order='F', dtype=float).reshape(0, 0)

        r_out, info = mb01od('U', 'N', n, alpha, beta, r, h, x, e)

        assert info == 0
        assert r_out.shape == (0, 0)


class TestMB01ODMathematicalProperties:
    """Test mathematical properties - symmetry preservation."""

    def test_result_symmetry_upper(self):
        """
        Test that result is symmetric when UPLO='U'.

        The matrix formula H*X*E' + E*X*H' is symmetric when X is symmetric.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n = 4
        alpha = 1.0
        beta = 1.0

        r_base = np.random.randn(n, n)
        r_sym = (r_base + r_base.T) / 2.0
        r_in = np.asfortranarray(r_sym, dtype=float)

        h = make_upper_hessenberg(np.random.randn(n, n)).astype(float, order='F')

        x_base = np.random.randn(n, n)
        x_sym = (x_base + x_base.T) / 2.0
        x_in = np.asfortranarray(x_sym, dtype=float)

        e = make_upper_triangular(np.random.randn(n, n) + np.eye(n)).astype(float, order='F')

        r_out, info = mb01od('U', 'N', n, alpha, beta, r_in.copy(), h.copy(), x_in.copy(), e)

        assert info == 0

        r_full = make_symmetric(r_out, 'U')
        np.testing.assert_allclose(r_full, r_full.T, rtol=1e-14)

    def test_result_symmetry_lower(self):
        """
        Test that result is symmetric when UPLO='L'.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n = 4
        alpha = 0.5
        beta = 2.0

        r_base = np.random.randn(n, n)
        r_sym = (r_base + r_base.T) / 2.0
        r_in = np.asfortranarray(r_sym, dtype=float)

        h = make_upper_hessenberg(np.random.randn(n, n)).astype(float, order='F')

        x_base = np.random.randn(n, n)
        x_sym = (x_base + x_base.T) / 2.0
        x_in = np.asfortranarray(x_sym, dtype=float)

        e = make_upper_triangular(np.random.randn(n, n) + np.eye(n)).astype(float, order='F')

        r_out, info = mb01od('L', 'T', n, alpha, beta, r_in.copy(), h.copy(), x_in.copy(), e)

        assert info == 0

        r_full = make_symmetric(r_out, 'L')
        np.testing.assert_allclose(r_full, r_full.T, rtol=1e-14)

    def test_formula_consistency_trans_n_vs_t(self):
        """
        Test that TRANS='N' and TRANS='T' give consistent results.

        For TRANS='N': H*X*E' + E*X*H'
        For TRANS='T': H'*X*E + E'*X*H

        These are related but not equal unless H, E have special structure.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n = 3
        alpha = 0.0
        beta = 1.0

        r_n = np.zeros((n, n), order='F', dtype=float)
        r_t = np.zeros((n, n), order='F', dtype=float)

        h = make_upper_hessenberg(np.random.randn(n, n)).astype(float, order='F')
        x_sym = np.eye(n, order='F', dtype=float)
        e = make_upper_triangular(np.random.randn(n, n) + np.eye(n)).astype(float, order='F')

        r_out_n, info_n = mb01od('U', 'N', n, alpha, beta, r_n.copy(), h.copy(), x_sym.copy(), e)
        r_out_t, info_t = mb01od('U', 'T', n, alpha, beta, r_t.copy(), h.copy(), x_sym.copy(), e)

        assert info_n == 0
        assert info_t == 0

        expected_n = h @ x_sym @ e.T + e @ x_sym @ h.T
        expected_t = h.T @ x_sym @ e + e.T @ x_sym @ h

        np.testing.assert_allclose(np.triu(r_out_n), np.triu(expected_n), rtol=1e-13)
        np.testing.assert_allclose(np.triu(r_out_t), np.triu(expected_t), rtol=1e-13)


class TestMB01ODLargerMatrices:
    """Test with larger matrices."""

    def test_n5_upper_notrans(self):
        """
        Test with 5x5 matrices, UPLO='U', TRANS='N'.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n = 5
        alpha = 1.0
        beta = 1.0

        r_base = np.random.randn(n, n)
        r_sym = (r_base + r_base.T) / 2.0
        r_in = np.asfortranarray(r_sym, dtype=float)

        h = make_upper_hessenberg(np.random.randn(n, n)).astype(float, order='F')

        x_base = np.random.randn(n, n)
        x_sym = (x_base + x_base.T) / 2.0
        x_in = np.asfortranarray(x_sym, dtype=float)

        e = make_upper_triangular(np.random.randn(n, n) + np.eye(n)).astype(float, order='F')

        r_out, info = mb01od('U', 'N', n, alpha, beta, r_in.copy(), h.copy(), x_in.copy(), e)

        assert info == 0

        x_full = make_symmetric(x_sym, 'U')
        product = h @ x_full @ e.T + e @ x_full @ h.T
        r_expected = alpha * r_sym + beta * product

        np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-12)


class TestMB01ODErrorHandling:
    """Test error handling for invalid parameters."""

    def test_invalid_uplo(self):
        """Test invalid UPLO parameter."""
        n = 2
        r = np.eye(n, order='F', dtype=float)
        h = np.eye(n, order='F', dtype=float)
        x = np.eye(n, order='F', dtype=float)
        e = np.eye(n, order='F', dtype=float)

        r_out, info = mb01od('X', 'N', n, 1.0, 1.0, r, h, x, e)

        assert info == -1

    def test_invalid_trans(self):
        """Test invalid TRANS parameter."""
        n = 2
        r = np.eye(n, order='F', dtype=float)
        h = np.eye(n, order='F', dtype=float)
        x = np.eye(n, order='F', dtype=float)
        e = np.eye(n, order='F', dtype=float)

        r_out, info = mb01od('U', 'X', n, 1.0, 1.0, r, h, x, e)

        assert info == -2

    def test_negative_n(self):
        """Test negative N parameter."""
        r = np.eye(2, order='F', dtype=float)
        h = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float)
        e = np.eye(2, order='F', dtype=float)

        r_out, info = mb01od('U', 'N', -1, 1.0, 1.0, r, h, x, e)

        assert info == -3
