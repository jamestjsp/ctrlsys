"""
Tests for SB04OW - Solving periodic Sylvester equation with matrices in periodic Schur form.

Solves:
    A * R - L * B = scale * C
    D * L - R * E = scale * F

where (A, D), (B, E) are in periodic Schur form (A, B upper quasi-triangular,
D, E upper triangular).
"""

import numpy as np
import pytest
from ctrlsys import sb04ow


def verify_periodic_sylvester(a, b, c_in, d, e, f_in, r_out, l_out, scale):
    """
    Verify the periodic Sylvester equations:
        A * R - L * B = scale * C
        D * L - R * E = scale * F
    """
    residual1 = a @ r_out - l_out @ b - scale * c_in
    residual2 = d @ l_out - r_out @ e - scale * f_in
    return np.linalg.norm(residual1, 'fro'), np.linalg.norm(residual2, 'fro')


class TestSB04OW:
    """Tests for sb04ow periodic Sylvester solver."""

    def test_basic_1x1(self):
        """
        Test 1x1 case (M=1, N=1).

        Equations:
            A[0,0]*R - L*B[0,0] = scale*C
            D[0,0]*L - R*E[0,0] = scale*F

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        m, n = 1, 1
        a = np.array([[2.0]], order='F', dtype=float)
        b = np.array([[3.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[4.0]], order='F', dtype=float)
        e = np.array([[5.0]], order='F', dtype=float)
        f = np.array([[2.0]], order='F', dtype=float)

        c_in = c.copy()
        f_in = f.copy()

        r, l, scale, info = sb04ow(m, n, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        res1, res2 = verify_periodic_sylvester(a, b, c_in, d, e, f_in, r, l, scale)
        np.testing.assert_allclose(res1, 0.0, atol=1e-14)
        np.testing.assert_allclose(res2, 0.0, atol=1e-14)

    def test_basic_2x2_diagonal(self):
        """
        Test 2x2 case with diagonal blocks (no 2x2 eigenvalue blocks).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        m, n = 2, 2
        a = np.array([[1.0, 0.5],
                      [0.0, 2.0]], order='F', dtype=float)
        b = np.array([[3.0, 0.3],
                      [0.0, 4.0]], order='F', dtype=float)
        c = np.array([[1.0, 2.0],
                      [3.0, 4.0]], order='F', dtype=float)
        d = np.array([[5.0, 0.2],
                      [0.0, 6.0]], order='F', dtype=float)
        e = np.array([[7.0, 0.1],
                      [0.0, 8.0]], order='F', dtype=float)
        f = np.array([[5.0, 6.0],
                      [7.0, 8.0]], order='F', dtype=float)

        c_in = c.copy()
        f_in = f.copy()

        r, l, scale, info = sb04ow(m, n, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        res1, res2 = verify_periodic_sylvester(a, b, c_in, d, e, f_in, r, l, scale)
        np.testing.assert_allclose(res1, 0.0, atol=1e-13)
        np.testing.assert_allclose(res2, 0.0, atol=1e-13)

    def test_with_2x2_block_in_a(self):
        """
        Test with 2x2 eigenvalue block in A (complex conjugate eigenvalues).

        A is quasi-triangular with a 2x2 block (A[1,0] != 0).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        m, n = 2, 2
        a = np.array([[1.0, 0.5],
                      [0.3, 2.0]], order='F', dtype=float)
        b = np.array([[3.0, 0.3],
                      [0.0, 4.0]], order='F', dtype=float)
        c = np.array([[1.0, 2.0],
                      [3.0, 4.0]], order='F', dtype=float)
        d = np.array([[5.0, 0.2],
                      [0.0, 6.0]], order='F', dtype=float)
        e = np.array([[7.0, 0.1],
                      [0.0, 8.0]], order='F', dtype=float)
        f = np.array([[5.0, 6.0],
                      [7.0, 8.0]], order='F', dtype=float)

        c_in = c.copy()
        f_in = f.copy()

        r, l, scale, info = sb04ow(m, n, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        res1, res2 = verify_periodic_sylvester(a, b, c_in, d, e, f_in, r, l, scale)
        np.testing.assert_allclose(res1, 0.0, atol=1e-13)
        np.testing.assert_allclose(res2, 0.0, atol=1e-13)

    def test_with_2x2_block_in_b(self):
        """
        Test with 2x2 eigenvalue block in B (complex conjugate eigenvalues).

        B is quasi-triangular with a 2x2 block (B[1,0] != 0).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        m, n = 2, 2
        a = np.array([[1.0, 0.5],
                      [0.0, 2.0]], order='F', dtype=float)
        b = np.array([[3.0, 0.3],
                      [0.4, 4.0]], order='F', dtype=float)
        c = np.array([[1.0, 2.0],
                      [3.0, 4.0]], order='F', dtype=float)
        d = np.array([[5.0, 0.2],
                      [0.0, 6.0]], order='F', dtype=float)
        e = np.array([[7.0, 0.1],
                      [0.0, 8.0]], order='F', dtype=float)
        f = np.array([[5.0, 6.0],
                      [7.0, 8.0]], order='F', dtype=float)

        c_in = c.copy()
        f_in = f.copy()

        r, l, scale, info = sb04ow(m, n, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        res1, res2 = verify_periodic_sylvester(a, b, c_in, d, e, f_in, r, l, scale)
        np.testing.assert_allclose(res1, 0.0, atol=1e-13)
        np.testing.assert_allclose(res2, 0.0, atol=1e-13)

    def test_larger_system(self):
        """
        Test larger system with mixed block structure.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)

        m, n = 4, 3
        a = np.array([
            [1.0, 0.3, 0.1, 0.05],
            [0.2, 2.0, 0.2, 0.1],
            [0.0, 0.0, 3.0, 0.4],
            [0.0, 0.0, 0.0, 4.0]
        ], order='F', dtype=float)
        b = np.array([
            [2.0, 0.3, 0.1],
            [0.0, 3.0, 0.2],
            [0.0, 0.0, 4.0]
        ], order='F', dtype=float)
        c = np.random.randn(m, n).astype(float, order='F')
        d = np.array([
            [5.0, 0.1, 0.05, 0.02],
            [0.0, 6.0, 0.1, 0.05],
            [0.0, 0.0, 7.0, 0.2],
            [0.0, 0.0, 0.0, 8.0]
        ], order='F', dtype=float)
        e = np.array([
            [3.0, 0.2, 0.1],
            [0.0, 4.0, 0.15],
            [0.0, 0.0, 5.0]
        ], order='F', dtype=float)
        f = np.random.randn(m, n).astype(float, order='F')

        c_in = c.copy()
        f_in = f.copy()

        r, l, scale, info = sb04ow(m, n, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        res1, res2 = verify_periodic_sylvester(a, b, c_in, d, e, f_in, r, l, scale)
        np.testing.assert_allclose(res1, 0.0, atol=1e-12)
        np.testing.assert_allclose(res2, 0.0, atol=1e-12)

    def test_rectangular_m_gt_n(self):
        """
        Test rectangular case M > N.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)

        m, n = 3, 2
        a = np.array([
            [1.0, 0.3, 0.1],
            [0.0, 2.0, 0.2],
            [0.0, 0.0, 3.0]
        ], order='F', dtype=float)
        b = np.array([
            [2.0, 0.4],
            [0.0, 3.0]
        ], order='F', dtype=float)
        c = np.random.randn(m, n).astype(float, order='F')
        d = np.array([
            [4.0, 0.1, 0.05],
            [0.0, 5.0, 0.1],
            [0.0, 0.0, 6.0]
        ], order='F', dtype=float)
        e = np.array([
            [3.0, 0.2],
            [0.0, 4.0]
        ], order='F', dtype=float)
        f = np.random.randn(m, n).astype(float, order='F')

        c_in = c.copy()
        f_in = f.copy()

        r, l, scale, info = sb04ow(m, n, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        res1, res2 = verify_periodic_sylvester(a, b, c_in, d, e, f_in, r, l, scale)
        np.testing.assert_allclose(res1, 0.0, atol=1e-13)
        np.testing.assert_allclose(res2, 0.0, atol=1e-13)

    def test_rectangular_m_lt_n(self):
        """
        Test rectangular case M < N.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)

        m, n = 2, 3
        a = np.array([
            [1.0, 0.3],
            [0.0, 2.0]
        ], order='F', dtype=float)
        b = np.array([
            [2.0, 0.4, 0.1],
            [0.0, 3.0, 0.2],
            [0.0, 0.0, 4.0]
        ], order='F', dtype=float)
        c = np.random.randn(m, n).astype(float, order='F')
        d = np.array([
            [4.0, 0.1],
            [0.0, 5.0]
        ], order='F', dtype=float)
        e = np.array([
            [3.0, 0.2, 0.1],
            [0.0, 4.0, 0.15],
            [0.0, 0.0, 5.0]
        ], order='F', dtype=float)
        f = np.random.randn(m, n).astype(float, order='F')

        c_in = c.copy()
        f_in = f.copy()

        r, l, scale, info = sb04ow(m, n, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        res1, res2 = verify_periodic_sylvester(a, b, c_in, d, e, f_in, r, l, scale)
        np.testing.assert_allclose(res1, 0.0, atol=1e-13)
        np.testing.assert_allclose(res2, 0.0, atol=1e-13)

    def test_equation_residuals_are_machine_precision(self):
        """
        Mathematical property test: solution residuals at machine precision.

        Uses well-separated eigenvalues to avoid common eigenvalue issues.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)

        m, n = 3, 3
        a = np.array([
            [1.0, 0.5, 0.1],
            [0.0, 2.0, 0.3],
            [0.0, 0.0, 3.0]
        ], order='F', dtype=float)
        b = np.array([
            [4.0, 0.4, 0.1],
            [0.0, 5.0, 0.2],
            [0.0, 0.0, 6.0]
        ], order='F', dtype=float)
        c = np.random.randn(m, n).astype(float, order='F')
        d = np.array([
            [7.0, 0.1, 0.05],
            [0.0, 8.0, 0.1],
            [0.0, 0.0, 9.0]
        ], order='F', dtype=float)
        e = np.array([
            [10.0, 0.2, 0.1],
            [0.0, 11.0, 0.15],
            [0.0, 0.0, 12.0]
        ], order='F', dtype=float)
        f = np.random.randn(m, n).astype(float, order='F')

        c_in = c.copy()
        f_in = f.copy()

        r, l, scale, info = sb04ow(m, n, a, b, c, d, e, f)

        assert info == 0

        eq1_lhs = a @ r - l @ b
        eq1_rhs = scale * c_in
        eq2_lhs = d @ l - r @ e
        eq2_rhs = scale * f_in

        np.testing.assert_allclose(eq1_lhs, eq1_rhs, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(eq2_lhs, eq2_rhs, rtol=1e-14, atol=1e-14)


class TestSB04OWErrors:
    """Tests for sb04ow error handling."""

    def test_invalid_m(self):
        """Test error when M <= 0."""
        m, n = 0, 2
        a = np.array([[1.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        c = np.zeros((1, 2), order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)
        e = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        f = np.zeros((1, 2), order='F', dtype=float)

        with pytest.raises(ValueError):
            sb04ow(m, n, a, b, c, d, e, f)

    def test_invalid_n(self):
        """Test error when N <= 0."""
        m, n = 2, 0
        a = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.zeros((2, 1), order='F', dtype=float)
        d = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        e = np.array([[1.0]], order='F', dtype=float)
        f = np.zeros((2, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            sb04ow(m, n, a, b, c, d, e, f)
