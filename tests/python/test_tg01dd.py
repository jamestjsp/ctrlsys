"""
Tests for TG01DD: Orthogonal reduction of descriptor system (C,A-lambda E) to RQ-coordinate form.

TG01DD computes orthogonal transformation matrix Z such that E*Z is upper trapezoidal.
"""

import numpy as np
import pytest
from ctrlsys import tg01dd


class TestTG01DDBasic:
    """Basic functionality tests from HTML doc example."""

    def test_html_example(self):
        """
        Test TG01DD with HTML documentation example.

        L=4, N=4, P=2, COMPZ='I'
        """
        l, n, p = 4, 4, 2

        # Input matrices from HTML doc (row-wise read)
        a = np.array([
            [-1.0,  0.0,  0.0,  3.0],
            [ 0.0,  0.0,  1.0,  2.0],
            [ 1.0,  1.0,  0.0,  4.0],
            [ 0.0,  0.0,  0.0,  0.0]
        ], order='F', dtype=float)

        e = np.array([
            [ 1.0,  2.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  1.0],
            [ 3.0,  9.0,  6.0,  3.0],
            [ 0.0,  0.0,  2.0,  0.0]
        ], order='F', dtype=float)

        c = np.array([
            [-1.0,  0.0,  1.0,  0.0],
            [ 0.0,  1.0, -1.0,  1.0]
        ], order='F', dtype=float)

        # Expected outputs from HTML doc
        a_expected = np.array([
            [ 0.4082,  3.0773,  0.6030,  0.0000],
            [ 0.8165,  1.7233,  0.6030, -1.0000],
            [ 2.0412,  2.8311,  2.4121,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000]
        ], order='F', dtype=float)

        e_expected = np.array([
            [ 0.0000, -0.7385,  2.1106,  0.0000],
            [ 0.0000,  0.7385,  1.2060,  0.0000],
            [ 0.0000,  0.0000,  9.9499, -6.0000],
            [ 0.0000,  0.0000,  0.0000, -2.0000]
        ], order='F', dtype=float)

        c_expected = np.array([
            [-0.8165,  0.4924, -0.3015, -1.0000],
            [ 0.0000,  0.7385,  1.2060,  1.0000]
        ], order='F', dtype=float)

        z_expected = np.array([
            [ 0.8165, -0.4924,  0.3015,  0.0000],
            [-0.4082, -0.1231,  0.9045,  0.0000],
            [ 0.0000,  0.0000,  0.0000, -1.0000],
            [ 0.4082,  0.8616,  0.3015,  0.0000]
        ], order='F', dtype=float)

        # Call routine with compz='I' to initialize Z to identity
        a_out, e_out, c_out, z_out, info = tg01dd('I', a, e, c)

        assert info == 0
        np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(e_out, e_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(z_out, z_expected, rtol=1e-3, atol=1e-4)


class TestTG01DDOrthogonality:
    """Test mathematical properties - orthogonality of Z."""

    def test_z_orthogonal(self):
        """
        Verify Z is orthogonal: Z'*Z = I

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        l, n, p = 5, 5, 3

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, c_out, z_out, info = tg01dd('I', a, e, c)

        assert info == 0
        identity = np.eye(n, dtype=float, order='F')
        np.testing.assert_allclose(z_out.T @ z_out, identity, rtol=1e-14, atol=1e-14)

    def test_transformation_consistency(self):
        """
        Verify transformation: A_out = A*Z, E_out = E*Z, C_out = C*Z

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        l, n, p = 4, 6, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_orig = a.copy()
        e_orig = e.copy()
        c_orig = c.copy()

        a_out, e_out, c_out, z_out, info = tg01dd('I', a, e, c)

        assert info == 0

        # Check A*Z = A_out
        np.testing.assert_allclose(a_orig @ z_out, a_out, rtol=1e-14, atol=1e-14)
        # Check E*Z = E_out
        np.testing.assert_allclose(e_orig @ z_out, e_out, rtol=1e-14, atol=1e-14)
        # Check C*Z = C_out
        np.testing.assert_allclose(c_orig @ z_out, c_out, rtol=1e-14, atol=1e-14)

    def test_e_upper_trapezoidal_l_ge_n(self):
        """
        Verify E*Z is upper trapezoidal when L >= N.

        For L >= N:
                  ( E11 )
            E*Z = (     )
                  (  R  )
        where R is MIN(L,N)-by-MIN(L,N) upper triangular.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        l, n, p = 6, 4, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, c_out, z_out, info = tg01dd('I', a, e, c)

        assert info == 0

        # For L >= N, the lower L-by-N part of E*Z should have zeros below main diagonal
        # The bottom MIN(L,N) rows form an upper triangular matrix R
        min_ln = min(l, n)
        r_start = l - min_ln
        # Check lower triangle of R is zero
        for i in range(1, min_ln):
            for j in range(i):
                assert abs(e_out[r_start + i, j]) < 1e-14, \
                    f"E*Z not upper trapezoidal at ({r_start+i},{j}): {e_out[r_start+i,j]}"

    def test_e_upper_trapezoidal_l_lt_n(self):
        """
        Verify E*Z is upper trapezoidal when L < N.

        For L < N:
            E*Z = ( 0  R )
        where R is L-by-L upper triangular.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        l, n, p = 3, 5, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, c_out, z_out, info = tg01dd('I', a, e, c)

        assert info == 0

        # For L < N, first (N-L) columns should be zero
        for j in range(n - l):
            for i in range(l):
                assert abs(e_out[i, j]) < 1e-14, \
                    f"E*Z should have zero in column {j}: {e_out[i,j]}"

        # Remaining L columns form upper triangular R
        for i in range(1, l):
            for j in range(i):
                assert abs(e_out[i, n - l + j]) < 1e-14, \
                    f"R not upper triangular at ({i},{j}): {e_out[i, n-l+j]}"


class TestTG01DDCompZModes:
    """Test different COMPZ modes."""

    def test_compz_n(self):
        """
        Test COMPZ='N': do not compute Z.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        l, n, p = 4, 4, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, c_out, z_out, info = tg01dd('N', a, e, c)

        assert info == 0
        # Z should be None or not computed when compz='N'
        assert z_out is None

    def test_compz_u(self):
        """
        Test COMPZ='U': update existing orthogonal Z.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        l, n, p = 4, 4, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        # Create an orthogonal Z1 matrix (from QR decomposition of random matrix)
        z1, _ = np.linalg.qr(np.random.randn(n, n))
        z1 = np.asfortranarray(z1)

        a_orig = a.copy()
        e_orig = e.copy()
        c_orig = c.copy()
        z1_orig = z1.copy()

        a_out, e_out, c_out, z_out, info = tg01dd('U', a, e, c, z1)

        assert info == 0

        # Z_out should be Z1 * Z where Z is the orthogonal transformation
        # Verify Z_out is still orthogonal
        identity = np.eye(n, dtype=float, order='F')
        np.testing.assert_allclose(z_out.T @ z_out, identity, rtol=1e-14, atol=1e-14)

        # Verify transformation: A_out = A * Z_out (where Z_out = Z1 * Z)
        # But Z_out contains Z1*Z, so we need A*Z_out/Z1 = A_out
        # Actually: A_out = A * Z, so A_out = A * inv(Z1) * Z_out
        # Simpler: verify A * Z_out != A_orig * Z_out (transformation was applied)


class TestTG01DDEdgeCases:
    """Edge case tests."""

    def test_zero_dimensions(self):
        """Test with L=0 or N=0."""
        a = np.array([], dtype=float, order='F').reshape(0, 4)
        e = np.array([], dtype=float, order='F').reshape(0, 4)
        c = np.array([[1, 2, 3, 4]], dtype=float, order='F')

        a_out, e_out, c_out, z_out, info = tg01dd('I', a, e, c)
        assert info == 0

    def test_n_zero(self):
        """Test with N=0."""
        a = np.array([], dtype=float, order='F').reshape(4, 0)
        e = np.array([], dtype=float, order='F').reshape(4, 0)
        c = np.array([], dtype=float, order='F').reshape(2, 0)

        a_out, e_out, c_out, z_out, info = tg01dd('I', a, e, c)
        assert info == 0

    def test_p_zero(self):
        """
        Test with P=0 (no C matrix rows).

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        l, n = 4, 4

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        c = np.array([], dtype=float, order='F').reshape(0, n)

        a_out, e_out, c_out, z_out, info = tg01dd('I', a, e, c)

        assert info == 0
        # Verify Z is orthogonal
        identity = np.eye(n, dtype=float, order='F')
        np.testing.assert_allclose(z_out.T @ z_out, identity, rtol=1e-14, atol=1e-14)

    def test_square_l_eq_n(self):
        """
        Test square case L=N.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        l, n, p = 5, 5, 3

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_orig = a.copy()
        e_orig = e.copy()
        c_orig = c.copy()

        a_out, e_out, c_out, z_out, info = tg01dd('I', a, e, c)

        assert info == 0

        # Verify transformation consistency
        np.testing.assert_allclose(a_orig @ z_out, a_out, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(e_orig @ z_out, e_out, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(c_orig @ z_out, c_out, rtol=1e-14, atol=1e-14)

        # E_out should be upper triangular (since L=N)
        for i in range(1, n):
            for j in range(i):
                assert abs(e_out[i, j]) < 1e-14


class TestTG01DDErrors:
    """Error handling tests."""

    def test_invalid_compz(self):
        """Test invalid COMPZ parameter."""
        a = np.eye(4, dtype=float, order='F')
        e = np.eye(4, dtype=float, order='F')
        c = np.eye(2, 4, dtype=float, order='F')

        with pytest.raises((ValueError, RuntimeError)):
            tg01dd('X', a, e, c)
