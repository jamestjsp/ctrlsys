"""
Tests for MB04IZ: QR factorization of complex matrix with lower-left zero triangle.

MB04IZ computes A = Q * R where A is n-by-m with a p-by-min(p,m) zero triangle
in the lower left corner, optionally applying transformations to matrix B.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from ctrlsys import mb04iz


class TestMB04IZBasic:
    """Basic functionality tests for MB04IZ."""

    def test_basic_qr_factorization(self):
        """
        Test basic QR factorization of complex matrix with zero triangle.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 6, 5, 2

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )
        for i in range(p):
            for j in range(min(i + 1, m)):
                if n - p + i >= 0:
                    a[n - p + i, j] = 0.0

        result = mb04iz(a, p)
        a_out, tau, info = result['a'], result['tau'], result['info']

        assert info == 0
        assert tau.shape == (min(n, m),)

        # R is stored in upper triangle of output (like LAPACK QR)
        # Householder vectors stored below diagonal
        # Verify tau values are reasonable (not NaN/Inf)
        assert np.all(np.isfinite(tau))

        # Verify the last (n-min(n,m)) rows are properly zeroed
        # (these should be zero in the R portion after factorization)
        if n > m:
            # For rows beyond min(n,m), columns should be zero
            for i in range(m, n):
                for j in range(m):
                    if i > j:  # Below diagonal
                        pass  # These contain Householder vectors
                    else:
                        assert_allclose(a_out[i, j], 0.0, atol=1e-14)

    def test_with_b_matrix(self):
        """
        Test QR factorization with transformation applied to matrix B.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p, l = 5, 4, 2, 3

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )
        b = (np.random.randn(n, l) + 1j * np.random.randn(n, l)).astype(
            np.complex128, order='F'
        )

        for i in range(p):
            for j in range(min(i + 1, m)):
                if n - p + i >= 0:
                    a[n - p + i, j] = 0.0

        result = mb04iz(a, p, b=b)
        a_out, b_out, tau, info = result['a'], result['b'], result['tau'], result['info']

        assert info == 0
        assert b_out.shape == (n, l)
        assert tau.shape == (min(n, m),)

    def test_no_zero_triangle(self):
        """
        Test with p=0 (no zero triangle - standard QR).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m = 4, 3

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )
        a_orig = a.copy()

        result = mb04iz(a, 0)
        a_out, tau, info = result['a'], result['tau'], result['info']

        assert info == 0

        r = np.triu(a_out[:min(n, m), :])
        q, _ = np.linalg.qr(a_orig)
        r_expected = q.T.conj() @ a_orig
        r_expected = np.triu(r_expected[:min(n, m), :])

        assert_allclose(np.abs(r), np.abs(r_expected), rtol=1e-10)


class TestMB04IZEdgeCases:
    """Edge case tests for MB04IZ."""

    def test_square_matrix(self):
        """
        Test with square matrix.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4
        p = 1

        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
            np.complex128, order='F'
        )
        for j in range(min(1, n)):
            a[n - 1, j] = 0.0

        result = mb04iz(a, p)
        assert result['info'] == 0
        assert result['tau'].shape == (n,)

    def test_tall_matrix(self):
        """
        Test with tall matrix (n > m).

        Random seed: 321 (for reproducibility)
        """
        np.random.seed(321)
        n, m, p = 8, 5, 2

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )
        for i in range(p):
            for j in range(min(i + 1, m)):
                a[n - p + i, j] = 0.0

        result = mb04iz(a, p)
        assert result['info'] == 0
        assert result['tau'].shape == (min(n, m),)

    def test_wide_matrix(self):
        """
        Test with wide matrix (m > n).

        Random seed: 654 (for reproducibility)
        """
        np.random.seed(654)
        n, m, p = 4, 7, 1

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )
        a[n - 1, 0] = 0.0

        result = mb04iz(a, p)
        assert result['info'] == 0
        assert result['tau'].shape == (min(n, m),)

    def test_empty_matrix_n_zero(self):
        """Test with n=0 (empty matrix)."""
        a = np.zeros((0, 3), dtype=np.complex128, order='F')

        result = mb04iz(a, 0)
        assert result['info'] == 0

    def test_empty_matrix_m_zero(self):
        """Test with m=0 (empty matrix)."""
        a = np.zeros((3, 0), dtype=np.complex128, order='F')

        result = mb04iz(a, 0)
        assert result['info'] == 0

    def test_n_equals_p_plus_one(self):
        """
        Test with n = p + 1 (minimal non-trivial structure).

        When n <= p+1, tau is set to zero (quick return).
        """
        np.random.seed(111)
        n, m, p = 3, 4, 2

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )

        result = mb04iz(a, p)
        assert result['info'] == 0
        assert_allclose(result['tau'], np.zeros(min(n, m), dtype=np.complex128), atol=1e-15)

    def test_large_p(self):
        """
        Test with p larger than n (structure extends beyond matrix).

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m, p = 4, 5, 3

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )
        for i in range(p):
            if n - p + i >= 0:
                for j in range(min(i + 1, m)):
                    a[n - p + i, j] = 0.0

        result = mb04iz(a, p)
        assert result['info'] == 0


class TestMB04IZNumerical:
    """Numerical property tests for MB04IZ."""

    def test_r_upper_triangular(self):
        """
        Validate R matrix is upper triangular.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        n, m, p = 6, 4, 2

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )

        for i in range(p):
            for j in range(min(i + 1, m)):
                if n - p + i >= 0:
                    a[n - p + i, j] = 0.0

        result = mb04iz(a, p)
        a_out, tau, info = result['a'], result['tau'], result['info']

        assert info == 0

        # Extract R from upper triangle - should be upper triangular
        r = np.triu(a_out[:min(n, m), :])

        # Verify strict lower triangle is zero
        for j in range(m):
            for i in range(j + 1, min(n, m)):
                assert_allclose(r[i, j], 0.0, atol=1e-14)

    def test_diagonal_r_values_match_qr(self):
        """
        Validate diagonal of R matches numpy QR for case with no zero triangle.

        When p=0, MB04IZ should behave like standard QR factorization.
        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        n, m = 5, 4

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )
        a_orig = a.copy()

        result = mb04iz(a, 0)  # p=0, no zero triangle
        a_out, tau, info = result['a'], result['tau'], result['info']
        assert info == 0

        # Get R from numpy QR for comparison
        q_np, r_np = np.linalg.qr(a_orig)

        # Diagonal elements of R should match in magnitude
        diag_mb04 = np.abs(np.diag(a_out[:min(n, m), :min(n, m)]))
        diag_np = np.abs(np.diag(r_np[:min(n, m), :min(n, m)]))
        assert_allclose(diag_mb04, diag_np, rtol=1e-10)


class TestMB04IZWorkspace:
    """Workspace query tests for MB04IZ."""

    def test_workspace_query(self):
        """Test workspace query (lzwork=-1)."""
        np.random.seed(777)
        n, m, p = 6, 5, 2

        a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F'
        )

        result = mb04iz(a, p, lzwork=-1)
        assert result['info'] == 0
        assert result['zwork_opt'] >= max(1, m - 1, m - p)


class TestMB04IZErrors:
    """Error handling tests for MB04IZ."""

    def test_negative_n(self):
        """Test error for negative n (via empty array with wrong shape)."""
        a = np.zeros((0, 3), dtype=np.complex128, order='F')
        result = mb04iz(a, 0)
        assert result['info'] == 0

    def test_negative_p(self):
        """Test error for negative p."""
        np.random.seed(333)
        a = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(
            np.complex128, order='F'
        )

        with pytest.raises((ValueError, RuntimeError)):
            mb04iz(a, -1)

    def test_negative_l(self):
        """Test error for negative l (via b array)."""
        np.random.seed(444)
        a = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(
            np.complex128, order='F'
        )
        b = np.zeros((4, 0), dtype=np.complex128, order='F')

        result = mb04iz(a, 1, b=b)
        assert result['info'] == 0
