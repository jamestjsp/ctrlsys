"""
Tests for mb03ah - Compute Givens rotations for real Wilkinson shift polynomial.

MB03AH computes two Givens rotations (C1,S1) and (C2,S2) such that the
orthogonal matrix Q makes the first column of the real Wilkinson double/single
shift polynomial parallel to the first unit vector.

Tests validate:
1. Givens rotation property: C^2 + S^2 = 1
2. C2=1, S2=0 for single shift (SHFT='S') or N=2
3. Numerical consistency with known periodic Hessenberg products
"""

import numpy as np
import pytest
from ctrlsys import mb03ah


class TestMB03AHBasic:
    """Basic functionality tests for mb03ah."""

    def test_single_shift_basic(self):
        """
        Test single shift mode with simple periodic Hessenberg product.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        k = 2  # number of factors
        n = 3  # order of factors

        # Create simple upper triangular matrices for the product
        # A[:,:,0] is triangular, A[:,:,1] is upper Hessenberg
        a = np.zeros((n, n, k), order='F', dtype=float)

        # First factor: upper triangular
        a[:, :, 0] = np.array([
            [2.0, 1.0, 0.5],
            [0.0, 1.5, 0.3],
            [0.0, 0.0, 1.0]
        ], order='F')

        # Second factor: upper Hessenberg (last factor)
        a[:, :, 1] = np.array([
            [1.0, 0.5, 0.2],
            [0.4, 1.2, 0.3],
            [0.0, 0.3, 0.8]
        ], order='F')

        # AMAP: factor 1 at position 0, Hessenberg (factor 2) at position 1
        amap = np.array([1, 2], dtype=np.int32)

        # All signatures = 1
        s = np.array([1, 1], dtype=np.int32)

        sinv = 1

        c1, s1, c2, s2, info = mb03ah('S', k, n, amap, s, sinv, a)

        assert info == 0, f"mb03ah returned info={info}"

        # Property 1: Givens rotation property
        np.testing.assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14,
                                   err_msg="First rotation not normalized")

        # Property 2: Single shift means C2=1, S2=0
        np.testing.assert_allclose(c2, 1.0, rtol=1e-14,
                                   err_msg="C2 should be 1 for single shift")
        np.testing.assert_allclose(s2, 0.0, atol=1e-14,
                                   err_msg="S2 should be 0 for single shift")

    def test_double_shift_basic(self):
        """
        Test double shift mode with N > 2.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        k = 2
        n = 4  # N > 2 required for double shift

        a = np.zeros((n, n, k), order='F', dtype=float)

        # First factor: upper triangular
        a[:, :, 0] = np.triu(np.random.randn(n, n).astype(float, order='F'))
        # Make diagonal positive for numerical stability
        for i in range(n):
            if a[i, i, 0] < 0.1:
                a[i, i, 0] = 1.0

        # Second factor: upper Hessenberg
        a[:, :, 1] = np.triu(np.random.randn(n, n).astype(float, order='F'), k=-1)

        amap = np.array([1, 2], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)
        sinv = 1

        c1, s1, c2, s2, info = mb03ah('D', k, n, amap, s, sinv, a)

        assert info == 0, f"mb03ah returned info={info}"

        # Both rotations should be normalized
        np.testing.assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14,
                                   err_msg="First rotation not normalized")
        np.testing.assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14,
                                   err_msg="Second rotation not normalized")

    def test_n_equals_2_forces_single_shift(self):
        """
        When N=2, even if SHFT='D', routine should behave like single shift.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        k = 2
        n = 2  # N=2 forces single shift behavior

        a = np.zeros((n, n, k), order='F', dtype=float)

        a[:, :, 0] = np.array([
            [2.0, 1.0],
            [0.0, 1.5]
        ], order='F')

        a[:, :, 1] = np.array([
            [1.0, 0.5],
            [0.3, 0.8]
        ], order='F')

        amap = np.array([1, 2], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)
        sinv = 1

        # Request double shift but N=2 should force single shift
        c1, s1, c2, s2, info = mb03ah('D', k, n, amap, s, sinv, a)

        assert info == 0

        # Should behave like single shift
        np.testing.assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
        np.testing.assert_allclose(c2, 1.0, rtol=1e-14,
                                   err_msg="C2 should be 1 when N=2")
        np.testing.assert_allclose(s2, 0.0, atol=1e-14,
                                   err_msg="S2 should be 0 when N=2")


class TestMB03AHSignatures:
    """Tests for different signature configurations."""

    def test_negative_sinv(self):
        """
        Test with negative SINV (reciprocal eigenvalues).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        k = 2
        n = 3

        a = np.zeros((n, n, k), order='F', dtype=float)

        a[:, :, 0] = np.array([
            [2.0, 1.0, 0.5],
            [0.0, 1.5, 0.3],
            [0.0, 0.0, 1.0]
        ], order='F')

        a[:, :, 1] = np.array([
            [1.0, 0.5, 0.2],
            [0.4, 1.2, 0.3],
            [0.0, 0.3, 0.8]
        ], order='F')

        amap = np.array([1, 2], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)
        sinv = -1  # Negative SINV

        c1, s1, c2, s2, info = mb03ah('S', k, n, amap, s, sinv, a)

        assert info == 0

        # Rotations should still be normalized
        np.testing.assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
        np.testing.assert_allclose(c2, 1.0, rtol=1e-14)
        np.testing.assert_allclose(s2, 0.0, atol=1e-14)

    def test_mixed_signatures(self):
        """
        Test with mixed signature array.

        Random seed: 321 (for reproducibility)
        """
        np.random.seed(321)

        k = 3
        n = 4

        a = np.zeros((n, n, k), order='F', dtype=float)

        # Create upper triangular factors
        for i in range(k - 1):
            a[:, :, i] = np.triu(np.eye(n) + 0.5 * np.random.randn(n, n))
            for j in range(n):
                if abs(a[j, j, i]) < 0.5:
                    a[j, j, i] = 1.0

        # Last factor is Hessenberg
        a[:, :, k-1] = np.triu(np.eye(n) + 0.3 * np.random.randn(n, n), k=-1)

        amap = np.array([1, 2, 3], dtype=np.int32)
        s = np.array([1, -1, 1], dtype=np.int32)  # Mixed signatures
        sinv = 1

        c1, s1, c2, s2, info = mb03ah('D', k, n, amap, s, sinv, a)

        assert info == 0

        np.testing.assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
        np.testing.assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


class TestMB03AHMultipleFactors:
    """Tests with various numbers of factors."""

    def test_single_factor(self):
        """
        Test with K=1 (single factor, which is the Hessenberg matrix).

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)

        k = 1
        n = 3

        a = np.zeros((n, n, k), order='F', dtype=float)

        # Single Hessenberg factor
        a[:, :, 0] = np.array([
            [2.0, 1.0, 0.5],
            [0.5, 1.5, 0.3],
            [0.0, 0.4, 1.0]
        ], order='F')

        amap = np.array([1], dtype=np.int32)
        s = np.array([1], dtype=np.int32)
        sinv = 1

        c1, s1, c2, s2, info = mb03ah('S', k, n, amap, s, sinv, a)

        assert info == 0
        np.testing.assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
        np.testing.assert_allclose(c2, 1.0, rtol=1e-14)
        np.testing.assert_allclose(s2, 0.0, atol=1e-14)

    def test_many_factors(self):
        """
        Test with many factors (K=5).

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)

        k = 5
        n = 4

        a = np.zeros((n, n, k), order='F', dtype=float)

        # Create upper triangular factors
        for i in range(k - 1):
            a[:, :, i] = np.triu(np.eye(n) + 0.3 * np.random.randn(n, n))
            for j in range(n):
                if abs(a[j, j, i]) < 0.5:
                    a[j, j, i] = 1.0

        # Last factor is Hessenberg
        a[:, :, k-1] = np.triu(np.eye(n) + 0.3 * np.random.randn(n, n), k=-1)

        amap = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        s = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        sinv = 1

        c1, s1, c2, s2, info = mb03ah('D', k, n, amap, s, sinv, a)

        assert info == 0
        np.testing.assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
        np.testing.assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


class TestMB03AHEdgeCases:
    """Edge case tests."""

    def test_identity_factors(self):
        """
        Test with identity-like factors.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)

        k = 2
        n = 3

        a = np.zeros((n, n, k), order='F', dtype=float)

        # Both factors close to identity
        a[:, :, 0] = np.eye(n, dtype=float, order='F')
        a[:, :, 1] = np.eye(n, dtype=float, order='F')
        a[1, 0, 1] = 0.1  # Small subdiagonal for Hessenberg

        amap = np.array([1, 2], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)
        sinv = 1

        c1, s1, c2, s2, info = mb03ah('S', k, n, amap, s, sinv, a)

        assert info == 0
        np.testing.assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)

    def test_large_n(self):
        """
        Test with larger matrix order (N=10).

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)

        k = 2
        n = 10

        a = np.zeros((n, n, k), order='F', dtype=float)

        # Upper triangular first factor
        a[:, :, 0] = np.triu(np.eye(n) + 0.2 * np.random.randn(n, n))
        for i in range(n):
            if abs(a[i, i, 0]) < 0.5:
                a[i, i, 0] = 1.0

        # Hessenberg second factor
        a[:, :, 1] = np.triu(np.eye(n) + 0.2 * np.random.randn(n, n), k=-1)

        amap = np.array([1, 2], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)
        sinv = 1

        c1, s1, c2, s2, info = mb03ah('D', k, n, amap, s, sinv, a)

        assert info == 0
        np.testing.assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
        np.testing.assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
