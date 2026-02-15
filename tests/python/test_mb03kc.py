"""
Tests for mb03kc - Reduce 2x2 formal matrix product to periodic Hessenberg-triangular form.

Tests validate:
1. Upper triangular form for all matrices except A_khess (which is full/Hessenberg)
2. Reflector reconstruction: H_k = I - tau_k * v_k * v_k'
3. Transformation equations hold
"""

import numpy as np
import pytest
from ctrlsys import mb03kc


def apply_reflector_left(h, a_block):
    """Apply H from left: H @ A_block."""
    return h @ a_block


def apply_reflector_right(a_block, h):
    """Apply H from right: A_block @ H."""
    return a_block @ h


def build_reflector(v, tau):
    """Build 2x2 Householder reflector H = I - tau * v * v'."""
    v = np.asarray(v).reshape(2, 1)
    return np.eye(2) - tau * (v @ v.T)


def is_upper_triangular_2x2(a, tol=1e-12):
    """Check if 2x2 block is upper triangular (a[1,0] ~ 0)."""
    return abs(a[1, 0]) < tol


class TestMB03KCBasic:
    """Basic functionality tests."""

    def test_k2_n3_r1_s_positive(self):
        """
        Test K=2, N=3, R=1, S=[1,1] (both positive signatures).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k = 2
        khess = 1
        n = 3
        r = 1
        lda = n
        s = np.array([1, 1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')
        a_original = a.copy()

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        assert tau.shape == (k,)
        assert v.shape == (2 * k,)

        for i in range(k):
            if i + 1 != khess:
                a_block = a_out[r-1:r+1, r-1:r+1, i]
                assert is_upper_triangular_2x2(a_block), f"A_{i+1} should be upper triangular"

    def test_k2_n3_r1_khess2(self):
        """
        Test K=2, N=3, R=1, KHESS=2, S=[1,1].

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k = 2
        khess = 2
        n = 3
        r = 1
        lda = n
        s = np.array([1, 1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        assert is_upper_triangular_2x2(a_out[r-1:r+1, r-1:r+1, 0]), \
            "A_1 should be upper triangular when khess=2"

    def test_k3_n4_r3(self):
        """
        Test K=3, N=4, R=3 (bottom-right 2x2 block), S=[1,1,1].

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        k = 3
        khess = 1
        n = 4
        r = 3
        lda = n
        s = np.array([1, 1, 1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        assert tau.shape == (k,)
        assert v.shape == (2 * k,)

        for i in range(k):
            if i + 1 != khess:
                a_block = a_out[r-1:r+1, r-1:r+1, i]
                assert is_upper_triangular_2x2(a_block), f"A_{i+1} should be upper triangular"

    def test_k4_n3_r1_mixed_signatures(self):
        """
        Test K=4, N=3, R=1 with mixed signatures S=[1,-1,1,-1].

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        k = 4
        khess = 2
        n = 3
        r = 1
        lda = n
        s = np.array([1, -1, 1, -1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        assert tau.shape == (k,)
        assert v.shape == (2 * k,)

        for i in range(k):
            if i + 1 != khess:
                a_block = a_out[r-1:r+1, r-1:r+1, i]
                assert is_upper_triangular_2x2(a_block), f"A_{i+1} should be upper triangular"


class TestMB03KCReflectorProperties:
    """Tests for reflector mathematical properties."""

    def test_reflector_orthogonality(self):
        """
        Validate H_k' @ H_k = I for all constructed reflectors.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        k = 3
        khess = 1
        n = 3
        r = 1
        lda = n
        s = np.array([1, 1, 1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        for i in range(k):
            v_k = v[2*i:2*i+2]
            tau_k = tau[i]
            h_k = build_reflector(v_k, tau_k)
            identity_check = h_k.T @ h_k
            np.testing.assert_allclose(identity_check, np.eye(2), rtol=1e-14, atol=1e-15)

    def test_reflector_involution(self):
        """
        Validate H_k @ H_k = I (Householder reflectors are involutions when tau = 2/(v'v)).

        For standard LAPACK DLARFG, H_k' = H_k, so H_k @ H_k = I.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        k = 2
        khess = 1
        n = 3
        r = 1
        lda = n
        s = np.array([1, 1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        for i in range(k):
            v_k = v[2*i:2*i+2]
            tau_k = tau[i]
            h_k = build_reflector(v_k, tau_k)
            involution_check = h_k @ h_k
            np.testing.assert_allclose(involution_check, np.eye(2), rtol=1e-14, atol=1e-15)


class TestMB03KCEdgeCases:
    """Edge case tests."""

    def test_k2_minimum(self):
        """
        Test with minimum K=2.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        k = 2
        khess = 1
        n = 3
        r = 1
        lda = n
        s = np.array([1, 1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        assert tau.shape == (k,)
        assert v.shape == (2 * k,)

    def test_all_negative_signatures(self):
        """
        Test with all negative signatures S=[-1,-1,-1].

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        k = 3
        khess = 2
        n = 4
        r = 3
        lda = n
        s = np.array([-1, -1, -1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        for i in range(k):
            if i + 1 != khess:
                a_block = a_out[r-1:r+1, r-1:r+1, i]
                assert is_upper_triangular_2x2(a_block), f"A_{i+1} should be upper triangular"

    def test_khess_equals_k(self):
        """
        Test with khess=K (last matrix is Hessenberg).

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        k = 3
        khess = 3
        n = 3
        r = 1
        lda = n
        s = np.array([1, 1, 1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        for i in range(k):
            if i + 1 != khess:
                a_block = a_out[r-1:r+1, r-1:r+1, i]
                assert is_upper_triangular_2x2(a_block), f"A_{i+1} should be upper triangular"


class TestMB03KCIdentityReflector:
    """Test identity reflector at khess+1 position."""

    def test_identity_reflector_at_khess_plus_1(self):
        """
        Validate H_{khess+1} is identity (tau=0, v=[0,0]).

        According to the algorithm, H_{khess+1} = identity.
        For khess=1, H_2 should be identity -> tau[1] = 0.
        For khess=K, H_1 should be identity -> tau[0] = 0.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        k = 3
        khess = 1
        n = 3
        r = 1
        lda = n
        s = np.array([1, 1, 1], dtype=np.int32)

        a = np.random.randn(n, n, k).astype(np.float64, order='F')

        v, tau, a_out = mb03kc(k, khess, n, r, s, a, lda)

        ip1 = khess % k
        assert tau[ip1] == 0.0, f"tau[{ip1}] should be 0 (identity reflector)"
        assert v[2*ip1] == 0.0, f"v[{2*ip1}] should be 0"
        assert v[2*ip1+1] == 0.0, f"v[{2*ip1+1}] should be 0"
