"""
Tests for mb03bc - Product singular value decomposition of 2x2 triangular factors.

MB03BC computes Givens rotations so that the product of K 2x2 upper triangular
matrices is diagonalized.
"""

import numpy as np
import pytest
from ctrlsys import mb03bc


def get_macpar():
    """Get machine parameters for DLAMCH."""
    import sys
    macpar = np.array([
        sys.float_info.max,      # DLAMCH('O') overflow threshold
        sys.float_info.min,      # DLAMCH('U') underflow threshold
        sys.float_info.min,      # DLAMCH('S') safe minimum
        sys.float_info.epsilon,  # DLAMCH('E') relative precision
        2.0                      # DLAMCH('B') base
    ], dtype=float, order='F')
    return macpar


def givens_matrix(c, s):
    """Create 2x2 Givens rotation matrix [[c, s], [-s, c]]."""
    return np.array([[c, s], [-s, c]], dtype=float, order='F')


def test_mb03bc_basic_k2():
    """
    Basic test with K=2 factors.

    For K=2, MB03BC computes SVD of the product A(:,:,2)^S(2).
    Verifies that the result diagonalizes the transformed product.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    k = 2

    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([[1.0, 0.5], [0.0, 0.8]], dtype=float)
    a[:, :, 1] = np.array([[2.0, -0.3], [0.0, 1.5]], dtype=float)

    amap = np.array([1, 2], dtype=np.int32, order='F')
    s = np.array([1, 1], dtype=np.int32, order='F')
    sinv = 1
    macpar = get_macpar()

    a_out, cv, sv = mb03bc(k, amap, s, sinv, a, macpar)

    assert len(cv) == k
    assert len(sv) == k

    for i in range(k):
        norm = cv[i]**2 + sv[i]**2
        np.testing.assert_allclose(norm, 1.0, rtol=1e-14)


def test_mb03bc_k3_identity_signature():
    """
    Test with K=3 factors and identity signature (all +1).

    Verifies Givens rotations produce orthogonal matrices and
    the transformed product becomes diagonal.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    k = 3

    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([[1.0, 0.2], [0.0, 0.9]], dtype=float)
    a[:, :, 1] = np.array([[1.5, 0.1], [0.0, 1.2]], dtype=float)
    a[:, :, 2] = np.array([[0.8, -0.1], [0.0, 0.7]], dtype=float)

    a_original = a.copy()

    amap = np.array([1, 2, 3], dtype=np.int32, order='F')
    s = np.array([1, 1, 1], dtype=np.int32, order='F')
    sinv = 1
    macpar = get_macpar()

    a_out, cv, sv = mb03bc(k, amap, s, sinv, a, macpar)

    assert len(cv) == k
    assert len(sv) == k

    for i in range(k):
        c, si = cv[i], sv[i]
        norm = c**2 + si**2
        np.testing.assert_allclose(norm, 1.0, rtol=1e-14)

    product = a_original[:, :, 1] @ a_original[:, :, 2]

    left_givens = givens_matrix(cv[0], sv[0])
    right_givens = givens_matrix(cv[k-1], -sv[k-1])
    diag = left_givens @ product @ right_givens

    np.testing.assert_allclose(diag[1, 0], 0.0, atol=1e-10)


def test_mb03bc_mixed_signature():
    """
    Test with mixed signature array (S has +1 and -1).

    When S(i) != SINV, the factor is effectively transposed and
    sign of off-diagonal is changed.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    k = 3

    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([[1.0, 0.3], [0.0, 0.8]], dtype=float)
    a[:, :, 1] = np.array([[1.2, -0.2], [0.0, 1.1]], dtype=float)
    a[:, :, 2] = np.array([[0.9, 0.15], [0.0, 0.85]], dtype=float)

    amap = np.array([1, 2, 3], dtype=np.int32, order='F')
    s = np.array([1, -1, 1], dtype=np.int32, order='F')
    sinv = 1
    macpar = get_macpar()

    a_out, cv, sv = mb03bc(k, amap, s, sinv, a, macpar)

    for i in range(k):
        norm = cv[i]**2 + sv[i]**2
        np.testing.assert_allclose(norm, 1.0, rtol=1e-14)


def test_mb03bc_k1_edge():
    """
    Edge case: K=1.

    With only one factor, the routine still produces valid Givens rotations.
    """
    k = 1

    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([[2.0, 1.0], [0.0, 1.0]], dtype=float)

    amap = np.array([1], dtype=np.int32, order='F')
    s = np.array([1], dtype=np.int32, order='F')
    sinv = 1
    macpar = get_macpar()

    a_out, cv, sv = mb03bc(k, amap, s, sinv, a, macpar)

    assert len(cv) == k
    assert len(sv) == k

    norm = cv[0]**2 + sv[0]**2
    np.testing.assert_allclose(norm, 1.0, rtol=1e-14)


def test_mb03bc_permuted_amap():
    """
    Test with permuted AMAP accessing factors in different order.

    AMAP allows accessing factors in arbitrary order.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    k = 3

    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([[1.0, 0.1], [0.0, 0.9]], dtype=float)
    a[:, :, 1] = np.array([[1.3, 0.2], [0.0, 1.1]], dtype=float)
    a[:, :, 2] = np.array([[0.7, -0.05], [0.0, 0.6]], dtype=float)

    amap = np.array([3, 1, 2], dtype=np.int32, order='F')
    s = np.array([1, 1, 1], dtype=np.int32, order='F')
    sinv = 1
    macpar = get_macpar()

    a_out, cv, sv = mb03bc(k, amap, s, sinv, a, macpar)

    for i in range(k):
        norm = cv[i]**2 + sv[i]**2
        np.testing.assert_allclose(norm, 1.0, rtol=1e-14)


def test_mb03bc_diagonal_product_preserved():
    """
    Mathematical property: Givens rotations preserve singular values.

    The diagonal of the transformed product should contain the
    singular values of the original product.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    k = 2

    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    a[:, :, 1] = np.array([[3.0, 1.0], [0.0, 2.0]], dtype=float)

    a_original = a.copy()

    amap = np.array([1, 2], dtype=np.int32, order='F')
    s = np.array([1, 1], dtype=np.int32, order='F')
    sinv = 1
    macpar = get_macpar()

    product = a_original[:, :, 1]
    expected_sv = np.linalg.svd(product, compute_uv=False)

    a_out, cv, sv = mb03bc(k, amap, s, sinv, a, macpar)

    left_givens = givens_matrix(cv[0], sv[0])
    right_givens = givens_matrix(cv[k-1], -sv[k-1])
    diag = left_givens @ product @ right_givens

    computed_sv = np.sort(np.abs([diag[0, 0], diag[1, 1]]))[::-1]
    np.testing.assert_allclose(computed_sv, expected_sv, rtol=1e-10)


def test_mb03bc_larger_leading_dims():
    """
    Test with larger leading dimensions (LDA1, LDA2 > 2).

    The routine should work correctly when storage has extra padding.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    k = 2
    lda1, lda2 = 4, 3

    a = np.zeros((lda1, lda2, k), dtype=float, order='F')
    a[0, 0, 0] = 1.0
    a[0, 1, 0] = 0.5
    a[1, 1, 0] = 0.8

    a[0, 0, 1] = 2.0
    a[0, 1, 1] = -0.3
    a[1, 1, 1] = 1.5

    amap = np.array([1, 2], dtype=np.int32, order='F')
    s = np.array([1, 1], dtype=np.int32, order='F')
    sinv = 1
    macpar = get_macpar()

    a_out, cv, sv = mb03bc(k, amap, s, sinv, a, macpar)

    for i in range(k):
        norm = cv[i]**2 + sv[i]**2
        np.testing.assert_allclose(norm, 1.0, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
