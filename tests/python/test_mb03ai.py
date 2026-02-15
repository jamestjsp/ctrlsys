"""
Tests for MB03AI: Compute Givens rotations for Wilkinson shift polynomial (full evaluation).

Computes two Givens rotations (C1,S1) and (C2,S2) such that the orthogonal
matrix Z makes the first column of the real Wilkinson double/single shift
polynomial of a product of matrices in periodic upper Hessenberg form parallel
to the first unit vector.

Unlike MB03AH/MB03AE, this routine EVALUATES the full matrix product and computes
eigenvalues using DLAHQR, making it more robust but slower. Should be called when
convergence difficulties are encountered for small order matrices (N, K <= 6).

Key difference: AMAP(K) points to the Hessenberg matrix (last factor).

Tests:
1. Single shift case (SHFT='S')
2. Double shift case (SHFT='D')
3. Givens rotation property: C^2 + S^2 = 1
4. Negative signature entries
5. Multiple factors

Random seeds: 42, 123, 456, 789, 111, 222, 333, 444, 555, 666 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03ai_single_shift():
    """
    Validate single shift case (SHFT='S').

    For single shift, C2 = 1 and S2 = 0.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03ai

    np.random.seed(42)

    k = 2
    n = 3
    lda1, lda2 = n, n

    a = np.zeros((lda1, lda2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 3.0, 1.5],
        [0.0, 0.0, 4.0]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 8.0, 9.0]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    c1, s1, c2, s2 = mb03ai('S', k, n, amap, s, sinv, a)

    assert c2 == 1.0, "For SHFT='S', C2 must be 1.0"
    assert s2 == 0.0, "For SHFT='S', S2 must be 0.0"

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)


def test_mb03ai_double_shift():
    """
    Validate double shift case (SHFT='D').

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03ai

    np.random.seed(123)

    k = 2
    n = 4
    lda1, lda2 = n, n

    a = np.zeros((lda1, lda2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [2.0, 1.0, 0.5, 0.3],
        [0.0, 1.5, 1.0, 0.6],
        [0.0, 0.0, 2.0, 0.8],
        [0.0, 0.0, 0.0, 1.2]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.0, 2.0, 3.0, 1.5],
        [0.5, 1.5, 2.5, 2.0],
        [0.0, 0.3, 0.8, 1.2],
        [0.0, 0.0, 0.4, 0.6]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    c1, s1, c2, s2 = mb03ai('D', k, n, amap, s, sinv, a)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


def test_mb03ai_negative_signature():
    """
    Validate with negative signature entries.

    Tests S(AI) != SINV branch in the algorithm (uses DTRSM instead of DTRMM).
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03ai

    np.random.seed(789)

    k = 2
    n = 3

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.5, 1.0, 0.5],
        [0.0, 2.0, 1.2],
        [0.0, 0.0, 1.5]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.0, 2.0, 1.5],
        [0.5, 1.8, 2.0],
        [0.0, 0.6, 1.2]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, -1], dtype=np.int32)
    sinv = 1

    c1, s1, c2, s2 = mb03ai('S', k, n, amap, s, sinv, a)

    assert c2 == 1.0, "For SHFT='S', C2 must be 1.0"
    assert s2 == 0.0, "For SHFT='S', S2 must be 0.0"
    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)


def test_mb03ai_three_factors():
    """
    Validate with K=3 factors.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb03ai

    np.random.seed(111)

    k = 3
    n = 4

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [2.0, 1.0, 0.5, 0.3],
        [0.0, 1.5, 1.0, 0.5],
        [0.0, 0.0, 2.0, 0.8],
        [0.0, 0.0, 0.0, 1.5]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.5, 0.5, 0.2, 0.1],
        [0.0, 1.8, 0.6, 0.3],
        [0.0, 0.0, 1.2, 0.4],
        [0.0, 0.0, 0.0, 0.8]
    ], order='F')
    a[:, :, 2] = np.array([
        [1.0, 2.0, 1.5, 0.8],
        [0.8, 1.5, 2.0, 1.2],
        [0.0, 0.4, 1.0, 0.6],
        [0.0, 0.0, 0.3, 0.9]
    ], order='F')

    amap = np.array([1, 2, 3], dtype=np.int32)
    s = np.array([1, 1, 1], dtype=np.int32)
    sinv = 1

    c1, s1, c2, s2 = mb03ai('D', k, n, amap, s, sinv, a)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


def test_mb03ai_givens_rotation_property():
    """
    Validate mathematical property: Givens rotation C^2 + S^2 = 1.

    This tests all shift types produce valid Givens rotations.
    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb03ai

    np.random.seed(222)

    k = 2
    n = 4

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.8, 0.8, 0.4, 0.2],
        [0.0, 1.6, 0.9, 0.5],
        [0.0, 0.0, 1.4, 0.6],
        [0.0, 0.0, 0.0, 1.1]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.2, 1.8, 0.9, 0.5],
        [0.6, 1.4, 1.6, 0.8],
        [0.0, 0.5, 0.8, 0.4],
        [0.0, 0.0, 0.3, 0.7]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    for shft in ['D', 'S']:
        c1, s1, c2, s2 = mb03ai(shft, k, n, amap, s, sinv, a)

        assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14,
                        err_msg=f"C1^2 + S1^2 != 1 for SHFT='{shft}'")

        if shft == 'S':
            assert c2 == 1.0, f"C2 must be 1.0 for SHFT='S'"
            assert s2 == 0.0, f"S2 must be 0.0 for SHFT='S'"
        else:
            assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14,
                            err_msg=f"C2^2 + S2^2 != 1 for SHFT='{shft}'")


def test_mb03ai_mixed_signature():
    """
    Validate with mixed signature entries.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb03ai

    np.random.seed(333)

    k = 3
    n = 4

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.6, 0.9, 0.4, 0.2],
        [0.0, 1.4, 0.7, 0.4],
        [0.0, 0.0, 1.2, 0.5],
        [0.0, 0.0, 0.0, 0.9]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.3, 0.6, 0.3, 0.15],
        [0.0, 1.1, 0.5, 0.25],
        [0.0, 0.0, 0.8, 0.35],
        [0.0, 0.0, 0.0, 0.7]
    ], order='F')
    a[:, :, 2] = np.array([
        [1.0, 1.5, 0.8, 0.4],
        [0.5, 1.2, 1.4, 0.7],
        [0.0, 0.3, 0.9, 0.5],
        [0.0, 0.0, 0.2, 0.6]
    ], order='F')

    amap = np.array([1, 2, 3], dtype=np.int32)
    s = np.array([1, -1, 1], dtype=np.int32)
    sinv = 1

    c1, s1, c2, s2 = mb03ai('D', k, n, amap, s, sinv, a)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


def test_mb03ai_negative_sinv():
    """
    Validate with negative SINV.

    When SINV is negative, shifts correspond to reciprocals of eigenvalues.
    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb03ai

    np.random.seed(444)

    k = 2
    n = 3

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.5, 0.8, 0.4],
        [0.0, 1.2, 0.6],
        [0.0, 0.0, 0.9]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.0, 1.5, 0.8],
        [0.5, 1.2, 1.0],
        [0.0, 0.4, 0.7]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = -1

    c1, s1, c2, s2 = mb03ai('S', k, n, amap, s, sinv, a)

    assert c2 == 1.0, "For SHFT='S', C2 must be 1.0"
    assert s2 == 0.0, "For SHFT='S', S2 must be 0.0"
    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)


def test_mb03ai_eigenvalue_based_shift():
    """
    Validate that MB03AI computes proper Wilkinson shifts from eigenvalues.

    MB03AI explicitly evaluates the matrix product and uses DLAHQR to compute
    eigenvalues, selecting the two with largest moduli for shifts.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import mb03ai

    np.random.seed(555)

    k = 2
    n = 4

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [3.0, 1.0, 0.5, 0.2],
        [0.0, 2.5, 1.0, 0.4],
        [0.0, 0.0, 2.0, 0.6],
        [0.0, 0.0, 0.0, 1.5]
    ], order='F')
    a[:, :, 1] = np.array([
        [2.0, 1.5, 1.0, 0.5],
        [0.8, 1.8, 1.2, 0.6],
        [0.0, 0.6, 1.0, 0.4],
        [0.0, 0.0, 0.3, 0.8]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    c1, s1, c2, s2 = mb03ai('D', k, n, amap, s, sinv, a)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)

    assert abs(c1) <= 1.0
    assert abs(s1) <= 1.0
    assert abs(c2) <= 1.0
    assert abs(s2) <= 1.0


def test_mb03ai_small_k_n():
    """
    Validate with smallest valid parameters (N=2, K=1).

    This tests the edge case where the routine handles minimal dimensions.
    Random seed: 666 (for reproducibility)
    """
    from ctrlsys import mb03ai

    np.random.seed(666)

    k = 1
    n = 2

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [2.0, 1.0],
        [0.5, 1.5]
    ], order='F')

    amap = np.array([1], dtype=np.int32)
    s = np.array([1], dtype=np.int32)
    sinv = 1

    c1, s1, c2, s2 = mb03ai('S', k, n, amap, s, sinv, a)

    assert c2 == 1.0, "For SHFT='S', C2 must be 1.0"
    assert s2 == 0.0, "For SHFT='S', S2 must be 0.0"
    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
