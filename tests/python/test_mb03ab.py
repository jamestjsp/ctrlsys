"""
Tests for MB03AB: Compute Givens rotations for Wilkinson shift polynomial.

Computes two Givens rotations (C1,S1) and (C2,S2) such that the orthogonal
matrix Z makes the first column of the real Wilkinson double shift polynomial
of a product of matrices in periodic upper Hessenberg form parallel to the
first unit vector.

Tests:
1. Basic case: Complex conjugate shifts (SHFT='C')
2. Basic case: Two real identical shifts (SHFT='D')
3. Basic case: Two real distinct shifts (SHFT='R')
4. Single shift case (SHFT='S')
5. Givens rotation property: C^2 + S^2 = 1

Random seeds: 42, 123, 456 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03ab_single_shift():
    """
    Validate single shift case (SHFT='S').

    For single shift, C2 = 1 and S2 = 0.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03ab

    np.random.seed(42)

    k = 2
    n = 3
    lda1, lda2 = n, n

    a = np.zeros((lda1, lda2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 8.0, 9.0]
    ], order='F')
    a[:, :, 1] = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 3.0, 1.5],
        [0.0, 0.0, 4.0]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1
    w1 = 0.0
    w2 = 1.5

    c1, s1, c2, s2 = mb03ab('S', k, n, amap, s, sinv, a, w1, w2)

    assert c2 == 1.0, "For SHFT='S', C2 must be 1.0"
    assert s2 == 0.0, "For SHFT='S', S2 must be 0.0"

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)


def test_mb03ab_complex_conjugate_shifts():
    """
    Validate complex conjugate shifts (SHFT='C').

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03ab

    np.random.seed(123)

    k = 2
    n = 3
    lda1, lda2 = n, n

    a = np.zeros((lda1, lda2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [0.0, 0.3, 0.8]
    ], order='F')
    a[:, :, 1] = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 1.5, 1.0],
        [0.0, 0.0, 2.0]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1
    w1 = 0.5
    w2 = 0.3

    c1, s1, c2, s2 = mb03ab('C', k, n, amap, s, sinv, a, w1, w2)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


def test_mb03ab_real_identical_shifts():
    """
    Validate two real identical shifts (SHFT='D').

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03ab

    np.random.seed(456)

    k = 2
    n = 3

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [2.0, 1.0, 0.5],
        [1.0, 2.5, 1.5],
        [0.0, 0.8, 1.2]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.5, 0.5, 0.2],
        [0.0, 2.0, 0.8],
        [0.0, 0.0, 1.8]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1
    w1 = 1.0
    w2 = 1.0

    c1, s1, c2, s2 = mb03ab('D', k, n, amap, s, sinv, a, w1, w2)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


def test_mb03ab_real_distinct_shifts():
    """
    Validate two real distinct shifts (SHFT='R').

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03ab

    np.random.seed(789)

    k = 2
    n = 3

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.5, 2.5, 1.0],
        [0.7, 1.8, 2.2],
        [0.0, 0.5, 0.9]
    ], order='F')
    a[:, :, 1] = np.array([
        [2.2, 1.5, 0.8],
        [0.0, 1.2, 1.0],
        [0.0, 0.0, 0.6]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1
    w1 = 0.8
    w2 = 1.2

    c1, s1, c2, s2 = mb03ab('R', k, n, amap, s, sinv, a, w1, w2)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


def test_mb03ab_negative_signature():
    """
    Validate with negative signature entries.

    Tests S(AI) != SINV branch in the algorithm.
    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb03ab

    np.random.seed(111)

    k = 2
    n = 3

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.0, 2.0, 1.5],
        [0.5, 1.8, 2.0],
        [0.0, 0.6, 1.2]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.5, 1.0, 0.5],
        [0.0, 2.0, 1.2],
        [0.0, 0.0, 1.5]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, -1], dtype=np.int32)
    sinv = 1

    w1 = 0.5
    w2 = 0.8

    c1, s1, c2, s2 = mb03ab('R', k, n, amap, s, sinv, a, w1, w2)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


def test_mb03ab_three_factors():
    """
    Validate with K=3 factors.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb03ab

    np.random.seed(222)

    k = 3
    n = 3

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.0, 2.0, 1.5],
        [0.8, 1.5, 2.0],
        [0.0, 0.4, 1.0]
    ], order='F')
    a[:, :, 1] = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 1.5, 1.0],
        [0.0, 0.0, 2.0]
    ], order='F')
    a[:, :, 2] = np.array([
        [1.5, 0.5, 0.2],
        [0.0, 1.8, 0.6],
        [0.0, 0.0, 1.2]
    ], order='F')

    amap = np.array([1, 2, 3], dtype=np.int32)
    s = np.array([1, 1, 1], dtype=np.int32)
    sinv = 1
    w1 = 1.0
    w2 = 0.5

    c1, s1, c2, s2 = mb03ab('C', k, n, amap, s, sinv, a, w1, w2)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)


def test_mb03ab_givens_rotation_property():
    """
    Validate mathematical property: Givens rotation C^2 + S^2 = 1.

    This tests all shift types produce valid Givens rotations.
    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb03ab

    np.random.seed(333)

    k = 2
    n = 3

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.2, 1.8, 0.9],
        [0.6, 1.4, 1.6],
        [0.0, 0.5, 0.8]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.8, 0.8, 0.4],
        [0.0, 1.6, 0.9],
        [0.0, 0.0, 1.4]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    for shft in ['C', 'D', 'R', 'S']:
        w1 = 0.5
        w2 = 0.3 if shft != 'D' else 0.5

        c1, s1, c2, s2 = mb03ab(shft, k, n, amap, s, sinv, a, w1, w2)

        assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14,
                        err_msg=f"C1^2 + S1^2 != 1 for SHFT='{shft}'")

        if shft == 'S':
            assert c2 == 1.0, f"C2 must be 1.0 for SHFT='S'"
            assert s2 == 0.0, f"S2 must be 0.0 for SHFT='S'"
        else:
            assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14,
                            err_msg=f"C2^2 + S2^2 != 1 for SHFT='{shft}'")


def test_mb03ab_mixed_signature():
    """
    Validate with mixed signature entries.

    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb03ab

    np.random.seed(444)

    k = 3
    n = 3

    a = np.zeros((n, n, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.0, 1.5, 0.8],
        [0.5, 1.2, 1.4],
        [0.0, 0.3, 0.9]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.6, 0.9, 0.4],
        [0.0, 1.4, 0.7],
        [0.0, 0.0, 1.2]
    ], order='F')
    a[:, :, 2] = np.array([
        [1.3, 0.6, 0.3],
        [0.0, 1.1, 0.5],
        [0.0, 0.0, 0.8]
    ], order='F')

    amap = np.array([1, 2, 3], dtype=np.int32)
    s = np.array([1, -1, 1], dtype=np.int32)
    sinv = 1

    w1 = 0.6
    w2 = 0.4

    c1, s1, c2, s2 = mb03ab('R', k, n, amap, s, sinv, a, w1, w2)

    assert_allclose(c1**2 + s1**2, 1.0, rtol=1e-14)
    assert_allclose(c2**2 + s2**2, 1.0, rtol=1e-14)
