"""
Tests for MB03BA: Compute maps for Hessenberg index and signature array.

Computes suitable maps (AMAP, QMAP) for periodic QZ algorithms based on
Hessenberg index H and signature array S.

Tests:
1. Basic case: S[H] = 1 (positive signature)
2. Basic case: S[H] = -1 (negative signature)
3. K = 1 (single factor)
4. H at boundary positions
5. Inverse mapping property (mathematical correctness)

Random seeds: N/A (deterministic integer-only routine)
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal


def test_mb03ba_positive_signature():
    """
    Validate map computation when S[H-1] = 1 (1-based: S[H] = 1).

    When S[H] = 1:
    - SMULT = 1
    - AMAP[I] = (H-1+I-1) mod K + 1 in Fortran indexing
    - QMAP identical to AMAP
    """
    from ctrlsys import mb03ba

    k = 4
    h = 2
    s = np.array([1, 1, -1, 1], dtype=np.int32, order='F')

    smult, amap, qmap = mb03ba(k, h, s)

    assert smult == 1

    # When S[H] = 1: AMAP and QMAP are cyclic permutations starting from H
    # For K=4, H=2: AMAP = [2, 3, 4, 1], QMAP = [2, 3, 4, 1]
    expected_amap = np.array([2, 3, 4, 1], dtype=np.int32)
    expected_qmap = np.array([2, 3, 4, 1], dtype=np.int32)

    assert_array_equal(amap, expected_amap)
    assert_array_equal(qmap, expected_qmap)


def test_mb03ba_negative_signature():
    """
    Validate map computation when S[H-1] = -1 (1-based: S[H] = -1).

    When S[H] = -1:
    - SMULT = -1
    - AMAP and QMAP have different formulas involving reversal
    """
    from ctrlsys import mb03ba

    k = 4
    h = 2
    s = np.array([1, -1, 1, -1], dtype=np.int32, order='F')

    smult, amap, qmap = mb03ba(k, h, s)

    assert smult == -1

    # For K=4, H=2, S[H]=-1:
    # AMAP: I=1,2: AMAP[I]=H-I+1=2-I+1 -> [2,1]
    #       I=3,4: AMAP[I]=H+1-I+K=3-I+4 -> [4,3]
    # AMAP = [2, 1, 4, 3]
    expected_amap = np.array([2, 1, 4, 3], dtype=np.int32)

    # QMAP: TEMP = H mod K + 1 = 2 mod 4 + 1 = 3
    # I=3,2,1: QMAP[TEMP-I+1]=QMAP[3-I+1]=I -> QMAP[1]=3, QMAP[2]=2, QMAP[3]=1
    # I=4: QMAP[TEMP+K-I+1]=QMAP[3+4-4+1]=4 -> QMAP[4]=4
    # QMAP = [3, 2, 1, 4]
    expected_qmap = np.array([3, 2, 1, 4], dtype=np.int32)

    assert_array_equal(amap, expected_amap)
    assert_array_equal(qmap, expected_qmap)


def test_mb03ba_single_factor():
    """
    Validate K=1 (single factor case).
    """
    from ctrlsys import mb03ba

    k = 1
    h = 1
    s = np.array([1], dtype=np.int32, order='F')

    smult, amap, qmap = mb03ba(k, h, s)

    assert smult == 1
    assert_array_equal(amap, np.array([1], dtype=np.int32))
    assert_array_equal(qmap, np.array([1], dtype=np.int32))


def test_mb03ba_single_factor_negative():
    """
    Validate K=1 with negative signature.
    """
    from ctrlsys import mb03ba

    k = 1
    h = 1
    s = np.array([-1], dtype=np.int32, order='F')

    smult, amap, qmap = mb03ba(k, h, s)

    assert smult == -1
    # For K=1, H=1: AMAP[1]=H-1+1=1, QMAP: TEMP=1%1+1=1, I=1: QMAP[1]=1
    assert_array_equal(amap, np.array([1], dtype=np.int32))
    assert_array_equal(qmap, np.array([1], dtype=np.int32))


def test_mb03ba_h_at_k():
    """
    Validate when H = K (boundary case).
    """
    from ctrlsys import mb03ba

    k = 3
    h = 3
    s = np.array([1, 1, 1], dtype=np.int32, order='F')

    smult, amap, qmap = mb03ba(k, h, s)

    assert smult == 1
    # H=K, S[H]=1: AMAP = QMAP = [3, 1, 2]
    expected_amap = np.array([3, 1, 2], dtype=np.int32)
    expected_qmap = np.array([3, 1, 2], dtype=np.int32)

    assert_array_equal(amap, expected_amap)
    assert_array_equal(qmap, expected_qmap)


def test_mb03ba_h_equals_one():
    """
    Validate when H = 1.
    """
    from ctrlsys import mb03ba

    k = 4
    h = 1
    s = np.array([1, -1, 1, -1], dtype=np.int32, order='F')

    smult, amap, qmap = mb03ba(k, h, s)

    assert smult == 1
    # H=1, S[1]=1: AMAP = QMAP = [1, 2, 3, 4]
    expected_amap = np.array([1, 2, 3, 4], dtype=np.int32)
    expected_qmap = np.array([1, 2, 3, 4], dtype=np.int32)

    assert_array_equal(amap, expected_amap)
    assert_array_equal(qmap, expected_qmap)


def test_mb03ba_permutation_property():
    """
    Validate that AMAP is a permutation of 1..K.

    Mathematical property: AMAP must be a valid permutation.
    """
    from ctrlsys import mb03ba

    k = 5
    h = 3
    s = np.array([1, -1, 1, -1, 1], dtype=np.int32, order='F')

    smult, amap, qmap = mb03ba(k, h, s)

    # Both AMAP and QMAP must be permutations of [1, 2, ..., K]
    expected_set = set(range(1, k + 1))
    assert set(amap) == expected_set, f"AMAP {amap} is not a permutation of 1..{k}"
    assert set(qmap) == expected_set, f"QMAP {qmap} is not a permutation of 1..{k}"


def test_mb03ba_negative_signature_permutation():
    """
    Validate permutation property with negative signature.
    """
    from ctrlsys import mb03ba

    k = 5
    h = 3
    s = np.array([1, -1, -1, 1, -1], dtype=np.int32, order='F')

    smult, amap, qmap = mb03ba(k, h, s)

    assert smult == -1

    expected_set = set(range(1, k + 1))
    assert set(amap) == expected_set
    assert set(qmap) == expected_set


def test_mb03ba_larger_k():
    """
    Validate with larger K value.
    """
    from ctrlsys import mb03ba

    k = 8
    h = 5
    s = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32, order='F')

    smult, amap, qmap = mb03ba(k, h, s)

    assert smult == 1
    # H=5, K=8, S[H]=1: cyclic shift starting from H
    # AMAP = QMAP = [5, 6, 7, 8, 1, 2, 3, 4]
    expected = np.array([5, 6, 7, 8, 1, 2, 3, 4], dtype=np.int32)

    assert_array_equal(amap, expected)
    assert_array_equal(qmap, expected)


def test_mb03ba_bijection_property():
    from ctrlsys import mb03ba

    for k in range(2, 7):
        for h in range(1, k + 1):
            for sign in [1, -1]:
                s = np.array([sign if i == h - 1 else 1 for i in range(k)], dtype=np.int32, order='F')
                smult, amap, qmap = mb03ba(k, h, s)
                assert set(amap) == set(range(1, k + 1))
                assert set(qmap) == set(range(1, k + 1))
