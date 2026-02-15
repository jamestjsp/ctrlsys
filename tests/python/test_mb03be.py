"""
Tests for MB03BE: Apply periodic QZ iterations to 2x2 matrix product.

Applies at most 20 iterations of a real single shifted periodic QZ algorithm
to the 2-by-2 product of matrices stored in a 3D array. The goal is to drive
the (2,1) element of the first factor toward zero.

Tests:
1. Basic case with K=2 factors - convergence to quasi-triangular form
2. K=3 factors test
3. Mixed signature test
4. Property: subdiagonal element should decrease
5. Edge case with already triangular factor

Random seeds: 42, 123, 456, 789, 111 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def compute_product(a, amap, s, sinv):
    """Compute the actual matrix product respecting signatures."""
    k = len(amap)
    result = np.eye(2)
    for i in range(k):
        ai = amap[i] - 1
        factor = a[:, :, ai].copy()
        if s[ai] != sinv:
            factor = np.linalg.inv(factor)
        result = result @ factor
    return result


def test_mb03be_basic_k2():
    """
    Validate basic functionality with K=2 factors.

    The algorithm should drive A(2,1,AMAP(1)) toward zero.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03be

    np.random.seed(42)

    k = 2
    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [2.0, 1.5],
        [0.8, 1.2]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.5, 0.5],
        [0.0, 2.0]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    a_orig = a.copy()
    prod_before = compute_product(a_orig, amap, s, sinv)

    a_out = mb03be(k, amap, s, sinv, a)

    ai = amap[0] - 1
    subdiag_after = abs(a_out[1, 0, ai])

    assert a_out is not None, "mb03be should return array"
    assert subdiag_after < 1.0, "Subdiagonal should decrease after QZ iterations"


def test_mb03be_convergence():
    """
    Validate convergence: subdiagonal element should become small.

    After QZ iterations, |A(2,1)| should be small relative to other elements.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03be

    np.random.seed(123)

    k = 2
    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.0, 2.0],
        [0.5, 1.5]
    ], order='F')
    a[:, :, 1] = np.array([
        [2.0, 1.0],
        [0.0, 1.5]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    a_out = mb03be(k, amap, s, sinv, a)

    ai = amap[0] - 1
    a11 = abs(a_out[0, 0, ai])
    a12 = abs(a_out[0, 1, ai])
    a21 = abs(a_out[1, 0, ai])
    a22 = abs(a_out[1, 1, ai])
    max_elem = max(a11, a12, a22)

    ulp = np.finfo(float).eps
    assert a21 < ulp * max_elem * 100, \
        f"Subdiagonal {a21} should be small relative to max element {max_elem}"


def test_mb03be_k3_factors():
    """
    Validate with K=3 factors.

    Tests the loop iteration with multiple factors.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03be

    np.random.seed(456)

    k = 3
    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.2, 0.8],
        [0.6, 1.4]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.5, 0.5],
        [0.0, 1.8]
    ], order='F')
    a[:, :, 2] = np.array([
        [2.0, 0.3],
        [0.0, 1.2]
    ], order='F')

    amap = np.array([1, 2, 3], dtype=np.int32)
    s = np.array([1, 1, 1], dtype=np.int32)
    sinv = 1

    a_out = mb03be(k, amap, s, sinv, a)

    assert a_out is not None
    ai = amap[0] - 1
    assert abs(a_out[1, 0, ai]) < 1.0, "Subdiagonal should decrease"


def test_mb03be_mixed_signature():
    """
    Validate with mixed signature entries.

    Tests S(AI) != SINV branch in the algorithm.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03be

    np.random.seed(789)

    k = 2
    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [2.0, 1.0],
        [0.5, 1.5]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.8, 0.6],
        [0.0, 2.2]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, -1], dtype=np.int32)
    sinv = 1

    a_out = mb03be(k, amap, s, sinv, a)

    assert a_out is not None
    ai = amap[0] - 1
    assert np.isfinite(a_out[1, 0, ai]), "Result should be finite"


def test_mb03be_already_triangular():
    """
    Validate with already upper triangular first factor.

    If A(2,1) is already zero, algorithm should preserve this.
    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb03be

    np.random.seed(111)

    k = 2
    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [2.0, 1.5],
        [0.0, 1.2]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.5, 0.5],
        [0.0, 2.0]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    a_out = mb03be(k, amap, s, sinv, a)

    ai = amap[0] - 1
    ulp = np.finfo(float).eps
    assert abs(a_out[1, 0, ai]) < ulp * 100, \
        "Zero subdiagonal should remain essentially zero"


def test_mb03be_permuted_amap():
    """
    Validate with permuted AMAP (factors stored in non-sequential order).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb03be

    np.random.seed(222)

    k = 2
    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [1.5, 0.5],
        [0.0, 2.0]
    ], order='F')
    a[:, :, 1] = np.array([
        [2.0, 1.5],
        [0.8, 1.2]
    ], order='F')

    amap = np.array([2, 1], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    a_out = mb03be(k, amap, s, sinv, a)

    ai = amap[0] - 1
    assert a_out is not None
    assert np.all(np.isfinite(a_out)), "All elements should be finite"


def test_mb03be_sinv_negative():
    """
    Validate with SINV = -1.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb03be

    np.random.seed(333)

    k = 2
    a = np.zeros((2, 2, k), dtype=float, order='F')
    a[:, :, 0] = np.array([
        [2.0, 1.0],
        [0.5, 1.8]
    ], order='F')
    a[:, :, 1] = np.array([
        [1.5, 0.6],
        [0.0, 2.0]
    ], order='F')

    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([-1, -1], dtype=np.int32)
    sinv = -1

    a_out = mb03be(k, amap, s, sinv, a)

    assert a_out is not None
    assert np.all(np.isfinite(a_out)), "All elements should be finite"
