"""
Tests for MA02RD: Sort vector D and rearrange E with same permutation

MA02RD sorts the elements of an n-vector D in increasing or decreasing order,
and rearranges the elements of an n-vector E using the same permutations.

Uses Quick Sort with Insertion sort fallback for arrays of length <= 20.

Random seeds used for reproducibility:
- test_ma02rd_increasing_basic: 42
- test_ma02rd_decreasing_basic: 123
- test_ma02rd_permutation_consistency: 456
- test_ma02rd_stability: 789
"""

import numpy as np
import pytest


def test_ma02rd_increasing_basic():
    """
    Test MA02RD with ID='I' - sort D in increasing order.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 10

    d = np.random.randn(n).astype(float, order='F')
    e = np.random.randn(n).astype(float, order='F')

    d_orig = d.copy()
    e_orig = e.copy()

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('I', d, e)

    assert info == 0

    # Check D is sorted in increasing order
    for i in range(n - 1):
        assert d_out[i] <= d_out[i + 1], f"D not increasing at {i}: {d_out[i]} > {d_out[i+1]}"

    # Check E was rearranged with same permutation
    # For each element in d_out, find its original position and verify e_out
    for i in range(n):
        orig_idx = np.where(np.isclose(d_orig, d_out[i]))[0]
        if len(orig_idx) == 1:
            j = orig_idx[0]
            np.testing.assert_allclose(e_out[i], e_orig[j], rtol=1e-14)


def test_ma02rd_decreasing_basic():
    """
    Test MA02RD with ID='D' - sort D in decreasing order.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 10

    d = np.random.randn(n).astype(float, order='F')
    e = np.random.randn(n).astype(float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('D', d, e)

    assert info == 0

    # Check D is sorted in decreasing order
    for i in range(n - 1):
        assert d_out[i] >= d_out[i + 1], f"D not decreasing at {i}: {d_out[i]} < {d_out[i+1]}"


def test_ma02rd_permutation_consistency():
    """
    Mathematical property test: E follows exact same permutation as D.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 8

    # Create D with unique values for easy tracking
    d = np.arange(n, dtype=float)
    np.random.shuffle(d)
    d = d.astype(float, order='F')

    # Create E = 100 + D so we can verify permutation exactly
    e = (100.0 + d).astype(float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('I', d, e)

    assert info == 0

    # After sorting, e_out should equal 100 + d_out
    np.testing.assert_allclose(e_out, 100.0 + d_out, rtol=1e-14, atol=1e-15)


def test_ma02rd_quick_sort_threshold():
    """
    Test MA02RD with array size > 20 to trigger Quick Sort path.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 100

    d = np.random.randn(n).astype(float, order='F')
    e = np.asarray(np.arange(n), dtype=float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('I', d, e)

    assert info == 0

    # Check sorted
    for i in range(n - 1):
        assert d_out[i] <= d_out[i + 1]


def test_ma02rd_insertion_sort_threshold():
    """
    Test MA02RD with array size <= 20 to use Insertion sort path.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 15

    d = np.random.randn(n).astype(float, order='F')
    e = np.asarray(np.arange(n), dtype=float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('D', d, e)

    assert info == 0

    # Check sorted decreasing
    for i in range(n - 1):
        assert d_out[i] >= d_out[i + 1]


def test_ma02rd_invalid_id():
    """
    Test MA02RD with invalid ID parameter - should return info=-1.
    """
    n = 5
    d = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=float, order='F')
    e = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float, order='F')

    from ctrlsys import ma02rd

    with pytest.raises(ValueError):
        ma02rd('X', d, e)


def test_ma02rd_n_zero():
    """
    Test MA02RD with N=0 (edge case).
    """
    d = np.array([], dtype=float, order='F')
    e = np.array([], dtype=float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('I', d, e)

    assert info == 0
    assert len(d_out) == 0
    assert len(e_out) == 0


def test_ma02rd_n_one():
    """
    Test MA02RD with N=1 (single element, trivially sorted).
    """
    d = np.array([5.0], dtype=float, order='F')
    e = np.array([10.0], dtype=float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('I', d, e)

    assert info == 0
    np.testing.assert_allclose(d_out, [5.0], rtol=1e-14)
    np.testing.assert_allclose(e_out, [10.0], rtol=1e-14)


def test_ma02rd_already_sorted():
    """
    Test MA02RD with already sorted array.
    """
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
    e = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('I', d, e)

    assert info == 0
    np.testing.assert_allclose(d_out, [1.0, 2.0, 3.0, 4.0, 5.0], rtol=1e-14)
    np.testing.assert_allclose(e_out, [10.0, 20.0, 30.0, 40.0, 50.0], rtol=1e-14)


def test_ma02rd_reverse_sorted():
    """
    Test MA02RD with reverse sorted array.
    """
    d = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=float, order='F')
    e = np.array([50.0, 40.0, 30.0, 20.0, 10.0], dtype=float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('I', d, e)

    assert info == 0
    np.testing.assert_allclose(d_out, [1.0, 2.0, 3.0, 4.0, 5.0], rtol=1e-14)
    np.testing.assert_allclose(e_out, [10.0, 20.0, 30.0, 40.0, 50.0], rtol=1e-14)


def test_ma02rd_duplicate_values():
    """
    Test MA02RD with duplicate values in D.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)

    # Create array with duplicates
    d = np.array([3.0, 1.0, 3.0, 2.0, 1.0], dtype=float, order='F')
    e = np.array([30.0, 10.0, 31.0, 20.0, 11.0], dtype=float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('I', d, e)

    assert info == 0

    # Check sorted
    for i in range(len(d) - 1):
        assert d_out[i] <= d_out[i + 1]

    # Check all original elements preserved
    np.testing.assert_allclose(sorted(d_out), sorted(d), rtol=1e-14)
    np.testing.assert_allclose(sorted(e_out), sorted(e), rtol=1e-14)


def test_ma02rd_large_array():
    """
    Test MA02RD with large array to stress Quick Sort.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 1000

    d = np.random.randn(n).astype(float, order='F')
    e = np.asarray(np.arange(n), dtype=float, order='F')

    from ctrlsys import ma02rd
    d_out, e_out, info = ma02rd('I', d, e)

    assert info == 0

    # Check sorted
    assert np.all(np.diff(d_out) >= 0)
