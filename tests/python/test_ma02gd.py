"""
Tests for MA02GD: Column interchanges on a matrix.

Performs series of column swaps based on pivot indices.
Column-oriented counterpart of LAPACK's DLASWP (row swaps).
"""
import numpy as np
import pytest
from ctrlsys import ma02gd


"""Basic functionality tests."""

def test_simple_swap_incx_1():
    """
    Basic column swaps with INCX=1.

    IPIV[0]=1, IPIV[1]=2 means:
    - Swap col 0 with col 0 (no-op)
    - Swap col 1 with col 1 (no-op)
    """
    n = 3  # rows
    k1, k2 = 1, 2
    incx = 1

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], order='F', dtype=float)

    ipiv = np.array([1, 2], dtype=np.int32)

    a_out = ma02gd(n, a, k1, k2, ipiv, incx)

    # No change since IPIV[k-1] == k for all k
    expected = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_swap_columns():
    """
    Swap column 1 with column 3.

    IPIV = [3] means swap col 1 with col 3.
    """
    n = 2  # rows
    k1, k2 = 1, 1
    incx = 1

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], order='F', dtype=float)

    ipiv = np.array([3], dtype=np.int32)

    a_out = ma02gd(n, a, k1, k2, ipiv, incx)

    # Col 0 and col 2 should be swapped
    expected = np.array([
        [3.0, 2.0, 1.0],
        [6.0, 5.0, 4.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_multiple_swaps():
    """
    Multiple swaps: swap col 1 with col 2, then col 2 with col 3.
    """
    n = 2
    k1, k2 = 1, 2
    incx = 1

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], order='F', dtype=float)

    # IPIV[0]=2 means swap col 1 with col 2
    # IPIV[1]=3 means swap col 2 with col 3
    ipiv = np.array([2, 3], dtype=np.int32)

    a_out = ma02gd(n, a, k1, k2, ipiv, incx)

    # After swap 1: [2,1,3], [5,4,6]
    # After swap 2: [2,3,1], [5,6,4]
    expected = np.array([
        [2.0, 3.0, 1.0],
        [5.0, 6.0, 4.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_negative_incx():
    """
    Negative INCX: swaps in reverse order.
    """
    n = 2
    k1, k2 = 1, 2
    incx = -1

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], order='F', dtype=float)

    # With incx=-1, loop goes K2 down to K1
    # IPIV index calculation: JX = 1 + (1 - K2)*INCX = 1 + (1-2)*(-1) = 2
    # K=2: JP = IPIV[2-1] = IPIV[1] = 3, swap col 2 with col 3
    # K=1: JP = IPIV[1-1] = IPIV[0] = 2, swap col 1 with col 2
    ipiv = np.array([2, 3], dtype=np.int32)

    a_out = ma02gd(n, a, k1, k2, ipiv, incx)

    # After swap col 2 with col 3: [1,3,2], [4,6,5]
    # After swap col 1 with col 2: [3,1,2], [6,4,5]
    expected = np.array([
        [3.0, 1.0, 2.0],
        [6.0, 4.0, 5.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)


"""Edge case tests."""

def test_incx_zero():
    """INCX=0 should return without changes (quick return)."""
    n = 2
    k1, k2 = 1, 2
    incx = 0

    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    ipiv = np.array([2, 1], dtype=np.int32)
    a_orig = a.copy()

    a_out = ma02gd(n, a, k1, k2, ipiv, incx)

    # No change
    np.testing.assert_allclose(a_out, a_orig, rtol=1e-14)

def test_n_zero():
    """N=0 should return without changes (quick return)."""
    n = 0
    k1, k2 = 1, 2
    incx = 1

    a = np.zeros((0, 3), order='F', dtype=float)
    ipiv = np.array([2, 1], dtype=np.int32)

    a_out = ma02gd(n, a, k1, k2, ipiv, incx)

    assert a_out.shape == (0, 3)

def test_single_row():
    """Single row matrix."""
    n = 1
    k1, k2 = 1, 3
    incx = 1

    a = np.array([[1.0, 2.0, 3.0, 4.0]], order='F', dtype=float)
    ipiv = np.array([2, 3, 4], dtype=np.int32)

    a_out = ma02gd(n, a, k1, k2, ipiv, incx)

    # Swap col1 with col2: [2,1,3,4]
    # Swap col2 with col3: [2,3,1,4]
    # Swap col3 with col4: [2,3,4,1]
    expected = np.array([[2.0, 3.0, 4.0, 1.0]], order='F', dtype=float)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)


"""Mathematical property validation tests."""

def test_involution():
    """
    Applying the same permutation twice should return original.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m = 4, 4
    k1, k2 = 1, 4
    incx = 1

    a = np.random.randn(n, m).astype(float, order='F')
    a_orig = a.copy()

    # Identity permutation (each column stays)
    ipiv = np.array([1, 2, 3, 4], dtype=np.int32)

    a_out = ma02gd(n, a, k1, k2, ipiv, incx)

    np.testing.assert_allclose(a_out, a_orig, rtol=1e-14)

def test_double_swap_identity():
    """
    Swapping same columns twice returns to original.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3
    k1, k2 = 1, 1
    incx = 1

    a = np.random.randn(n, 3).astype(float, order='F')
    a_orig = a.copy()

    # Swap col 1 with col 3
    ipiv = np.array([3], dtype=np.int32)

    a_swapped = ma02gd(n, a.copy(), k1, k2, ipiv, incx)
    a_back = ma02gd(n, a_swapped.copy(), k1, k2, ipiv, incx)

    np.testing.assert_allclose(a_back, a_orig, rtol=1e-14)
