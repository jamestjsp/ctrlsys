"""
Tests for MA02GZ: Column interchanges on a complex matrix.

Performs series of column swaps based on pivot indices.
Column-oriented counterpart of LAPACK's ZLASWP (row swaps) for complex matrices.
Complex version of MA02GD.
"""
import numpy as np
import pytest
from ctrlsys import ma02gz


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
        [1.0+1j, 2.0+2j, 3.0+3j],
        [4.0+4j, 5.0+5j, 6.0+6j],
        [7.0+7j, 8.0+8j, 9.0+9j]
    ], order='F', dtype=complex)

    ipiv = np.array([1, 2], dtype=np.int32)

    a_out = ma02gz(n, a, k1, k2, ipiv, incx)

    expected = np.array([
        [1.0+1j, 2.0+2j, 3.0+3j],
        [4.0+4j, 5.0+5j, 6.0+6j],
        [7.0+7j, 8.0+8j, 9.0+9j]
    ], order='F', dtype=complex)

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
        [1.0+0j, 2.0+1j, 3.0+2j],
        [4.0-1j, 5.0+0j, 6.0+1j]
    ], order='F', dtype=complex)

    ipiv = np.array([3], dtype=np.int32)

    a_out = ma02gz(n, a, k1, k2, ipiv, incx)

    expected = np.array([
        [3.0+2j, 2.0+1j, 1.0+0j],
        [6.0+1j, 5.0+0j, 4.0-1j]
    ], order='F', dtype=complex)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_multiple_swaps():
    """
    Multiple swaps: swap col 1 with col 2, then col 2 with col 3.
    """
    n = 2
    k1, k2 = 1, 2
    incx = 1

    a = np.array([
        [1.0+1j, 2.0+2j, 3.0+3j],
        [4.0+4j, 5.0+5j, 6.0+6j]
    ], order='F', dtype=complex)

    ipiv = np.array([2, 3], dtype=np.int32)

    a_out = ma02gz(n, a, k1, k2, ipiv, incx)

    # After swap 1: cols [2,1,3] -> [2+2j, 1+1j, 3+3j]
    # After swap 2: cols [2,3,1] -> [2+2j, 3+3j, 1+1j]
    expected = np.array([
        [2.0+2j, 3.0+3j, 1.0+1j],
        [5.0+5j, 6.0+6j, 4.0+4j]
    ], order='F', dtype=complex)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_negative_incx():
    """
    Negative INCX: swaps in reverse order.
    """
    n = 2
    k1, k2 = 1, 2
    incx = -1

    a = np.array([
        [1.0+0j, 2.0+0j, 3.0+0j],
        [4.0+0j, 5.0+0j, 6.0+0j]
    ], order='F', dtype=complex)

    ipiv = np.array([2, 3], dtype=np.int32)

    a_out = ma02gz(n, a, k1, k2, ipiv, incx)

    # With incx=-1, loop goes K2 down to K1
    # After swap col 2 with col 3: [1,3,2]
    # After swap col 1 with col 2: [3,1,2]
    expected = np.array([
        [3.0+0j, 1.0+0j, 2.0+0j],
        [6.0+0j, 4.0+0j, 5.0+0j]
    ], order='F', dtype=complex)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)


"""Edge case tests."""

def test_incx_zero():
    """INCX=0 should return without changes (quick return)."""
    n = 2
    k1, k2 = 1, 2
    incx = 0

    a = np.array([
        [1.0+1j, 2.0+2j],
        [3.0+3j, 4.0+4j]
    ], order='F', dtype=complex)

    ipiv = np.array([2, 1], dtype=np.int32)
    a_orig = a.copy()

    a_out = ma02gz(n, a, k1, k2, ipiv, incx)

    np.testing.assert_allclose(a_out, a_orig, rtol=1e-14)

def test_n_zero():
    """N=0 should return without changes (quick return)."""
    n = 0
    k1, k2 = 1, 2
    incx = 1

    a = np.zeros((0, 3), order='F', dtype=complex)
    ipiv = np.array([2, 1], dtype=np.int32)

    a_out = ma02gz(n, a, k1, k2, ipiv, incx)

    assert a_out.shape == (0, 3)

def test_single_row():
    """Single row complex matrix."""
    n = 1
    k1, k2 = 1, 3
    incx = 1

    a = np.array([[1.0+1j, 2.0+2j, 3.0+3j, 4.0+4j]], order='F', dtype=complex)
    ipiv = np.array([2, 3, 4], dtype=np.int32)

    a_out = ma02gz(n, a, k1, k2, ipiv, incx)

    # Swap col1 with col2: [2+2j,1+1j,3+3j,4+4j]
    # Swap col2 with col3: [2+2j,3+3j,1+1j,4+4j]
    # Swap col3 with col4: [2+2j,3+3j,4+4j,1+1j]
    expected = np.array([[2.0+2j, 3.0+3j, 4.0+4j, 1.0+1j]], order='F', dtype=complex)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)


"""Mathematical property validation tests."""

def test_involution():
    """
    Applying identity permutation should not change matrix.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m = 4, 4
    k1, k2 = 1, 4
    incx = 1

    a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    a_orig = a.copy()

    ipiv = np.array([1, 2, 3, 4], dtype=np.int32)

    a_out = ma02gz(n, a, k1, k2, ipiv, incx)

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

    a = (np.random.randn(n, 3) + 1j * np.random.randn(n, 3)).astype(complex, order='F')
    a_orig = a.copy()

    ipiv = np.array([3], dtype=np.int32)

    a_swapped = ma02gz(n, a.copy(), k1, k2, ipiv, incx)
    a_back = ma02gz(n, a_swapped.copy(), k1, k2, ipiv, incx)

    np.testing.assert_allclose(a_back, a_orig, rtol=1e-14)

def test_preserves_column_norms():
    """
    Column swapping should preserve Frobenius norm.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m = 5, 4
    k1, k2 = 1, 3
    incx = 1

    a = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    norm_before = np.linalg.norm(a, 'fro')

    ipiv = np.array([2, 3, 4], dtype=np.int32)

    a_out = ma02gz(n, a.copy(), k1, k2, ipiv, incx)
    norm_after = np.linalg.norm(a_out, 'fro')

    np.testing.assert_allclose(norm_after, norm_before, rtol=1e-14)

def test_incx_greater_than_one():
    """
    Test with INCX > 1 (stride through IPIV).
    """
    n = 2
    k1, k2 = 1, 2
    incx = 2

    a = np.array([
        [1.0+0j, 2.0+0j, 3.0+0j],
        [4.0+0j, 5.0+0j, 6.0+0j]
    ], order='F', dtype=complex)

    # With incx=2, JX starts at k1-1=0
    # K=1: JP = IPIV[0] = 3, swap col 1 with col 3
    # K=2: JP = IPIV[2] = 1, swap col 2 with col 1
    ipiv = np.array([3, 0, 1], dtype=np.int32)

    a_out = ma02gz(n, a, k1, k2, ipiv, incx)

    # After swap col 1 with col 3: [3,2,1]
    # After swap col 2 with col 1: [2,3,1]
    expected = np.array([
        [2.0+0j, 3.0+0j, 1.0+0j],
        [5.0+0j, 6.0+0j, 4.0+0j]
    ], order='F', dtype=complex)

    np.testing.assert_allclose(a_out, expected, rtol=1e-14)
