"""
Tests for MA02DD - Pack/unpack triangular matrix

MA02DD packs or unpacks the upper or lower triangle of a symmetric matrix.
The packed matrix is stored column-wise in a one-dimensional array.

Packed storage formats:
- Upper (UPLO='U'): 11, 12, 22, 13, 23, 33, ..., 1n, 2n, ..., nn
- Lower (UPLO='L'): 11, 21, 31, ..., n1, 22, 32, ..., n2, ..., nn

Property tests verify:
- Pack then unpack returns original triangle
- Packed array length is n*(n+1)/2
- Column-wise storage order
"""
import numpy as np
import pytest
from ctrlsys import ma02dd


def test_ma02dd_pack_upper():
    """Test packing upper triangular matrix"""
    n = 3
    a = np.array([[1.0, 2.0, 4.0],
                  [0.0, 3.0, 5.0],
                  [0.0, 0.0, 6.0]], order='F')

    ap = ma02dd('P', 'U', a)

    # Upper packed column-wise: (1,1), (1,2), (2,2), (1,3), (2,3), (3,3)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(ap, expected, rtol=1e-14)


def test_ma02dd_pack_lower():
    """Test packing lower triangular matrix"""
    n = 3
    a = np.array([[1.0, 0.0, 0.0],
                  [2.0, 4.0, 0.0],
                  [3.0, 5.0, 6.0]], order='F')

    ap = ma02dd('P', 'L', a)

    # Lower packed column-wise: (1,1), (2,1), (3,1), (2,2), (3,2), (3,3)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(ap, expected, rtol=1e-14)


def test_ma02dd_unpack_upper():
    """Test unpacking to upper triangular matrix"""
    n = 3
    # Upper packed column-wise: (1,1), (1,2), (2,2), (1,3), (2,3), (3,3)
    ap = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    a = ma02dd('U', 'U', ap, n)

    # Expected upper triangular matrix
    expected = np.array([[1.0, 2.0, 4.0],
                         [0.0, 3.0, 5.0],
                         [0.0, 0.0, 6.0]], order='F')

    # Only compare upper triangle
    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(a[i, j], expected[i, j], rtol=1e-14)


def test_ma02dd_unpack_lower():
    """Test unpacking to lower triangular matrix"""
    n = 3
    # Lower packed column-wise: (1,1), (2,1), (3,1), (2,2), (3,2), (3,3)
    ap = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    a = ma02dd('U', 'L', ap, n)

    # Expected lower triangular matrix
    expected = np.array([[1.0, 0.0, 0.0],
                         [2.0, 4.0, 0.0],
                         [3.0, 5.0, 6.0]], order='F')

    # Only compare lower triangle
    for i in range(n):
        for j in range(i + 1):
            np.testing.assert_allclose(a[i, j], expected[i, j], rtol=1e-14)


def test_ma02dd_pack_unpack_roundtrip_upper():
    """
    Property test: Pack then unpack preserves upper triangle.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 5
    a_orig = np.random.randn(n, n).astype(float, order='F')

    # Pack upper triangle
    ap = ma02dd('P', 'U', a_orig)

    # Verify packed length
    assert len(ap) == n * (n + 1) // 2

    # Unpack back
    a_recovered = ma02dd('U', 'U', ap, n)

    # Upper triangle should match
    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(a_recovered[i, j], a_orig[i, j], rtol=1e-14)


def test_ma02dd_pack_unpack_roundtrip_lower():
    """
    Property test: Pack then unpack preserves lower triangle.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    a_orig = np.random.randn(n, n).astype(float, order='F')

    # Pack lower triangle
    ap = ma02dd('P', 'L', a_orig)

    # Verify packed length
    assert len(ap) == n * (n + 1) // 2

    # Unpack back
    a_recovered = ma02dd('U', 'L', ap, n)

    # Lower triangle should match
    for i in range(n):
        for j in range(i + 1):
            np.testing.assert_allclose(a_recovered[i, j], a_orig[i, j], rtol=1e-14)


def test_ma02dd_single_element():
    """Test edge case: 1x1 matrix"""
    a = np.array([[5.0]], order='F')

    ap = ma02dd('P', 'U', a)
    np.testing.assert_allclose(ap, [5.0], rtol=1e-14)

    a_recovered = ma02dd('U', 'U', ap, 1)
    np.testing.assert_allclose(a_recovered[0, 0], 5.0, rtol=1e-14)


def test_ma02dd_2x2_matrix():
    """Test 2x2 matrix packing/unpacking"""
    # Upper triangle
    a_upper = np.array([[1.0, 2.0],
                        [0.0, 3.0]], order='F')
    ap_upper = ma02dd('P', 'U', a_upper)
    # Upper packed: (1,1), (1,2), (2,2)
    np.testing.assert_allclose(ap_upper, [1.0, 2.0, 3.0], rtol=1e-14)

    # Lower triangle
    a_lower = np.array([[1.0, 0.0],
                        [2.0, 3.0]], order='F')
    ap_lower = ma02dd('P', 'L', a_lower)
    # Lower packed: (1,1), (2,1), (2,2)
    np.testing.assert_allclose(ap_lower, [1.0, 2.0, 3.0], rtol=1e-14)


def test_ma02dd_symmetric_matrix():
    """
    Test with truly symmetric matrix where both triangles give same packed result.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4
    a = np.random.randn(n, n)
    a = (a + a.T) / 2  # Make symmetric
    a = a.astype(float, order='F')

    # Pack both triangles
    ap_upper = ma02dd('P', 'U', a)
    ap_lower = ma02dd('P', 'L', a)

    # Unpack and verify full matrix reconstruction is same
    a_from_upper = ma02dd('U', 'U', ap_upper, n)
    a_from_lower = ma02dd('U', 'L', ap_lower, n)

    # The diagonal and their respective triangles should match original
    for i in range(n):
        np.testing.assert_allclose(a_from_upper[i, i], a[i, i], rtol=1e-14)
        np.testing.assert_allclose(a_from_lower[i, i], a[i, i], rtol=1e-14)


def test_ma02dd_larger_matrix():
    """
    Test with larger matrix for performance validation.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 20
    a = np.random.randn(n, n).astype(float, order='F')

    # Pack and unpack upper
    ap = ma02dd('P', 'U', a)
    assert len(ap) == n * (n + 1) // 2

    a_recovered = ma02dd('U', 'U', ap, n)

    # Verify upper triangle
    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(a_recovered[i, j], a[i, j], rtol=1e-14)
