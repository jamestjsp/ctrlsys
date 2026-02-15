"""
Tests for SB04NV: Construct right-hand sides for Sylvester solver (2 RHS case).

This routine constructs right-hand sides D for a system of equations in
Hessenberg form solved via SB04NX (case with 2 right-hand sides).

Tests numerical correctness using:
1. Basic functionality with ABSCHR='B', UL='U'
2. Alternative mode ABSCHR='A', UL='U'
3. Lower Hessenberg case (UL='L')
4. Mathematical property: D structure verification
5. Edge cases (empty, 1x1)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb04nv_abschr_b_upper():
    """
    Validate basic functionality with ABSCHR='B' (AB contains B), UL='U' (upper).

    For ABSCHR='B':
    - D is 2*N elements (N = rows of C)
    - C is N-by-M, AB is M-by-M upper Hessenberg (matrix B)
    - D = [C[:, indx-1], C[:, indx]] interleaved, minus corrections from AB

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04nv

    np.random.seed(42)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)

    indx = 2

    d = sb04nv('B', 'U', indx, c, ab)

    assert d.shape == (2 * n,)

    d_expected_col1 = c[:, indx - 1].copy()
    d_expected_col2 = c[:, indx].copy()

    for k in range(indx - 1):
        d_expected_col1 -= c[:, k] * ab[k, indx - 1]
        d_expected_col2 -= c[:, k] * ab[k, indx]

    d_expected = np.empty(2 * n)
    d_expected[0::2] = d_expected_col1
    d_expected[1::2] = d_expected_col2

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nv_abschr_a_upper():
    """
    Validate with ABSCHR='A' (AB contains A), UL='U' (upper).

    For ABSCHR='A':
    - D is 2*M elements (M = columns of C)
    - C is N-by-M, AB is N-by-N upper Hessenberg (matrix A)
    - D = [C[indx-1, :], C[indx, :]] interleaved, minus corrections from AB

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04nv

    np.random.seed(123)
    n = 4
    m = 3

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(n, n).astype(float, order='F'), k=-1)

    indx = 1

    d = sb04nv('A', 'U', indx, c, ab)

    assert d.shape == (2 * m,)

    d_expected_row1 = c[indx - 1, :].copy()
    d_expected_row2 = c[indx, :].copy()

    for k in range(indx + 1, n):
        d_expected_row1 -= c[k, :] * ab[indx - 1, k]
        d_expected_row2 -= c[k, :] * ab[indx, k]

    d_expected = np.empty(2 * m)
    d_expected[0::2] = d_expected_row1
    d_expected[1::2] = d_expected_row2

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nv_lower_hessenberg():
    """
    Validate with UL='L' (lower Hessenberg).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04nv

    np.random.seed(456)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)

    indx = 2

    d = sb04nv('B', 'L', indx, c, ab)

    assert d.shape == (2 * n,)

    d_expected_col1 = c[:, indx - 1].copy()
    d_expected_col2 = c[:, indx].copy()

    for k in range(indx + 1, m):
        d_expected_col1 -= c[:, k] * ab[k, indx - 1]
        d_expected_col2 -= c[:, k] * ab[k, indx]

    d_expected = np.empty(2 * n)
    d_expected[0::2] = d_expected_col1
    d_expected[1::2] = d_expected_col2

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nv_first_index():
    """
    Test with indx=1 (first valid index).

    For ABSCHR='B', UL='U', indx=1 means no prior columns to subtract.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04nv

    np.random.seed(789)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)

    indx = 1

    d = sb04nv('B', 'U', indx, c, ab)

    d_expected = np.empty(2 * n)
    d_expected[0::2] = c[:, 0]
    d_expected[1::2] = c[:, 1]

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nv_last_index():
    """
    Test with indx at last valid position.

    For ABSCHR='B', UL='U', this tests the full accumulation.
    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import sb04nv

    np.random.seed(101)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)

    indx = m - 1

    d = sb04nv('B', 'U', indx, c, ab)

    d_expected_col1 = c[:, indx - 1].copy()
    d_expected_col2 = c[:, indx].copy()

    for k in range(indx - 1):
        d_expected_col1 -= c[:, k] * ab[k, indx - 1]
        d_expected_col2 -= c[:, k] * ab[k, indx]

    d_expected = np.empty(2 * n)
    d_expected[0::2] = d_expected_col1
    d_expected[1::2] = d_expected_col2

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nv_1x1():
    """
    Test with 1x1 matrix.
    """
    from ctrlsys import sb04nv

    c = np.array([[5.0]], dtype=float, order='F')
    ab = np.array([[3.0]], dtype=float, order='F')

    d = sb04nv('B', 'U', 1, c, ab)

    assert d.shape == (2,)


def test_sb04nv_identity_ab():
    """
    Test with identity AB matrix (no off-diagonal contributions).

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import sb04nv

    np.random.seed(202)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.eye(m, dtype=float, order='F')

    indx = 2

    d = sb04nv('B', 'U', indx, c, ab)

    d_expected = np.empty(2 * n)
    d_expected[0::2] = c[:, indx - 1]
    d_expected[1::2] = c[:, indx]

    assert_allclose(d, d_expected, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
