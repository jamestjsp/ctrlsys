"""
Tests for SB04NW: Construct right-hand side for Sylvester solver (1 RHS case).

This routine constructs right-hand side D for a system of equations in
Hessenberg form solved via SB04NY (case with 1 right-hand side).

Tests numerical correctness using:
1. Basic functionality with ABSCHR='B', UL='U'
2. Alternative mode ABSCHR='A', UL='U'
3. Lower Hessenberg case (UL='L')
4. Mathematical property: D structure verification
5. Edge cases (1x1)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb04nw_abschr_b_upper():
    """
    Validate basic functionality with ABSCHR='B' (AB contains B), UL='U' (upper).

    For ABSCHR='B':
    - D is N elements (N = rows of C)
    - C is N-by-M, AB is M-by-M upper Hessenberg (matrix B)
    - D = C[:, indx-1] minus corrections from prior columns

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04nw

    np.random.seed(42)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)

    indx = 3

    d = sb04nw('B', 'U', indx, c, ab)

    assert d.shape == (n,)

    d_expected = c[:, indx - 1].copy()
    for k in range(indx - 1):
        d_expected -= c[:, k] * ab[k, indx - 1]

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nw_abschr_a_upper():
    """
    Validate with ABSCHR='A' (AB contains A), UL='U' (upper).

    For ABSCHR='A':
    - D is M elements (M = columns of C)
    - C is N-by-M, AB is N-by-N upper Hessenberg (matrix A)
    - D = C[indx-1, :] minus corrections from subsequent rows

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04nw

    np.random.seed(123)
    n = 4
    m = 3

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(n, n).astype(float, order='F'), k=-1)

    indx = 2

    d = sb04nw('A', 'U', indx, c, ab)

    assert d.shape == (m,)

    d_expected = c[indx - 1, :].copy()
    for k in range(indx, n):
        d_expected -= c[k, :] * ab[indx - 1, k]

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nw_lower_hessenberg():
    """
    Validate with UL='L' (lower Hessenberg).

    For ABSCHR='B', UL='L':
    - Corrections come from columns after indx

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04nw

    np.random.seed(456)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)

    indx = 2

    d = sb04nw('B', 'L', indx, c, ab)

    assert d.shape == (n,)

    d_expected = c[:, indx - 1].copy()
    for k in range(indx, m):
        d_expected -= c[:, k] * ab[k, indx - 1]

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nw_abschr_a_lower():
    """
    Validate with ABSCHR='A', UL='L' (lower Hessenberg A).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04nw

    np.random.seed(789)
    n = 4
    m = 3

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.tril(np.random.randn(n, n).astype(float, order='F'), k=1)

    indx = 3

    d = sb04nw('A', 'L', indx, c, ab)

    assert d.shape == (m,)

    d_expected = c[indx - 1, :].copy()
    for k in range(indx - 1):
        d_expected -= c[k, :] * ab[indx - 1, k]

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nw_first_index():
    """
    Test with indx=1 (first valid index).

    For ABSCHR='B', UL='U', indx=1 means no prior columns to subtract.
    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import sb04nw

    np.random.seed(101)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)

    indx = 1

    d = sb04nw('B', 'U', indx, c, ab)

    assert_allclose(d, c[:, 0], rtol=1e-14)


def test_sb04nw_last_index():
    """
    Test with indx at last position.

    For ABSCHR='B', UL='U', this tests the full accumulation.
    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import sb04nw

    np.random.seed(202)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)

    indx = m

    d = sb04nw('B', 'U', indx, c, ab)

    d_expected = c[:, indx - 1].copy()
    for k in range(indx - 1):
        d_expected -= c[:, k] * ab[k, indx - 1]

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04nw_1x1():
    """
    Test with 1x1 matrix.
    """
    from ctrlsys import sb04nw

    c = np.array([[5.0]], dtype=float, order='F')
    ab = np.array([[3.0]], dtype=float, order='F')

    d = sb04nw('B', 'U', 1, c, ab)

    assert d.shape == (1,)
    assert_allclose(d, np.array([5.0]), rtol=1e-14)


def test_sb04nw_identity_ab():
    """
    Test with identity AB matrix (no off-diagonal contributions).

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import sb04nw

    np.random.seed(303)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.eye(m, dtype=float, order='F')

    indx = 3

    d = sb04nw('B', 'U', indx, c, ab)

    assert_allclose(d, c[:, indx - 1], rtol=1e-14)


def test_sb04nw_compare_different_indices():
    """
    Compare results for different indices on same data.

    Random seed: 404 (for reproducibility)
    """
    from ctrlsys import sb04nw

    np.random.seed(404)
    n = 3
    m = 5

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)

    results = []
    for indx in range(1, m + 1):
        d = sb04nw('B', 'U', indx, c, ab)
        results.append(d.copy())

    for i, d in enumerate(results):
        indx = i + 1
        d_expected = c[:, indx - 1].copy()
        for k in range(indx - 1):
            d_expected -= c[:, k] * ab[k, indx - 1]
        assert_allclose(d, d_expected, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
