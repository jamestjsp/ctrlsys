"""
Tests for SB04RV: Construct right-hand sides for quasi-Hessenberg Sylvester solver (2 RHS).

This routine constructs right-hand sides D for a system of equations in
quasi-Hessenberg form solved via SB04RX (case with 2 right-hand sides).

For the Sylvester equation X + AXB = C:
- If ABSCHR='B': AB contains B (M-by-M), BA contains A (N-by-N)
  - Output D has 2*N elements (interleaved columns)
  - Computes D from columns INDX and INDX+1 of C with corrections
- If ABSCHR='A': AB contains A (N-by-N), BA contains B (M-by-M)
  - Output D has 2*M elements (interleaved rows)
  - Computes D from rows INDX and INDX+1 of C with corrections

Tests numerical correctness using:
1. Basic functionality with ABSCHR='B', UL='U'
2. Alternative mode ABSCHR='A', UL='U'
3. Lower Hessenberg case (UL='L')
4. Edge cases: first index, last index
5. Mathematical verification against manual computation
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb04rv_abschr_b_upper():
    """
    Validate basic functionality with ABSCHR='B' (AB contains B), UL='U' (upper).

    For ABSCHR='B', UL='U':
    - D is 2*N elements, stored as 2-row matrix (interleaved)
    - D[:, j] = C[:, indx+j-1] - BA * sum_k(C[:, k] * AB[k, indx+j-1]) for k < indx

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04rv

    np.random.seed(42)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)  # M-by-M upper Hessenberg
    ba = np.random.randn(n, n).astype(float, order='F')  # N-by-N

    indx = 2

    d = sb04rv('B', 'U', indx, c, ab, ba)

    assert d.shape == (2 * n,)

    # Manual computation of expected D
    # Step 1: Start with columns INDX-1 and INDX of C (0-indexed: indx-1 and indx)
    d_expected_col1 = c[:, indx - 1].copy()
    d_expected_col2 = c[:, indx].copy()

    # Step 2: For UL='U' with indx > 1, compute dwork = C[:, 0:indx-1] @ AB[0:indx-1, indx-1:indx+1]
    if indx > 1:
        dwork1 = c[:, :indx - 1] @ ab[:indx - 1, indx - 1]
        dwork2 = c[:, :indx - 1] @ ab[:indx - 1, indx]
        # Step 3: D -= BA @ dwork
        d_expected_col1 -= ba @ dwork1
        d_expected_col2 -= ba @ dwork2

    d_expected = np.empty(2 * n)
    d_expected[0::2] = d_expected_col1
    d_expected[1::2] = d_expected_col2

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rv_abschr_a_upper():
    """
    Validate with ABSCHR='A' (AB contains A), UL='U' (upper).

    For ABSCHR='A', UL='U':
    - D is 2*M elements
    - Works with rows INDX and INDX+1 of C
    - For upper Hessenberg, accumulates from rows > indx+1

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04rv

    np.random.seed(123)
    n = 4
    m = 3

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(n, n).astype(float, order='F'), k=-1)  # N-by-N upper Hessenberg
    ba = np.random.randn(m, m).astype(float, order='F')  # M-by-M

    indx = 1

    d = sb04rv('A', 'U', indx, c, ab, ba)

    assert d.shape == (2 * m,)

    # Manual computation
    d_expected_row1 = c[indx - 1, :].copy()
    d_expected_row2 = c[indx, :].copy()

    # For UL='U' with indx < n-1, accumulate from rows > indx+1
    if indx < n - 1:
        # dwork = C[indx+1:, :].T @ AB[indx-1:indx+1, indx+1:].T
        dwork1 = c[indx + 1:, :].T @ ab[indx - 1, indx + 1:]
        dwork2 = c[indx + 1:, :].T @ ab[indx, indx + 1:]
        # D -= BA.T @ dwork
        d_expected_row1 -= ba.T @ dwork1
        d_expected_row2 -= ba.T @ dwork2

    d_expected = np.empty(2 * m)
    d_expected[0::2] = d_expected_row1
    d_expected[1::2] = d_expected_row2

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rv_lower_hessenberg_b():
    """
    Validate with UL='L' (lower Hessenberg), ABSCHR='B'.

    For ABSCHR='B', UL='L':
    - Works with columns INDX and INDX+1 of C
    - For lower Hessenberg, accumulates from columns > indx+1

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04rv

    np.random.seed(456)
    n = 3
    m = 5

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)  # M-by-M lower Hessenberg
    ba = np.random.randn(n, n).astype(float, order='F')  # N-by-N

    indx = 2

    d = sb04rv('B', 'L', indx, c, ab, ba)

    assert d.shape == (2 * n,)

    # Manual computation
    d_expected_col1 = c[:, indx - 1].copy()
    d_expected_col2 = c[:, indx].copy()

    # For UL='L' with indx < m-1, accumulate from columns > indx+1
    if indx < m - 1:
        dwork1 = c[:, indx + 1:] @ ab[indx + 1:, indx - 1]
        dwork2 = c[:, indx + 1:] @ ab[indx + 1:, indx]
        d_expected_col1 -= ba @ dwork1
        d_expected_col2 -= ba @ dwork2

    d_expected = np.empty(2 * n)
    d_expected[0::2] = d_expected_col1
    d_expected[1::2] = d_expected_col2

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rv_lower_hessenberg_a():
    """
    Validate with UL='L' (lower Hessenberg), ABSCHR='A'.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04rv

    np.random.seed(789)
    n = 5
    m = 3

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.tril(np.random.randn(n, n).astype(float, order='F'), k=1)  # N-by-N lower Hessenberg
    ba = np.random.randn(m, m).astype(float, order='F')  # M-by-M

    indx = 3

    d = sb04rv('A', 'L', indx, c, ab, ba)

    assert d.shape == (2 * m,)

    # Manual computation
    d_expected_row1 = c[indx - 1, :].copy()
    d_expected_row2 = c[indx, :].copy()

    # For UL='L' with indx > 1, accumulate from rows < indx
    if indx > 1:
        dwork1 = c[:indx - 1, :].T @ ab[indx - 1, :indx - 1]
        dwork2 = c[:indx - 1, :].T @ ab[indx, :indx - 1]
        d_expected_row1 -= ba.T @ dwork1
        d_expected_row2 -= ba.T @ dwork2

    d_expected = np.empty(2 * m)
    d_expected[0::2] = d_expected_row1
    d_expected[1::2] = d_expected_row2

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rv_first_index():
    """
    Test with indx=1 (first valid index).

    For ABSCHR='B', UL='U', indx=1 means no prior columns to process.
    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import sb04rv

    np.random.seed(101)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    ba = np.random.randn(n, n).astype(float, order='F')

    indx = 1

    d = sb04rv('B', 'U', indx, c, ab, ba)

    # With indx=1 and UL='U', no corrections needed
    d_expected = np.empty(2 * n)
    d_expected[0::2] = c[:, 0]
    d_expected[1::2] = c[:, 1]

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rv_last_index():
    """
    Test with indx at last valid position for 2 RHS.

    For ABSCHR='B', UL='U', tests full accumulation path.
    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import sb04rv

    np.random.seed(202)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    ba = np.random.randn(n, n).astype(float, order='F')

    indx = m - 1  # Last valid index for 2 columns

    d = sb04rv('B', 'U', indx, c, ab, ba)

    d_expected_col1 = c[:, indx - 1].copy()
    d_expected_col2 = c[:, indx].copy()

    if indx > 1:
        dwork1 = c[:, :indx - 1] @ ab[:indx - 1, indx - 1]
        dwork2 = c[:, :indx - 1] @ ab[:indx - 1, indx]
        d_expected_col1 -= ba @ dwork1
        d_expected_col2 -= ba @ dwork2

    d_expected = np.empty(2 * n)
    d_expected[0::2] = d_expected_col1
    d_expected[1::2] = d_expected_col2

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rv_identity_ba():
    """
    Test with identity BA matrix (simplifies to no transformation).

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import sb04rv

    np.random.seed(303)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    ba = np.eye(n, dtype=float, order='F')  # Identity

    indx = 2

    d = sb04rv('B', 'U', indx, c, ab, ba)

    d_expected_col1 = c[:, indx - 1].copy()
    d_expected_col2 = c[:, indx].copy()

    if indx > 1:
        dwork1 = c[:, :indx - 1] @ ab[:indx - 1, indx - 1]
        dwork2 = c[:, :indx - 1] @ ab[:indx - 1, indx]
        d_expected_col1 -= dwork1  # BA is identity
        d_expected_col2 -= dwork2

    d_expected = np.empty(2 * n)
    d_expected[0::2] = d_expected_col1
    d_expected[1::2] = d_expected_col2

    assert_allclose(d, d_expected, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
