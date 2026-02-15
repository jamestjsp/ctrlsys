"""
Tests for SB04RW: Construct right-hand side for Hessenberg Sylvester solver (1 RHS).

This routine constructs right-hand side D for a system of equations in
Hessenberg form solved via SB04RY (case with 1 right-hand side).

For the Sylvester equation X + AXB = C:
- If ABSCHR='B': AB contains B (M-by-M), BA contains A (N-by-N)
  - Output D has N elements (column)
  - Computes D from column INDX of C with corrections
- If ABSCHR='A': AB contains A (N-by-N), BA contains B (M-by-M)
  - Output D has M elements (row)
  - Computes D from row INDX of C with corrections

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


def test_sb04rw_abschr_b_upper():
    """
    Validate basic functionality with ABSCHR='B' (AB contains B), UL='U' (upper).

    For ABSCHR='B', UL='U':
    - D is N elements (single column)
    - D = C[:, indx-1] - BA * C[:, 0:indx-1] @ AB[0:indx-1, indx-1] for indx > 1

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04rw

    np.random.seed(42)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)  # M-by-M upper Hessenberg
    ba = np.random.randn(n, n).astype(float, order='F')  # N-by-N

    indx = 3

    d = sb04rw('B', 'U', indx, c, ab, ba)

    assert d.shape == (n,)

    # Manual computation of expected D
    # Step 1: Start with column INDX of C (1-indexed: indx, 0-indexed: indx-1)
    d_expected = c[:, indx - 1].copy()

    # Step 2: For UL='U' with indx > 1, compute dwork = C[:, 0:indx-1] @ AB[0:indx-1, indx-1]
    if indx > 1:
        dwork = c[:, :indx - 1] @ ab[:indx - 1, indx - 1]
        # Step 3: D -= BA @ dwork
        d_expected -= ba @ dwork

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rw_abschr_a_upper():
    """
    Validate with ABSCHR='A' (AB contains A), UL='U' (upper).

    For ABSCHR='A', UL='U':
    - D is M elements (single row)
    - Works with row INDX of C
    - For upper Hessenberg, accumulates from rows > indx

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04rw

    np.random.seed(123)
    n = 4
    m = 3

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(n, n).astype(float, order='F'), k=-1)  # N-by-N upper Hessenberg
    ba = np.random.randn(m, m).astype(float, order='F')  # M-by-M

    indx = 2

    d = sb04rw('A', 'U', indx, c, ab, ba)

    assert d.shape == (m,)

    # Manual computation
    d_expected = c[indx - 1, :].copy()

    # For UL='U' with indx < n, accumulate from rows > indx
    if indx < n:
        # dwork = C[indx:, :].T @ AB[indx-1, indx:].T
        dwork = c[indx:, :].T @ ab[indx - 1, indx:]
        # D -= BA.T @ dwork
        d_expected -= ba.T @ dwork

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rw_lower_hessenberg_b():
    """
    Validate with UL='L' (lower Hessenberg), ABSCHR='B'.

    For ABSCHR='B', UL='L':
    - Works with column INDX of C
    - For lower Hessenberg, accumulates from columns > indx

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04rw

    np.random.seed(456)
    n = 3
    m = 5

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)  # M-by-M lower Hessenberg
    ba = np.random.randn(n, n).astype(float, order='F')  # N-by-N

    indx = 2

    d = sb04rw('B', 'L', indx, c, ab, ba)

    assert d.shape == (n,)

    # Manual computation
    d_expected = c[:, indx - 1].copy()

    # For UL='L' with indx < m, accumulate from columns > indx
    if indx < m:
        dwork = c[:, indx:] @ ab[indx:, indx - 1]
        d_expected -= ba @ dwork

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rw_lower_hessenberg_a():
    """
    Validate with UL='L' (lower Hessenberg), ABSCHR='A'.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04rw

    np.random.seed(789)
    n = 5
    m = 3

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.tril(np.random.randn(n, n).astype(float, order='F'), k=1)  # N-by-N lower Hessenberg
    ba = np.random.randn(m, m).astype(float, order='F')  # M-by-M

    indx = 3

    d = sb04rw('A', 'L', indx, c, ab, ba)

    assert d.shape == (m,)

    # Manual computation
    d_expected = c[indx - 1, :].copy()

    # For UL='L' with indx > 1, accumulate from rows < indx
    if indx > 1:
        dwork = c[:indx - 1, :].T @ ab[indx - 1, :indx - 1]
        d_expected -= ba.T @ dwork

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rw_first_index():
    """
    Test with indx=1 (first valid index).

    For ABSCHR='B', UL='U', indx=1 means no prior columns to process.
    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import sb04rw

    np.random.seed(101)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    ba = np.random.randn(n, n).astype(float, order='F')

    indx = 1

    d = sb04rw('B', 'U', indx, c, ab, ba)

    # With indx=1 and UL='U', no corrections needed
    d_expected = c[:, 0].copy()

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rw_last_index():
    """
    Test with indx at last valid position.

    For ABSCHR='B', UL='U', tests full accumulation path.
    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import sb04rw

    np.random.seed(202)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    ba = np.random.randn(n, n).astype(float, order='F')

    indx = m  # Last column

    d = sb04rw('B', 'U', indx, c, ab, ba)

    d_expected = c[:, indx - 1].copy()

    if indx > 1:
        dwork = c[:, :indx - 1] @ ab[:indx - 1, indx - 1]
        d_expected -= ba @ dwork

    assert_allclose(d, d_expected, rtol=1e-14)


def test_sb04rw_identity_ba():
    """
    Test with identity BA matrix (simplifies to no transformation).

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import sb04rw

    np.random.seed(303)
    n = 3
    m = 4

    c = np.random.randn(n, m).astype(float, order='F')
    ab = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    ba = np.eye(n, dtype=float, order='F')  # Identity

    indx = 3

    d = sb04rw('B', 'U', indx, c, ab, ba)

    d_expected = c[:, indx - 1].copy()

    if indx > 1:
        dwork = c[:, :indx - 1] @ ab[:indx - 1, indx - 1]
        d_expected -= dwork  # BA is identity

    assert_allclose(d, d_expected, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
