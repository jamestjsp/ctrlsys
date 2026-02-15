"""
Tests for SB04MW - Solve linear algebraic system with upper Hessenberg compact storage.

Solves a linear system whose coefficient matrix is in upper Hessenberg form,
stored compactly, row-wise. Uses Gaussian elimination with partial pivoting.
"""
import pytest
import numpy as np


def test_sb04mw_basic_2x2():
    """
    Test basic 2x2 system.

    For m=2, D has length 2*(2+1)/2 + 2*2 = 3 + 4 = 7
    """
    from ctrlsys import sb04mw

    m = 2
    d_len = m * (m + 1) // 2 + 2 * m  # 3 + 4 = 7

    d = np.zeros(d_len, dtype=float, order='F')
    # Upper Hessenberg 2x2: [[a11, a12], [a21, a22]]
    # Row 1: a11=2, a12=1
    # Row 2: a21=1, a22=3
    d[0] = 2.0
    d[1] = 1.0
    d[2] = 1.0
    d[3] = 3.0
    # RHS
    d[5] = 5.0   # b1 = 2*1 + 1*3 = 5
    d[6] = 10.0  # b2 = 1*1 + 3*3 = 10

    d_out, ipr, info = sb04mw(m, d.copy())

    assert info == 0

    # Extract solution
    x = np.zeros(m)
    for i in range(m):
        x[i] = d_out[ipr[i] - 1]

    np.testing.assert_allclose(x, [1.0, 3.0], rtol=1e-12)


def test_sb04mw_3x3():
    """
    Test 3x3 upper Hessenberg system.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04mw

    np.random.seed(42)

    m = 3
    d_len = m * (m + 1) // 2 + 2 * m  # 6 + 6 = 12

    d = np.zeros(d_len, dtype=float, order='F')
    # Upper Hessenberg: nonzeros on and above main diagonal + one subdiagonal
    # Row 1: a11, a12, a13 (3 elements)
    # Row 2: a21, a22, a23 (3 elements)
    # Row 3: a32, a33 (2 elements)

    # Diagonally dominant for stability
    d[0] = 5.0   # a11
    d[1] = 1.0   # a12
    d[2] = 0.5   # a13
    d[3] = 1.0   # a21
    d[4] = 4.0   # a22
    d[5] = 1.0   # a23
    d[6] = 0.8   # a32
    d[7] = 3.5   # a33

    # RHS for known solution [1, 2, 3]
    d[9] = 5.0*1 + 1.0*2 + 0.5*3    # 8.5
    d[10] = 1.0*1 + 4.0*2 + 1.0*3   # 12.0
    d[11] = 0.8*2 + 3.5*3           # 12.1

    d_out, ipr, info = sb04mw(m, d.copy())

    assert info == 0

    x = np.zeros(m)
    for i in range(m):
        x[i] = d_out[ipr[i] - 1]

    np.testing.assert_allclose(x, [1.0, 2.0, 3.0], rtol=1e-10)


def test_sb04mw_singular():
    """
    Test detection of singular matrix.
    """
    from ctrlsys import sb04mw

    m = 2
    d_len = m * (m + 1) // 2 + 2 * m

    d = np.zeros(d_len, dtype=float, order='F')
    d[5] = 1.0  # RHS only

    d_out, ipr, info = sb04mw(m, d.copy())

    assert info == 1


def test_sb04mw_m0():
    """
    Test with m=0 (empty system).
    """
    from ctrlsys import sb04mw

    d = np.array([], dtype=float, order='F')
    d_out, ipr, info = sb04mw(0, d)

    assert info == 0
