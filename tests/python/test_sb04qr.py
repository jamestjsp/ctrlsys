"""
Tests for SB04QR - Solve linear algebraic system with special compact storage.

Solves a linear system whose coefficient matrix has zeros below the third
subdiagonal and zero elements on the third subdiagonal with even column indices.
Matrix stored compactly, row-wise.
"""
import pytest
import numpy as np


def test_sb04qr_basic_4x4():
    """
    Test basic 4x4 system (m must be even).

    For m=4, D has length 4*4/2 + 4*4 = 8 + 16 = 24
    """
    from ctrlsys import sb04qr

    m = 4
    d_len = m * m // 2 + 4 * m  # 8 + 16 = 24

    d = np.zeros(d_len, dtype=float, order='F')
    # The matrix has special sparsity pattern
    # For m=4: row lengths are [4, 4, 4, 3] (decreasing by 2 for i>=4, even i)

    # Diagonally dominant values for stability
    d[0] = 5.0   # a11
    d[1] = 1.0   # a12
    d[2] = 0.5   # a13
    d[3] = 0.2   # a14
    d[4] = 1.0   # a21
    d[5] = 4.0   # a22
    d[6] = 1.0   # a23
    d[7] = 0.3   # a24
    d[8] = 0.8   # a31
    d[9] = 0.7   # a32
    d[10] = 3.5  # a33
    d[11] = 0.5  # a34
    d[12] = 0.6  # a42
    d[13] = 0.5  # a43
    d[14] = 3.0  # a44

    # RHS at position m*m/2 + 3*m = 8 + 12 = 20
    rhs_start = m * m // 2 + 3 * m
    d[rhs_start] = 7.7
    d[rhs_start + 1] = 6.3
    d[rhs_start + 2] = 5.0
    d[rhs_start + 3] = 3.6

    d_out, ipr, info = sb04qr(m, d.copy())

    assert info == 0 or info == 1

    # Extract solution
    x = np.zeros(m)
    for i in range(m):
        x[i] = d_out[ipr[i] - 1]

    # Verify finite solution
    assert np.all(np.isfinite(x))


def test_sb04qr_2x2():
    """
    Test smallest case m=2.

    For m=2, D has length 2*2/2 + 4*2 = 2 + 8 = 10
    """
    from ctrlsys import sb04qr

    m = 2
    d_len = m * m // 2 + 4 * m  # 2 + 8 = 10

    d = np.zeros(d_len, dtype=float, order='F')
    # For m=2, row lengths are [2, 2]
    d[0] = 3.0  # a11
    d[1] = 1.0  # a12
    d[2] = 1.0  # a21
    d[3] = 4.0  # a22

    # RHS at position m*m/2 + 3*m = 2 + 6 = 8
    d[8] = 5.0   # b1 = 3*1 + 1*2 = 5
    d[9] = 9.0   # b2 = 1*1 + 4*2 = 9

    d_out, ipr, info = sb04qr(m, d.copy())

    assert info == 0

    x = np.zeros(m)
    for i in range(m):
        x[i] = d_out[ipr[i] - 1]

    np.testing.assert_allclose(x, [1.0, 2.0], rtol=1e-12)


def test_sb04qr_singular():
    """
    Test detection of singular matrix.
    """
    from ctrlsys import sb04qr

    m = 2
    d_len = m * m // 2 + 4 * m

    d = np.zeros(d_len, dtype=float, order='F')
    d[8] = 1.0  # RHS only

    d_out, ipr, info = sb04qr(m, d.copy())

    assert info == 1


def test_sb04qr_m0():
    """
    Test with m=0 (empty system).
    """
    from ctrlsys import sb04qr

    d = np.array([], dtype=float, order='F')
    d_out, ipr, info = sb04qr(0, d)

    assert info == 0
