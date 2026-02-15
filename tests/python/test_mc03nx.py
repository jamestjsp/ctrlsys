"""
Tests for MC03NX - Companion pencil construction from polynomial matrix.

Constructs the pencil s*E-A related to a polynomial matrix P(s) of degree dp:
    P(s) = P(0) + P(1)*s + ... + P(dp)*s^dp

Output matrices A and E have structure:
    | I              |           | O          -P(dp) |
    |   .            |           | I .           .   |
A = |     .          |  and  E = |   . .         .   |
    |       .        |           |     . O       .   |
    |         I      |           |       I  O -P(2)  |
    |           P(0) |           |          I -P(1)  |

A is DP*MP by (DP-1)*MP+NP
E is DP*MP by (DP-1)*MP+NP

This is a helper routine for MC03ND.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mc03nx_basic_2x2_degree2():
    """
    Test basic case: 2x2 polynomial matrix of degree 2.

    P(s) = P(0) + P(1)*s + P(2)*s^2

    A dimensions: DP*MP x (DP-1)*MP+NP = 2*2 x 1*2+2 = 4x4
    E dimensions: same as A = 4x4

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc03nx

    mp, np_, dp = 2, 2, 2

    P0 = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    P1 = np.array([[5.0, 6.0], [7.0, 8.0]], order='F', dtype=float)
    P2 = np.array([[9.0, 10.0], [11.0, 12.0]], order='F', dtype=float)

    p = np.zeros((mp, np_, dp + 1), order='F', dtype=float)
    p[:, :, 0] = P0
    p[:, :, 1] = P1
    p[:, :, 2] = P2

    a, e = mc03nx(p)

    nrows = dp * mp
    ncols = (dp - 1) * mp + np_
    assert a.shape == (nrows, ncols), f"A shape {a.shape} != ({nrows}, {ncols})"
    assert e.shape == (nrows, ncols), f"E shape {e.shape} != ({nrows}, {ncols})"

    # Verify A structure: identity block on left, P(0) on bottom-right
    # A = | I     0  |
    #     | 0   P(0) |
    assert_allclose(a[:2, :2], np.eye(2), rtol=1e-14)
    assert_allclose(a[2:, 2:], P0, rtol=1e-14)
    assert_allclose(a[:2, 2:], np.zeros((2, 2)), rtol=1e-14)
    assert_allclose(a[2:, :2], np.zeros((2, 2)), rtol=1e-14)


def test_mc03nx_1x1_degree1():
    """
    Test minimal case: 1x1 polynomial matrix of degree 1.

    P(s) = P(0) + P(1)*s

    A dimensions: 1*1 x 0*1+1 = 1x1
    E dimensions: 1x1
    """
    from ctrlsys import mc03nx

    mp, np_, dp = 1, 1, 1

    P0 = np.array([[[2.0]]], order='F', dtype=float)
    P1 = np.array([[[3.0]]], order='F', dtype=float)

    p = np.zeros((mp, np_, dp + 1), order='F', dtype=float)
    p[0, 0, 0] = 2.0
    p[0, 0, 1] = 3.0

    a, e = mc03nx(p)

    assert a.shape == (1, 1)
    assert e.shape == (1, 1)

    assert_allclose(a[0, 0], 2.0, rtol=1e-14)
    assert_allclose(e[0, 0], -3.0, rtol=1e-14)


def test_mc03nx_2x3_degree2():
    """
    Test non-square case: 2x3 polynomial matrix of degree 2.

    P(s) = P(0) + P(1)*s + P(2)*s^2

    A dimensions: 2*2 x 1*2+3 = 4x5
    E dimensions: 4x5
    """
    from ctrlsys import mc03nx

    mp, np_, dp = 2, 3, 2

    P0 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], order='F', dtype=float)
    P1 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], order='F', dtype=float)
    P2 = np.array([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], order='F', dtype=float)

    p = np.zeros((mp, np_, dp + 1), order='F', dtype=float)
    p[:, :, 0] = P0
    p[:, :, 1] = P1
    p[:, :, 2] = P2

    a, e = mc03nx(p)

    nrows = dp * mp
    ncols = (dp - 1) * mp + np_
    assert a.shape == (nrows, ncols), f"A shape {a.shape} != ({nrows}, {ncols})"
    assert e.shape == (nrows, ncols), f"E shape {e.shape} != ({nrows}, {ncols})"


def test_mc03nx_invalid_degree():
    """
    Test that degree 0 (size dp+1=1 for 3rd axis) is rejected.
    """
    from ctrlsys import mc03nx

    p = np.zeros((2, 2, 1), order='F', dtype=float)

    with pytest.raises(ValueError, match="Polynomial degree dp must be >= 1"):
        mc03nx(p)


def test_mc03nx_identity_structure():
    """
    Verify the identity block structure in A matrix.

    For a 3x3 polynomial of degree 3:
    A is 9x12 with identity blocks on the diagonal (upper-left 6x6).
    """
    from ctrlsys import mc03nx

    mp, np_, dp = 3, 3, 3

    p = np.zeros((mp, np_, dp + 1), order='F', dtype=float)
    for k in range(dp + 1):
        p[:, :, k] = np.eye(3) * (k + 1)

    a, e = mc03nx(p)

    nrows = dp * mp
    ncols = (dp - 1) * mp + np_
    hb = (dp - 1) * mp

    id_block = a[:hb, :hb]
    expected = np.eye(hb)
    assert_allclose(id_block, expected, rtol=1e-14)


def test_mc03nx_e_matrix_structure():
    """
    Verify E matrix has correct structure.

    E has:
    - Zero block in top-left mp x hb
    - Identity shift in middle
    - Negated polynomial coefficients on right
    """
    from ctrlsys import mc03nx

    mp, np_, dp = 2, 2, 2

    P0 = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    P1 = np.array([[5.0, 6.0], [7.0, 8.0]], order='F', dtype=float)
    P2 = np.array([[9.0, 10.0], [11.0, 12.0]], order='F', dtype=float)

    p = np.zeros((mp, np_, dp + 1), order='F', dtype=float)
    p[:, :, 0] = P0
    p[:, :, 1] = P1
    p[:, :, 2] = P2

    a, e = mc03nx(p)

    hb = (dp - 1) * mp

    # Right block should contain negated P(dp), P(dp-1), ..., P(1)
    # First mp rows: -P(2)
    assert_allclose(e[:mp, hb:], -P2, rtol=1e-14)
    # Next mp rows: -P(1)
    assert_allclose(e[mp:2*mp, hb:], -P1, rtol=1e-14)
