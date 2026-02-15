"""
Tests for UD01CD: Reading sparse matrix polynomial coefficients.

UD01CD reads the elements of a sparse matrix polynomial P(s) where each
nonzero (i,j) element is specified by (row, col, degree, coefficients).
P(s) = P(0) + P(1)*s + ... + P(dp-1)*s^(dp-1) + P(dp)*s^dp

Unlike UD01BD which reads dense data, UD01CD only stores nonzero elements,
with all other elements initialized to zero.
"""

import numpy as np
import pytest


def test_ud01cd_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    From HTML doc:
    - MP=4, NP=3, DP=2 (4x3 matrices, degree 2 polynomial)
    - Sparse elements:
      (1,1,1): coeffs [1.0, 1.0] -> P(1,1,s) = 1 + s
      (2,2,2): coeffs [2.0, 0.0, 1.0] -> P(2,2,s) = 2 + s^2
      (3,3,2): coeffs [0.0, 3.0, 1.0] -> P(3,3,s) = 3s + s^2
      (4,1,0): coeffs [4.0] -> P(4,1,s) = 4

    Expected output matrices:
    P(0) = [[1, 0, 0], [0, 2, 0], [0, 0, 0], [4, 0, 0]]
    P(1) = [[1, 0, 0], [0, 0, 0], [0, 0, 3], [0, 0, 0]]
    P(2) = [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 4, 3, 2

    rows = np.array([1, 2, 3, 4], dtype=np.int32)
    cols = np.array([1, 2, 3, 1], dtype=np.int32)
    degrees = np.array([1, 2, 2, 0], dtype=np.int32)
    coeffs = np.array([1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 3.0, 1.0, 4.0], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == 0
    assert p.shape == (mp, np_dim, dp + 1)

    p0_expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ], order='F', dtype=float)

    p1_expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    p2_expected = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(p[:, :, 0], p0_expected, rtol=1e-14)
    np.testing.assert_allclose(p[:, :, 1], p1_expected, rtol=1e-14)
    np.testing.assert_allclose(p[:, :, 2], p2_expected, rtol=1e-14)


def test_ud01cd_empty_sparse():
    """
    Test edge case: no nonzero elements (all zeros).
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 2, 2, 1

    rows = np.array([], dtype=np.int32)
    cols = np.array([], dtype=np.int32)
    degrees = np.array([], dtype=np.int32)
    coeffs = np.array([], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == 0
    assert p.shape == (mp, np_dim, dp + 1)
    np.testing.assert_allclose(p, np.zeros((mp, np_dim, dp + 1)), rtol=1e-14)


def test_ud01cd_single_element():
    """
    Test with a single nonzero polynomial element.
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 3, 3, 2

    rows = np.array([2], dtype=np.int32)
    cols = np.array([3], dtype=np.int32)
    degrees = np.array([2], dtype=np.int32)
    coeffs = np.array([1.0, 2.0, 3.0], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == 0

    p_expected = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    p_expected[1, 2, 0] = 1.0
    p_expected[1, 2, 1] = 2.0
    p_expected[1, 2, 2] = 3.0

    np.testing.assert_allclose(p, p_expected, rtol=1e-14)


def test_ud01cd_degree_zero():
    """
    Test edge case: degree 0 polynomial (constant matrices only).
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 2, 2, 0

    rows = np.array([1, 2], dtype=np.int32)
    cols = np.array([1, 2], dtype=np.int32)
    degrees = np.array([0, 0], dtype=np.int32)
    coeffs = np.array([5.0, 7.0], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == 0
    assert p.shape == (mp, np_dim, 1)

    p0_expected = np.array([[5.0, 0.0], [0.0, 7.0]], order='F', dtype=float)
    np.testing.assert_allclose(p[:, :, 0], p0_expected, rtol=1e-14)


def test_ud01cd_error_mp_invalid():
    """
    Test error: MP < 1.
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 0, 3, 2
    rows = np.array([1], dtype=np.int32)
    cols = np.array([1], dtype=np.int32)
    degrees = np.array([0], dtype=np.int32)
    coeffs = np.array([1.0], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == -1


def test_ud01cd_error_np_invalid():
    """
    Test error: NP < 1.
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 2, 0, 2
    rows = np.array([1], dtype=np.int32)
    cols = np.array([1], dtype=np.int32)
    degrees = np.array([0], dtype=np.int32)
    coeffs = np.array([1.0], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == -2


def test_ud01cd_error_dp_invalid():
    """
    Test error: DP < 0.
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 2, 3, -1
    rows = np.array([1], dtype=np.int32)
    cols = np.array([1], dtype=np.int32)
    degrees = np.array([0], dtype=np.int32)
    coeffs = np.array([1.0], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == -3


def test_ud01cd_warning_invalid_index():
    """
    Test warning: invalid row/column/degree values (INFO=1).

    Per Fortran spec, invalid indices cause warning but routine continues.
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 2, 2, 1

    rows = np.array([1, 5], dtype=np.int32)
    cols = np.array([1, 1], dtype=np.int32)
    degrees = np.array([0, 0], dtype=np.int32)
    coeffs = np.array([1.0, 2.0], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == 1
    np.testing.assert_allclose(p[0, 0, 0], 1.0, rtol=1e-14)


def test_ud01cd_warning_invalid_degree():
    """
    Test warning: degree > DP+1 causes INFO=1 warning.
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 2, 2, 1

    rows = np.array([1, 2], dtype=np.int32)
    cols = np.array([1, 2], dtype=np.int32)
    degrees = np.array([0, 5], dtype=np.int32)
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == 1
    np.testing.assert_allclose(p[0, 0, 0], 1.0, rtol=1e-14)


def test_ud01cd_sparsity_preservation():
    """
    Mathematical property: only specified elements should be nonzero.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ud01cd

    np.random.seed(42)

    mp, np_dim, dp = 5, 4, 3

    nelem = 3
    rows = np.array([1, 3, 5], dtype=np.int32)
    cols = np.array([2, 4, 1], dtype=np.int32)
    degrees = np.array([1, 2, 0], dtype=np.int32)
    coeffs = np.random.randn(1 + 2 + 3 + 1)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == 0

    coeff_idx = 0
    for e in range(nelem):
        i, j, d = rows[e] - 1, cols[e] - 1, degrees[e]
        for k in range(d + 1):
            assert p[i, j, k] == coeffs[coeff_idx]
            coeff_idx += 1

    for i in range(mp):
        for j in range(np_dim):
            is_specified = False
            for e in range(nelem):
                if rows[e] - 1 == i and cols[e] - 1 == j:
                    is_specified = True
                    break
            if not is_specified:
                np.testing.assert_allclose(p[i, j, :], 0.0, rtol=1e-14)


def test_ud01cd_multiple_same_position():
    """
    Test overwriting: if same (i,j) specified twice, later value wins.
    """
    from ctrlsys import ud01cd

    mp, np_dim, dp = 2, 2, 1

    rows = np.array([1, 1], dtype=np.int32)
    cols = np.array([1, 1], dtype=np.int32)
    degrees = np.array([0, 1], dtype=np.int32)
    coeffs = np.array([5.0, 10.0, 20.0], dtype=float)

    p, info = ud01cd(mp, np_dim, dp, rows, cols, degrees, coeffs)

    assert info == 0
    np.testing.assert_allclose(p[0, 0, 0], 10.0, rtol=1e-14)
    np.testing.assert_allclose(p[0, 0, 1], 20.0, rtol=1e-14)
