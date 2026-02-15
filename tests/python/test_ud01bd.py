"""
Tests for UD01BD: Reading/copying matrix polynomial coefficients.

UD01BD reads the coefficients of a matrix polynomial P(s) from input data.
P(s) = P(0) + P(1)*s + ... + P(dp-1)*s^(dp-1) + P(dp)*s^dp

In the C implementation, instead of file I/O, the routine copies from a
source data array to the output polynomial array P.
"""

import numpy as np
import pytest


def test_ud01bd_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    From HTML doc:
    - MP=4, NP=3, DP=2 (4x3 matrices, degree 2 polynomial)
    - P(0), P(1), P(2) coefficient matrices
    """
    from ctrlsys import ud01bd

    mp, np_dim, dp = 4, 3, 2

    # Input data from HTML example (row-by-row for each coefficient matrix)
    # P0
    p0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 4.0],
        [0.0, 4.0, 8.0],
        [0.0, 6.0, 12.0]
    ], order='F', dtype=float)

    # P1
    p1 = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ], order='F', dtype=float)

    # P2
    p2 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    # Stack as input data (flattened row-by-row as Fortran reads)
    # Fortran reads: for each k in [0, dp]: for each i in [0, mp): read row i
    data = np.concatenate([
        p0.flatten(order='C'),  # row-by-row
        p1.flatten(order='C'),
        p2.flatten(order='C')
    ])

    # Call routine
    p, info = ud01bd(mp, np_dim, dp, data)

    assert info == 0

    # Verify shapes
    assert p.shape == (mp, np_dim, dp + 1)

    # Verify numerical values
    np.testing.assert_allclose(p[:, :, 0], p0, rtol=1e-14)
    np.testing.assert_allclose(p[:, :, 1], p1, rtol=1e-14)
    np.testing.assert_allclose(p[:, :, 2], p2, rtol=1e-14)


def test_ud01bd_degree_zero():
    """
    Test edge case: degree 0 polynomial (single constant matrix).
    """
    from ctrlsys import ud01bd

    mp, np_dim, dp = 2, 2, 0

    # Single 2x2 matrix
    p0 = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    data = p0.flatten(order='C')  # row-by-row

    p, info = ud01bd(mp, np_dim, dp, data)

    assert info == 0
    assert p.shape == (mp, np_dim, 1)
    np.testing.assert_allclose(p[:, :, 0], p0, rtol=1e-14)


def test_ud01bd_single_element():
    """
    Test edge case: 1x1 polynomial matrices.
    """
    from ctrlsys import ud01bd

    mp, np_dim, dp = 1, 1, 3

    # Scalar coefficients: P(s) = 1 + 2s + 3s^2 + 4s^3
    data = np.array([1.0, 2.0, 3.0, 4.0])

    p, info = ud01bd(mp, np_dim, dp, data)

    assert info == 0
    assert p.shape == (1, 1, 4)
    np.testing.assert_allclose(p[0, 0, :], [1.0, 2.0, 3.0, 4.0], rtol=1e-14)


def test_ud01bd_error_mp_invalid():
    """
    Test error: MP < 1.
    """
    from ctrlsys import ud01bd

    mp, np_dim, dp = 0, 3, 2
    data = np.array([1.0, 2.0, 3.0])

    p, info = ud01bd(mp, np_dim, dp, data)

    assert info == -1


def test_ud01bd_error_np_invalid():
    """
    Test error: NP < 1.
    """
    from ctrlsys import ud01bd

    mp, np_dim, dp = 2, 0, 2
    data = np.array([1.0, 2.0, 3.0])

    p, info = ud01bd(mp, np_dim, dp, data)

    assert info == -2


def test_ud01bd_error_dp_invalid():
    """
    Test error: DP < 0.
    """
    from ctrlsys import ud01bd

    mp, np_dim, dp = 2, 3, -1
    data = np.array([1.0, 2.0, 3.0])

    p, info = ud01bd(mp, np_dim, dp, data)

    assert info == -3


def test_ud01bd_rectangular_matrices():
    """
    Test with non-square coefficient matrices (MP != NP).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ud01bd

    np.random.seed(42)

    mp, np_dim, dp = 3, 5, 1

    # Generate random coefficient matrices
    p0 = np.random.randn(mp, np_dim).astype(float, order='F')
    p1 = np.random.randn(mp, np_dim).astype(float, order='F')

    # Flatten row-by-row
    data = np.concatenate([
        p0.flatten(order='C'),
        p1.flatten(order='C')
    ])

    p, info = ud01bd(mp, np_dim, dp, data)

    assert info == 0
    assert p.shape == (mp, np_dim, dp + 1)
    np.testing.assert_allclose(p[:, :, 0], p0, rtol=1e-14)
    np.testing.assert_allclose(p[:, :, 1], p1, rtol=1e-14)


def test_ud01bd_data_preservation():
    """
    Mathematical property: data should be exactly preserved (no transformation).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ud01bd

    np.random.seed(123)

    mp, np_dim, dp = 4, 3, 2

    # Generate random data
    n_elements = mp * np_dim * (dp + 1)
    data = np.random.randn(n_elements)

    p, info = ud01bd(mp, np_dim, dp, data)

    assert info == 0

    # Reconstruct data from output and compare
    reconstructed = []
    for k in range(dp + 1):
        reconstructed.append(p[:, :, k].flatten(order='C'))
    reconstructed = np.concatenate(reconstructed)

    np.testing.assert_allclose(reconstructed, data, rtol=1e-14)
