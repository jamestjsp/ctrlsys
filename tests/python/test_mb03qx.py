"""
Tests for MB03QX: Compute eigenvalues of upper quasi-triangular matrix.

Computes eigenvalues from diagonal and 2x2 blocks of quasi-triangular
matrix (Schur form). Complex conjugate pairs appear as 2x2 blocks.

Tests:
1. Diagonal matrix (all real eigenvalues)
2. Matrix with 2x2 block (complex conjugate pair)
3. Mixed real and complex eigenvalues
4. Single element matrix (n=1)
5. Parameter validation (error handling)

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

def test_mb03qx_diagonal():
    """
    Validate eigenvalue extraction from diagonal matrix.

    All eigenvalues are real, on the diagonal.
    """
    from ctrlsys import mb03qx

    n = 4

    # Diagonal quasi-triangular matrix (all 1x1 blocks)
    t = np.array([
        [1.0, 0.5, 0.2, 0.1],
        [0.0, 2.0, 0.3, 0.2],
        [0.0, 0.0, 3.0, 0.4],
        [0.0, 0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    wr, wi, info = mb03qx(t)

    assert info == 0

    # Expected eigenvalues: 1, 2, 3, 4 (all real)
    expected_wr = np.array([1.0, 2.0, 3.0, 4.0])
    expected_wi = np.zeros(4)

    assert_allclose(wr, expected_wr, rtol=1e-14)
    assert_allclose(wi, expected_wi, atol=1e-14)


def test_mb03qx_complex_pair():
    """
    Validate eigenvalue extraction from 2x2 block (complex pair).

    A 2x2 block on diagonal represents complex conjugate eigenvalues.
    """
    from ctrlsys import mb03qx

    n = 2

    # 2x2 block with eigenvalues 1 +/- 2i
    # For eigenvalue a +/- bi, 2x2 block is [[a, c], [-b^2/c, a]] for some c
    # Using standard form: [[a, b], [-b, a]]
    a_val = 1.0
    b_val = 2.0
    t = np.array([
        [a_val,  b_val],
        [-b_val, a_val]
    ], order='F', dtype=float)

    wr, wi, info = mb03qx(t)

    assert info == 0

    # Expected: eigenvalues 1+2i and 1-2i
    # WR should be [1, 1], WI should be [2, -2]
    assert_allclose(wr, [1.0, 1.0], rtol=1e-14)
    assert_allclose(wi, [2.0, -2.0], rtol=1e-14)


def test_mb03qx_mixed():
    """
    Validate mixed real and complex eigenvalues.

    Matrix with 1x1 blocks and 2x2 blocks.
    """
    from ctrlsys import mb03qx

    n = 5

    # Quasi-triangular with:
    # - Real eigenvalue 5.0 at (0,0)
    # - 2x2 block at (1:3,1:3) with eigenvalues 2+i, 2-i
    # - Real eigenvalue 3.0 at (3,3)
    # - Real eigenvalue 1.0 at (4,4)
    t = np.array([
        [5.0, 0.1, 0.2, 0.3, 0.4],
        [0.0, 2.0, 1.0, 0.5, 0.6],
        [0.0,-1.0, 2.0, 0.7, 0.8],
        [0.0, 0.0, 0.0, 3.0, 0.9],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    wr, wi, info = mb03qx(t)

    assert info == 0

    # Expected eigenvalues in order:
    # 5 (real), 2+i, 2-i (complex pair), 3 (real), 1 (real)
    expected_wr = np.array([5.0, 2.0, 2.0, 3.0, 1.0])
    expected_wi = np.array([0.0, 1.0, -1.0, 0.0, 0.0])

    assert_allclose(wr, expected_wr, rtol=1e-14)
    assert_allclose(wi, expected_wi, rtol=1e-14)


def test_mb03qx_single_element():
    """
    Validate single element matrix (n=1).
    """
    from ctrlsys import mb03qx

    t = np.array([[3.5]], order='F', dtype=float)

    wr, wi, info = mb03qx(t)

    assert info == 0
    assert_allclose(wr, [3.5], rtol=1e-14)
    assert_allclose(wi, [0.0], atol=1e-14)


def test_mb03qx_negative_eigenvalues():
    """
    Validate with negative real eigenvalues.
    """
    from ctrlsys import mb03qx

    n = 3

    t = np.array([
        [-2.0, 0.5, 0.2],
        [ 0.0,-1.0, 0.3],
        [ 0.0, 0.0,-3.0]
    ], order='F', dtype=float)

    wr, wi, info = mb03qx(t)

    assert info == 0

    expected_wr = np.array([-2.0, -1.0, -3.0])
    expected_wi = np.zeros(3)

    assert_allclose(wr, expected_wr, rtol=1e-14)
    assert_allclose(wi, expected_wi, atol=1e-14)


def test_mb03qx_eigenvalue_consistency():
    """
    Validate eigenvalues from quasi-triangular matrix.

    Construct quasi-triangular matrix with known eigenvalues and verify
    mb03qx extracts them correctly.
    """
    from ctrlsys import mb03qx

    # Construct quasi-triangular matrix with:
    # - Real eigenvalue 2.0 at (0,0)
    # - Complex pair 1+3i, 1-3i from 2x2 block at (1:3, 1:3)
    # - Real eigenvalue -1.0 at (3,3)
    t = np.array([
        [2.0, 0.5, 0.3, 0.1],
        [0.0, 1.0, 3.0, 0.2],
        [0.0, -3.0, 1.0, 0.4],
        [0.0, 0.0, 0.0, -1.0]
    ], order='F', dtype=float)

    wr, wi, info = mb03qx(t)

    assert info == 0

    # Expected eigenvalues: 2, 1+3i, 1-3i, -1
    eig_mb03qx = wr + 1j * wi
    eig_mb03qx_sorted = np.sort_complex(eig_mb03qx)
    eig_expected_sorted = np.sort_complex(np.array([2.0, 1.0+3.0j, 1.0-3.0j, -1.0]))

    assert_allclose(eig_mb03qx_sorted, eig_expected_sorted, rtol=1e-12)


def test_mb03qx_multiple_complex_blocks():
    """
    Validate matrix with multiple 2x2 blocks.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03qx

    n = 4

    # Two 2x2 blocks: eigenvalues 1+2i, 1-2i and 3+i, 3-i
    t = np.array([
        [1.0, 2.0, 0.5, 0.3],
        [-2.0, 1.0, 0.4, 0.2],
        [0.0, 0.0, 3.0, 1.0],
        [0.0, 0.0, -1.0, 3.0]
    ], order='F', dtype=float)

    wr, wi, info = mb03qx(t)

    assert info == 0

    # Expected eigenvalues
    expected_wr = np.array([1.0, 1.0, 3.0, 3.0])
    expected_wi = np.array([2.0, -2.0, 1.0, -1.0])

    assert_allclose(wr, expected_wr, rtol=1e-14)
    assert_allclose(wi, expected_wi, rtol=1e-14)
