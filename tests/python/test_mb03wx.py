"""
Tests for MB03WX: Eigenvalues of a product of matrices in periodic Schur form.

MB03WX computes eigenvalues of T = T_1*T_2*...*T_p where:
- T_1 is upper quasi-triangular (real Schur form)
- T_2, ..., T_p are upper triangular
"""

import numpy as np
import pytest
from ctrlsys import mb03wx


def test_mb03wx_single_matrix():
    """
    Test with P=1 (single quasi-triangular matrix).

    Eigenvalues should be diagonal elements for 1x1 blocks and
    eigenvalues of 2x2 blocks for complex conjugate pairs.

    Random seed: 42 (for reproducibility)
    """
    n = 4
    p = 1

    # Create upper quasi-triangular matrix (real Schur form)
    # 2x2 block in positions [0:2,0:2] with complex eigenvalues
    # Two 1x1 blocks on diagonal at positions [2,2] and [3,3]
    t1 = np.array([
        [2.0, 3.0, 0.5, 0.3],
        [-1.0, 2.0, 0.2, 0.1],
        [0.0, 0.0, 4.0, 0.4],
        [0.0, 0.0, 0.0, 5.0]
    ], order='F', dtype=float)

    # Stack into 3D array (n x n x p)
    t = np.zeros((n, n, p), order='F', dtype=float)
    t[:, :, 0] = t1

    wr, wi, info = mb03wx(n, p, t)

    assert info == 0

    # For single matrix, eigenvalues are from T_1 directly
    # 2x2 block [[2, 3], [-1, 2]] has eigenvalues 2 +/- sqrt(3)i
    # 1x1 blocks give eigenvalues 4 and 5

    # Expected eigenvalues from 2x2 block: 2 +/- sqrt(3)*i
    expected_wr = np.array([2.0, 2.0, 4.0, 5.0])
    expected_wi = np.array([np.sqrt(3.0), -np.sqrt(3.0), 0.0, 0.0])

    np.testing.assert_allclose(wr, expected_wr, rtol=1e-14)
    np.testing.assert_allclose(wi, expected_wi, rtol=1e-14)


def test_mb03wx_two_triangular_matrices():
    """
    Test with P=2: T = T_1 * T_2 where both are upper triangular.

    For triangular matrices, eigenvalues of product are products of diagonals.

    Random seed: 123 (for reproducibility)
    """
    n = 3
    p = 2

    # Upper triangular T_1
    t1 = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 3.0, 0.3],
        [0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    # Upper triangular T_2
    t2 = np.array([
        [1.5, 0.2, 0.1],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 0.5]
    ], order='F', dtype=float)

    # Stack into 3D array (n x n x p)
    t = np.zeros((n, n, p), order='F', dtype=float)
    t[:, :, 0] = t1
    t[:, :, 1] = t2

    wr, wi, info = mb03wx(n, p, t)

    assert info == 0

    # For triangular matrices, eigenvalues are products of diagonal elements
    # lambda_i = T_1(i,i) * T_2(i,i)
    expected_wr = np.array([2.0 * 1.5, 3.0 * 2.0, 4.0 * 0.5])
    expected_wi = np.zeros(n)

    np.testing.assert_allclose(wr, expected_wr, rtol=1e-14)
    np.testing.assert_allclose(wi, expected_wi, rtol=1e-14)


def test_mb03wx_complex_eigenvalue_pair():
    """
    Test eigenvalue computation for 2x2 block with complex eigenvalues
    through a product of P=3 matrices.

    Tests that complex conjugate pairs are properly identified and
    computed from the periodic Schur form.

    Random seed: 456 (for reproducibility)
    """
    n = 2
    p = 3

    # T_1: quasi-triangular (2x2 block with complex eigenvalues)
    # eigenvalues of T_1 alone: 1 +/- 2i (from [[1, 4], [-1, 1]])
    t1 = np.array([
        [1.0, 4.0],
        [-1.0, 1.0]
    ], order='F', dtype=float)

    # T_2: upper triangular with diagonal [2, 3]
    t2 = np.array([
        [2.0, 0.5],
        [0.0, 3.0]
    ], order='F', dtype=float)

    # T_3: upper triangular with diagonal [0.5, 0.5]
    t3 = np.array([
        [0.5, 0.1],
        [0.0, 0.5]
    ], order='F', dtype=float)

    # Stack into 3D array (n x n x p)
    t = np.zeros((n, n, p), order='F', dtype=float)
    t[:, :, 0] = t1
    t[:, :, 1] = t2
    t[:, :, 2] = t3

    wr, wi, info = mb03wx(n, p, t)

    assert info == 0

    # Compute expected eigenvalues by forming 2x2 product explicitly
    # T11_prod = T_1(1,1) * T_2(1,1) * T_3(1,1) = 1 * 2 * 0.5 = 1
    # T22_prod = T_1(2,2) * T_2(2,2) * T_3(2,2) = 1 * 3 * 0.5 = 1.5
    # T12_prod computed by the algorithm
    # T21 = T_1(2,1) * T22_prod_2_to_p = -1 * T_2(1,1)*T_3(1,1) = -1 * 1 = -1

    # The 2x2 product matrix is:
    # [A11, A12]   where A11 = T_1(1,1)*T11 = 1*1 = 1
    # [A21, A22]         A22 = T_1(2,1)*T12 + T_1(2,2)*T22 = ...
    # with T11 = prod of T_j(1,1) for j=2..p = 2*0.5 = 1
    #      T22 = prod of T_j(2,2) for j=2..p = 3*0.5 = 1.5
    #      T12 computed iteratively

    # Let's compute the 2x2 product matrix A = T_1 * D where D is the
    # accumulated diagonal/upper triangular part
    t11 = 1.0
    t12 = 0.0
    t22 = 1.0
    for j in range(1, p):
        t22_new = t22 * t[1, 1, j]
        t12_new = t11 * t[0, 1, j] + t12 * t[1, 1, j]
        t11_new = t11 * t[0, 0, j]
        t11, t12, t22 = t11_new, t12_new, t22_new

    a11 = t1[0, 0] * t11
    a12 = t1[0, 0] * t12 + t1[0, 1] * t22
    a21 = t1[1, 0] * t11
    a22 = t1[1, 0] * t12 + t1[1, 1] * t22

    # Eigenvalues of 2x2 matrix [[a11, a12], [a21, a22]]
    a_2x2 = np.array([[a11, a12], [a21, a22]])
    expected_eigs = np.linalg.eigvals(a_2x2)
    expected_eigs = np.sort_complex(expected_eigs)

    # Sort computed eigenvalues
    computed_eigs = np.sort_complex(wr + 1j * wi)

    np.testing.assert_allclose(computed_eigs.real, expected_eigs.real, rtol=1e-14)
    np.testing.assert_allclose(np.abs(computed_eigs.imag), np.abs(expected_eigs.imag), rtol=1e-14)

    # wi(1) should be positive for complex pair (by convention)
    if wi[0] != 0:
        assert wi[0] > 0


def test_mb03wx_mixed_blocks():
    """
    Test with mixed 1x1 and 2x2 blocks.

    T_1 has structure:
    - 1x1 block at (0,0)
    - 2x2 block at (1:3, 1:3)
    - 1x1 block at (3,3)

    Random seed: 789 (for reproducibility)
    """
    n = 4
    p = 2

    # T_1: quasi-triangular with mixed blocks
    t1 = np.array([
        [3.0, 0.5, 0.3, 0.2],
        [0.0, 1.0, 2.0, 0.4],
        [0.0, -0.5, 1.0, 0.1],
        [0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    # T_2: upper triangular
    t2 = np.array([
        [2.0, 0.3, 0.1, 0.05],
        [0.0, 1.5, 0.2, 0.1],
        [0.0, 0.0, 2.0, 0.15],
        [0.0, 0.0, 0.0, 0.5]
    ], order='F', dtype=float)

    t = np.zeros((n, n, p), order='F', dtype=float)
    t[:, :, 0] = t1
    t[:, :, 1] = t2

    wr, wi, info = mb03wx(n, p, t)

    assert info == 0

    # Eigenvalue 1: T_1(0,0) * T_2(0,0) = 3 * 2 = 6
    assert np.isclose(wr[0], 6.0, rtol=1e-14)
    assert np.isclose(wi[0], 0.0, atol=1e-14)

    # Eigenvalue 4: T_1(3,3) * T_2(3,3) = 2 * 0.5 = 1
    assert np.isclose(wr[3], 1.0, rtol=1e-14)
    assert np.isclose(wi[3], 0.0, atol=1e-14)

    # Eigenvalues 2,3: from 2x2 block product
    # Compute the 2x2 product explicitly
    t11 = t2[1, 1]  # 1.5
    t12 = t2[1, 2]  # 0.2
    t22 = t2[2, 2]  # 2.0

    a11 = t1[1, 1] * t11  # 1 * 1.5 = 1.5
    a12 = t1[1, 1] * t12 + t1[1, 2] * t22  # 1*0.2 + 2*2 = 4.2
    a21 = t1[2, 1] * t11  # -0.5 * 1.5 = -0.75
    a22 = t1[2, 1] * t12 + t1[2, 2] * t22  # -0.5*0.2 + 1*2 = 1.9

    block_2x2 = np.array([[a11, a12], [a21, a22]])
    block_eigs = np.linalg.eigvals(block_2x2)

    # The eigenvalues from wr[1:3] + wi[1:3]*1j should match
    computed = wr[1] + 1j * wi[1], wr[2] + 1j * wi[2]
    computed = np.sort_complex(np.array(computed))
    expected = np.sort_complex(block_eigs)

    np.testing.assert_allclose(computed.real, expected.real, rtol=1e-14)
    np.testing.assert_allclose(computed.imag, expected.imag, rtol=1e-14, atol=1e-14)


def test_mb03wx_large_p():
    """
    Test with larger number of factors (P=5).

    Verifies correct accumulation over multiple triangular matrices.

    Random seed: 999 (for reproducibility)
    """
    n = 3
    p = 5

    # T_1: upper triangular (no 2x2 blocks for simplicity)
    diags = [[2.0, 3.0, 1.0], [1.5, 0.5, 2.0], [1.0, 2.0, 0.5],
             [2.0, 1.0, 1.5], [0.5, 2.0, 1.0]]

    t = np.zeros((n, n, p), order='F', dtype=float)
    for k in range(p):
        t[0, 0, k] = diags[k][0]
        t[1, 1, k] = diags[k][1]
        t[2, 2, k] = diags[k][2]
        # Add some upper triangular entries
        t[0, 1, k] = 0.1 * (k + 1)
        t[0, 2, k] = 0.05 * (k + 1)
        t[1, 2, k] = 0.08 * (k + 1)

    wr, wi, info = mb03wx(n, p, t)

    assert info == 0

    # For triangular matrices, eigenvalues are products of diagonals
    expected_wr = np.zeros(n)
    for i in range(n):
        prod = 1.0
        for k in range(p):
            prod *= diags[k][i]
        expected_wr[i] = prod

    np.testing.assert_allclose(wr, expected_wr, rtol=1e-14)
    np.testing.assert_allclose(wi, np.zeros(n), atol=1e-14)


def test_mb03wx_n_zero():
    """
    Test edge case N=0 (quick return).
    """
    n = 0
    p = 1

    t = np.zeros((1, 1, p), order='F', dtype=float)

    wr, wi, info = mb03wx(n, p, t)

    assert info == 0
    assert len(wr) == 0
    assert len(wi) == 0


def test_mb03wx_n_one():
    """
    Test edge case N=1 (single eigenvalue).
    """
    n = 1
    p = 3

    t = np.zeros((n, n, p), order='F', dtype=float)
    t[0, 0, 0] = 2.0
    t[0, 0, 1] = 3.0
    t[0, 0, 2] = 0.5

    wr, wi, info = mb03wx(n, p, t)

    assert info == 0
    assert len(wr) == 1
    assert len(wi) == 1
    np.testing.assert_allclose(wr[0], 2.0 * 3.0 * 0.5, rtol=1e-14)
    np.testing.assert_allclose(wi[0], 0.0, atol=1e-14)


def test_mb03wx_invalid_n():
    """
    Test error handling for invalid N < 0.
    """
    n = -1
    p = 1

    t = np.zeros((1, 1, p), order='F', dtype=float)

    wr, wi, info = mb03wx(n, p, t)

    assert info == -1


def test_mb03wx_invalid_p():
    """
    Test error handling for invalid P < 1.
    """
    n = 2
    p = 0

    t = np.zeros((2, 2, 1), order='F', dtype=float)

    wr, wi, info = mb03wx(n, p, t)

    assert info == -2


def test_mb03wx_eigenvalue_product_property():
    """
    Validate mathematical property: determinant of product equals product of determinants.

    For triangular matrices: det(T) = prod(diagonal elements)
    Eigenvalues should multiply to give determinant.

    Random seed: 321 (for reproducibility)
    """
    np.random.seed(321)
    n = 3
    p = 2

    # Create random upper triangular matrices with positive diagonals
    t = np.zeros((n, n, p), order='F', dtype=float)
    for k in range(p):
        for i in range(n):
            t[i, i, k] = np.random.uniform(0.5, 2.0)
            for j in range(i + 1, n):
                t[i, j, k] = np.random.uniform(-1.0, 1.0)

    wr, wi, info = mb03wx(n, p, t)
    assert info == 0

    # For triangular matrices, eigenvalues are products of diagonals
    for i in range(n):
        expected = 1.0
        for k in range(p):
            expected *= t[i, i, k]
        np.testing.assert_allclose(wr[i], expected, rtol=1e-14)
        np.testing.assert_allclose(wi[i], 0.0, atol=1e-14)
