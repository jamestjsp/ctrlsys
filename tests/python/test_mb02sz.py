"""
Tests for MB02SZ: LU factorization of complex upper Hessenberg matrix.

Computes P*L*U factorization where:
- P is permutation matrix
- L is unit lower bidiagonal
- U is upper triangular
"""

import numpy as np
import pytest
from ctrlsys import mb02sz


"""Basic functionality tests."""

def test_2x2_hessenberg_no_pivot():
    """
    Test 2x2 upper Hessenberg matrix where no pivoting needed.

    H = [[2+1j, 3+0j],
         [1+0j, 4+2j]]

    LU factorization: H = P*L*U
    With no pivoting, pivot indices = [1, 2] (1-based).
    """
    h = np.array([
        [2.0 + 1.0j, 3.0 + 0.0j],
        [1.0 + 0.0j, 4.0 + 2.0j]
    ], order='F', dtype=complex)

    h_orig = h.copy()
    h_result, ipiv, info = mb02sz(h)

    assert info == 0
    assert ipiv.shape == (2,)

    u = np.triu(h_result)
    l = np.eye(2, dtype=complex)
    l[1, 0] = h_result[1, 0]

    p = np.eye(2, dtype=complex)
    if ipiv[0] == 2:
        p = np.array([[0, 1], [1, 0]], dtype=complex)

    h_reconstructed = p @ l @ u
    np.testing.assert_allclose(h_reconstructed, h_orig, rtol=1e-14)

def test_3x3_hessenberg_with_pivot():
    """
    Test 3x3 upper Hessenberg matrix requiring pivoting.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h_orig = h.copy()
    h_result, ipiv, info = mb02sz(h)

    assert info == 0
    assert ipiv.shape == (n,)
    assert all(1 <= p <= n for p in ipiv)

def test_4x4_random_hessenberg():
    """
    Test 4x4 random upper Hessenberg matrix.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h_orig = h.copy()
    h_result, ipiv, info = mb02sz(h)

    assert info == 0


"""Test mathematical properties of LU factorization."""

def test_lu_factorization_solves_system():
    """
    Validate LU factorization by solving H*x = b and checking residual.

    This is the mathematically correct way to verify LU factorization:
    if the factorization is correct, solving should give accurate results.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 5

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()
    h[np.diag_indices(n)] += 5.0

    h_orig = h.copy()
    b = np.random.randn(n, 1) + 1j * np.random.randn(n, 1)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    h_result, ipiv, info = mb02sz(h)
    assert info == 0

    from ctrlsys import mb02rz
    x, info_solve = mb02rz('N', h_result, ipiv, b)
    assert info_solve == 0

    residual = h_orig @ x - b_orig
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)

def test_pivot_indices_valid():
    """
    Validate pivot indices are valid (i or i+1 only for Hessenberg).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 6

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h_result, ipiv, info = mb02sz(h)

    assert info == 0
    for j in range(n):
        assert ipiv[j] == j + 1 or (j < n - 1 and ipiv[j] == j + 2)


"""Edge cases and boundary conditions."""

def test_n_equals_1():
    """Test 1x1 matrix (trivial case)."""
    h = np.array([[3.0 + 2.0j]], order='F', dtype=complex)

    h_result, ipiv, info = mb02sz(h)

    assert info == 0
    assert ipiv[0] == 1
    np.testing.assert_allclose(h_result[0, 0], 3.0 + 2.0j, rtol=1e-14)

def test_identity_matrix():
    """Test identity matrix."""
    n = 4
    h = np.eye(n, dtype=complex, order='F')

    h_result, ipiv, info = mb02sz(h)

    assert info == 0
    np.testing.assert_allclose(np.triu(h_result), np.eye(n, dtype=complex), rtol=1e-14)

def test_diagonal_matrix():
    """Test diagonal matrix (special case of upper Hessenberg)."""
    n = 3
    h = np.diag([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j]).astype(complex, order='F')

    h_result, ipiv, info = mb02sz(h)

    assert info == 0
    for j in range(n):
        assert ipiv[j] == j + 1


"""Test error conditions."""

def test_singular_matrix():
    """
    Test detection of singular matrix.

    When a diagonal of U becomes zero, info > 0.
    """
    h = np.array([
        [0.0 + 0.0j, 1.0 + 0.0j],
        [0.0 + 0.0j, 2.0 + 0.0j]
    ], order='F', dtype=complex)

    h_result, ipiv, info = mb02sz(h)

    assert info == 1

def test_zero_subdiagonal_causes_singularity():
    """Test matrix with zero elements leading to singularity."""
    h = np.array([
        [0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 4.0 + 0.0j]
    ], order='F', dtype=complex)

    h_result, ipiv, info = mb02sz(h)

    assert info > 0


"""Test with specific known values."""

def test_known_factorization():
    """
    Test with a specific matrix where factorization is known.

    H = [[4+0j, 2+1j],
         [2+0j, 5+0j]]

    Since |2| < |4|, no pivot at first step.
    L[2,1] = (2+0j)/(4+0j) = 0.5
    U = [[4+0j, 2+1j], [0, 5-(0.5)*(2+1j)]] = [[4+0j, 2+1j], [0, 4-0.5j]]
    """
    h = np.array([
        [4.0 + 0.0j, 2.0 + 1.0j],
        [2.0 + 0.0j, 5.0 + 0.0j]
    ], order='F', dtype=complex)

    h_result, ipiv, info = mb02sz(h)

    assert info == 0
    assert ipiv[0] == 1
    assert ipiv[1] == 2

    np.testing.assert_allclose(h_result[0, 0], 4.0 + 0.0j, rtol=1e-14)
    np.testing.assert_allclose(h_result[0, 1], 2.0 + 1.0j, rtol=1e-14)
    np.testing.assert_allclose(h_result[1, 0], 0.5 + 0.0j, rtol=1e-14)
    np.testing.assert_allclose(h_result[1, 1], 4.0 - 0.5j, rtol=1e-14)
