"""
Tests for MB02RZ: Solve complex Hessenberg system using LU factorization.

Solves H*X=B, H'*X=B, or H^H*X=B using factorization from MB02SZ.
"""

import numpy as np
import pytest
from ctrlsys import mb02sz, mb02rz


"""Basic functionality tests."""

def test_no_transpose_2x2():
    """
    Test H*X=B with 2x2 system, no transpose.

    H = [[2+1j, 3+0j],
         [1+0j, 4+2j]]
    B = [[1+0j], [2+1j]]

    Solve H*X = B for X.
    """
    h = np.array([
        [2.0 + 1.0j, 3.0 + 0.0j],
        [1.0 + 0.0j, 4.0 + 2.0j]
    ], order='F', dtype=complex)

    b = np.array([
        [1.0 + 0.0j],
        [2.0 + 1.0j]
    ], order='F', dtype=complex)

    h_orig = h.copy()
    b_orig = b.copy()

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    x, info = mb02rz('N', h_lu, ipiv, b)
    assert info == 0

    residual = h_orig @ x - b_orig
    np.testing.assert_allclose(residual, 0.0, atol=1e-14)

def test_transpose_2x2():
    """
    Test H'*X=B with 2x2 system, transpose (not conjugate).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 2

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    b = np.random.randn(n, 1) + 1j * np.random.randn(n, 1)
    b = np.asfortranarray(b)

    h_orig = h.copy()
    b_orig = b.copy()

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    x, info = mb02rz('T', h_lu, ipiv, b)
    assert info == 0

    residual = h_orig.T @ x - b_orig
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)

def test_conjugate_transpose_2x2():
    """
    Test H^H*X=B with 2x2 system, conjugate transpose.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 2

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    b = np.random.randn(n, 1) + 1j * np.random.randn(n, 1)
    b = np.asfortranarray(b)

    h_orig = h.copy()
    b_orig = b.copy()

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    x, info = mb02rz('C', h_lu, ipiv, b)
    assert info == 0

    residual = h_orig.conj().T @ x - b_orig
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)


"""Test mathematical properties."""

def test_solution_accuracy_4x4():
    """
    Validate solution accuracy for 4x4 system.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4
    nrhs = 2

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    b = np.random.randn(n, nrhs) + 1j * np.random.randn(n, nrhs)
    b = np.asfortranarray(b)

    h_orig = h.copy()
    b_orig = b.copy()

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    x, info = mb02rz('N', h_lu, ipiv, b)
    assert info == 0

    residual = h_orig @ x - b_orig
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)

def test_all_transpose_modes_give_correct_solution():
    """
    Validate all three transpose modes give correct solutions.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    b = np.random.randn(n, 1) + 1j * np.random.randn(n, 1)
    b = np.asfortranarray(b)

    h_orig = h.copy()
    b_orig = b.copy()

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    b_n = b_orig.copy()
    x_n, info = mb02rz('N', h_lu, ipiv, b_n)
    assert info == 0
    np.testing.assert_allclose(h_orig @ x_n, b_orig, atol=1e-13)

    b_t = b_orig.copy()
    x_t, info = mb02rz('T', h_lu, ipiv, b_t)
    assert info == 0
    np.testing.assert_allclose(h_orig.T @ x_t, b_orig, atol=1e-13)

    b_c = b_orig.copy()
    x_c, info = mb02rz('C', h_lu, ipiv, b_c)
    assert info == 0
    np.testing.assert_allclose(h_orig.conj().T @ x_c, b_orig, atol=1e-13)

def test_multiple_rhs():
    """
    Test with multiple right-hand sides.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 4
    nrhs = 5

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    b = np.random.randn(n, nrhs) + 1j * np.random.randn(n, nrhs)
    b = np.asfortranarray(b)

    h_orig = h.copy()
    b_orig = b.copy()

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    x, info = mb02rz('N', h_lu, ipiv, b)
    assert info == 0
    assert x.shape == (n, nrhs)

    residual = h_orig @ x - b_orig
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)


"""Edge cases and boundary conditions."""

def test_n_equals_1():
    """Test 1x1 system."""
    h = np.array([[2.0 + 3.0j]], order='F', dtype=complex)
    b = np.array([[4.0 + 5.0j]], order='F', dtype=complex)

    h_orig = h.copy()

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    x, info = mb02rz('N', h_lu, ipiv, b)
    assert info == 0

    expected = np.array([[4.0 + 5.0j]]) / (2.0 + 3.0j)
    np.testing.assert_allclose(x, expected, rtol=1e-14)

def test_identity_system():
    """Test with identity matrix (trivial solution)."""
    n = 3
    h = np.eye(n, dtype=complex, order='F')
    b = np.array([[1.0 + 0.0j], [2.0 + 1.0j], [3.0 + 2.0j]], order='F', dtype=complex)

    b_orig = b.copy()

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    x, info = mb02rz('N', h_lu, ipiv, b)
    assert info == 0

    np.testing.assert_allclose(x, b_orig, rtol=1e-14)


"""Test error conditions."""

def test_invalid_trans_parameter():
    """Test invalid trans parameter returns error code."""
    h = np.array([[1.0 + 0.0j]], order='F', dtype=complex)
    b = np.array([[1.0 + 0.0j]], order='F', dtype=complex)

    h_lu, ipiv, info_lu = mb02sz(h)

    x, info = mb02rz('X', h_lu, ipiv, b)
    assert info == -1


"""Integration tests with MB02SZ."""

def test_factorize_then_solve_workflow():
    """
    Test typical workflow: factorize once, solve multiple times.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 6

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h_orig = h.copy()

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    for _ in range(3):
        b = np.random.randn(n, 1) + 1j * np.random.randn(n, 1)
        b = np.asfortranarray(b)
        b_orig = b.copy()

        x, info = mb02rz('N', h_lu, ipiv, b)
        assert info == 0

        residual = h_orig @ x - b_orig
        np.testing.assert_allclose(residual, 0.0, atol=1e-13)
