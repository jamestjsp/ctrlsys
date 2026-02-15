"""
Tests for MB02TZ: Condition number estimation of complex Hessenberg matrix.

Estimates reciprocal condition number using LU factorization from MB02SZ.
"""

import numpy as np
import pytest
from ctrlsys import mb02sz, mb02tz


"""Basic functionality tests."""

def test_well_conditioned_2x2():
    """
    Test well-conditioned 2x2 matrix.

    H = [[2+0j, 1+0j],
         [0+0j, 2+0j]]

    This is upper triangular (special Hessenberg), well-conditioned.
    """
    h = np.array([
        [2.0 + 0.0j, 1.0 + 0.0j],
        [0.0 + 0.0j, 2.0 + 0.0j]
    ], order='F', dtype=complex)

    hnorm = np.linalg.norm(h, 1)

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    rcond, info = mb02tz('1', hnorm, h_lu, ipiv)
    assert info == 0
    assert 0.0 < rcond <= 1.0

    actual_cond = np.linalg.cond(np.array([
        [2.0 + 0.0j, 1.0 + 0.0j],
        [0.0 + 0.0j, 2.0 + 0.0j]
    ]), 1)

    np.testing.assert_allclose(rcond, 1.0 / actual_cond, rtol=0.5)

def test_identity_matrix_condition():
    """
    Test identity matrix has condition number 1.

    rcond should be 1.0.
    """
    n = 3
    h = np.eye(n, dtype=complex, order='F')

    hnorm = np.linalg.norm(h, 1)

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    rcond, info = mb02tz('1', hnorm, h_lu, ipiv)
    assert info == 0

    np.testing.assert_allclose(rcond, 1.0, rtol=1e-10)

def test_one_norm_vs_infinity_norm():
    """
    Test both 1-norm and infinity-norm condition estimates.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h_orig = h.copy()

    hnorm_1 = np.linalg.norm(h_orig, 1)
    hnorm_inf = np.linalg.norm(h_orig, np.inf)

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    rcond_1, info = mb02tz('1', hnorm_1, h_lu, ipiv)
    assert info == 0
    assert 0.0 < rcond_1 <= 1.0

    rcond_inf, info = mb02tz('I', hnorm_inf, h_lu, ipiv)
    assert info == 0
    assert 0.0 < rcond_inf <= 1.0


"""Test mathematical properties of condition estimation."""

def test_rcond_bounds():
    """
    Validate 0 < rcond <= 1 for non-singular matrices.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 5

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h[np.diag_indices(n)] += 5.0

    hnorm = np.linalg.norm(h, 1)

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    rcond, info = mb02tz('1', hnorm, h_lu, ipiv)
    assert info == 0
    assert rcond > 0.0
    assert rcond <= 1.0

def test_ill_conditioned_matrix():
    """
    Test ill-conditioned matrix has small rcond.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h[0, 0] = 1e10 + 0j
    h[n - 1, n - 1] = 1e-10 + 0j

    hnorm = np.linalg.norm(h, 1)

    h_lu, ipiv, info_lu = mb02sz(h)
    if info_lu == 0:
        rcond, info = mb02tz('1', hnorm, h_lu, ipiv)
        assert info == 0
        assert rcond < 1e-8

def test_condition_estimate_reasonable():
    """
    Validate condition estimate is within reasonable range of true value.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h[np.diag_indices(n)] += 3.0

    h_orig = h.copy()
    hnorm = np.linalg.norm(h_orig, 1)

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    rcond, info = mb02tz('1', hnorm, h_lu, ipiv)
    assert info == 0

    true_cond = np.linalg.cond(h_orig, 1)
    estimated_cond = 1.0 / rcond

    assert estimated_cond >= true_cond / 100
    assert estimated_cond <= true_cond * 100


"""Edge cases and boundary conditions."""

def test_n_equals_1():
    """Test 1x1 matrix."""
    h = np.array([[3.0 + 4.0j]], order='F', dtype=complex)
    hnorm = abs(h[0, 0])

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    rcond, info = mb02tz('1', hnorm, h_lu, ipiv)
    assert info == 0

    np.testing.assert_allclose(rcond, 1.0, rtol=1e-10)

def test_norm_option_O():
    """Test 'O' as equivalent to '1' for 1-norm."""
    n = 3
    np.random.seed(999)

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h[np.diag_indices(n)] += 2.0

    h_orig = h.copy()
    hnorm = np.linalg.norm(h_orig, 1)

    h_lu1, ipiv1, _ = mb02sz(h.copy())
    h_lu2, ipiv2, _ = mb02sz(h_orig.copy())

    rcond_1, info1 = mb02tz('1', hnorm, h_lu1, ipiv1)
    rcond_o, info2 = mb02tz('O', hnorm, h_lu2, ipiv2)

    assert info1 == 0
    assert info2 == 0
    np.testing.assert_allclose(rcond_1, rcond_o, rtol=1e-14)


"""Test error conditions."""

def test_negative_hnorm():
    """Test negative hnorm raises error or returns error code."""
    h = np.array([[1.0 + 0.0j]], order='F', dtype=complex)

    h_lu, ipiv, _ = mb02sz(h)

    rcond, info = mb02tz('1', -1.0, h_lu, ipiv)
    assert info == -3

def test_invalid_norm_parameter():
    """Test invalid norm parameter."""
    h = np.array([[1.0 + 0.0j]], order='F', dtype=complex)

    h_lu, ipiv, _ = mb02sz(h)

    rcond, info = mb02tz('X', 1.0, h_lu, ipiv)
    assert info == -1


"""Integration tests with MB02SZ."""

def test_factorize_then_estimate_workflow():
    """
    Test typical workflow: factorize, estimate condition.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 5

    h = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn() + 1j * np.random.randn()

    h[np.diag_indices(n)] += 3.0

    h_orig = h.copy()
    hnorm = np.linalg.norm(h_orig, 1)

    h_lu, ipiv, info_lu = mb02sz(h)
    assert info_lu == 0

    rcond, info = mb02tz('1', hnorm, h_lu, ipiv)
    assert info == 0
    assert 0 < rcond <= 1

def test_zero_hnorm_gives_zero_rcond():
    """Test that hnorm=0 returns rcond=0."""
    h = np.array([[1.0 + 0.0j]], order='F', dtype=complex)

    h_lu, ipiv, _ = mb02sz(h)

    rcond, info = mb02tz('1', 0.0, h_lu, ipiv)
    assert info == 0
    assert rcond == 0.0
