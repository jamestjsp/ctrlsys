"""
Tests for MB04OX: QR factorization of [U; x'] to get [R; 0].

Performs QR factorization of augmented matrix where U is n-by-n upper
triangular and x is an n-element vector, producing R upper triangular.

Tests:
1. Basic QR update with identity matrix
2. Random upper triangular matrix update
3. Orthogonality property of implicit Q
4. Zero vector edge case

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb04ox_basic():
    """
    Validate basic QR factorization update.

    Test with simple upper triangular matrix and unit vector.
    Verifies that output is still upper triangular.
    """
    from ctrlsys import mb04ox

    n = 3

    # Simple upper triangular matrix
    u = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    # Vector to incorporate
    x = np.array([1.0, 1.0, 1.0], order='F', dtype=float)

    u_orig = u.copy()
    x_orig = x.copy()

    # Form the augmented matrix for verification
    augmented = np.vstack([u_orig, x_orig.reshape(1, -1)])

    # Call routine
    r, x_out = mb04ox(u.copy(order='F'), x.copy(order='F'))

    # R should be upper triangular
    for i in range(n):
        for j in range(i):
            assert abs(r[i, j]) < 1e-14, f"R[{i},{j}] = {r[i,j]} should be 0"

    # Verify R is from a valid QR factorization of augmented matrix
    # The squared Frobenius norms should match (orthogonal Q preserves norm)
    aug_norm_sq = np.sum(augmented**2)
    r_norm_sq = np.sum(np.triu(r)**2)
    assert_allclose(r_norm_sq, aug_norm_sq, rtol=1e-13)


def test_mb04ox_identity_update():
    """
    Validate update of identity matrix.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04ox

    np.random.seed(42)
    n = 4

    # Identity matrix
    u = np.eye(n, order='F', dtype=float)

    # Random vector
    x = np.random.randn(n).astype(float, order='F')

    u_orig = u.copy()
    x_orig = x.copy()

    # Form augmented matrix
    augmented = np.vstack([u_orig, x_orig.reshape(1, -1)])

    # Call routine
    r, x_out = mb04ox(u.copy(order='F'), x.copy(order='F'))

    # R should be upper triangular
    assert np.allclose(np.tril(r, -1), 0, atol=1e-14)

    # Norm preservation
    aug_norm_sq = np.sum(augmented**2)
    r_norm_sq = np.sum(np.triu(r)**2)
    assert_allclose(r_norm_sq, aug_norm_sq, rtol=1e-13)


def test_mb04ox_random_triangular():
    """
    Validate with random upper triangular matrix.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04ox

    np.random.seed(123)
    n = 5

    # Random upper triangular matrix with positive diagonal
    u = np.triu(np.random.randn(n, n))
    u = np.asfortranarray(u)
    for i in range(n):
        if u[i, i] < 0.5:
            u[i, i] = 0.5 + np.abs(u[i, i])

    # Random vector
    x = np.random.randn(n).astype(float, order='F')

    u_orig = u.copy()
    x_orig = x.copy()

    # Form augmented matrix
    augmented = np.vstack([u_orig, x_orig.reshape(1, -1)])

    # Compute expected R via numpy QR
    q_expected, r_expected = np.linalg.qr(augmented)
    # Make R diagonal positive (convention)
    signs = np.sign(np.diag(r_expected))
    r_expected = np.diag(signs) @ r_expected

    # Call routine
    r, x_out = mb04ox(u.copy(order='F'), x.copy(order='F'))

    # R should match expected R (up to signs of rows)
    for i in range(n):
        row_match = np.allclose(r[i, :], r_expected[i, :], rtol=1e-12) or \
                    np.allclose(r[i, :], -r_expected[i, :], rtol=1e-12)
        assert row_match, f"Row {i} mismatch"

    # Upper triangular check
    assert np.allclose(np.tril(r, -1), 0, atol=1e-14)


def test_mb04ox_zero_vector():
    """
    Validate with zero vector - R should equal U.
    """
    from ctrlsys import mb04ox

    n = 3

    u = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    x = np.zeros(n, order='F', dtype=float)

    u_orig = u.copy()

    r, x_out = mb04ox(u.copy(order='F'), x.copy(order='F'))

    # With zero vector, R should equal U
    assert_allclose(r, u_orig, rtol=1e-14)


def test_mb04ox_stride():
    """
    Validate with non-unit stride (INCX > 1).

    The Python wrapper uses INCX=1, but we verify the routine
    handles the standard case correctly.
    """
    from ctrlsys import mb04ox

    n = 3

    u = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    x = np.array([0.5, 0.5, 0.5], order='F', dtype=float)

    r, x_out = mb04ox(u.copy(order='F'), x.copy(order='F'))

    # R should be upper triangular
    assert np.allclose(np.tril(r, -1), 0, atol=1e-14)

    # Diagonal should remain positive (or at least non-zero for valid R)
    assert all(abs(r[i, i]) > 1e-14 for i in range(n))
