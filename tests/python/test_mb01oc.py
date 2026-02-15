"""
Tests for MB01OC: Special symmetric rank 2k operation with Hessenberg matrix.

Computes:
    R := alpha*R + beta*H*X + beta*X*H'     (TRANS='N')
or
    R := alpha*R + beta*H'*X + beta*X*H     (TRANS='T' or 'C')

where R and X are N-by-N symmetric matrices and H is N-by-N upper Hessenberg.
"""

import numpy as np
import pytest
from ctrlsys import mb01oc


def make_symmetric(a, uplo='U'):
    """Create symmetric matrix from triangle."""
    if uplo == 'U':
        return np.triu(a) + np.triu(a, 1).T
    else:
        return np.tril(a) + np.tril(a, -1).T


def make_upper_hessenberg(n):
    """Create upper Hessenberg matrix (upper triangular plus subdiagonal)."""
    np.random.seed(12345)
    h = np.random.randn(n, n).astype(float, order='F')
    for i in range(2, n):
        for j in range(i - 1):
            h[i, j] = 0.0
    return h


def test_mb01oc_basic_upper_notrans():
    """
    Test basic upper triangle, no transpose case.
    R := alpha*R + beta*H*X + beta*X*H'

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3
    alpha = 0.5
    beta = 1.0

    r = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0]
    ], order='F', dtype=float)

    x = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    x_full = make_symmetric(x, 'U')

    r_out, info = mb01oc('U', 'N', n, alpha, beta, r, h, x)

    assert info == 0

    r_full_in = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ], order='F', dtype=float)

    hx = h @ x_full
    xht = x_full @ h.T
    r_expected_full = alpha * r_full_in + beta * hx + beta * xht

    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(r_out[i, j], r_expected_full[i, j], rtol=1e-14)


def test_mb01oc_basic_lower_notrans():
    """
    Test basic lower triangle, no transpose case.
    R := alpha*R + beta*H*X + beta*X*H'

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3
    alpha = 1.0
    beta = 0.5

    r = np.array([
        [2.0, 0.0, 0.0],
        [1.0, 3.0, 0.0],
        [0.5, 0.3, 4.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 1.0],
        [3.0, 4.0, 2.0],
        [0.0, 5.0, 3.0]
    ], order='F', dtype=float)

    x = np.array([
        [1.0, 0.0, 0.0],
        [0.2, 2.0, 0.0],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    x_full = make_symmetric(x, 'L')
    r_full_in = make_symmetric(r, 'L')

    r_out, info = mb01oc('L', 'N', n, alpha, beta, r, h, x)

    assert info == 0

    hx = h @ x_full
    xht = x_full @ h.T
    r_expected_full = alpha * r_full_in + beta * hx + beta * xht

    for i in range(n):
        for j in range(i + 1):
            np.testing.assert_allclose(r_out[i, j], r_expected_full[i, j], rtol=1e-14)


def test_mb01oc_basic_upper_trans():
    """
    Test basic upper triangle, transpose case.
    R := alpha*R + beta*H'*X + beta*X*H

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3
    alpha = 1.0
    beta = 2.0

    r = np.array([
        [1.0, 1.0, 1.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    h = np.array([
        [0.5, 1.0, 0.5],
        [2.0, 1.5, 1.0],
        [0.0, 3.0, 2.5]
    ], order='F', dtype=float)

    x = np.array([
        [0.5, 0.2, 0.1],
        [0.0, 1.0, 0.3],
        [0.0, 0.0, 0.8]
    ], order='F', dtype=float)

    x_full = make_symmetric(x, 'U')
    r_full_in = make_symmetric(r, 'U')

    r_out, info = mb01oc('U', 'T', n, alpha, beta, r, h, x)

    assert info == 0

    htx = h.T @ x_full
    xh = x_full @ h
    r_expected_full = alpha * r_full_in + beta * htx + beta * xh

    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(r_out[i, j], r_expected_full[i, j], rtol=1e-14)


def test_mb01oc_basic_lower_trans():
    """
    Test basic lower triangle, transpose case.
    R := alpha*R + beta*H'*X + beta*X*H

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3
    alpha = 0.25
    beta = 1.5

    r = np.array([
        [3.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [1.0, 1.0, 1.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 0.5, 0.25],
        [2.0, 1.0, 0.5],
        [0.0, 2.0, 1.0]
    ], order='F', dtype=float)

    x = np.array([
        [0.4, 0.0, 0.0],
        [0.2, 0.6, 0.0],
        [0.1, 0.15, 0.8]
    ], order='F', dtype=float)

    x_full = make_symmetric(x, 'L')
    r_full_in = make_symmetric(r, 'L')

    r_out, info = mb01oc('L', 'T', n, alpha, beta, r, h, x)

    assert info == 0

    htx = h.T @ x_full
    xh = x_full @ h
    r_expected_full = alpha * r_full_in + beta * htx + beta * xh

    for i in range(n):
        for j in range(i + 1):
            np.testing.assert_allclose(r_out[i, j], r_expected_full[i, j], rtol=1e-14)


def test_mb01oc_alpha_zero():
    """
    Test alpha=0 case: R := beta*H*X + beta*X*H'.

    Random seed: 100 (for reproducibility)
    """
    np.random.seed(100)
    n = 4
    alpha = 0.0
    beta = 1.0

    r = np.random.randn(n, n).astype(float, order='F')
    r = np.triu(r)

    h = make_upper_hessenberg(n)

    x = np.random.randn(n, n).astype(float, order='F')
    x = np.triu(x)
    x_full = make_symmetric(x, 'U')

    r_out, info = mb01oc('U', 'N', n, alpha, beta, r, h, x)

    assert info == 0

    hx = h @ x_full
    xht = x_full @ h.T
    r_expected_full = beta * hx + beta * xht

    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(r_out[i, j], r_expected_full[i, j], rtol=1e-14)


def test_mb01oc_beta_zero():
    """
    Test beta=0 case: R := alpha*R.

    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n = 4
    alpha = 2.5
    beta = 0.0

    r = np.random.randn(n, n).astype(float, order='F')
    r = np.triu(r)
    r_copy = r.copy()

    h = make_upper_hessenberg(n)
    x = np.random.randn(n, n).astype(float, order='F')
    x = np.triu(x)

    r_out, info = mb01oc('U', 'N', n, alpha, beta, r, h, x)

    assert info == 0

    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(r_out[i, j], alpha * r_copy[i, j], rtol=1e-14)


def test_mb01oc_alpha_beta_zero():
    """
    Test alpha=0, beta=0 case: R := 0.

    Random seed: 300 (for reproducibility)
    """
    np.random.seed(300)
    n = 3
    alpha = 0.0
    beta = 0.0

    r = np.random.randn(n, n).astype(float, order='F')
    r = np.triu(r)

    h = make_upper_hessenberg(n)
    x = np.random.randn(n, n).astype(float, order='F')
    x = np.triu(x)

    r_out, info = mb01oc('U', 'N', n, alpha, beta, r, h, x)

    assert info == 0

    for i in range(n):
        for j in range(i, n):
            assert r_out[i, j] == 0.0


def test_mb01oc_n_zero():
    """
    Test n=0 case (quick return).
    """
    n = 0
    alpha = 1.0
    beta = 1.0

    r = np.array([], dtype=float, order='F').reshape(0, 0)
    h = np.array([], dtype=float, order='F').reshape(0, 0)
    x = np.array([], dtype=float, order='F').reshape(0, 0)

    r_out, info = mb01oc('U', 'N', n, alpha, beta, r, h, x)

    assert info == 0


def test_mb01oc_n_one():
    """
    Test n=1 edge case.
    """
    n = 1
    alpha = 0.5
    beta = 2.0

    r = np.array([[3.0]], order='F', dtype=float)
    h = np.array([[2.0]], order='F', dtype=float)
    x = np.array([[1.5]], order='F', dtype=float)

    r_out, info = mb01oc('U', 'N', n, alpha, beta, r, h, x)

    assert info == 0
    expected = alpha * 3.0 + beta * 2.0 * 1.5 + beta * 1.5 * 2.0
    np.testing.assert_allclose(r_out[0, 0], expected, rtol=1e-14)


def test_mb01oc_invalid_uplo():
    """Test invalid uplo parameter."""
    n = 2
    r = np.zeros((n, n), order='F', dtype=float)
    h = np.zeros((n, n), order='F', dtype=float)
    x = np.zeros((n, n), order='F', dtype=float)

    r_out, info = mb01oc('X', 'N', n, 1.0, 1.0, r, h, x)
    assert info == -1


def test_mb01oc_invalid_trans():
    """Test invalid trans parameter."""
    n = 2
    r = np.zeros((n, n), order='F', dtype=float)
    h = np.zeros((n, n), order='F', dtype=float)
    x = np.zeros((n, n), order='F', dtype=float)

    r_out, info = mb01oc('U', 'X', n, 1.0, 1.0, r, h, x)
    assert info == -2


def test_mb01oc_invalid_n():
    """Test invalid n parameter (negative)."""
    n = -1
    r = np.zeros((1, 1), order='F', dtype=float)
    h = np.zeros((1, 1), order='F', dtype=float)
    x = np.zeros((1, 1), order='F', dtype=float)

    r_out, info = mb01oc('U', 'N', n, 1.0, 1.0, r, h, x)
    assert info == -3


def test_mb01oc_mathematical_symmetry():
    """
    Validate that output R preserves symmetry in the computed triangle.

    The full operation H*X + X*H' is symmetric when X is symmetric.
    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n = 5
    alpha = 1.0
    beta = 1.0

    r = np.random.randn(n, n).astype(float, order='F')
    r_full = make_symmetric(r, 'U')
    r_upper = np.triu(r_full).astype(float, order='F')

    h = make_upper_hessenberg(n)

    x = np.random.randn(n, n).astype(float, order='F')
    x = np.triu(x)
    x_full = make_symmetric(x, 'U')

    r_out, info = mb01oc('U', 'N', n, alpha, beta, r_upper, h, x)

    assert info == 0

    r_expected_full = alpha * r_full + beta * (h @ x_full) + beta * (x_full @ h.T)

    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(r_out[i, j], r_expected_full[i, j], rtol=1e-13)

    for i in range(n):
        for j in range(i + 1, n):
            np.testing.assert_allclose(r_expected_full[i, j], r_expected_full[j, i], rtol=1e-14)


def test_mb01oc_larger_matrix():
    """
    Test with larger matrix to validate scaling.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 10
    alpha = 0.3
    beta = 1.2

    r = np.random.randn(n, n).astype(float, order='F')
    r = np.triu(r)
    r_full = make_symmetric(r, 'U')

    h = make_upper_hessenberg(n)

    x = np.random.randn(n, n).astype(float, order='F')
    x = np.triu(x)
    x_full = make_symmetric(x, 'U')

    r_out, info = mb01oc('U', 'N', n, alpha, beta, r, h, x)

    assert info == 0

    hx = h @ x_full
    xht = x_full @ h.T
    r_expected_full = alpha * r_full + beta * hx + beta * xht

    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(r_out[i, j], r_expected_full[i, j], rtol=1e-13)
