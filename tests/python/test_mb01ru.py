"""
Tests for MB01RU: Compute R = alpha*R + beta*op(A)*X*op(A)'

MB01RU computes the matrix formula:
    R_out = alpha*R + beta*op(A)*X*op(A)'
where R and X are symmetric matrices and op(A) = A or A'.
"""

import numpy as np
import pytest
from ctrlsys import mb01ru


def test_mb01ru_basic_no_trans():
    """
    Test basic functionality: R = alpha*R + beta*A*X*A'

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n = 3, 2
    alpha, beta = 1.5, 0.5

    r = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 3.0],
        [1.0, 3.0, 6.0]
    ], order='F', dtype=float)

    a = np.random.randn(m, n).astype(float, order='F')

    x = np.array([
        [3.0, 1.0],
        [1.0, 2.0]
    ], order='F', dtype=float)

    r_orig = r.copy()
    x_orig = x.copy()

    r_expected = alpha * r_orig + beta * (a @ x @ a.T)

    r_out, info = mb01ru('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)
    np.testing.assert_allclose(x, x_orig, rtol=1e-15)


def test_mb01ru_basic_trans():
    """
    Test transpose case: R = alpha*R + beta*A'*X*A

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, n = 2, 3
    alpha, beta = 2.0, 1.0

    r = np.array([
        [5.0, 2.0],
        [2.0, 4.0]
    ], order='F', dtype=float)

    a = np.random.randn(n, m).astype(float, order='F')

    x = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.7],
        [0.3, 0.7, 1.5]
    ], order='F', dtype=float)

    r_orig = r.copy()
    x_orig = x.copy()

    r_expected = alpha * r_orig + beta * (a.T @ x @ a)

    r_out, info = mb01ru('U', 'T', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)
    np.testing.assert_allclose(x, x_orig, rtol=1e-15)


def test_mb01ru_lower_triangular():
    """
    Test lower triangular storage: R = alpha*R + beta*A*X*A'

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, n = 3, 2
    alpha, beta = 1.0, 1.0

    r = np.array([
        [3.0, 0.0, 0.0],
        [1.0, 4.0, 0.0],
        [2.0, 1.5, 5.0]
    ], order='F', dtype=float)

    a = np.random.randn(m, n).astype(float, order='F')

    x = np.array([
        [2.0, 0.0],
        [0.5, 1.5]
    ], order='F', dtype=float)

    r_sym = r + np.tril(r, -1).T
    x_sym = x + np.tril(x, -1).T

    r_expected = alpha * r_sym + beta * (a @ x_sym @ a.T)

    r_out, info = mb01ru('L', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.tril(r_out), np.tril(r_expected), rtol=1e-14)


def test_mb01ru_alpha_zero():
    """
    Test alpha=0: R = beta*A*X*A' (R input ignored)

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m, n = 2, 2
    alpha, beta = 0.0, 1.0

    r = np.array([
        [999.0, 888.0],
        [888.0, 777.0]
    ], order='F', dtype=float)

    a = np.random.randn(m, n).astype(float, order='F')

    x = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ], order='F', dtype=float)

    r_expected = beta * (a @ x @ a.T)

    r_out, info = mb01ru('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01ru_beta_zero():
    """
    Test beta=0: R = alpha*R (X and A ignored)

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m, n = 3, 2
    alpha, beta = 2.5, 0.0

    r = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ], order='F', dtype=float)

    a = np.random.randn(m, n).astype(float, order='F')
    x = np.random.randn(n, n).astype(float, order='F')

    r_orig = r.copy()
    r_expected = alpha * r_orig

    r_out, info = mb01ru('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01ru_alpha_beta_zero():
    """
    Test alpha=0, beta=0: R = 0
    """
    m, n = 2, 2
    alpha, beta = 0.0, 0.0

    r = np.array([
        [5.0, 3.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    a = np.ones((m, n), order='F', dtype=float)
    x = np.ones((n, n), order='F', dtype=float)

    r_out, info = mb01ru('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.zeros((m, m)), rtol=1e-15)


def test_mb01ru_m_zero():
    """
    Test quick return for m=0
    """
    m, n = 0, 3
    alpha, beta = 1.0, 1.0

    r = np.array([], order='F', dtype=float).reshape(0, 0)
    a = np.array([], order='F', dtype=float).reshape(0, n)
    x = np.eye(n, order='F', dtype=float)

    r_out, info = mb01ru('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0


def test_mb01ru_n_zero():
    """
    Test for n=0: reduces to R = alpha*R
    """
    m, n = 2, 0
    alpha, beta = 3.0, 1.0

    r = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ], order='F', dtype=float)

    a = np.array([], order='F', dtype=float).reshape(m, 0)
    x = np.array([], order='F', dtype=float).reshape(0, 0)

    r_orig = r.copy()

    r_out, info = mb01ru('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(alpha * r_orig), rtol=1e-14)


def test_mb01ru_symmetry_preservation():
    """
    Validate mathematical property: result is symmetric.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    m, n = 4, 3
    alpha, beta = 1.0, 1.0

    r_raw = np.random.randn(m, m)
    r = ((r_raw + r_raw.T) / 2).astype(float, order='F')

    a = np.random.randn(m, n).astype(float, order='F')

    x_raw = np.random.randn(n, n)
    x = ((x_raw + x_raw.T) / 2).astype(float, order='F')

    r_expected = alpha * r + beta * (a @ x @ a.T)

    np.testing.assert_allclose(r_expected, r_expected.T, rtol=1e-14)

    r_out, info = mb01ru('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0

    r_full = r_out.copy()
    for i in range(m):
        for j in range(i + 1, m):
            r_full[j, i] = r_full[i, j]

    np.testing.assert_allclose(r_full, r_expected, rtol=1e-14)


def test_mb01ru_invalid_uplo():
    """
    Test error handling for invalid UPLO parameter
    """
    m, n = 2, 2
    r = np.eye(m, order='F', dtype=float)
    a = np.eye(m, n, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)

    r_out, info = mb01ru('X', 'N', m, n, 1.0, 1.0, r, a, x)

    assert info == -1


def test_mb01ru_invalid_trans():
    """
    Test error handling for invalid TRANS parameter
    """
    m, n = 2, 2
    r = np.eye(m, order='F', dtype=float)
    a = np.eye(m, n, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)

    r_out, info = mb01ru('U', 'X', m, n, 1.0, 1.0, r, a, x)

    assert info == -2
