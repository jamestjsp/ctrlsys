"""
Tests for MB01RD: Symmetric rank-k matrix update.

Computes R = alpha*R + beta*op(A)*X*op(A)' where R and X are symmetric.
"""

import numpy as np
import pytest


def test_mb01rd_basic_upper_notrans():
    """
    Validate basic functionality with UPLO='U', TRANS='N'.

    R = alpha*R + beta*A*X*A'
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb01rd

    np.random.seed(42)
    m, n = 3, 2
    alpha, beta = 1.0, 1.0

    r = np.array([
        [4.0, 2.0, 1.0],
        [0.0, 3.0, 2.0],
        [0.0, 0.0, 5.0]
    ], order='F', dtype=float)

    a = np.array([
        [1.0, 2.0],
        [3.0, 1.0],
        [2.0, 4.0]
    ], order='F', dtype=float)

    x = np.array([
        [2.0, 0.5],
        [0.5, 1.0]
    ], order='F', dtype=float)

    x_full = np.array([
        [2.0, 0.5],
        [0.5, 1.0]
    ], order='F', dtype=float)
    r_full = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 3.0, 2.0],
        [1.0, 2.0, 5.0]
    ], order='F', dtype=float)

    r_expected_full = alpha * r_full + beta * (a @ x_full @ a.T)

    r_out, info = mb01rd('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected_full), rtol=1e-14)


def test_mb01rd_basic_lower_notrans():
    """
    Validate basic functionality with UPLO='L', TRANS='N'.

    R = alpha*R + beta*A*X*A'
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb01rd

    np.random.seed(123)
    m, n = 3, 2
    alpha, beta = 2.0, 0.5

    r = np.array([
        [4.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [1.0, 2.0, 5.0]
    ], order='F', dtype=float)

    a = np.array([
        [1.0, 2.0],
        [3.0, 1.0],
        [2.0, 4.0]
    ], order='F', dtype=float)

    x = np.array([
        [2.0, 0.5],
        [0.5, 1.0]
    ], order='F', dtype=float)

    x_full = np.array([
        [2.0, 0.5],
        [0.5, 1.0]
    ], order='F', dtype=float)
    r_full = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 3.0, 2.0],
        [1.0, 2.0, 5.0]
    ], order='F', dtype=float)

    r_expected_full = alpha * r_full + beta * (a @ x_full @ a.T)

    r_out, info = mb01rd('L', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.tril(r_out), np.tril(r_expected_full), rtol=1e-14)


def test_mb01rd_transpose():
    """
    Validate TRANS='T' mode: R = alpha*R + beta*A'*X*A.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb01rd

    np.random.seed(456)
    m, n = 2, 3
    alpha, beta = 1.5, -1.0

    r = np.array([
        [3.0, 1.0],
        [0.0, 4.0]
    ], order='F', dtype=float)

    a = np.array([
        [1.0, 2.0],
        [3.0, 1.0],
        [2.0, 4.0]
    ], order='F', dtype=float)

    x = np.array([
        [2.0, 0.5, 1.0],
        [0.5, 1.0, 0.0],
        [1.0, 0.0, 3.0]
    ], order='F', dtype=float)

    r_full = np.array([
        [3.0, 1.0],
        [1.0, 4.0]
    ], order='F', dtype=float)
    x_full = np.array([
        [2.0, 0.5, 1.0],
        [0.5, 1.0, 0.0],
        [1.0, 0.0, 3.0]
    ], order='F', dtype=float)

    r_expected_full = alpha * r_full + beta * (a.T @ x_full @ a)

    r_out, info = mb01rd('U', 'T', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected_full), rtol=1e-14)


def test_mb01rd_alpha_zero():
    """
    Validate special case: alpha=0 -> R = beta*op(A)*X*op(A)'.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb01rd

    np.random.seed(789)
    m, n = 3, 2
    alpha, beta = 0.0, 1.0

    r = np.ones((m, m), order='F', dtype=float)

    a = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ], order='F', dtype=float)

    x = np.array([
        [2.0, 0.0],
        [0.0, 3.0]
    ], order='F', dtype=float)

    x_full = np.array([
        [2.0, 0.0],
        [0.0, 3.0]
    ], order='F', dtype=float)

    r_expected_full = beta * (a @ x_full @ a.T)

    r_out, info = mb01rd('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected_full), rtol=1e-14)


def test_mb01rd_beta_zero():
    """
    Validate special case: beta=0 -> R = alpha*R.

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb01rd

    np.random.seed(999)
    m, n = 3, 2
    alpha, beta = 2.5, 0.0

    r = np.array([
        [4.0, 2.0, 1.0],
        [0.0, 3.0, 2.0],
        [0.0, 0.0, 5.0]
    ], order='F', dtype=float)

    a = np.zeros((m, n), order='F', dtype=float)
    x = np.zeros((n, n), order='F', dtype=float)

    r_full = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 3.0, 2.0],
        [1.0, 2.0, 5.0]
    ], order='F', dtype=float)

    r_expected_full = alpha * r_full

    r_out, info = mb01rd('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected_full), rtol=1e-14)


def test_mb01rd_property_symmetry_preservation():
    """
    Validate that result R is symmetric (upper part equals lower part transpose).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb01rd

    np.random.seed(111)
    m, n = 4, 3
    alpha, beta = 1.0, 1.0

    r = np.array([
        [4.0, 2.0, 1.0, 0.5],
        [0.0, 3.0, 2.0, 1.0],
        [0.0, 0.0, 5.0, 0.5],
        [0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    a = np.random.randn(m, n).astype(float, order='F')

    x = np.array([
        [2.0, 0.5, 1.0],
        [0.0, 1.0, 0.3],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    r_out, info = mb01rd('U', 'N', m, n, alpha, beta, r, a, x)
    assert info == 0

    r_out_full = np.triu(r_out) + np.triu(r_out, 1).T
    np.testing.assert_allclose(r_out_full, r_out_full.T, rtol=1e-14)


def test_mb01rd_property_identity_transform():
    """
    Validate mathematical property: With A=I, result equals alpha*R + beta*X.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb01rd

    np.random.seed(222)
    m = 3
    n = m
    alpha, beta = 1.0, 1.0

    r = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    a = np.eye(m, order='F', dtype=float)

    x = np.array([
        [7.0, 1.0, 2.0],
        [0.0, 8.0, 3.0],
        [0.0, 0.0, 9.0]
    ], order='F', dtype=float)

    r_full = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ], order='F', dtype=float)
    x_full = np.array([
        [7.0, 1.0, 2.0],
        [1.0, 8.0, 3.0],
        [2.0, 3.0, 9.0]
    ], order='F', dtype=float)

    r_expected = alpha * r_full + beta * x_full

    r_out, info = mb01rd('U', 'N', m, n, alpha, beta, r, a, x)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01rd_error_invalid_uplo():
    """Test error handling: invalid UPLO parameter."""
    from ctrlsys import mb01rd

    m, n = 3, 2
    r = np.eye(m, order='F', dtype=float)
    a = np.eye(m, n, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)

    with pytest.raises(ValueError, match="Parameter 1"):
        mb01rd('X', 'N', m, n, 1.0, 1.0, r, a, x)


def test_mb01rd_error_invalid_trans():
    """Test error handling: invalid TRANS parameter."""
    from ctrlsys import mb01rd

    m, n = 3, 2
    r = np.eye(m, order='F', dtype=float)
    a = np.eye(m, n, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)

    with pytest.raises(ValueError, match="Parameter 2"):
        mb01rd('U', 'X', m, n, 1.0, 1.0, r, a, x)


def test_mb01rd_error_m_negative():
    """Test error handling: M < 0."""
    from ctrlsys import mb01rd

    r = np.eye(1, order='F', dtype=float)
    a = np.eye(1, 2, order='F', dtype=float)
    x = np.eye(2, order='F', dtype=float)

    with pytest.raises(ValueError, match="Parameter 3"):
        mb01rd('U', 'N', -1, 2, 1.0, 1.0, r, a, x)


def test_mb01rd_error_n_negative():
    """Test error handling: N < 0."""
    from ctrlsys import mb01rd

    r = np.eye(3, order='F', dtype=float)
    a = np.eye(3, 1, order='F', dtype=float)
    x = np.eye(1, order='F', dtype=float)

    with pytest.raises(ValueError, match="Parameter 4"):
        mb01rd('U', 'N', 3, -1, 1.0, 1.0, r, a, x)


def test_mb01rd_edge_case_m_zero():
    """Test edge case: M=0."""
    from ctrlsys import mb01rd

    m, n = 0, 2
    r = np.zeros((1, 1), order='F', dtype=float)
    a = np.zeros((1, n), order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)

    r_out, info = mb01rd('U', 'N', m, n, 1.0, 1.0, r, a, x)
    assert info == 0


def test_mb01rd_edge_case_n_zero():
    """Test edge case: N=0 -> R = alpha*R."""
    from ctrlsys import mb01rd

    m, n = 3, 0
    alpha = 2.0

    r = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    a = np.zeros((m, 1), order='F', dtype=float)
    x = np.zeros((1, 1), order='F', dtype=float)

    r_full = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ], order='F', dtype=float)

    r_expected = alpha * r_full

    r_out, info = mb01rd('U', 'N', m, n, alpha, 1.0, r, a, x)
    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)
