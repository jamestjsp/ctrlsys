"""
Tests for MB01RX: Triangular symmetric rank-k update.

Computes R = alpha*R + beta*op(A)*B or R = alpha*R + beta*B*op(A)
where only upper or lower triangle is computed/stored.
"""

import numpy as np
import pytest


def test_mb01rx_side_left_uplo_upper_trans_no():
    """
    Test SIDE='L', UPLO='U', TRANS='N': R = alpha*R + beta*A*B (upper triangle).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb01rx

    np.random.seed(42)
    m, n = 3, 2
    alpha, beta = 2.0, 0.5

    # Input matrices (column-major)
    r = np.array([
        [4.0, 1.0, 2.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 5.0]
    ], order='F', dtype=float)

    a = np.array([
        [1.0, 2.0],
        [3.0, 1.0],
        [2.0, 4.0]
    ], order='F', dtype=float)

    b = np.array([
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 1.0]
    ], order='F', dtype=float)

    # Expected: R_new[i,j] = alpha*R[i,j] + beta*sum_k(A[i,k]*B[k,j]) for i <= j
    # Only upper triangle computed
    r_expected = np.array([
        [10.0, 4.5, 6.5],
        [0.0, 8.5, 7.0],
        [0.0, 0.0, 15.0]
    ], order='F', dtype=float)

    r_out, info = mb01rx('L', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    # Upper triangle only
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01rx_side_left_uplo_lower_trans_yes():
    """
    Test SIDE='L', UPLO='L', TRANS='T': R = alpha*R + beta*A'*B (lower triangle).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb01rx

    np.random.seed(123)
    m, n = 3, 4
    alpha, beta = 1.5, -1.0

    r = np.array([
        [2.0, 0.0, 0.0],
        [1.0, 3.0, 0.0],
        [2.0, 1.0, 4.0]
    ], order='F', dtype=float)

    a = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 2.0],
        [3.0, 1.0, 2.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 2.0],
        [3.0, 1.0, 2.0]
    ], order='F', dtype=float)

    # Expected: R_new[i,j] = alpha*R[i,j] + beta*sum_k(A[k,i]*B[k,j]) for i >= j
    r_expected = np.array([
        [-12.0, 0.0, 0.0],
        [-7.5, -5.5, 0.0],
        [-12.0, -9.5, -12.0]
    ], order='F', dtype=float)

    r_out, info = mb01rx('L', 'L', 'T', m, n, alpha, beta, r, a, b)

    assert info == 0
    # Lower triangle only
    np.testing.assert_allclose(np.tril(r_out), np.tril(r_expected), rtol=1e-14)


def test_mb01rx_side_right_uplo_upper_trans_no():
    """
    Test SIDE='R', UPLO='U', TRANS='N': R = alpha*R + beta*B*A (upper triangle).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb01rx

    np.random.seed(456)
    m, n = 3, 2
    alpha, beta = 0.5, 2.0

    r = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 2.0, 3.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 3.0]
    ], order='F', dtype=float)

    a = np.array([
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 1.0]
    ], order='F', dtype=float)

    # Expected: R_new[i,j] = alpha*R[i,j] + beta*sum_k(B[i,k]*A[k,j]) for i <= j
    r_expected = np.array([
        [8.5, 11.0, 10.5],
        [0.0, 9.0, 15.5],
        [0.0, 0.0, 12.5]
    ], order='F', dtype=float)

    r_out, info = mb01rx('R', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01rx_side_right_uplo_lower_trans_yes():
    """
    Test SIDE='R', UPLO='L', TRANS='T': R = alpha*R + beta*B*A' (lower triangle).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb01rx

    np.random.seed(789)
    m, n = 4, 3
    alpha, beta = 0.0, 1.0

    # When alpha=0, R is not referenced on input
    r = np.zeros((m, m), order='F', dtype=float)

    b = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 2.0],
        [3.0, 1.0, 2.0]
    ], order='F', dtype=float)

    a = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 2.0],
        [3.0, 1.0, 2.0]
    ], order='F', dtype=float)

    # Expected: R_new[i,j] = beta*sum_k(B[i,k]*A[k,j]) for i >= j
    r_expected = np.array([
        [6.0, 0.0, 0.0, 0.0],
        [7.0, 14.0, 0.0, 0.0],
        [7.0, 10.0, 9.0, 0.0],
        [7.0, 13.0, 9.0, 14.0]
    ], order='F', dtype=float)

    r_out, info = mb01rx('R', 'L', 'T', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.tril(r_out), np.tril(r_expected), rtol=1e-14)


def test_mb01rx_alpha_zero():
    """
    Test special case: alpha=0, beta=1 -> R = op(A)*B (upper triangle).

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb01rx

    np.random.seed(999)
    m, n = 3, 2
    alpha, beta = 0.0, 1.0

    # R input is ignored when alpha=0
    r = np.ones((m, m), order='F', dtype=float)

    a = np.array([
        [1.0, 2.0],
        [3.0, 1.0],
        [2.0, 4.0]
    ], order='F', dtype=float)

    b = np.array([
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 1.0]
    ], order='F', dtype=float)

    # Expected: R_new = A*B (upper triangle only)
    r_expected = np.array([
        [4.0, 5.0, 5.0],
        [0.0, 5.0, 10.0],
        [0.0, 0.0, 10.0]
    ], order='F', dtype=float)

    r_out, info = mb01rx('L', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01rx_beta_zero():
    """
    Test special case: alpha=2, beta=0 -> R = 2*R (upper triangle).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb01rx

    np.random.seed(111)
    m, n = 3, 2
    alpha, beta = 2.0, 0.0

    r = np.array([
        [4.0, 1.0, 2.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 5.0]
    ], order='F', dtype=float)

    # A and B are not referenced when beta=0
    a = np.zeros((m, n), order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)

    # Expected: R_new = 2*R (upper triangle)
    r_expected = np.array([
        [8.0, 2.0, 4.0],
        [5.0, 6.0, 2.0],
        [6.0, 7.0, 10.0]
    ], order='F', dtype=float)

    r_out, info = mb01rx('L', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01rx_alpha_beta_zero():
    """
    Test special case: alpha=0, beta=0 -> R = 0 (upper triangle).
    """
    from ctrlsys import mb01rx

    m, n = 3, 2
    alpha, beta = 0.0, 0.0

    r = np.array([
        [4.0, 1.0, 2.0],
        [5.0, 3.0, 1.0],
        [6.0, 7.0, 5.0]
    ], order='F', dtype=float)

    a = np.zeros((m, n), order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)

    # Expected: R = 0 (upper triangle)
    r_expected = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [6.0, 7.0, 0.0]
    ], order='F', dtype=float)

    r_out, info = mb01rx('L', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01rx_error_invalid_side():
    """Test error handling: invalid SIDE parameter."""
    from ctrlsys import mb01rx

    m, n = 3, 2
    r = np.eye(m, order='F', dtype=float)
    a = np.eye(m, n, order='F', dtype=float)
    b = np.eye(n, m, order='F', dtype=float)

    r_out, info = mb01rx('X', 'U', 'N', m, n, 1.0, 1.0, r, a, b)
    assert info == -1


def test_mb01rx_error_invalid_uplo():
    """Test error handling: invalid UPLO parameter."""
    from ctrlsys import mb01rx

    m, n = 3, 2
    r = np.eye(m, order='F', dtype=float)
    a = np.eye(m, n, order='F', dtype=float)
    b = np.eye(n, m, order='F', dtype=float)

    r_out, info = mb01rx('L', 'X', 'N', m, n, 1.0, 1.0, r, a, b)
    assert info == -2


def test_mb01rx_error_invalid_trans():
    """Test error handling: invalid TRANS parameter."""
    from ctrlsys import mb01rx

    m, n = 3, 2
    r = np.eye(m, order='F', dtype=float)
    a = np.eye(m, n, order='F', dtype=float)
    b = np.eye(n, m, order='F', dtype=float)

    r_out, info = mb01rx('L', 'U', 'X', m, n, 1.0, 1.0, r, a, b)
    assert info == -3


def test_mb01rx_error_m_negative():
    """Test error handling: M < 0."""
    from ctrlsys import mb01rx

    m, n = -1, 2
    r = np.eye(1, order='F', dtype=float)
    a = np.eye(1, 2, order='F', dtype=float)
    b = np.eye(2, 1, order='F', dtype=float)

    r_out, info = mb01rx('L', 'U', 'N', m, n, 1.0, 1.0, r, a, b)
    assert info == -4


def test_mb01rx_error_n_negative():
    """Test error handling: N < 0."""
    from ctrlsys import mb01rx

    m, n = 3, -1
    r = np.eye(m, order='F', dtype=float)
    a = np.eye(m, 1, order='F', dtype=float)
    b = np.eye(1, m, order='F', dtype=float)

    r_out, info = mb01rx('L', 'U', 'N', m, n, 1.0, 1.0, r, a, b)
    assert info == -5


def test_mb01rx_property_symmetry():
    """
    Validate mathematical property: Result should be symmetric when B = A'.

    For SIDE='L', UPLO='U': R = alpha*R + beta*A*A'
    For SIDE='R', UPLO='L': R = alpha*R + beta*A'*A
    Both should produce symmetric R.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb01rx

    np.random.seed(222)
    m, n = 4, 3
    alpha, beta = 0.0, 1.0

    a = np.random.randn(m, n).astype(float, order='F')

    # Test 1: R = A*A' (upper triangle)
    r1 = np.zeros((m, m), order='F', dtype=float)
    r1_out, info = mb01rx('L', 'U', 'N', m, n, alpha, beta, r1, a, a.T.copy(order='F'))
    assert info == 0

    # Full symmetric matrix
    r1_full = a @ a.T
    np.testing.assert_allclose(np.triu(r1_out), np.triu(r1_full), rtol=1e-14)

    # Test 2: R = A'*A (lower triangle) - different dimensions
    m2, n2 = n, m
    r2 = np.zeros((m2, m2), order='F', dtype=float)
    # For SIDE='L', TRANS='T': need A(n2 x m2) = A(4x3)
    a2 = a.copy(order='F')  # 4x3 - correct dimensions
    b2 = a.copy(order='F')  # 4x3 - same as A for symmetric result
    r2_out, info = mb01rx('L', 'L', 'T', m2, n2, alpha, beta, r2, a2, b2)
    assert info == 0

    r2_full = a.T @ a
    np.testing.assert_allclose(np.tril(r2_out), np.tril(r2_full), rtol=1e-14)


def test_mb01rx_property_scale_equivalence():
    """
    Validate mathematical property: Scaling equivalence.

    R = alpha*R0 + beta*A*B should equal R0 scaled by alpha plus beta*(A*B).

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb01rx

    np.random.seed(333)
    m, n = 3, 2
    alpha, beta = 1.5, 2.0

    r0 = np.random.randn(m, m).astype(float, order='F')
    r0 = np.triu(r0)  # Upper triangular

    a = np.random.randn(m, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    # Compute via routine
    r_routine, info = mb01rx('L', 'U', 'N', m, n, alpha, beta, r0.copy(order='F'), a, b)
    assert info == 0

    # Compute manually
    r_manual = alpha * r0 + beta * (a @ b)

    np.testing.assert_allclose(np.triu(r_routine), np.triu(r_manual), rtol=1e-14)


def test_mb01rx_edge_case_m_zero():
    """Test edge case: M=0 (empty matrix)."""
    from ctrlsys import mb01rx

    m, n = 0, 2
    r = np.zeros((1, 1), order='F', dtype=float)
    a = np.zeros((1, n), order='F', dtype=float)
    b = np.zeros((n, 1), order='F', dtype=float)

    r_out, info = mb01rx('L', 'U', 'N', m, n, 1.0, 1.0, r, a, b)
    assert info == 0


def test_mb01rx_edge_case_n_zero():
    """Test edge case: N=0 (R = alpha*R)."""
    from ctrlsys import mb01rx

    m, n = 3, 0
    alpha = 2.5

    r = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    a = np.zeros((m, 1), order='F', dtype=float)
    b = np.zeros((1, m), order='F', dtype=float)

    r_expected = alpha * r

    r_out, info = mb01rx('L', 'U', 'N', m, n, alpha, 1.0, r, a, b)
    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)
