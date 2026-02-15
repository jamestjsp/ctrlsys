"""
Tests for MB04LD - LQ factorization of special structured block matrix.

MB04LD computes [[L A], [0 B]] * Q = [[L_bar 0], [C D]] where L and L_bar are lower triangular.

This is useful for Kalman filter square-root covariance updates.
"""

import numpy as np
import pytest


def test_mb04ld_full_basic():
    """
    Basic test with full matrix A (UPLO='F').

    Validates:
    - L_bar is lower triangular
    - Orthogonal transformation preserves Frobenius norm
    - Output matrices have correct dimensions

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n, m, p = 3, 4, 2

    L = np.tril(np.random.randn(n, n)).astype(float, order='F')
    L += np.diag(np.abs(np.diag(L)) + 1.0)

    A = np.random.randn(n, m).astype(float, order='F')
    B = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04ld

    L_bar, A_out, D, C, tau = mb04ld('F', n, m, p, L.copy(), A.copy(), B.copy())

    # Test 1: L_bar is lower triangular (strict upper should be zero)
    for i in range(n):
        for j in range(i + 1, n):
            assert abs(L_bar[i, j]) < 1e-14, f"L_bar not lower triangular at ({i},{j})"

    # Test 2: Output matrix dimensions
    assert L_bar.shape == (n, n), f"L_bar shape mismatch: {L_bar.shape}"
    assert D.shape == (p, m), f"D shape mismatch: {D.shape}"
    assert C.shape == (p, n), f"C shape mismatch: {C.shape}"
    assert len(tau) == n, f"tau length mismatch: {len(tau)}"

    # Test 3: Verify orthogonal transformation (Frobenius norm preservation)
    original_norm_sq = np.linalg.norm(L)**2 + np.linalg.norm(A)**2 + np.linalg.norm(B)**2
    result_norm_sq = np.linalg.norm(L_bar)**2 + np.linalg.norm(C)**2 + np.linalg.norm(D)**2

    np.testing.assert_allclose(original_norm_sq, result_norm_sq, rtol=1e-13, atol=1e-14,
                               err_msg="Orthogonal transformation should preserve Frobenius norm")


def test_mb04ld_lower_trapezoidal():
    """
    Test with lower trapezoidal matrix A (UPLO='L').

    When UPLO='L', A is lower trapezoidal (N-by-min(N,M) lower triangular part).
    Only elements on or below diagonal are used.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n, m, p = 4, 5, 3

    L = np.tril(np.random.randn(n, n)).astype(float, order='F')
    L += np.diag(np.abs(np.diag(L)) + 2.0)

    A = np.zeros((n, m), order='F', dtype=float)
    for i in range(n):
        for j in range(min(i + 1, m)):
            A[i, j] = np.random.randn()

    B = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04ld

    L_bar, A_out, D, C, tau = mb04ld('L', n, m, p, L.copy(), A.copy(), B.copy())

    assert np.allclose(L_bar, np.tril(L_bar), rtol=1e-14), "L_bar should be lower triangular"

    assert L_bar.shape == (n, n)
    assert C.shape == (p, n)
    assert D.shape == (p, m)
    assert len(tau) == n


def test_mb04ld_norm_preservation():
    """
    Mathematical property test: Orthogonal transformations preserve Frobenius norm.

    Property: ||X*Q||_F = ||X||_F for orthogonal Q
    Applied to: ||[[L_bar 0],[C D]]||_F = ||[[L A],[0 B]]||_F

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n, m, p = 3, 4, 2

    L = np.tril(np.random.randn(n, n)).astype(float, order='F')
    L += np.diag(np.abs(np.diag(L)) + 3.0)

    A = np.random.randn(n, m).astype(float, order='F')
    B = np.random.randn(p, m).astype(float, order='F')

    block_row1 = np.hstack([L, A])
    block_row2 = np.hstack([np.zeros((p, n)), B])
    original_norm = np.linalg.norm(np.vstack([block_row1, block_row2]), 'fro')

    from ctrlsys import mb04ld

    L_bar, A_out, D, C, tau = mb04ld('F', n, m, p, L.copy(), A.copy(), B.copy())

    result_row1 = np.hstack([L_bar, np.zeros((n, m))])
    result_row2 = np.hstack([C, D])
    result_norm = np.linalg.norm(np.vstack([result_row1, result_row2]), 'fro')

    np.testing.assert_allclose(original_norm, result_norm, rtol=1e-13, atol=1e-14,
                               err_msg="Frobenius norm must be preserved by orthogonal transformation")


def test_mb04ld_zero_m():
    """
    Edge case: M=0 (no columns in A, B).

    When M=0, transformation is trivial - L unchanged, C and D empty.
    """
    from ctrlsys import mb04ld

    n, m, p = 3, 0, 2

    L = np.tril(np.random.randn(n, n)).astype(float, order='F')
    L += np.diag(np.abs(np.diag(L)) + 1.0)

    A = np.zeros((n, 1), order='F', dtype=float)
    B = np.zeros((p, 1), order='F', dtype=float)

    L_bar, A_out, D, C, tau = mb04ld('F', n, m, p, L.copy(), A.copy(), B.copy())

    np.testing.assert_allclose(L_bar, L, rtol=1e-14, atol=1e-14,
                               err_msg="L should be unchanged when M=0")


def test_mb04ld_single_row():
    """
    Edge case: Single row (N=1).

    Simplest non-trivial case for LQ factorization.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n, m, p = 1, 3, 2

    L = np.array([[2.5]], order='F', dtype=float)
    A = np.array([[1.0, 2.0, 1.5]], order='F', dtype=float)
    B = np.array([[1.0, 2.0, 1.5], [2.0, 1.0, 2.5]], order='F', dtype=float)

    from ctrlsys import mb04ld

    L_bar, A_out, D, C, tau = mb04ld('F', n, m, p, L.copy(), A.copy(), B.copy())

    assert L_bar.shape == (1, 1)
    assert abs(L_bar[0, 0]) > 1e-10

    assert C.shape == (p, 1)
    assert D.shape == (p, m)
    assert len(tau) == 1


def test_mb04ld_householder_structure():
    """
    Verify Householder reflector structure.

    Each tau[i] corresponds to a Householder transformation.
    Property: |tau[i]| should be in range [0, 2] for normalized reflectors.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)

    n, m, p = 4, 5, 3

    L = np.tril(np.random.randn(n, n)).astype(float, order='F')
    L += np.diag(np.abs(np.diag(L)) + 2.0)

    A = np.random.randn(n, m).astype(float, order='F')
    B = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04ld

    L_bar, A_out, D, C, tau = mb04ld('F', n, m, p, L.copy(), A.copy(), B.copy())

    for i in range(n):
        assert 0.0 <= tau[i] <= 2.0, f"tau[{i}]={tau[i]} outside valid range [0, 2]"


def test_mb04ld_diagonal_l():
    """
    Special case: L is diagonal matrix.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    n, m, p = 3, 4, 2

    L = np.diag([2.0, 3.0, 1.5]).astype(float, order='F')
    A = np.random.randn(n, m).astype(float, order='F')
    B = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04ld

    L_bar, A_out, D, C, tau = mb04ld('F', n, m, p, L.copy(), A.copy(), B.copy())

    assert np.allclose(L_bar, np.tril(L_bar), rtol=1e-14)

    original_norm = np.linalg.norm(L)**2 + np.linalg.norm(A)**2 + np.linalg.norm(B)**2
    result_norm = np.linalg.norm(L_bar)**2 + np.linalg.norm(C)**2 + np.linalg.norm(D)**2
    np.testing.assert_allclose(original_norm, result_norm, rtol=1e-13, atol=1e-14)


def test_mb04ld_lower_trap_m_less_n():
    """
    Test lower trapezoidal with M < N.

    When M < N and UPLO='L', A is N-by-M with lower triangular structure.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    n, m, p = 5, 3, 2

    L = np.tril(np.random.randn(n, n)).astype(float, order='F')
    L += np.diag(np.abs(np.diag(L)) + 1.5)

    A = np.tril(np.random.randn(n, m)).astype(float, order='F')

    B = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04ld

    L_bar, A_out, D, C, tau = mb04ld('L', n, m, p, L.copy(), A.copy(), B.copy())

    assert np.allclose(L_bar, np.tril(L_bar), rtol=1e-14)
    assert L_bar.shape == (n, n)
    assert C.shape == (p, n)
    assert D.shape == (p, m)
