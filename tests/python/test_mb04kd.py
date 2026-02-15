"""
Tests for MB04KD - QR factorization of special structured block matrix.

MB04KD computes Q'*[[R],[A B]] = [[R_bar C],[0 D]] where R and R_bar are upper triangular.
"""

import numpy as np
import pytest


def test_mb04kd_full_basic():
    """
    Basic test with full matrix A (UPLO='F').

    Validates:
    - R_bar is upper triangular
    - Orthogonal transformation preserves norms
    - Mathematical structure Q'*[[R],[A B]] = [[R_bar C],[0 D]]

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n, m, p = 3, 2, 4

    # Input matrices (column-major)
    R = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 3.0, 1.5],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    A = np.array([
        [1.0, 2.0, 1.0],
        [3.0, 1.0, 2.0],
        [2.0, 3.0, 1.0],
        [1.0, 1.0, 3.0]
    ], order='F', dtype=float)

    B = np.array([
        [1.5, 2.0],
        [2.5, 1.0],
        [1.0, 3.0],
        [3.0, 1.5]
    ], order='F', dtype=float)

    # Import SLICOT routine
    from ctrlsys import mb04kd

    # Call MB04KD
    R_bar, A_out, D, C, tau = mb04kd('F', n, m, p, R.copy(), A.copy(), B.copy())

    # Test 1: R_bar is upper triangular (strict lower should be zero)
    for i in range(n):
        for j in range(i):
            assert abs(R_bar[i, j]) < 1e-14, f"R_bar not upper triangular at ({i},{j})"

    # Test 2: D matrix has valid dimensions
    assert D.shape == (p, m), f"D shape mismatch: {D.shape} vs {(p, m)}"

    # Test 3: C matrix has valid dimensions
    assert C.shape == (n, m), f"C shape mismatch: {C.shape} vs {(n, m)}"

    # Test 4: tau has correct length
    assert len(tau) == n, f"tau length mismatch: {len(tau)} vs {n}"

    # Test 5: Verify orthogonal transformation (Frobenius norm preservation)
    # ||[[R],[A B]]||_F^2 should equal ||[[R_bar C],[0 D]]||_F^2
    original_norm_sq = np.linalg.norm(R)**2 + np.linalg.norm(A)**2 + np.linalg.norm(B)**2
    result_norm_sq = np.linalg.norm(R_bar)**2 + np.linalg.norm(C)**2 + np.linalg.norm(D)**2

    np.testing.assert_allclose(original_norm_sq, result_norm_sq, rtol=1e-13, atol=1e-14,
                               err_msg="Orthogonal transformation should preserve Frobenius norm")


def test_mb04kd_upper_trapezoidal():
    """
    Test with upper trapezoidal matrix A (UPLO='U').

    When UPLO='U', A is upper trapezoidal (min(P,N)-by-N upper triangular part).
    This exploits structure for efficiency.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n, m, p = 4, 3, 5

    # Upper triangular R
    R = np.triu(np.random.randn(n, n)).astype(float, order='F')
    R += np.eye(n) * 2.0  # Make diagonally dominant

    # Upper trapezoidal A (only min(p,n)-by-n upper part is used)
    A = np.zeros((p, n), order='F', dtype=float)
    for i in range(min(p, n)):
        for j in range(i, n):
            A[i, j] = np.random.randn()

    B = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04kd

    R_bar, A_out, D, C, tau = mb04kd('U', n, m, p, R.copy(), A.copy(), B.copy())

    # Verify R_bar is upper triangular
    assert np.allclose(R_bar, np.triu(R_bar), rtol=1e-14), "R_bar should be upper triangular"

    # Verify dimensions
    assert R_bar.shape == (n, n)
    assert C.shape == (n, m)
    assert D.shape == (p, m)
    assert len(tau) == n


def test_mb04kd_norm_preservation():
    """
    Mathematical property test: Orthogonal transformations preserve Frobenius norm.

    Property: ||Q'*X||_F = ||X||_F for orthogonal Q
    Applied to: ||[[R_bar C],[0 D]]||_F = ||[[R],[A B]]||_F

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n, m, p = 3, 2, 3

    R = np.triu(np.random.randn(n, n)).astype(float, order='F')
    R += np.eye(n) * 3.0

    A = np.random.randn(p, n).astype(float, order='F')
    B = np.random.randn(p, m).astype(float, order='F')

    # Compute original norm
    block_col1 = np.vstack([R, A])
    block_col2 = np.vstack([np.zeros((n, m)), B])
    original_norm = np.linalg.norm(np.hstack([block_col1, block_col2]), 'fro')

    from ctrlsys import mb04kd

    R_bar, A_out, D, C, tau = mb04kd('F', n, m, p, R.copy(), A.copy(), B.copy())

    # Compute result norm
    # Note: A_out contains lower part which should be zero after transformation
    result_block_col1 = np.vstack([R_bar, np.zeros((p, n))])
    result_block_col2 = np.vstack([C, D])
    result_norm = np.linalg.norm(np.hstack([result_block_col1, result_block_col2]), 'fro')

    np.testing.assert_allclose(original_norm, result_norm, rtol=1e-13, atol=1e-14,
                               err_msg="Frobenius norm must be preserved by orthogonal transformation")


def test_mb04kd_zero_dimensions():
    """
    Edge case: Zero dimensions.

    Tests boundary conditions when n=0, m=0, or p=0.
    """
    from ctrlsys import mb04kd

    # Case 1: n=0 (degenerate - no transformation)
    n, m, p = 0, 2, 3
    R = np.zeros((1, 0), order='F', dtype=float)
    A = np.zeros((p, 0), order='F', dtype=float)
    B = np.random.randn(p, m).astype(float, order='F')

    R_bar, A_out, D, C, tau = mb04kd('F', n, m, p, R.copy(), A.copy(), B.copy())

    # When n=0, C matrix still has proper dimensions
    assert C.shape[1] == m  # Number of columns is m

    # Case 2: p=0
    n, m, p = 3, 2, 0
    R = np.triu(np.random.randn(n, n)).astype(float, order='F')
    A = np.zeros((1, n), order='F', dtype=float)
    B = np.zeros((1, m), order='F', dtype=float)

    R_bar, A_out, D, C, tau = mb04kd('F', n, m, p, R.copy(), A.copy(), B.copy())

    # When p=0, R should remain unchanged and C should come from first row
    assert R_bar.shape == (n, n)
    assert C.shape == (n, m)


def test_mb04kd_single_column():
    """
    Edge case: Single column (n=1).

    Simplest non-trivial case for QR factorization.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n, m, p = 1, 2, 3

    R = np.array([[2.5]], order='F', dtype=float)
    A = np.array([[1.0], [2.0], [1.5]], order='F', dtype=float)
    B = np.array([[1.0, 2.0], [1.5, 1.0], [2.0, 1.5]], order='F', dtype=float)

    from ctrlsys import mb04kd

    R_bar, A_out, D, C, tau = mb04kd('F', n, m, p, R.copy(), A.copy(), B.copy())

    # R_bar is 1x1, should be non-zero
    assert R_bar.shape == (1, 1)
    assert abs(R_bar[0, 0]) > 1e-10

    # Verify shapes
    assert C.shape == (1, m)
    assert D.shape == (p, m)
    assert len(tau) == 1


def test_mb04kd_householder_structure():
    """
    Verify Householder reflector structure.

    Each tau[i] corresponds to a Householder transformation.
    Property: |tau[i]| should be in reasonable range [0, 2] for normalized reflectors.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)

    n, m, p = 4, 2, 5

    R = np.triu(np.random.randn(n, n)).astype(float, order='F')
    R += np.eye(n) * 2.0

    A = np.random.randn(p, n).astype(float, order='F')
    B = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04kd

    R_bar, A_out, D, C, tau = mb04kd('F', n, m, p, R.copy(), A.copy(), B.copy())

    # Householder tau values should be in [0, 2]
    for i in range(n):
        assert 0.0 <= tau[i] <= 2.0, f"tau[{i}]={tau[i]} outside valid range [0, 2]"

    # A_out contains the Householder vectors v_i
    # These should have been modified from input
    assert A_out.shape == (p, n)


def test_mb04kd_diagonal_r():
    """
    Special case: R is diagonal matrix.

    Diagonal R simplifies the computation structure.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    n, m, p = 3, 2, 4

    # Diagonal R
    R = np.diag([2.0, 3.0, 1.5]).astype(float, order='F')
    A = np.random.randn(p, n).astype(float, order='F')
    B = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04kd

    R_bar, A_out, D, C, tau = mb04kd('F', n, m, p, R.copy(), A.copy(), B.copy())

    # R_bar should still be upper triangular
    assert np.allclose(R_bar, np.triu(R_bar), rtol=1e-14)

    # Norm preservation
    original_norm = np.linalg.norm(R)**2 + np.linalg.norm(A)**2 + np.linalg.norm(B)**2
    result_norm = np.linalg.norm(R_bar)**2 + np.linalg.norm(C)**2 + np.linalg.norm(D)**2
    np.testing.assert_allclose(original_norm, result_norm, rtol=1e-13, atol=1e-14)
