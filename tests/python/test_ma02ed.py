"""
Tests for MA02ED: Store by symmetry the upper or lower triangle of a symmetric matrix

Random seeds used for reproducibility:
- test_ma02ed_upper: 42
- test_ma02ed_lower: 123
- test_ma02ed_symmetry_property: 456
- test_ma02ed_no_op: 789
"""

import numpy as np
import pytest


def test_ma02ed_upper_basic():
    """
    Test MA02ED with UPLO='U' - construct lower triangle from upper.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    # Create symmetric matrix
    A_full = np.random.randn(n, n)
    A_full = (A_full + A_full.T) / 2  # Make symmetric

    # Extract upper triangle only
    A = np.triu(A_full).astype(float, order='F')

    # Expected: full symmetric matrix
    A_expected = A_full.copy(order='F')

    # Import and call
    from ctrlsys import ma02ed
    ma02ed('U', A)

    # Validate symmetry achieved
    np.testing.assert_allclose(A, A_expected, rtol=1e-14, atol=1e-15)

    # Validate symmetry property
    np.testing.assert_allclose(A, A.T, rtol=1e-14, atol=1e-15)


def test_ma02ed_lower_basic():
    """
    Test MA02ED with UPLO='L' - construct upper triangle from lower.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 5

    # Create symmetric matrix
    A_full = np.random.randn(n, n)
    A_full = (A_full + A_full.T) / 2  # Make symmetric

    # Extract lower triangle only
    A = np.tril(A_full).astype(float, order='F')

    # Expected: full symmetric matrix
    A_expected = A_full.copy(order='F')

    # Import and call
    from ctrlsys import ma02ed
    ma02ed('L', A)

    # Validate symmetry achieved
    np.testing.assert_allclose(A, A_expected, rtol=1e-14, atol=1e-15)

    # Validate symmetry property
    np.testing.assert_allclose(A, A.T, rtol=1e-14, atol=1e-15)


def test_ma02ed_symmetry_property():
    """
    Mathematical property test: A = A^T after completion.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 6

    # Test upper triangle input
    A_upper = np.random.randn(n, n).astype(float, order='F')
    A_upper = np.triu(A_upper)

    from ctrlsys import ma02ed
    ma02ed('U', A_upper)

    # Validate perfect symmetry
    np.testing.assert_allclose(A_upper, A_upper.T, rtol=1e-14, atol=1e-15)

    # Test lower triangle input
    A_lower = np.random.randn(n, n).astype(float, order='F')
    A_lower = np.tril(A_lower)

    ma02ed('L', A_lower)

    # Validate perfect symmetry
    np.testing.assert_allclose(A_lower, A_lower.T, rtol=1e-14, atol=1e-15)


def test_ma02ed_no_op():
    """
    Test MA02ED with invalid UPLO - should be no-op.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3

    A = np.random.randn(n, n).astype(float, order='F')
    A_original = A.copy()

    from ctrlsys import ma02ed
    ma02ed('X', A)  # Invalid UPLO

    # Matrix should be unchanged
    np.testing.assert_allclose(A, A_original, rtol=1e-14, atol=1e-15)


def test_ma02ed_n_zero():
    """
    Test MA02ED with N=0 (edge case).
    """
    A = np.array([], dtype=float, order='F').reshape(0, 0)

    from ctrlsys import ma02ed
    ma02ed('U', A)

    # Should handle gracefully
    assert A.shape == (0, 0)


def test_ma02ed_n_one():
    """
    Test MA02ED with N=1 (single element).
    """
    A = np.array([[5.0]], dtype=float, order='F')

    from ctrlsys import ma02ed
    ma02ed('U', A)

    # Single element is already symmetric
    assert A[0, 0] == 5.0


def test_ma02ed_diagonal_preservation():
    """
    Test that diagonal elements are preserved in both UPLO modes.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 4

    # Create diagonal matrix with random diagonal
    diag_vals = np.random.randn(n)
    A_upper = np.diag(diag_vals).astype(float, order='F')
    A_lower = np.diag(diag_vals).astype(float, order='F')

    from ctrlsys import ma02ed
    ma02ed('U', A_upper)
    ma02ed('L', A_lower)

    # Diagonal should be unchanged
    np.testing.assert_allclose(np.diag(A_upper), diag_vals, rtol=1e-14)
    np.testing.assert_allclose(np.diag(A_lower), diag_vals, rtol=1e-14)

    # Both should be diagonal matrices
    np.testing.assert_allclose(A_upper, np.diag(diag_vals), rtol=1e-14)
    np.testing.assert_allclose(A_lower, np.diag(diag_vals), rtol=1e-14)
