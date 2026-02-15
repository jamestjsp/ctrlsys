"""
Tests for MA02ES: Store by skew-symmetry the upper or lower triangle of a skew-symmetric matrix

Random seeds used for reproducibility:
- test_ma02es_upper_basic: 42
- test_ma02es_lower_basic: 123
- test_ma02es_skew_symmetry_property: 456
- test_ma02es_no_op: 789
"""

import numpy as np
import pytest


def test_ma02es_upper_basic():
    """
    Test MA02ES with UPLO='U' - construct lower triangle from upper.
    Diagonal is set to zero. Lower = -Upper^T.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    A = np.random.randn(n, n).astype(float, order='F')
    A_upper = np.triu(A, k=1)

    A_expected = A_upper - A_upper.T

    from ctrlsys import ma02es
    ma02es('U', A_upper)

    np.testing.assert_allclose(A_upper, A_expected, rtol=1e-14, atol=1e-15)

    for i in range(n):
        assert A_upper[i, i] == 0.0

    np.testing.assert_allclose(A_upper, -A_upper.T, rtol=1e-14, atol=1e-15)


def test_ma02es_lower_basic():
    """
    Test MA02ES with UPLO='L' - construct upper triangle from lower.
    Diagonal is set to zero. Upper = -Lower^T.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 5

    A = np.random.randn(n, n).astype(float, order='F')
    A_lower = np.tril(A, k=-1)

    A_expected = A_lower - A_lower.T

    from ctrlsys import ma02es
    ma02es('L', A_lower)

    np.testing.assert_allclose(A_lower, A_expected, rtol=1e-14, atol=1e-15)

    for i in range(n):
        assert A_lower[i, i] == 0.0

    np.testing.assert_allclose(A_lower, -A_lower.T, rtol=1e-14, atol=1e-15)


def test_ma02es_skew_symmetry_property():
    """
    Mathematical property test: A = -A^T after completion.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 6

    A_upper = np.random.randn(n, n).astype(float, order='F')
    A_upper = np.triu(A_upper, k=1)

    from ctrlsys import ma02es
    ma02es('U', A_upper)

    np.testing.assert_allclose(A_upper, -A_upper.T, rtol=1e-14, atol=1e-15)

    A_lower = np.random.randn(n, n).astype(float, order='F')
    A_lower = np.tril(A_lower, k=-1)

    ma02es('L', A_lower)

    np.testing.assert_allclose(A_lower, -A_lower.T, rtol=1e-14, atol=1e-15)


def test_ma02es_no_op():
    """
    Test MA02ES with invalid UPLO - should be no-op.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3

    A = np.random.randn(n, n).astype(float, order='F')
    A_original = A.copy()

    from ctrlsys import ma02es
    ma02es('X', A)

    np.testing.assert_allclose(A, A_original, rtol=1e-14, atol=1e-15)


def test_ma02es_n_zero():
    """
    Test MA02ES with N=0 (edge case).
    """
    A = np.array([], dtype=float, order='F').reshape(0, 0)

    from ctrlsys import ma02es
    ma02es('U', A)

    assert A.shape == (0, 0)


def test_ma02es_n_one():
    """
    Test MA02ES with N=1 (single element).
    Diagonal must be set to zero for skew-symmetric.
    """
    A = np.array([[5.0]], dtype=float, order='F')

    from ctrlsys import ma02es
    ma02es('U', A)

    assert A[0, 0] == 0.0


def test_ma02es_diagonal_zeroed():
    """
    Test that diagonal elements are set to zero in both UPLO modes.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 4

    diag_vals = np.random.randn(n)
    A_upper = np.diag(diag_vals).astype(float, order='F')
    A_lower = np.diag(diag_vals).astype(float, order='F')

    from ctrlsys import ma02es
    ma02es('U', A_upper)
    ma02es('L', A_lower)

    np.testing.assert_allclose(np.diag(A_upper), np.zeros(n), rtol=1e-14)
    np.testing.assert_allclose(np.diag(A_lower), np.zeros(n), rtol=1e-14)

    np.testing.assert_allclose(A_upper, np.zeros((n, n)), rtol=1e-14)
    np.testing.assert_allclose(A_lower, np.zeros((n, n)), rtol=1e-14)


def test_ma02es_specific_values():
    """
    Test MA02ES with specific known values.

    For UPLO='U', given upper triangle:
    [0  2  3]
    [0  0  5]
    [0  0  0]

    Expected skew-symmetric result:
    [ 0  2  3]
    [-2  0  5]
    [-3 -5  0]
    """
    A = np.array([
        [0.0, 2.0, 3.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0]
    ], dtype=float, order='F')

    A_expected = np.array([
        [ 0.0,  2.0,  3.0],
        [-2.0,  0.0,  5.0],
        [-3.0, -5.0,  0.0]
    ], dtype=float, order='F')

    from ctrlsys import ma02es
    ma02es('U', A)

    np.testing.assert_allclose(A, A_expected, rtol=1e-14, atol=1e-15)


def test_ma02es_specific_values_lower():
    """
    Test MA02ES with specific known values (lower triangle input).

    For UPLO='L', given lower triangle:
    [ 0  0  0]
    [-2  0  0]
    [-3 -5  0]

    Expected skew-symmetric result:
    [ 0  2  3]
    [-2  0  5]
    [-3 -5  0]
    """
    A = np.array([
        [ 0.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-3.0,-5.0, 0.0]
    ], dtype=float, order='F')

    A_expected = np.array([
        [ 0.0,  2.0,  3.0],
        [-2.0,  0.0,  5.0],
        [-3.0, -5.0,  0.0]
    ], dtype=float, order='F')

    from ctrlsys import ma02es
    ma02es('L', A)

    np.testing.assert_allclose(A, A_expected, rtol=1e-14, atol=1e-15)
