"""
Tests for MB02XD - Solve A'*A*X = B using Cholesky factorization.

MB02XD computes the solution to a system of linear equations A'*A*X = B,
where A'*A is a symmetric positive definite matrix, using Cholesky
factorization.
"""

import numpy as np
import pytest
import ctrlsys


def test_mb02xd_basic_full_upper():
    """
    Test MB02XD with full storage, upper triangle.

    Solve A'*A*X = B where A is a 4x3 matrix (M=4, N=3).

    Uses STOR='F' (full), UPLO='U' (upper).
    """
    m, n, nrhs = 4, 3, 2

    np.random.seed(42)
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
        [1.0, 1.0, 2.0]
    ], order='F', dtype=np.float64)

    B = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ], order='F', dtype=np.float64)

    AtA = A.T @ A
    X_expected = np.linalg.solve(AtA, B)

    x, ata, info = slicot.mb02xd('F', 'U', m, n, nrhs, A, B)

    assert info == 0, f"MB02XD failed with info={info}"
    np.testing.assert_allclose(x, X_expected, rtol=1e-13, atol=1e-14)

    L_expected = np.linalg.cholesky(AtA).T
    np.testing.assert_allclose(np.triu(ata), L_expected, rtol=1e-13, atol=1e-14)


def test_mb02xd_full_lower():
    """
    Test MB02XD with full storage, lower triangle.

    Random seed: 123 (for reproducibility)
    """
    m, n, nrhs = 5, 3, 1

    np.random.seed(123)
    A = np.random.randn(m, n).astype(np.float64, order='F')
    B = np.random.randn(n, nrhs).astype(np.float64, order='F')

    AtA = A.T @ A
    X_expected = np.linalg.solve(AtA, B)

    x, ata, info = slicot.mb02xd('F', 'L', m, n, nrhs, A, B)

    assert info == 0, f"MB02XD failed with info={info}"
    np.testing.assert_allclose(x, X_expected, rtol=1e-13, atol=1e-14)


def test_mb02xd_packed_upper():
    """
    Test MB02XD with packed storage, upper triangle.

    For packed storage with UPLO='U', the upper triangle is stored
    column by column: A(1,1), A(1,2), A(2,2), A(1,3), A(2,3), A(3,3), ...

    Random seed: 456 (for reproducibility)
    """
    m, n, nrhs = 4, 3, 2

    np.random.seed(456)
    A = np.random.randn(m, n).astype(np.float64, order='F')
    B = np.random.randn(n, nrhs).astype(np.float64, order='F')

    AtA = A.T @ A
    X_expected = np.linalg.solve(AtA, B)

    x, ata, info = slicot.mb02xd('P', 'U', m, n, nrhs, A, B)

    assert info == 0, f"MB02XD failed with info={info}"
    np.testing.assert_allclose(x, X_expected, rtol=1e-13, atol=1e-14)


def test_mb02xd_packed_lower():
    """
    Test MB02XD with packed storage, lower triangle.

    For packed storage with UPLO='L', the lower triangle is stored
    column by column: A(1,1), A(2,1), A(3,1), ..., A(2,2), A(3,2), ...

    Random seed: 789 (for reproducibility)
    """
    m, n, nrhs = 6, 4, 3

    np.random.seed(789)
    A = np.random.randn(m, n).astype(np.float64, order='F')
    B = np.random.randn(n, nrhs).astype(np.float64, order='F')

    AtA = A.T @ A
    X_expected = np.linalg.solve(AtA, B)

    x, ata, info = slicot.mb02xd('P', 'L', m, n, nrhs, A, B)

    assert info == 0, f"MB02XD failed with info={info}"
    np.testing.assert_allclose(x, X_expected, rtol=1e-13, atol=1e-14)


def test_mb02xd_singular_matrix():
    """
    Test MB02XD with singular A'*A matrix (rank deficient A).

    Should return info > 0 indicating singularity.
    """
    m, n, nrhs = 4, 3, 1

    A = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0]
    ], order='F', dtype=np.float64)

    B = np.array([[1.0], [2.0], [3.0]], order='F', dtype=np.float64)

    x, ata, info = slicot.mb02xd('F', 'U', m, n, nrhs, A, B)

    assert info > 0, f"Expected info > 0 for singular matrix, got {info}"


def test_mb02xd_parameter_errors():
    """
    Test MB02XD parameter validation.
    """
    m, n, nrhs = 4, 3, 2
    A = np.ones((m, n), order='F', dtype=np.float64)
    B = np.ones((n, nrhs), order='F', dtype=np.float64)

    x, ata, info = slicot.mb02xd('X', 'U', m, n, nrhs, A, B)
    assert info == -2, f"Expected info=-2 for invalid STOR, got {info}"

    A = np.ones((m, n), order='F', dtype=np.float64)
    B = np.ones((n, nrhs), order='F', dtype=np.float64)
    x, ata, info = slicot.mb02xd('F', 'U', -1, n, nrhs, A, B)
    assert info == -5, f"Expected info=-5 for M<0, got {info}"


def test_mb02xd_residual_property():
    """
    Mathematical property test: verify residual A'*A*X - B is small.

    This validates that the solution X truly satisfies the normal equations.

    Random seed: 999 (for reproducibility)
    """
    m, n, nrhs = 10, 5, 3

    np.random.seed(999)
    A = np.random.randn(m, n).astype(np.float64, order='F')
    B_orig = np.random.randn(n, nrhs).astype(np.float64, order='F')
    B = B_orig.copy(order='F')

    AtA = A.T @ A

    x, ata, info = slicot.mb02xd('F', 'U', m, n, nrhs, A, B)

    assert info == 0, f"MB02XD failed with info={info}"

    residual = AtA @ x - B_orig

    residual_norm = np.linalg.norm(residual, 'fro')
    B_norm = np.linalg.norm(B_orig, 'fro')

    relative_residual = residual_norm / B_norm
    assert relative_residual < 1e-13, f"Relative residual {relative_residual} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
