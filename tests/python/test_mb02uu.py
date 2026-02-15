"""
Tests for MB02UU: Solve linear system using LU with complete pivoting.

Solves A * x = scale * RHS using the LU factorization from MB02UV.
"""

import numpy as np
import pytest


def test_mb02uu_basic():
    """
    Validate basic linear system solution.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb02uv, mb02uu

    np.random.seed(42)
    n = 4

    a = np.array([
        [4.0, 2.0, 1.0, 3.0],
        [2.0, 5.0, 3.0, 1.0],
        [1.0, 3.0, 6.0, 2.0],
        [3.0, 1.0, 2.0, 7.0]
    ], order='F', dtype=float)
    a_orig = a.copy()

    rhs = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)
    rhs_orig = rhs.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info == 0

    x, scale = mb02uu(n, a_lu, rhs, ipiv, jpiv)

    assert scale > 0 and scale <= 1.0

    residual = a_orig @ x - scale * rhs_orig
    np.testing.assert_allclose(residual, np.zeros(n), rtol=1e-12, atol=1e-13)


def test_mb02uu_identity():
    """
    Validate solution with identity matrix: x = rhs.
    """
    from ctrlsys import mb02uv, mb02uu

    n = 3
    a = np.eye(n, order='F', dtype=float)
    rhs = np.array([1.0, 2.0, 3.0], dtype=float)
    rhs_orig = rhs.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info == 0

    x, scale = mb02uu(n, a_lu, rhs, ipiv, jpiv)

    assert scale == 1.0
    np.testing.assert_allclose(x, rhs_orig, rtol=1e-14)


def test_mb02uu_1x1():
    """
    Validate edge case: 1x1 system.
    """
    from ctrlsys import mb02uv, mb02uu

    n = 1
    a = np.array([[2.0]], order='F', dtype=float)
    rhs = np.array([4.0], dtype=float)

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info == 0

    x, scale = mb02uu(n, a_lu, rhs, ipiv, jpiv)

    assert scale == 1.0
    np.testing.assert_allclose(x[0], 2.0, rtol=1e-14)


def test_mb02uu_2x2():
    """
    Validate 2x2 system with known solution.
    """
    from ctrlsys import mb02uv, mb02uu

    n = 2
    a = np.array([
        [2.0, 1.0],
        [1.0, 3.0]
    ], order='F', dtype=float)
    a_orig = a.copy()

    x_true = np.array([1.0, 2.0], dtype=float)
    rhs = a_orig @ x_true
    rhs_orig = rhs.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info == 0

    x, scale = mb02uu(n, a_lu, rhs, ipiv, jpiv)

    np.testing.assert_allclose(x, scale * x_true, rtol=1e-14)


def test_mb02uu_property_solution_correctness():
    """
    Validate mathematical property: A*x = scale*rhs.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb02uv, mb02uu

    np.random.seed(123)
    n = 5

    a = np.random.randn(n, n).astype(float, order='F')
    a = a + 5.0 * np.eye(n, dtype=float)
    a_orig = a.copy()

    rhs = np.random.randn(n).astype(float)
    rhs_orig = rhs.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info == 0

    x, scale = mb02uu(n, a_lu, rhs, ipiv, jpiv)

    residual = a_orig @ x - scale * rhs_orig
    np.testing.assert_allclose(residual, np.zeros(n), rtol=1e-12, atol=1e-13)


def test_mb02uu_multiple_rhs():
    """
    Validate solving multiple right-hand sides sequentially.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb02uv, mb02uu

    np.random.seed(456)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    a = a + 5.0 * np.eye(n, dtype=float)
    a_orig = a.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info == 0

    for i in range(3):
        rhs = np.random.randn(n).astype(float)
        rhs_orig = rhs.copy()

        x, scale = mb02uu(n, a_lu, rhs, ipiv, jpiv)

        residual = a_orig @ x - scale * rhs_orig
        np.testing.assert_allclose(residual, np.zeros(n), rtol=1e-12, atol=1e-13)


def test_mb02uu_well_conditioned():
    """
    Validate solution for well-conditioned matrix.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb02uv, mb02uu

    np.random.seed(789)
    n = 6

    a = np.diag(np.arange(1, n + 1, dtype=float))
    a = a.astype(float, order='F')
    a_orig = a.copy()

    rhs = np.arange(1, n + 1, dtype=float)
    rhs_orig = rhs.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info == 0

    x, scale = mb02uu(n, a_lu, rhs, ipiv, jpiv)

    x_expected = rhs_orig / np.diag(a_orig)

    np.testing.assert_allclose(x, scale * x_expected, rtol=1e-14)


def test_mb02uu_scaling():
    """
    Validate that scaling is applied correctly to prevent overflow.

    Test with a matrix that could cause numerical issues.
    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import mb02uv, mb02uu

    np.random.seed(888)
    n = 4

    a = np.array([
        [1e10, 1.0, 1.0, 1.0],
        [1.0, 1e-10, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 2.0]
    ], order='F', dtype=float)
    a_orig = a.copy()

    rhs = np.array([1e10, 1.0, 4.0, 5.0], dtype=float)
    rhs_orig = rhs.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info >= 0

    x, scale = mb02uu(n, a_lu, rhs, ipiv, jpiv)

    assert 0 < scale <= 1.0


def test_mb02uu_random_larger():
    """
    Validate solution for larger random system.

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb02uv, mb02uu

    np.random.seed(999)
    n = 10

    a = np.random.randn(n, n).astype(float, order='F')
    a = a + 10.0 * np.eye(n, dtype=float)
    a_orig = a.copy()

    rhs = np.random.randn(n).astype(float)
    rhs_orig = rhs.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info == 0

    x, scale = mb02uu(n, a_lu, rhs, ipiv, jpiv)

    residual = a_orig @ x - scale * rhs_orig
    np.testing.assert_allclose(residual, np.zeros(n), rtol=1e-11, atol=1e-12)
