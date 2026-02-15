"""
Tests for MB02TD: Estimate reciprocal condition number of upper Hessenberg matrix.

Uses numpy only.
"""

import numpy as np


def test_mb02td_basic():
    """
    Test MB02TD with the example from MB02SD HTML documentation.

    Input: 5x5 upper Hessenberg matrix (after LU factorization by MB02SD)
    NORM = 'O' (1-norm)
    Expected RCOND = 0.1554D-01 (approximately 0.01554)
    """
    from ctrlsys import mb02sd, mb02td

    n = 5
    h = np.array([
        [1.0, 2.0, 6.0, 3.0, 5.0],
        [-2.0, -1.0, -1.0, 0.0, -2.0],
        [0.0, 3.0, 1.0, 5.0, 1.0],
        [0.0, 0.0, 2.0, 0.0, -4.0],
        [0.0, 0.0, 0.0, 1.0, 4.0]
    ], order='F', dtype=float)

    hnorm_1 = np.max(np.sum(np.abs(h), axis=0))

    h_lu, ipiv, info_sd = mb02sd(n, h)
    assert info_sd == 0

    rcond, info = mb02td('O', n, hnorm_1, h_lu, ipiv)

    assert info == 0
    np.testing.assert_allclose(rcond, 0.01554, rtol=0.1)


def test_mb02td_infinity_norm():
    """
    Test MB02TD with infinity norm.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb02sd, mb02td

    np.random.seed(42)
    n = 4

    h = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn()

    hnorm_inf = np.max(np.sum(np.abs(h), axis=1))

    h_lu, ipiv, info_sd = mb02sd(n, h)
    assert info_sd == 0

    rcond, info = mb02td('I', n, hnorm_inf, h_lu, ipiv)

    assert info == 0
    assert 0.0 < rcond <= 1.0


def test_mb02td_well_conditioned():
    """
    Test MB02TD with well-conditioned matrix (RCOND close to 1).

    Uses identity-like Hessenberg matrix.
    """
    from ctrlsys import mb02sd, mb02td

    n = 3
    h = np.array([
        [1.0, 0.0, 0.0],
        [0.01, 1.0, 0.0],
        [0.0, 0.01, 1.0]
    ], order='F', dtype=float)

    hnorm_1 = np.max(np.sum(np.abs(h), axis=0))

    h_lu, ipiv, info_sd = mb02sd(n, h)
    assert info_sd == 0

    rcond, info = mb02td('1', n, hnorm_1, h_lu, ipiv)

    assert info == 0
    assert rcond > 0.9


def test_mb02td_ill_conditioned():
    """
    Test MB02TD with ill-conditioned matrix (small RCOND).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb02sd, mb02td

    n = 4
    h = np.array([
        [1e10, 1.0, 0.0, 0.0],
        [1.0, 1e-10, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0]
    ], order='F', dtype=float)

    hnorm_1 = np.max(np.sum(np.abs(h), axis=0))

    h_lu, ipiv, info_sd = mb02sd(n, h)
    assert info_sd == 0

    rcond, info = mb02td('O', n, hnorm_1, h_lu, ipiv)

    assert info == 0
    assert rcond < 1e-8


def test_mb02td_n1():
    """Test MB02TD with 1x1 matrix."""
    from ctrlsys import mb02sd, mb02td

    n = 1
    h = np.array([[5.0]], order='F', dtype=float)
    hnorm = 5.0

    h_lu, ipiv, info_sd = mb02sd(n, h)
    assert info_sd == 0

    rcond, info = mb02td('O', n, hnorm, h_lu, ipiv)

    assert info == 0
    np.testing.assert_allclose(rcond, 1.0, rtol=1e-14)


def test_mb02td_n0():
    """Test MB02TD with n=0 (quick return, RCOND=1)."""
    from ctrlsys import mb02td

    n = 0
    h = np.zeros((1, 1), order='F', dtype=float)
    ipiv = np.zeros(1, dtype=np.int32)

    rcond, info = mb02td('O', n, 0.0, h, ipiv)

    assert info == 0
    np.testing.assert_allclose(rcond, 1.0, rtol=1e-14)


def test_mb02td_hnorm_zero():
    """Test MB02TD with HNORM=0 (quick return, RCOND=0)."""
    from ctrlsys import mb02sd, mb02td

    n = 2
    h = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    h_lu, ipiv, info_sd = mb02sd(n, h)
    assert info_sd == 0

    rcond, info = mb02td('O', n, 0.0, h_lu, ipiv)

    assert info == 0
    np.testing.assert_allclose(rcond, 0.0, atol=1e-15)


def test_mb02td_property_bounds():
    """
    Test mathematical property: 0 <= RCOND <= 1.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb02sd, mb02td

    np.random.seed(456)

    for _ in range(5):
        n = np.random.randint(2, 8)

        h = np.zeros((n, n), order='F', dtype=float)
        for i in range(n):
            for j in range(n):
                if j >= i - 1:
                    h[i, j] = np.random.randn()

        hnorm = np.max(np.sum(np.abs(h), axis=0))

        h_lu, ipiv, info_sd = mb02sd(n, h)
        if info_sd > 0:
            continue

        rcond, info = mb02td('O', n, hnorm, h_lu, ipiv)

        assert info == 0
        assert 0.0 <= rcond <= 1.0


def test_mb02td_invalid_norm():
    """Test MB02TD with invalid NORM parameter."""
    from ctrlsys import mb02td

    n = 2
    h = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    ipiv = np.array([1, 2], dtype=np.int32)

    rcond, info = mb02td('X', n, 1.0, h, ipiv)

    assert info == -1


def test_mb02td_invalid_n():
    """Test MB02TD with invalid N parameter."""
    from ctrlsys import mb02td

    h = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    ipiv = np.array([1, 2], dtype=np.int32)

    rcond, info = mb02td('O', -1, 1.0, h, ipiv)

    assert info == -2


def test_mb02td_invalid_hnorm():
    """Test MB02TD with invalid HNORM parameter (negative)."""
    from ctrlsys import mb02td

    n = 2
    h = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    ipiv = np.array([1, 2], dtype=np.int32)

    rcond, info = mb02td('O', n, -1.0, h, ipiv)

    assert info == -3
