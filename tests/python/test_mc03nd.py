"""
Tests for MC03ND: Minimal polynomial basis for right nullspace of polynomial matrix.

Computes K(s) such that P(s) * K(s) = 0.
"""

import numpy as np
import pytest


def test_mc03nd_html_example():
    """
    Test MC03ND with HTML documentation example.

    Input: 5x4 polynomial matrix P(s) of degree 2.
    Expected output: DK=1, kernel with 2 columns.
    """
    from ctrlsys import mc03nd

    mp, np_dim, dp = 5, 4, 2

    # P(0) - constant coefficient (5x4)
    # Data read row-by-row from HTML
    p0 = np.array([
        [2.0, 2.0, 0.0, 3.0],
        [0.0, 4.0, 0.0, 6.0],
        [8.0, 8.0, 0.0, 12.0],
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0, 3.0]
    ], order='F', dtype=float)

    # P(1) - linear coefficient (5x4)
    p1 = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [4.0, 0.0, 4.0, 0.0],
        [2.0, 2.0, 0.0, 3.0],
        [3.0, 2.0, 1.0, 3.0]
    ], order='F', dtype=float)

    # P(2) - quadratic coefficient (5x4)
    p2 = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    # Stack into 3D array (mp x np_dim x (dp+1))
    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    p[:, :, 0] = p0
    p[:, :, 1] = p1
    p[:, :, 2] = p2

    tol = 0.0

    dk, gam, nullsp, ker, info = mc03nd(mp, np_dim, dp, p, tol)

    assert info == 0
    assert dk == 1  # Degree of kernel is 1

    # From HTML: nk = SUM(GAM(i)) for i=1..DK+1
    # GAM should have DK+1=2 elements
    nk = sum(gam[:dk + 1])
    assert nk == 2  # Two kernel columns

    # Validate NULLSP (right nullspace vectors in condensed form)
    # Expected from HTML (4 rows, M1 columns where M1 = sum(i*GAM(i)))
    nullsp_expected = np.array([
        [0.0000, 0.0000, 0.0000],
        [-0.8321, 0.0000, 0.1538],
        [0.0000, -1.0000, 0.0000],
        [0.5547, 0.0000, 0.2308]
    ], order='F', dtype=float)

    m1 = sum((i + 1) * gam[i] for i in range(dk + 1))
    assert m1 == 3  # 1*GAM(0) + 2*GAM(1) = 3
    np.testing.assert_allclose(nullsp[:np_dim, :m1], nullsp_expected, rtol=1e-3, atol=1e-3)

    # Validate KER by checking P(s)*K(s) = 0 (kernel basis is not unique)
    for s in [0.0, 0.5, 1.0, -1.0, 2.0]:
        ps = p0 + p1*s + p2*s**2
        ks = np.zeros((np_dim, nk), order='F', dtype=float)
        for k in range(dk + 1):
            ks += ker[:np_dim, :nk, k] * (s**k)
        product = ps @ ks
        np.testing.assert_allclose(product, np.zeros_like(product), atol=1e-12)


def test_mc03nd_no_nullspace():
    """
    Test MC03ND when polynomial matrix has no right nullspace.

    A square, full rank polynomial matrix should have DK=-1.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc03nd

    np.random.seed(42)

    mp, np_dim, dp = 3, 3, 1

    # Create a simple diagonal polynomial matrix (full rank)
    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    # P(0) = identity
    p[:, :, 0] = np.eye(mp, dtype=float)
    # P(1) = 2*identity
    p[:, :, 1] = 2.0 * np.eye(mp, dtype=float)

    tol = 0.0

    dk, gam, nullsp, ker, info = mc03nd(mp, np_dim, dp, p, tol)

    assert info == 0
    assert dk == -1  # No right nullspace


def test_mc03nd_nullspace_verification():
    """
    Verify that P(s)*K(s) = 0 holds for the computed kernel.

    Tests the polynomial matrix equation residual at specific s values.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mc03nd

    mp, np_dim, dp = 5, 4, 2

    # Use same data as HTML example
    p0 = np.array([
        [2.0, 2.0, 0.0, 3.0],
        [0.0, 4.0, 0.0, 6.0],
        [8.0, 8.0, 0.0, 12.0],
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0, 3.0]
    ], order='F', dtype=float)

    p1 = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [4.0, 0.0, 4.0, 0.0],
        [2.0, 2.0, 0.0, 3.0],
        [3.0, 2.0, 1.0, 3.0]
    ], order='F', dtype=float)

    p2 = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    p[:, :, 0] = p0
    p[:, :, 1] = p1
    p[:, :, 2] = p2

    tol = 0.0

    dk, gam, nullsp, ker, info = mc03nd(mp, np_dim, dp, p, tol)

    assert info == 0
    assert dk >= 0

    nk = sum(gam[:dk + 1])

    # Evaluate P(s)*K(s) at several s values
    test_s_values = [0.0, 0.5, 1.0, -1.0, 2.0]

    for s in test_s_values:
        # Compute P(s) = P(0) + P(1)*s + P(2)*s^2
        ps = np.zeros((mp, np_dim), order='F', dtype=float)
        for k in range(dp + 1):
            ps += p[:, :, k] * (s ** k)

        # Compute K(s) = K(0) + K(1)*s + ... + K(dk)*s^dk
        ks = np.zeros((np_dim, nk), order='F', dtype=float)
        for k in range(dk + 1):
            ks += ker[:np_dim, :nk, k] * (s ** k)

        # Compute P(s) * K(s) - should be zero
        product = ps @ ks

        np.testing.assert_allclose(product, np.zeros_like(product), rtol=1e-10, atol=1e-10)


def test_mc03nd_parameter_errors():
    """
    Test MC03ND parameter validation.
    """
    from ctrlsys import mc03nd

    mp, np_dim, dp = 5, 4, 2
    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    tol = 0.0

    # Test invalid dp (must be >= 1)
    with pytest.raises(ValueError):
        mc03nd(mp, np_dim, 0, p, tol)


def test_mc03nd_polynomial_coefficients():
    """
    Verify polynomial coefficient identity for the kernel.

    For K(s) to satisfy P(s)*K(s) = 0 as a polynomial identity,
    all coefficient matrices of the product must be zero.

    Uses HTML doc example data.
    """
    from ctrlsys import mc03nd

    mp, np_dim, dp = 5, 4, 2

    # Same data as HTML example
    p = np.zeros((mp, np_dim, dp + 1), order='F', dtype=float)
    p[:, :, 0] = np.array([
        [2.0, 2.0, 0.0, 3.0],
        [0.0, 4.0, 0.0, 6.0],
        [8.0, 8.0, 0.0, 12.0],
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0, 3.0]
    ], order='F')

    p[:, :, 1] = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [4.0, 0.0, 4.0, 0.0],
        [2.0, 2.0, 0.0, 3.0],
        [3.0, 2.0, 1.0, 3.0]
    ], order='F')

    p[:, :, 2] = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0]
    ], order='F')

    tol = 0.0

    dk, gam, nullsp, ker, info = mc03nd(mp, np_dim, dp, p, tol)

    assert info == 0
    assert dk >= 0

    nk = sum(gam[:dk + 1])

    # Verify polynomial coefficient identity:
    # P(s)*K(s) = sum_{m=0}^{dp+dk} C_m * s^m where
    # C_m = sum_{i+j=m} P_i * K_j
    # All C_m must be zero for polynomial identity

    for m in range(dp + dk + 1):
        coeff = np.zeros((mp, nk), order='F', dtype=float)
        for i in range(dp + 1):
            j = m - i
            if 0 <= j <= dk:
                coeff += p[:, :, i] @ ker[:np_dim, :nk, j]
        np.testing.assert_allclose(coeff, np.zeros_like(coeff), rtol=1e-10, atol=1e-10)
