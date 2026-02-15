"""
Tests for MB02SD: LU factorization of upper Hessenberg matrix.

Uses numpy only.
"""

import numpy as np


def _reconstruct_hessenberg_lu(h_out, ipiv):
    """Reconstruct H from Hessenberg LU factorization H = P*L*U."""
    n = h_out.shape[0]
    rec = h_out.copy()
    for j in range(n - 2, -1, -1):
        mult = rec[j + 1, j]
        rec[j + 1, j + 1:] += mult * rec[j, j + 1:]
        rec[j + 1, j] = mult * rec[j, j]
        jp = ipiv[j] - 1
        if jp != j:
            rec[j, j:], rec[jp, j:] = rec[jp, j:].copy(), rec[j, j:].copy()
    return rec


def test_mb02sd_basic():
    """
    Test MB02SD with a basic upper Hessenberg matrix.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb02sd

    np.random.seed(42)
    n = 4

    h = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn()

    h_orig = h.copy()
    h_out, ipiv, info = mb02sd(n, h)

    assert info == 0
    assert h_out.shape == (n, n)
    assert ipiv.shape == (n,)

    np.testing.assert_allclose(_reconstruct_hessenberg_lu(h_out, ipiv), h_orig, atol=1e-12)


def test_mb02sd_n3():
    """
    Test MB02SD with 3x3 upper Hessenberg matrix.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb02sd

    np.random.seed(123)
    n = 3

    h = np.array([
        [2.0, 1.0, 3.0],
        [4.0, 5.0, 2.0],
        [0.0, 3.0, 1.0]
    ], order='F', dtype=float)
    h_orig = h.copy()

    h_out, ipiv, info = mb02sd(n, h)

    assert info == 0
    assert h_out.shape == (n, n)
    assert ipiv.shape == (n,)

    np.testing.assert_allclose(_reconstruct_hessenberg_lu(h_out, ipiv), h_orig, atol=1e-12)


def test_mb02sd_singular():
    """
    Test MB02SD with a singular matrix (should return info > 0).
    """
    from ctrlsys import mb02sd

    n = 3
    h = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    h_out, ipiv, info = mb02sd(n, h)

    assert info > 0


def test_mb02sd_n1():
    """Test MB02SD with 1x1 matrix."""
    from ctrlsys import mb02sd

    n = 1
    h = np.array([[5.0]], order='F', dtype=float)

    h_out, ipiv, info = mb02sd(n, h)

    assert info == 0
    assert ipiv[0] == 1
    np.testing.assert_allclose(h_out[0, 0], 5.0, rtol=1e-14)


def test_mb02sd_n0():
    """Test MB02SD with n=0 (quick return)."""
    from ctrlsys import mb02sd

    n = 0
    h = np.zeros((1, 1), order='F', dtype=float)

    h_out, ipiv, info = mb02sd(n, h)

    assert info == 0
