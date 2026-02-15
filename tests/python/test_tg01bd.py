"""
Tests for TG01BD - Reduction of descriptor system to generalized Hessenberg form.

TG01BD reduces matrices A and E of the descriptor system pencil
S = (A, B; C, 0) - lambda*(E, 0; 0, 0) to generalized upper Hessenberg form
using orthogonal transformations: Q' * A * Z = H, Q' * E * Z = T.

Test data sources:
- Mathematical properties of Hessenberg-triangular form
- Random matrices with verified transformations
"""

import numpy as np
import pytest

from ctrlsys import tg01bd


def test_tg01bd_basic():
    """
    Test basic transformation with identity E.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()
    e_orig = e.copy()

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01bd(
        'G', 'I', 'I', 1, n, a, e, b, c, q, z
    )

    assert info == 0

    # H should be upper Hessenberg (zeros below first subdiagonal)
    for i in range(2, n):
        for j in range(i - 1):
            assert abs(a_out[i, j]) < 1e-14, f"H[{i},{j}] = {a_out[i, j]} should be zero"

    # T should be upper triangular
    for i in range(1, n):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-14, f"T[{i},{j}] = {e_out[i, j]} should be zero"


def test_tg01bd_orthogonality():
    """
    Test that Q and Z are orthogonal.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 5, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01bd(
        'G', 'I', 'I', 1, n, a, e, b, c, q, z
    )

    assert info == 0

    # Q should be orthogonal: Q'*Q = I
    qtq = q_out.T @ q_out
    np.testing.assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14)

    # Z should be orthogonal: Z'*Z = I
    ztz = z_out.T @ z_out
    np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14)


def test_tg01bd_transformation_consistency():
    """
    Test that Q' * A * Z = H and Q' * E * Z = T.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 4, 1, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01bd(
        'G', 'I', 'I', 1, n, a, e, b, c, q, z
    )

    assert info == 0

    # H = Q' * A * Z
    h_check = q_out.T @ a_orig @ z_out
    np.testing.assert_allclose(a_out, h_check, rtol=1e-13, atol=1e-14)

    # T = Q' * E * Z
    t_check = q_out.T @ e_orig @ z_out
    np.testing.assert_allclose(e_out, t_check, rtol=1e-13, atol=1e-14)

    # B_out = Q' * B
    b_check = q_out.T @ b_orig
    np.testing.assert_allclose(b_out, b_check, rtol=1e-13, atol=1e-14)

    # C_out = C * Z
    c_check = c_orig @ z_out
    np.testing.assert_allclose(c_out, c_check, rtol=1e-13, atol=1e-14)


def test_tg01bd_upper_triangular_e():
    """
    Test with E already upper triangular (JOBE='U').

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    # Create upper triangular E
    e = np.triu(np.random.randn(n, n)).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01bd(
        'U', 'I', 'I', 1, n, a, e, b, c, q, z
    )

    assert info == 0

    # E should remain upper triangular
    for i in range(1, n):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-14


def test_tg01bd_no_q_z():
    """
    Test with COMPQ='N' and COMPZ='N' (don't compute Q, Z).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 3, 1, 1

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    q = np.zeros((n, n), order='F', dtype=float)
    z = np.zeros((n, n), order='F', dtype=float)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01bd(
        'G', 'N', 'N', 1, n, a, e, b, c, q, z
    )

    assert info == 0

    # H should still be upper Hessenberg
    for i in range(2, n):
        for j in range(i - 1):
            assert abs(a_out[i, j]) < 1e-14


def test_tg01bd_partial_range():
    """
    Test with partial active range (ILO > 1 or IHI < N).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m, p = 5, 2, 2
    ilo, ihi = 2, 4

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.random.randn(n, n).astype(float, order='F')
    # Make A and E upper triangular in rows/cols 1:ILO-1 and IHI+1:N
    for i in range(ilo - 1):
        for j in range(i):
            a[i, j] = 0.0
            e[i, j] = 0.0
    for i in range(ihi, n):
        for j in range(i):
            a[i, j] = 0.0
            e[i, j] = 0.0

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01bd(
        'G', 'I', 'I', ilo, ihi, a, e, b, c, q, z
    )

    assert info == 0


def test_tg01bd_invalid_params():
    """
    Test error handling for invalid parameters.
    """
    n, m, p = 3, 1, 1

    a = np.zeros((n, n), order='F', dtype=float)
    e = np.zeros((n, n), order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)
    c = np.zeros((p, n), order='F', dtype=float)
    q = np.zeros((n, n), order='F', dtype=float)
    z = np.zeros((n, n), order='F', dtype=float)

    # Invalid ILO
    _, _, _, _, _, _, info = tg01bd(
        'G', 'I', 'I', 0, n, a.copy(), e.copy(), b.copy(), c.copy(), q.copy(), z.copy()
    )
    assert info == -7

    # Invalid IHI
    _, _, _, _, _, _, info = tg01bd(
        'G', 'I', 'I', 1, n + 1, a.copy(), e.copy(), b.copy(), c.copy(), q.copy(), z.copy()
    )
    assert info == -8
