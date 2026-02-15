"""
Tests for MB03JD - Eigenvalue reordering for real skew-Hamiltonian/Hamiltonian pencil.

MB03JD moves eigenvalues with strictly negative real parts of an N-by-N real
skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the
leading principal subpencil, while keeping the triangular form.

On entry:
        (  A  D  )      (  B  F  )
    S = (        ), H = (        ),
        (  0  A' )      (  0 -B' )

where A is upper triangular and B is upper quasi-triangular.
"""

import numpy as np
import pytest


def test_mb03jd_basic_n4():
    """
    Test basic eigenvalue reordering with N=4 (smallest valid case).

    Random seed: 42 (for reproducibility)
    Validates that:
    1. Output info is 0 (success)
    2. Output neig >= 0 and neig <= m
    3. Q is orthogonal when COMPQ='I'
    """
    from ctrlsys import mb03jd

    np.random.seed(42)

    n = 4
    m = n // 2

    a = np.array([
        [1.0, 0.5],
        [0.0, -0.8]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.3],
        [-0.3, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.7, 0.2],
        [0.0, 0.4]
    ], order='F', dtype=float)

    f = np.array([
        [1.0, 0.1],
        [0.1, 0.8]
    ], order='F', dtype=float)

    a_out, d_out, b_out, f_out, q, neig, info = mb03jd('I', n, a, d, b, f)

    assert info == 0
    assert neig >= 0
    assert neig <= m

    np.testing.assert_allclose(q @ q.T, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03jd_no_transform():
    """
    Test with COMPQ='N' (no transformation matrix computed).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03jd

    np.random.seed(123)

    n = 4
    m = n // 2

    a = np.array([
        [1.2, 0.3],
        [0.0, -0.5]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.2],
        [-0.2, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.6, 0.15],
        [0.0, 0.35]
    ], order='F', dtype=float)

    f = np.array([
        [0.9, 0.05],
        [0.05, 0.7]
    ], order='F', dtype=float)

    a_out, d_out, b_out, f_out, q, neig, info = mb03jd('N', n, a, d, b, f)

    assert info == 0
    assert neig >= 0


def test_mb03jd_zero_dimension():
    """
    Test with N=0 (quick return case).
    """
    from ctrlsys import mb03jd

    n = 0

    a = np.zeros((0, 0), order='F', dtype=float)
    d = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, 0), order='F', dtype=float)
    f = np.zeros((0, 0), order='F', dtype=float)

    a_out, d_out, b_out, f_out, q, neig, info = mb03jd('N', n, a, d, b, f)

    assert info == 0
    assert neig == 0


def test_mb03jd_invalid_n_odd():
    """
    Test error handling for odd N (N must be even).
    """
    from ctrlsys import mb03jd

    n = 3

    a = np.zeros((1, 1), order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, 1), order='F', dtype=float)
    f = np.zeros((1, 1), order='F', dtype=float)

    a_out, d_out, b_out, f_out, q, neig, info = mb03jd('N', n, a, d, b, f)

    assert info == -2


def test_mb03jd_update_mode():
    """
    Test with COMPQ='U' (update existing transformation).

    Random seed: 456 (for reproducibility)
    Validates that existing Q matrix is updated correctly.
    """
    from ctrlsys import mb03jd

    np.random.seed(456)

    n = 4
    m = n // 2

    a = np.array([
        [0.8, 0.4],
        [0.0, -0.6]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.15],
        [-0.15, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.5, 0.12],
        [0.0, 0.32]
    ], order='F', dtype=float)

    f = np.array([
        [0.85, 0.08],
        [0.08, 0.75]
    ], order='F', dtype=float)

    q_init = np.eye(n, order='F', dtype=float)

    a_out, d_out, b_out, f_out, q, neig, info = mb03jd('U', n, a, d, b, f, q=q_init)

    assert info == 0
    assert neig >= 0

    np.testing.assert_allclose(q @ q.T, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03jd_orthogonality_preserved():
    """
    Test that Q remains orthogonal for various inputs.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03jd

    np.random.seed(789)

    n = 6
    m = n // 2

    a = np.triu(np.random.randn(m, m).astype(float))
    a = np.asfortranarray(a)

    d_upper = np.triu(np.random.randn(m, m), 1)
    d = d_upper - d_upper.T
    d = np.triu(d)
    d = np.asfortranarray(d)

    b = np.triu(np.random.randn(m, m).astype(float))
    b = np.asfortranarray(b)

    f_diag = np.diag(np.random.rand(m) + 0.5)
    f_upper = np.triu(np.random.randn(m, m), 1) * 0.1
    f = f_diag + f_upper + f_upper.T
    f = np.triu(f)
    f = np.asfortranarray(f)

    a_out, d_out, b_out, f_out, q, neig, info = mb03jd('I', n, a, d, b, f)

    assert info in [0, 1, 2]
    if info == 0:
        np.testing.assert_allclose(q @ q.T, np.eye(n), rtol=1e-13, atol=1e-13)


def test_mb03jd_determinant_preserved():
    """
    Test that det(Q) = +/- 1 for orthogonal transformation.

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb03jd

    np.random.seed(999)

    n = 4
    m = n // 2

    a = np.array([
        [0.9, 0.35],
        [0.0, -0.55]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.1],
        [-0.1, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.58, 0.18],
        [0.0, 0.38]
    ], order='F', dtype=float)

    f = np.array([
        [0.88, 0.06],
        [0.06, 0.78]
    ], order='F', dtype=float)

    a_out, d_out, b_out, f_out, q, neig, info = mb03jd('I', n, a, d, b, f)

    assert info == 0

    det_q = np.linalg.det(q)
    np.testing.assert_allclose(np.abs(det_q), 1.0, rtol=1e-13)


def test_mb03jd_n8_larger():
    """
    Test with larger N=8 case.

    Random seed: 1234 (for reproducibility)
    """
    from ctrlsys import mb03jd

    np.random.seed(1234)

    n = 8
    m = n // 2

    a = np.triu(np.random.randn(m, m).astype(float))
    a = np.asfortranarray(a)

    d = np.zeros((m, m), order='F', dtype=float)
    for i in range(m):
        for j in range(i+1, m):
            d[i, j] = np.random.randn() * 0.1

    b = np.triu(np.random.randn(m, m).astype(float))
    b = np.asfortranarray(b)

    f = np.zeros((m, m), order='F', dtype=float)
    for i in range(m):
        f[i, i] = np.random.rand() + 0.5
        for j in range(i+1, m):
            f[i, j] = np.random.randn() * 0.1

    a_out, d_out, b_out, f_out, q, neig, info = mb03jd('I', n, a, d, b, f)

    assert info in [0, 1, 2]
    if info == 0:
        np.testing.assert_allclose(q @ q.T, np.eye(n), rtol=1e-13, atol=1e-13)
        assert neig >= 0
        assert neig <= m


def test_mb03jd_invalid_compq():
    """
    Test error handling for invalid COMPQ parameter.
    """
    from ctrlsys import mb03jd

    n = 4
    m = n // 2

    a = np.zeros((m, m), order='F', dtype=float)
    d = np.zeros((m, m), order='F', dtype=float)
    b = np.zeros((m, m), order='F', dtype=float)
    f = np.zeros((m, m), order='F', dtype=float)

    a_out, d_out, b_out, f_out, q, neig, info = mb03jd('X', n, a, d, b, f)

    assert info == -1
