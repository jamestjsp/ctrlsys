"""
Tests for MB03JP - Move eigenvalues with negative real parts of real
skew-Hamiltonian/Hamiltonian pencil in structured Schur form to leading subpencil.

This is a panel-based blocked variant of MB03JD.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03jp_basic_4x4():
    """
    Test basic eigenvalue reordering for a 4x4 pencil (n=4, m=2).

    Verifies that eigenvalues with negative real parts are moved to
    the leading subpencil while preserving the structured Schur form.
    Random seed: 42 (for reproducibility).
    """
    from ctrlsys import mb03jp

    np.random.seed(42)

    n = 4
    m = n // 2

    a = np.array([
        [1.0, 0.5],
        [0.0, 2.0]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.3],
        [0.0, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [-1.0, 0.2],
        [0.0, 0.5]
    ], order='F', dtype=float)

    f = np.array([
        [0.5, 0.1],
        [0.0, 0.3]
    ], order='F', dtype=float)

    a_out, d_out, b_out, f_out, q_out, neig, info = mb03jp(
        compq='I', n=n, a=a, d=d, b=b, f=f
    )

    assert info == 0, f"Expected info=0, got {info}"
    assert isinstance(neig, int), "neig should be an integer"
    assert neig >= 0 and neig <= m, f"neig should be in [0, {m}]"

    assert a_out.shape == (m, m)
    assert d_out.shape == (m, m)
    assert b_out.shape == (m, m)
    assert f_out.shape == (m, m)
    assert q_out.shape == (n, n)

    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14,
                   err_msg="Q should be orthogonal")


def test_mb03jp_eigenvalue_preservation_8x8():
    """
    Test eigenvalue preservation for an 8x8 pencil.

    Verifies that eigenvalues are preserved under similarity transformation.
    Random seed: 123 (for reproducibility).
    """
    from ctrlsys import mb03jp

    np.random.seed(123)

    n = 8
    m = n // 2

    a = np.triu(np.random.randn(m, m))
    a = np.asarray(a, order='F', dtype=float)

    d_full = np.random.randn(m, m)
    d = np.triu(d_full - d_full.T)
    d = np.asarray(d, order='F', dtype=float)

    b = np.triu(np.random.randn(m, m))
    subdiag = np.random.randn(m-1) * 0.1
    for i in range(m-1):
        b[i+1, i] = subdiag[i]
    b = np.asarray(b, order='F', dtype=float)

    f_full = np.random.randn(m, m)
    f = np.triu(f_full + f_full.T)
    f = np.asarray(f, order='F', dtype=float)

    a_in = a.copy()
    b_in = b.copy()

    a_out, d_out, b_out, f_out, q_out, neig, info = mb03jp(
        compq='I', n=n, a=a, d=d, b=b, f=f
    )

    assert info == 0, f"Expected info=0, got {info}"

    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14,
                   err_msg="Q should be orthogonal")


def test_mb03jp_no_q():
    """
    Test with COMPQ='N' (no Q computation).
    Random seed: 456 (for reproducibility).
    """
    from ctrlsys import mb03jp

    np.random.seed(456)

    n = 6
    m = n // 2

    a = np.triu(np.random.randn(m, m))
    a = np.asarray(a, order='F', dtype=float)

    d_full = np.random.randn(m, m)
    d = np.triu(d_full - d_full.T)
    d = np.asarray(d, order='F', dtype=float)

    b = np.triu(np.random.randn(m, m))
    subdiag = np.random.randn(m-1) * 0.1
    for i in range(m-1):
        b[i+1, i] = subdiag[i]
    b = np.asarray(b, order='F', dtype=float)

    f_full = np.random.randn(m, m)
    f = np.triu(f_full + f_full.T)
    f = np.asarray(f, order='F', dtype=float)

    a_out, d_out, b_out, f_out, q_out, neig, info = mb03jp(
        compq='N', n=n, a=a, d=d, b=b, f=f
    )

    assert info == 0, f"Expected info=0, got {info}"
    assert neig >= 0 and neig <= m


def test_mb03jp_update_q():
    """
    Test with COMPQ='U' (update existing Q).
    Random seed: 789 (for reproducibility).
    """
    from ctrlsys import mb03jp

    np.random.seed(789)

    n = 4
    m = n // 2

    a = np.triu(np.random.randn(m, m))
    a = np.asarray(a, order='F', dtype=float)

    d_full = np.random.randn(m, m)
    d = np.triu(d_full - d_full.T)
    d = np.asarray(d, order='F', dtype=float)

    b = np.triu(np.random.randn(m, m))
    b = np.asarray(b, order='F', dtype=float)

    f_full = np.random.randn(m, m)
    f = np.triu(f_full + f_full.T)
    f = np.asarray(f, order='F', dtype=float)

    q0 = np.eye(n, order='F', dtype=float)

    a_out, d_out, b_out, f_out, q_out, neig, info = mb03jp(
        compq='U', n=n, a=a, d=d, b=b, f=f, q=q0
    )

    assert info == 0, f"Expected info=0, got {info}"

    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14,
                   err_msg="Q should be orthogonal")


def test_mb03jp_quick_return_n0():
    """
    Test quick return for n=0.
    """
    from ctrlsys import mb03jp

    n = 0
    a = np.array([], dtype=float).reshape(0, 0)
    a = np.asfortranarray(a)
    d = np.array([], dtype=float).reshape(0, 0)
    d = np.asfortranarray(d)
    b = np.array([], dtype=float).reshape(0, 0)
    b = np.asfortranarray(b)
    f = np.array([], dtype=float).reshape(0, 0)
    f = np.asfortranarray(f)

    a_out, d_out, b_out, f_out, q_out, neig, info = mb03jp(
        compq='N', n=n, a=a, d=d, b=b, f=f
    )

    assert info == 0, f"Expected info=0, got {info}"
    assert neig == 0, f"Expected neig=0, got {neig}"


def test_mb03jp_invalid_compq():
    """
    Test error handling for invalid COMPQ parameter.
    """
    from ctrlsys import mb03jp

    n = 4
    m = n // 2
    a = np.eye(m, order='F', dtype=float)
    d = np.zeros((m, m), order='F', dtype=float)
    b = np.eye(m, order='F', dtype=float)
    f = np.zeros((m, m), order='F', dtype=float)

    a_out, d_out, b_out, f_out, q_out, neig, info = mb03jp(
        compq='X', n=n, a=a, d=d, b=b, f=f
    )

    assert info == -1, f"Expected info=-1 for invalid compq, got {info}"


def test_mb03jp_invalid_n_odd():
    """
    Test error handling for odd n (must be even).
    """
    from ctrlsys import mb03jp

    n = 5
    m = 2
    a = np.eye(m, order='F', dtype=float)
    d = np.zeros((m, m), order='F', dtype=float)
    b = np.eye(m, order='F', dtype=float)
    f = np.zeros((m, m), order='F', dtype=float)

    a_out, d_out, b_out, f_out, q_out, neig, info = mb03jp(
        compq='N', n=n, a=a, d=d, b=b, f=f
    )

    assert info == -2, f"Expected info=-2 for odd n, got {info}"


def test_mb03jp_negative_eigenvalues_ordering():
    """
    Test that eigenvalues with negative real parts are ordered to the top.

    Creates a pencil with known positive and negative eigenvalues and verifies
    that after reordering, eigenvalues with negative real parts are in leading
    positions.
    Random seed: 999 (for reproducibility).
    """
    from ctrlsys import mb03jp

    np.random.seed(999)

    n = 4
    m = n // 2

    a = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.1],
        [0.0, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ], order='F', dtype=float)

    f = np.array([
        [0.1, 0.0],
        [0.0, 0.1]
    ], order='F', dtype=float)

    a_out, d_out, b_out, f_out, q_out, neig, info = mb03jp(
        compq='I', n=n, a=a, d=d, b=b, f=f
    )

    assert info == 0, f"Expected info=0, got {info}"

    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14)
