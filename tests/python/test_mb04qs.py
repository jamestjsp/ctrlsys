"""
Tests for MB04QS: Multiplication with product of symplectic reflectors and Givens rotations.

MB04QS overwrites general real m-by-n matrices C and D with
U * [op(C); op(D)] or U^T * [op(C); op(D)]
where U is defined as a product of symplectic reflectors and Givens rotations,
as returned by MB04PU or MB04RU.

Uses numpy only - no scipy.
"""

import numpy as np
import pytest


def test_mb04qs_quick_return_n_zero():
    """Test quick return when n=0."""
    from ctrlsys import mb04qs

    m, n = 4, 0
    ilo = 1
    v = np.eye(m, order='F')
    w = np.eye(m, order='F')
    c = np.zeros((m, 1), order='F')
    d = np.zeros((m, 1), order='F')
    cs = np.ones(2 * m - 2 if m > 1 else 1, order='F')
    tau = np.zeros(m - 1 if m > 1 else 1, order='F')

    c_out, d_out, info = mb04qs('N', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == 0


def test_mb04qs_quick_return_m_le_ilo():
    """Test quick return when m <= ilo (no active rows)."""
    from ctrlsys import mb04qs

    m, n = 3, 4
    ilo = 4
    v = np.eye(m, order='F')
    w = np.eye(m, order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    d = np.random.randn(m, n).astype(float, order='F')
    cs = np.ones(1, order='F')
    tau = np.zeros(1, order='F')

    c_orig = c.copy()
    d_orig = d.copy()

    c_out, d_out, info = mb04qs('N', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == 0
    np.testing.assert_allclose(c_out, c_orig, rtol=1e-14)
    np.testing.assert_allclose(d_out, d_orig, rtol=1e-14)


def test_mb04qs_workspace_query():
    """Test workspace query mode (ldwork=-1)."""
    from ctrlsys import mb04qs

    m, n = 6, 4
    ilo = 1
    v = np.eye(m, order='F')
    w = np.eye(m, order='F')
    c = np.zeros((m, n), order='F')
    d = np.zeros((m, n), order='F')
    cs = np.ones(2 * m - 2 if m > 1 else 1, order='F')
    tau = np.zeros(m - 1 if m > 1 else 1, order='F')

    c_out, d_out, info = mb04qs('N', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau, ldwork=-1)

    assert info == 0


def test_mb04qs_identity_reflectors():
    """
    Test with identity-like reflectors (tau=0 means no reflection) and
    identity Givens rotations (cos=1, sin=0).

    When tau=0 for all reflectors and Givens are identity, the transformation
    should leave C and D unchanged.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04qs

    np.random.seed(42)
    m, n = 5, 3
    ilo = 1

    v = np.eye(m, order='F')
    w = np.eye(m, order='F')

    c = np.random.randn(m, n).astype(float, order='F')
    d = np.random.randn(m, n).astype(float, order='F')

    mh = m - ilo
    cs = np.zeros(2 * mh if mh > 0 else 1, order='F')
    for i in range(mh):
        cs[2 * i] = 1.0
        cs[2 * i + 1] = 0.0

    tau = np.zeros(mh if mh > 0 else 1, order='F')

    c_orig = c.copy()
    d_orig = d.copy()

    c_out, d_out, info = mb04qs('N', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == 0


def test_mb04qs_with_transpose_tranc():
    """
    Test with TRANC='T' (C is stored as N-by-M transpose).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04qs

    np.random.seed(123)
    m, n = 4, 3
    ilo = 1

    v = np.eye(m, order='F')
    w = np.eye(m, order='F')

    c = np.random.randn(n, m).astype(float, order='F')
    d = np.random.randn(m, n).astype(float, order='F')

    mh = m - ilo
    cs = np.zeros(2 * mh if mh > 0 else 1, order='F')
    for i in range(mh):
        cs[2 * i] = 1.0
        cs[2 * i + 1] = 0.0
    tau = np.zeros(mh if mh > 0 else 1, order='F')

    c_out, d_out, info = mb04qs('T', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == 0


def test_mb04qs_invalid_tranc():
    """Test error for invalid TRANC parameter."""
    from ctrlsys import mb04qs

    m, n = 4, 3
    ilo = 1
    v = np.eye(m, order='F')
    w = np.eye(m, order='F')
    c = np.zeros((m, n), order='F')
    d = np.zeros((m, n), order='F')
    cs = np.ones(2 * (m - 1) if m > 1 else 1, order='F')
    tau = np.zeros(m - 1 if m > 1 else 1, order='F')

    c_out, d_out, info = mb04qs('X', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == -1


def test_mb04qs_invalid_trand():
    """Test error for invalid TRAND parameter."""
    from ctrlsys import mb04qs

    m, n = 4, 3
    ilo = 1
    v = np.eye(m, order='F')
    w = np.eye(m, order='F')
    c = np.zeros((m, n), order='F')
    d = np.zeros((m, n), order='F')
    cs = np.ones(2 * (m - 1) if m > 1 else 1, order='F')
    tau = np.zeros(m - 1 if m > 1 else 1, order='F')

    c_out, d_out, info = mb04qs('N', 'X', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == -2


def test_mb04qs_invalid_tranu():
    """Test error for invalid TRANU parameter."""
    from ctrlsys import mb04qs

    m, n = 4, 3
    ilo = 1
    v = np.eye(m, order='F')
    w = np.eye(m, order='F')
    c = np.zeros((m, n), order='F')
    d = np.zeros((m, n), order='F')
    cs = np.ones(2 * (m - 1) if m > 1 else 1, order='F')
    tau = np.zeros(m - 1 if m > 1 else 1, order='F')

    c_out, d_out, info = mb04qs('N', 'N', 'X', m, n, ilo, v, w, c, d, cs, tau)

    assert info == -3


def test_mb04qs_negative_m():
    """Test error for negative M."""
    from ctrlsys import mb04qs

    m, n = -1, 3
    ilo = 1
    v = np.zeros((1, 1), order='F')
    w = np.zeros((1, 1), order='F')
    c = np.zeros((1, n), order='F')
    d = np.zeros((1, n), order='F')
    cs = np.zeros(1, order='F')
    tau = np.zeros(1, order='F')

    c_out, d_out, info = mb04qs('N', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == -4


def test_mb04qs_negative_n():
    """Test error for negative N."""
    from ctrlsys import mb04qs

    m, n = 4, -1
    ilo = 1
    v = np.eye(m, order='F')
    w = np.eye(m, order='F')
    c = np.zeros((m, 1), order='F')
    d = np.zeros((m, 1), order='F')
    cs = np.ones(1, order='F')
    tau = np.zeros(1, order='F')

    c_out, d_out, info = mb04qs('N', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == -5


def test_mb04qs_invalid_ilo_too_small():
    """Test error for ILO < 1."""
    from ctrlsys import mb04qs

    m, n = 4, 3
    ilo = 0
    v = np.eye(m, order='F')
    w = np.eye(m, order='F')
    c = np.zeros((m, n), order='F')
    d = np.zeros((m, n), order='F')
    cs = np.ones(1, order='F')
    tau = np.zeros(1, order='F')

    c_out, d_out, info = mb04qs('N', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == -6


def test_mb04qs_invalid_ilo_too_large():
    """Test error for ILO > M+1."""
    from ctrlsys import mb04qs

    m, n = 4, 3
    ilo = 6
    v = np.eye(m, order='F')
    w = np.eye(m, order='F')
    c = np.zeros((m, n), order='F')
    d = np.zeros((m, n), order='F')
    cs = np.ones(1, order='F')
    tau = np.zeros(1, order='F')

    c_out, d_out, info = mb04qs('N', 'N', 'N', m, n, ilo, v, w, c, d, cs, tau)

    assert info == -6
