"""
Tests for MB04QB: Blocked version of symplectic reflector/rotation application.

MB04QB applies a product of symplectic reflectors and Givens rotations
to overwrite m-by-n matrices C and D with Q*[op(C);op(D)] or Q^T*[op(C);op(D)].

Uses numpy only - no scipy.
"""

import numpy as np


def test_mb04qb_quick_return_n_zero():
    """Test quick return when n=0."""
    from ctrlsys import mb04qb

    m, n, k = 4, 0, 2
    v = np.eye(m, k, order='F')
    w = np.eye(m, k, order='F')
    c = np.zeros((m, 1), order='F')
    d = np.zeros((m, 1), order='F')
    cs = np.ones(2 * k, order='F')
    tau = np.zeros(k, order='F')

    c_out, d_out, info = mb04qb(
        'N', 'N', 'N', 'C', 'C', m, n, k,
        v, w, c, d, cs, tau
    )

    assert info == 0


def test_mb04qb_quick_return_k_zero():
    """Test quick return when k=0."""
    from ctrlsys import mb04qb

    m, n, k = 4, 3, 0
    v = np.zeros((m, 1), order='F')
    w = np.zeros((m, 1), order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    d = np.random.randn(m, n).astype(float, order='F')
    cs = np.zeros(1, order='F')
    tau = np.zeros(1, order='F')

    c_orig = c.copy()
    d_orig = d.copy()

    c_out, d_out, info = mb04qb(
        'N', 'N', 'N', 'C', 'C', m, n, k,
        v, w, c, d, cs, tau
    )

    assert info == 0
    np.testing.assert_allclose(c_out, c_orig, rtol=1e-14)
    np.testing.assert_allclose(d_out, d_orig, rtol=1e-14)


def test_mb04qb_workspace_query():
    """Test workspace query mode (ldwork=-1)."""
    from ctrlsys import mb04qb

    m, n, k = 10, 8, 5
    v = np.eye(m, k, order='F')
    w = np.eye(m, k, order='F')
    c = np.random.randn(m, n).astype(float, order='F')
    d = np.random.randn(m, n).astype(float, order='F')
    cs = np.ones(2 * k, order='F')
    tau = np.zeros(k, order='F')

    c_out, d_out, info = mb04qb(
        'N', 'N', 'N', 'C', 'C', m, n, k,
        v, w, c, d, cs, tau, ldwork=-1
    )

    assert info == 0


def test_mb04qb_identity_reflectors():
    """
    Test with identity-like reflectors (tau=0 means no reflection).

    When tau=0 for all reflectors, the transformation should be close
    to applying only the Givens rotations on identity matrix portions.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04qb

    np.random.seed(42)
    m, n, k = 6, 4, 3

    v = np.eye(m, k, order='F')
    w = np.eye(m, k, order='F')

    c = np.random.randn(m, n).astype(float, order='F')
    d = np.random.randn(m, n).astype(float, order='F')

    cs = np.zeros(2 * k, order='F')
    for i in range(k):
        cs[2 * i] = 1.0
        cs[2 * i + 1] = 0.0

    tau = np.zeros(k, order='F')

    c_orig = c.copy()
    d_orig = d.copy()

    c_out, d_out, info = mb04qb(
        'N', 'N', 'N', 'C', 'C', m, n, k,
        v, w, c_orig, d_orig, cs, tau
    )

    assert info == 0


def test_mb04qb_invalid_tranc():
    """Test error for invalid TRANC parameter."""
    from ctrlsys import mb04qb

    m, n, k = 4, 3, 2
    v = np.eye(m, k, order='F')
    w = np.eye(m, k, order='F')
    c = np.zeros((m, n), order='F')
    d = np.zeros((m, n), order='F')
    cs = np.ones(2 * k, order='F')
    tau = np.zeros(k, order='F')

    c_out, d_out, info = mb04qb(
        'X', 'N', 'N', 'C', 'C', m, n, k,
        v, w, c, d, cs, tau
    )

    assert info == -1


def test_mb04qb_invalid_trand():
    """Test error for invalid TRAND parameter."""
    from ctrlsys import mb04qb

    m, n, k = 4, 3, 2
    v = np.eye(m, k, order='F')
    w = np.eye(m, k, order='F')
    c = np.zeros((m, n), order='F')
    d = np.zeros((m, n), order='F')
    cs = np.ones(2 * k, order='F')
    tau = np.zeros(k, order='F')

    c_out, d_out, info = mb04qb(
        'N', 'X', 'N', 'C', 'C', m, n, k,
        v, w, c, d, cs, tau
    )

    assert info == -2


def test_mb04qb_negative_m():
    """Test error for negative M."""
    from ctrlsys import mb04qb

    m, n, k = -1, 3, 0
    v = np.zeros((1, 1), order='F')
    w = np.zeros((1, 1), order='F')
    c = np.zeros((1, n), order='F')
    d = np.zeros((1, n), order='F')
    cs = np.zeros(1, order='F')
    tau = np.zeros(1, order='F')

    c_out, d_out, info = mb04qb(
        'N', 'N', 'N', 'C', 'C', m, n, k,
        v, w, c, d, cs, tau
    )

    assert info == -6


def test_mb04qb_k_greater_than_m():
    """Test error for K > M."""
    from ctrlsys import mb04qb

    m, n, k = 3, 4, 5
    v = np.eye(k, k, order='F')
    w = np.eye(k, k, order='F')
    c = np.zeros((m, n), order='F')
    d = np.zeros((m, n), order='F')
    cs = np.ones(2 * k, order='F')
    tau = np.zeros(k, order='F')

    c_out, d_out, info = mb04qb(
        'N', 'N', 'N', 'C', 'C', m, n, k,
        v, w, c, d, cs, tau
    )

    assert info == -8
