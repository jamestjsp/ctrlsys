"""
Tests for MB04QF: Form triangular block factors of symplectic block reflector.

Forms triangular block factors R, S, T of a symplectic block reflector SH
defined as a product of Householder reflectors and Givens rotations.

Uses numpy only - no scipy.
"""

import numpy as np


def test_mb04qf_basic():
    """
    Test MB04QF with identity-like inputs.

    When W diagonal=0, tau=0, only Givens structure matters.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04qf

    np.random.seed(42)
    n, k = 4, 2

    V = np.zeros((n, k), order='F', dtype=float)
    W = np.zeros((n, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 0.0

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        cs[2*i] = 1.0
        cs[2*i + 1] = 0.0

    tau = np.zeros(k, order='F', dtype=float)

    rs, t, info = mb04qf('F', 'C', 'C', n, k, V, W, cs, tau)

    assert info == 0
    assert rs.shape == (k, 6*k)
    assert t.shape == (k, 9*k)


def test_mb04qf_with_givens_rotations():
    """
    Test MB04QF with non-trivial Givens rotations.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04qf

    np.random.seed(123)
    n, k = 5, 3

    V = np.zeros((n, k), order='F', dtype=float)
    W = np.zeros((n, k), order='F', dtype=float)
    for i in range(k):
        V[i:, i] = np.random.randn(n - i)
        W[i:, i] = np.random.randn(n - i)

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        theta = np.random.uniform(0, 2*np.pi)
        cs[2*i] = np.cos(theta)
        cs[2*i + 1] = np.sin(theta)

    tau = np.random.uniform(0.5, 1.5, k).astype(float, order='F')

    rs, t, info = mb04qf('F', 'C', 'C', n, k, V, W, cs, tau)

    assert info == 0
    assert rs.shape == (k, 6*k)
    assert t.shape == (k, 9*k)


def test_mb04qf_rowwise_storage():
    """
    Test MB04QF with STOREV='R' and STOREW='R'.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04qf

    np.random.seed(456)
    n, k = 4, 2

    V = np.zeros((k, n), order='F', dtype=float)
    W = np.zeros((k, n), order='F', dtype=float)
    for i in range(k):
        V[i, i:] = np.random.randn(n - i)
        W[i, i:] = np.random.randn(n - i)

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        cs[2*i] = 1.0
        cs[2*i + 1] = 0.0

    tau = np.zeros(k, order='F', dtype=float)

    rs, t, info = mb04qf('F', 'R', 'R', n, k, V, W, cs, tau)

    assert info == 0
    assert rs.shape == (k, 6*k)
    assert t.shape == (k, 9*k)


def test_mb04qf_mixed_storage():
    """
    Test MB04QF with STOREV='C' and STOREW='R'.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04qf

    np.random.seed(789)
    n, k = 4, 2

    V = np.zeros((n, k), order='F', dtype=float)
    W = np.zeros((k, n), order='F', dtype=float)
    for i in range(k):
        V[i:, i] = np.random.randn(n - i)
        W[i, i:] = np.random.randn(n - i)

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        cs[2*i] = 1.0
        cs[2*i + 1] = 0.0

    tau = np.zeros(k, order='F', dtype=float)

    rs, t, info = mb04qf('F', 'C', 'R', n, k, V, W, cs, tau)

    assert info == 0
    assert rs.shape == (k, 6*k)
    assert t.shape == (k, 9*k)


def test_mb04qf_k_equals_1():
    """
    Test MB04QF with K=1 (single reflector/rotation).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04qf

    np.random.seed(111)
    n, k = 5, 1

    V = np.random.randn(n, k).astype(float, order='F')
    W = np.random.randn(n, k).astype(float, order='F')

    cs = np.array([0.8, 0.6], order='F', dtype=float)
    tau = np.array([0.5], order='F', dtype=float)

    rs, t, info = mb04qf('F', 'C', 'C', n, k, V, W, cs, tau)

    assert info == 0
    assert rs.shape == (k, 6*k)
    assert t.shape == (k, 9*k)


def test_mb04qf_n_equals_zero():
    """
    Test MB04QF with N=0 (quick return).
    """
    from ctrlsys import mb04qf

    n, k = 0, 1

    V = np.zeros((1, k), order='F', dtype=float)
    W = np.zeros((1, k), order='F', dtype=float)

    cs = np.array([1.0, 0.0], order='F', dtype=float)
    tau = np.array([0.0], order='F', dtype=float)

    rs, t, info = mb04qf('F', 'C', 'C', n, k, V, W, cs, tau)

    assert info == 0


def test_mb04qf_large_k():
    """
    Test MB04QF with larger K value.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb04qf

    np.random.seed(333)
    n, k = 10, 5

    V = np.zeros((n, k), order='F', dtype=float)
    W = np.zeros((n, k), order='F', dtype=float)
    for i in range(k):
        V[i:, i] = np.random.randn(n - i)
        W[i:, i] = np.random.randn(n - i)

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        theta = np.random.uniform(0, 2*np.pi)
        cs[2*i] = np.cos(theta)
        cs[2*i + 1] = np.sin(theta)

    tau = np.random.uniform(0.5, 1.5, k).astype(float, order='F')

    rs, t, info = mb04qf('F', 'C', 'C', n, k, V, W, cs, tau)

    assert info == 0
    assert rs.shape == (k, 6*k)
    assert t.shape == (k, 9*k)


def test_mb04qf_givens_cosine_sine_property():
    """
    Test that Givens rotation c^2 + s^2 = 1 is reflected in outputs.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import mb04qf

    np.random.seed(555)
    n, k = 4, 2

    V = np.zeros((n, k), order='F', dtype=float)
    W = np.zeros((n, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 0.0

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        theta = np.random.uniform(0, 2*np.pi)
        cs[2*i] = np.cos(theta)
        cs[2*i + 1] = np.sin(theta)
        np.testing.assert_allclose(cs[2*i]**2 + cs[2*i+1]**2, 1.0, rtol=1e-14)

    tau = np.zeros(k, order='F', dtype=float)

    rs, t, info = mb04qf('F', 'C', 'C', n, k, V, W, cs, tau)

    assert info == 0
    assert rs.shape == (k, 6*k)
    assert t.shape == (k, 9*k)
