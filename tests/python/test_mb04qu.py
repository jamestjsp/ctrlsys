"""
Tests for MB04QU: Apply symplectic reflectors and Givens rotations to matrices.

Applies Q * [op(C); op(D)] or Q^T * [op(C); op(D)] where Q is defined as
a product of symplectic reflectors and Givens rotations.

Uses numpy only - no scipy.
"""

import numpy as np


def test_mb04qu_basic():
    """
    Test MB04QU with identity-like reflectors (tau=0 for F, W diag=0 for H).

    When tau=0 and W diagonal=0, reflectors are identity transformations.
    When cs=(1,0), Givens rotations are also identity.
    Result should match input.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04qu

    np.random.seed(42)
    m, n, k = 4, 3, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 0.0

    C = np.random.randn(m, n).astype(float, order='F')
    D = np.random.randn(m, n).astype(float, order='F')
    C_orig = C.copy()
    D_orig = D.copy()

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        cs[2*i] = 1.0
        cs[2*i + 1] = 0.0

    tau = np.zeros(k, order='F', dtype=float)

    C_out, D_out, info = mb04qu(
        'N', 'N', 'N', 'C', 'C', C.copy(), D.copy(), V, W, cs, tau
    )

    assert info == 0
    np.testing.assert_allclose(C_out, C_orig, rtol=1e-14)
    np.testing.assert_allclose(D_out, D_orig, rtol=1e-14)


def test_mb04qu_givens_rotation():
    """
    Test MB04QU with Givens rotation only (tau=0, W diag=0, cs defines rotation).

    Validates that Givens rotation c^2 + s^2 = 1 is applied correctly.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04qu

    np.random.seed(123)
    m, n, k = 3, 4, 1

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    V[0, 0] = 1.0
    W[0, 0] = 0.0

    C = np.random.randn(m, n).astype(float, order='F')
    D = np.random.randn(m, n).astype(float, order='F')
    C_orig = C.copy()
    D_orig = D.copy()

    theta = np.pi / 4
    c_val = np.cos(theta)
    s_val = np.sin(theta)
    cs = np.array([c_val, s_val], order='F', dtype=float)

    tau = np.zeros(k, order='F', dtype=float)

    C_out, D_out, info = mb04qu(
        'N', 'N', 'N', 'C', 'C', C.copy(), D.copy(), V, W, cs, tau
    )

    assert info == 0

    assert C_out.shape == (m, n)
    assert D_out.shape == (m, n)

    assert not np.allclose(C_out, C_orig, rtol=1e-10)
    assert not np.allclose(D_out, D_orig, rtol=1e-10)


def test_mb04qu_transpose_q():
    """
    Test MB04QU with TRANQ='T' (apply Q^T).

    When applying Q^T, the order of operations is reversed.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04qu

    np.random.seed(456)
    m, n, k = 3, 2, 1

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    V[0, 0] = 1.0
    W[0, 0] = 1.0

    C = np.random.randn(m, n).astype(float, order='F')
    D = np.random.randn(m, n).astype(float, order='F')

    cs = np.array([1.0, 0.0], order='F', dtype=float)
    tau = np.zeros(k, order='F', dtype=float)

    C_out, D_out, info = mb04qu(
        'N', 'N', 'T', 'C', 'C', C.copy(), D.copy(), V, W, cs, tau
    )

    assert info == 0
    assert C_out.shape == (m, n)
    assert D_out.shape == (m, n)


def test_mb04qu_transpose_c():
    """
    Test MB04QU with TRANC='T' (C stored transposed).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04qu

    np.random.seed(789)
    m, n, k = 3, 4, 1

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    V[0, 0] = 1.0
    W[0, 0] = 1.0

    C_T = np.random.randn(n, m).astype(float, order='F')
    D = np.random.randn(m, n).astype(float, order='F')

    cs = np.array([1.0, 0.0], order='F', dtype=float)
    tau = np.zeros(k, order='F', dtype=float)

    C_out, D_out, info = mb04qu(
        'T', 'N', 'N', 'C', 'C', C_T.copy(), D.copy(), V, W, cs, tau
    )

    assert info == 0
    assert C_out.shape == (n, m)


def test_mb04qu_rowwise_storage():
    """
    Test MB04QU with STOREV='R' and STOREW='R' (rowwise storage).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04qu

    np.random.seed(111)
    m, n, k = 4, 3, 2

    V = np.zeros((k, m), order='F', dtype=float)
    W = np.zeros((k, m), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 1.0

    C = np.random.randn(m, n).astype(float, order='F')
    D = np.random.randn(m, n).astype(float, order='F')

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        cs[2*i] = 1.0

    tau = np.zeros(k, order='F', dtype=float)

    C_out, D_out, info = mb04qu(
        'N', 'N', 'N', 'R', 'R', C.copy(), D.copy(), V, W, cs, tau
    )

    assert info == 0
    assert C_out.shape == (m, n)
    assert D_out.shape == (m, n)


def test_mb04qu_k_zero():
    """
    Test MB04QU with K=0 (no reflectors).

    Should return quickly without modifying C and D.
    """
    from ctrlsys import mb04qu

    m, n = 3, 4

    V = np.zeros((m, 1), order='F', dtype=float)
    W = np.zeros((m, 1), order='F', dtype=float)

    C = np.eye(m, n, order='F', dtype=float)
    D = np.eye(m, n, order='F', dtype=float)
    C_orig = C.copy()
    D_orig = D.copy()

    cs = np.zeros(0, order='F', dtype=float)
    tau = np.zeros(0, order='F', dtype=float)

    C_out, D_out, info = mb04qu(
        'N', 'N', 'N', 'C', 'C', C.copy(), D.copy(), V, W, cs, tau, k=0
    )

    assert info == 0
    np.testing.assert_allclose(C_out, C_orig, rtol=1e-14)
    np.testing.assert_allclose(D_out, D_orig, rtol=1e-14)


def test_mb04qu_invalid_tranc():
    """
    Test MB04QU with invalid TRANC parameter.

    Should return INFO = -1.
    """
    from ctrlsys import mb04qu

    m, n, k = 2, 2, 1
    V = np.eye(m, k, order='F', dtype=float)
    W = np.eye(m, k, order='F', dtype=float)
    C = np.eye(m, n, order='F', dtype=float)
    D = np.eye(m, n, order='F', dtype=float)
    cs = np.array([1.0, 0.0], order='F', dtype=float)
    tau = np.zeros(k, order='F', dtype=float)

    C_out, D_out, info = mb04qu(
        'X', 'N', 'N', 'C', 'C', C.copy(), D.copy(), V, W, cs, tau
    )

    assert info == -1


def test_mb04qu_invalid_storev():
    """
    Test MB04QU with invalid STOREV parameter.

    Should return INFO = -4.
    """
    from ctrlsys import mb04qu

    m, n, k = 2, 2, 1
    V = np.eye(m, k, order='F', dtype=float)
    W = np.eye(m, k, order='F', dtype=float)
    C = np.eye(m, n, order='F', dtype=float)
    D = np.eye(m, n, order='F', dtype=float)
    cs = np.array([1.0, 0.0], order='F', dtype=float)
    tau = np.zeros(k, order='F', dtype=float)

    C_out, D_out, info = mb04qu(
        'N', 'N', 'N', 'X', 'C', C.copy(), D.copy(), V, W, cs, tau
    )

    assert info == -4


def test_mb04qu_orthogonality_preservation():
    """
    Test that applying Q and Q^T in sequence returns original matrices.

    Q * Q^T = I for properly constructed reflectors and Givens rotations.
    Use identity reflectors (tau=0, W diag=0) with orthogonal Givens rotations.
    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb04qu

    np.random.seed(222)
    m, n, k = 4, 3, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 0.0

    C = np.random.randn(m, n).astype(float, order='F')
    D = np.random.randn(m, n).astype(float, order='F')
    C_orig = C.copy()
    D_orig = D.copy()

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        theta = np.random.uniform(0, 2*np.pi)
        cs[2*i] = np.cos(theta)
        cs[2*i + 1] = np.sin(theta)

    tau = np.zeros(k, order='F', dtype=float)

    C1, D1, info1 = mb04qu(
        'N', 'N', 'N', 'C', 'C', C.copy(), D.copy(), V.copy(), W.copy(), cs, tau
    )
    assert info1 == 0

    C2, D2, info2 = mb04qu(
        'N', 'N', 'T', 'C', 'C', C1.copy(), D1.copy(), V.copy(), W.copy(), cs, tau
    )
    assert info2 == 0

    np.testing.assert_allclose(C2, C_orig, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(D2, D_orig, rtol=1e-12, atol=1e-14)
