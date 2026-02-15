"""
Tests for MB04QC: Apply symplectic block reflector to matrices.

Applies the orthogonal symplectic block reflector Q or Q^T to a real
2m-by-n matrix [op(A); op(B)] from the left.

Uses numpy only - no scipy.
"""

import numpy as np


def test_mb04qc_basic():
    """
    Test MB04QC with identity-like block reflector.

    When R, S, T are zero matrices, the reflector is identity.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(42)
    m, n, k = 4, 3, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')
    A_orig = A.copy()
    B_orig = B.copy()

    A_copy = A.copy()
    B_copy = B.copy()
    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A_copy, B_copy
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)

    np.testing.assert_allclose(A_out, A_orig, atol=1e-12)
    np.testing.assert_allclose(B_out, B_orig, atol=1e-12)


def test_mb04qc_transpose_q():
    """
    Test MB04QC with TRANQ='T' (apply Q^T).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(123)
    m, n, k = 4, 3, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'T', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_transpose_a():
    """
    Test MB04QC with TRANA='T' (A stored transposed).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(456)
    m, n, k = 4, 3, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A_T = np.random.randn(n, m).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'T', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A_T.copy(), B.copy()
    )

    assert A_out.shape == (n, m)
    assert B_out.shape == (m, n)


def test_mb04qc_strab_n():
    """
    Test MB04QC with STRAB='N' (no zero structure).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(789)
    m, n, k = 5, 4, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i:, i] = np.random.randn(m - i)
        W[i:, i] = np.random.randn(m - i)

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'N', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_rowwise_storage():
    """
    Test MB04QC with STOREV='R' and STOREW='R'.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(111)
    m, n, k = 4, 3, 2

    V = np.zeros((k, m), order='F', dtype=float)
    W = np.zeros((k, m), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'R', 'R', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_k_equals_1():
    """
    Test MB04QC with K=1 (single reflector).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(222)
    m, n, k = 5, 4, 1

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    V[0, 0] = 1.0
    W[0, 0] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_m_equals_k():
    """
    Test MB04QC with M=K (degenerate case).

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(333)
    m, n, k = 3, 4, 3

    V = np.eye(m, k, order='F', dtype=float)
    W = np.eye(m, k, order='F', dtype=float)

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_n_equals_zero():
    """
    Test MB04QC with N=0 (quick return).
    """
    from ctrlsys import mb04qc

    m, n, k = 4, 0, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.zeros((m, 1), order='F', dtype=float)
    B = np.zeros((m, 1), order='F', dtype=float)

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == A.shape


def test_mb04qc_with_mb04qf():
    """
    Test MB04QC integration with MB04QF block factors.

    MB04QF computes the RS and T factors, MB04QC applies them.
    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb04qf, mb04qc

    np.random.seed(444)
    m, n, k = 5, 4, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i:, i] = np.random.randn(m - i)
        W[i:, i] = np.random.randn(m - i)
        W[i, i] = 0.0

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        theta = np.random.uniform(0, 2*np.pi)
        cs[2*i] = np.cos(theta)
        cs[2*i + 1] = np.sin(theta)

    tau = np.zeros(k, order='F', dtype=float)

    rs, t, info_qf = mb04qf('F', 'C', 'C', m, k, V.copy(), W.copy(), cs, tau)
    assert info_qf == 0

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')
    A_orig = A.copy()
    B_orig = B.copy()

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)

    A_round, B_round = mb04qc(
        'Z', 'N', 'N', 'T', 'F', 'C', 'C', m, n, k, V, W, rs, t, A_out.copy(), B_out.copy()
    )
    np.testing.assert_allclose(A_round, A_orig, atol=1e-12)
    np.testing.assert_allclose(B_round, B_orig, atol=1e-12)
