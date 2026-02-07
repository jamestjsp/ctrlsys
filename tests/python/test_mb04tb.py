"""
Tests for MB04TB: Symplectic URV decomposition (blocked version).

MB04TB computes H = U * R * V^T where H is a 2N-by-2N Hamiltonian matrix.

Uses numpy only - no scipy.
"""

import numpy as np
from slicot import mb04tb, mb04wr


def test_mb04tb_n5_basic():
    """
    Test MB04TB with N=5 using HTML doc example data.

    Test data from SLICOT HTML documentation.
    """
    n = 5

    A = np.array([
        [0.4643, 0.3655, 0.6853, 0.5090, 0.3718],
        [0.3688, 0.6460, 0.4227, 0.6798, 0.5135],
        [0.7458, 0.5043, 0.9419, 0.9717, 0.9990],
        [0.7140, 0.4941, 0.7802, 0.5272, 0.1220],
        [0.7418, 0.0339, 0.7441, 0.0436, 0.6564]
    ], order='F', dtype=float)

    B = np.array([
        [-0.4643, -0.3688, -0.7458, -0.7140, -0.7418],
        [-0.3655, -0.6460, -0.5043, -0.4941, -0.0339],
        [-0.6853, -0.4227, -0.9419, -0.7802, -0.7441],
        [-0.5090, -0.6798, -0.9717, -0.5272, -0.0436],
        [-0.3718, -0.5135, -0.9990, -0.1220, -0.6564]
    ], order='F', dtype=float)

    G = np.array([
        [0.7933, 1.5765, 1.0711, 1.0794, 0.8481],
        [1.5765, 0.1167, 1.5685, 0.8756, 0.5037],
        [1.0711, 1.5685, 0.9902, 0.3858, 0.2109],
        [1.0794, 0.8756, 0.3858, 1.8834, 1.4338],
        [0.8481, 0.5037, 0.2109, 1.4338, 0.1439]
    ], order='F', dtype=float)

    Q = np.array([
        [1.0786, 1.5264, 1.1721, 1.5343, 0.4756],
        [1.5264, 0.8644, 0.6872, 1.1379, 0.6499],
        [1.1721, 0.6872, 1.5194, 1.1197, 1.0158],
        [1.5343, 1.1379, 1.1197, 0.6612, 0.2004],
        [0.4756, 0.6499, 1.0158, 0.2004, 1.2188]
    ], order='F', dtype=float)

    A_orig = A.copy()
    B_orig = B.copy()
    G_orig = G.copy()
    Q_orig = Q.copy()
    ilo = 1

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'N', 'N', n, ilo, A, B, G, Q
    )

    assert info == 0

    assert A_out.shape == (n, n)
    assert B_out.shape == (n, n)
    assert G_out.shape == (n, n)
    assert Q_out.shape == (n, n)

    q1_u = A_out.copy()
    q2_u = Q_out.copy()
    u1, u2, info_u = mb04wr('U', 'N', n, ilo, q1_u, q2_u, csl, taul)
    assert info_u == 0

    q1_v = B_out.copy()
    q2_v = Q_out.copy()
    v1, v2, info_v = mb04wr('V', 'N', n, ilo, q1_v, q2_v, csr, taur)
    assert info_v == 0

    U = np.block([[u1, u2], [-u2, u1]])
    V = np.block([[v1.T, v2.T], [-v2.T, v1.T]])

    np.testing.assert_allclose(U.T @ U, np.eye(2*n), atol=1e-12)
    np.testing.assert_allclose(V.T @ V, np.eye(2*n), atol=1e-12)

    R11 = np.triu(A_out)
    R12 = G_out.copy()
    R22_hess = np.tril(B_out, 1)
    Q_zero = np.zeros((n, n))
    R = np.block([[R11, R12], [Q_zero, R22_hess]])

    H = np.block([[A_orig, G_orig], [Q_orig, B_orig]])
    residual = np.linalg.norm(H @ V - U @ R, 'fro')
    h_norm = np.linalg.norm(H, 'fro')
    assert residual / h_norm < 1e-12


def test_mb04tb_quick_return_n_zero():
    """Test quick return when N=0."""
    from slicot import mb04tb

    n = 0
    ilo = 1

    A = np.zeros((1, 1), order='F')
    B = np.zeros((1, 1), order='F')
    G = np.zeros((1, 1), order='F')
    Q = np.zeros((1, 1), order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'N', 'N', n, ilo, A, B, G, Q
    )

    assert info == 0


def test_mb04tb_quick_return_nh_zero():
    """Test quick return when NH=0 (ilo=n+1)."""
    from slicot import mb04tb

    n = 3
    ilo = n + 1

    A = np.eye(n, order='F')
    B = np.eye(n, order='F')
    G = np.zeros((n, n), order='F')
    Q = np.zeros((n, n), order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'N', 'N', n, ilo, A, B, G, Q
    )

    assert info == 0


def test_mb04tb_workspace_query():
    """Test workspace query mode (ldwork=-1)."""
    from slicot import mb04tb

    n = 5
    ilo = 1

    A = np.eye(n, order='F')
    B = np.eye(n, order='F')
    G = np.zeros((n, n), order='F')
    Q = np.zeros((n, n), order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'N', 'N', n, ilo, A, B, G, Q, ldwork=-1
    )

    assert info == 0


def test_mb04tb_invalid_trana():
    """Test error for invalid TRANA parameter."""
    from slicot import mb04tb

    n = 3
    ilo = 1

    A = np.eye(n, order='F')
    B = np.eye(n, order='F')
    G = np.zeros((n, n), order='F')
    Q = np.zeros((n, n), order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'X', 'N', n, ilo, A, B, G, Q
    )

    assert info == -1


def test_mb04tb_invalid_tranb():
    """Test error for invalid TRANB parameter."""
    from slicot import mb04tb

    n = 3
    ilo = 1

    A = np.eye(n, order='F')
    B = np.eye(n, order='F')
    G = np.zeros((n, n), order='F')
    Q = np.zeros((n, n), order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'N', 'X', n, ilo, A, B, G, Q
    )

    assert info == -2


def test_mb04tb_negative_n():
    """Test error for negative N."""
    from slicot import mb04tb

    n = -1
    ilo = 1

    A = np.zeros((1, 1), order='F')
    B = np.zeros((1, 1), order='F')
    G = np.zeros((1, 1), order='F')
    Q = np.zeros((1, 1), order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'N', 'N', n, ilo, A, B, G, Q
    )

    assert info == -3


def test_mb04tb_invalid_ilo():
    """Test error for invalid ILO."""
    from slicot import mb04tb

    n = 4
    ilo = 0

    A = np.eye(n, order='F')
    B = np.eye(n, order='F')
    G = np.zeros((n, n), order='F')
    Q = np.zeros((n, n), order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'N', 'N', n, ilo, A, B, G, Q
    )

    assert info == -4
