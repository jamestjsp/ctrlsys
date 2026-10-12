"""
Tests for MB04TB: Symplectic URV decomposition (blocked version).

MB04TB computes H = U * R * V^T where H is a 2N-by-2N Hamiltonian matrix.

Uses numpy; Fortran-reference checks call the SLICOT library via ctypes.
"""

import ctypes

import numpy as np
import pytest
from ctrlsys import mb04tb, mb04ts, mb04wr
from fortran_reference import reference_cdll


def _fortran_mb04tb(trana, tranb, n, ilo, a, b, g, q, ldwork):
    lib = reference_cdll()
    a, b, g, q = (np.array(x, dtype=float, order="F") for x in (a, b, g, q))
    csl = np.zeros(2 * n)
    csr = np.zeros(2 * n)
    taul = np.zeros(n)
    taur = np.zeros(n)
    dwork = np.zeros(ldwork)
    info = ctypes.c_int(0)

    def i(v):
        return ctypes.byref(ctypes.c_int(v))

    def p(x):
        return x.ctypes.data_as(ctypes.c_void_p)

    lib.mb04tb_(
        trana.encode(), tranb.encode(), i(n), i(ilo),
        p(a), i(n), p(b), i(n), p(g), i(n), p(q), i(n),
        p(csl), p(csr), p(taul), p(taur), p(dwork), i(ldwork),
        ctypes.byref(info), ctypes.c_size_t(1), ctypes.c_size_t(1),
    )
    return a, b, g, q, csl, csr[:2 * n - 2], taul, taur[:n - 1], info.value


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
    from ctrlsys import mb04tb

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
    from ctrlsys import mb04tb

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
    from ctrlsys import mb04tb

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
    from ctrlsys import mb04tb

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
    from ctrlsys import mb04tb

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
    from ctrlsys import mb04tb

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
    from ctrlsys import mb04tb

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


def test_mb04tb_n20_blocked():
    """
    Test MB04TB with N=20 to exercise blocked algorithm path.

    Validates H = U * R * V^T decomposition with mathematical residual check.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 20
    ilo = 1

    A = np.random.randn(n, n).astype(float, order='F')
    G_upper = np.random.randn(n, n)
    G = (G_upper + G_upper.T).astype(float, order='F')
    Q_upper = np.random.randn(n, n)
    Q = (Q_upper + Q_upper.T).astype(float, order='F')
    B = -A.T.copy(order='F')

    A_orig = A.copy()
    B_orig = B.copy()
    G_orig = G.copy()
    Q_orig = Q.copy()

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'N', 'N', n, ilo, A, B, G, Q
    )

    assert info == 0

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

    np.testing.assert_allclose(U.T @ U, np.eye(2*n), atol=1e-10)
    np.testing.assert_allclose(V.T @ V, np.eye(2*n), atol=1e-10)

    R11 = np.triu(A_out)
    R12 = G_out.copy()
    R22_hess = np.tril(B_out, 1)
    Q_zero = np.zeros((n, n))
    R = np.block([[R11, R12], [Q_zero, R22_hess]])

    H = np.block([[A_orig, G_orig], [Q_orig, B_orig]])
    residual = np.linalg.norm(H @ V - U @ R, 'fro')
    h_norm = np.linalg.norm(H, 'fro')
    assert residual / h_norm < 1e-10


@pytest.mark.parametrize("trana,tranb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
def test_mb04tb_blocked_path_matches_unblocked_mb04ts(trana, tranb):
    """Blocked MB03XU panels must reproduce the unblocked MB04TS reduction."""
    n, ilo, ldwork = 200, 2, 30000
    rng = np.random.default_rng(7)
    a, b, g, q = (np.asfortranarray(rng.standard_normal((n, n))) for _ in range(4))

    blk = mb04tb(trana, tranb, n, ilo, a.copy(order="F"), b.copy(order="F"),
                 g.copy(order="F"), q.copy(order="F"), ldwork=ldwork)
    unb = mb04ts(trana, tranb, n, ilo, a.copy(order="F"), b.copy(order="F"),
                 g.copy(order="F"), q.copy(order="F"))

    assert blk[-1] == 0
    assert unb[-1] == 0
    for got, want in zip(blk[:4], unb[:4]):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)
    np.testing.assert_allclose(blk[6][ilo - 1:], unb[6][ilo - 1:], rtol=0, atol=1e-10)
    np.testing.assert_allclose(blk[7][ilo - 1:n - 1], unb[7][ilo - 1:n - 1], rtol=0, atol=1e-10)


@pytest.mark.parametrize("trana,tranb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
def test_mb04tb_blocked_path_matches_fortran_reference(trana, tranb):
    """
    n=200, ilo=2, ldwork=30000 forces NB = ldwork/(16n+5) = 9 < NX, so the
    MB03XU panel loop runs. Inputs are passed bit-identical to the reference.
    """
    n, ilo, ldwork = 200, 2, 30000
    rng = np.random.default_rng(7)
    a, b, g, q = (np.asfortranarray(rng.standard_normal((n, n))) for _ in range(4))

    ref = _fortran_mb04tb(trana, tranb, n, ilo, a, b, g, q, ldwork)
    out = mb04tb(trana, tranb, n, ilo, a.copy(order="F"), b.copy(order="F"),
                 g.copy(order="F"), q.copy(order="F"), ldwork=ldwork)

    assert ref[-1] == 0
    assert out[-1] == 0
    for got, want in zip(out[:4], ref[:4]):
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(out[4], ref[4], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(out[5][:2 * n - 2], ref[5], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(out[6], ref[6], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(out[7][:n - 1], ref[7], rtol=1e-10, atol=1e-10)


def test_mb04tb_hamiltonian_property():
    """
    Validate Hamiltonian structure: J * H is symmetric where J = [[0, I], [-I, 0]].

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 8

    A = np.random.randn(n, n).astype(float, order='F')
    G_upper = np.random.randn(n, n)
    G = (G_upper + G_upper.T).astype(float, order='F')
    Q_upper = np.random.randn(n, n)
    Q = (Q_upper + Q_upper.T).astype(float, order='F')
    B = -A.T.copy(order='F')

    H = np.block([[A, G], [Q, B]])
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    JH = J @ H
    np.testing.assert_allclose(JH, JH.T, atol=1e-14)

    A_orig = A.copy()
    B_orig = B.copy()
    G_orig = G.copy()
    Q_orig = Q.copy()

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04tb(
        'N', 'N', n, 1, A, B, G, Q
    )
    assert info == 0

    q1_u = A_out.copy()
    q2_u = Q_out.copy()
    u1, u2, info_u = mb04wr('U', 'N', n, 1, q1_u, q2_u, csl, taul)
    assert info_u == 0

    q1_v = B_out.copy()
    q2_v = Q_out.copy()
    v1, v2, info_v = mb04wr('V', 'N', n, 1, q1_v, q2_v, csr, taur)
    assert info_v == 0

    U = np.block([[u1, u2], [-u2, u1]])
    V = np.block([[v1.T, v2.T], [-v2.T, v1.T]])

    R11 = np.triu(A_out)
    R12 = G_out.copy()
    R22_hess = np.tril(B_out, 1)
    Q_zero = np.zeros((n, n))
    R = np.block([[R11, R12], [Q_zero, R22_hess]])

    H_orig = np.block([[A_orig, G_orig], [Q_orig, B_orig]])
    residual = np.linalg.norm(H_orig @ V - U @ R, 'fro')
    h_norm = np.linalg.norm(H_orig, 'fro')
    assert residual / h_norm < 1e-12
