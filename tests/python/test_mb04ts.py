"""
Tests for MB04TS: Symplectic URV decomposition (unblocked version).

Computes H = U * R * V^T where:
- H is a 2N-by-2N Hamiltonian matrix with blocks [op(A) G; Q op(B)]
- U, V are 2N-by-2N orthogonal symplectic matrices
- R is block upper triangular: [op(R11) R12; 0 op(R22)]
  where op(R11) is upper triangular and op(R22) is lower Hessenberg

Uses numpy only - no scipy.
"""

import numpy as np


def test_mb04ts_basic():
    """
    Test MB04TS with data from HTML documentation example.

    Input: 5x5 matrices A, B, G, Q forming Hamiltonian structure.
    Validates decomposition H = U * R * V^T properties.
    """
    from ctrlsys import mb04ts

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

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
        'N', 'N', n, 1, A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    assert A_out.shape == (n, n)
    assert B_out.shape == (n, n)
    assert G_out.shape == (n, n)
    assert Q_out.shape == (n, n)
    assert csl.shape == (2*n,)
    assert csr.shape == (2*n - 2,) if n > 1 else (0,)
    assert taul.shape == (n,)
    assert taur.shape == (n - 1,) if n > 1 else (0,)

    np.testing.assert_allclose(abs(A_out[0, 0]), 3.0684, rtol=1e-3)
    np.testing.assert_allclose(abs(A_out[0, 1]), 4.6724, rtol=1e-3)


def test_mb04ts_small_2x2():
    """
    Test MB04TS with 2x2 matrices.

    Random seed: 42 (for reproducibility)
    Validates basic decomposition structure.
    """
    from ctrlsys import mb04ts

    np.random.seed(42)
    n = 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, n).astype(float, order='F')
    G = np.random.randn(n, n).astype(float, order='F')
    G = G + G.T
    Q = np.random.randn(n, n).astype(float, order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
        'N', 'N', n, 1, A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0
    assert A_out.shape == (n, n)
    assert B_out.shape == (n, n)
    assert csl.shape == (2*n,)
    assert csr.shape == (2*n - 2,)
    assert taul.shape == (n,)
    assert taur.shape == (n - 1,)


def test_mb04ts_transpose_a():
    """
    Test MB04TS with TRANA='T' (transposed A).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04ts

    np.random.seed(123)
    n = 3

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, n).astype(float, order='F')
    G = np.random.randn(n, n).astype(float, order='F')
    G = G + G.T
    Q = np.random.randn(n, n).astype(float, order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
        'T', 'N', n, 1, A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0
    assert A_out.shape == (n, n)


def test_mb04ts_transpose_b():
    """
    Test MB04TS with TRANB='T' (transposed B).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04ts

    np.random.seed(456)
    n = 3

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, n).astype(float, order='F')
    G = np.random.randn(n, n).astype(float, order='F')
    G = G + G.T
    Q = np.random.randn(n, n).astype(float, order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
        'N', 'T', n, 1, A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0
    assert B_out.shape == (n, n)


def test_mb04ts_n_equals_1():
    """
    Test MB04TS with N=1 edge case.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04ts

    np.random.seed(789)
    n = 1

    A = np.array([[2.5]], order='F', dtype=float)
    B = np.array([[-1.2]], order='F', dtype=float)
    G = np.array([[0.8]], order='F', dtype=float)
    Q = np.array([[0.5]], order='F', dtype=float)

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
        'N', 'N', n, 1, A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0
    assert A_out.shape == (1, 1)
    assert B_out.shape == (1, 1)
    assert csl.shape == (2,)
    assert taul.shape == (1,)


def test_mb04ts_ilo_parameter():
    """
    Test MB04TS with ILO > 1 (partial factorization).

    ILO indicates rows/cols 1:ILO-1 are already reduced.
    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb04ts

    np.random.seed(999)
    n = 4
    ilo = 2

    A = np.random.randn(n, n).astype(float, order='F')
    np.fill_diagonal(A, np.abs(np.diag(A)) + 1)
    B = np.random.randn(n, n).astype(float, order='F')
    G = np.random.randn(n, n).astype(float, order='F')
    G = G + G.T
    Q = np.random.randn(n, n).astype(float, order='F')

    Q[:ilo-1, :] = 0.0
    Q[:, :ilo-1] = 0.0

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
        'N', 'N', n, ilo, A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0


def test_mb04ts_invalid_trana():
    """
    Test MB04TS with invalid TRANA parameter.

    Should return INFO = -1.
    """
    from ctrlsys import mb04ts

    n = 2
    A = np.eye(n, order='F', dtype=float)
    B = np.eye(n, order='F', dtype=float)
    G = np.eye(n, order='F', dtype=float)
    Q = np.eye(n, order='F', dtype=float)

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
        'X', 'N', n, 1, A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == -1


def test_mb04ts_invalid_ilo():
    """
    Test MB04TS with invalid ILO parameter.

    Should return INFO = -4.
    """
    from ctrlsys import mb04ts

    n = 3
    A = np.eye(n, order='F', dtype=float)
    B = np.eye(n, order='F', dtype=float)
    G = np.eye(n, order='F', dtype=float)
    Q = np.eye(n, order='F', dtype=float)

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
        'N', 'N', n, 0, A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == -4


def test_mb04ts_givens_rotation_property():
    """
    Test that Givens rotation cosines/sines satisfy c^2 + s^2 = 1.

    This is a fundamental property of Givens rotations.
    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import mb04ts

    np.random.seed(555)
    n = 4

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, n).astype(float, order='F')
    G = np.random.randn(n, n).astype(float, order='F')
    G = G + G.T
    Q = np.random.randn(n, n).astype(float, order='F')

    A_out, B_out, G_out, Q_out, csl, csr, taul, taur, info = mb04ts(
        'N', 'N', n, 1, A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    for i in range(n):
        c = csl[2*i]
        s = csl[2*i + 1]
        np.testing.assert_allclose(c*c + s*s, 1.0, rtol=1e-14,
            err_msg=f"CSL Givens rotation {i}: c^2+s^2 = {c*c + s*s}")

    for i in range(n - 1):
        c = csr[2*i]
        s = csr[2*i + 1]
        np.testing.assert_allclose(c*c + s*s, 1.0, rtol=1e-14,
            err_msg=f"CSR Givens rotation {i}: c^2+s^2 = {c*c + s*s}")
