"""
Tests for MB03XU: Panel reduction for blocked Hamiltonian matrix.

MB03XU is an auxiliary routine called by MB04TB that reduces 2*NB columns
and rows of a (K+2N)-by-(K+2N) Hamiltonian matrix H using orthogonal
symplectic transformations:

    H = [op(A)   G  ]
        [  Q   op(B)]

Returns update matrices XA, XB, XG, XQ, YA, YB, YG, YQ such that:

    UU' * H * VV = [op(Aout)+U*YA'+XA*V'     G+U*YG'+XG*V'    ]
                   [  Qout+U*YQ'+XQ*V'   op(Bout)+U*YB'+XB*V' ]

Uses numpy only - no scipy.
"""

import numpy as np


def test_mb03xu_basic_ltra_false_ltrb_false():
    """
    Test MB03XU with LTRA=False, LTRB=False (no transpose).

    Random seed: 42 (for reproducibility)
    Uses small matrices to validate basic functionality.
    """
    from ctrlsys import mb03xu

    np.random.seed(42)
    n = 5
    k = 2
    nb = 2

    A = np.random.randn(k + n, n).astype(float, order='F')
    B = np.random.randn(n, k + n).astype(float, order='F')
    G = np.random.randn(k + n, k + n).astype(float, order='F')
    G = (G + G.T) / 2
    Q = np.random.randn(n, n).astype(float, order='F')

    (A_out, B_out, G_out, Q_out,
     XA, XB, XG, XQ, YA, YB, YG, YQ,
     CSL, CSR, TAUL, TAUR, info) = mb03xu(
        False, False, n, k, nb,
        A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    assert A_out.shape == (k + n, n)
    assert B_out.shape == (n, k + n)
    assert G_out.shape == (k + n, k + n)
    assert Q_out.shape == (n, n)

    assert XA.shape == (n, 2*nb)
    assert XB.shape == (k + n, 2*nb)
    assert XG.shape == (k + n, 2*nb)
    assert XQ.shape == (n, 2*nb)

    assert YA.shape == (k + n, 2*nb)
    assert YB.shape == (n, 2*nb)
    assert YG.shape == (k + n, 2*nb)
    assert YQ.shape == (n, 2*nb)

    assert CSL.shape == (2*nb,)
    assert CSR.shape == (2*nb,)
    assert TAUL.shape == (nb,)
    assert TAUR.shape == (nb,)


def test_mb03xu_ltra_true_ltrb_true():
    """
    Test MB03XU with LTRA=True, LTRB=True (both transposed).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03xu

    np.random.seed(123)
    n = 4
    k = 1
    nb = 2

    A = np.random.randn(n, k + n).astype(float, order='F')
    B = np.random.randn(k + n, n).astype(float, order='F')
    G = np.random.randn(k + n, k + n).astype(float, order='F')
    G = (G + G.T) / 2
    Q = np.random.randn(n, n).astype(float, order='F')

    (A_out, B_out, G_out, Q_out,
     XA, XB, XG, XQ, YA, YB, YG, YQ,
     CSL, CSR, TAUL, TAUR, info) = mb03xu(
        True, True, n, k, nb,
        A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    assert A_out.shape == (n, k + n)
    assert B_out.shape == (k + n, n)


def test_mb03xu_ltra_true_ltrb_false():
    """
    Test MB03XU with LTRA=True, LTRB=False (A transposed only).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03xu

    np.random.seed(456)
    n = 4
    k = 2
    nb = 2

    A = np.random.randn(n, k + n).astype(float, order='F')
    B = np.random.randn(n, k + n).astype(float, order='F')
    G = np.random.randn(k + n, k + n).astype(float, order='F')
    G = (G + G.T) / 2
    Q = np.random.randn(n, n).astype(float, order='F')

    (A_out, B_out, G_out, Q_out,
     XA, XB, XG, XQ, YA, YB, YG, YQ,
     CSL, CSR, TAUL, TAUR, info) = mb03xu(
        True, False, n, k, nb,
        A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    assert A_out.shape == (n, k + n)
    assert B_out.shape == (n, k + n)


def test_mb03xu_ltra_false_ltrb_true():
    """
    Test MB03XU with LTRA=False, LTRB=True (B transposed only).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03xu

    np.random.seed(789)
    n = 4
    k = 1
    nb = 2

    A = np.random.randn(k + n, n).astype(float, order='F')
    B = np.random.randn(k + n, n).astype(float, order='F')
    G = np.random.randn(k + n, k + n).astype(float, order='F')
    G = (G + G.T) / 2
    Q = np.random.randn(n, n).astype(float, order='F')

    (A_out, B_out, G_out, Q_out,
     XA, XB, XG, XQ, YA, YB, YG, YQ,
     CSL, CSR, TAUL, TAUR, info) = mb03xu(
        False, True, n, k, nb,
        A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    assert A_out.shape == (k + n, n)
    assert B_out.shape == (k + n, n)


def test_mb03xu_nb_equals_1():
    """
    Test MB03XU with NB=1 (minimum block size).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb03xu

    np.random.seed(111)
    n = 4
    k = 1
    nb = 1

    A = np.random.randn(k + n, n).astype(float, order='F')
    B = np.random.randn(n, k + n).astype(float, order='F')
    G = np.random.randn(k + n, k + n).astype(float, order='F')
    G = (G + G.T) / 2
    Q = np.random.randn(n, n).astype(float, order='F')

    (A_out, B_out, G_out, Q_out,
     XA, XB, XG, XQ, YA, YB, YG, YQ,
     CSL, CSR, TAUL, TAUR, info) = mb03xu(
        False, False, n, k, nb,
        A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    assert XA.shape == (n, 2*nb)
    assert CSL.shape == (2*nb,)


def test_mb03xu_k_equals_0():
    """
    Test MB03XU with K=0 (no offset).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb03xu

    np.random.seed(222)
    n = 5
    k = 0
    nb = 2

    A = np.random.randn(k + n, n).astype(float, order='F')
    B = np.random.randn(n, k + n).astype(float, order='F')
    G = np.random.randn(k + n, k + n).astype(float, order='F')
    G = (G + G.T) / 2
    Q = np.random.randn(n, n).astype(float, order='F')

    (A_out, B_out, G_out, Q_out,
     XA, XB, XG, XQ, YA, YB, YG, YQ,
     CSL, CSR, TAUL, TAUR, info) = mb03xu(
        False, False, n, k, nb,
        A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    assert A_out.shape == (n, n)
    assert B_out.shape == (n, n)


def test_mb03xu_givens_rotation_property():
    """
    Test that Givens rotation cosines/sines satisfy c^2 + s^2 = 1.

    Fundamental property of Givens rotations.
    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb03xu

    np.random.seed(333)
    n = 6
    k = 2
    nb = 3

    A = np.random.randn(k + n, n).astype(float, order='F')
    B = np.random.randn(n, k + n).astype(float, order='F')
    G = np.random.randn(k + n, k + n).astype(float, order='F')
    G = (G + G.T) / 2
    Q = np.random.randn(n, n).astype(float, order='F')

    (A_out, B_out, G_out, Q_out,
     XA, XB, XG, XQ, YA, YB, YG, YQ,
     CSL, CSR, TAUL, TAUR, info) = mb03xu(
        False, False, n, k, nb,
        A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    for i in range(nb):
        c = CSL[2*i]
        s = CSL[2*i + 1]
        np.testing.assert_allclose(c*c + s*s, 1.0, rtol=1e-14,
            err_msg=f"CSL Givens rotation {i}: c^2+s^2 = {c*c + s*s}")

    for i in range(nb - 1):
        c = CSR[2*i]
        s = CSR[2*i + 1]
        np.testing.assert_allclose(c*c + s*s, 1.0, rtol=1e-14,
            err_msg=f"CSR Givens rotation {i}: c^2+s^2 = {c*c + s*s}")


def test_mb03xu_taul_reflector_property():
    """
    Test that Householder reflector tau values are valid (0 <= tau <= 2).

    For elementary reflectors H = I - tau * v * v', tau must be in [0, 2].
    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb03xu

    np.random.seed(444)
    n = 5
    k = 2
    nb = 2

    A = np.random.randn(k + n, n).astype(float, order='F')
    B = np.random.randn(n, k + n).astype(float, order='F')
    G = np.random.randn(k + n, k + n).astype(float, order='F')
    G = (G + G.T) / 2
    Q = np.random.randn(n, n).astype(float, order='F')

    (A_out, B_out, G_out, Q_out,
     XA, XB, XG, XQ, YA, YB, YG, YQ,
     CSL, CSR, TAUL, TAUR, info) = mb03xu(
        False, False, n, k, nb,
        A.copy(), B.copy(), G.copy(), Q.copy()
    )

    assert info == 0

    for i in range(nb):
        assert 0.0 <= TAUL[i] <= 2.0 + 1e-10, f"TAUL[{i}] = {TAUL[i]} out of range"

    for i in range(nb - 1):
        assert 0.0 <= TAUR[i] <= 2.0 + 1e-10, f"TAUR[{i}] = {TAUR[i]} out of range"
