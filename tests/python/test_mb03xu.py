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

Uses numpy; Fortran-reference checks call the SLICOT library via ctypes.
"""

import ctypes

import numpy as np
import pytest

from fortran_reference import reference_cdll


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


@pytest.mark.parametrize("ltra,ltrb", [(False, False), (True, True), (True, False), (False, True)])
@pytest.mark.parametrize("n,k,nb", [(4, 0, 1), (8, 3, 3), (24, 16, 16)])
def test_mb03xu_matches_fortran_reference(ltra, ltrb, n, k, nb):
    from ctrlsys import mb03xu

    lib = reference_cdll()
    rng = np.random.default_rng(100 + n + k + nb)
    kn = k + n
    a = np.asfortranarray(rng.standard_normal((n, kn) if ltra else (kn, n)))
    b = np.asfortranarray(rng.standard_normal((kn, n) if ltrb else (n, kn)))
    g = np.asfortranarray(rng.standard_normal((kn, kn)))
    q = np.asfortranarray(rng.standard_normal((n, n)))

    ref = [x.copy(order="F") for x in (a, b, g, q)]
    lds = [n, kn, kn, n, kn, n, kn, n]
    work = [np.zeros((ld, 2 * nb), order="F") for ld in lds]
    vecs = [np.zeros(2 * nb), np.zeros(2 * nb), np.zeros(nb), np.zeros(nb)]

    def i(v):
        return ctypes.byref(ctypes.c_int(v))

    def p(x):
        return x.ctypes.data_as(ctypes.c_void_p)

    # MB03XU.f strides some B panels with LDQ; padding Q to LDB keeps the
    # reference correct when LDB > N.
    q_ref = np.zeros((b.shape[0], n), order="F")
    q_ref[:n] = ref[3]
    args = [i(int(ltra)), i(int(ltrb)), i(n), i(k), i(nb)]
    for x in ref[:3] + [q_ref]:
        args += [p(x), i(x.shape[0])]
    for x, ld in zip(work, lds):
        args += [p(x), i(ld)]
    args += [p(x) for x in vecs] + [p(np.zeros(5 * nb))]
    lib.mb03xu_(*args)
    ref[3] = q_ref[:n]

    out = mb03xu(ltra, ltrb, n, k, nb, a.copy(order="F"), b.copy(order="F"),
                 g.copy(order="F"), q.copy(order="F"))
    for got, want in zip(out[:16], ref + work + vecs):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)



@pytest.mark.parametrize("ltra,ltrb", [(False, False), (True, True), (True, False), (False, True)])
def test_mb03xu_independent_of_padded_ldb(ltra, ltrb):
    """B panels must use LDB (reference MB03XU.f wrongly passes LDQ)."""
    from ctrlsys import _slicot, mb03xu

    n, k, nb, pad = 10, 3, 4, 7
    kn = k + n
    rng = np.random.default_rng(31)
    a = np.asfortranarray(rng.standard_normal((n, kn) if ltra else (kn, n)))
    b = np.asfortranarray(rng.standard_normal((kn, n) if ltrb else (n, kn)))
    g = np.asfortranarray(rng.standard_normal((kn, kn)))
    q = np.asfortranarray(rng.standard_normal((n, n)))

    want = mb03xu(ltra, ltrb, n, k, nb, a.copy(order="F"), b.copy(order="F"),
                  g.copy(order="F"), q.copy(order="F"))

    ldb = b.shape[0] + pad
    b_pad = np.zeros((ldb, b.shape[1]), order="F")
    b_pad[:b.shape[0]] = b
    a_c, g_c, q_c = (x.copy(order="F") for x in (a, g, q))
    lds = [n, kn, kn, n, kn, n, kn, n]
    work = [np.zeros((ld, 2 * nb), order="F") for ld in lds]
    vecs = [np.zeros(2 * nb), np.zeros(2 * nb), np.zeros(nb), np.zeros(nb)]

    def p(x):
        return x.ctypes.data_as(ctypes.c_void_p)

    lib = ctypes.CDLL(_slicot.__file__)
    args = [ctypes.c_bool(ltra), ctypes.c_bool(ltrb), ctypes.c_int32(n),
            ctypes.c_int32(k), ctypes.c_int32(nb)]
    for x, ld in ((a_c, a.shape[0]), (b_pad, ldb), (g_c, kn), (q_c, n)):
        args += [p(x), ctypes.c_int32(ld)]
    for x, ld in zip(work, lds):
        args += [p(x), ctypes.c_int32(ld)]
    args += [p(x) for x in vecs] + [p(np.zeros(5 * nb))]
    lib.mb03xu.restype = None
    lib.mb03xu(*args)

    got = [a_c, b_pad[:b.shape[0]], g_c, q_c] + work + vecs
    for x, y in zip(got, want[:16]):
        np.testing.assert_allclose(x, y, rtol=1e-13, atol=1e-13)
