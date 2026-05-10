import ctypes

import numpy as np
import pytest

from ctrlsys import _slicot


def _mb01vd_call(trana, tranb, ma, na, mb, nb, alpha, beta, a, b, c):
    lib = ctypes.CDLL(_slicot.__file__)
    routine = lib.mb01vd
    routine.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    routine.restype = None

    mc = ma * mb
    lda = a.shape[0]
    ldb = b.shape[0]
    ldc = c.shape[0]
    mc_out = ctypes.c_int(-1)
    nc_out = ctypes.c_int(-1)
    info = ctypes.c_int(-99)

    routine(
        trana.encode("ascii"),
        tranb.encode("ascii"),
        ma,
        na,
        mb,
        nb,
        alpha,
        beta,
        a,
        lda,
        b,
        ldb,
        c,
        ldc,
        ctypes.byref(mc_out),
        ctypes.byref(nc_out),
        ctypes.byref(info),
    )

    return c, mc_out.value, nc_out.value, info.value


def _make_a(trana, ma, na, sparse):
    if trana == "N":
        shape = (ma, na)
    else:
        shape = (na, ma)
    values = np.arange(1, shape[0] * shape[1] + 1, dtype=np.float64).reshape(shape, order="F")
    if sparse:
        values.fill(0.0)
        values[0, 0] = 1.0
    return np.array(values, dtype=np.float64, order="F")


def _make_b(tranb, mb, nb):
    if tranb == "N":
        shape = (mb, nb)
    else:
        shape = (nb, mb)
    values = np.linspace(-2.0, 2.5, shape[0] * shape[1], dtype=np.float64).reshape(shape, order="F")
    return np.array(values, dtype=np.float64, order="F")


@pytest.mark.parametrize("trana,tranb", [
    ("N", "N"),
    ("T", "N"),
    ("N", "T"),
    ("T", "T"),
])
@pytest.mark.parametrize("sparse", [False, True])
def test_mb01vd_transpose_sparse_dense_parity(trana, tranb, sparse):
    ma, na = 3, 2
    mb, nb = 2, 3
    alpha = -1.25
    beta = 0.5

    a = _make_a(trana, ma, na, sparse)
    b = _make_b(tranb, mb, nb)
    mc = ma * mb
    nc = na * nb
    c_initial = np.linspace(0.25, 3.25, mc * nc, dtype=np.float64).reshape((mc, nc), order="F")
    c = np.array(c_initial, dtype=np.float64, order="F")

    op_a = a if trana == "N" else a.T
    op_b = b if tranb == "N" else b.T
    expected = alpha * np.kron(op_a, op_b) + beta * c_initial

    c_out, mc_out, nc_out, info = _mb01vd_call(
        trana, tranb, ma, na, mb, nb, alpha, beta, a, b, c
    )

    assert info == 0
    assert mc_out == mc
    assert nc_out == nc
    np.testing.assert_allclose(c_out, expected, rtol=1e-14, atol=1e-14)


def test_mb01vd_alpha_zero_scales_existing_output():
    ma, na = 3, 2
    mb, nb = 2, 3
    a = _make_a("N", ma, na, sparse=False)
    b = _make_b("N", mb, nb)
    c_initial = np.linspace(-1.0, 1.0, ma * mb * na * nb, dtype=np.float64).reshape(
        (ma * mb, na * nb), order="F"
    )
    c = np.array(c_initial, dtype=np.float64, order="F")

    c_out, mc_out, nc_out, info = _mb01vd_call(
        "N", "N", ma, na, mb, nb, 0.0, -0.25, a, b, c
    )

    assert info == 0
    assert mc_out == ma * mb
    assert nc_out == na * nb
    np.testing.assert_allclose(c_out, -0.25 * c_initial, rtol=0, atol=0)
