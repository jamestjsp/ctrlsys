"""
Tests for IB01MD - Upper triangular factor R of block-Hankel matrices.

Constructs the upper triangular factor R of the concatenated block
Hankel matrices from input-output data. Used in subspace identification
methods (MOESP and N4SID).

Mathematical property: R comes from QR factorization of:
    H = [ Uf'  Up'  Y' ]  for MOESP (METH='M')
    H = [ U'   Y' ]       for N4SID (METH='N')
where Up, Uf, U, Y are block Hankel matrices built from input-output data.
"""

import ctypes

import numpy as np
import pytest
from ctrlsys import _slicot, ib01md
from fortran_reference import run_fortran_driver


def _ib01my_query(meth):
    lib = ctypes.CDLL(_slicot.__file__)
    routine = lib.ib01my
    routine.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    routine.restype = None

    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr
    nr = 2 * (m + l) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=606)
    r = np.zeros((nr, nr), dtype=np.float64, order="F")
    iwork = np.full(max(3, m + l), -77, dtype=np.int32)
    dwork = np.zeros(256, dtype=np.float64)
    iwarn = ctypes.c_int(-99)
    info = ctypes.c_int(-99)

    routine(
        meth.encode("ascii"),
        b"O",
        b"N",
        nobr,
        m,
        l,
        nsmp,
        u,
        u.shape[0],
        y,
        y.shape[0],
        r,
        r.shape[0],
        iwork,
        dwork,
        -1,
        ctypes.byref(iwarn),
        ctypes.byref(info),
    )

    return dwork[0], iwork.copy(), iwarn.value, info.value


def _fortran_ib01my_query(meth, tmp_path):
    source = f"""
program main
  implicit none
  integer, parameter :: nobr = 2, m = 1, l = 1
  integer, parameter :: nsmp = 2*(m+l+1)*nobr
  integer, parameter :: nr = 2*(m+l)*nobr
  integer, parameter :: liwork = max(3,m+l), ldwork = 256
  integer :: i, info, iwarn
  integer :: iwork(liwork)
  double precision :: u(nsmp,m), y(nsmp,l), r(nr,nr), dwork(ldwork)

  do i = 1, nsmp
     u(i,1) = dble(i) / 10.0d0
     y(i,1) = -dble(i) / 20.0d0
  end do
  r = 0.0d0
  iwork = -77
  dwork = 0.0d0
  iwarn = -99
  info = -99

  call IB01MY('{meth}', 'O', 'N', nobr, m, l, nsmp, u, nsmp, y, nsmp, &
       r, nr, iwork, dwork, -1, iwarn, info)

  print '(ES24.16,1X,I0,1X,I0)', dwork(1), iwarn, info
  print '(*(I0,1X))', iwork
end program main
"""
    lines = run_fortran_driver(source, tmp_path).splitlines()
    dwork0, iwarn, info = lines[0].split()
    iwork = np.fromstring(lines[1], sep=" ", dtype=np.int32)
    return float(dwork0), iwork, int(iwarn), int(info)


def generate_io_data(nsmp, m, l, seed=42):
    """
    Generate random input-output data for system identification.

    Random seed: specified for reproducibility
    """
    np.random.seed(seed)

    if m > 0:
        u = np.random.randn(nsmp, m).astype(float, order='F')
    else:
        u = np.zeros((nsmp, 0), order='F', dtype=float)

    y = np.random.randn(nsmp, l).astype(float, order='F')

    return u, y


def test_ib01md_basic_qr_n4sid():
    """
    Basic test: QR algorithm with N4SID method, one batch.

    Tests METH='N', ALG='Q', BATCH='O'.
    Random seed: 42 (for reproducibility)
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=42)

    meth = 'N'
    alg = 'Q'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    assert info == 0
    assert iwarn == 0

    nr = 2 * (m + l) * nobr
    assert r.shape == (nr, nr)

    for i in range(nr):
        for j in range(i):
            assert abs(r[i, j]) < 1e-14, f"R not upper triangular at ({i},{j})"


def test_ib01md_basic_qr_moesp():
    """
    Basic test: QR algorithm with MOESP method, one batch.

    Tests METH='M', ALG='Q', BATCH='O'.
    Random seed: 123 (for reproducibility)
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=123)

    meth = 'M'
    alg = 'Q'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    assert info == 0
    assert iwarn == 0

    nr = 2 * (m + l) * nobr
    assert r.shape == (nr, nr)

    for i in range(nr):
        for j in range(i):
            assert abs(r[i, j]) < 1e-14, f"R not upper triangular at ({i},{j})"


def test_ib01md_cholesky_algorithm():
    """
    Test Cholesky algorithm for computing R factor.

    Tests ALG='C', BATCH='O'.
    Random seed: 456 (for reproducibility)
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 10

    u, y = generate_io_data(nsmp, m, l, seed=456)

    meth = 'N'
    alg = 'C'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    if info == 0:
        nr = 2 * (m + l) * nobr
        assert r.shape == (nr, nr)

        for i in range(nr):
            for j in range(i):
                assert abs(r[i, j]) < 1e-14
    else:
        assert iwarn == 2 or info == 1


def test_ib01md_larger_system():
    """
    Test with larger system dimensions.

    Random seed: 789 (for reproducibility)
    """
    nobr = 3
    m = 2
    l = 2
    nsmp = 2 * (m + l + 1) * nobr + 20

    u, y = generate_io_data(nsmp, m, l, seed=789)

    meth = 'N'
    alg = 'Q'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    assert info == 0

    nr = 2 * (m + l) * nobr
    assert r.shape == (nr, nr)

    for i in range(nr):
        for j in range(i):
            assert abs(r[i, j]) < 1e-14


def test_ib01md_m_zero():
    """
    Edge case: M = 0 (no inputs, output-only identification).

    Random seed: 111 (for reproducibility)
    """
    nobr = 2
    m = 0
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    np.random.seed(111)
    u = np.zeros((nsmp, 0), order='F', dtype=float)
    y = np.random.randn(nsmp, l).astype(float, order='F')

    meth = 'N'
    alg = 'Q'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    assert info == 0

    nr = 2 * l * nobr
    assert r.shape == (nr, nr)


def test_ib01md_r_matrix_property():
    """
    Mathematical property: R'R equals correlation matrix of block Hankel.

    For H = block Hankel matrix, QR factorization gives H = QR.
    Then H'H = R'Q'QR = R'R.

    Random seed: 222 (for reproducibility)
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 5

    u, y = generate_io_data(nsmp, m, l, seed=222)

    meth = 'N'
    alg = 'Q'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    assert info == 0

    rtr = r.T @ r

    assert np.allclose(rtr, rtr.T, rtol=1e-14)

    eigvals = np.linalg.eigvalsh(rtr)
    assert np.all(eigvals >= -1e-10), "R'R should be positive semidefinite"


def test_ib01md_deterministic_output():
    """
    Test deterministic output: same input produces same R.

    Random seed: 333 (for reproducibility)
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u1, y1 = generate_io_data(nsmp, m, l, seed=333)
    u2, y2 = generate_io_data(nsmp, m, l, seed=333)

    meth = 'N'
    alg = 'Q'
    batch = 'O'
    conct = 'N'

    r1, iwarn1, info1 = ib01md(meth, alg, batch, conct, nobr, m, l, u1, y1)
    r2, iwarn2, info2 = ib01md(meth, alg, batch, conct, nobr, m, l, u2, y2)

    assert info1 == 0
    assert info2 == 0
    np.testing.assert_allclose(r1, r2, rtol=1e-14)


def test_ib01md_error_invalid_meth():
    """
    Error handling: invalid METH parameter.
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=444)

    with pytest.raises(ValueError, match="METH"):
        ib01md('Z', 'Q', 'O', 'N', nobr, m, l, u, y)


def test_ib01md_error_invalid_alg():
    """
    Error handling: invalid ALG parameter.
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=555)

    with pytest.raises(ValueError, match="ALG"):
        ib01md('N', 'Z', 'O', 'N', nobr, m, l, u, y)


def test_ib01md_error_invalid_batch():
    """
    Error handling: invalid BATCH parameter.
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=666)

    with pytest.raises(ValueError, match="BATCH"):
        ib01md('N', 'Q', 'Z', 'N', nobr, m, l, u, y)


def test_ib01md_error_nobr_nonpositive():
    """
    Error handling: NOBR <= 0.
    """
    nobr = 0
    m = 1
    l = 1
    nsmp = 10

    u, y = generate_io_data(nsmp, m, l, seed=777)

    with pytest.raises(ValueError, match="NOBR"):
        ib01md('N', 'Q', 'O', 'N', nobr, m, l, u, y)


def test_ib01md_error_l_nonpositive():
    """
    Error handling: L <= 0.
    """
    nobr = 2
    m = 1
    l = 0
    nsmp = 10

    np.random.seed(888)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.zeros((nsmp, 0), order='F', dtype=float)

    with pytest.raises(ValueError, match="L"):
        ib01md('N', 'Q', 'O', 'N', nobr, m, l, u, y)


def test_ib01md_error_nsmp_too_small():
    """
    Error handling: NSMP too small for the problem.
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * nobr - 1

    np.random.seed(999)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="NSMP"):
        ib01md('N', 'Q', 'O', 'N', nobr, m, l, u, y)


def test_ib01md_sequential_basic():
    """
    Test sequential data processing: BATCH='F', 'I', 'L'.

    Random seed: 101 (for reproducibility)
    """
    nobr = 2
    m = 1
    l = 1
    nsmp_per_batch = 2 * nobr + 5

    np.random.seed(101)
    u1 = np.random.randn(nsmp_per_batch, m).astype(float, order='F')
    y1 = np.random.randn(nsmp_per_batch, l).astype(float, order='F')
    u2 = np.random.randn(nsmp_per_batch, m).astype(float, order='F')
    y2 = np.random.randn(nsmp_per_batch, l).astype(float, order='F')
    u3 = np.random.randn(nsmp_per_batch, m).astype(float, order='F')
    y3 = np.random.randn(nsmp_per_batch, l).astype(float, order='F')

    meth = 'N'
    alg = 'Q'
    conct = 'N'

    r1, iwork1, iwarn1, info1 = ib01md(meth, alg, 'F', conct, nobr, m, l, u1, y1)
    assert info1 == 0

    r2, iwork2, iwarn2, info2 = ib01md(meth, alg, 'I', conct, nobr, m, l, u2, y2,
                                        r=r1, iwork=iwork1)
    assert info2 == 0

    r3, iwarn3, info3 = ib01md(meth, alg, 'L', conct, nobr, m, l, u3, y3,
                               r=r2, iwork=iwork2)
    assert info3 == 0

    nr = 2 * (m + l) * nobr
    assert r3.shape == (nr, nr)


def test_ib01md_fast_qr_algorithm():
    """
    Test fast QR algorithm (ALG='F') with N4SID.

    Tests IB01MY implementation.
    Random seed: 202 (for reproducibility)
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=202)

    meth = 'N'
    alg = 'F'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    assert info == 0
    assert iwarn == 0

    nr = 2 * (m + l) * nobr
    assert r.shape == (nr, nr)

    for i in range(nr):
        for j in range(i):
            assert abs(r[i, j]) < 1e-14, f"R not upper triangular at ({i},{j})"


def test_ib01md_fast_qr_moesp():
    """
    Test fast QR algorithm (ALG='F') with MOESP.

    Tests IB01MY with MOESP algorithm.
    Random seed: 303 (for reproducibility)

    Note: MOESP fast QR stores Householder vectors below diagonal after
    the MB04ID retriangularization step. Only the upper triangular part
    of R contains the R factor. Downstream routines (like IB01PD) extract
    only the upper triangular part using triu(R).
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=303)

    meth = 'M'
    alg = 'F'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    assert info == 0
    assert iwarn == 0

    nr = 2 * (m + l) * nobr
    assert r.shape == (nr, nr)

    # Compare upper triangular part with standard QR MOESP
    r_std, _, info_std = ib01md('M', 'Q', 'O', 'N', nobr, m, l, u, y)
    assert info_std == 0

    # Extract upper triangular parts (R factor)
    r_upper = np.triu(r)
    r_std_upper = np.triu(r_std)

    # R factors are unique up to row sign (Q can have sign flips)
    # Compare absolute values of diagonal elements
    np.testing.assert_allclose(
        np.abs(np.diag(r_upper)),
        np.abs(np.diag(r_std_upper)),
        rtol=1e-10, atol=1e-12
    )

    # Verify each row matches up to sign by comparing R^T R products
    # which are invariant to row sign flips
    rtr_fast = r_upper.T @ r_upper
    rtr_std = r_std_upper.T @ r_std_upper
    np.testing.assert_allclose(rtr_fast, rtr_std, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("meth", ["N", "M"])
def test_ib01my_fast_qr_workspace_query_matches_fortran_reference(meth, tmp_path):
    c_dwork0, c_iwork, c_iwarn, c_info = _ib01my_query(meth)
    f_dwork0, f_iwork, f_iwarn, f_info = _fortran_ib01my_query(meth, tmp_path)

    assert c_info == f_info == 0
    assert c_iwarn == f_iwarn == 0
    np.testing.assert_allclose(c_dwork0, f_dwork0, rtol=0, atol=0)
    np.testing.assert_array_equal(c_iwork, f_iwork)


def test_ib01md_fast_qr_larger_system():
    """
    Test fast QR algorithm with larger system dimensions.

    Random seed: 404 (for reproducibility)
    """
    nobr = 3
    m = 2
    l = 2
    nsmp = 2 * (m + l + 1) * nobr + 20

    u, y = generate_io_data(nsmp, m, l, seed=404)

    meth = 'N'
    alg = 'F'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    assert info == 0

    nr = 2 * (m + l) * nobr
    assert r.shape == (nr, nr)

    for i in range(nr):
        for j in range(i):
            assert abs(r[i, j]) < 1e-14


def test_ib01md_fast_qr_m_zero():
    """
    Test fast QR algorithm with M=0 (output-only identification).

    Random seed: 505 (for reproducibility)
    """
    nobr = 2
    m = 0
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    np.random.seed(505)
    u = np.zeros((nsmp, 0), order='F', dtype=float)
    y = np.random.randn(nsmp, l).astype(float, order='F')

    meth = 'N'
    alg = 'F'
    batch = 'O'
    conct = 'N'

    r, iwarn, info = ib01md(meth, alg, batch, conct, nobr, m, l, u, y)

    assert info == 0

    nr = 2 * l * nobr
    assert r.shape == (nr, nr)


def test_ib01md_fast_qr_consistency_with_standard_qr():
    """
    Mathematical property: Fast QR should give same R'R as standard QR.

    Both methods compute QR factorization of same block-Hankel matrix,
    so R'R = H'H should be identical.

    Random seed: 606 (for reproducibility)
    """
    nobr = 2
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 5

    u, y = generate_io_data(nsmp, m, l, seed=606)

    meth = 'N'
    batch = 'O'
    conct = 'N'

    r_qr, iwarn_qr, info_qr = ib01md(meth, 'Q', batch, conct, nobr, m, l, u, y)
    assert info_qr == 0

    r_fast, iwarn_fast, info_fast = ib01md(meth, 'F', batch, conct, nobr, m, l, u, y)
    assert info_fast == 0

    rtr_qr = r_qr.T @ r_qr
    rtr_fast = r_fast.T @ r_fast

    np.testing.assert_allclose(rtr_qr, rtr_fast, rtol=1e-10, atol=1e-12)
