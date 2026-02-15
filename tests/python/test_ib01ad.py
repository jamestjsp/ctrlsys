"""
Tests for IB01AD - System identification driver.

IB01AD preprocesses input-output data for estimating state-space matrices
and finds an estimate of the system order using MOESP or N4SID method.
This driver calls IB01MD (R factor), IB01ND (SVD), IB01OD (order estimation).

Mathematical property: The estimated order n is determined by the number of
significant singular values from the SVD of the triangular factor R.
"""

import numpy as np
import os
import pytest
from ctrlsys import ib01ad


def load_html_example_data():
    """Load shared IB01 HTML example data from NPZ file."""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'ib01_html_example_data.npz')
    data = np.load(data_path)
    return data['u'], data['y'], int(data['nobr']), int(data['m']), int(data['l']), int(data['nsmp'])


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


def test_ib01ad_html_example():
    """
    Test using HTML documentation example.

    From IB01AD.html:
    NOBR=15, N=0 (auto-estimate), M=1, L=1, NSMP=1000
    RCOND=0.0, TOL=-1.0, METH='M', ALG='C', JOBD='N', BATCH='O', CONCT='N', CTRL='N'

    Expected output: N=4
    Singular values: 69.8841, 14.9963, 3.6675, 1.9677, 0.3000, ...

    Test validates numerical correctness of order estimation.
    """
    u, y, nobr, m, l, nsmp = load_html_example_data()

    meth = 'M'
    alg = 'C'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = -1.0

    n, r, sv, iwarn, info = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    assert info == 0
    assert n == 4

    sv_expected = np.array([69.8841, 14.9963, 3.6675, 1.9677, 0.3000, 0.2078,
                            0.1651, 0.1373, 0.1133, 0.1059, 0.0856, 0.0784,
                            0.0733, 0.0678, 0.0571])
    np.testing.assert_allclose(sv[:4], sv_expected[:4], rtol=1e-3, atol=1e-4)


def test_ib01ad_basic_n4sid():
    """
    Basic test: N4SID method with QR algorithm.

    Tests METH='N', ALG='Q', BATCH='O'.
    Random seed: 42 (for reproducibility)
    """
    nobr = 3
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=42)

    meth = 'N'
    alg = 'Q'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = 0.0

    n, r, sv, iwarn, info = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    assert info == 0
    assert iwarn == 0
    assert n >= 0
    assert n <= nobr

    lnobr = l * nobr
    assert len(sv) == lnobr
    for i in range(lnobr - 1):
        assert sv[i] >= sv[i + 1], f"SV not descending at {i}"


def test_ib01ad_basic_moesp():
    """
    Basic test: MOESP method with QR algorithm.

    Tests METH='M', ALG='Q', BATCH='O'.
    Random seed: 123 (for reproducibility)
    """
    nobr = 3
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=123)

    meth = 'M'
    alg = 'Q'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = 0.0

    n, r, sv, iwarn, info = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    assert info == 0
    assert n >= 0

    lnobr = l * nobr
    nr = 2 * (m + l) * nobr
    assert len(sv) == lnobr
    assert r.shape == (nr, nr)


def test_ib01ad_moesp_jobd_m():
    """
    MOESP with JOBD='M' (compute B,D using MOESP approach).

    Random seed: 456 (for reproducibility)
    """
    nobr = 4
    m = 2
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 20

    u, y = generate_io_data(nsmp, m, l, seed=456)

    meth = 'M'
    alg = 'Q'
    jobd = 'M'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = 0.0

    n, r, sv, iwarn, info = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    assert info == 0
    assert n >= 0


def test_ib01ad_larger_system():
    """
    Test with larger system dimensions.

    Random seed: 789 (for reproducibility)
    """
    nobr = 5
    m = 2
    l = 2
    nsmp = 2 * (m + l + 1) * nobr + 50

    u, y = generate_io_data(nsmp, m, l, seed=789)

    meth = 'N'
    alg = 'Q'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = -1.0

    n, r, sv, iwarn, info = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    assert info == 0

    lnobr = l * nobr
    nr = 2 * (m + l) * nobr
    assert len(sv) == lnobr
    assert r.shape == (nr, nr)


def test_ib01ad_m_zero():
    """
    Edge case: M = 0 (no inputs, output-only identification).

    Random seed: 111 (for reproducibility)
    """
    nobr = 3
    m = 0
    l = 2
    nsmp = 2 * (m + l + 1) * nobr

    np.random.seed(111)
    u = np.zeros((nsmp, 0), order='F', dtype=float)
    y = np.random.randn(nsmp, l).astype(float, order='F')

    meth = 'M'
    alg = 'Q'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = 0.0

    n, r, sv, iwarn, info = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    assert info == 0

    lnobr = l * nobr
    nr = 2 * l * nobr
    assert len(sv) == lnobr


def test_ib01ad_sv_descending_property():
    """
    Mathematical property: singular values must be in descending order.

    Random seed: 222 (for reproducibility)
    """
    nobr = 4
    m = 1
    l = 2
    nsmp = 2 * (m + l + 1) * nobr + 10

    u, y = generate_io_data(nsmp, m, l, seed=222)

    meth = 'N'
    alg = 'Q'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = 0.0

    n, r, sv, iwarn, info = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    assert info == 0

    for i in range(len(sv) - 1):
        assert sv[i] >= sv[i + 1], f"SV not descending: sv[{i}]={sv[i]} < sv[{i+1}]={sv[i+1]}"


def test_ib01ad_deterministic_output():
    """
    Test deterministic output: same input produces same results.

    Random seed: 333 (for reproducibility)
    """
    nobr = 3
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u1, y1 = generate_io_data(nsmp, m, l, seed=333)
    u2, y2 = generate_io_data(nsmp, m, l, seed=333)

    meth = 'N'
    alg = 'Q'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = 0.0

    n1, r1, sv1, iwarn1, info1 = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u1, y1, rcond, tol)
    n2, r2, sv2, iwarn2, info2 = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u2, y2, rcond, tol)

    assert info1 == 0
    assert info2 == 0
    assert n1 == n2
    np.testing.assert_allclose(sv1, sv2, rtol=1e-14)
    np.testing.assert_allclose(r1, r2, rtol=1e-14)


def test_ib01ad_error_invalid_meth():
    """
    Error handling: invalid METH parameter.
    """
    nobr = 3
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=444)

    with pytest.raises(ValueError, match="METH"):
        ib01ad('X', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)


def test_ib01ad_error_invalid_alg():
    """
    Error handling: invalid ALG parameter.
    """
    nobr = 3
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=555)

    with pytest.raises(ValueError, match="ALG"):
        ib01ad('N', 'X', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)


def test_ib01ad_error_invalid_batch():
    """
    Error handling: invalid BATCH parameter.
    """
    nobr = 3
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr

    u, y = generate_io_data(nsmp, m, l, seed=666)

    with pytest.raises(ValueError, match="BATCH"):
        ib01ad('N', 'Q', 'N', 'X', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)


def test_ib01ad_error_nobr_nonpositive():
    """
    Error handling: NOBR <= 0.
    """
    nobr = 0
    m = 1
    l = 1
    nsmp = 10

    u, y = generate_io_data(nsmp, m, l, seed=777)

    with pytest.raises(ValueError, match="NOBR"):
        ib01ad('N', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)


def test_ib01ad_error_l_nonpositive():
    """
    Error handling: L <= 0.
    """
    nobr = 3
    m = 1
    l = 0
    nsmp = 10

    np.random.seed(888)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.zeros((nsmp, 0), order='F', dtype=float)

    with pytest.raises(ValueError, match="L"):
        ib01ad('N', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)


def test_ib01ad_error_nsmp_too_small():
    """
    Error handling: NSMP too small for the problem.
    """
    nobr = 3
    m = 1
    l = 1
    nsmp = 2 * nobr - 1

    np.random.seed(999)
    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    with pytest.raises(ValueError, match="NSMP"):
        ib01ad('N', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)


def test_ib01ad_cholesky_algorithm():
    """
    Test with Cholesky algorithm (ALG='C').

    Random seed: 101 (for reproducibility)
    """
    nobr = 3
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 10

    u, y = generate_io_data(nsmp, m, l, seed=101)

    meth = 'N'
    alg = 'C'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = 0.0

    n, r, sv, iwarn, info = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    if info == 0:
        assert n >= 0
        assert len(sv) == l * nobr
    else:
        assert iwarn == 2 or info == 1


def test_ib01ad_fast_qr_algorithm():
    """
    Test with fast QR algorithm (ALG='F').

    Random seed: 202 (for reproducibility)
    """
    nobr = 3
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 10

    u, y = generate_io_data(nsmp, m, l, seed=202)

    meth = 'N'
    alg = 'F'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = 0.0

    n, r, sv, iwarn, info = ib01ad(meth, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    assert info == 0
    assert n >= 0
    assert len(sv) == l * nobr


def test_ib01ad_order_estimation_consistency():
    """
    Mathematical property: order estimation consistency between methods.

    For same data, different algorithms should give similar order estimates.
    Random seed: 303 (for reproducibility)
    """
    nobr = 4
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 20

    u, y = generate_io_data(nsmp, m, l, seed=303)

    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol = -1.0

    n_qr, _, sv_qr, _, info_qr = ib01ad('N', 'Q', jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)
    n_f, _, sv_f, _, info_f = ib01ad('N', 'F', jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol)

    assert info_qr == 0
    assert info_f == 0

    assert abs(n_qr - n_f) <= 1
