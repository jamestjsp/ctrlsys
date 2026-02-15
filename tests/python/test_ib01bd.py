"""
Tests for IB01BD - System matrices estimation from N4SID/MOESP.

IB01BD estimates the system matrices A, C, B, D, the noise covariance
matrices Q, Ry, S, and the Kalman gain matrix K of a linear time-invariant
state-space model from the processed triangular factor R provided by IB01AD.

Mathematical properties tested:
- State equation: x(k+1) = A*x(k) + B*u(k)
- Output equation: y(k) = C*x(k) + D*u(k)
- Covariance symmetry: Q = Q^T, Ry = Ry^T
- Covariance positive semidefiniteness
"""

import numpy as np
import os
import pytest
from ctrlsys import ib01ad, ib01bd


def load_html_example_data():
    """Load shared IB01 HTML example data from NPZ file."""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'ib01_html_example_data.npz')
    data = np.load(data_path)
    return data['u'], data['y'], int(data['nobr']), int(data['m']), int(data['l']), int(data['nsmp'])


def test_ib01bd_html_example():
    """
    Test using HTML documentation example.

    From IB01BD.html:
    NOBR=15, N=4 (given), M=1, L=1, NSMP=1000
    METH='C', JOB='A', JOBCK='K', RCOND=0.0, TOL=-1.0

    Expected outputs:
    A = [[0.8924, 0.3887, 0.1285, 0.1716],
         [-0.0837, 0.6186, -0.6273, -0.4582],
         [0.0052, 0.1307, 0.6685, -0.6755],
         [0.0055, 0.0734, -0.2148, 0.4788]]

    C = [-0.4442, 0.6663, 0.3961, 0.4102]

    B = [-0.2142, -0.1968, 0.0525, 0.0361]^T

    D = [-0.0041]

    K = [-1.9513, -0.1867, 0.6348, -0.3486]^T
    """
    u, y, nobr, m, l, nsmp = load_html_example_data()
    n = 4

    meth_ib01ad = 'M'
    alg = 'C'
    jobd = 'N'
    batch = 'O'
    conct = 'N'
    ctrl = 'N'
    rcond = 0.0
    tol_ad = -1.0

    _n_est, r, _sv, _iwarn_ad, info_ad = ib01ad(meth_ib01ad, alg, jobd, batch, conct, ctrl, nobr, m, l, u, y, rcond, tol_ad)
    assert info_ad == 0

    # Use Combined method (MOESP for A,C then N4SID for B,D) with Kalman gain
    meth = 'C'
    job = 'A'
    jobck = 'K'
    tol = 0.0

    A, C, B, D, Q, Ry, S, K, iwarn, info = ib01bd(meth, job, jobck, nobr, n, m, l, nsmp, r, tol)

    assert info == 0

    # Expected values from IB01BD.html documentation
    A_expected = np.array([
        [0.8924, 0.3887, 0.1285, 0.1716],
        [-0.0837, 0.6186, -0.6273, -0.4582],
        [0.0052, 0.1307, 0.6685, -0.6755],
        [0.0055, 0.0734, -0.2148, 0.4788]
    ], order='F', dtype=float)

    C_expected = np.array([[-0.4442, 0.6663, 0.3961, 0.4102]], order='F', dtype=float)

    B_expected = np.array([[-0.2142], [-0.1968], [0.0525], [0.0361]], order='F', dtype=float)

    D_expected = np.array([[-0.0041]], order='F', dtype=float)

    K_expected = np.array([[-1.9513], [-0.1867], [0.6348], [-0.3486]], order='F', dtype=float)

    # Expected covariance matrices from IB01BD.html documentation
    Q_expected = np.array([
        [0.0052, 0.0005, -0.0017, 0.0009],
        [0.0005, 0.0000, -0.0002, 0.0001],
        [-0.0017, -0.0002, 0.0006, -0.0003],
        [0.0009, 0.0001, -0.0003, 0.0002]
    ], order='F', dtype=float)

    Ry_expected = np.array([[0.0012]], order='F', dtype=float)

    S_expected = np.array([[-0.0025], [-0.0002], [0.0008], [-0.0005]], order='F', dtype=float)

    # Verify shapes
    assert A.shape == (n, n)
    assert C.shape == (l, n)
    assert B.shape == (n, m)
    assert D.shape == (l, m)
    assert K.shape == (n, l)
    assert Q.shape == (n, n)
    assert Ry.shape == (l, l)
    assert S.shape == (n, l)

    # Verify values (HTML shows 4 decimal places, use rtol=5e-3 for some tolerance)
    np.testing.assert_allclose(A, A_expected, rtol=5e-3, atol=1e-3)
    np.testing.assert_allclose(C, C_expected, rtol=5e-3, atol=1e-3)
    np.testing.assert_allclose(B, B_expected, rtol=5e-3, atol=1e-3)
    np.testing.assert_allclose(D, D_expected, rtol=5e-2, atol=1e-3)
    np.testing.assert_allclose(K, K_expected, rtol=5e-3, atol=1e-3)

    # Verify covariance matrices (HTML shows 4 decimal places)
    np.testing.assert_allclose(Q, Q_expected, rtol=5e-2, atol=5e-4)
    np.testing.assert_allclose(Ry, Ry_expected, rtol=5e-2, atol=5e-4)
    np.testing.assert_allclose(S, S_expected, rtol=5e-2, atol=5e-4)

    # Verify covariance matrices are symmetric
    np.testing.assert_allclose(Q, Q.T, rtol=1e-14)


def test_ib01bd_moesp_basic():
    """
    Basic test with MOESP method (METH='M').

    Tests system matrices A, C, B, D estimation without covariances.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    nobr = 4
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 50
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)
    assert info_ad == 0

    A, C, B, D, _Q, _Ry, _S, _K, _iwarn, info = ib01bd('M', 'A', 'N', nobr, n, m, l, nsmp, r, 0.0)

    assert info == 0
    assert A.shape == (n, n)
    assert C.shape == (l, n)
    assert B.shape == (n, m)
    assert D.shape == (l, m)


def test_ib01bd_n4sid_basic():
    """
    Basic test with N4SID method (METH='N').

    Tests system matrices estimation using pure N4SID approach.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    nobr = 4
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 50
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, info_ad = ib01ad('N', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)
    assert info_ad == 0

    A, C, B, D, _Q, _Ry, _S, _K, _iwarn, info = ib01bd('N', 'A', 'N', nobr, n, m, l, nsmp, r, 0.0)

    assert info == 0
    assert A.shape == (n, n)
    assert C.shape == (l, n)
    assert B.shape == (n, m)
    assert D.shape == (l, m)


def test_ib01bd_job_c_only():
    """
    Test JOB='C': compute only A and C matrices.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    nobr = 4
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 50
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)
    assert info_ad == 0

    A, C, _B, _D, _Q, _Ry, _S, _K, _iwarn, info = ib01bd('M', 'C', 'N', nobr, n, m, l, nsmp, r, 0.0)

    assert info == 0
    assert A.shape == (n, n)
    assert C.shape == (l, n)


def test_ib01bd_covariance_symmetry():
    """
    Mathematical property: covariance matrices must be symmetric.

    Q = Q^T and Ry = Ry^T
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    nobr = 5
    m = 1
    l = 2
    nsmp = 2 * (m + l + 1) * nobr + 100
    n = 3

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)
    assert info_ad == 0

    A, C, _B, _D, Q, Ry, _S, _K, _iwarn, info = ib01bd('M', 'A', 'C', nobr, n, m, l, nsmp, r, 0.0)

    assert info == 0
    assert A.shape == (n, n)
    assert C.shape == (l, n)
    assert Q.shape == (n, n)
    assert Ry.shape == (l, l)
    # Verify symmetry
    np.testing.assert_allclose(Q, Q.T, rtol=1e-10)
    np.testing.assert_allclose(Ry, Ry.T, rtol=1e-10)


def test_ib01bd_with_kalman_gain():
    """
    Test JOBCK='K': compute covariances and Kalman gain.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    nobr = 5
    m = 1
    l = 1
    nsmp = 2 * (m + l + 1) * nobr + 100
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)
    assert info_ad == 0

    A, _C, B, _D, Q, Ry, S, K, _iwarn, info = ib01bd('M', 'A', 'K', nobr, n, m, l, nsmp, r, 0.0)

    assert info == 0
    assert A.shape == (n, n)
    assert B.shape == (n, m)
    assert Q.shape == (n, n)
    assert Ry.shape == (l, l)
    assert S.shape == (n, l)
    assert K.shape == (n, l)
    # Verify covariance symmetry
    np.testing.assert_allclose(Q, Q.T, rtol=1e-10)
    np.testing.assert_allclose(Ry, Ry.T, rtol=1e-10)


def test_ib01bd_larger_system():
    """
    Test with larger system dimensions.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    nobr = 6
    m = 2
    l = 2
    nsmp = 2 * (m + l + 1) * nobr + 150
    n = 4

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)
    assert info_ad == 0

    A, C, B, D, _Q, _Ry, _S, _K, _iwarn, info = ib01bd('M', 'A', 'N', nobr, n, m, l, nsmp, r, 0.0)

    assert info == 0
    assert A.shape == (n, n)
    assert C.shape == (l, n)
    assert B.shape == (n, m)
    assert D.shape == (l, m)


def test_ib01bd_m_zero():
    """
    Edge case: M=0 (output-only identification).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    nobr = 4
    m = 0
    l = 2
    nsmp = 2 * (l + 1) * nobr + 50
    n = 2

    u = np.zeros((nsmp, 0), order='F', dtype=float)
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)
    assert info_ad == 0

    A, C, _B, _D, _Q, _Ry, _S, _K, _iwarn, info = ib01bd('M', 'C', 'N', nobr, n, m, l, nsmp, r, 0.0)

    assert info == 0
    assert A.shape == (n, n)
    assert C.shape == (l, n)


def test_ib01bd_error_invalid_meth():
    """
    Error handling: invalid METH parameter.
    """
    np.random.seed(444)
    nobr = 3
    m = 1
    l = 1
    nsmp = 50
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, _info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)

    with pytest.raises(ValueError, match="METH"):
        ib01bd('X', 'A', 'N', nobr, n, m, l, nsmp, r, 0.0)


def test_ib01bd_error_invalid_job():
    """
    Error handling: invalid JOB parameter.
    """
    np.random.seed(555)
    nobr = 3
    m = 1
    l = 1
    nsmp = 50
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, _info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)

    with pytest.raises(ValueError, match="JOB"):
        ib01bd('M', 'X', 'N', nobr, n, m, l, nsmp, r, 0.0)


def test_ib01bd_error_invalid_jobck():
    """
    Error handling: invalid JOBCK parameter.
    """
    np.random.seed(666)
    nobr = 3
    m = 1
    l = 1
    nsmp = 50
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, _info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)

    with pytest.raises(ValueError, match="JOBCK"):
        ib01bd('M', 'A', 'X', nobr, n, m, l, nsmp, r, 0.0)


def test_ib01bd_error_nobr_too_small():
    """
    Error handling: NOBR <= 1.
    """
    np.random.seed(777)
    nobr = 3
    m = 1
    l = 1
    nsmp = 50
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, _info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)

    with pytest.raises(ValueError, match="NOBR"):
        ib01bd('M', 'A', 'N', 1, n, m, l, nsmp, r, 0.0)


def test_ib01bd_error_n_invalid():
    """
    Error handling: N <= 0 or N >= NOBR.
    """
    np.random.seed(888)
    nobr = 4
    m = 1
    l = 1
    nsmp = 50
    _n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, _info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)

    with pytest.raises(ValueError, match="N"):
        ib01bd('M', 'A', 'N', nobr, 0, m, l, nsmp, r, 0.0)

    with pytest.raises(ValueError, match="N"):
        ib01bd('M', 'A', 'N', nobr, nobr, m, l, nsmp, r, 0.0)


def test_ib01bd_error_l_nonpositive():
    """
    Error handling: L <= 0.
    """
    np.random.seed(999)
    nobr = 3
    m = 1
    l = 1
    nsmp = 50
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r, _sv, _iwarn_ad, _info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)

    with pytest.raises(ValueError, match="L"):
        ib01bd('M', 'A', 'N', nobr, n, m, 0, nsmp, r, 0.0)


def test_ib01bd_deterministic_output():
    """
    Test deterministic output: same input produces same results.

    Random seed: 101 (for reproducibility)
    """
    np.random.seed(101)
    nobr = 4
    m = 1
    l = 1
    nsmp = 100
    n = 2

    u = np.random.randn(nsmp, m).astype(float, order='F')
    y = np.random.randn(nsmp, l).astype(float, order='F')

    _n_est, r1, _sv, _iwarn_ad, _info_ad = ib01ad('M', 'Q', 'N', 'O', 'N', 'N', nobr, m, l, u, y, 0.0, 0.0)
    r2 = r1.copy()

    A1, C1, B1, D1, _Q1, _Ry1, _S1, _K1, _iwarn1, info1 = ib01bd('M', 'A', 'N', nobr, n, m, l, nsmp, r1, 0.0)
    A2, C2, B2, D2, _Q2, _Ry2, _S2, _K2, _iwarn2, info2 = ib01bd('M', 'A', 'N', nobr, n, m, l, nsmp, r2, 0.0)

    assert info1 == info2
    np.testing.assert_allclose(A1, A2, rtol=1e-14)
    np.testing.assert_allclose(C1, C2, rtol=1e-14)
    np.testing.assert_allclose(B1, B2, rtol=1e-14)
    np.testing.assert_allclose(D1, D2, rtol=1e-14)
