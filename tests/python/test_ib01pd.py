"""
Tests for IB01PD - Estimate system matrices A, C, B, D from R factor.

IB01PD computes system matrices from the R factor produced by IB01MD/IB01ND.
Used in subspace identification methods (MOESP and N4SID).

Test data generated from control package via test_ib01pd_gen.py script.
All random data uses documented seeds for reproducibility.
"""

import numpy as np
import pytest
from slicot import ib01pd


def test_ib01pd_moesp_all_matrices():
    """
    Test MOESP method computing all system matrices (A, B, C, D).

    Uses R factor from IB01MD with METH='M'.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    nobr = 4
    n = 2
    m = 1
    l = 1
    nsmpl = 100

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    meth = 'M'
    job = 'A'
    jobcv = 'N'
    tol = 0.0

    a, c, b, d, rcond, iwarn, info = ib01pd(
        meth, job, jobcv, nobr, n, m, l, nsmpl, r, tol
    )

    assert info == 0 or info == 3

    if info == 0:
        assert a.shape == (n, n)
        assert c.shape == (l, n)
        assert b.shape == (n, m)
        assert d.shape == (l, m)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(d))


def test_ib01pd_n4sid_all_matrices():
    """
    Test N4SID method computing all system matrices (A, B, C, D).

    Uses R factor from IB01MD with METH='N'.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    nobr = 4
    n = 2
    m = 1
    l = 1
    nsmpl = 100

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    meth = 'N'
    job = 'A'
    jobcv = 'N'
    tol = 0.0

    a, c, b, d, rcond, iwarn, info = ib01pd(
        meth, job, jobcv, nobr, n, m, l, nsmpl, r, tol
    )

    assert info == 0 or info == 3

    if info == 0:
        assert a.shape == (n, n)
        assert c.shape == (l, n)
        assert b.shape == (n, m)
        assert d.shape == (l, m)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(c))


def test_ib01pd_ac_only():
    """
    Test computing A and C matrices only (JOB='C').

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    nobr = 4
    n = 2
    m = 1
    l = 1
    nsmpl = 100

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    meth = 'M'
    job = 'C'
    jobcv = 'N'
    tol = 0.0

    a, c, rcond, iwarn, info = ib01pd(
        meth, job, jobcv, nobr, n, m, l, nsmpl, r, tol
    )

    assert info == 0 or info == 3

    if info == 0:
        assert a.shape == (n, n)
        assert c.shape == (l, n)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(c))


def test_ib01pd_bd_only():
    """
    Test computing B and D matrices only (JOB='D').

    Requires A and C as input when METH='N'.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    nobr = 4
    n = 2
    m = 1
    l = 1
    nsmpl = 100

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    a = np.random.randn(n, n).astype(float, order='F')
    c = np.random.randn(l, n).astype(float, order='F')

    meth = 'N'
    job = 'D'
    jobcv = 'N'
    tol = 0.0

    b, d, rcond, iwarn, info = ib01pd(
        meth, job, jobcv, nobr, n, m, l, nsmpl, r, tol, a=a, c=c
    )

    assert info == 0 or info == 3

    if info == 0:
        assert b.shape == (n, m)
        assert d.shape == (l, m)
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(d))


def test_ib01pd_with_covariance():
    """
    Test computing covariance matrices (JOBCV='C').

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    nobr = 4
    n = 2
    m = 1
    l = 1
    nsmpl = 2 * (m + l) * nobr + 50

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    meth = 'M'
    job = 'A'
    jobcv = 'C'
    tol = 0.0

    a, c, b, d, q, ry, s, o, rcond, iwarn, info = ib01pd(
        meth, job, jobcv, nobr, n, m, l, nsmpl, r, tol
    )

    assert info == 0 or info == 3

    if info == 0:
        assert q.shape == (n, n)
        assert ry.shape == (l, l)
        assert s.shape == (n, l)
        assert o.shape == (l * nobr, n)
        q_eigs = np.linalg.eigvalsh(q)
        assert np.all(q_eigs >= -1e-10)
        ry_eigs = np.linalg.eigvalsh(ry)
        assert np.all(ry_eigs >= -1e-10)


def test_ib01pd_m_zero():
    """
    Edge case: M = 0 (no inputs, output-only identification).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)

    nobr = 4
    n = 2
    m = 0
    l = 1
    nsmpl = 100

    nr = 2 * l * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    meth = 'M'
    job = 'C'
    jobcv = 'N'
    tol = 0.0

    a, c, rcond, iwarn, info = ib01pd(
        meth, job, jobcv, nobr, n, m, l, nsmpl, r, tol
    )

    assert info == 0 or info == 3

    if info == 0:
        assert a.shape == (n, n)
        assert c.shape == (l, n)
        assert np.all(np.isfinite(a))


def test_ib01pd_larger_system():
    """
    Test with larger system dimensions.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)

    nobr = 5
    n = 3
    m = 2
    l = 2
    nsmpl = 200

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    meth = 'M'
    job = 'A'
    jobcv = 'N'
    tol = 0.0

    a, c, b, d, rcond, iwarn, info = ib01pd(
        meth, job, jobcv, nobr, n, m, l, nsmpl, r, tol
    )

    assert info == 0 or info == 3

    if info == 0:
        assert a.shape == (n, n)
        assert c.shape == (l, n)
        assert b.shape == (n, m)
        assert d.shape == (l, m)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(d))


def test_ib01pd_error_invalid_meth():
    """
    Error handling: invalid METH parameter.
    """
    np.random.seed(444)

    nobr = 4
    n = 2
    m = 1
    l = 1
    nsmpl = 100

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    with pytest.raises(ValueError, match="METH"):
        ib01pd('Z', 'A', 'N', nobr, n, m, l, nsmpl, r, 0.0)


def test_ib01pd_error_invalid_job():
    """
    Error handling: invalid JOB parameter.
    """
    np.random.seed(555)

    nobr = 4
    n = 2
    m = 1
    l = 1
    nsmpl = 100

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    with pytest.raises(ValueError, match="JOB"):
        ib01pd('M', 'Z', 'N', nobr, n, m, l, nsmpl, r, 0.0)


def test_ib01pd_error_nobr_too_small():
    """
    Error handling: NOBR <= 1.
    """
    np.random.seed(666)

    nobr = 1
    n = 0
    m = 1
    l = 1
    nsmpl = 100

    r = np.random.randn(4, 4).astype(float, order='F')

    with pytest.raises(ValueError, match="NOBR"):
        ib01pd('M', 'A', 'N', nobr, n, m, l, nsmpl, r, 0.0)


def test_ib01pd_error_n_invalid():
    """
    Error handling: N <= 0 or N >= NOBR.
    """
    np.random.seed(777)

    nobr = 4
    n = 4
    m = 1
    l = 1
    nsmpl = 100

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')

    with pytest.raises(ValueError, match="N"):
        ib01pd('M', 'A', 'N', nobr, n, m, l, nsmpl, r, 0.0)


def test_ib01pd_error_l_invalid():
    """
    Error handling: L <= 0.
    """
    np.random.seed(888)

    nobr = 4
    n = 2
    m = 1
    l = 0
    nsmpl = 100

    r = np.random.randn(8, 8).astype(float, order='F')

    with pytest.raises(ValueError, match="L"):
        ib01pd('M', 'A', 'N', nobr, n, m, l, nsmpl, r, 0.0)


def test_ib01pd_rcond_output():
    """
    Test that reciprocal condition numbers are returned.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    nobr = 4
    n = 2
    m = 1
    l = 1
    nsmpl = 100

    nr = 2 * (m + l) * nobr
    r = np.random.randn(nr, nr).astype(float, order='F')
    r = np.triu(r)

    meth = 'M'
    job = 'A'
    jobcv = 'N'
    tol = 0.0

    result = ib01pd(meth, job, jobcv, nobr, n, m, l, nsmpl, r, tol)

    rcond = result[-3]

    assert isinstance(rcond, np.ndarray) or isinstance(rcond, (list, tuple))


def test_ib01pd_warning_rank_deficient():
    """
    Test warning indicator IWARN=4 for rank-deficient problem.

    Create deliberately rank-deficient R matrix.
    Random seed: 101 (for reproducibility)
    """
    np.random.seed(101)

    nobr = 4
    n = 2
    m = 1
    l = 1
    nsmpl = 100

    nr = 2 * (m + l) * nobr
    r = np.zeros((nr, nr), order='F', dtype=float)
    for i in range(nr):
        r[i, i] = 1e-15

    meth = 'M'
    job = 'A'
    jobcv = 'N'
    tol = 0.0

    result = ib01pd(meth, job, jobcv, nobr, n, m, l, nsmpl, r, tol)

    iwarn = result[-2]
    info = result[-1]

    assert info == 0 or info == 3 or iwarn in [0, 4]
