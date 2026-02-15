"""
Tests for SB10MD: D-step in D-K iteration for continuous-time systems.

SB10MD performs the D-step in the D-K iteration for robust control synthesis.
It estimates the structured singular value (mu) at given frequencies and
optionally fits a state-space D-scaling system.

The routine:
1. Computes W(jw) = D + C*inv(jw*I - A)*B for each frequency
2. Uses AB13MD to estimate D(jw) scaling and mu(jw)
3. If QUTOL >= 0, fits state-space realizations to D(jw) block by block
4. Assembles the fitted blocks into the D-scaling system (AD, BD, CD, DD)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb10md_basic():
    """
    Test basic functionality: estimate mu(jw) for a simple closed-loop system.

    This tests the core path through SB10MD with a simple SISO-like structure.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb10md

    np.random.seed(42)

    nc = 2
    mp = 2
    lendat = 10
    f = 0
    ord_max = 2
    mnb = 2
    nblock = np.array([1, 1], dtype=np.int32)
    itype = np.array([2, 2], dtype=np.int32)
    qutol = 2.0

    a = np.array([[-1.0, 0.5],
                  [0.0, -2.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0, 0.0],
                  [0.0, 0.0]], order='F', dtype=float)

    omega = np.logspace(-1, 1, lendat)

    (a_out, b_out, c_out, d_out, totord, ad, bd, cd, dd, mju, info
    ) = sb10md(nc, mp, lendat, f, ord_max, mnb, nblock, itype, qutol,
               a, b, c, d, omega)

    assert info == 0
    assert len(mju) == lendat
    assert all(mju >= 0)


def test_sb10md_mu_only():
    """
    Test mu-only mode: QUTOL < 0 skips D(s) fitting.

    When QUTOL < 0, only mu(jw) is estimated, not the D-scaling system.
    The TOTORD should be 0 and AD/BD/CD/DD are not referenced.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb10md

    np.random.seed(123)

    nc = 3
    mp = 2
    lendat = 20
    f = 0
    ord_max = 2
    mnb = 2
    nblock = np.array([1, 1], dtype=np.int32)
    itype = np.array([2, 2], dtype=np.int32)
    qutol = -1.0

    a = np.array([[-1.0, 0.1, 0.0],
                  [0.0, -2.0, 0.2],
                  [0.1, 0.0, -3.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.5, 0.5]], order='F', dtype=float)
    c = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.1, 0.0],
                  [0.0, 0.1]], order='F', dtype=float)

    omega = np.logspace(-1, 2, lendat)

    (a_out, b_out, c_out, d_out, totord, ad, bd, cd, dd, mju, info
    ) = sb10md(nc, mp, lendat, f, ord_max, mnb, nblock, itype, qutol,
               a, b, c, d, omega)

    assert info == 0
    assert totord == 0
    assert len(mju) == lendat
    assert all(mju >= 0)


def test_sb10md_single_complex_block():
    """
    Test with single complex block: mu = sigma_max.

    For single complex full block, mu equals the largest singular value.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb10md

    np.random.seed(456)

    nc = 2
    mp = 2
    lendat = 5
    f = 0
    ord_max = 1
    mnb = 1
    nblock = np.array([2], dtype=np.int32)
    itype = np.array([2], dtype=np.int32)
    qutol = -1.0

    a = np.array([[-1.0, 0.0],
                  [0.0, -2.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0, 0.0],
                  [0.0, 0.0]], order='F', dtype=float)

    omega = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

    (a_out, b_out, c_out, d_out, totord, ad, bd, cd, dd, mju, info
    ) = sb10md(nc, mp, lendat, f, ord_max, mnb, nblock, itype, qutol,
               a, b, c, d, omega)

    assert info == 0

    for i, w in enumerate(omega):
        jw_I = 1j * w * np.eye(nc)
        W = c @ np.linalg.solve(jw_I - a, b) + d
        sigma_max = np.linalg.svd(W, compute_uv=False)[0]
        assert_allclose(mju[i], sigma_max, rtol=1e-6, atol=1e-8)


def test_sb10md_with_f_parameter():
    """
    Test with F > 0: adds identity block I_f to D-scaling system.

    F is the size of the I_f block added to the D-scaling output system.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb10md

    np.random.seed(789)

    nc = 2
    mp = 2
    lendat = 8
    f = 1
    ord_max = 2
    mnb = 2
    nblock = np.array([1, 1], dtype=np.int32)
    itype = np.array([2, 2], dtype=np.int32)
    qutol = 2.0

    a = np.array([[-1.0, 0.2],
                  [-0.1, -1.5]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.1, 0.0],
                  [0.0, 0.1]], order='F', dtype=float)

    omega = np.logspace(-1, 1, lendat)

    (a_out, b_out, c_out, d_out, totord, ad, bd, cd, dd, mju, info
    ) = sb10md(nc, mp, lendat, f, ord_max, mnb, nblock, itype, qutol,
               a, b, c, d, omega)

    assert info == 0
    assert len(mju) == lendat
    assert all(mju >= 0)


def test_sb10md_quick_return_mp_zero():
    """
    Test quick return when MP = 0.

    Random seed: N/A (deterministic)
    """
    from ctrlsys import sb10md

    nc = 2
    mp = 0
    lendat = 5
    f = 0
    ord_max = 1
    mnb = 1
    nblock = np.array([1], dtype=np.int32)
    itype = np.array([2], dtype=np.int32)
    qutol = -1.0

    a = np.array([[-1.0, 0.0],
                  [0.0, -2.0]], order='F', dtype=float)
    b = np.zeros((2, 0), order='F', dtype=float)
    c = np.zeros((0, 2), order='F', dtype=float)
    d = np.zeros((0, 0), order='F', dtype=float)

    omega = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

    (a_out, b_out, c_out, d_out, totord, ad, bd, cd, dd, mju, info
    ) = sb10md(nc, mp, lendat, f, ord_max, mnb, nblock, itype, qutol,
               a, b, c, d, omega)

    assert info == 0


def test_sb10md_quick_return_nc_zero():
    """
    Test quick return when NC = 0.

    Random seed: N/A (deterministic)
    """
    from ctrlsys import sb10md

    nc = 0
    mp = 2
    lendat = 5
    f = 0
    ord_max = 1
    mnb = 2
    nblock = np.array([1, 1], dtype=np.int32)
    itype = np.array([2, 2], dtype=np.int32)
    qutol = -1.0

    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, 2), order='F', dtype=float)
    c = np.zeros((2, 0), order='F', dtype=float)
    d = np.array([[0.5, 0.0],
                  [0.0, 0.5]], order='F', dtype=float)

    omega = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

    (a_out, b_out, c_out, d_out, totord, ad, bd, cd, dd, mju, info
    ) = sb10md(nc, mp, lendat, f, ord_max, mnb, nblock, itype, qutol,
               a, b, c, d, omega)

    assert info == 0


def test_sb10md_block_structure_error():
    """
    Test error: sum of block sizes must equal MP.

    Should return INFO = 3.
    """
    from ctrlsys import sb10md

    nc = 2
    mp = 4
    lendat = 5
    f = 0
    ord_max = 1
    mnb = 2
    nblock = np.array([1, 2], dtype=np.int32)
    itype = np.array([2, 2], dtype=np.int32)
    qutol = -1.0

    a = np.array([[-1.0, 0.0],
                  [0.0, -2.0]], order='F', dtype=float)
    b = np.zeros((2, 4), order='F', dtype=float)
    c = np.zeros((4, 2), order='F', dtype=float)
    d = np.zeros((4, 4), order='F', dtype=float)

    omega = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

    (a_out, b_out, c_out, d_out, totord, ad, bd, cd, dd, mju, info
    ) = sb10md(nc, mp, lendat, f, ord_max, mnb, nblock, itype, qutol,
               a, b, c, d, omega)

    assert info == 3


def test_sb10md_mu_positive():
    """
    Test mathematical property: mu(jw) >= 0 for all frequencies.

    The structured singular value is non-negative by definition.
    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import sb10md

    np.random.seed(999)

    nc = 3
    mp = 3
    lendat = 15
    f = 0
    ord_max = 2
    mnb = 3
    nblock = np.array([1, 1, 1], dtype=np.int32)
    itype = np.array([2, 2, 2], dtype=np.int32)
    qutol = -1.0

    a = -np.eye(nc) + 0.1 * np.random.randn(nc, nc)
    a = np.asfortranarray(a)
    b = np.random.randn(nc, mp).astype(float, order='F')
    c = np.random.randn(mp, nc).astype(float, order='F')
    d = 0.1 * np.eye(mp, dtype=float, order='F')

    omega = np.logspace(-2, 2, lendat)

    (a_out, b_out, c_out, d_out, totord, ad, bd, cd, dd, mju, info
    ) = sb10md(nc, mp, lendat, f, ord_max, mnb, nblock, itype, qutol,
               a, b, c, d, omega)

    assert info == 0
    assert all(mju >= 0), "mu must be non-negative at all frequencies"
