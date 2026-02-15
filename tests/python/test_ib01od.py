"""
Tests for IB01OD - Estimate system order from Hankel singular values.

This routine estimates the system order based on singular values from the
QR factorization of concatenated block Hankel matrices.
"""

import numpy as np
import pytest
from ctrlsys import ib01od


def test_ib01od_tol_positive():
    """
    Basic test: estimate order with positive tolerance.

    Tests order estimation where TOL >= 0. The estimate is indicated by
    the index of the last singular value >= TOL.
    """
    nobr = 5
    l = 2
    sv = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001],
                  order='F', dtype=float)
    tol = 0.5
    ctrl = 'N'  # No user confirmation

    n, iwarn, info = ib01od(ctrl, nobr, l, sv, tol)

    assert info == 0
    assert iwarn == 0
    assert n == 5  # sv[0:5] >= 0.5, sv[5] < 0.5
    assert np.all(sv >= 0)
    assert np.all(sv[:-1] >= sv[1:])


def test_ib01od_tol_zero():
    """
    Test with TOL = 0: use default tolerance NOBR*EPS*SV(1).

    When TOL = 0, an internally computed default is used.
    """
    nobr = 4
    l = 2
    sv = np.array([1.0, 0.5, 0.1, 1e-10, 1e-14, 1e-15, 1e-16, 1e-17],
                  order='F', dtype=float)
    tol = 0.0
    ctrl = 'N'

    n, iwarn, info = ib01od(ctrl, nobr, l, sv, tol)

    assert info == 0
    assert iwarn == 0
    assert n >= 1
    assert n <= nobr
    assert np.all(sv[:n] > 0)


def test_ib01od_tol_negative():
    """
    Test with TOL < 0: estimate based on largest logarithmic gap.

    When TOL < 0, the estimate is indicated by the index of the singular
    value that has the largest logarithmic gap to its successor.
    """
    nobr = 5
    l = 2
    # Clear gap between sv[2] and sv[3]: log10(0.5) - log10(0.001) ~ 2.7
    sv = np.array([10.0, 5.0, 0.5, 0.001, 0.0005, 0.0001, 0.00005, 0.00001,
                   0.000005, 0.000001], order='F', dtype=float)
    tol = -1.0
    ctrl = 'N'

    n, iwarn, info = ib01od(ctrl, nobr, l, sv, tol)

    assert info == 0
    assert iwarn == 0
    assert n == 3  # Largest gap after sv[2]
    gap_at_n = np.log10(sv[n-1]) - np.log10(sv[n])
    for i in range(nobr * l - 1):
        if i != n - 1 and sv[i+1] > 0:
            assert gap_at_n >= np.log10(sv[i]) - np.log10(sv[i+1]) - 1e-10


def test_ib01od_all_zero_singular_values():
    """
    Edge case: all singular values are zero.

    Returns IWARN = 3 and N = 0.
    """
    nobr = 4
    l = 2
    sv = np.zeros(nobr * l, order='F', dtype=float)
    tol = 0.0
    ctrl = 'N'

    n, iwarn, info = ib01od(ctrl, nobr, l, sv, tol)

    assert info == 0
    assert iwarn == 3  # All SVs zero warning
    assert n == 0


def test_ib01od_single_nonzero_sv():
    """
    Edge case: only first singular value is non-zero.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    nobr = 5
    l = 2
    sv = np.zeros(nobr * l, order='F', dtype=float)
    sv[0] = 1.0  # Only first SV is nonzero
    tol = -1.0  # Use gap-based estimation
    ctrl = 'N'

    n, iwarn, info = ib01od(ctrl, nobr, l, sv, tol)

    assert info == 0
    assert iwarn == 0
    assert n == 1


def test_ib01od_monotonic_property():
    """
    Mathematical property: order increases with decreasing tolerance.

    For descending SVs: tol1 > tol2 implies n1 <= n2 when using positive TOL.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    nobr = 6
    l = 2
    sv = np.sort(np.abs(np.random.randn(nobr * l)))[::-1].astype(float, order='F')
    ctrl = 'N'

    # Test with decreasing tolerances
    tol_high = sv[2] + 0.001
    tol_low = sv[4] + 0.001

    n_high, _, info1 = ib01od(ctrl, nobr, l, sv, tol_high)
    n_low, _, info2 = ib01od(ctrl, nobr, l, sv, tol_low)

    assert info1 == 0
    assert info2 == 0
    assert n_high <= n_low  # Lower tolerance -> higher (or equal) order


def test_ib01od_gap_detection_property():
    """
    Mathematical property: gap-based method detects clear transitions.

    When SVs have a distinct pattern [high, high, ..., low, low, ...],
    the gap-based method should identify the transition.
    """
    nobr = 5
    l = 2

    # SVs: [100, 90, 80] >> [0.01, 0.005, ...] - clear gap at position 3
    sv = np.array([100.0, 90.0, 80.0, 0.01, 0.005, 0.002, 0.001, 0.0005,
                   0.0002, 0.0001], order='F', dtype=float)
    tol = -1.0
    ctrl = 'N'

    n, iwarn, info = ib01od(ctrl, nobr, l, sv, tol)

    assert info == 0
    assert n == 3  # Gap after position 3 (80 >> 0.01)


def test_ib01od_confirm_mode():
    """
    Test with CTRL = 'C' (user confirmation mode).

    In non-interactive library mode, this should call IB01OY which
    validates and potentially adjusts n.
    """
    nobr = 4
    l = 2
    sv = np.array([10.0, 5.0, 2.0, 0.1, 0.05, 0.01, 0.005, 0.001],
                  order='F', dtype=float)
    tol = 1.0
    ctrl = 'C'

    n, iwarn, info = ib01od(ctrl, nobr, l, sv, tol)

    assert info == 0
    assert n >= 0
    assert n <= nobr - 1  # n <= NOBR - 1 as per spec


def test_ib01od_error_invalid_ctrl():
    """
    Error handling: invalid CTRL parameter.
    """
    nobr = 4
    l = 2
    sv = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01],
                  order='F', dtype=float)
    tol = 0.5
    ctrl = 'X'  # Invalid

    with pytest.raises(ValueError, match="CTRL"):
        ib01od(ctrl, nobr, l, sv, tol)


def test_ib01od_error_nobr_nonpositive():
    """
    Error handling: NOBR <= 0.
    """
    nobr = 0
    l = 2
    sv = np.array([1.0, 0.5], order='F', dtype=float)
    tol = 0.1
    ctrl = 'N'

    with pytest.raises(ValueError, match="NOBR"):
        ib01od(ctrl, nobr, l, sv, tol)


def test_ib01od_error_l_nonpositive():
    """
    Error handling: L <= 0.
    """
    nobr = 4
    l = 0
    sv = np.array([1.0, 0.5, 0.1, 0.01], order='F', dtype=float)
    tol = 0.1
    ctrl = 'N'

    with pytest.raises(ValueError, match="L"):
        ib01od(ctrl, nobr, l, sv, tol)


def test_ib01od_sv_length_validation():
    """
    Error handling: SV array too short.
    """
    nobr = 4
    l = 2
    sv = np.array([10.0, 5.0, 2.0], order='F', dtype=float)  # Only 3, need 8
    tol = 0.5
    ctrl = 'N'

    with pytest.raises(ValueError, match="SV"):
        ib01od(ctrl, nobr, l, sv, tol)


def test_ib01od_large_system():
    """
    Larger system test: NOBR=20, L=3.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    nobr = 20
    l = 3
    n_sv = nobr * l  # 60 SVs

    # Generate descending SVs with clear gap at position 10
    sv_high = np.linspace(100.0, 10.0, 10)
    sv_low = np.linspace(0.1, 0.001, n_sv - 10)
    sv = np.concatenate([sv_high, sv_low]).astype(float, order='F')

    tol = -1.0  # Gap-based
    ctrl = 'N'

    n, iwarn, info = ib01od(ctrl, nobr, l, sv, tol)

    assert info == 0
    assert n == 10  # Gap at position 10
    assert np.all(sv[:n] > sv[n])
