"""
Tests for IB01OY - User confirmation of system order

This routine validates and potentially modifies the system order estimate.
Non-interactive version for library use.
"""

import numpy as np
import pytest
from ctrlsys import ib01oy


def test_ib01oy_basic():
    """
    Basic validation: accept estimated order within bounds.

    Tests parameter validation with valid inputs.
    """
    ns = 10
    nmax = 8
    n = 5
    sv = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
                  order='F', dtype=float)

    n_out, info = ib01oy(ns, nmax, n, sv)

    assert info == 0
    assert n_out == 5
    assert 0 <= n_out <= nmax
    assert np.all(sv[:-1] >= sv[1:])


def test_ib01oy_n_equals_nmax():
    """
    Edge case: N equals NMAX (boundary condition).
    """
    ns = 10
    nmax = 7
    n = 7
    sv = np.linspace(10.0, 0.001, ns).astype(float, order='F')

    n_out, info = ib01oy(ns, nmax, n, sv)

    assert info == 0
    assert n_out == 7
    assert 0 <= n_out <= nmax


def test_ib01oy_n_zero():
    """
    Edge case: N = 0 (valid minimum order).
    """
    ns = 5
    nmax = 5
    n = 0
    sv = np.array([5.0, 2.0, 1.0, 0.5, 0.1], order='F', dtype=float)

    n_out, info = ib01oy(ns, nmax, n, sv)

    assert info == 0
    assert n_out == 0


def test_ib01oy_singular_values_descending():
    """
    Validate routine accepts descending singular values.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    ns = 8
    nmax = 6
    n = 4

    # Generate descending singular values
    sv = np.sort(np.random.rand(ns))[::-1].astype(float, order='F')

    n_out, info = ib01oy(ns, nmax, n, sv)

    assert info == 0
    assert n_out == 4
    assert 0 <= n_out <= nmax
    assert np.all(np.diff(sv) <= 1e-15)


def test_ib01oy_error_ns_nonpositive():
    """
    Error handling: NS <= 0.
    """
    ns = 0
    nmax = 5
    n = 3
    sv = np.array([1.0], order='F', dtype=float)

    with pytest.raises(ValueError, match="NS must be positive"):
        ib01oy(ns, nmax, n, sv)


def test_ib01oy_error_ns_negative():
    """
    Error handling: NS < 0.
    """
    ns = -5
    nmax = 5
    n = 3
    sv = np.array([1.0], order='F', dtype=float)

    with pytest.raises(ValueError, match="NS must be positive"):
        ib01oy(ns, nmax, n, sv)


def test_ib01oy_error_nmax_negative():
    """
    Error handling: NMAX < 0.
    """
    ns = 10
    nmax = -1
    n = 3
    sv = np.linspace(10.0, 0.1, ns).astype(float, order='F')

    with pytest.raises(ValueError, match="NMAX must be in range"):
        ib01oy(ns, nmax, n, sv)


def test_ib01oy_error_nmax_exceeds_ns():
    """
    Error handling: NMAX > NS.
    """
    ns = 5
    nmax = 10
    n = 3
    sv = np.linspace(10.0, 0.1, ns).astype(float, order='F')

    with pytest.raises(ValueError, match="NMAX must be in range"):
        ib01oy(ns, nmax, n, sv)


def test_ib01oy_error_n_negative():
    """
    Error handling: N < 0.
    """
    ns = 10
    nmax = 8
    n = -2
    sv = np.linspace(10.0, 0.1, ns).astype(float, order='F')

    with pytest.raises(ValueError, match="N must be in range"):
        ib01oy(ns, nmax, n, sv)


def test_ib01oy_error_n_exceeds_ns():
    """
    Error handling: N > NS.
    """
    ns = 5
    nmax = 5
    n = 7
    sv = np.linspace(10.0, 0.1, ns).astype(float, order='F')

    with pytest.raises(ValueError, match="N must be in range"):
        ib01oy(ns, nmax, n, sv)


def test_ib01oy_sv_length_validation():
    """
    Error handling: SV array too short.
    """
    ns = 10
    nmax = 8
    n = 5
    sv = np.array([10.0, 5.0, 2.0], order='F', dtype=float)  # Only 3 elements

    with pytest.raises(ValueError, match="SV must have at least NS elements"):
        ib01oy(ns, nmax, n, sv)


def test_ib01oy_large_system():
    """
    Large system: NS=100, validate performance.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    ns = 100
    nmax = 80
    n = 50

    # Generate descending singular values
    sv = np.sort(np.random.rand(ns))[::-1].astype(float, order='F')

    n_out, info = ib01oy(ns, nmax, n, sv)

    assert info == 0
    assert n_out == 50
    assert 0 <= n_out <= nmax
