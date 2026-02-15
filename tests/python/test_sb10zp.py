"""
Tests for SB10ZP - Transform SISO system to stable and minimum phase.

SB10ZP computes a stable and minimum phase SISO system from a given
SISO system by reflecting unstable poles/zeros to their stable counterparts.
"""

import numpy as np
import pytest
from ctrlsys import sb10zp


def test_sb10zp_continuous_unstable_pole():
    """
    Test continuous-time system with unstable pole at +2.

    System: A = [[2, 1], [0, -1]], B = [1, 0]', C = [1, 1], D = 2
    Pole at +2 (unstable) should be mirrored to -2.
    After transformation, all eigenvalues should have negative real parts.
    """
    a = np.array([[2.0, 1.0], [0.0, -1.0]], order='F', dtype=float)
    b = np.array([1.0, 0.0], order='F', dtype=float)
    c = np.array([1.0, 1.0], order='F', dtype=float)
    d = np.array([2.0], order='F', dtype=float)

    discfl = 0

    a_out, b_out, c_out, d_out, n_out, info = sb10zp(discfl, a, b, c, d)

    assert info == 0, f"sb10zp failed with info={info}"
    np.testing.assert_allclose(d_out[0], 2.0, rtol=1e-10,
                               err_msg="D should be preserved")

    if n_out > 0:
        eigs = np.linalg.eigvals(a_out[:n_out, :n_out])
        assert np.all(eigs.real < 0), f"System should be stable, got eigs={eigs}"


def test_sb10zp_already_stable():
    """
    Test continuous-time system that is already stable.

    System: A = [[-1, 0], [0, -2]], B = [1, 1]', C = [1, 0], D = 1
    Poles at -1, -2 (stable). System should remain similar.
    """
    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
    b = np.array([1.0, 1.0], order='F', dtype=float)
    c = np.array([1.0, 0.0], order='F', dtype=float)
    d = np.array([1.0], order='F', dtype=float)

    discfl = 0

    a_out, b_out, c_out, d_out, n_out, info = sb10zp(discfl, a, b, c, d)

    assert info == 0, f"sb10zp failed with info={info}"
    np.testing.assert_allclose(d_out[0], 1.0, rtol=1e-10,
                               err_msg="D should be preserved")

    if n_out > 0:
        eigs = np.linalg.eigvals(a_out[:n_out, :n_out])
        assert np.all(eigs.real < 0), f"System should be stable, got eigs={eigs}"


def test_sb10zp_zero_order():
    """
    Test zero-order system (N=0).
    Should return immediately with info = 0.
    """
    a = np.array([[0.0]], order='F', dtype=float)
    b = np.array([0.0], order='F', dtype=float)
    c = np.array([0.0], order='F', dtype=float)
    d = np.array([5.0], order='F', dtype=float)

    a_empty = a[:0, :0]
    b_empty = b[:0]
    c_empty = c[:0]

    discfl = 0

    a_out, b_out, c_out, d_out, n_out, info = sb10zp(discfl, a_empty, b_empty, c_empty, d)

    assert info == 0, f"sb10zp failed with info={info}"
    assert n_out == 0


def test_sb10zp_zero_d_error():
    """
    Test that D = 0 causes info = 3 (inverse system cannot be computed).
    """
    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
    b = np.array([1.0, 0.0], order='F', dtype=float)
    c = np.array([1.0, 0.0], order='F', dtype=float)
    d = np.array([0.0], order='F', dtype=float)

    discfl = 0

    a_out, b_out, c_out, d_out, n_out, info = sb10zp(discfl, a, b, c, d)

    assert info == 3, f"Expected info=3 for D=0, got info={info}"


def test_sb10zp_eigenvalue_stability_property():
    """
    Validate mathematical property: all output eigenvalues have negative real parts.

    For a continuous-time system, sb10zp should transform any unstable poles
    to their stable counterparts (reflected across imaginary axis).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3
    a = np.random.randn(n, n).astype(float, order='F')
    a[0, 0] = 2.0
    b = np.random.randn(n).astype(float, order='F')
    c = np.random.randn(n).astype(float, order='F')
    d = np.array([1.5], order='F', dtype=float)

    discfl = 0

    a_out, b_out, c_out, d_out, n_out, info = sb10zp(discfl, a, b, c, d)

    if info == 0 and n_out > 0:
        eigs = np.linalg.eigvals(a_out[:n_out, :n_out])
        assert np.all(eigs.real < 1e-10),             f"All eigenvalues should have non-positive real parts, got {eigs}"


def test_sb10zp_d_preservation():
    """
    Validate that D is preserved through transformation.

    The DC gain structure should be maintained.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 2
    a = np.array([[-1.0, 0.5], [0.0, -2.0]], order='F', dtype=float)
    b = np.random.randn(n).astype(float, order='F')
    c = np.random.randn(n).astype(float, order='F')
    d_val = 3.14159
    d = np.array([d_val], order='F', dtype=float)

    discfl = 0

    a_out, b_out, c_out, d_out, n_out, info = sb10zp(discfl, a, b, c, d)

    assert info == 0, f"sb10zp failed with info={info}"
    np.testing.assert_allclose(d_out[0], d_val, rtol=1e-14,
                               err_msg="D should be preserved exactly")
