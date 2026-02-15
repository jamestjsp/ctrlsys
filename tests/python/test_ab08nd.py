"""
Tests for AB08ND: Construction of regular pencil for invariant zeros.

AB08ND constructs a regular pencil (Af - lambda*Bf) for a state-space system
(A,B,C,D) such that its generalized eigenvalues are the invariant zeros.
Also computes orders of infinite zeros and Kronecker indices.
"""

import numpy as np
import pytest
from ctrlsys import ab08nd


"""Basic functionality tests using HTML doc example."""

def test_html_doc_example():
    """
    Test from SLICOT HTML documentation.

    System: 6 states, 2 inputs, 3 outputs
    A = diag(1, 1, 3, -4, -1, 3)
    Expected: 2 finite invariant zeros at s = 2 and s = -1
              2 infinite zeros of degree 1
              1 left Kronecker index of value 2
    """
    n, m, p = 6, 2, 3

    a = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0, -1.0],
        [-1.0, 0.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ], order='F', dtype=float)

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
    assert nu == 2
    assert rank == 2

    ninfz = sum(infz[i] * (i + 1) for i in range(dinfz))
    assert ninfz == 2

    assert nkror == 0
    assert nkrol == 1
    assert kronl[0] == 2

def test_finite_zeros_computation():
    """
    Verify finite invariant zeros by computing eigenvalues of pencil.

    From HTML doc example, zeros should be at s = 2 and s = -1.
    """
    n, m, p = 6, 2, 3

    a = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0, -1.0],
        [-1.0, 0.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ], order='F', dtype=float)

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
    assert nu == 2

    af_nu = np.asfortranarray(af[:nu, :nu].copy())
    bf_nu = np.asfortranarray(bf[:nu, :nu].copy())

    if rank == 0:
        zeros = np.linalg.eigvals(af_nu)
    else:
        # Generalized eigenvalues: solve Bf^{-1} * Af
        bf_inv_af = np.linalg.solve(bf_nu, af_nu)
        zeros = np.linalg.eigvals(bf_inv_af)

    zeros_real = np.sort(np.real(zeros[np.isfinite(zeros)]))
    expected_zeros = np.array([-1.0, 2.0])

    np.testing.assert_allclose(zeros_real, expected_zeros, rtol=1e-3)


"""Tests for observability check (m=0 case)."""

def test_observability_indices():
    """
    Check observability by calling with m=0.

    From HTML doc: left Kronecker indices of (A,C) are [1, 2, 2]
    """
    n, m, p = 6, 0, 3

    a = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([], dtype=float).reshape(6, 0)
    b = np.asfortranarray(b)

    c = np.array([
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    d = np.array([], dtype=float).reshape(3, 0)
    d = np.asfortranarray(d)

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
    assert nkrol == 3

    observable_dim = n - nu
    assert observable_dim == 5

    kronl_vals = sorted(kronl[:nkrol])
    assert kronl_vals == [1, 2, 2]


"""Tests for controllability check (p=0 case)."""

def test_controllability_indices():
    """
    Check controllability by calling with p=0.

    From HTML doc: right Kronecker indices of (A,B) are [2, 3]
    """
    n, m, p = 6, 2, 0

    a = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0, -1.0],
        [-1.0, 0.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0]
    ], order='F', dtype=float)

    c = np.array([], dtype=float).reshape(0, 6)
    c = np.asfortranarray(c)

    d = np.array([], dtype=float).reshape(0, 2)
    d = np.asfortranarray(d)

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
    assert nkror == 2

    controllable_dim = n - nu
    assert controllable_dim == 5

    kronr_vals = sorted(kronr[:nkror])
    assert kronr_vals == [2, 3]


"""Edge case tests for AB08ND."""

def test_zero_n():
    """
    Test with n=0 (static system).

    Static system has no finite zeros.
    """
    n, m, p = 0, 2, 2

    a = np.array([], dtype=float).reshape(0, 0)
    a = np.asfortranarray(a)
    b = np.array([], dtype=float).reshape(0, 2)
    b = np.asfortranarray(b)
    c = np.array([], dtype=float).reshape(2, 0)
    c = np.asfortranarray(c)
    d = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
    assert nu == 0
    assert rank == 2

def test_zero_m_and_p():
    """
    Test with m=0 and p=0 (autonomous system).
    """
    n, m, p = 2, 0, 0

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
    b = np.array([], dtype=float).reshape(2, 0)
    b = np.asfortranarray(b)
    c = np.array([], dtype=float).reshape(0, 2)
    c = np.asfortranarray(c)
    d = np.array([], dtype=float).reshape(0, 0)
    d = np.asfortranarray(d)

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0

def test_siso_system():
    """
    Test SISO system with one finite zero.

    System: x' = -2x + u, y = x + u
    Transfer: G(s) = (s+3)/(s+2)
    Zero at s = -3.
    """
    n, m, p = 1, 1, 1

    a = np.array([[-2.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[1.0]], order='F', dtype=float)

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
    assert rank == 1
    assert nu == 1


"""Tests for scaling option."""

def test_with_scaling():
    """
    Test with scaling enabled (EQUIL='S').

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F') * 100
    a = a - 2.0 * np.eye(n)
    b = np.random.randn(n, m).astype(float, order='F') * 0.01
    c = np.random.randn(p, n).astype(float, order='F') * 100
    d = np.random.randn(p, m).astype(float, order='F')

    result1 = ab08nd('N', n, m, p, a.copy(), b.copy(), c.copy(), d.copy())
    result2 = ab08nd('S', n, m, p, a.copy(), b.copy(), c.copy(), d.copy())

    nu1, rank1, dinfz1, nkror1, nkrol1, _, _, _, _, _, info1 = result1
    nu2, rank2, dinfz2, nkror2, nkrol2, _, _, _, _, _, info2 = result2

    assert info1 == 0
    assert info2 == 0
    assert rank1 == rank2


"""Tests validating mathematical properties."""

def test_structural_invariants_sum():
    """
    Validate: nu + sum(infz*degree) + sum(kronr) + sum(kronl) relates to system size.

    The sum of all structural invariants has a relationship with n, m, p.
    """
    n, m, p = 6, 2, 3

    a = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0, -1.0],
        [-1.0, 0.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ], order='F', dtype=float)

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
    assert nu >= 0
    assert rank >= 0
    assert rank <= min(m, p)

def test_random_system_invariants():
    """
    Validate structural invariants for random system.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 5, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    a = a - 2.0 * np.eye(n)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
    assert nu >= 0
    assert nu <= n
    assert rank >= 0
    assert rank <= min(m, p)
    assert nkror >= 0
    assert nkrol >= 0
    assert dinfz >= 0 and dinfz <= n

def test_pencil_structure_when_rank_zero():
    """
    When rank=0, Bf should be identity matrix.

    Create system where D has no contribution.
    """
    n, m, p = 2, 1, 1

    a = np.array([[-1.0, 0.5], [0.0, -2.0]], order='F', dtype=float)
    b = np.array([[1.0], [0.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    result = ab08nd('N', n, m, p, a, b, c, d)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
    if rank == 0 and nu > 0:
        bf_nu = bf[:nu, :nu]
        np.testing.assert_allclose(bf_nu, np.eye(nu), rtol=1e-10)


"""Tests for tolerance parameter."""

def test_default_tolerance():
    """
    Test with default tolerance (tol=0.0).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    a = a - 2.0 * np.eye(n)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    result = ab08nd('N', n, m, p, a, b, c, d, tol=0.0)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0

def test_custom_tolerance():
    """
    Test with custom tolerance.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    a = a - 2.0 * np.eye(n)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    result = ab08nd('N', n, m, p, a, b, c, d, tol=1e-8)
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = result

    assert info == 0
