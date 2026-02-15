"""
Tests for AB08NW: Construction of regular pencil for finite Smith zeros.

AB08NW extracts from the system pencil a regular pencil Af-lambda*Ef whose
generalized eigenvalues are the finite Smith zeros. Also computes orders
of infinite zeros and Kronecker indices.

Unlike AB08ND, this routine outputs E matrix (from Af-lambda*Ef) instead of Bf.
"""

import numpy as np
import pytest
from ctrlsys import ab08nw


def test_html_doc_example():
    """
    Test from SLICOT HTML documentation.

    System: 6 states, 2 inputs, 3 outputs
    A = diag(1, 1, 3, -4, -1, 3)
    Expected results:
      - 2 finite invariant zeros at s = 2 and s = -1
      - 2 infinite zeros of order 1
      - 1 left Kronecker index of value 2
      - 0 right Kronecker indices
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

    result = ab08nw('N', n, m, p, a, b, c, d)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0
    assert nfz == 2
    assert nrank == 8  # n + min(m, p) = 6 + 2

    # Number of infinite zeros
    assert niz == 2

    # Kronecker structure
    assert nkror == 0
    assert nkrol == 1
    assert kronl[0] == 2


def test_finite_zeros_computation():
    """
    Verify finite zeros by computing generalized eigenvalues of Af - lambda*Ef.

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

    result = ab08nw('N', n, m, p, a, b, c, d)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0
    assert nfz == 2

    # Extract nfz x nfz submatrices for the pencil
    af_nfz = np.asfortranarray(af[:nfz, :nfz].copy())
    e_nfz = np.asfortranarray(e[:nfz, :nfz].copy())

    # Compute generalized eigenvalues: solve E^{-1} * Af
    e_inv_af = np.linalg.solve(e_nfz, af_nfz)
    zeros = np.linalg.eigvals(e_inv_af)

    zeros_real = np.sort(np.real(zeros[np.isfinite(zeros)]))
    expected_zeros = np.array([-1.0, 2.0])

    np.testing.assert_allclose(zeros_real, expected_zeros, rtol=1e-3)


def test_observability_indices():
    """
    Check observability by calling with m=0.

    From HTML doc: left Kronecker indices of (A,C) are [1, 2, 2]
    The observable subspace dimension is 5 (out of 6).
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

    result = ab08nw('N', n, m, p, a, b, c, d)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0

    # Observable dimension = n - nfz
    observable_dim = n - nfz
    assert observable_dim == 5

    # Left Kronecker indices
    assert nkrol == 3
    kronl_vals = sorted(kronl[:nkrol])
    assert kronl_vals == [1, 2, 2]


def test_controllability_indices():
    """
    Check controllability by calling with p=0.

    From HTML doc: right Kronecker indices of (A,B) are [2, 3]
    The controllable subspace dimension is 5 (out of 6).
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

    result = ab08nw('N', n, m, p, a, b, c, d)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0

    # Controllable dimension = n - nfz
    controllable_dim = n - nfz
    assert controllable_dim == 5

    # Right Kronecker indices
    assert nkror == 2
    kronr_vals = sorted(kronr[:nkror])
    assert kronr_vals == [2, 3]


def test_zero_dimensions():
    """
    Test with n=m=p=0.

    Quick return case: should return immediately.
    """
    n, m, p = 0, 0, 0

    a = np.array([], dtype=float).reshape(0, 0)
    a = np.asfortranarray(a)
    b = np.array([], dtype=float).reshape(0, 0)
    b = np.asfortranarray(b)
    c = np.array([], dtype=float).reshape(0, 0)
    c = np.asfortranarray(c)
    d = np.array([], dtype=float).reshape(0, 0)
    d = np.asfortranarray(d)

    result = ab08nw('N', n, m, p, a, b, c, d)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0
    assert nfz == 0
    assert nrank == 0
    assert dinfz == 0


def test_siso_system():
    """
    Test SISO system with finite zeros.

    System: x' = -2x + u, y = x + u (standard form)
    Transfer function: G(s) = (s+3)/(s+2)
    Has one finite zero at s = -3.
    """
    n, m, p = 1, 1, 1

    a = np.array([[-2.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[1.0]], order='F', dtype=float)

    result = ab08nw('N', n, m, p, a, b, c, d)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0
    assert nfz == 1
    assert nrank == 2  # n + min(m, p) = 1 + 1

    # Verify the finite zero
    af_val = af[0, 0]
    e_val = e[0, 0]
    finite_zero = af_val / e_val
    np.testing.assert_allclose(finite_zero, -3.0, rtol=1e-10)


def test_with_scaling():
    """
    Test with scaling enabled (EQUIL='S').

    Scaling should not change structural invariants.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 2, 2

    # Create ill-conditioned system
    a = np.random.randn(n, n).astype(float, order='F') * 100
    a = a - 2.0 * np.eye(n)
    b = np.random.randn(n, m).astype(float, order='F') * 0.01
    c = np.random.randn(p, n).astype(float, order='F') * 100
    d = np.random.randn(p, m).astype(float, order='F')

    result1 = ab08nw('N', n, m, p, a.copy(), b.copy(), c.copy(), d.copy())
    result2 = ab08nw('S', n, m, p, a.copy(), b.copy(), c.copy(), d.copy())

    (_, _, nfz1, nrank1, _, _, nkror1, _, nkrol1, _, _, _, _, info1) = result1
    (_, _, nfz2, nrank2, _, _, nkror2, _, nkrol2, _, _, _, _, info2) = result2

    assert info1 == 0
    assert info2 == 0
    assert nfz1 == nfz2
    assert nrank1 == nrank2


def test_infinite_eigenvalue_structure():
    """
    Validate infinite eigenvalue multiplicities.

    From HTML doc example:
      - 2 infinite zeros of order 1
      - ninfe = 2 with infe = [2, 2] (multiplicities of infinite eigenvalues)
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

    result = ab08nw('N', n, m, p, a, b, c, d)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0
    assert niz == 2
    assert ninfe == 2

    # Verify multiplicities
    assert infe[0] == 2
    assert infe[1] == 2


def test_infz_degrees():
    """
    Validate infz array contains degrees of infinite elementary divisors.

    From HTML doc: 2 infinite zeros of order 1, so infz[0] = 2 (degree 1).
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

    result = ab08nw('N', n, m, p, a, b, c, d)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0
    assert dinfz == 1  # Maximum degree of infinite elementary divisors
    assert infz[0] == 2  # 2 infinite zeros of degree 1


def test_pencil_regularity():
    """
    Verify Ef is invertible (pencil is regular).

    For a valid reduced pencil, Ef should have full rank.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    a = a - 2.0 * np.eye(n)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    result = ab08nw('N', n, m, p, a, b, c, d)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0

    if nfz > 0:
        e_nfz = e[:nfz, :nfz]
        cond = np.linalg.cond(e_nfz)
        assert cond < 1e10  # Should be well-conditioned


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

    result = ab08nw('N', n, m, p, a, b, c, d, tol=0.0)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

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

    result = ab08nw('N', n, m, p, a, b, c, d, tol=1e-8)
    (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
     infz, kronr, infe, kronl, info) = result

    assert info == 0
