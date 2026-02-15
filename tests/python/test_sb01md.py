"""
Tests for SB01MD: Single-input pole assignment.

Determines the one-dimensional state feedback matrix G for a linear
time-invariant single-input system dX/dt = A*X + B*U, such that the
closed-loop system dX/dt = (A - B*G)*X has desired poles.

The system must be preliminarily reduced to orthogonal canonical form
using AB01MD.

Test data from SLICOT HTML documentation example:
- 4x4 system, single input
- Desired poles: -1 (4 times, all real)
- Expected G: [1.0, 29.0, 93.0, -76.0]

Mathematical properties tested:
- Closed-loop eigenvalues match desired poles
- Z matrix is orthogonal: Z'*Z = Z*Z' = I
- Closed-loop matrix can be reconstructed: A_cl = Z * S * Z'

Random seeds: 42, 123, 456 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_basic_html_doc_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    System: 4x4 continuous-time, single-input
    First reduced to canonical form via AB01MD, then SB01MD assigns poles.

    Input data from HTML doc:
    - N=4
    - A matrix (read row-wise from HTML)
    - B vector
    - Desired poles: WR=[-1,-1,-1,-1], WI=[0,0,0,0]

    Expected output: G = [1.0000, 29.0000, 93.0000, -76.0000]
    """
    from ctrlsys import ab01md, sb01md

    n = 4
    tol = 0.0

    # A matrix from HTML doc (read row-wise)
    a = np.array([
        [-1.0,  0.0,  2.0, -3.0],
        [ 1.0, -4.0,  3.0, -1.0],
        [ 0.0,  2.0,  4.0, -5.0],
        [ 0.0,  0.0, -1.0, -2.0]
    ], order='F', dtype=float)

    # B vector from HTML doc
    b = np.array([1.0, 0.0, 0.0, 0.0], order='F', dtype=float)

    # Desired poles from HTML doc
    wr = np.array([-1.0, -1.0, -1.0, -1.0], dtype=float)
    wi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # Step 1: Reduce to canonical form using AB01MD
    result_ab = ab01md('I', a.copy(order='F'), b.copy(order='F'), tol)
    a_can, b_can, ncont, z, tau, info_ab = result_ab
    assert info_ab == 0, f"AB01MD failed with info={info_ab}"
    assert ncont == n, f"Expected full controllability (ncont={n}), got {ncont}"

    # Step 2: Apply SB01MD for pole assignment
    result_sb = sb01md(ncont, n, a_can.copy(order='F'), b_can.copy(order='F'),
                        wr.copy(), wi.copy(), z.copy(order='F'))
    a_out, b_out, z_out, g, info_sb = result_sb
    assert info_sb == 0, f"SB01MD failed with info={info_sb}"

    # Expected G from HTML doc
    g_expected = np.array([1.0, 29.0, 93.0, -76.0], dtype=float)

    # Validate G (rtol=1e-3 matches HTML doc 4-decimal precision)
    assert_allclose(g, g_expected, rtol=1e-3, atol=1e-4)


def test_closed_loop_eigenvalue_placement():
    """
    Validate that closed-loop eigenvalues match desired poles.

    Mathematical property: eigenvalues of (A - B*G) should equal desired poles.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab01md, sb01md

    n = 4
    tol = 0.0

    # Original system from HTML doc
    a = np.array([
        [-1.0,  0.0,  2.0, -3.0],
        [ 1.0, -4.0,  3.0, -1.0],
        [ 0.0,  2.0,  4.0, -5.0],
        [ 0.0,  0.0, -1.0, -2.0]
    ], order='F', dtype=float)

    b = np.array([1.0, 0.0, 0.0, 0.0], order='F', dtype=float)
    b_matrix = b.reshape(-1, 1)  # For matrix math

    # Desired poles
    wr = np.array([-1.0, -1.0, -1.0, -1.0], dtype=float)
    wi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    desired_poles = wr + 1j * wi

    # Reduce to canonical form
    result_ab = ab01md('I', a.copy(order='F'), b.copy(order='F'), tol)
    a_can, b_can, ncont, z, tau, info_ab = result_ab
    assert info_ab == 0

    # Pole assignment
    result_sb = sb01md(ncont, n, a_can.copy(order='F'), b_can.copy(order='F'),
                        wr.copy(), wi.copy(), z.copy(order='F'))
    a_out, b_out, z_out, g, info_sb = result_sb
    assert info_sb == 0

    # Compute closed-loop matrix: A_cl = A - B*G
    # G is (ncont,), B is (n,), so B*G is outer product
    g_row = g.reshape(1, -1)  # (1, ncont)
    a_closed = a - b_matrix @ g_row

    # Compute eigenvalues of closed-loop system
    eigs = np.linalg.eigvals(a_closed)
    eigs_sorted = np.sort(eigs.real)
    desired_sorted = np.sort(desired_poles.real)

    # Validate eigenvalues match desired poles (relaxed tolerance for numerical algorithm)
    assert_allclose(eigs_sorted, desired_sorted, rtol=1e-3, atol=1e-3)


def test_z_orthogonality():
    """
    Validate that output Z matrix is orthogonal.

    Mathematical property: Z'*Z = Z*Z' = I
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab01md, sb01md

    n = 4
    tol = 0.0

    # System from HTML doc
    a = np.array([
        [-1.0,  0.0,  2.0, -3.0],
        [ 1.0, -4.0,  3.0, -1.0],
        [ 0.0,  2.0,  4.0, -5.0],
        [ 0.0,  0.0, -1.0, -2.0]
    ], order='F', dtype=float)

    b = np.array([1.0, 0.0, 0.0, 0.0], order='F', dtype=float)

    wr = np.array([-1.0, -1.0, -1.0, -1.0], dtype=float)
    wi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # Reduce to canonical form
    result_ab = ab01md('I', a.copy(order='F'), b.copy(order='F'), tol)
    a_can, b_can, ncont, z, tau, info_ab = result_ab
    assert info_ab == 0

    # Pole assignment
    result_sb = sb01md(ncont, n, a_can.copy(order='F'), b_can.copy(order='F'),
                        wr.copy(), wi.copy(), z.copy(order='F'))
    a_out, b_out, z_out, g, info_sb = result_sb
    assert info_sb == 0

    # Use NCONT-by-NCONT part of Z (as per docs)
    z_ncont = z_out[:ncont, :ncont]

    # Check orthogonality
    assert_allclose(z_ncont.T @ z_ncont, np.eye(ncont), atol=1e-12)
    assert_allclose(z_ncont @ z_ncont.T, np.eye(ncont), atol=1e-12)


def test_schur_form_reconstruction():
    """
    Validate closed-loop matrix reconstruction: A_cl = Z * S * Z'.

    Mathematical property: S (output A) is the Schur form of closed-loop matrix.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import ab01md, sb01md

    n = 4
    tol = 0.0

    # System from HTML doc
    a = np.array([
        [-1.0,  0.0,  2.0, -3.0],
        [ 1.0, -4.0,  3.0, -1.0],
        [ 0.0,  2.0,  4.0, -5.0],
        [ 0.0,  0.0, -1.0, -2.0]
    ], order='F', dtype=float)

    b = np.array([1.0, 0.0, 0.0, 0.0], order='F', dtype=float)
    b_matrix = b.reshape(-1, 1)

    wr = np.array([-1.0, -1.0, -1.0, -1.0], dtype=float)
    wi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # Reduce to canonical form
    result_ab = ab01md('I', a.copy(order='F'), b.copy(order='F'), tol)
    a_can, b_can, ncont, z, tau, info_ab = result_ab
    assert info_ab == 0

    # Pole assignment
    result_sb = sb01md(ncont, n, a_can.copy(order='F'), b_can.copy(order='F'),
                        wr.copy(), wi.copy(), z.copy(order='F'))
    s_out, b_out, z_out, g, info_sb = result_sb
    assert info_sb == 0

    # Compute closed-loop matrix: A_cl = A - B*G
    g_row = g.reshape(1, -1)
    a_closed = a - b_matrix @ g_row

    # NCONT-by-NCONT Schur form and Z
    s_ncont = s_out[:ncont, :ncont]
    z_ncont = z_out[:ncont, :ncont]

    # Verify: A_cl = Z * S * Z' (similarity transformation)
    a_closed_recon = z_ncont @ s_ncont @ z_ncont.T
    assert_allclose(a_closed, a_closed_recon, rtol=1e-10, atol=1e-10)


def test_complex_conjugate_poles():
    """
    Test assignment of complex conjugate pole pair.

    Complex poles must appear consecutively in WR/WI arrays.
    """
    from ctrlsys import ab01md, sb01md

    n = 2
    tol = 0.0

    # Simple controllable 2x2 system
    a = np.array([
        [0.0, 1.0],
        [-2.0, -3.0]
    ], order='F', dtype=float)

    b = np.array([0.0, 1.0], order='F', dtype=float)
    b_matrix = b.reshape(-1, 1)

    # Desired complex conjugate pair: -1 +/- j
    wr = np.array([-1.0, -1.0], dtype=float)
    wi = np.array([1.0, -1.0], dtype=float)  # Consecutive conjugate pair

    # Reduce to canonical form
    result_ab = ab01md('I', a.copy(order='F'), b.copy(order='F'), tol)
    a_can, b_can, ncont, z, tau, info_ab = result_ab
    assert info_ab == 0
    assert ncont == n

    # Pole assignment
    result_sb = sb01md(ncont, n, a_can.copy(order='F'), b_can.copy(order='F'),
                        wr.copy(), wi.copy(), z.copy(order='F'))
    a_out, b_out, z_out, g, info_sb = result_sb
    assert info_sb == 0

    # Compute closed-loop eigenvalues
    g_row = g.reshape(1, -1)
    a_closed = a - b_matrix @ g_row
    eigs = np.linalg.eigvals(a_closed)

    # Verify complex conjugate pair
    expected_poles = np.array([-1.0 + 1.0j, -1.0 - 1.0j])
    for expected in expected_poles:
        found = min(abs(e - expected) for e in eigs)
        assert found < 1e-10, f"Expected pole {expected} not found"


def test_ncont_less_than_n():
    """
    Test when controllable order is less than full order.

    When NCONT < N, only NCONT poles are assigned.
    """
    from ctrlsys import ab01md, sb01md

    n = 3
    tol = 0.0

    # System with uncontrollable mode
    # B has no effect on third state (decoupled)
    a = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 3.0]  # Third mode coupled but not controllable
    ], order='F', dtype=float)

    b = np.array([1.0, 1.0, 0.0], order='F', dtype=float)

    # Reduce to canonical form
    result_ab = ab01md('I', a.copy(order='F'), b.copy(order='F'), tol)
    a_can, b_can, ncont, z, tau, info_ab = result_ab
    assert info_ab == 0

    if ncont > 0:
        # Assign poles for controllable part only
        wr = np.array([-1.0] * ncont, dtype=float)
        wi = np.array([0.0] * ncont, dtype=float)

        result_sb = sb01md(ncont, n, a_can.copy(order='F'), b_can.copy(order='F'),
                            wr.copy(), wi.copy(), z.copy(order='F'))
        a_out, b_out, z_out, g, info_sb = result_sb
        assert info_sb == 0
        assert len(g) == ncont


def test_ncont_zero_quick_return():
    """
    Test quick return when NCONT=0 (no controllable part).
    """
    from ctrlsys import sb01md

    ncont = 0
    n = 2

    # Dummy arrays
    a = np.array([[1.0, 0.0], [0.0, 2.0]], order='F', dtype=float)
    b = np.array([0.0, 0.0], order='F', dtype=float)
    wr = np.array([], dtype=float)
    wi = np.array([], dtype=float)
    z = np.eye(2, order='F', dtype=float)

    result = sb01md(ncont, n, a.copy(order='F'), b.copy(order='F'),
                    wr, wi, z.copy(order='F'))
    a_out, b_out, z_out, g, info = result

    assert info == 0
    assert len(g) == 0 or g.size == 0


def test_n_equals_one():
    """
    Test 1x1 case (special handling in algorithm).
    """
    from ctrlsys import ab01md, sb01md

    n = 1
    tol = 0.0

    # 1x1 system
    a = np.array([[2.0]], order='F', dtype=float)
    b = np.array([1.0], order='F', dtype=float)

    # Desired pole
    wr = np.array([-1.0], dtype=float)
    wi = np.array([0.0], dtype=float)

    # Reduce to canonical form
    result_ab = ab01md('I', a.copy(order='F'), b.copy(order='F'), tol)
    a_can, b_can, ncont, z, tau, info_ab = result_ab
    assert info_ab == 0

    # Pole assignment
    result_sb = sb01md(ncont, n, a_can.copy(order='F'), b_can.copy(order='F'),
                        wr.copy(), wi.copy(), z.copy(order='F'))
    a_out, b_out, z_out, g, info_sb = result_sb
    assert info_sb == 0

    # For 1x1: closed-loop = A - B*G = wr
    # G = (A - wr) / B = (2 - (-1)) / 1 = 3
    assert_allclose(g, [3.0], rtol=1e-10)


def test_invalid_ncont_negative():
    """
    Test error handling for NCONT < 0.
    """
    from ctrlsys import sb01md

    ncont = -1
    n = 2

    a = np.array([[1.0, 0.0], [0.0, 2.0]], order='F', dtype=float)
    b = np.array([1.0, 0.0], order='F', dtype=float)
    wr = np.array([-1.0, -1.0], dtype=float)
    wi = np.array([0.0, 0.0], dtype=float)
    z = np.eye(2, order='F', dtype=float)

    result = sb01md(ncont, n, a.copy(order='F'), b.copy(order='F'),
                    wr.copy(), wi.copy(), z.copy(order='F'))
    a_out, b_out, z_out, g, info = result

    assert info == -1  # First parameter error


def test_invalid_n_less_than_ncont():
    """
    Test error handling for N < NCONT.
    """
    from ctrlsys import sb01md

    ncont = 3
    n = 2  # N < NCONT is invalid

    a = np.array([[1.0, 0.0], [0.0, 2.0]], order='F', dtype=float)
    b = np.array([1.0, 0.0], order='F', dtype=float)
    wr = np.array([-1.0, -1.0, -1.0], dtype=float)
    wi = np.array([0.0, 0.0, 0.0], dtype=float)
    z = np.eye(2, order='F', dtype=float)

    result = sb01md(ncont, n, a.copy(order='F'), b.copy(order='F'),
                    wr.copy(), wi.copy(), z.copy(order='F'))
    a_out, b_out, z_out, g, info = result

    assert info == -2  # Second parameter error
