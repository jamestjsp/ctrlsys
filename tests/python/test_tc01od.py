"""
Tests for TC01OD - Dual of left/right polynomial matrix representation.

TC01OD finds the dual right (left) polynomial matrix representation of
a given left (right) polynomial matrix representation by transposing
the coefficient matrices.

Q(s)*inv(P(s)) <-> inv(P(s))*Q(s)
"""

import numpy as np
import pytest
from ctrlsys import tc01od


"""Basic functionality tests using HTML doc example."""

def test_left_to_right_dual_2x2():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Input: Left matrix fraction with M=2, P=2, INDLIM=3
    Expected: Transposed P and Q coefficient matrices

    Data from SLICOT-Reference/doc/TC01OD.html
    """
    m = 2
    p = 2
    indlim = 3

    # PCOEFF input data (P x P x INDLIM for left representation)
    # READ: ((( PCOEFF(I,J,K), K=1,INDLIM ), J=1,PORM ), I=1,PORM)
    # Data lines (row-major in K, then J, then I):
    #   2.0   3.0   1.0   <- I=1, J=1, K=1,2,3
    #   4.0  -1.0  -1.0   <- I=1, J=2, K=1,2,3
    #   5.0   7.0  -6.0   <- I=2, J=1, K=1,2,3
    #   3.0   2.0   2.0   <- I=2, J=2, K=1,2,3
    pcoeff = np.zeros((p, p, indlim), dtype=float, order='F')
    pcoeff[0, 0, :] = [2.0, 3.0, 1.0]
    pcoeff[0, 1, :] = [4.0, -1.0, -1.0]
    pcoeff[1, 0, :] = [5.0, 7.0, -6.0]
    pcoeff[1, 1, :] = [3.0, 2.0, 2.0]

    # QCOEFF input data (P x M x INDLIM)
    # READ: ((( QCOEFF(I,J,K), K=1,INDLIM ), J=1,M ), I=1,P)
    # Data lines:
    #   6.0  -1.0   5.0   <- I=1, J=1, K=1,2,3
    #   1.0   7.0   5.0   <- I=1, J=2, K=1,2,3
    #   1.0   1.0   1.0   <- I=2, J=1, K=1,2,3
    #   4.0   1.0  -1.0   <- I=2, J=2, K=1,2,3
    qcoeff = np.zeros((p, m, indlim), dtype=float, order='F')
    qcoeff[0, 0, :] = [6.0, -1.0, 5.0]
    qcoeff[0, 1, :] = [1.0, 7.0, 5.0]
    qcoeff[1, 0, :] = [1.0, 1.0, 1.0]
    qcoeff[1, 1, :] = [4.0, 1.0, -1.0]

    # Expected PCOEFF output (transposed - swap (i,j) indices)
    # element (1,1) is    2.00   3.00   1.00  <- same (diagonal)
    # element (1,2) is    5.00   7.00  -6.00  <- was (2,1)
    # element (2,1) is    4.00  -1.00  -1.00  <- was (1,2)
    # element (2,2) is    3.00   2.00   2.00  <- same (diagonal)
    pcoeff_expected = np.zeros((p, p, indlim), dtype=float, order='F')
    pcoeff_expected[0, 0, :] = [2.0, 3.0, 1.0]
    pcoeff_expected[0, 1, :] = [5.0, 7.0, -6.0]
    pcoeff_expected[1, 0, :] = [4.0, -1.0, -1.0]
    pcoeff_expected[1, 1, :] = [3.0, 2.0, 2.0]

    # Expected QCOEFF output (M x P x INDLIM - transposed)
    # element (1,1) is    6.00  -1.00   5.00  <- was (1,1)
    # element (1,2) is    1.00   1.00   1.00  <- was (2,1)
    # element (2,1) is    1.00   7.00   5.00  <- was (1,2)
    # element (2,2) is    4.00   1.00  -1.00  <- was (2,2)
    qcoeff_expected = np.zeros((m, p, indlim), dtype=float, order='F')
    qcoeff_expected[0, 0, :] = [6.0, -1.0, 5.0]
    qcoeff_expected[0, 1, :] = [1.0, 1.0, 1.0]
    qcoeff_expected[1, 0, :] = [1.0, 7.0, 5.0]
    qcoeff_expected[1, 1, :] = [4.0, 1.0, -1.0]

    # Call TC01OD
    pcoeff_out, qcoeff_out, info = tc01od('L', m, p, pcoeff, qcoeff)

    assert info == 0
    np.testing.assert_allclose(pcoeff_out, pcoeff_expected, rtol=1e-14)
    np.testing.assert_allclose(qcoeff_out, qcoeff_expected, rtol=1e-14)


"""Mathematical property tests for transposition."""

def test_double_transpose_identity_square():
    """
    Validate involution property: applying dual twice returns original.

    (A^T)^T = A for all coefficient matrices.
    For square systems (m=p), pcoeff dimensions are the same for L and R.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, p, indlim = 3, 3, 4

    # Create random polynomial matrices
    pcoeff_orig = np.random.randn(p, p, indlim).astype(float, order='F')
    qcoeff_orig = np.random.randn(p, m, indlim).astype(float, order='F')

    # First transpose (left -> right dual)
    pcoeff1, qcoeff1, info1 = tc01od('L', m, p, pcoeff_orig.copy(), qcoeff_orig.copy())
    assert info1 == 0

    # Second transpose (right -> left dual)
    pcoeff2, qcoeff2, info2 = tc01od('R', m, p, pcoeff1.copy(), qcoeff1.copy())
    assert info2 == 0

    # Should return to original
    np.testing.assert_allclose(pcoeff2, pcoeff_orig, rtol=1e-14)
    np.testing.assert_allclose(qcoeff2, qcoeff_orig, rtol=1e-14)

def test_right_representation_transpose():
    """
    Test right representation transposition (R -> L dual).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, p, indlim = 2, 3, 3
    mplim = max(m, p)

    # For right representation, PORM = M
    # LDQCO1 >= max(1,m,p), LDQCO2 >= max(1,m,p)
    pcoeff = np.random.randn(m, m, indlim).astype(float, order='F')
    qcoeff = np.zeros((mplim, mplim, indlim), dtype=float, order='F')
    qcoeff[:p, :m, :] = np.random.randn(p, m, indlim)

    qcoeff_input = qcoeff[:p, :m, :].copy()

    pcoeff_out, qcoeff_out, info = tc01od('R', m, p, pcoeff.copy(), qcoeff.copy())
    assert info == 0

    # Verify PCOEFF transposed
    for k in range(indlim):
        np.testing.assert_allclose(pcoeff_out[:, :, k], pcoeff[:, :, k].T, rtol=1e-14)

    # Verify QCOEFF transposed (P x M -> M x P)
    for k in range(indlim):
        np.testing.assert_allclose(qcoeff_out[:m, :p, k], qcoeff_input[:, :, k].T, rtol=1e-14)


"""Edge case tests."""

def test_scalar_system_m1_p1():
    """
    Test scalar system (M=1, P=1).

    For scalar systems, transposition is identity.
    """
    m, p, indlim = 1, 1, 2

    pcoeff = np.array([[[1.0], [2.0]]]).reshape((1, 1, 2), order='F')
    qcoeff = np.array([[[3.0], [4.0]]]).reshape((1, 1, 2), order='F')

    pcoeff_out, qcoeff_out, info = tc01od('L', m, p, pcoeff.copy(), qcoeff.copy())

    assert info == 0
    np.testing.assert_allclose(pcoeff_out, pcoeff, rtol=1e-14)
    np.testing.assert_allclose(qcoeff_out, qcoeff, rtol=1e-14)

def test_rectangular_m_greater_than_p():
    """
    Test rectangular system with M > P.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, p, indlim = 4, 2, 3

    # Left representation: PORM = P
    pcoeff = np.random.randn(p, p, indlim).astype(float, order='F')
    qcoeff = np.random.randn(max(m, p), max(m, p), indlim).astype(float, order='F')

    # Fill only P x M part
    qcoeff_input = np.zeros_like(qcoeff)
    qcoeff_input[:p, :m, :] = np.random.randn(p, m, indlim)

    pcoeff_out, qcoeff_out, info = tc01od('L', m, p, pcoeff.copy(), qcoeff_input.copy())
    assert info == 0

    # Verify PCOEFF transposed
    for k in range(indlim):
        np.testing.assert_allclose(pcoeff_out[:, :, k], pcoeff[:, :, k].T, rtol=1e-14)

def test_rectangular_p_greater_than_m():
    """
    Test rectangular system with P > M.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m, p, indlim = 2, 4, 3

    # Left representation: PORM = P
    pcoeff = np.random.randn(p, p, indlim).astype(float, order='F')
    qcoeff = np.zeros((max(m, p), max(m, p), indlim), dtype=float, order='F')
    qcoeff[:p, :m, :] = np.random.randn(p, m, indlim)

    pcoeff_out, qcoeff_out, info = tc01od('L', m, p, pcoeff.copy(), qcoeff.copy())
    assert info == 0


"""Error handling tests."""

def test_invalid_leri():
    """Test invalid LERI parameter."""
    m, p, indlim = 2, 2, 2
    pcoeff = np.zeros((2, 2, 2), dtype=float, order='F')
    qcoeff = np.zeros((2, 2, 2), dtype=float, order='F')

    _, _, info = tc01od('X', m, p, pcoeff, qcoeff)
    assert info == -1

def test_negative_m():
    """Test negative M parameter."""
    pcoeff = np.zeros((2, 2, 2), dtype=float, order='F')
    qcoeff = np.zeros((2, 2, 2), dtype=float, order='F')

    _, _, info = tc01od('L', -1, 2, pcoeff, qcoeff)
    assert info == -2

def test_negative_p():
    """Test negative P parameter."""
    pcoeff = np.zeros((2, 2, 2), dtype=float, order='F')
    qcoeff = np.zeros((2, 2, 2), dtype=float, order='F')

    _, _, info = tc01od('L', 2, -1, pcoeff, qcoeff)
    assert info == -3

def test_zero_indlim():
    """Test INDLIM = 0 (invalid, must be >= 1)."""
    pcoeff = np.zeros((2, 2, 1), dtype=float, order='F')
    qcoeff = np.zeros((2, 2, 1), dtype=float, order='F')

    # indlim is derived from array shape, so we can't test this directly
    # The C code will check based on the array's third dimension
    pass

def test_zero_m_quick_return():
    """Test M = 0 quick return."""
    pcoeff = np.zeros((2, 2, 2), dtype=float, order='F')
    qcoeff = np.zeros((2, 2, 2), dtype=float, order='F')

    pcoeff_out, qcoeff_out, info = tc01od('L', 0, 2, pcoeff, qcoeff)
    assert info == 0

def test_zero_p_quick_return():
    """Test P = 0 quick return."""
    pcoeff = np.zeros((2, 2, 2), dtype=float, order='F')
    qcoeff = np.zeros((2, 2, 2), dtype=float, order='F')

    pcoeff_out, qcoeff_out, info = tc01od('L', 2, 0, pcoeff, qcoeff)
    assert info == 0
