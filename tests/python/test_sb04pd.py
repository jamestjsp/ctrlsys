"""
Tests for SB04PD - Solution of continuous-time or discrete-time Sylvester equations

SB04PD solves for X either:
  Continuous-time: op(A)*X + ISGN*X*op(B) = scale*C
  Discrete-time:   op(A)*X*op(B) + ISGN*X = scale*C

where op(M) = M or M**T, and ISGN = 1 or -1.
"""

import numpy as np
import pytest
from ctrlsys import sb04pd


def test_sb04pd_discrete_basic():
    """
    Test discrete-time Sylvester equation using HTML documentation example.

    Equation: A*X*B + X = C (DICO='D', ISGN=1, TRANA='N', TRANB='N')

    Data from SB04PD HTML documentation.
    """
    m = 3
    n = 2

    # A matrix (3x3) - read row-wise from HTML doc
    a = np.array([
        [2.0, 1.0, 3.0],
        [0.0, 2.0, 1.0],
        [6.0, 1.0, 2.0]
    ], order='F', dtype=float)

    # B matrix (2x2) - read row-wise from HTML doc
    b = np.array([
        [2.0, 1.0],
        [1.0, 6.0]
    ], order='F', dtype=float)

    # C matrix (3x2) - read row-wise from HTML doc (right-hand side)
    c = np.array([
        [2.0, 1.0],
        [1.0, 4.0],
        [0.0, 5.0]
    ], order='F', dtype=float)

    # Expected solution X from HTML doc (same structure as C)
    x_expected = np.array([
        [-0.3430,  0.1995],
        [-0.1856,  0.4192],
        [ 0.6922, -0.2952]
    ], order='F', dtype=float)

    # Call routine - FACTA='N', FACTB='N' means Schur factorization computed
    x, a_out, u, b_out, v, scale, info = sb04pd(
        'D', 'N', 'N', 'N', 'N', 1, a, b, c
    )

    assert info == 0
    assert abs(scale - 1.0) < 1e-4

    # Validate solution (HTML doc shows 4 decimal places)
    np.testing.assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)


def test_sb04pd_continuous_basic():
    """
    Test continuous-time Sylvester equation: A*X + X*B = C

    Equation: op(A)*X + ISGN*X*op(B) = scale*C with DICO='C'
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m = 3
    n = 2

    # Create well-conditioned matrices with separated eigenvalues
    a = np.array([
        [-2.0, 0.5, 0.0],
        [ 0.0, -3.0, 0.5],
        [ 0.0,  0.0, -4.0]
    ], order='F', dtype=float)

    b = np.array([
        [-1.0, 0.2],
        [ 0.0, -5.0]
    ], order='F', dtype=float)

    # Random solution X
    x_true = np.random.randn(m, n).astype(float, order='F')

    # Compute C = A*X + X*B (ISGN=1)
    c = a @ x_true + x_true @ b
    c = np.asfortranarray(c)

    # Solve
    x, a_out, u, b_out, v, scale, info = sb04pd(
        'C', 'N', 'N', 'N', 'N', 1, a.copy(), b.copy(), c.copy()
    )

    assert info == 0
    assert abs(scale - 1.0) < 1e-10

    # Validate solution matches true X
    np.testing.assert_allclose(x, x_true, rtol=1e-13, atol=1e-14)


def test_sb04pd_residual_verification_continuous():
    """
    Verify continuous-time Sylvester equation residual: A*X + X*B - C ≈ 0

    Mathematical property test: equation residual should be near machine precision.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m = 4
    n = 3

    # Create stable matrices (all eigenvalues have negative real parts)
    a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
    a[0, 1] = 0.5
    a[1, 2] = 0.3

    b = np.diag([-0.5, -1.5, -2.5]).astype(float, order='F')
    b[0, 1] = 0.2

    # Random C
    c = np.random.randn(m, n).astype(float, order='F')
    c_original = c.copy()

    # Solve A*X + X*B = C
    x, a_out, u, b_out, v, scale, info = sb04pd(
        'C', 'N', 'N', 'N', 'N', 1, a.copy(), b.copy(), c.copy()
    )

    assert info == 0

    # Verify residual: A*X + X*B should equal scale*C_original
    residual = a @ x + x @ b - scale * c_original
    res_norm = np.linalg.norm(residual, 'fro')
    c_norm = np.linalg.norm(c_original, 'fro')

    # Relative residual should be near machine precision
    rel_residual = res_norm / c_norm if c_norm > 0 else res_norm
    assert rel_residual < 1e-13


def test_sb04pd_residual_verification_discrete():
    """
    Verify discrete-time Sylvester equation residual: A*X*B + X - C ≈ 0

    Mathematical property test: equation residual should be near machine precision.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m = 3
    n = 4

    # Create matrices with eigenvalues having product not equal to -1
    a = np.array([
        [0.5, 0.1, 0.0],
        [0.0, 0.3, 0.1],
        [0.0, 0.0, 0.2]
    ], order='F', dtype=float)

    b = np.array([
        [0.4, 0.1, 0.0, 0.0],
        [0.0, 0.6, 0.1, 0.0],
        [0.0, 0.0, 0.5, 0.1],
        [0.0, 0.0, 0.0, 0.3]
    ], order='F', dtype=float)

    # Random C
    c = np.random.randn(m, n).astype(float, order='F')
    c_original = c.copy()

    # Solve A*X*B + X = C (ISGN=1)
    x, a_out, u, b_out, v, scale, info = sb04pd(
        'D', 'N', 'N', 'N', 'N', 1, a.copy(), b.copy(), c.copy()
    )

    assert info == 0

    # Verify residual: A*X*B + X should equal scale*C_original
    residual = a @ x @ b + x - scale * c_original
    res_norm = np.linalg.norm(residual, 'fro')
    c_norm = np.linalg.norm(c_original, 'fro')

    # Relative residual should be near machine precision
    rel_residual = res_norm / c_norm if c_norm > 0 else res_norm
    assert rel_residual < 1e-13


def test_sb04pd_schur_form_input():
    """
    Test with matrices already in Schur form (FACTA='S', FACTB='S').

    When matrices are already quasi-triangular, no factorization is needed.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m = 3
    n = 2

    # A in upper triangular (Schur) form
    a = np.array([
        [2.0, 0.5, 0.1],
        [0.0, 3.0, 0.2],
        [0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    # B in upper triangular (Schur) form
    b = np.array([
        [1.0, 0.3],
        [0.0, 2.0]
    ], order='F', dtype=float)

    # Random C
    c = np.random.randn(m, n).astype(float, order='F')
    c_original = c.copy()

    # Solve A*X*B + X = C with Schur form inputs
    x, a_out, u, b_out, v, scale, info = sb04pd(
        'D', 'S', 'S', 'N', 'N', 1, a.copy(), b.copy(), c.copy()
    )

    assert info == 0

    # Verify residual
    residual = a @ x @ b + x - scale * c_original
    res_norm = np.linalg.norm(residual, 'fro')
    c_norm = np.linalg.norm(c_original, 'fro')
    rel_residual = res_norm / c_norm if c_norm > 0 else res_norm
    assert rel_residual < 1e-13


def test_sb04pd_transpose_a():
    """
    Test with A transposed: A'*X + X*B = C

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m = 3
    n = 2

    # A matrix
    a = np.array([
        [-2.0, 0.5, 0.0],
        [ 0.1, -3.0, 0.5],
        [ 0.0,  0.2, -4.0]
    ], order='F', dtype=float)

    # B matrix
    b = np.array([
        [-1.0, 0.2],
        [ 0.1, -5.0]
    ], order='F', dtype=float)

    # Random solution X
    x_true = np.random.randn(m, n).astype(float, order='F')

    # Compute C = A'*X + X*B (TRANA='T', ISGN=1)
    c = a.T @ x_true + x_true @ b
    c = np.asfortranarray(c)

    # Solve
    x, a_out, u, b_out, v, scale, info = sb04pd(
        'C', 'N', 'N', 'T', 'N', 1, a.copy(), b.copy(), c.copy()
    )

    assert info == 0
    np.testing.assert_allclose(x, x_true, rtol=1e-13, atol=1e-14)


def test_sb04pd_isgn_minus_one():
    """
    Test with ISGN=-1: A*X - X*B = C

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    m = 3
    n = 2

    # A matrix with negative eigenvalues
    a = np.array([
        [-2.0, 0.5, 0.0],
        [ 0.0, -3.0, 0.5],
        [ 0.0,  0.0, -4.0]
    ], order='F', dtype=float)

    # B matrix with positive eigenvalues (to avoid singularity)
    b = np.array([
        [1.0, 0.2],
        [0.0, 5.0]
    ], order='F', dtype=float)

    # Random solution X
    x_true = np.random.randn(m, n).astype(float, order='F')

    # Compute C = A*X - X*B (ISGN=-1)
    c = a @ x_true - x_true @ b
    c = np.asfortranarray(c)

    # Solve
    x, a_out, u, b_out, v, scale, info = sb04pd(
        'C', 'N', 'N', 'N', 'N', -1, a.copy(), b.copy(), c.copy()
    )

    assert info == 0
    np.testing.assert_allclose(x, x_true, rtol=1e-13, atol=1e-14)


def test_sb04pd_zero_dimensions():
    """
    Test edge case with zero dimensions (quick return).
    """
    # M = 0
    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.array([[1.0, 0.0], [0.0, 2.0]], order='F', dtype=float)
    c = np.zeros((0, 2), order='F', dtype=float)

    x, a_out, u, b_out, v, scale, info = sb04pd(
        'C', 'N', 'N', 'N', 'N', 1, a, b, c
    )

    assert info == 0
    assert scale == 1.0


def test_sb04pd_invalid_dico():
    """
    Test error handling for invalid DICO parameter.
    """
    a = np.eye(2, order='F', dtype=float)
    b = np.eye(2, order='F', dtype=float)
    c = np.ones((2, 2), order='F', dtype=float)

    x, a_out, u, b_out, v, scale, info = sb04pd(
        'X', 'N', 'N', 'N', 'N', 1, a, b, c
    )

    assert info == -1


def test_sb04pd_invalid_isgn():
    """
    Test error handling for invalid ISGN parameter (must be 1 or -1).
    """
    a = np.eye(2, order='F', dtype=float)
    b = np.eye(2, order='F', dtype=float)
    c = np.ones((2, 2), order='F', dtype=float)

    x, a_out, u, b_out, v, scale, info = sb04pd(
        'C', 'N', 'N', 'N', 'N', 0, a, b, c
    )

    assert info == -6
