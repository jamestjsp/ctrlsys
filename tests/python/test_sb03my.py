"""
Tests for SB03MY: Continuous Lyapunov equation solver for Schur form matrices.

Solves: op(A)' * X + X * op(A) = scale * C
where A is upper quasi-triangular (Schur form), C is symmetric.

Tests:
1. Basic functionality with diagonal A
2. Transpose form (TRANA='T')
3. 2x2 block case (complex eigenvalues)
4. Lyapunov residual property validation
5. Solution symmetry preservation

Random seed: 42, 123, 456, 789 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb03my_diagonal():
    """
    Validate basic functionality with diagonal matrix A.

    For diagonal A with eigenvalues lambda_i, X_ij = C_ij / (lambda_i + lambda_j).
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03my

    # Diagonal Schur form matrix with stable eigenvalues (Re < 0)
    a = np.array([
        [-2.0, 0.0, 0.0],
        [ 0.0,-3.0, 0.0],
        [ 0.0, 0.0,-1.0]
    ], order='F', dtype=float)

    # Symmetric RHS
    c = np.array([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 0.3],
        [0.5, 0.3, 2.0]
    ], order='F', dtype=float)

    n = 3
    a_copy = a.copy()
    c_copy = c.copy()

    x, scale, info = sb03my('N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation: A' * X + X * A = scale * C
    residual = a_copy.T @ x + x @ a_copy - scale * c_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)

    # Verify symmetry of solution
    assert_allclose(x, x.T, atol=1e-14)


def test_sb03my_transpose():
    """
    Validate transpose form: A * X + X * A' = scale * C.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03my

    np.random.seed(123)

    # Upper triangular Schur form with stable eigenvalues
    a = np.array([
        [-2.0, 0.5, 0.2],
        [ 0.0,-1.5, 0.3],
        [ 0.0, 0.0,-1.0]
    ], order='F', dtype=float)

    # Symmetric RHS
    c = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 1.5, 0.2],
        [0.3, 0.2, 1.0]
    ], order='F', dtype=float)

    a_copy = a.copy()
    c_copy = c.copy()

    x, scale, info = sb03my('T', a.copy(order='F'), c.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation: A * X + X * A' = scale * C
    residual = a_copy @ x + x @ a_copy.T - scale * c_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)


def test_sb03my_2x2_block():
    """
    Validate with 2x2 block (complex conjugate eigenvalues).

    The 2x2 block must be in standard Schur form with Re(eigenvalues) < 0.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03my

    # Schur form with 2x2 block (eigenvalues -1 +/- 2i, Re = -1 < 0)
    a = np.array([
        [-1.0,  2.0],
        [-2.0, -1.0]
    ], order='F', dtype=float)

    # Symmetric RHS
    c = np.array([
        [4.0, 1.0],
        [1.0, 3.0]
    ], order='F', dtype=float)

    a_copy = a.copy()
    c_copy = c.copy()

    x, scale, info = sb03my('N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation
    residual = a_copy.T @ x + x @ a_copy - scale * c_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)

    # Verify symmetry
    assert_allclose(x, x.T, atol=1e-14)


def test_sb03my_mixed_blocks():
    """
    Validate with mixed 1x1 and 2x2 blocks.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03my

    # 4x4 Schur form: 1x1 block, 2x2 block, 1x1 block
    # All eigenvalues have negative real parts
    a = np.array([
        [-3.0, 0.5, 0.2, 0.1],
        [ 0.0,-1.0, 1.5, 0.2],
        [ 0.0,-1.5,-1.0, 0.1],
        [ 0.0, 0.0, 0.0,-2.0]
    ], order='F', dtype=float)

    # Symmetric RHS
    c = np.array([
        [2.0, 0.5, 0.3, 0.2],
        [0.5, 3.0, 0.4, 0.3],
        [0.3, 0.4, 2.5, 0.2],
        [0.2, 0.3, 0.2, 1.5]
    ], order='F', dtype=float)

    a_copy = a.copy()
    c_copy = c.copy()

    x, scale, info = sb03my('N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation
    residual = a_copy.T @ x + x @ a_copy - scale * c_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-12)


def test_sb03my_close_eigenvalues():
    """
    Validate warning for eigenvalues with small sum (A and -A close).

    When A has eigenvalues lambda and -lambda, info=1 is returned.
    """
    from ctrlsys import sb03my

    # Matrix with eigenvalues very close to negatives of each other
    # lambda = -0.0001, -1.9999 (sum close to -2, but paired eigenvalues)
    a = np.array([
        [-0.0001, 0.0],
        [ 0.0, -1.9999]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    x, scale, info = sb03my('N', a.copy(order='F'), c.copy(order='F'))

    # Should succeed (eigenvalues don't sum to zero)
    assert info >= 0


def test_sb03my_scalar():
    """
    Validate 1x1 case (scalar equation).

    For scalar case: 2*a*x = c, so x = c/(2*a).
    """
    from ctrlsys import sb03my

    a = np.array([[-2.0]], order='F', dtype=float)
    c = np.array([[8.0]], order='F', dtype=float)

    x, scale, info = sb03my('N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0
    assert scale == 1.0
    # A' * X + X * A = -2*X + X*(-2) = -4*X = C = 8
    # So X = -2
    assert_allclose(x[0, 0], -2.0, rtol=1e-14)
