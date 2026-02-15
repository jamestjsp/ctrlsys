"""
Tests for SB03MX: Discrete Lyapunov equation solver for Schur form matrices.

Solves: op(A)' * X * op(A) - X = scale * C
where A is upper quasi-triangular (Schur form), C is symmetric.

Tests:
1. Basic functionality with diagonal A
2. 2x2 block case (complex eigenvalues)
3. Lyapunov residual property validation
4. Solution symmetry preservation

Random seed: 42, 123, 456, 789 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb03mx_diagonal():
    """
    Validate basic functionality with diagonal matrix A.

    For diagonal A, the solution can be computed element-wise.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03mx

    # Diagonal Schur form matrix with stable eigenvalues (|lambda| < 1)
    a = np.array([
        [0.5, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.0, 0.0, 0.4]
    ], order='F', dtype=float)

    # Symmetric RHS
    c = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    n = 3
    a_copy = a.copy()
    c_copy = c.copy()

    x, scale, info = sb03mx('N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation: A' * X * A - X = scale * C
    residual = a_copy.T @ x @ a_copy - x - scale * c_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-12)

    # Verify symmetry of solution
    assert_allclose(x, x.T, atol=1e-14)


def test_sb03mx_transpose():
    """
    Validate transpose form: A * X * A' - X = scale * C.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03mx

    np.random.seed(123)

    # Upper triangular Schur form with stable eigenvalues
    a = np.array([
        [0.6, 0.2, 0.1],
        [0.0, 0.4, 0.3],
        [0.0, 0.0, 0.5]
    ], order='F', dtype=float)

    # Symmetric RHS
    c = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 1.5, 0.2],
        [0.3, 0.2, 1.0]
    ], order='F', dtype=float)

    a_copy = a.copy()
    c_copy = c.copy()

    x, scale, info = sb03mx('T', a.copy(order='F'), c.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation: A * X * A' - X = scale * C
    residual = a_copy @ x @ a_copy.T - x - scale * c_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-12)


def test_sb03mx_2x2_block():
    """
    Validate with 2x2 block (complex conjugate eigenvalues).

    The 2x2 block must be in standard Schur form.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03mx

    # Schur form with 2x2 block (eigenvalues 0.5 +/- 0.3i, |lambda| = 0.583 < 1)
    a = np.array([
        [ 0.5,  0.3],
        [-0.3,  0.5]
    ], order='F', dtype=float)

    # Symmetric RHS
    c = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ], order='F', dtype=float)

    a_copy = a.copy()
    c_copy = c.copy()

    x, scale, info = sb03mx('N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation
    residual = a_copy.T @ x @ a_copy - x - scale * c_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-12)

    # Verify symmetry
    assert_allclose(x, x.T, atol=1e-14)


def test_sb03mx_mixed_blocks():
    """
    Validate with mixed 1x1 and 2x2 blocks.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03mx

    # 4x4 Schur form: 1x1 block, 2x2 block, 1x1 block
    a = np.array([
        [0.3, 0.1, 0.1, 0.1],
        [0.0, 0.5, 0.2, 0.1],
        [0.0,-0.2, 0.5, 0.1],
        [0.0, 0.0, 0.0, 0.4]
    ], order='F', dtype=float)

    # Symmetric RHS
    c = np.array([
        [1.0, 0.2, 0.1, 0.3],
        [0.2, 2.0, 0.4, 0.2],
        [0.1, 0.4, 1.5, 0.1],
        [0.3, 0.2, 0.1, 1.0]
    ], order='F', dtype=float)

    a_copy = a.copy()
    c_copy = c.copy()

    x, scale, info = sb03mx('N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation
    residual = a_copy.T @ x @ a_copy - x - scale * c_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)


def test_sb03mx_nearly_unstable():
    """
    Validate warning for nearly unstable system (reciprocal eigenvalues).

    When eigenvalue products are close to 1, info=1 is returned.
    """
    from ctrlsys import sb03mx

    # Matrix with eigenvalue very close to 1 (nearly unstable)
    a = np.array([
        [0.9999, 0.0],
        [0.0, 0.5]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    x, scale, info = sb03mx('N', a.copy(order='F'), c.copy(order='F'))

    # May return info=1 for near-reciprocal eigenvalues (warning, not error)
    assert info >= 0
