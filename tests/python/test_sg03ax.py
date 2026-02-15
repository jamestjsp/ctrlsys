"""
Tests for SG03AX: Generalized discrete Lyapunov equation solver.

Solves reduced generalized discrete-time Lyapunov equation:
    TRANS='N':  A' * X * A - E' * X * E = scale * Y
    TRANS='T':  A * X * A' - E * X * E' = scale * Y

where A is upper quasitriangular, E is upper triangular (generalized Schur form),
and Y is symmetric.

Tests:
1. Basic functionality with simple matrices
2. Transpose equation form
3. Lyapunov residual property validation
4. 2x2 block handling
5. Solution symmetry preservation

Random seed: 42, 123, 456, 789 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sg03ax_basic():
    """
    Validate basic functionality with simple diagonal matrices.

    For diagonal A and E, the solution can be computed element-wise.
    """
    from ctrlsys import sg03ax

    # Diagonal A and E (generalized Schur form is trivial)
    # Eigenvalues of pencil: lambda = A(i,i)/E(i,i) must have |lambda| != 1
    a = np.array([
        [0.5, 0.0, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.0, 0.4]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    # Symmetric RHS
    y = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    a_copy = a.copy()
    e_copy = e.copy()
    y_copy = y.copy()

    x, scale, info = sg03ax('N', a.copy(order='F'), e.copy(order='F'), y.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation: A' * X * A - E' * X * E = scale * Y
    residual = a_copy.T @ x @ a_copy - e_copy.T @ x @ e_copy - scale * y_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-12)

    # Verify symmetry of solution
    assert_allclose(x, x.T, atol=1e-14)


def test_sg03ax_transpose():
    """
    Validate transpose form: A * X * A' - E * X * E' = scale * Y.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sg03ax

    # Upper triangular A and E
    a = np.array([
        [0.5, 0.1, 0.1],
        [0.0, 0.6, 0.2],
        [0.0, 0.0, 0.4]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.1, 0.1],
        [0.0, 1.0, 0.1],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    # Symmetric RHS
    y = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 1.5, 0.2],
        [0.3, 0.2, 1.0]
    ], order='F', dtype=float)

    a_copy = a.copy()
    e_copy = e.copy()
    y_copy = y.copy()

    x, scale, info = sg03ax('T', a.copy(order='F'), e.copy(order='F'), y.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation: A * X * A' - E * X * E' = scale * Y
    residual = a_copy @ x @ a_copy.T - e_copy @ x @ e_copy.T - scale * y_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-12)


def test_sg03ax_2x2_block():
    """
    Validate with 2x2 block in A (complex conjugate eigenvalues).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sg03ax

    # A has a 2x2 block (complex eigenvalues), E is upper triangular
    # Eigenvalues: 0.3 +/- 0.4i (magnitude ~0.5 < 1)
    a = np.array([
        [ 0.3,  0.4],
        [-0.4,  0.3]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], order='F', dtype=float)

    # Symmetric RHS
    y = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ], order='F', dtype=float)

    a_copy = a.copy()
    e_copy = e.copy()
    y_copy = y.copy()

    x, scale, info = sg03ax('N', a.copy(order='F'), e.copy(order='F'), y.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation
    residual = a_copy.T @ x @ a_copy - e_copy.T @ x @ e_copy - scale * y_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-12)

    # Verify symmetry
    assert_allclose(x, x.T, atol=1e-14)


def test_sg03ax_mixed_blocks():
    """
    Validate with mixed 1x1 and 2x2 blocks.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sg03ax

    # 4x4 A: 2x2 block + 2 1x1 blocks
    a = np.array([
        [0.4, 0.3, 0.1, 0.1],
        [-0.3, 0.4, 0.1, 0.1],
        [0.0, 0.0, 0.5, 0.1],
        [0.0, 0.0, 0.0, 0.3]
    ], order='F', dtype=float)

    # Upper triangular E
    e = np.array([
        [1.0, 0.1, 0.1, 0.1],
        [0.0, 1.0, 0.1, 0.1],
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    # Symmetric RHS
    y = np.array([
        [2.0, 0.5, 0.3, 0.2],
        [0.5, 3.0, 0.4, 0.3],
        [0.3, 0.4, 2.5, 0.2],
        [0.2, 0.3, 0.2, 1.5]
    ], order='F', dtype=float)

    a_copy = a.copy()
    e_copy = e.copy()
    y_copy = y.copy()

    x, scale, info = sg03ax('N', a.copy(order='F'), e.copy(order='F'), y.copy(order='F'))

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov equation
    residual = a_copy.T @ x @ a_copy - e_copy.T @ x @ e_copy - scale * y_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)


def test_sg03ax_scalar():
    """
    Validate 1x1 case (scalar equation).

    For scalar: a^2 * x - e^2 * x = y, so x = y / (a^2 - e^2).
    """
    from ctrlsys import sg03ax

    a = np.array([[0.5]], order='F', dtype=float)
    e = np.array([[1.0]], order='F', dtype=float)
    y = np.array([[0.75]], order='F', dtype=float)

    x, scale, info = sg03ax('N', a.copy(order='F'), e.copy(order='F'), y.copy(order='F'))

    assert info == 0
    assert scale == 1.0

    # A' * X * A - E' * X * E = 0.25 * x - 1.0 * x = -0.75 * x = Y = 0.75
    # So x = -1.0
    assert_allclose(x[0, 0], -1.0, rtol=1e-14)


def test_sg03ax_standard_lyapunov():
    """
    Validate that with E=I, reduces to standard discrete Lyapunov.

    Compares result with standard Lyapunov equation solution.
    """
    from ctrlsys import sg03ax

    # When E = I, equation becomes A' * X * A - X = Y
    a = np.array([
        [0.5, 0.1],
        [0.0, 0.6]
    ], order='F', dtype=float)

    e = np.eye(2, order='F', dtype=float)

    y = np.array([
        [1.0, 0.3],
        [0.3, 2.0]
    ], order='F', dtype=float)

    a_copy = a.copy()
    y_copy = y.copy()

    x, scale, info = sg03ax('N', a.copy(order='F'), e.copy(order='F'), y.copy(order='F'))

    assert info == 0

    # Verify standard discrete Lyapunov: A' * X * A - X = scale * Y
    residual = a_copy.T @ x @ a_copy - x - scale * y_copy
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)
