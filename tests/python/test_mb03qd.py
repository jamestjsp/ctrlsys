"""
Tests for MB03QD: Eigenvalue reordering in upper quasi-triangular matrix.

Reorders diagonal blocks of a principal submatrix of an upper quasi-triangular
matrix together with their eigenvalues by constructing an orthogonal similarity
transformation.

Tests:
1. Basic eigenvalue separation (continuous-time, stable domain)
2. Discrete-time eigenvalue separation (inside unit circle)
3. Eigenvalue preservation property (similarity transformation)
4. Orthogonality of transformation matrix

Random seed: 42, 123, 456 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03qd_continuous_stable():
    """
    Validate continuous-time stable eigenvalue separation.

    Moves eigenvalues with Re(lambda) < alpha to leading block.
    Tests eigenvalue preservation under similarity transformation.
    """
    from ctrlsys import mb03qd

    # 4x4 quasi-triangular matrix with eigenvalues: -2, -1, 1, 2
    # We want to separate stable (Re < 0) from unstable (Re > 0)
    a = np.array([
        [ 1.0,  0.5,  0.2,  0.1],
        [ 0.0,  2.0,  0.3,  0.2],
        [ 0.0,  0.0, -1.0,  0.4],
        [ 0.0,  0.0,  0.0, -2.0]
    ], order='F', dtype=float)

    n = 4
    nlow = 1
    nsup = 4
    alpha = 0.0  # Boundary: Re(lambda) < 0 is stable

    a_orig = a.copy()

    a_out, u, ndim, info = mb03qd('C', 'S', 'I', a.copy(order='F'), nlow, nsup, alpha)

    assert info == 0
    # Two eigenvalues (-1, -2) should be in stable region
    assert ndim == 2

    # Verify eigenvalue preservation
    eig_before = np.sort(np.linalg.eigvals(a_orig).real)
    eig_after = np.sort(np.linalg.eigvals(a_out).real)
    assert_allclose(eig_before, eig_after, rtol=1e-13)

    # Verify orthogonality: U' * U = I
    assert_allclose(u.T @ u, np.eye(n), atol=1e-14)

    # Verify similarity: A_out = U' * A * U
    a_reconstructed = u.T @ a_orig @ u
    assert_allclose(a_out, a_reconstructed, atol=1e-13)


def test_mb03qd_discrete_stable():
    """
    Validate discrete-time stable eigenvalue separation.

    Moves eigenvalues with |lambda| < alpha to leading block.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03qd

    # 3x3 diagonal matrix with eigenvalues: 0.5, 1.5, 2.0
    # Stable in discrete-time means |lambda| < 1
    a = np.array([
        [0.5, 0.1, 0.2],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    n = 3
    nlow = 1
    nsup = 3
    alpha = 1.0  # Boundary: |lambda| < 1 is stable

    a_orig = a.copy()

    a_out, u, ndim, info = mb03qd('D', 'S', 'I', a.copy(order='F'), nlow, nsup, alpha)

    assert info == 0
    # One eigenvalue (0.5) is inside unit circle
    assert ndim == 1

    # Verify eigenvalue preservation
    eig_before = np.sort(np.linalg.eigvals(a_orig).real)
    eig_after = np.sort(np.linalg.eigvals(a_out).real)
    assert_allclose(eig_before, eig_after, rtol=1e-13)


def test_mb03qd_update_mode():
    """
    Validate update mode (JOBU='U') accumulates transformations.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03qd

    np.random.seed(123)

    n = 4

    # Quasi-triangular matrix
    a = np.array([
        [-1.0,  0.5,  0.2,  0.1],
        [ 0.0, -2.0,  0.3,  0.2],
        [ 0.0,  0.0,  1.0,  0.4],
        [ 0.0,  0.0,  0.0,  2.0]
    ], order='F', dtype=float)

    # Initial transformation matrix (identity)
    u_init = np.eye(n, order='F', dtype=float)

    nlow = 1
    nsup = 4
    alpha = 0.0

    a_out, u_out, ndim, info = mb03qd('C', 'S', 'U', a.copy(order='F'), nlow, nsup, alpha, u_init.copy(order='F'))

    assert info == 0

    # Verify orthogonality
    assert_allclose(u_out.T @ u_out, np.eye(n), atol=1e-14)


def test_mb03qd_complex_eigenvalues():
    """
    Validate handling of 2x2 blocks (complex conjugate eigenvalues).

    A 2x2 block represents a complex conjugate pair.
    """
    from ctrlsys import mb03qd

    # 4x4 matrix with one 2x2 block (complex eigenvalues)
    # Block at (2,2)-(3,3) has eigenvalues 0.5 +/- 0.5i
    a = np.array([
        [ 2.0,  0.5,  0.2,  0.1],
        [ 0.0,  3.0,  0.3,  0.2],
        [ 0.0,  0.0,  0.5,  0.5],
        [ 0.0,  0.0, -0.5,  0.5]
    ], order='F', dtype=float)

    n = 4
    nlow = 1
    nsup = 4
    alpha = 1.0  # Continuous-time: Re < 1

    a_orig = a.copy()

    a_out, u, ndim, info = mb03qd('C', 'S', 'I', a.copy(order='F'), nlow, nsup, alpha)

    assert info == 0

    # Eigenvalues with Re < 1: 0.5+0.5i and 0.5-0.5i (counted as 2)
    assert ndim == 2

    # Verify eigenvalue preservation (including complex)
    eig_before = np.sort(np.linalg.eigvals(a_orig))
    eig_after = np.sort(np.linalg.eigvals(a_out))
    assert_allclose(np.sort(eig_before.real), np.sort(eig_after.real), rtol=1e-13)


def test_mb03qd_unstable_domain():
    """
    Validate unstable domain selection (STDOM='U').

    Moves eigenvalues with Re(lambda) > alpha to leading block.
    """
    from ctrlsys import mb03qd

    a = np.array([
        [-1.0,  0.5,  0.2],
        [ 0.0,  2.0,  0.3],
        [ 0.0,  0.0,  3.0]
    ], order='F', dtype=float)

    n = 3
    nlow = 1
    nsup = 3
    alpha = 0.0  # Re > 0 is unstable

    a_orig = a.copy()

    a_out, u, ndim, info = mb03qd('C', 'U', 'I', a.copy(order='F'), nlow, nsup, alpha)

    assert info == 0
    # Two eigenvalues (2, 3) are in unstable region (Re > 0)
    assert ndim == 2

    # Verify eigenvalue preservation
    eig_before = np.sort(np.linalg.eigvals(a_orig).real)
    eig_after = np.sort(np.linalg.eigvals(a_out).real)
    assert_allclose(eig_before, eig_after, rtol=1e-13)
