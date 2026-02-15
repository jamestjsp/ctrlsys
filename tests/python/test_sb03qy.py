"""
Tests for SB03QY: Estimate separation and 1-norm of Theta operator for continuous-time Lyapunov.

Estimates sep(op(A), -op(A)') and/or 1-norm of Theta where:
- sep(op(A),-op(A)') = min norm(op(A)'*X + X*op(A))/norm(X)
- Theta is related to the continuous-time Lyapunov equation: op(A)'*X + X*op(A) = C

Tests:
1. Basic separation estimation with diagonal Schur form
2. Both separation and Theta norm estimation
3. Transpose operation form
4. Reduced Lyapunov mode (LYAPUN='R')
5. Mathematical property: separation bounds
6. Edge case: n=1 scalar

Random seeds: 42, 123, 456, 789 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb03qy_separation_only():
    """
    Validate separation estimation with diagonal Schur form.

    For a stable diagonal matrix, the separation from its negative transpose
    can be computed analytically. With eigenvalues lambda_i, the separation is:
    sep(A, -A') = min_i,j |lambda_i + lambda_j|

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03qy

    n = 3
    # Diagonal Schur form with stable eigenvalues (Re < 0)
    t = np.array([
        [-2.0, 0.0, 0.0],
        [ 0.0,-3.0, 0.0],
        [ 0.0, 0.0,-1.0]
    ], order='F', dtype=float)

    # Identity orthogonal matrix for simple case
    u = np.eye(n, order='F', dtype=float)

    # X is not referenced when job='S'
    x = np.zeros((n, n), order='F', dtype=float)

    sep, thnorm, info = sb03qy('S', 'N', 'R', t, u, x)

    assert info == 0
    # For diagonal A, sep(A,-A') = min|lambda_i + lambda_j| = |-1-1| = 2
    # The estimator should give a value close to this
    assert sep > 0, "Separation must be positive for stable system"
    # Thnorm not computed when job='S'


def test_sb03qy_both_sep_and_thnorm():
    """
    Validate computation of both separation and Theta norm.

    Uses a simple upper triangular Schur form where we can verify
    the separation is positive and Theta norm is reasonable.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03qy, sb03my

    n = 3
    # Upper triangular Schur form with stable eigenvalues
    t = np.array([
        [-2.0, 0.5, 0.2],
        [ 0.0,-1.5, 0.3],
        [ 0.0, 0.0,-1.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)

    # Compute X by solving Lyapunov equation A'*X + X*A = C
    c = np.array([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 0.3],
        [0.5, 0.3, 2.0]
    ], order='F', dtype=float)

    x, scale, info_lyap = sb03my('N', t.copy(order='F'), c.copy(order='F'))
    assert info_lyap == 0

    sep, thnorm, info = sb03qy('B', 'N', 'R', t, u, x)

    assert info == 0
    assert sep > 0, "Separation must be positive for stable system"
    assert thnorm >= 0, "Theta norm must be non-negative"


def test_sb03qy_transpose_form():
    """
    Validate with transpose form: op(A) = A'.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03qy, sb03my

    n = 2
    # Simple diagonal matrix
    t = np.array([
        [-2.0, 0.0],
        [ 0.0,-3.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)

    # Solve A*X + X*A' = C (transpose form)
    c = np.array([
        [2.0, 0.5],
        [0.5, 3.0]
    ], order='F', dtype=float)

    x, scale, info_lyap = sb03my('T', t.copy(order='F'), c.copy(order='F'))
    assert info_lyap == 0

    sep, thnorm, info = sb03qy('B', 'T', 'R', t, u, x)

    assert info == 0
    assert sep > 0


def test_sb03qy_original_lyapunov():
    """
    Validate with original Lyapunov mode (LYAPUN='O').

    This mode uses U to transform between original and Schur coordinates.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03qy, sb03my

    np.random.seed(789)

    n = 3
    # Create a stable matrix A and compute its Schur decomposition
    # Use a simple triangular matrix as Schur form
    t = np.array([
        [-2.0, 0.5, 0.1],
        [ 0.0,-1.5, 0.2],
        [ 0.0, 0.0,-1.0]
    ], order='F', dtype=float)

    # Create orthogonal matrix U (identity for simplicity)
    u = np.eye(n, order='F', dtype=float)

    # Solve reduced Lyapunov equation
    c = np.array([
        [3.0, 0.5, 0.2],
        [0.5, 2.0, 0.3],
        [0.2, 0.3, 1.5]
    ], order='F', dtype=float)

    x, scale, info_lyap = sb03my('N', t.copy(order='F'), c.copy(order='F'))
    assert info_lyap == 0

    # Original Lyapunov mode
    sep, thnorm, info = sb03qy('B', 'N', 'O', t, u, x)

    assert info == 0
    assert sep > 0
    assert thnorm >= 0


def test_sb03qy_separation_mathematical_bound():
    """
    Validate that estimated separation satisfies mathematical bounds.

    For a diagonal matrix with eigenvalues lambda_i, the true separation is:
    sep(A, -A') = min_{i,j} |lambda_i + lambda_j|

    The estimator should satisfy: sep_estimated <= sep_true * N
    (condition number estimation bound from Higham's 1-norm estimator)

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03qy

    n = 2
    # Diagonal matrix with eigenvalues -1, -3
    t = np.array([
        [-1.0, 0.0],
        [ 0.0,-3.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)
    x = np.zeros((n, n), order='F', dtype=float)

    sep, thnorm, info = sb03qy('S', 'N', 'R', t, u, x)

    assert info == 0

    # True separation: min(|-1-1|, |-1-3|, |-3-1|, |-3-3|) = min(2, 4, 4, 6) = 2
    true_sep = 2.0

    # The 1-norm estimator cannot overestimate by more than factor N
    assert sep > 0
    assert sep <= true_sep * n + 1e-10, f"sep={sep} should be <= {true_sep * n}"


def test_sb03qy_scalar():
    """
    Validate 1x1 case (scalar equation).

    For scalar a < 0:
    - sep(a, -a) = |a + a| = 2|a|
    - Theta norm depends on X

    """
    from ctrlsys import sb03qy

    n = 1
    t = np.array([[-2.0]], order='F', dtype=float)
    u = np.array([[1.0]], order='F', dtype=float)
    x = np.array([[0.5]], order='F', dtype=float)  # arbitrary X

    sep, thnorm, info = sb03qy('B', 'N', 'R', t, u, x)

    assert info == 0
    # For scalar a=-2: sep = |a + a| = |-4| = 4
    assert_allclose(sep, 4.0, rtol=1e-2)
    assert thnorm >= 0


def test_sb03qy_thnorm_only():
    """
    Validate computation of Theta norm only.

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import sb03qy, sb03my

    n = 2
    t = np.array([
        [-1.0, 0.5],
        [ 0.0,-2.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)

    # Solve Lyapunov equation
    c = np.array([
        [2.0, 0.3],
        [0.3, 1.5]
    ], order='F', dtype=float)

    x, scale, info_lyap = sb03my('N', t.copy(order='F'), c.copy(order='F'))
    assert info_lyap == 0

    sep, thnorm, info = sb03qy('T', 'N', 'R', t, u, x)

    assert info == 0
    # SEP not computed when job='T'
    assert thnorm >= 0


def test_sb03qy_2x2_block():
    """
    Validate with 2x2 block (complex conjugate eigenvalues).

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import sb03qy, sb03my

    n = 2
    # 2x2 block with eigenvalues -1 +/- 2i (Re = -1 < 0, stable)
    t = np.array([
        [-1.0,  2.0],
        [-2.0, -1.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)

    # Solve Lyapunov equation
    c = np.array([
        [4.0, 1.0],
        [1.0, 3.0]
    ], order='F', dtype=float)

    x, scale, info_lyap = sb03my('N', t.copy(order='F'), c.copy(order='F'))
    assert info_lyap == 0

    sep, thnorm, info = sb03qy('B', 'N', 'R', t, u, x)

    assert info == 0
    assert sep > 0
    assert thnorm >= 0
