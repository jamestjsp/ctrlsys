"""
Tests for SB04RY: Solve Hessenberg system with one right-hand side.

This routine solves a system of equations in Hessenberg form with one RHS.
The system matrix is: H = I + LAMBDA * A
where A is the Hessenberg matrix. The system H*x = d is solved via QR
decomposition with Givens rotations.

Parameters:
- rc: 'R' for row transformations, 'C' for column transformations
- ul: 'U' if A is upper Hessenberg, 'L' if lower Hessenberg
- m: Order of matrix A
- a: M-by-M Hessenberg matrix
- lambd: Scalar multiplier for A
- d: M-element RHS vector (input), solution vector (output)
- tol: Tolerance for near-singularity detection

Tests numerical correctness using:
1. Basic functionality with upper Hessenberg, row transformations
2. Upper Hessenberg with column transformations
3. Lower Hessenberg cases
4. Edge case: M=0 (quick return)
5. Error handling: singular system (info=1)
6. Mathematical property: verify solution satisfies original system
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def build_hessenberg_system(a, lambd, ul='U'):
    """
    Build the M x M Hessenberg system matrix H = I + LAMBDA * A.

    For upper Hessenberg A (UL='U'):
        A has zeros below the first subdiagonal

    For lower Hessenberg A (UL='L'):
        A has zeros above the first superdiagonal
    """
    m = a.shape[0]
    if m == 0:
        return np.zeros((0, 0), dtype=float, order='F')

    H = np.eye(m, dtype=float, order='F') + lambd * a
    return H


def test_sb04ry_upper_row_basic():
    """
    Validate basic functionality with RC='R', UL='U' (upper Hessenberg, row transforms).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04ry

    np.random.seed(42)
    m = 4

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)  # Upper Hessenberg
    lambd = 0.5

    d_init = np.random.randn(m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04ry('R', 'U', a, lambd, d, tol)

    assert info == 0

    H = build_hessenberg_system(a, lambd, 'U')
    d_reconstructed = H @ d_out

    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04ry_upper_col_basic():
    """
    Validate with RC='C', UL='U' (upper Hessenberg, column transforms).

    Column transforms solve H^T * x = d (transpose system), not H * x = d.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04ry

    np.random.seed(123)
    m = 3

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    lambd = 1.2

    d_init = np.random.randn(m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04ry('C', 'U', a, lambd, d, tol)

    assert info == 0

    H = build_hessenberg_system(a, lambd, 'U')
    d_reconstructed = H.T @ d_out

    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04ry_lower_row_basic():
    """
    Validate with RC='R', UL='L' (lower Hessenberg, row transforms).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04ry

    np.random.seed(456)
    m = 4

    a = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)  # Lower Hessenberg
    lambd = 0.3

    d_init = np.random.randn(m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04ry('R', 'L', a, lambd, d, tol)

    assert info == 0

    H = build_hessenberg_system(a, lambd, 'L')
    d_reconstructed = H @ d_out

    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04ry_lower_col_basic():
    """
    Validate with RC='C', UL='L' (lower Hessenberg, column transforms).

    Column transforms solve H^T * x = d (transpose system).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04ry

    np.random.seed(789)
    m = 3

    a = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)
    lambd = 2.0

    d_init = np.random.randn(m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04ry('C', 'L', a, lambd, d, tol)

    assert info == 0

    H = build_hessenberg_system(a, lambd, 'L')
    d_reconstructed = H.T @ d_out

    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04ry_m_zero():
    """
    Test edge case M=0 (quick return).
    """
    from ctrlsys import sb04ry

    m = 0
    a = np.zeros((0, 0), dtype=float, order='F')
    d = np.zeros(0, dtype=float, order='F')

    d_out, info = sb04ry('R', 'U', a, 1.0, d, 1e-12)

    assert info == 0
    assert d_out.shape == (0,)


def test_sb04ry_m_one():
    """
    Test smallest non-trivial case M=1.

    For M=1, the system becomes scalar: (1 + lambda*a) * x = d

    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import sb04ry

    np.random.seed(101)
    m = 1

    a = np.array([[2.5]], dtype=float, order='F')
    lambd = 0.4

    d_init = np.array([1.0], dtype=float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04ry('R', 'U', a, lambd, d, tol)

    assert info == 0

    H = np.array([[1.0 + lambd * a[0, 0]]], dtype=float, order='F')
    d_reconstructed = H @ d_out
    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04ry_singular_system():
    """
    Test error handling when system is nearly singular.

    Set lambda = -1/a so that I + lambda*A becomes singular.
    """
    from ctrlsys import sb04ry

    m = 2
    a = np.eye(m, dtype=float, order='F')
    lambd = -1.0  # I + (-1)*I = 0 (singular)

    d = np.array([1.0, 2.0], dtype=float, order='F')

    tol = 1e-12

    d_out, info = sb04ry('R', 'U', a, lambd, d, tol)

    assert info == 1


def test_sb04ry_solution_property():
    """
    Mathematical property test: verify H @ x = d.

    Tests that the solution x satisfies the original linear system exactly.

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import sb04ry

    np.random.seed(202)
    m = 5

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    lambd = np.random.randn()

    d_init = np.random.randn(m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04ry('R', 'U', a, lambd, d, tol)

    if info == 0:
        H = build_hessenberg_system(a, lambd, 'U')

        residual = H @ d_out - d_init
        residual_norm = np.linalg.norm(residual) / np.linalg.norm(d_init)

        assert residual_norm < 1e-10


def test_sb04ry_identity_case():
    """
    Test with lambda=0 (system reduces to identity).

    When lambda=0, H = I, so x = d exactly.

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import sb04ry

    np.random.seed(303)
    m = 3

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    lambd = 0.0

    d_init = np.random.randn(m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04ry('R', 'U', a, lambd, d, tol)

    assert info == 0
    assert_allclose(d_out, d_init, rtol=1e-14, atol=1e-15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
