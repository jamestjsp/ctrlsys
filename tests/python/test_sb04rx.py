"""
Tests for SB04RX: Solve quasi-Hessenberg system with two right-hand sides.

This routine solves a system of equations in quasi-Hessenberg form
(Hessenberg form plus two consecutive offdiagonals) with two right-hand sides.

The system matrix is formed as:
    H = I kron A * LAMBDA + LAMBDA kron A
where LAMBDA is the 2x2 block [[lambd1, lambd2], [lambd3, lambd4]]
and A is the Hessenberg matrix. The system H*x = d is solved via QR
decomposition with Givens rotations.

Parameters:
- rc: 'R' for row transformations, 'C' for column transformations
- ul: 'U' if A is upper Hessenberg, 'L' if lower Hessenberg
- m: Order of matrix A
- a: M-by-M Hessenberg matrix
- lambd1, lambd2, lambd3, lambd4: 2x2 block elements
- d: 2*M RHS vector (input), solution vector (output), stored row-wise
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


def build_quasi_hessenberg_system(a, lambd1, lambd2, lambd3, lambd4, ul='U'):
    """
    Build the 2M x 2M quasi-Hessenberg system matrix H.

    For upper Hessenberg A:
        H[2i-1:2i, 2j-1:2j] = A[i,j] * LAMBDA + delta_{i,j} * I

    where LAMBDA = [[lambd1, lambd2], [lambd3, lambd4]] and delta is Kronecker delta.
    """
    m = a.shape[0]
    if m == 0:
        return np.zeros((0, 0), dtype=float, order='F')

    m2 = 2 * m
    H = np.zeros((m2, m2), dtype=float, order='F')

    for j in range(m):
        for i in range(m):
            if ul == 'U':
                if i <= j + 1:  # Upper Hessenberg: main + lower diagonal
                    val = a[i, j]
                else:
                    val = 0.0
            else:  # Lower Hessenberg
                if i >= j - 1:  # Main + upper diagonal
                    val = a[i, j]
                else:
                    val = 0.0

            i2 = 2 * i
            j2 = 2 * j
            H[i2, j2] = val * lambd1
            H[i2, j2 + 1] = val * lambd2
            H[i2 + 1, j2] = val * lambd3
            H[i2 + 1, j2 + 1] = val * lambd4

    for j in range(m):
        j2 = 2 * j
        H[j2, j2] += 1.0
        H[j2 + 1, j2 + 1] += 1.0

    return H


def test_sb04rx_upper_row_basic():
    """
    Validate basic functionality with RC='R', UL='U' (upper Hessenberg, row transforms).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb04rx

    np.random.seed(42)
    m = 4

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)  # Upper Hessenberg
    lambd1 = 0.5
    lambd2 = 0.2
    lambd3 = -0.1
    lambd4 = 0.8

    d_init = np.random.randn(2 * m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04rx('R', 'U', a, lambd1, lambd2, lambd3, lambd4, d, tol)

    assert info == 0

    H = build_quasi_hessenberg_system(a, lambd1, lambd2, lambd3, lambd4, 'U')
    d_reconstructed = H @ d_out

    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04rx_upper_col_basic():
    """
    Validate with RC='C', UL='U' (upper Hessenberg, column transforms).

    Column transforms solve H^T * x = d (transpose system), not H * x = d.
    This is because column Givens rotations transform both matrix and RHS,
    resulting in a solution to the transposed system.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb04rx

    np.random.seed(123)
    m = 3

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    lambd1 = 1.2
    lambd2 = -0.3
    lambd3 = 0.4
    lambd4 = 1.5

    d_init = np.random.randn(2 * m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04rx('C', 'U', a, lambd1, lambd2, lambd3, lambd4, d, tol)

    assert info == 0

    H = build_quasi_hessenberg_system(a, lambd1, lambd2, lambd3, lambd4, 'U')
    d_reconstructed = H.T @ d_out

    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04rx_lower_row_basic():
    """
    Validate with RC='R', UL='L' (lower Hessenberg, row transforms).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb04rx

    np.random.seed(456)
    m = 4

    a = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)  # Lower Hessenberg
    lambd1 = 0.3
    lambd2 = 0.6
    lambd3 = 0.1
    lambd4 = 0.9

    d_init = np.random.randn(2 * m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04rx('R', 'L', a, lambd1, lambd2, lambd3, lambd4, d, tol)

    assert info == 0

    H = build_quasi_hessenberg_system(a, lambd1, lambd2, lambd3, lambd4, 'L')
    d_reconstructed = H @ d_out

    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04rx_lower_col_basic():
    """
    Validate with RC='C', UL='L' (lower Hessenberg, column transforms).

    Column transforms solve H^T * x = d (transpose system), not H * x = d.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb04rx

    np.random.seed(789)
    m = 3

    a = np.tril(np.random.randn(m, m).astype(float, order='F'), k=1)
    lambd1 = 2.0
    lambd2 = -0.5
    lambd3 = 0.7
    lambd4 = 1.8

    d_init = np.random.randn(2 * m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04rx('C', 'L', a, lambd1, lambd2, lambd3, lambd4, d, tol)

    assert info == 0

    H = build_quasi_hessenberg_system(a, lambd1, lambd2, lambd3, lambd4, 'L')
    d_reconstructed = H.T @ d_out

    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04rx_m_zero():
    """
    Test edge case M=0 (quick return).
    """
    from ctrlsys import sb04rx

    m = 0
    a = np.zeros((0, 0), dtype=float, order='F')
    d = np.zeros(0, dtype=float, order='F')

    d_out, info = sb04rx('R', 'U', a, 1.0, 0.0, 0.0, 1.0, d, 1e-12)

    assert info == 0
    assert d_out.shape == (0,)


def test_sb04rx_m_one():
    """
    Test smallest non-trivial case M=1.

    For M=1, the system becomes 2x2:
    H = [[a*lambd1 + 1, a*lambd2],
         [a*lambd3, a*lambd4 + 1]]

    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import sb04rx

    np.random.seed(101)
    m = 1

    a = np.array([[2.5]], dtype=float, order='F')
    lambd1 = 0.4
    lambd2 = 0.2
    lambd3 = 0.1
    lambd4 = 0.6

    d_init = np.array([1.0, 2.0], dtype=float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04rx('R', 'U', a, lambd1, lambd2, lambd3, lambd4, d, tol)

    assert info == 0

    H = np.array([
        [a[0, 0] * lambd1 + 1, a[0, 0] * lambd2],
        [a[0, 0] * lambd3, a[0, 0] * lambd4 + 1]
    ], dtype=float, order='F')

    d_reconstructed = H @ d_out
    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


def test_sb04rx_singular_system():
    """
    Test error handling when system is nearly singular.

    Set up a system that results in a singular or nearly singular
    quasi-Hessenberg matrix.
    """
    from ctrlsys import sb04rx

    m = 2
    a = np.eye(m, dtype=float, order='F')
    lambd1 = -1.0
    lambd2 = 0.0
    lambd3 = 0.0
    lambd4 = -1.0

    d = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')

    tol = 1e-12

    d_out, info = sb04rx('R', 'U', a, lambd1, lambd2, lambd3, lambd4, d, tol)

    assert info == 1


def test_sb04rx_solution_property():
    """
    Mathematical property test: verify H @ x = d.

    Tests that the solution x satisfies the original linear system exactly.

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import sb04rx

    np.random.seed(202)
    m = 5

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    lambd1 = np.random.randn()
    lambd2 = np.random.randn()
    lambd3 = np.random.randn()
    lambd4 = np.random.randn()

    d_init = np.random.randn(2 * m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04rx('R', 'U', a, lambd1, lambd2, lambd3, lambd4, d, tol)

    if info == 0:
        H = build_quasi_hessenberg_system(a, lambd1, lambd2, lambd3, lambd4, 'U')

        residual = H @ d_out - d_init
        residual_norm = np.linalg.norm(residual) / np.linalg.norm(d_init)

        assert residual_norm < 1e-10


def test_sb04rx_identity_lambda():
    """
    Test with identity LAMBDA block (lambd1=lambd4=1, lambd2=lambd3=0).

    This simplifies to (I + A*I) x = d, where the system matrix is
    identity-like structure with A on diagonal blocks.

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import sb04rx

    np.random.seed(303)
    m = 3

    a = np.triu(np.random.randn(m, m).astype(float, order='F'), k=-1)
    a = a * 0.1  # Scale to ensure well-conditioned

    lambd1 = 1.0
    lambd2 = 0.0
    lambd3 = 0.0
    lambd4 = 1.0

    d_init = np.random.randn(2 * m).astype(float, order='F')
    d = d_init.copy()

    tol = 1e-12

    d_out, info = sb04rx('R', 'U', a, lambd1, lambd2, lambd3, lambd4, d, tol)

    assert info == 0

    H = build_quasi_hessenberg_system(a, lambd1, lambd2, lambd3, lambd4, 'U')
    d_reconstructed = H @ d_out

    assert_allclose(d_reconstructed, d_init, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
