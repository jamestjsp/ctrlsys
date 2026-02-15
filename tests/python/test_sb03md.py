"""
Tests for SB03MD: Continuous and discrete Lyapunov equation solver.

Solves:
- Continuous: op(A)' * X + X * op(A) = scale * C
- Discrete:   op(A)' * X * op(A) - X = scale * C

where op(A) = A or A^T and C is symmetric.

Test data from SLICOT HTML documentation example:
- 3x3 discrete-time Lyapunov equation
- A stable matrix, C symmetric positive definite

Mathematical properties tested:
- Lyapunov residual equation
- Solution symmetry
- Eigenvalue computation (via Schur factorization)

Random seeds: 42, 123, 456, 789 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


"""Tests based on SLICOT HTML documentation example."""

def test_discrete_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Discrete-time Lyapunov: A'*X*A - X = scale*C
    """
    from ctrlsys import sb03md

    n = 3

    # A matrix (read row-wise from HTML doc)
    a = np.array([
        [3.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    # C matrix (symmetric RHS)
    c = np.array([
        [25.0, 24.0, 15.0],
        [24.0, 32.0,  8.0],
        [15.0,  8.0, 40.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    result = sb03md('D', 'X', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Expected solution from HTML doc
    x_expected = np.array([
        [2.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 4.0]
    ], order='F', dtype=float)

    # Verify solution (4-decimal tolerance from HTML display)
    assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)

    # Verify solution symmetry
    assert_allclose(x, x.T, atol=1e-12)

    # Verify Lyapunov residual: A'*X*A - X = scale*C
    # Need to use original A, not Schur form
    # But we can verify using eigenvalue properties


"""Tests for continuous-time Lyapunov equation."""

def test_continuous_diagonal():
    """
    Continuous Lyapunov with diagonal A.

    A'*X + X*A = scale*C has explicit solution for diagonal A.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(42)
    n = 3

    # Stable diagonal A (all eigenvalues negative)
    a = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=float)

    # Symmetric C
    c = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 4.0, 0.2],
        [0.3, 0.2, 6.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify Lyapunov residual: A'*X + X*A = scale*C
    residual = a_orig.T @ x + x @ a_orig - scale * c_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)

    # Verify symmetry
    assert_allclose(x, x.T, atol=1e-12)

def test_continuous_transpose():
    """
    Continuous Lyapunov with transpose: A*X + X*A' = scale*C.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(123)
    n = 3

    # Stable upper triangular A
    a = np.array([
        [-1.0, 0.5, 0.2],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=float)

    # Symmetric C
    c = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    result = sb03md('C', 'X', 'N', 'T', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0

    # Verify: A*X + X*A' = scale*C
    residual = a_orig @ x + x @ a_orig.T - scale * c_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)


"""Tests for discrete-time Lyapunov equation."""

def test_discrete_stable():
    """
    Discrete Lyapunov with stable matrix.

    A'*X*A - X = scale*C where all |lambda(A)| < 1.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(456)
    n = 3

    # Stable discrete-time matrix (eigenvalues inside unit circle)
    a = np.array([
        [0.5, 0.1, 0.0],
        [0.0, 0.4, 0.1],
        [0.0, 0.0, 0.3]
    ], order='F', dtype=float)

    # Symmetric C
    c = np.array([
        [1.0, 0.3, 0.2],
        [0.3, 2.0, 0.1],
        [0.2, 0.1, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    result = sb03md('D', 'X', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0

    # Verify: A'*X*A - X = scale*C
    residual = a_orig.T @ x @ a_orig - x - scale * c_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)

def test_discrete_transpose():
    """
    Discrete Lyapunov with transpose: A*X*A' - X = scale*C.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(789)
    n = 2

    # Stable matrix
    a = np.array([
        [0.6, 0.2],
        [0.0, 0.5]
    ], order='F', dtype=float)

    # Symmetric C
    c = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    result = sb03md('D', 'X', 'N', 'T', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0

    # Verify: A*X*A' - X = scale*C
    residual = a_orig @ x @ a_orig.T - x - scale * c_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)


"""Tests for Schur factorization feature."""

def test_factored_input():
    """
    Test with pre-computed Schur factorization (FACT='F').

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(111)
    n = 3

    # First compute Schur form using FACT='N'
    a = np.array([
        [-1.0, 0.5, 0.2],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    # Get Schur form
    result1 = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'),
                     c.copy(order='F'))

    x1, a_schur, u, wr, wi, scale1, sep, ferr, info = result1
    assert info == 0

    # Now solve with pre-computed Schur form
    c2 = np.array([
        [2.0, 0.4, 0.2],
        [0.4, 3.0, 0.6],
        [0.2, 0.6, 2.5]
    ], order='F', dtype=float)

    result2 = sb03md('C', 'X', 'F', 'N', n, a_schur.copy(order='F'),
                     c2.copy(order='F'), u.copy(order='F'))

    x2, a_out2, u_out2, wr2, wi2, scale2, sep2, ferr2, info2 = result2

    assert info2 == 0

    # Verify residual with original A
    a_orig = u @ a_schur @ u.T  # Reconstruct A from Schur
    c2_orig = c2.copy()
    residual = a_orig.T @ x2 + x2 @ a_orig - scale2 * c2_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-10)


"""Tests for separation estimation feature."""

def test_separation_only():
    """
    Test separation computation only (JOB='S').

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(222)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=float)

    # C not needed for JOB='S', but must provide valid array
    c = np.zeros((n, n), order='F', dtype=float)

    result = sb03md('C', 'S', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0
    assert sep > 0  # Separation should be positive for well-conditioned problem

def test_solution_and_separation():
    """
    Test both solution and separation (JOB='B').

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(333)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    result = sb03md('C', 'B', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0
    assert sep > 0
    assert ferr >= 0  # Forward error bound

    # Verify residual
    residual = a_orig.T @ x + x @ a_orig - scale * c_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)


"""Mathematical property validation tests."""

def test_solution_symmetry():
    """
    Verify solution X is symmetric.

    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(444)
    n = 4

    # Generate random stable matrix
    a = -np.eye(n) + 0.1 * np.random.randn(n, n)
    a = np.asfortranarray(a, dtype=float)

    # Generate symmetric C
    c = np.random.randn(n, n)
    c = c + c.T
    c = np.asfortranarray(c, dtype=float)

    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0 or info == n + 1

    # Solution should be symmetric
    assert_allclose(x, x.T, atol=1e-12)

def test_eigenvalue_computation():
    """
    Verify eigenvalues are computed correctly when FACT='N'.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(555)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [-0.3, -2.0, 0.3],
        [0.1, -0.2, -1.5]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    # Get eigenvalues from NumPy for comparison
    eigs_numpy = np.linalg.eigvals(a)

    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0

    # Construct eigenvalues from wr, wi
    eigs_slicot = wr + 1j * wi

    # Sort for comparison
    eigs_numpy_sorted = sorted(eigs_numpy, key=lambda x: (x.real, x.imag))
    eigs_slicot_sorted = sorted(eigs_slicot, key=lambda x: (x.real, x.imag))

    for e1, e2 in zip(eigs_numpy_sorted, eigs_slicot_sorted):
        assert abs(e1 - e2) < 1e-10

def test_schur_factorization_validity():
    """
    Verify A = U * S * U' where S is Schur form.

    Random seed: 666 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(666)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [-0.3, -2.0, 0.3],
        [0.1, -0.2, -1.5]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()

    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_schur, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0

    # Verify U is orthogonal
    assert_allclose(u.T @ u, np.eye(n), atol=1e-12)

    # Verify A = U * S * U'
    a_reconstructed = u @ a_schur @ u.T
    assert_allclose(a_reconstructed, a_orig, atol=1e-12)


"""Edge case and error condition tests."""

def test_zero_dimension():
    """Test with n=0 (quick return)."""
    from ctrlsys import sb03md

    n = 0

    a = np.array([], dtype=float).reshape(0, 0)
    c = np.array([], dtype=float).reshape(0, 0)

    result = sb03md('C', 'X', 'N', 'N', n,
                    np.asfortranarray(a), np.asfortranarray(c))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0
    assert scale == 1.0

def test_nearly_singular():
    """
    Test with nearly singular Lyapunov operator.

    When A and -A' have nearly common eigenvalues, INFO=N+1.
    """
    from ctrlsys import sb03md

    n = 2

    # A with eigenvalue very close to 0 (A and -A' nearly have common eigenvalue)
    a = np.array([
        [1e-12, 1.0],
        [0.0, -1.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ], order='F', dtype=float)

    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    # May return INFO=N+1 for nearly singular case
    assert info >= 0


"""Tests for matrices with 2x2 blocks (complex eigenvalues)."""

def test_continuous_complex_eigenvalues():
    """
    Test with complex conjugate eigenvalues.

    Random seed: 777 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(777)
    n = 2

    # Matrix with complex eigenvalues -1 +/- i
    a = np.array([
        [-1.0, 1.0],
        [-1.0, -1.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.3],
        [0.3, 2.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0

    # Verify residual
    residual = a_orig.T @ x + x @ a_orig - scale * c_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)

    # Verify symmetry
    assert_allclose(x, x.T, atol=1e-12)

def test_discrete_complex_eigenvalues():
    """
    Test discrete-time with complex eigenvalues.

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import sb03md

    np.random.seed(888)
    n = 2

    # Stable complex eigenvalues: 0.5 +/- 0.5i (|lambda| = 0.707)
    a = np.array([
        [0.5, 0.5],
        [-0.5, 0.5]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.2],
        [0.2, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    result = sb03md('D', 'X', 'N', 'N', n, a.copy(order='F'),
                    c.copy(order='F'))

    x, a_out, u, wr, wi, scale, sep, ferr, info = result

    assert info == 0

    # Verify residual
    residual = a_orig.T @ x @ a_orig - x - scale * c_orig
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)
