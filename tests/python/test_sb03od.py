"""
Tests for SB03OD - Lyapunov equation solver computing Cholesky factor.

SB03OD solves for X = op(U)'*op(U) either the stable continuous-time Lyapunov equation:
    op(A)'*X + X*op(A) = -scale^2*op(B)'*op(B)
or the convergent discrete-time Lyapunov equation:
    op(A)'*X*op(A) - X = -scale^2*op(B)'*op(B)

where A is N-by-N, op(B) is M-by-N, U is upper triangular (Cholesky factor).
"""
import numpy as np
import pytest
import ctrlsys


"""Test using the example from SLICOT HTML documentation."""

def test_continuous_nofact_notrans():
    """
    Test continuous-time, compute Schur factorization, no transpose.

    Example data from SLICOT HTML doc for SB03OD.
    N=4, M=5, DICO='C', FACT='N', TRANS='N'
    """
    n, m = 4, 5

    a = np.array([
        [-1.0, 37.0, -12.0, -12.0],
        [-1.0, -10.0, 0.0, 4.0],
        [2.0, -4.0, 7.0, -6.0],
        [2.0, 2.0, 7.0, -9.0]
    ], order='F', dtype=np.float64)

    b = np.array([
        [1.0, 2.5, 1.0, 3.5],
        [0.0, 1.0, 0.0, 1.0],
        [-1.0, -2.5, -1.0, -1.5],
        [1.0, 2.5, 4.0, -5.5],
        [-1.0, -2.5, -4.0, 3.5]
    ], order='F', dtype=np.float64)

    u_expected = np.array([
        [1.0, 3.0, 2.0, -1.0],
        [0.0, 1.0, -1.0, 1.0],
        [0.0, 0.0, 1.0, -2.0],
        [0.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=np.float64)

    u, q, wr, wi, scale, info = slicot.sb03od('C', 'N', 'N', a, b)

    assert info == 0
    assert scale == pytest.approx(1.0, abs=1e-4)

    u_result = np.triu(u[:n, :n])
    np.testing.assert_allclose(u_result, u_expected, rtol=1e-3, atol=1e-4)


"""Test Lyapunov equation residual properties."""

def test_continuous_residual_verification():
    """
    Verify A'*X + X*A = -scale^2*B'*B for continuous-time.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m = 3, 4

    a = -np.eye(n) - 0.5 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()

    b = np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    u, q, wr, wi, scale, info = slicot.sb03od('C', 'N', 'N', a, b)

    assert info == 0
    assert 0 < scale <= 1.0

    u_result = np.triu(u[:n, :n])
    x = u_result.T @ u_result

    rhs = -scale**2 * b_orig.T @ b_orig
    residual = a_orig.T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

def test_discrete_residual_verification():
    """
    Verify A'*X*A - X = -scale^2*B'*B for discrete-time.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m = 3, 4

    a = 0.5 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()

    b = np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    u, q, wr, wi, scale, info = slicot.sb03od('D', 'N', 'N', a, b)

    assert info == 0
    assert 0 < scale <= 1.0

    u_result = np.triu(u[:n, :n])
    x = u_result.T @ u_result

    rhs = -scale**2 * b_orig.T @ b_orig
    residual = a_orig.T @ x @ a_orig - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


"""Test transpose option (TRANS='T')."""

def test_continuous_transpose():
    """
    Test op(K)=K' for continuous-time.

    Equation: A*X + X*A' = -scale^2*B*B', where X = U*U'.
    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n, m = 3, 4

    a = -np.eye(n) - 0.5 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()

    b = np.random.randn(n, m)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    u, q, wr, wi, scale, info = slicot.sb03od('C', 'N', 'T', a, b)

    assert info == 0
    assert 0 < scale <= 1.0

    u_result = np.triu(u[:n, :n])
    x = u_result @ u_result.T

    rhs = -scale**2 * b_orig @ b_orig.T
    residual = a_orig @ x + x @ a_orig.T - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

def test_discrete_transpose():
    """
    Test op(K)=K' for discrete-time.

    Equation: A*X*A' - X = -scale^2*B*B', where X = U*U'.
    Random seed: 201 (for reproducibility)
    """
    np.random.seed(201)
    n, m = 3, 4

    a = 0.5 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()

    b = np.random.randn(n, m)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    u, q, wr, wi, scale, info = slicot.sb03od('D', 'N', 'T', a, b)

    assert info == 0
    assert 0 < scale <= 1.0

    u_result = np.triu(u[:n, :n])
    x = u_result @ u_result.T

    rhs = -scale**2 * b_orig @ b_orig.T
    residual = a_orig @ x @ a_orig.T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


"""Test FACT='F' (Schur factorization provided)."""

def test_continuous_schur_provided():
    """
    Test with Schur factorization already provided.

    Random seed: 300 (for reproducibility)
    """
    np.random.seed(300)
    n, m = 3, 4

    s = np.array([
        [-1.0, 0.5, 0.3],
        [0.0, -2.0, 0.2],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=np.float64)

    q_in = np.eye(n, order='F', dtype=np.float64)

    b = np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    a_orig = q_in @ s @ q_in.T

    u, q, wr, wi, scale, info = slicot.sb03od('C', 'F', 'N', s, b, q_in)

    assert info == 0
    assert 0 < scale <= 1.0

    u_result = np.triu(u[:n, :n])
    x = u_result.T @ u_result

    rhs = -scale**2 * b_orig.T @ b_orig
    residual = a_orig.T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


"""Test edge cases."""

def test_zero_rhs():
    """Test with zero RHS: U should be nearly zero."""
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.3],
        [0.0, -2.0, 0.2],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=np.float64)

    b = np.zeros((n, n), order='F', dtype=np.float64)

    u, q, wr, wi, scale, info = slicot.sb03od('C', 'N', 'N', a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(u[:n, :n]), 0.0, atol=1e-14)

def test_n_zero():
    """Test N=0: quick return."""
    n, m = 0, 3

    a = np.zeros((0, 0), order='F', dtype=np.float64)
    b = np.zeros((m, 0), order='F', dtype=np.float64)

    u, q, wr, wi, scale, info = slicot.sb03od('C', 'N', 'N', a, b)

    assert info == 0


"""Test error handling."""

def test_unstable_continuous():
    """Test unstable A (positive eigenvalue) returns info=2."""
    n, m = 2, 2

    a = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ], order='F', dtype=np.float64)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=np.float64)

    u, q, wr, wi, scale, info = slicot.sb03od('C', 'N', 'N', a, b)

    assert info == 2

def test_non_convergent_discrete():
    """Test non-convergent A (eigenvalue > 1) returns info=2."""
    n, m = 2, 2

    a = np.array([
        [2.0, 0.0],
        [0.0, 0.5]
    ], order='F', dtype=np.float64)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=np.float64)

    u, q, wr, wi, scale, info = slicot.sb03od('D', 'N', 'N', a, b)

    assert info == 2

def test_invalid_dico():
    """Test invalid DICO parameter."""
    n, m = 2, 2

    a = np.array([[-1.0, 0.0], [0.0, -1.0]], order='F', dtype=np.float64)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)

    with pytest.raises(ValueError, match="sb03od parameter error"):
        slicot.sb03od('X', 'N', 'N', a, b)


"""Test larger systems for robustness."""

def test_6x6_continuous():
    """
    Test 6x6 continuous-time.

    Random seed: 600 (for reproducibility)
    """
    np.random.seed(600)
    n, m = 6, 8

    a = -np.eye(n) - 0.3 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()

    b = np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    u, q, wr, wi, scale, info = slicot.sb03od('C', 'N', 'N', a, b)

    assert info == 0
    assert 0 < scale <= 1.0

    u_result = np.triu(u[:n, :n])
    x = u_result.T @ u_result

    rhs = -scale**2 * b_orig.T @ b_orig
    residual = a_orig.T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)

def test_8x8_discrete_transpose():
    """
    Test 8x8 discrete-time with transpose.

    Random seed: 800 (for reproducibility)
    """
    np.random.seed(800)
    n, m = 8, 6

    a = 0.3 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()

    b_orig = np.random.randn(n, m)
    b_orig = np.asfortranarray(b_orig)

    b = np.zeros((n, max(m, n)), order='F', dtype=np.float64)
    b[:n, :m] = b_orig

    u, q, wr, wi, scale, info = slicot.sb03od('D', 'N', 'T', a, b)

    assert info == 0
    assert 0 < scale <= 1.0

    u_result = np.triu(u[:n, :n])
    x = u_result @ u_result.T

    rhs = -scale**2 * b_orig @ b_orig.T
    residual = a_orig @ x @ a_orig.T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)
