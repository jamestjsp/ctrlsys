"""
Tests for SB01BY: Pole placement for N=1 or N=2 systems.

Constructs feedback matrix F such that A + B*F has prescribed eigenvalues.
Eigenvalues specified by sum S and product P (for N=2).
"""
import numpy as np
import pytest
from ctrlsys import sb01by


"""Basic functionality tests."""

def test_n1_single_pole():
    """
    N=1: Single pole assignment.
    A = [2], B = [1], target eigenvalue = -1
    F should satisfy: A + B*F = -1, so F = -1 - 2 = -3
    """
    n, m = 1, 1
    s = -1.0  # Target eigenvalue
    p = 0.0   # Not used for N=1
    tol = 1e-10

    a = np.array([[2.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)

    f, info = sb01by(n, m, s, p, a, b, tol)

    assert info == 0
    assert f.shape == (m, n)

    # Verify: A + B*F should have eigenvalue s
    a_orig = np.array([[2.0]], order='F', dtype=float)
    b_orig = np.array([[1.0]], order='F', dtype=float)
    closed_loop = a_orig + b_orig @ f

    np.testing.assert_allclose(closed_loop[0, 0], s, rtol=1e-13)

def test_n1_multiple_inputs():
    """
    N=1, M=3: Multiple input channels.
    """
    n, m = 1, 3
    s = 0.0  # Target eigenvalue
    p = 0.0
    tol = 1e-10

    a = np.array([[5.0]], order='F', dtype=float)
    b = np.array([[1.0, 2.0, 3.0]], order='F', dtype=float)

    f, info = sb01by(n, m, s, p, a, b, tol)

    assert info == 0
    assert f.shape == (m, n)

    # Verify closed-loop eigenvalue
    a_orig = np.array([[5.0]], order='F', dtype=float)
    b_orig = np.array([[1.0, 2.0, 3.0]], order='F', dtype=float)
    closed_loop = a_orig + b_orig @ f

    np.testing.assert_allclose(closed_loop[0, 0], s, rtol=1e-13)

def test_n2_real_poles():
    """
    N=2: Two real poles assignment.
    Target eigenvalues: -1 and -2
    Sum = -3, Product = 2
    """
    n, m = 2, 2
    s = -3.0   # Sum of eigenvalues
    p = 2.0    # Product of eigenvalues
    tol = 1e-10

    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    f, info = sb01by(n, m, s, p, a.copy(), b.copy(), tol)

    assert info == 0
    assert f.shape == (m, n)

    # Verify closed-loop eigenvalues
    closed_loop = a + b @ f
    eigs = np.linalg.eigvals(closed_loop)

    # Check sum and product
    np.testing.assert_allclose(np.sum(eigs), s, rtol=1e-10)
    np.testing.assert_allclose(np.prod(eigs), p, rtol=1e-10)

def test_n2_complex_poles():
    """
    N=2: Complex conjugate poles.
    Target eigenvalues: -1 +/- 2i
    Sum = -2, Product = 1 + 4 = 5
    """
    n, m = 2, 1
    s = -2.0   # Sum: 2 * real = 2 * (-1) = -2
    p = 5.0    # Product: real^2 + imag^2 = 1 + 4 = 5
    tol = 1e-10

    a = np.array([
        [0.0, 1.0],
        [-1.0, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    f, info = sb01by(n, m, s, p, a.copy(), b.copy(), tol)

    assert info == 0
    assert f.shape == (m, n)

    # Verify closed-loop eigenvalues
    closed_loop = a + b @ f
    eigs = np.linalg.eigvals(closed_loop)

    # Sum and product
    np.testing.assert_allclose(np.sum(eigs.real), s, rtol=1e-10)
    np.testing.assert_allclose(np.prod(eigs), p, rtol=1e-10)


"""Tests for uncontrollable systems (INFO=1)."""

def test_n1_uncontrollable():
    """
    N=1 with B effectively zero.
    """
    n, m = 1, 1
    s = 0.0
    p = 0.0
    tol = 1e-10

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1e-15]], order='F', dtype=float)  # Below tolerance

    f, info = sb01by(n, m, s, p, a, b, tol)

    assert info == 1  # Uncontrollable

def test_n2_uncontrollable():
    """
    N=2 with uncontrollable pair.
    """
    n, m = 2, 1
    s = 0.0
    p = 0.0
    tol = 1e-10

    # Uncontrollable: B in null space of observability
    a = np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0],
        [0.0]
    ], order='F', dtype=float)

    f, info = sb01by(n, m, s, p, a.copy(), b.copy(), tol)

    assert info == 1  # Uncontrollable


"""Mathematical property validation tests."""

def test_minimum_frobenius_norm():
    """
    Verify F has minimum Frobenius norm (for overdetermined case).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m = 1, 3
    s = 0.0
    p = 0.0
    tol = 1e-12

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)

    f, info = sb01by(n, m, s, p, a.copy(), b.copy(), tol)

    assert info == 0

    # Closed-loop eigenvalue should be s
    a_orig = np.array([[1.0]])
    b_orig = np.array([[1.0, 1.0, 1.0]])
    closed_loop = a_orig + b_orig @ f

    np.testing.assert_allclose(closed_loop[0, 0], s, rtol=1e-12)

    # F should have equal elements (minimum norm solution)
    # B*F = s - a = -1, so sum(f) = -1, minimum norm: f = [-1/3, -1/3, -1/3]
    expected_f = np.array([[-1.0/3], [-1.0/3], [-1.0/3]])
    np.testing.assert_allclose(f, expected_f, rtol=1e-12)

def test_eigenvalue_equations():
    """
    Verify closed-loop satisfies characteristic polynomial.

    For N=2: det(sI - A - BF) = s^2 - (sum)*s + product = 0

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m = 2, 2
    s_sum = -4.0  # Sum of desired eigenvalues
    p_prod = 5.0  # Product of desired eigenvalues
    tol = 1e-12

    a = np.array([
        [0.0, 1.0],
        [-2.0, -3.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ], order='F', dtype=float)

    f, info = sb01by(n, m, s_sum, p_prod, a.copy(), b.copy(), tol)

    assert info == 0

    # Verify eigenvalues of closed-loop
    closed_loop = a + b @ f
    eigs = np.linalg.eigvals(closed_loop)

    # Characteristic polynomial: s^2 - trace*s + det = 0
    trace = np.trace(closed_loop)
    det = np.linalg.det(closed_loop)

    np.testing.assert_allclose(trace, s_sum, rtol=1e-10)
    np.testing.assert_allclose(det, p_prod, rtol=1e-10)

def test_state_feedback_stability():
    """
    Verify pole placement achieves stability.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m = 2, 1
    # Place poles at -1 +/- i (stable)
    s_sum = -2.0
    p_prod = 2.0  # (-1)^2 + 1^2 = 2
    tol = 1e-12

    # Unstable open-loop system
    a = np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [1.0]
    ], order='F', dtype=float)

    f, info = sb01by(n, m, s_sum, p_prod, a.copy(), b.copy(), tol)

    assert info == 0

    # Verify closed-loop is stable
    closed_loop = a + b @ f
    eigs = np.linalg.eigvals(closed_loop)

    # All eigenvalues should have negative real part
    assert all(e.real < 0 for e in eigs)
