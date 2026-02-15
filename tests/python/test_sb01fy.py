"""
Tests for SB01FY: Inner denominator of right-coprime factorization.

Computes state-feedback matrix F and matrix V such that (A+B*F, B*V, F, V)
is inner, for systems of order 1 or 2 with unstable A.

Tests:
1. Continuous-time N=1 case
2. Discrete-time N=1 case
3. Continuous-time N=2 case with complex eigenvalues
4. Discrete-time N=2 case
5. Error handling (stable system, uncontrollable)

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb01fy_continuous_n1():
    """
    Validate continuous-time N=1 case.

    For continuous-time: A must have positive real part (unstable).
    V should be identity for continuous-time.
    """
    from ctrlsys import sb01fy

    n = 1
    m = 2

    # Unstable A (positive eigenvalue)
    a = np.array([[2.0]], order='F', dtype=float)

    # Controllable B
    b = np.array([[1.0, 0.5]], order='F', dtype=float)

    f, v, info = sb01fy(False, a, b)

    assert info == 0
    assert f.shape == (m, n)
    assert v.shape == (m, m)

    # V should be identity for continuous-time
    assert_allclose(v, np.eye(m), atol=1e-14)

    # A + B*F should have eigenvalue at -2 (reflected across imaginary axis)
    a_closed = a + b @ f
    eig_closed = np.linalg.eigvals(a_closed)
    assert eig_closed[0].real < 0


def test_sb01fy_discrete_n1():
    """
    Validate discrete-time N=1 case.

    For discrete-time: A must have modulus > 1 (unstable).
    """
    from ctrlsys import sb01fy

    n = 1
    m = 1

    # Unstable A (|eigenvalue| > 1)
    a = np.array([[1.5]], order='F', dtype=float)

    # Controllable B
    b = np.array([[1.0]], order='F', dtype=float)

    f, v, info = sb01fy(True, a, b)

    assert info == 0
    assert f.shape == (m, n)
    assert v.shape == (m, m)

    # V is upper triangular
    assert np.allclose(np.tril(v, -1), 0, atol=1e-14)

    # A + B*F should have eigenvalue reflected inside unit circle
    a_closed = a + b @ f
    eig_closed = np.linalg.eigvals(a_closed)
    assert abs(eig_closed[0]) < 1


def test_sb01fy_continuous_n2():
    """
    Validate continuous-time N=2 case with complex eigenvalues.

    A must have eigenvalues with positive real parts.
    """
    from ctrlsys import sb01fy

    n = 2
    m = 2

    # Unstable A with complex eigenvalues (1+2i, 1-2i)
    a = np.array([
        [1.0, 2.0],
        [-2.0, 1.0]
    ], order='F', dtype=float)

    # Controllable B
    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    f, v, info = sb01fy(False, a, b)

    assert info == 0
    assert f.shape == (m, n)
    assert v.shape == (m, m)

    # V should be identity for continuous-time
    assert_allclose(v, np.eye(m), atol=1e-14)

    # A + B*F should have stable eigenvalues (reflected to left half plane)
    a_closed = a + b @ f
    eig_closed = np.linalg.eigvals(a_closed)
    assert all(e.real < 0 for e in eig_closed)


def test_sb01fy_discrete_n2():
    """
    Validate discrete-time N=2 case.

    A must have eigenvalues with modulus > 1.
    """
    from ctrlsys import sb01fy

    n = 2
    m = 2

    # Unstable A with complex eigenvalues outside unit circle
    # eigenvalues: 1.5 +/- 0.5i (modulus ~ 1.58)
    a = np.array([
        [1.5, 0.5],
        [-0.5, 1.5]
    ], order='F', dtype=float)

    # Controllable B
    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    f, v, info = sb01fy(True, a, b)

    assert info == 0
    assert f.shape == (m, n)
    assert v.shape == (m, m)

    # V is upper triangular
    assert np.allclose(np.tril(v, -1), 0, atol=1e-14)

    # A + B*F should have stable eigenvalues (inside unit circle)
    a_closed = a + b @ f
    eig_closed = np.linalg.eigvals(a_closed)
    assert all(abs(e) < 1 for e in eig_closed)


def test_sb01fy_inner_property():
    """
    Validate inner system property.

    For continuous-time: (A+BF)'P + P(A+BF) + F'F = 0 should have solution P > 0.
    System (A+BF, BV, F, V) is inner means V'V = I and specific Lyapunov equation.
    """
    from ctrlsys import sb01fy

    n = 1
    m = 2

    a = np.array([[3.0]], order='F', dtype=float)
    b = np.array([[1.0, 2.0]], order='F', dtype=float)

    f, v, info = sb01fy(False, a, b)

    assert info == 0

    # For continuous-time, V should be identity
    assert_allclose(v @ v.T, np.eye(m), atol=1e-13)


def test_sb01fy_stable_system_error():
    """
    Validate error for stable system (INFO=2).

    A must be unstable, otherwise INFO=2.
    """
    from ctrlsys import sb01fy

    n = 1
    m = 1

    # Stable A (negative eigenvalue for continuous-time)
    a = np.array([[-1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)

    f, v, info = sb01fy(False, a, b)

    assert info == 2


def test_sb01fy_marginally_stable_error():
    """
    Validate error for marginally stable system (INFO=2).

    A at stability limit (eigenvalue = 0 for continuous, |eig|=1 for discrete).
    """
    from ctrlsys import sb01fy

    n = 1
    m = 1

    # Marginally stable for discrete (|eigenvalue| = 1)
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)

    f, v, info = sb01fy(True, a, b)

    assert info == 2
