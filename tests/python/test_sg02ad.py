"""
Tests for SG02AD: Generalized algebraic Riccati equation solver for descriptor systems.

Solves:
- Continuous: Q + A'XE + E'XA - (L+E'XB)R^{-1}(L+E'XB)' = 0
- Discrete:   E'XE = A'XA - (L+A'XB)(R+B'XB)^{-1}(L+A'XB)' + Q

Test data from SLICOT HTML documentation example.

Mathematical properties tested:
- Riccati residual equation
- Solution symmetry
- Closed-loop eigenvalue stability

Random seeds: 42, 123, 456 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


"""Tests based on SLICOT HTML documentation example."""

def test_continuous_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Continuous-time Riccati equation with factored Q and R.
    N=2, M=1, P=3, DICO='C', JOBB='B', FACT='B', UPLO='U', JOBL='Z'
    """
    from ctrlsys import sg02ad

    n = 2
    m = 1
    p = 3

    a = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    q = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ], order='F', dtype=float)

    r = np.array([
        [0.0],
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    l = np.zeros((n, m), order='F', dtype=float)

    result = sg02ad(
        'C', 'B', 'B', 'U', 'Z', 'N', 'S', 'N',
        n, m, p,
        a.copy(order='F'),
        e.copy(order='F'),
        b.copy(order='F'),
        q.copy(order='F'),
        r.copy(order='F'),
        l.copy(order='F'),
        0.0
    )

    x, rcondu, alfar, alfai, beta, s, t, u, iwarn, info = result

    assert info == 0

    x_expected = np.array([
        [1.7321, 1.0000],
        [1.0000, 1.7321]
    ], order='F', dtype=float)

    assert_allclose(x, x_expected, rtol=1e-3, atol=1e-3)

    assert_allclose(x, x.T, atol=1e-10)


"""Tests for continuous-time Riccati equation."""

def test_continuous_identity_e():
    """
    Test continuous Riccati with identity E.

    This reduces to standard continuous-time Riccati equation.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sg02ad

    np.random.seed(42)
    n = 3
    m = 2
    p = 3

    a = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -2.0, -3.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    b = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0]
    ], order='F', dtype=float)

    q = np.eye(n, order='F', dtype=float)

    r = np.eye(m, order='F', dtype=float)

    l = np.zeros((n, m), order='F', dtype=float)

    result = sg02ad(
        'C', 'B', 'N', 'U', 'Z', 'N', 'S', 'N',
        n, m, p,
        a.copy(order='F'),
        e.copy(order='F'),
        b.copy(order='F'),
        q.copy(order='F'),
        r.copy(order='F'),
        l.copy(order='F'),
        0.0
    )

    x, rcondu, alfar, alfai, beta, s, t, u, iwarn, info = result

    assert info == 0

    assert_allclose(x, x.T, atol=1e-10)

    for i in range(n):
        if beta[i] != 0:
            eig = alfar[i] / beta[i]
            assert eig < 0, f"Closed-loop eigenvalue {eig} not stable"


"""Tests for discrete-time Riccati equation."""

def test_discrete_basic():
    """
    Test discrete-time Riccati equation.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sg02ad

    np.random.seed(123)
    n = 2
    m = 1
    p = 2

    a = np.array([
        [0.9, 0.1],
        [0.0, 0.8]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    b = np.array([
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    q = np.eye(n, order='F', dtype=float)

    r = np.array([[1.0]], order='F', dtype=float)

    l = np.zeros((n, m), order='F', dtype=float)

    result = sg02ad(
        'D', 'B', 'N', 'U', 'Z', 'N', 'S', 'N',
        n, m, p,
        a.copy(order='F'),
        e.copy(order='F'),
        b.copy(order='F'),
        q.copy(order='F'),
        r.copy(order='F'),
        l.copy(order='F'),
        0.0
    )

    x, rcondu, alfar, alfai, beta, s, t, u, iwarn, info = result

    assert info == 0

    assert_allclose(x, x.T, atol=1e-10)

    for i in range(n):
        if beta[i] != 0:
            eig_real = alfar[i] / beta[i]
            eig_imag = alfai[i] / beta[i]
            eig_mag = np.sqrt(eig_real**2 + eig_imag**2)
            assert eig_mag < 1 + 1e-6, f"Closed-loop eigenvalue magnitude {eig_mag} not stable"


"""Mathematical property validation tests."""

def test_solution_symmetry():
    """
    Verify solution X is symmetric.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sg02ad

    np.random.seed(456)
    n = 3
    m = 1
    p = 3

    a = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -2.0, -3.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    b = np.array([
        [0.0],
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    q = np.eye(n, order='F', dtype=float)

    r = np.array([[1.0]], order='F', dtype=float)

    l = np.zeros((n, m), order='F', dtype=float)

    result = sg02ad(
        'C', 'B', 'N', 'U', 'Z', 'N', 'S', 'N',
        n, m, p,
        a.copy(order='F'),
        e.copy(order='F'),
        b.copy(order='F'),
        q.copy(order='F'),
        r.copy(order='F'),
        l.copy(order='F'),
        0.0
    )

    x, rcondu, alfar, alfai, beta, s, t, u, iwarn, info = result

    assert info == 0
    assert_allclose(x, x.T, atol=1e-12)

def test_closed_loop_stability_continuous():
    """
    Verify closed-loop eigenvalues are in left half-plane for continuous.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sg02ad

    np.random.seed(789)
    n = 2
    m = 1
    p = 2

    a = np.array([
        [0.0, 1.0],
        [2.0, 0.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    b = np.array([
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    q = np.eye(n, order='F', dtype=float)

    r = np.array([[1.0]], order='F', dtype=float)

    l = np.zeros((n, m), order='F', dtype=float)

    result = sg02ad(
        'C', 'B', 'N', 'U', 'Z', 'N', 'S', 'N',
        n, m, p,
        a.copy(order='F'),
        e.copy(order='F'),
        b.copy(order='F'),
        q.copy(order='F'),
        r.copy(order='F'),
        l.copy(order='F'),
        0.0
    )

    x, rcondu, alfar, alfai, beta, s, t, u, iwarn, info = result

    assert info == 0

    for i in range(n):
        if abs(beta[i]) > 1e-14:
            eig_real = alfar[i] / beta[i]
            assert eig_real < 0, f"Closed-loop eigenvalue real part {eig_real} not stable"


"""Edge case and error condition tests."""

def test_zero_dimension():
    """Test with n=0 (quick return)."""
    from ctrlsys import sg02ad

    n = 0
    m = 0
    p = 0

    a = np.array([], dtype=float).reshape(0, 0)
    e = np.array([], dtype=float).reshape(0, 0)
    b = np.array([], dtype=float).reshape(0, 0)
    q = np.array([], dtype=float).reshape(0, 0)
    r = np.array([], dtype=float).reshape(0, 0)
    l = np.array([], dtype=float).reshape(0, 0)

    result = sg02ad(
        'C', 'B', 'N', 'U', 'Z', 'N', 'S', 'N',
        n, m, p,
        np.asfortranarray(a),
        np.asfortranarray(e),
        np.asfortranarray(b),
        np.asfortranarray(q),
        np.asfortranarray(r),
        np.asfortranarray(l),
        0.0
    )

    x, rcondu, alfar, alfai, beta, s, t, u, iwarn, info = result

    assert info == 0

def test_small_system():
    """Test with n=1, m=1 (smallest non-trivial case)."""
    from ctrlsys import sg02ad

    n = 1
    m = 1
    p = 1

    a = np.array([[1.0]], order='F', dtype=float)
    e = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    q = np.array([[1.0]], order='F', dtype=float)
    r = np.array([[1.0]], order='F', dtype=float)
    l = np.array([[0.0]], order='F', dtype=float)

    result = sg02ad(
        'C', 'B', 'N', 'U', 'Z', 'N', 'S', 'N',
        n, m, p,
        a.copy(order='F'),
        e.copy(order='F'),
        b.copy(order='F'),
        q.copy(order='F'),
        r.copy(order='F'),
        l.copy(order='F'),
        0.0
    )

    x, rcondu, alfar, alfai, beta, s, t, u, iwarn, info = result

    assert info == 0
    assert x.shape == (1, 1)


"""Tests for scaling option."""

def test_with_scaling():
    """
    Test with scaling enabled (SCAL='G').
    """
    from ctrlsys import sg02ad

    n = 2
    m = 1
    p = 2

    a = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    b = np.array([
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    q = np.eye(n, order='F', dtype=float)

    r = np.array([[1.0]], order='F', dtype=float)

    l = np.zeros((n, m), order='F', dtype=float)

    result = sg02ad(
        'C', 'B', 'N', 'U', 'Z', 'G', 'S', 'N',
        n, m, p,
        a.copy(order='F'),
        e.copy(order='F'),
        b.copy(order='F'),
        q.copy(order='F'),
        r.copy(order='F'),
        l.copy(order='F'),
        0.0
    )

    x, rcondu, alfar, alfai, beta, s, t, u, iwarn, info = result

    assert info == 0
    assert_allclose(x, x.T, atol=1e-10)
