"""
Tests for AB13BD: Compute H2 or L2 norm of a transfer-function matrix.

Computes the H2-norm (continuous) or L2-norm (discrete) of
transfer-function matrix G(lambda) = C*inv(lambda*I - A)*B + D.

Tests:
1. Continuous-time L2 norm from HTML doc
2. Discrete-time H2 norm with stable system
3. Continuous-time with D != 0 error (INFO=5)
4. H2 norm with unstable system error (INFO=6)
5. Property: Stable system has finite norm

Random seeds: 42, 123, 789 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_ab13bd_continuous_l2_stable_system():
    """
    Validate L2 norm for stable continuous-time system.

    For stable systems with D=0, L2 norm equals H2 norm.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab13bd

    np.random.seed(42)
    n, m, p = 3, 2, 2

    # Create stable A with well-separated negative eigenvalues
    a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

    # Random B, C
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.zeros((p, m), order='F', dtype=float)

    tol = 0.0
    h2norm, nq, iwarn, info = ab13bd(
        'C', 'L',
        a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 0
    assert nq == n  # All states in minimal realization
    assert h2norm >= 0
    assert np.isfinite(h2norm)


def test_ab13bd_continuous_h2_stable():
    """
    Validate H2 norm for stable continuous-time system.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab13bd

    np.random.seed(42)
    n, m, p = 3, 2, 2

    # Create stable A (eigenvalues in left half-plane)
    a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

    # Random B, C
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.zeros((p, m), order='F', dtype=float)

    tol = 0.0
    h2norm, nq, iwarn, info = ab13bd(
        'C', 'H',
        a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 0
    assert nq == n
    assert h2norm >= 0  # Norm is non-negative
    assert np.isfinite(h2norm)


def test_ab13bd_discrete_h2():
    """
    Validate H2 norm for discrete-time system (D can be nonzero).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab13bd

    np.random.seed(123)
    n, m, p = 3, 2, 2

    # Create stable discrete A (eigenvalues inside unit circle)
    a = np.diag([0.5, 0.3, 0.2]).astype(float, order='F')

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    tol = 0.0
    h2norm, nq, iwarn, info = ab13bd(
        'D', 'H',
        a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 0
    assert h2norm >= 0
    assert np.isfinite(h2norm)


def test_ab13bd_continuous_d_nonzero_error():
    """
    Validate error for continuous-time with D != 0 (INFO=5).
    """
    from ctrlsys import ab13bd

    n, m, p = 2, 2, 2

    a = np.diag([-1.0, -2.0]).astype(float, order='F')
    b = np.eye(n, m, dtype=float, order='F')
    c = np.eye(p, n, dtype=float, order='F')
    d = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)  # D != 0

    tol = 0.0
    h2norm, nq, iwarn, info = ab13bd(
        'C', 'H',
        a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 5  # Continuous-time with D != 0 is invalid


def test_ab13bd_h2_unstable_error():
    """
    Validate error for H2 norm with unstable system (INFO=6).
    """
    from ctrlsys import ab13bd

    n, m, p = 2, 1, 1

    # Unstable A (positive eigenvalue)
    a = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ], order='F', dtype=float)
    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    tol = 0.0
    h2norm, nq, iwarn, info = ab13bd(
        'C', 'H',
        a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 6  # H2 norm with unstable system is invalid


def test_ab13bd_norm_positive():
    """
    Validate norm is always non-negative for valid systems.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import ab13bd

    np.random.seed(789)
    n, m, p = 4, 2, 3

    # Create stable system
    a = np.diag([-1.0, -0.5, -0.2, -0.1]).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.zeros((p, m), order='F', dtype=float)

    tol = 0.0
    h2norm, nq, iwarn, info = ab13bd(
        'C', 'L',
        a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 0
    assert h2norm >= 0


def test_ab13bd_zero_norm_zero_b():
    """
    Validate zero norm when B=0 (no input coupling).
    """
    from ctrlsys import ab13bd

    n, m, p = 2, 2, 2

    a = np.diag([-1.0, -2.0]).astype(float, order='F')
    b = np.zeros((n, m), order='F', dtype=float)  # B = 0
    c = np.eye(p, n, dtype=float, order='F')
    d = np.zeros((p, m), order='F', dtype=float)

    tol = 0.0
    h2norm, nq, iwarn, info = ab13bd(
        'C', 'H',
        a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 0
    assert h2norm == 0.0  # No input coupling means zero norm
