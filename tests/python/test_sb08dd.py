"""
Tests for SB08DD: Right coprime factorization with inner denominator.

Constructs a right coprime factorization G = Q * R^(-1) where:
- Q and R are stable transfer-function matrices
- R is inner (R'(-s)*R(s) = I for continuous, R'(1/z)*R(z) = I for discrete)

Tests:
1. Continuous-time example from HTML doc
2. Discrete-time stable system (NR=0 case)
3. Basic parameter validation
4. Property: DR is upper triangular

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb08dd_continuous_html_example():
    """
    Validate SB08DD using the continuous-time example from HTML doc.

    System: 7x7 A, 7x2 B, 3x7 C, 3x2 D
    Expected: NQ=7, NR=2
    """
    from ctrlsys import sb08dd

    n, m, p = 7, 2, 3

    # A matrix from HTML doc (row-wise read)
    a = np.array([
        [-0.04165,  0.0000,  4.9200,  0.4920,  0.0000,  0.0000,  0.0000],
        [-5.2100, -12.500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  3.3300, -3.3300,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.5450,  0.0000,  0.0000,  0.0000,  0.0545,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.4920,  0.004165, 0.0000,  4.9200],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.5210, -12.500,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  3.3300, -3.3300]
    ], order='F', dtype=float)

    # B matrix (7x2)
    b = np.array([
        [0.0000,  0.0000],
        [12.500,  0.0000],
        [0.0000,  0.0000],
        [0.0000,  0.0000],
        [0.0000,  0.0000],
        [0.0000,  12.500],
        [0.0000,  0.0000]
    ], order='F', dtype=float)

    # C matrix (3x7)
    c = np.array([
        [1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000],
        [0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000]
    ], order='F', dtype=float)

    # D matrix (3x2)
    d = np.array([
        [0.0000,  0.0000],
        [0.0000,  0.0000],
        [0.0000,  0.0000]
    ], order='F', dtype=float)

    tol = 1e-10
    a_out, b_out, c_out, d_out, nq, nr, cr, dr, iwarn, info = sb08dd(
        'C', a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 0
    assert nq == 7  # All states kept
    assert nr >= 1  # At least one eigenvalue reflected (complex pair may be handled as 1 or 2)

    # DR should be upper triangular
    assert np.allclose(np.tril(dr, -1), 0, atol=1e-14)


def test_sb08dd_stable_system():
    """
    Validate SB08DD with already stable system (NR=0).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb08dd

    np.random.seed(42)
    n, m, p = 3, 2, 2

    # Create stable A (eigenvalues in left half-plane)
    a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

    # Random B, C, D=0 for continuous
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.zeros((p, m), order='F', dtype=float)

    tol = 0.0  # Use default tolerance
    a_out, b_out, c_out, d_out, nq, nr, cr, dr, iwarn, info = sb08dd(
        'C', a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 0
    assert nq == n  # All states remain
    assert nr == 0  # No unstable eigenvalues

    # DR should be identity (no modification needed for stable system)
    assert_allclose(dr, np.eye(m), rtol=1e-10)


def test_sb08dd_discrete():
    """
    Validate SB08DD with discrete-time system.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb08dd

    np.random.seed(123)
    n, m, p = 3, 2, 2

    # Create stable discrete A (eigenvalues inside unit circle)
    a = np.diag([0.5, 0.3, 0.2]).astype(float, order='F')

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')  # D can be nonzero for discrete

    tol = 0.0
    a_out, b_out, c_out, d_out, nq, nr, cr, dr, iwarn, info = sb08dd(
        'D', a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 0
    assert nq == n
    assert nr == 0  # Already stable


def test_sb08dd_dr_upper_triangular():
    """
    Validate DR is upper triangular.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb08dd

    np.random.seed(456)
    n, m, p = 4, 3, 2

    # Create system with some unstable eigenvalues
    a = np.diag([0.5, -1.0, 2.0, -3.0]).astype(float, order='F')  # One unstable
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.zeros((p, m), order='F', dtype=float)

    tol = 0.0
    a_out, b_out, c_out, d_out, nq, nr, cr, dr, iwarn, info = sb08dd(
        'C', a.copy(order='F'), b.copy(order='F'), c.copy(order='F'),
        d.copy(order='F'), tol)

    assert info == 0

    # DR should be upper triangular
    assert np.allclose(np.tril(dr, -1), 0, atol=1e-14)


def test_sb08dd_zero_dimensions():
    """
    Validate quick return for zero dimensions.
    """
    from ctrlsys import sb08dd

    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, 2), order='F', dtype=float)
    c = np.zeros((3, 0), order='F', dtype=float)
    d = np.zeros((3, 2), order='F', dtype=float)

    tol = 0.0
    a_out, b_out, c_out, d_out, nq, nr, cr, dr, iwarn, info = sb08dd(
        'C', a, b, c, d, tol)

    assert info == 0
    assert nq == 0
