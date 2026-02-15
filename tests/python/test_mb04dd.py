"""
Tests for MB04DD - Balance a real Hamiltonian matrix.

MB04DD balances a real Hamiltonian matrix H = [A, G; Q, -A'] where A is NxN
and G, Q are symmetric NxN matrices. Balancing involves permuting to isolate
eigenvalues and diagonal similarity transformations.

Test data sources:
- Mathematical properties of Hamiltonian matrices
- Random test matrices
"""

import numpy as np
import pytest

from ctrlsys import mb04dd


def test_mb04dd_no_balancing():
    """
    Test with JOB='N' - no balancing, just set scale to 1.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    # QG has dimensions (n, n+1) - lower tri of Q and upper tri of G
    qg = np.random.randn(n, n + 1).astype(float, order='F')

    a_orig = a.copy()
    qg_orig = qg.copy()

    a_out, qg_out, ilo, scale, info = mb04dd('N', a, qg)

    assert info == 0
    assert ilo == 1
    np.testing.assert_allclose(scale, np.ones(n), rtol=1e-14)
    np.testing.assert_allclose(a_out, a_orig, rtol=1e-14)


def test_mb04dd_permute_only():
    """
    Test with JOB='P' - permute only.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.random.randn(n, n + 1).astype(float, order='F')

    a_out, qg_out, ilo, scale, info = mb04dd('P', a, qg)

    assert info == 0
    assert 1 <= ilo <= n + 1


def test_mb04dd_scale_only():
    """
    Test with JOB='S' - scale only.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.random.randn(n, n + 1).astype(float, order='F')

    a_out, qg_out, ilo, scale, info = mb04dd('S', a, qg)

    assert info == 0
    assert ilo == 1
    assert len(scale) == n
    assert all(s > 0 for s in scale)


def test_mb04dd_both():
    """
    Test with JOB='B' - both permute and scale.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.random.randn(n, n + 1).astype(float, order='F')

    a_out, qg_out, ilo, scale, info = mb04dd('B', a, qg)

    assert info == 0
    assert 1 <= ilo <= n + 1
    assert len(scale) == n


def test_mb04dd_zero_dimension():
    """
    Test with n=0 - empty matrix.
    """
    a = np.zeros((0, 0), order='F', dtype=float)
    qg = np.zeros((0, 1), order='F', dtype=float)

    a_out, qg_out, ilo, scale, info = mb04dd('B', a, qg)

    assert info == 0


def test_mb04dd_single_element():
    """
    Test with n=1 - single element matrix.
    """
    a = np.array([[2.0]], order='F', dtype=float)
    qg = np.array([[1.0, 3.0]], order='F', dtype=float)

    a_out, qg_out, ilo, scale, info = mb04dd('B', a, qg)

    assert info == 0


def test_mb04dd_diagonal_hamiltonian():
    """
    Test with diagonal A and zero Q, G - already balanced.
    """
    n = 3
    a = np.diag([1.0, 2.0, 3.0]).astype(float, order='F')
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    a_out, qg_out, ilo, scale, info = mb04dd('B', a, qg)

    assert info == 0
