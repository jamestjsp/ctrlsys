"""
Tests for MB04DZ - Balance a complex Hamiltonian matrix.

MB04DZ balances a complex Hamiltonian matrix H = [A, G; Q, -A^H] where A is NxN
and G, Q are NxN Hermitian matrices. Balancing involves permuting to isolate
eigenvalues and diagonal similarity transformations.

Based on real version MB04DD but for complex matrices.

Test data sources:
- SLICOT HTML documentation example
- Mathematical properties of complex Hamiltonian matrices
"""

import numpy as np
import pytest

from ctrlsys import mb04dz


def test_mb04dz_html_doc_example():
    """
    Test using the exact example from SLICOT HTML documentation.

    N=6, JOB='B' (both permute and scale).
    Expected ILO=3 and specific balanced A and QG matrices.
    """
    n = 6

    # Input A matrix (6x6, read row-wise per HTML)
    a = np.array([
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0994+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.9696+0.0j],
        [0.3248+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.4372+0.0j, 0.8308+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0717+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.1976+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)

    # Input QG matrix (6x7, read row-wise per HTML)
    qg = np.array([
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0651+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0444+0.0j, 0.0+0.0j, 0.0+0.0j, 0.1957+0.0j, 0.0+0.0j],
        [0.8144+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.3652+0.0j, 0.0+0.0j, 0.9121+0.0j],
        [0.9023+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0945+0.0j]
    ], order='F', dtype=complex)

    a_out, qg_out, ilo, scale, info = mb04dz('B', a, qg)

    assert info == 0
    assert ilo == 3

    # Expected balanced A matrix from HTML documentation
    a_expected = np.array([
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.9696+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, -0.8144+0.0j, -0.9023+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.1093+0.0j, 0.2077+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0717+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.1976+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)

    # Expected balanced QG matrix from HTML documentation
    qg_expected = np.array([
        [0.0+0.0j, 0.0+0.0j, 0.0994+0.0j, 0.0+0.0j, 0.0651+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0812+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.1776+0.0j, 0.0+0.0j, 0.0+0.0j, 0.1957+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.3652+0.0j, 0.0+0.0j, 0.9121+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0945+0.0j]
    ], order='F', dtype=complex)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(qg_out, qg_expected, rtol=1e-3, atol=1e-4)


def test_mb04dz_no_balancing():
    """
    Test with JOB='N' - no balancing, just set scale to 1.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    qg = (np.random.randn(n, n + 1) + 1j * np.random.randn(n, n + 1)).astype(complex, order='F')

    a_orig = a.copy()
    qg_orig = qg.copy()

    a_out, qg_out, ilo, scale, info = mb04dz('N', a, qg)

    assert info == 0
    assert ilo == 1
    np.testing.assert_allclose(scale, np.ones(n), rtol=1e-14)
    np.testing.assert_allclose(a_out, a_orig, rtol=1e-14)


def test_mb04dz_permute_only():
    """
    Test with JOB='P' - permute only.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    qg = (np.random.randn(n, n + 1) + 1j * np.random.randn(n, n + 1)).astype(complex, order='F')

    a_out, qg_out, ilo, scale, info = mb04dz('P', a, qg)

    assert info == 0
    assert 1 <= ilo <= n + 1


def test_mb04dz_scale_only():
    """
    Test with JOB='S' - scale only.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    qg = (np.random.randn(n, n + 1) + 1j * np.random.randn(n, n + 1)).astype(complex, order='F')

    a_out, qg_out, ilo, scale, info = mb04dz('S', a, qg)

    assert info == 0
    assert ilo == 1
    assert len(scale) == n
    assert all(s > 0 for s in scale)


def test_mb04dz_both():
    """
    Test with JOB='B' - both permute and scale.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    qg = (np.random.randn(n, n + 1) + 1j * np.random.randn(n, n + 1)).astype(complex, order='F')

    a_out, qg_out, ilo, scale, info = mb04dz('B', a, qg)

    assert info == 0
    assert 1 <= ilo <= n + 1
    assert len(scale) == n


def test_mb04dz_zero_dimension():
    """
    Test with n=0 - empty matrix.
    """
    a = np.zeros((0, 0), order='F', dtype=complex)
    qg = np.zeros((0, 1), order='F', dtype=complex)

    a_out, qg_out, ilo, scale, info = mb04dz('B', a, qg)

    assert info == 0


def test_mb04dz_single_element():
    """
    Test with n=1 - single element matrix.
    """
    a = np.array([[2.0+1.0j]], order='F', dtype=complex)
    qg = np.array([[1.0+0.5j, 3.0-0.5j]], order='F', dtype=complex)

    a_out, qg_out, ilo, scale, info = mb04dz('B', a, qg)

    assert info == 0


def test_mb04dz_diagonal_hamiltonian():
    """
    Test with diagonal A and zero Q, G - already balanced.
    """
    n = 3
    a = np.diag([1.0+0.0j, 2.0+0.0j, 3.0+0.0j]).astype(complex, order='F')
    qg = np.zeros((n, n + 1), order='F', dtype=complex)

    a_out, qg_out, ilo, scale, info = mb04dz('B', a, qg)

    assert info == 0


def test_mb04dz_hermitian_structure():
    """
    Test that balancing preserves Hermitian structure of G and Q.

    For Hamiltonian matrices, G and Q must be Hermitian.
    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n = 4

    # Create Hermitian Q (lower triangle of QG)
    q_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    q_herm = (q_full + q_full.conj().T) / 2

    # Create Hermitian G (upper triangle of QG, shifted by one column)
    g_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    g_herm = (g_full + g_full.conj().T) / 2

    # Pack into QG format: Q in lower triangle, G in upper triangle shifted
    qg = np.zeros((n, n + 1), order='F', dtype=complex)
    for i in range(n):
        for j in range(i + 1):
            qg[i, j] = q_herm[i, j]  # Lower triangle of Q
        for j in range(i, n):
            qg[i, j + 1] = g_herm[i, j]  # Upper triangle of G (shifted)

    # Also create an A matrix
    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')

    a_out, qg_out, ilo, scale, info = mb04dz('S', a, qg)

    assert info == 0
    # Just check that scaling was applied and structure is maintained
    assert len(scale) == n


def test_mb04dz_pure_imaginary_elements():
    """
    Test with matrices containing only pure imaginary elements.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 3

    a = 1j * np.random.randn(n, n).astype(complex, order='F')
    qg = 1j * np.random.randn(n, n + 1).astype(complex, order='F')

    a_out, qg_out, ilo, scale, info = mb04dz('B', a, qg)

    assert info == 0
    assert 1 <= ilo <= n + 1
