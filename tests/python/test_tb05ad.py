"""
Tests for TB05AD: Frequency response matrix of state-space system.

Computes G(freq) = C * (freq*I - A)^(-1) * B

Test data from SLICOT HTML documentation example (TB05AD.html).
"""

import numpy as np
import pytest
from ctrlsys import tb05ad


"""Basic functionality tests from HTML doc example."""

def test_html_example_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    System: n=3, m=1, p=2
    A = [[1, 2, 0], [4, -1, 0], [0, 0, 1]]
    B = [[1], [0], [1]]
    C = [[1, 0, -1], [0, 0, 1]]
    freq = 0 + 0.5j
    BALEIG = 'A', INITA = 'G'

    Expected G(freq):
    G[0,0] ~ (0.69, 0.35)
    G[1,0] ~ (-0.80, -0.40)
    """
    n, m, p = 3, 1, 2

    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    freq = 0.0 + 0.5j

    g, rcond, evre, evim, hinvb, info = tb05ad(
        'A', 'G', a, b, c, freq
    )

    assert info == 0

    g_expected = np.array([
        [0.69 + 0.35j],
        [-0.80 - 0.40j]
    ], order='F', dtype=complex)

    np.testing.assert_allclose(g, g_expected, rtol=1e-1, atol=0.01)

    assert rcond > 0.0

    eig_expected = np.array([3.0, -3.0, 1.0])
    eig_computed = evre + 1j * evim
    np.testing.assert_allclose(
        sorted(eig_computed.real),
        sorted(eig_expected),
        rtol=1e-13
    )

def test_html_example_hinvb():
    """
    Validate H^(-1)*B output from HTML doc example.

    Expected HINVB:
    HINVB[0,0] ~ (-0.11, -0.05)
    HINVB[1,0] ~ (-0.43, 0.00)
    HINVB[2,0] ~ (-0.80, -0.40)
    """
    n, m, p = 3, 1, 2

    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    freq = 0.0 + 0.5j

    g, rcond, evre, evim, hinvb, info = tb05ad(
        'A', 'G', a, b, c, freq
    )

    assert info == 0

    hinvb_expected = np.array([
        [-0.11 - 0.05j],
        [-0.43 + 0.00j],
        [-0.80 - 0.40j]
    ], order='F', dtype=complex)

    np.testing.assert_allclose(hinvb, hinvb_expected, rtol=1e-1, atol=0.01)


"""Test mathematical properties of frequency response."""

def test_frequency_response_definition():
    """
    Validate G = C*(freq*I - A)^(-1)*B holds exactly.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 2, 3

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(p, n).astype(float, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    freq = 1.0 + 2.0j

    g, rcond, evre, evim, hinvb, info = tb05ad(
        'N', 'G', a, b, c, freq
    )

    assert info == 0

    freq_i_minus_a = freq * np.eye(n) - a_orig
    hinvb_expected = np.linalg.solve(freq_i_minus_a, b_orig)
    g_expected = c_orig @ hinvb_expected

    np.testing.assert_allclose(g, g_expected, rtol=1e-13, atol=1e-14)

def test_hessenberg_transformation_preserves_response():
    """
    Validate frequency response is preserved under Hessenberg reduction.

    The transformation A -> Q^T A Q should not change G(freq).
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 5, 2, 3

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(p, n).astype(float, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    freq = 0.5 + 1.5j

    g_general, _, _, _, _, info = tb05ad(
        'N', 'G', a, b, c, freq
    )
    assert info == 0

    a_hess = a.copy(order='F')
    b_hess = b.copy(order='F')
    c_hess = c.copy(order='F')

    g_hessenberg, _, _, _, _, info = tb05ad(
        'N', 'H', a_hess, b_hess, c_hess, freq
    )
    assert info == 0

    np.testing.assert_allclose(g_general, g_hessenberg, rtol=1e-13, atol=1e-14)

def test_eigenvalues_preserved_under_balancing():
    """
    Validate eigenvalues are preserved under balancing transformation.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 4, 1, 2

    scale = np.array([1e-3, 1e0, 1e3, 1e0])
    a_raw = np.random.randn(n, n)
    a_orig = (np.diag(scale) @ a_raw @ np.diag(1.0/scale)).astype(float, order='F')
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(p, n).astype(float, order='F')

    eig_before = np.linalg.eigvals(a_orig)

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    freq = 1.0 + 1.0j
    _, _, evre, evim, _, info = tb05ad(
        'B', 'G', a, b, c, freq
    )
    assert info == 0 or info == 1

    eig_after = evre + 1j * evim

    np.testing.assert_allclose(
        sorted(eig_before, key=lambda x: (x.real, x.imag)),
        sorted(eig_after, key=lambda x: (x.real, x.imag)),
        rtol=1e-12
    )


"""Edge cases and boundary conditions."""

def test_zero_dimension():
    """Test n=0 edge case."""
    n, m, p = 0, 1, 2

    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, 1), order='F', dtype=float)
    c = np.zeros((2, 0), order='F', dtype=float)

    freq = 1.0 + 1.0j

    g, rcond, evre, evim, hinvb, info = tb05ad(
        'N', 'G', a, b, c, freq
    )

    assert info == 0
    assert g.shape == (p, m)
    np.testing.assert_array_equal(g, 0.0 + 0.0j)

def test_siso_system():
    """Test single-input single-output system."""
    np.random.seed(789)
    n, m, p = 3, 1, 1

    a_orig = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -2.0, -3.0]
    ], order='F', dtype=float)

    b_orig = np.array([[0.0], [0.0], [1.0]], order='F', dtype=float)
    c_orig = np.array([[1.0, 0.0, 0.0]], order='F', dtype=float)

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    freq = 1.0j

    g, rcond, evre, evim, hinvb, info = tb05ad(
        'C', 'G', a, b, c, freq
    )

    assert info == 0
    assert g.shape == (1, 1)

    freq_i_minus_a = freq * np.eye(n) - a_orig
    g_expected = c_orig @ np.linalg.solve(freq_i_minus_a, b_orig)
    np.testing.assert_allclose(g, g_expected, rtol=1e-13)


"""Test error conditions."""

def test_singular_matrix_detection():
    """
    Test detection when freq equals an eigenvalue.

    When freq is exactly an eigenvalue, (freq*I - A) is singular.
    """
    n, m, p = 2, 1, 1

    a = np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)

    freq = 1.0 + 0.0j

    g, rcond, evre, evim, hinvb, info = tb05ad(
        'C', 'G', a, b, c, freq
    )

    assert info == 2


"""Test different BALEIG and INITA modes."""

def test_mode_n_no_balance():
    """Test BALEIG='N' mode (no balancing, no eigenvalues)."""
    np.random.seed(111)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    freq = 1.0 + 0.5j

    g, rcond, evre, evim, hinvb, info = tb05ad(
        'N', 'G', a, b, c, freq
    )

    assert info == 0
    assert g.shape == (p, m)

def test_mode_c_condition_only():
    """Test BALEIG='C' mode (condition estimate only)."""
    np.random.seed(222)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    freq = 1.0 + 0.5j

    g, rcond, evre, evim, hinvb, info = tb05ad(
        'C', 'G', a, b, c, freq
    )

    assert info == 0
    assert rcond > 0.0
    assert rcond <= 1.0

def test_mode_b_balance_eigenvalues():
    """Test BALEIG='B' mode (balance + eigenvalues)."""
    np.random.seed(333)
    n, m, p = 4, 1, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    eig_expected = np.linalg.eigvals(a)

    freq = 2.0 + 1.0j

    g, rcond, evre, evim, hinvb, info = tb05ad(
        'B', 'G', a, b, c, freq
    )

    assert info == 0 or info == 1
    eig_computed = evre + 1j * evim
    np.testing.assert_allclose(
        sorted(eig_expected, key=lambda x: (x.real, x.imag)),
        sorted(eig_computed, key=lambda x: (x.real, x.imag)),
        rtol=1e-12
    )
