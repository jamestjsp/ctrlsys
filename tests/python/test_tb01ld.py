"""
Tests for TB01LD: Reduce state matrix to ordered Schur form.

Reduces system (A,B,C) to ordered Schur form with eigenvalues in a
specified domain of interest in the leading block. Applies orthogonal
transformation to B and C.

Tests:
1. HTML doc example (continuous, unstable domain)
2. Continuous stable domain separation
3. Discrete stable domain separation
4. Schur form input (JOBA='S')
5. Eigenvalue preservation property
6. Orthogonality of transformation

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tb01ld_html_example():
    """
    Validate using HTML documentation example.

    Continuous system, unstable domain (Re(lambda) > -1.0).
    Expected: NDIM=2, eigenvalues with Re > -1.0 in leading block.
    """
    from ctrlsys import tb01ld

    n, m, p = 5, 2, 3
    alpha = -1.0

    a = np.array([
        [-0.04165,  4.9200, -4.9200,       0,       0],
        [-1.387944, -3.3300,       0,       0,       0],
        [   0.5450,       0,       0, -0.5450,       0],
        [        0,       0,  4.9200, -0.04165, 4.9200],
        [        0,       0,       0, -1.387944, -3.3300]
    ], order='F', dtype=float)

    b = np.array([
        [0.0,    0.0],
        [3.3300, 0.0],
        [0.0,    0.0],
        [0.0,    0.0],
        [0.0, 3.3300]
    ], order='F', dtype=float)

    c = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]
    ], order='F', dtype=float)

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01ld(
        'C', 'U', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 2  # From HTML doc example

    # Expected eigenvalues from HTML doc (sorted)
    expected_wr = np.array([-0.7483, -0.7483, -1.6858, -1.6858, -1.8751])
    expected_wi = np.array([2.9940, -2.9940, 2.0311, -2.0311, 0.0])

    # First ndim eigenvalues should be in domain Re > -1.0
    for i in range(ndim):
        assert wr[i] > alpha, f"Eigenvalue {i} real part {wr[i]} not > {alpha}"

    # Eigenvalues match expected (sorted by real part, then imag)
    eig_computed = sorted(zip(wr, wi), key=lambda x: (x[0], -abs(x[1])))
    eig_expected = sorted(zip(expected_wr, expected_wi), key=lambda x: (x[0], -abs(x[1])))

    for i, ((wr_c, wi_c), (wr_e, wi_e)) in enumerate(zip(eig_computed, eig_expected)):
        assert_allclose(wr_c, wr_e, rtol=1e-3)
        assert_allclose(abs(wi_c), abs(wi_e), rtol=1e-3)


def test_tb01ld_continuous_stable():
    """
    Validate continuous-time stable domain separation.

    Move eigenvalues with Re(lambda) < 0 to leading block.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tb01ld

    np.random.seed(42)
    n, m, p = 4, 2, 2
    alpha = 0.0

    # Create matrix with known eigenvalues: -1, -2, 1, 2
    eigs = np.array([-1, -2, 1, 2])
    q, _ = np.linalg.qr(np.random.randn(n, n))
    a = q @ np.diag(eigs) @ q.T
    a = np.asfortranarray(a)

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_orig = a.copy()

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01ld(
        'C', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 2  # Two eigenvalues with Re < 0

    # First ndim eigenvalues should be stable (Re < 0)
    for i in range(ndim):
        assert wr[i] < alpha, f"Eigenvalue {i} real part {wr[i]} not < {alpha}"

    # Eigenvalue preservation
    eig_before = np.sort(np.linalg.eigvals(a_orig).real)
    eig_after = np.sort(wr)
    assert_allclose(eig_before, eig_after, rtol=1e-10)


def test_tb01ld_discrete_stable():
    """
    Validate discrete-time stable domain separation.

    Move eigenvalues with |lambda| < 1 to leading block.
    """
    from ctrlsys import tb01ld

    n, m, p = 3, 1, 1
    alpha = 1.0  # Unit circle

    # Eigenvalues: 0.5, 0.8, 1.5 (two inside, one outside)
    a = np.diag([0.5, 0.8, 1.5]).astype(float, order='F')
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01ld(
        'D', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 2  # Two eigenvalues inside unit circle

    # First ndim eigenvalues should be inside unit circle
    for i in range(ndim):
        modulus = np.sqrt(wr[i]**2 + wi[i]**2)
        assert modulus < alpha, f"Eigenvalue {i} modulus {modulus} not < {alpha}"


def test_tb01ld_schur_input():
    """
    Validate with Schur form input (JOBA='S').

    When A is already in Schur form, only reordering is needed.
    """
    from ctrlsys import tb01ld

    n, m, p = 3, 1, 1
    alpha = 0.0

    # Upper triangular (Schur form) with eigenvalues: -1, 2, 3
    a = np.array([
        [-1.0, 0.5, 0.3],
        [ 0.0, 2.0, 0.4],
        [ 0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01ld(
        'C', 'S', 'S', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 1  # One eigenvalue with Re < 0


def test_tb01ld_orthogonality():
    """
    Validate orthogonality of transformation matrix U.

    U should satisfy U'*U = I.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import tb01ld

    np.random.seed(123)
    n, m, p = 4, 2, 2
    alpha = 0.0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01ld(
        'C', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0

    # U should be orthogonal
    assert_allclose(u.T @ u, np.eye(n), atol=1e-14)
    assert_allclose(u @ u.T, np.eye(n), atol=1e-14)


def test_tb01ld_similarity():
    """
    Validate similarity transformation A_out = U'*A*U.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tb01ld

    np.random.seed(456)
    n, m, p = 4, 2, 2
    alpha = 0.0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_orig = a.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01ld(
        'C', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0

    # A_out = U'*A*U
    a_reconstructed = u.T @ a_orig @ u
    # Only compare upper quasi-triangular part (Schur form zeros below subdiagonal)
    for i in range(n):
        for j in range(i, n):
            assert_allclose(a_out[i, j], a_reconstructed[i, j], atol=1e-12)

    # B_out = U'*B
    b_reconstructed = u.T @ b_orig
    assert_allclose(b_out, b_reconstructed, atol=1e-12)

    # C_out = C*U
    c_reconstructed = c_orig @ u
    assert_allclose(c_out, c_reconstructed, atol=1e-12)


def test_tb01ld_eigenvalue_preservation():
    """
    Validate eigenvalues are preserved by similarity transformation.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tb01ld

    np.random.seed(789)
    n, m, p = 5, 2, 2
    alpha = 0.0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_orig = a.copy()

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01ld(
        'C', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0

    # Eigenvalues should be preserved
    eig_before = np.linalg.eigvals(a_orig)
    eig_computed = wr + 1j * wi

    eig_before_sorted = np.sort_complex(eig_before)
    eig_computed_sorted = np.sort_complex(eig_computed)

    assert_allclose(eig_before_sorted, eig_computed_sorted, rtol=1e-10)
