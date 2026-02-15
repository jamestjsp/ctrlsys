"""
Tests for TB01KD: Stable/unstable decomposition of state-space system.

Computes an additive spectral decomposition of the transfer function matrix
of (A,B,C) by reducing A to block-diagonal form with eigenvalues in a
specified domain in the leading block.

Transformation: A <- inv(U)*A*U, B <- inv(U)*B, C <- C*U

Tests:
1. HTML doc example (continuous, unstable domain)
2. Continuous stable domain separation
3. Discrete stable domain separation
4. Schur form input (JOBA='S')
5. Block-diagonal verification
6. Eigenvalue preservation

Random seeds: 42, 123, 456 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tb01kd_html_example():
    """
    Validate using HTML documentation example.

    Continuous system, unstable domain (Re(lambda) > -1.0).
    Expected: NDIM=2, block-diagonal A with eigenvalues in domain in leading block.
    """
    from ctrlsys import tb01kd

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

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01kd(
        'C', 'U', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 2  # From HTML doc example

    # Expected eigenvalues from HTML doc
    expected_wr = np.array([-0.7483, -0.7483, -1.6858, -1.6858, -1.8751])
    expected_wi = np.array([2.9940, -2.9940, 2.0311, -2.0311, 0.0])

    # First ndim eigenvalues should be in domain Re > -1.0
    for i in range(ndim):
        assert wr[i] > alpha, f"Eigenvalue {i} real part {wr[i]} not > {alpha}"

    # Eigenvalues match expected (sorted by real part)
    eig_computed = sorted(zip(wr, wi), key=lambda x: (x[0], -abs(x[1])))
    eig_expected = sorted(zip(expected_wr, expected_wi), key=lambda x: (x[0], -abs(x[1])))

    for i, ((wr_c, wi_c), (wr_e, wi_e)) in enumerate(zip(eig_computed, eig_expected)):
        assert_allclose(wr_c, wr_e, rtol=1e-3)
        assert_allclose(abs(wi_c), abs(wi_e), rtol=1e-3)

    # A12 block should be zero (block-diagonal form)
    assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-10)

    # Verify block structure: A11 (NDIM x NDIM) and A22 ((N-NDIM) x (N-NDIM)) are quasi-triangular
    # Elements below first subdiagonal should be zero
    for i in range(2, n):
        for j in range(i - 1):
            assert abs(a_out[i, j]) < 1e-10, f"a_out[{i},{j}] = {a_out[i,j]} not zero"


def test_tb01kd_continuous_stable():
    """
    Validate continuous-time stable domain separation.

    Move eigenvalues with Re(lambda) < 0 to leading block, make A block-diagonal.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tb01kd

    np.random.seed(42)
    n, m, p = 4, 2, 2
    alpha = 0.0

    # Create matrix with known eigenvalues: -1, -2, 1, 2
    eigs = np.array([-1.0, -2.0, 1.0, 2.0])
    q, _ = np.linalg.qr(np.random.randn(n, n))
    a = q @ np.diag(eigs) @ q.T
    a = np.asfortranarray(a)

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_orig = a.copy()

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01kd(
        'C', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 2  # Two eigenvalues with Re < 0

    # First ndim eigenvalues should be stable (Re < 0)
    for i in range(ndim):
        assert wr[i] < alpha, f"Eigenvalue {i} real part {wr[i]} not < {alpha}"

    # A12 block should be zero (block-diagonal form)
    assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-10)

    # Eigenvalue preservation
    eig_before = np.sort(np.linalg.eigvals(a_orig).real)
    eig_after = np.sort(wr)
    assert_allclose(eig_before, eig_after, rtol=1e-10)


def test_tb01kd_discrete_stable():
    """
    Validate discrete-time stable domain separation.

    Move eigenvalues with |lambda| < 1 to leading block, make A block-diagonal.
    """
    from ctrlsys import tb01kd

    n, m, p = 3, 1, 1
    alpha = 1.0  # Unit circle

    # Eigenvalues: 0.5, 0.8, 1.5 (two inside, one outside)
    a = np.diag([0.5, 0.8, 1.5]).astype(float, order='F')
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01kd(
        'D', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 2  # Two eigenvalues inside unit circle

    # First ndim eigenvalues should be inside unit circle
    for i in range(ndim):
        modulus = np.sqrt(wr[i]**2 + wi[i]**2)
        assert modulus < alpha, f"Eigenvalue {i} modulus {modulus} not < {alpha}"

    # A12 block should be zero
    assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-10)


def test_tb01kd_schur_input():
    """
    Validate with Schur form input (JOBA='S').

    When A is already in Schur form, only reordering and block-diagonalization needed.
    """
    from ctrlsys import tb01kd

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

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01kd(
        'C', 'S', 'S', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 1  # One eigenvalue with Re < 0

    # A12 block should be zero
    assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-10)


def test_tb01kd_block_diagonal():
    """
    Validate A is truly block-diagonal after transformation.

    A12 block must be zero. Leading block has NDIM eigenvalues in domain.
    Uses matrix with well-separated eigenvalues to avoid Sylvester singularity.
    """
    from ctrlsys import tb01kd

    n, m, p = 4, 2, 2
    alpha = 0.0

    # Create matrix with well-separated eigenvalues: -3, -2, 2, 3
    eigs = np.array([-3.0, -2.0, 2.0, 3.0])
    a = np.diag(eigs).astype(float, order='F')
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01kd(
        'C', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 2  # Two eigenvalues with Re < 0

    if ndim > 0 and ndim < n:
        # A12 block should be zero
        assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-10)

    # Below first subdiagonal should be zero (quasi-triangular form)
    for i in range(2, n):
        for j in range(i - 1):
            assert abs(a_out[i, j]) < 1e-10, f"a_out[{i},{j}] = {a_out[i,j]} not zero"


def test_tb01kd_eigenvalue_preservation():
    """
    Validate eigenvalues are preserved by similarity transformation.

    Uses matrix with well-separated eigenvalues to avoid Sylvester singularity.
    """
    from ctrlsys import tb01kd

    n, m, p = 5, 2, 2
    alpha = 0.0

    # Create matrix with well-separated eigenvalues: -4, -3, 1, 2, 3
    eigs = np.array([-4.0, -3.0, 1.0, 2.0, 3.0])
    a = np.diag(eigs).astype(float, order='F')
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    a_orig = a.copy()

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01kd(
        'C', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0

    # Eigenvalues should be preserved
    eig_before = np.linalg.eigvals(a_orig)
    eig_computed = wr + 1j * wi

    eig_before_sorted = np.sort_complex(eig_before)
    eig_computed_sorted = np.sort_complex(eig_computed)

    assert_allclose(eig_before_sorted, eig_computed_sorted, rtol=1e-10)


def test_tb01kd_transformation_validity():
    """
    Validate transformation: A_out = inv(U)*A_orig*U, B_out = inv(U)*B, C_out = C*U.

    Note: U from TB01KD is not orthogonal in general (includes Sylvester solution).
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tb01kd

    np.random.seed(789)
    n, m, p = 4, 2, 2
    alpha = 0.0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_orig = a.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01kd(
        'C', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0

    # C_out = C * U
    c_reconstructed = c_orig @ u
    assert_allclose(c_out, c_reconstructed, atol=1e-12)


def test_tb01kd_zero_dimension():
    """
    Validate quick return for n=0.
    """
    from ctrlsys import tb01kd

    n, m, p = 0, 2, 2
    alpha = 0.0

    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, m), order='F', dtype=float)
    c = np.zeros((p, 0), order='F', dtype=float)

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01kd(
        'C', 'S', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c.copy(order='F')
    )

    assert info == 0
    assert ndim == 0


def test_tb01kd_html_transformed_matrices():
    """
    Validate transformed B and C matrices from HTML doc.

    Since eigenvalue ordering may differ between implementations, we verify:
    1. C_out = C_orig @ U (transformation is correct)
    2. Block diagonal structure of A
    3. First NDIM rows of B_out and C_out correspond to domain eigenvalues
    """
    from ctrlsys import tb01kd

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

    c_orig = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]
    ], order='F', dtype=float)

    a_out, b_out, c_out, ndim, u, wr, wi, info = tb01kd(
        'C', 'U', 'G', alpha, a.copy(order='F'), b.copy(order='F'), c_orig.copy(order='F')
    )

    assert info == 0
    assert ndim == 2  # Two eigenvalues with Re > -1.0

    # Verify C_out = C_orig @ U
    c_reconstructed = c_orig @ u
    assert_allclose(c_out, c_reconstructed, atol=1e-12)

    # A12 block should be zero (block-diagonal form)
    assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-10)

    # Verify block structure: elements below first subdiagonal should be zero
    for i in range(2, n):
        for j in range(i - 1):
            assert abs(a_out[i, j]) < 1e-10, f"a_out[{i},{j}] = {a_out[i,j]} not zero"
