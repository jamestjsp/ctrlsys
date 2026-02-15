"""
Tests for AB01MD - Controllable realization for single-input systems.

AB01MD finds a controllable realization for the linear time-invariant
single-input system dX/dt = A * X + B * U, reducing to orthogonal canonical
form using orthogonal similarity transformations.
"""

import numpy as np
import pytest
from ctrlsys import ab01md


"""Basic functionality tests from HTML documentation example."""

def test_html_doc_example():
    """
    Test case from SLICOT HTML documentation.

    Input: N=3, TOL=0.0, JOBZ='I'
    A = [[1, 2, 0], [4, -1, 0], [0, 0, 1]]
    B = [1, 0, 1]

    Expected NCONT=3 (fully controllable)
    Expected transformed A (upper Hessenberg):
        [[1.0000,  1.4142,  0.0000],
         [2.8284, -1.0000,  2.8284],
         [0.0000,  1.4142,  1.0000]]
    Expected transformed B:
        [-1.4142, 0.0000, 0.0000]
    Expected Z:
        [[-0.7071,  0.0000, -0.7071],
         [ 0.0000, -1.0000,  0.0000],
         [-0.7071,  0.0000,  0.7071]]
    """
    n = 3

    # Input: A read row-wise from HTML doc
    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float, order='F')

    # Input: B vector
    b = np.array([1.0, 0.0, 1.0], dtype=float, order='F')

    tol = 0.0

    a_out, b_out, ncont, z, tau, info = ab01md('I', a, b, tol)

    assert info == 0
    assert ncont == 3

    # Expected A (upper Hessenberg, NCONT x NCONT block)
    a_expected = np.array([
        [1.0000, 1.4142, 0.0000],
        [2.8284, -1.0000, 2.8284],
        [0.0000, 1.4142, 1.0000]
    ], dtype=float, order='F')

    # Expected B (first NCONT elements)
    b_expected = np.array([-1.4142, 0.0, 0.0], dtype=float)

    # Expected Z (orthogonal transformation matrix)
    z_expected = np.array([
        [-0.7071, 0.0000, -0.7071],
        [0.0000, -1.0000, 0.0000],
        [-0.7071, 0.0000, 0.7071]
    ], dtype=float, order='F')

    # Validate outputs - HTML doc shows 4 decimal places
    np.testing.assert_allclose(a_out[:ncont, :ncont], a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b_out[:ncont], b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z, z_expected, rtol=1e-3, atol=1e-4)


"""Test mathematical properties of orthogonal transformation."""

def test_z_is_orthogonal():
    """
    Validate Z'*Z = I (orthogonality property).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    b[0] = 1.0  # Ensure non-zero for controllability

    _, _, ncont, z, _, info = ab01md('I', a, b, 0.0)

    assert info == 0

    # Z should be orthogonal: Z'*Z = I
    ztz = z.T @ z
    np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14)

def test_similarity_transformation():
    """
    Validate A_out = Z' * A_in * Z (similarity transformation property).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 5

    a_in = np.random.randn(n, n).astype(float, order='F')
    b_in = np.random.randn(n).astype(float, order='F')
    b_in[0] = 2.0  # Ensure non-zero

    a_copy = a_in.copy()

    a_out, _, ncont, z, _, info = ab01md('I', a_in, b_in, 0.0)

    assert info == 0

    # A_out = Z' * A_orig * Z
    a_transformed = z.T @ a_copy @ z
    np.testing.assert_allclose(a_out, a_transformed, rtol=1e-13, atol=1e-14)

def test_b_transformation():
    """
    Validate b_out = Z' * b_in (vector transformation).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    b_in = np.random.randn(n).astype(float, order='F')
    b_in[0] = 1.5

    b_copy = b_in.copy()

    _, b_out, ncont, z, _, info = ab01md('I', a, b_in, 0.0)

    assert info == 0

    # b_out = Z' * b_orig
    b_transformed = z.T @ b_copy
    np.testing.assert_allclose(b_out, b_transformed, rtol=1e-13, atol=1e-14)


"""Test upper Hessenberg structure of output A."""

def test_output_is_upper_hessenberg():
    """
    Validate output A has upper Hessenberg structure (zeros below first subdiagonal).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 6

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    b[0] = 3.0

    a_out, _, ncont, _, _, info = ab01md('I', a, b, 0.0)

    assert info == 0

    # Check upper Hessenberg structure: a[i,j] = 0 for i > j+1
    for i in range(2, ncont):
        for j in range(i - 1):
            assert abs(a_out[i, j]) < 1e-14, \
                f"Element ({i},{j}) should be zero: {a_out[i, j]}"


"""Test controllability detection."""

def test_uncontrollable_system():
    """
    Test system that is not fully controllable.

    A block-diagonal system where B only affects one block.
    """
    n = 4

    # Block diagonal A with two 2x2 blocks
    a = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    # B only affects first block
    b = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    _, _, ncont, _, _, info = ab01md('I', a, b, 0.0)

    assert info == 0
    # System is not fully controllable due to decoupled blocks
    assert ncont < n

def test_zero_b_vector():
    """
    Test with zero B vector - completely uncontrollable.
    """
    n = 3

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=float, order='F')

    b = np.zeros(n, dtype=float)

    _, _, ncont, z, _, info = ab01md('I', a, b, 0.0)

    assert info == 0
    assert ncont == 0
    # For zero B with JOBZ='I', Z should be identity
    np.testing.assert_allclose(z, np.eye(n), rtol=1e-14, atol=1e-14)


"""Test different JOBZ modes."""

def test_jobz_n():
    """
    Test JOBZ='N' mode (no orthogonal transformation stored).
    """
    n = 3

    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float, order='F')

    b = np.array([1.0, 0.0, 1.0], dtype=float)

    a_out, b_out, ncont, z, tau, info = ab01md('N', a, b, 0.0)

    assert info == 0
    assert ncont == 3
    # Z is not computed but returned as dummy

def test_jobz_f():
    """
    Test JOBZ='F' mode (factored form storage).
    """
    n = 3

    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float, order='F')

    b = np.array([1.0, 0.0, 1.0], dtype=float)

    a_out, b_out, ncont, z, tau, info = ab01md('F', a, b, 0.0)

    assert info == 0
    assert ncont == 3


"""Test edge cases and boundary conditions."""

def test_n_equals_1():
    """
    Test with N=1 (scalar system).
    """
    a = np.array([[2.0]], dtype=float, order='F')
    b = np.array([3.0], dtype=float)

    a_out, b_out, ncont, z, tau, info = ab01md('I', a, b, 0.0)

    assert info == 0
    assert ncont == 1
    # For n=1, A unchanged, B unchanged, Z is 1x1 (sign may change)
    np.testing.assert_allclose(abs(z[0, 0]), 1.0, rtol=1e-5)

def test_n_equals_0():
    """
    Test with N=0 (empty system).
    """
    a = np.array([], dtype=float, order='F').reshape(0, 0)
    b = np.array([], dtype=float)

    a_out, b_out, ncont, z, tau, info = ab01md('I', a, b, 0.0)

    assert info == 0
    assert ncont == 0

def test_large_system():
    """
    Test with larger system to verify scalability.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 20

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n).astype(float, order='F')
    b[0] = 5.0

    a_out, _, ncont, z, _, info = ab01md('I', a, b, 0.0)

    assert info == 0
    # Should be fully controllable (with high probability for random system)

    # Z should be orthogonal
    ztz = z.T @ z
    np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-13, atol=1e-13)


"""Test tolerance handling."""

def test_explicit_tolerance():
    """
    Test with explicit positive tolerance.
    """
    n = 3

    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float, order='F')

    b = np.array([1.0, 0.0, 1.0], dtype=float)

    # Use explicit tolerance
    tol = 1e-10

    a_out, b_out, ncont, z, tau, info = ab01md('I', a, b, tol)

    assert info == 0
    assert ncont == 3

def test_negative_tolerance_uses_default():
    """
    Test that negative tolerance uses default (N*EPS*max(norm(A), norm(B))).
    """
    n = 3

    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float, order='F')

    b = np.array([1.0, 0.0, 1.0], dtype=float)

    # Negative tol means use default
    tol = -1.0

    a_out, b_out, ncont, z, tau, info = ab01md('I', a, b, tol)

    assert info == 0
    assert ncont == 3


"""Test error handling."""

def test_invalid_jobz():
    """
    Test invalid JOBZ parameter.
    """
    n = 3
    a = np.eye(n, dtype=float, order='F')
    b = np.ones(n, dtype=float)

    with pytest.raises(ValueError, match="jobz must be"):
        ab01md('X', a, b, 0.0)

def test_mismatched_dimensions():
    """
    Test mismatched A and B dimensions.
    """
    a = np.eye(3, dtype=float, order='F')
    b = np.ones(4, dtype=float)  # Wrong size

    with pytest.raises(ValueError):
        ab01md('I', a, b, 0.0)
