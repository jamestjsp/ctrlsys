"""Tests for tb01md - Controller Hessenberg form reduction."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tb01md_upper_hessenberg_html_example():
    """
    Test upper controller Hessenberg form using HTML doc example.

    N=6, M=3, JOBU='I', UPLO='U'
    Validates mathematical properties rather than exact values since
    Householder reflector signs can differ between implementations.
    """
    from ctrlsys import tb01md

    # Input A: 6x6 matrix from SLICOT HTML doc
    a = np.array([
        [35.0,  1.0,  6.0, 26.0, 19.0, 24.0],
        [ 3.0, 32.0,  7.0, 21.0, 23.0, 25.0],
        [31.0,  9.0,  2.0, 22.0, 27.0, 20.0],
        [ 8.0, 28.0, 33.0, 17.0, 10.0, 15.0],
        [30.0,  5.0, 34.0, 12.0, 14.0, 16.0],
        [ 4.0, 36.0, 29.0, 13.0, 18.0, 11.0]
    ], order='F', dtype=float)

    # Input B: 6x3 matrix from SLICOT HTML doc
    b = np.array([
        [  1.0,  5.0, 11.0],
        [ -1.0,  4.0, 11.0],
        [ -5.0,  1.0,  9.0],
        [-11.0, -4.0,  5.0],
        [-19.0,-11.0, -1.0],
        [-29.0,-20.0, -9.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    b_orig = b.copy()
    n, m = 6, 3

    # Call tb01md with JOBU='I' to get transformation matrix
    a_out, b_out, u_out, info = tb01md('I', 'U', a, b)

    assert info == 0

    # Verify U is orthogonal
    assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-12, atol=1e-12)

    # Verify similarity transformation: A_out = U' @ A_orig @ U
    assert_allclose(a_out, u_out.T @ a_orig @ u_out, rtol=1e-12, atol=1e-12)

    # Verify: B_out = U' @ B_orig
    assert_allclose(b_out, u_out.T @ b_orig, rtol=1e-12, atol=1e-12)

    # Verify upper triangular structure of B (zeros below diagonal)
    for i in range(m, n):
        for j in range(m):
            assert_allclose(b_out[i, j], 0.0, atol=1e-14)
    for i in range(1, m):
        for j in range(i):
            assert_allclose(b_out[i, j], 0.0, atol=1e-14)

    # Verify upper Hessenberg structure of A (zeros in lower part for rows > m)
    for i in range(m + 1, n):
        for j in range(i - m - 1):
            assert_allclose(a_out[i, j], 0.0, atol=1e-14)

    # Verify eigenvalues preserved
    eig_before = np.linalg.eigvals(a_orig)
    eig_after = np.linalg.eigvals(a_out)
    assert_allclose(sorted(eig_before.real), sorted(eig_after.real), rtol=1e-12, atol=1e-12)


def test_tb01md_lower_hessenberg():
    """
    Test lower controller Hessenberg form (UPLO='L').

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tb01md

    np.random.seed(42)
    n, m = 4, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    a_orig = a.copy()
    b_orig = b.copy()

    # Call with JOBU='I' to get transformation matrix
    a_out, b_out, u_out, info = tb01md('I', 'L', a, b)

    assert info == 0

    # Verify U is orthogonal: U^T @ U = I
    assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-12, atol=1e-12)

    # Verify similarity: A_out = U^T @ A_orig @ U
    assert_allclose(a_out, u_out.T @ a_orig @ u_out, rtol=1e-12, atol=1e-12)

    # Verify: B_out = U^T @ B_orig
    assert_allclose(b_out, u_out.T @ b_orig, rtol=1e-12, atol=1e-12)

    # For lower Hessenberg with M=2, N=4:
    # B should have lower trapezoidal structure (zeros in upper part)
    # B(i,j) = 0 for i < n-m+j = 4-2+j = 2+j
    # j=0: i<2 => B[0,0]=B[1,0]=0
    # j=1: i<3 => B[0,1]=B[1,1]=B[2,1]=0
    assert_allclose(b_out[0, 0], 0.0, atol=1e-14)
    assert_allclose(b_out[1, 0], 0.0, atol=1e-14)
    assert_allclose(b_out[0, 1], 0.0, atol=1e-14)
    assert_allclose(b_out[1, 1], 0.0, atol=1e-14)
    assert_allclose(b_out[2, 1], 0.0, atol=1e-14)


def test_tb01md_accumulate_u():
    """
    Test U matrix accumulation mode (JOBU='U').

    When JOBU='U', the given U is updated with the transformations.
    If Q is the internal transformation, then U_out = U_init @ Q.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import tb01md

    np.random.seed(123)
    n, m = 5, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    a_orig = a.copy()
    b_orig = b.copy()

    # Create initial orthogonal U from QR decomposition
    q, _ = np.linalg.qr(np.random.randn(n, n))
    u_init = np.asfortranarray(q.astype(float))
    u_init_copy = u_init.copy()

    # Call with JOBU='U' to update the given U
    a_out, b_out, u_out, info = tb01md('U', 'U', a, b, u=u_init)

    assert info == 0

    # Verify U_out is still orthogonal
    assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-12, atol=1e-12)

    # When JOBU='U', the transformation is U_out = U_init @ Q where Q is the
    # Hessenberg reduction transformation. So U_out^T @ U_init is orthogonal.
    # Also verify: A_out = Q^T @ A_orig @ Q and B_out = Q^T @ B_orig
    # where Q = U_init^T @ U_out

    # Extract Q: Q = U_init^T @ U_out (since U_out = U_init @ Q)
    q_transform = u_init_copy.T @ u_out

    # Verify Q is orthogonal
    assert_allclose(q_transform.T @ q_transform, np.eye(n), rtol=1e-12, atol=1e-12)

    # Verify: A_out = Q^T @ A_orig @ Q
    assert_allclose(a_out, q_transform.T @ a_orig @ q_transform, rtol=1e-12, atol=1e-12)

    # Verify: B_out = Q^T @ B_orig
    assert_allclose(b_out, q_transform.T @ b_orig, rtol=1e-12, atol=1e-12)


def test_tb01md_eigenvalue_preservation():
    """
    Validate eigenvalue preservation under similarity transformation.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tb01md

    np.random.seed(456)
    n, m = 6, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    # Compute eigenvalues before transformation
    eig_before = np.linalg.eigvals(a)

    a_out, b_out, u_out, info = tb01md('N', 'U', a, b)

    assert info == 0

    # Compute eigenvalues after transformation
    eig_after = np.linalg.eigvals(a_out)

    # Eigenvalues should be preserved (sort for comparison)
    eig_before_sorted = np.sort(eig_before)
    eig_after_sorted = np.sort(eig_after)

    assert_allclose(eig_before_sorted.real, eig_after_sorted.real, rtol=1e-13, atol=1e-14)
    assert_allclose(eig_before_sorted.imag, eig_after_sorted.imag, rtol=1e-13, atol=1e-14)


def test_tb01md_m_geq_n():
    """
    Test case where M >= N (B becomes trapezoidal, A stays full).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tb01md

    np.random.seed(789)
    n, m = 3, 5  # m > n

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    a_orig = a.copy()
    b_orig = b.copy()

    a_out, b_out, u_out, info = tb01md('I', 'U', a, b)

    assert info == 0

    # Verify U is orthogonal
    assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-12, atol=1e-12)

    # Verify similarity transformation
    assert_allclose(a_out, u_out.T @ a_orig @ u_out, rtol=1e-12, atol=1e-12)
    assert_allclose(b_out, u_out.T @ b_orig, rtol=1e-12, atol=1e-12)


def test_tb01md_n_zero():
    """Test edge case with N=0."""
    from ctrlsys import tb01md

    a = np.empty((0, 0), order='F', dtype=float)
    b = np.empty((0, 2), order='F', dtype=float)

    a_out, b_out, u_out, info = tb01md('I', 'U', a, b)

    assert info == 0
    assert a_out.shape == (0, 0)
    assert b_out.shape == (0, 2)


def test_tb01md_m_zero():
    """Test edge case with M=0 - A should be unchanged."""
    from ctrlsys import tb01md

    np.random.seed(111)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.empty((n, 0), order='F', dtype=float)
    a_orig = a.copy()

    a_out, b_out, u_out, info = tb01md('I', 'U', a, b)

    assert info == 0
    # When M=0, A is unchanged
    assert_allclose(a_out, a_orig, rtol=1e-12, atol=1e-12)
    # U should be identity when no transformation needed
    assert_allclose(u_out, np.eye(n), rtol=1e-12, atol=1e-12)


def test_tb01md_invalid_jobu():
    """Test error handling for invalid JOBU parameter."""
    from ctrlsys import tb01md

    a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    b = np.array([[1.0], [2.0]], order='F', dtype=float)

    with pytest.raises(ValueError):
        tb01md('X', 'U', a, b)


def test_tb01md_invalid_uplo():
    """Test error handling for invalid UPLO parameter."""
    from ctrlsys import tb01md

    a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    b = np.array([[1.0], [2.0]], order='F', dtype=float)

    with pytest.raises(ValueError):
        tb01md('N', 'X', a, b)
