"""
Tests for MB03QY: 2x2 diagonal block eigenvalue computation and standardization.

Transforms a 2x2 diagonal block of an upper quasi-triangular matrix to
standard Schur form (split if real eigenvalues, standardize if complex).
"""
import numpy as np
import pytest
from ctrlsys import mb03qy


"""Basic functionality tests."""

def test_complex_eigenvalues():
    """
    Test 2x2 block with complex conjugate eigenvalues.
    Block at L=1 (0-indexed: position 0).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3
    l = 1  # 1-based position

    # Construct upper quasi-triangular with complex 2x2 block at (0,0)
    # Block: [[2, 3], [-1, 2]] has eigenvalues 2 +/- i
    a = np.array([
        [2.0, 3.0, 1.5],
        [-1.0, 2.0, 0.5],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)

    a_out, u_out, e1, e2, info = mb03qy(n, l, a, u)

    assert info == 0

    # Complex eigenvalues: e1=real part, e2=positive imaginary part
    # Block [[2, 3], [-1, 2]] has eigenvalues 2 +/- sqrt(3)i
    np.testing.assert_allclose(e1, 2.0, rtol=1e-14)
    np.testing.assert_allclose(e2, np.sqrt(3.0), rtol=1e-14)

    # Verify standard form: a[0,0] = a[1,1] and a[0,1]*a[1,0] < 0
    np.testing.assert_allclose(a_out[0, 0], a_out[1, 1], rtol=1e-14)
    assert a_out[0, 1] * a_out[1, 0] < 0

def test_real_eigenvalues():
    """
    Test 2x2 block with real eigenvalues that can be split.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 2
    l = 1  # 1-based

    # Block: [[4, 2], [1, 3]] has eigenvalues 5 and 2
    a = np.array([
        [4.0, 2.0],
        [1.0, 3.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)

    a_out, u_out, e1, e2, info = mb03qy(n, l, a, u)

    assert info == 0

    # For real eigenvalues: |e1| >= |e2|
    assert abs(e1) >= abs(e2) or np.isclose(abs(e1), abs(e2))

    # Block should be upper triangular (subdiagonal = 0)
    np.testing.assert_allclose(a_out[1, 0], 0.0, atol=1e-14)

    # Eigenvalues should be 5 and 2
    eigs = sorted([e1, e2], reverse=True)
    np.testing.assert_allclose(eigs, [5.0, 2.0], rtol=1e-14)

def test_similarity_transformation():
    """
    Verify similarity transformation preserves eigenvalues.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4
    l = 2  # Block at position 2 (0-indexed: rows 1-2)

    # Upper quasi-triangular matrix
    a = np.array([
        [1.0, 0.5, 0.3, 0.1],
        [0.0, 3.0, 2.0, 0.4],
        [0.0, -1.0, 3.0, 0.2],
        [0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)
    a_orig = a.copy()

    a_out, u_out, e1, e2, info = mb03qy(n, l, a, u)

    assert info == 0

    # Verify U is orthogonal
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-14, atol=1e-14)

    # Verify similarity: A_out = U^T * A_orig * U
    a_reconstructed = u_out.T @ a_orig @ u_out
    np.testing.assert_allclose(a_out, a_reconstructed, rtol=1e-13, atol=1e-14)


"""Error handling tests."""

def test_invalid_n():
    """N must be >= 2."""
    a = np.array([[1.0]], order='F', dtype=float)
    u = np.array([[1.0]], order='F', dtype=float)

    a_out, u_out, e1, e2, info = mb03qy(1, 1, a, u)
    assert info == -1

def test_invalid_l_low():
    """L must be >= 1."""
    a = np.eye(3, order='F', dtype=float)
    u = np.eye(3, order='F', dtype=float)

    a_out, u_out, e1, e2, info = mb03qy(3, 0, a, u)
    assert info == -2

def test_invalid_l_high():
    """L must be < N."""
    a = np.eye(3, order='F', dtype=float)
    u = np.eye(3, order='F', dtype=float)

    a_out, u_out, e1, e2, info = mb03qy(3, 3, a, u)
    assert info == -2


"""Mathematical property validation tests."""

def test_eigenvalue_preservation():
    """
    Verify eigenvalues of the 2x2 block are preserved.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3
    l = 1

    # 2x2 block with known eigenvalues: 1 +/- 2i
    a = np.array([
        [1.0, 4.0, 0.5],
        [-1.0, 1.0, 0.3],
        [0.0, 0.0, 5.0]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)

    # Compute eigenvalues of original 2x2 block
    block_orig = a[:2, :2].copy()
    eigs_orig = np.linalg.eigvals(block_orig)

    a_out, u_out, e1, e2, info = mb03qy(n, l, a, u)
    assert info == 0

    # Compare with returned eigenvalues
    # Complex case: e1 = real, e2 = positive imag
    if e2 != 0:
        eig_computed = [complex(e1, e2), complex(e1, -e2)]
        np.testing.assert_allclose(
            sorted(eigs_orig.real),
            sorted([e.real for e in eig_computed]),
            rtol=1e-14
        )
        np.testing.assert_allclose(
            sorted(np.abs(eigs_orig.imag)),
            sorted([abs(e.imag) for e in eig_computed]),
            rtol=1e-14
        )
    else:
        # Real case
        np.testing.assert_allclose(
            sorted(eigs_orig.real),
            sorted([e1, e2]),
            rtol=1e-14
        )

def test_orthogonality_of_transformation():
    """
    U * UT should remain orthogonal when U starts as identity.

    Random seed: 321 (for reproducibility)
    """
    np.random.seed(321)
    n = 5
    l = 3

    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 1.0
    a[1, 1] = 2.0
    # 2x2 block at l=3 (indices 2,3)
    a[2, 2] = 4.0
    a[2, 3] = 3.0
    a[3, 2] = -2.0
    a[3, 3] = 4.0
    a[4, 4] = 6.0

    u = np.eye(n, order='F', dtype=float)

    a_out, u_out, e1, e2, info = mb03qy(n, l, a, u)

    assert info == 0

    # U should be orthogonal
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(u_out @ u_out.T, np.eye(n), rtol=1e-14, atol=1e-14)
