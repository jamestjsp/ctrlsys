"""
Tests for MB05MY: Schur form, eigenvalues, and right eigenvectors.

Computes for an N-by-N real nonsymmetric matrix A:
  - Orthogonal matrix Q reducing A to real Schur form T
  - Eigenvalues (WR + i*WI)
  - Right eigenvectors R of T (upper triangular by construction)
"""
import numpy as np
import pytest
from ctrlsys import mb05my


"""Basic functionality tests."""

def test_small_real_eigenvalues():
    """
    Test 3x3 matrix with real eigenvalues.

    Matrix has eigenvalues 1, 2, 3.
    Random seed: 42 (for reproducibility)
    """
    a = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 2.0, 1.0],
        [0.0, 0.0, 3.0],
    ], order='F', dtype=float)

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0

    eig_computed = np.sort(wr)
    eig_expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(eig_computed, eig_expected, rtol=1e-14)

    np.testing.assert_allclose(wi, np.zeros(3), atol=1e-14)

def test_complex_eigenvalues_2x2():
    """
    Test 2x2 rotation matrix with complex conjugate eigenvalues.

    Matrix [[0, -1], [1, 0]] has eigenvalues +-i.
    """
    a = np.array([
        [0.0, -1.0],
        [1.0, 0.0],
    ], order='F', dtype=float)

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0

    np.testing.assert_allclose(wr, [0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(np.abs(wi), [1.0, 1.0], rtol=1e-14)
    assert wi[0] == -wi[1]

def test_4x4_mixed_eigenvalues():
    """
    Test 4x4 matrix with 2 real and 2 complex eigenvalues.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    a = np.random.randn(n, n).astype(float, order='F')

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0

    eig_numpy = np.linalg.eigvals(a)
    eig_computed = wr + 1j * wi

    np.testing.assert_allclose(
        sorted(eig_numpy, key=lambda x: (x.real, x.imag)),
        sorted(eig_computed, key=lambda x: (x.real, x.imag)),
        rtol=1e-12
    )


"""Mathematical property tests for Schur decomposition."""

def test_orthogonality_of_q():
    """
    Validate Q is orthogonal: Q^T * Q = I.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 5
    a = np.random.randn(n, n).astype(float, order='F')

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0
    np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q @ q.T, np.eye(n), rtol=1e-14, atol=1e-14)

def test_schur_decomposition_a_equals_q_t_qt():
    """
    Validate Schur decomposition: A = Q * T * Q^T.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4
    a = np.random.randn(n, n).astype(float, order='F')
    a_copy = a.copy()

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0
    a_reconstructed = q @ t @ q.T
    np.testing.assert_allclose(a_reconstructed, a_copy, rtol=1e-13, atol=1e-14)

def test_t_is_quasi_upper_triangular():
    """
    Validate T is quasi-upper triangular (real Schur form).

    For real eigenvalues: T is strictly upper triangular at that row.
    For complex pairs: 2x2 block on diagonal.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5
    a = np.random.randn(n, n).astype(float, order='F')

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0

    for i in range(n - 1):
        if wi[i] == 0.0 and wi[i + 1] == 0.0:
            np.testing.assert_allclose(t[i + 1, i], 0.0, atol=1e-14)

def test_eigenvalue_preservation():
    """
    Validate eigenvalues of T match eigenvalues of A.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n = 6
    a = np.random.randn(n, n).astype(float, order='F')

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0

    eig_a = np.linalg.eigvals(a)
    eig_t = np.linalg.eigvals(t)

    np.testing.assert_allclose(
        sorted(eig_a, key=lambda x: (x.real, x.imag)),
        sorted(eig_t, key=lambda x: (x.real, x.imag)),
        rtol=1e-13
    )


"""Mathematical property tests for eigenvectors."""

def test_eigenvector_equation_real():
    """
    Validate T * r = lambda * r for real eigenvalues.

    Random seed: 111 (for reproducibility)
    """
    a = np.array([
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 4.0],
    ], order='F', dtype=float)

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0

    for j in range(3):
        if wi[j] == 0.0:
            rj = r[:, j]
            lhs = t @ rj
            rhs = wr[j] * rj
            np.testing.assert_allclose(lhs, rhs, rtol=1e-14, atol=1e-14)

def test_r_is_upper_triangular():
    """
    Validate R is upper triangular by construction.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 5
    a = np.random.randn(n, n).astype(float, order='F')

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0

    for j in range(n):
        for i in range(j + 1, n):
            np.testing.assert_allclose(r[i, j], 0.0, atol=1e-14)


"""Tests for diagonal scaling (BALANC='S')."""

def test_with_scaling():
    """
    Test with BALANC='S' for badly scaled matrix.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 4
    d = np.diag([1e-5, 1.0, 1e3, 1e6])
    a_balanced = np.random.randn(n, n).astype(float, order='F')
    a = d @ a_balanced @ np.linalg.inv(d)
    a = np.asfortranarray(a)

    wr, wi, r, q, t, info = mb05my('S', a)

    assert info == 0

    eig_numpy = np.linalg.eigvals(a)
    eig_computed = wr + 1j * wi

    np.testing.assert_allclose(
        sorted(eig_numpy, key=lambda x: (x.real, x.imag)),
        sorted(eig_computed, key=lambda x: (x.real, x.imag)),
        rtol=1e-10
    )

def test_no_scaling():
    """
    Test with BALANC='N' (no scaling).

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 3
    a = np.random.randn(n, n).astype(float, order='F')

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0

    eig_numpy = np.linalg.eigvals(a)
    eig_computed = wr + 1j * wi

    np.testing.assert_allclose(
        sorted(eig_numpy, key=lambda x: (x.real, x.imag)),
        sorted(eig_computed, key=lambda x: (x.real, x.imag)),
        rtol=1e-14
    )


"""Edge case tests."""

def test_n_equals_zero():
    """Test with n=0 (empty matrix)."""
    a = np.array([], order='F', dtype=float).reshape(0, 0)

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0
    assert wr.shape == (0,)
    assert wi.shape == (0,)
    assert r.shape == (0, 0)
    assert q.shape == (0, 0)
    assert t.shape == (0, 0)

def test_n_equals_one():
    """Test scalar case."""
    a = np.array([[3.5]], order='F', dtype=float)

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0
    np.testing.assert_allclose(wr[0], 3.5, rtol=1e-14)
    np.testing.assert_allclose(wi[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(q[0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(t[0, 0], 3.5, rtol=1e-14)

def test_identity_matrix():
    """Test identity matrix (eigenvalues all 1)."""
    n = 4
    a = np.eye(n, order='F', dtype=float)

    wr, wi, r, q, t, info = mb05my('N', a)

    assert info == 0
    np.testing.assert_allclose(wr, np.ones(n), rtol=1e-14)
    np.testing.assert_allclose(wi, np.zeros(n), atol=1e-14)
    np.testing.assert_allclose(t, np.eye(n), rtol=1e-14, atol=1e-14)


"""Error handling tests."""

def test_invalid_balanc():
    """Test that invalid BALANC returns info=-1."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)

    wr, wi, r, q, t, info = mb05my('X', a)

    assert info == -1
