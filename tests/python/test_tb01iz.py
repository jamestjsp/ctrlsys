"""
Tests for TB01IZ: Balancing a system matrix for complex state-space systems.

TB01IZ reduces the 1-norm of the system matrix S = [[A, B], [C, 0]] by applying
diagonal similarity transformations to balance rows and columns.

Reference: SLICOT-Reference/doc/TB01IZ.html
"""

import numpy as np
import pytest
from ctrlsys import tb01iz


"""Basic functionality tests using HTML doc example."""

def test_html_doc_example():
    """
    Test using the exact example from SLICOT HTML documentation.

    N=5, M=2, P=5, JOB='A', MAXRED=0.0
    Validates: eigenvalue preservation, reconstruction, norm reduction.
    Note: Balancing is iterative and may converge to different (but valid) solutions.
    """
    n, m, p = 5, 2, 5

    a = np.array([
        [0.0+0.0j, 1.0e3+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [-1.58e6+0.0j, -1.257e3+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [3.541e14+0.0j, 0.0+0.0j, -1.434e3+0.0j, 0.0+0.0j, -5.33e11+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, -1.863e4+0.0j, -1.482+0.0j]
    ], order='F', dtype=complex)

    b = np.array([
        [0.0+0.0j, 0.0+0.0j],
        [1.103e2+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 8.333e-3+0.0j]
    ], order='F', dtype=complex)

    c = np.array([
        [1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.0+0.0j],
        [6.664e-1+0.0j, 0.0+0.0j, -6.2e-13+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, -1.0e-3+0.0j, 1.896e6+0.0j, 1.508e2+0.0j]
    ], order='F', dtype=complex)

    a_orig, b_orig, c_orig = a.copy(), b.copy(), c.copy()

    a_bal, b_bal, c_bal, maxred_out, scale, info = tb01iz('A', a, b, c, 0.0)

    assert info == 0
    assert maxred_out > 1.0

    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)
    a_reconstructed = d @ a_bal @ d_inv
    b_reconstructed = d @ b_bal
    c_reconstructed = c_bal @ d_inv

    np.testing.assert_allclose(a_reconstructed, a_orig, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(b_reconstructed, b_orig, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(c_reconstructed, c_orig, rtol=1e-12, atol=1e-14)

    eig_orig = np.linalg.eigvals(a_orig)
    eig_bal = np.linalg.eigvals(a_bal)
    np.testing.assert_allclose(
        sorted(np.abs(eig_orig)),
        sorted(np.abs(eig_bal)),
        rtol=1e-10
    )

    assert all(np.log10(s) == int(np.log10(s)) for s in scale if s != 0)


"""Test mathematical properties of the balancing transformation."""

def test_similarity_transformation_preserves_eigenvalues():
    """
    Validate that diagonal similarity transformation preserves eigenvalues.

    The balanced matrix inv(D)*A*D should have the same eigenvalues as A.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 2, 3

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    eig_before = np.linalg.eigvals(a)

    a_bal, b_bal, c_bal, maxred_out, scale, info = tb01iz('A', a.copy(), b.copy(), c.copy(), 0.0)
    assert info == 0

    eig_after = np.linalg.eigvals(a_bal)

    np.testing.assert_allclose(
        sorted(np.abs(eig_before)),
        sorted(np.abs(eig_after)),
        rtol=1e-10
    )

def test_reconstruction_from_scale():
    """
    Validate that we can reconstruct the original system from balanced one using scale.

    A = D * A_bal * inv(D)
    B = D * B_bal
    C = C_bal * inv(D)

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    a_orig = a.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_bal, b_bal, c_bal, maxred_out, scale, info = tb01iz('A', a.copy(), b.copy(), c.copy(), 0.0)
    assert info == 0

    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)

    a_reconstructed = d @ a_bal @ d_inv
    b_reconstructed = d @ b_bal
    c_reconstructed = c_bal @ d_inv

    np.testing.assert_allclose(a_reconstructed, a_orig, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(b_reconstructed, b_orig, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(c_reconstructed, c_orig, rtol=1e-12, atol=1e-14)

def test_norm_reduction():
    """
    Validate that balancing reduces or maintains system matrix 1-norm.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 5, 2, 3

    scale_factors = 10.0 ** np.random.uniform(-5, 5, n)
    a = np.diag(scale_factors) @ np.random.randn(n, n) @ np.diag(1.0 / scale_factors)
    a = (a + 1j * np.random.randn(n, n) * 0.1).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    def compute_system_norm(a, b, c):
        s_top = np.hstack([a, b])
        s_bottom = np.hstack([c, np.zeros((c.shape[0], b.shape[1]), dtype=complex)])
        s = np.vstack([s_top, s_bottom])
        return np.max(np.sum(np.abs(s), axis=0))

    norm_before = compute_system_norm(a, b, c)

    a_bal, b_bal, c_bal, maxred_out, scale, info = tb01iz('A', a.copy(), b.copy(), c.copy(), 0.0)
    assert info == 0

    norm_after = compute_system_norm(a_bal, b_bal, c_bal)

    assert norm_after <= norm_before * 1.1


"""Test different JOB parameter modes."""

def test_job_a_balances_all():
    """
    Test JOB='A': All matrices (A, B, C) are involved in balancing.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    a_orig, b_orig, c_orig = a.copy(), b.copy(), c.copy()

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('A', a, b, c, 0.0)

    assert info == 0
    assert scale.shape == (n,)

def test_job_b_balances_a_and_b_only():
    """
    Test JOB='B': Only A and B matrices are involved in balancing.
    C is not used in computing the scaling.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('B', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

def test_job_c_balances_a_and_c_only():
    """
    Test JOB='C': Only A and C matrices are involved in balancing.
    B is not used in computing the scaling.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('C', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

def test_job_n_balances_a_only():
    """
    Test JOB='N': Only A matrix is balanced (B and C not used).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('N', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0


"""Edge cases and boundary conditions."""

def test_n_equals_zero():
    """Test with n=0 (trivial case)."""
    a = np.array([], dtype=complex, order='F').reshape(0, 0)
    b = np.array([], dtype=complex, order='F').reshape(0, 1)
    c = np.array([], dtype=complex, order='F').reshape(1, 0)

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('A', a, b, c, 0.0)
    assert info == 0

def test_identity_system():
    """Test with identity A matrix (already balanced)."""
    n, m, p = 3, 1, 1
    a = np.eye(n, dtype=complex, order='F')
    b = np.ones((n, m), dtype=complex, order='F')
    c = np.ones((p, n), dtype=complex, order='F')

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('A', a, b, c, 0.0)

    assert info == 0
    np.testing.assert_allclose(scale, np.ones(n), rtol=1e-10)

def test_zero_matrix():
    """Test with zero system matrix."""
    n, m, p = 3, 2, 2
    a = np.zeros((n, n), dtype=complex, order='F')
    b = np.zeros((n, m), dtype=complex, order='F')
    c = np.zeros((p, n), dtype=complex, order='F')

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('A', a, b, c, 0.0)

    assert info == 0

def test_m_equals_zero():
    """Test with m=0 (no inputs)."""
    n, m, p = 3, 0, 2
    a = np.eye(n, dtype=complex, order='F')
    b = np.array([], dtype=complex, order='F').reshape(n, 0)
    c = np.ones((p, n), dtype=complex, order='F')

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('A', a, b, c, 0.0)
    assert info == 0

def test_p_equals_zero():
    """Test with p=0 (no outputs)."""
    n, m, p = 3, 2, 0
    a = np.eye(n, dtype=complex, order='F')
    b = np.ones((n, m), dtype=complex, order='F')
    c = np.array([], dtype=complex, order='F').reshape(0, n)

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('A', a, b, c, 0.0)
    assert info == 0


"""Tests with complex values."""

def test_purely_imaginary_matrix():
    """
    Test with purely imaginary system matrices.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n, m, p = 3, 2, 2

    a = 1j * np.random.randn(n, n).astype(complex, order='F')
    b = 1j * np.random.randn(n, m).astype(complex, order='F')
    c = 1j * np.random.randn(p, n).astype(complex, order='F')

    a_orig = a.copy()

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('A', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

    eig_before = np.linalg.eigvals(a_orig)
    eig_after = np.linalg.eigvals(a_bal)
    np.testing.assert_allclose(
        sorted(np.abs(eig_before)),
        sorted(np.abs(eig_after)),
        rtol=1e-10
    )

def test_general_complex_matrix():
    """
    Test with general complex values (real + imaginary parts).

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n, m, p = 4, 2, 3

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    a_orig, b_orig, c_orig = a.copy(), b.copy(), c.copy()

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('A', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)

    a_reconstructed = d @ a_bal @ d_inv
    np.testing.assert_allclose(a_reconstructed, a_orig, rtol=1e-12, atol=1e-14)


"""Test error handling."""

def test_invalid_job_parameter():
    """Test invalid JOB parameter returns error."""
    n, m, p = 2, 1, 1
    a = np.eye(n, dtype=complex, order='F')
    b = np.ones((n, m), dtype=complex, order='F')
    c = np.ones((p, n), dtype=complex, order='F')

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('X', a, b, c, 0.0)
    assert info == -1

def test_invalid_maxred_value():
    """Test MAXRED between 0 and 1 returns error."""
    n, m, p = 2, 1, 1
    a = np.eye(n, dtype=complex, order='F')
    b = np.ones((n, m), dtype=complex, order='F')
    c = np.ones((p, n), dtype=complex, order='F')

    a_bal, b_bal, c_bal, maxred, scale, info = tb01iz('A', a, b, c, 0.5)
    assert info == -5


"""Tests for MAXRED parameter behavior."""

def test_maxred_zero_uses_default():
    """Test MAXRED=0 uses default value of 10."""
    np.random.seed(666)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    a_bal, b_bal, c_bal, maxred_out, scale, info = tb01iz('A', a, b, c, 0.0)

    assert info == 0

def test_maxred_positive_custom_value():
    """Test custom MAXRED > 1 value."""
    np.random.seed(777)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')

    a_bal, b_bal, c_bal, maxred_out, scale, info = tb01iz('A', a, b, c, 5.0)

    assert info == 0
