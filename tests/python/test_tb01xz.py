"""
Tests for TB01XZ - Special similarity transformation of dual state-space system (complex)

TB01XZ applies the transformation:
  A <-- P * A' * P,  B <-- P * C',  C <-- B' * P

where P is the permutation matrix with 1s on the secondary diagonal (anti-identity).
Optionally, D is transposed. Matrix A can be band-limited.

This is the complex version of TB01XD. Note: A' means transpose, NOT conjugate transpose.
"""
import numpy as np
import pytest
from ctrlsys import tb01xz


def create_permutation_matrix(n):
    """
    Create the permutation matrix P with 1s on the secondary diagonal.

    P = [[0, ..., 0, 1],
         [0, ..., 1, 0],
         [...],
         [1, 0, ..., 0]]
    """
    P = np.zeros((n, n), order='F', dtype=complex)
    for i in range(n):
        P[i, n - 1 - i] = 1.0
    return P


"""Basic functionality tests."""

def test_small_system_full_matrix():
    """
    Validate basic transformation on 3x3 full matrix (complex).

    Tests: A <-- P*A'*P, B <-- P*C', C <-- B'*P, D <-- D'
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 3, 2, 2

    a_orig = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b_orig = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c_orig = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')
    d_orig = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(complex, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = n - 1
    ku = n - 1

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)

    a_expected = P @ a_orig.T @ P
    b_expected = P @ c_orig.T
    c_expected = b_orig.T @ P
    d_expected = d_orig.T

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out[:, :p], b_expected, rtol=1e-14)
    np.testing.assert_allclose(c_out[:m, :], c_expected, rtol=1e-14)
    np.testing.assert_allclose(d_out[:m, :p], d_expected, rtol=1e-14)

def test_non_square_io():
    """
    Validate transformation with M != P (more outputs than inputs).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 4, 2, 3
    maxmp = max(m, p)

    a_orig = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b_orig = np.zeros((n, maxmp), order='F', dtype=complex)
    b_orig[:, :m] = (np.random.randn(n, m) + 1j * np.random.randn(n, m))
    c_orig = np.zeros((maxmp, n), order='F', dtype=complex)
    c_orig[:p, :] = (np.random.randn(p, n) + 1j * np.random.randn(p, n))
    d_orig = np.zeros((maxmp, maxmp), order='F', dtype=complex)
    d_orig[:p, :m] = (np.random.randn(p, m) + 1j * np.random.randn(p, m))

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = n - 1
    ku = n - 1

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)

    a_expected = P @ a_orig.T @ P
    b_expected = P @ c_orig[:p, :].T
    c_expected = b_orig[:, :m].T @ P
    d_expected = d_orig[:p, :m].T

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out[:, :p], b_expected, rtol=1e-14)
    np.testing.assert_allclose(c_out[:m, :], c_expected, rtol=1e-14)
    np.testing.assert_allclose(d_out[:m, :p], d_expected, rtol=1e-14)


"""Mathematical property tests."""

def test_involution_property():
    """
    Validate involution: applying transformation twice returns original.

    For pertransposition: (P*A'*P)' = P*A*P
    Applying twice: P*(P*A'*P)'*P = P*P*A*P*P = A (since P^2 = I)

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 3, 2, 2

    a_orig = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b_orig = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c_orig = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')
    d_orig = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(complex, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = n - 1
    ku = n - 1

    a1, b1, c1, d1, info1 = tb01xz('D', n, m, p, kl, ku, a, b, c, d)
    assert info1 == 0

    a2, b2, c2, d2, info2 = tb01xz('D', n, p, m, kl, ku, a1, b1, c1, d1)
    assert info2 == 0

    np.testing.assert_allclose(a2, a_orig, rtol=1e-14)
    np.testing.assert_allclose(b2[:, :m], b_orig, rtol=1e-14)
    np.testing.assert_allclose(c2[:p, :], c_orig, rtol=1e-14)
    np.testing.assert_allclose(d2[:p, :m], d_orig, rtol=1e-14)

def test_eigenvalue_preservation():
    """
    Validate eigenvalue preservation under similarity transformation.

    P*A'*P is similar to A' (same eigenvalues as A since transpose preserves eigenvalues).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 5, 2, 3
    maxmp = max(m, p)

    a_orig = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b = np.zeros((n, maxmp), order='F', dtype=complex)
    b[:, :m] = (np.random.randn(n, m) + 1j * np.random.randn(n, m))
    c = np.zeros((maxmp, n), order='F', dtype=complex)
    c[:p, :] = (np.random.randn(p, n) + 1j * np.random.randn(p, n))
    d = np.zeros((maxmp, maxmp), order='F', dtype=complex)
    d[:p, :m] = (np.random.randn(p, m) + 1j * np.random.randn(p, m))

    eig_before = np.linalg.eigvals(a_orig)

    a = a_orig.copy(order='F')
    kl = n - 1
    ku = n - 1

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    eig_after = np.linalg.eigvals(a_out)

    np.testing.assert_allclose(
        sorted(eig_before.real),
        sorted(eig_after.real),
        rtol=1e-13
    )
    np.testing.assert_allclose(
        sorted(eig_before.imag),
        sorted(eig_after.imag),
        rtol=1e-13
    )


"""Tests for band matrix handling."""

def test_diagonal_matrix():
    """
    Validate band matrix with KL=KU=0 (diagonal matrix).

    For diagonal matrix, pertransposition reverses diagonal entries.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m, p = 4, 2, 2

    diag_vals = np.random.randn(n) + 1j * np.random.randn(n)
    a_orig = np.diag(diag_vals).astype(complex, order='F')
    b_orig = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c_orig = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')
    d_orig = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(complex, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = 0
    ku = 0

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)
    a_expected = P @ a_orig.T @ P

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)

def test_tridiagonal_matrix():
    """
    Validate band matrix with KL=KU=1 (tridiagonal).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, p = 5, 2, 2

    a_orig = np.zeros((n, n), order='F', dtype=complex)
    a_orig += np.diag(np.random.randn(n) + 1j * np.random.randn(n))
    a_orig += np.diag(np.random.randn(n - 1) + 1j * np.random.randn(n - 1), 1)
    a_orig += np.diag(np.random.randn(n - 1) + 1j * np.random.randn(n - 1), -1)
    a_orig = np.asfortranarray(a_orig)

    b_orig = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c_orig = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')
    d_orig = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(complex, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = 1
    ku = 1

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)
    a_expected = P @ a_orig.T @ P

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)


"""Edge case tests."""

def test_n_zero():
    """Test N=0 case (quick return with D transpose only)."""
    n, m, p = 0, 2, 3
    maxmp = max(m, p)

    a = np.zeros((1, 1), order='F', dtype=complex)
    b = np.zeros((1, maxmp), order='F', dtype=complex)
    c = np.zeros((maxmp, 1), order='F', dtype=complex)
    d_orig = (np.random.randn(maxmp, maxmp) + 1j * np.random.randn(maxmp, maxmp)).astype(complex, order='F')
    d_orig_pxm = d_orig[:p, :m].copy()
    d = d_orig.copy(order='F')

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, 0, 0, a, b, c, d)

    assert info == 0
    np.testing.assert_allclose(d_out[:m, :p], d_orig_pxm.T, rtol=1e-14)

def test_jobd_z_no_d_processing():
    """
    Test JOBD='Z' mode where D is zero (not processed).

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n, m, p = 3, 2, 2

    a_orig = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b_orig = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c_orig = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')
    d = np.zeros((max(m, p), max(m, p)), order='F', dtype=complex)

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    kl = n - 1
    ku = n - 1

    a_out, b_out, c_out, d_out, info = tb01xz('Z', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)

    a_expected = P @ a_orig.T @ P
    b_expected = P @ c_orig.T
    c_expected = b_orig.T @ P

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out[:, :p], b_expected, rtol=1e-14)
    np.testing.assert_allclose(c_out[:m, :], c_expected, rtol=1e-14)

def test_m_greater_than_p():
    """
    Test M > P case (more inputs than outputs).

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n, m, p = 3, 4, 2
    maxmp = max(m, p)

    a_orig = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b_orig = np.zeros((n, maxmp), order='F', dtype=complex)
    b_orig[:, :m] = (np.random.randn(n, m) + 1j * np.random.randn(n, m))
    c_orig = np.zeros((maxmp, n), order='F', dtype=complex)
    c_orig[:p, :] = (np.random.randn(p, n) + 1j * np.random.randn(p, n))
    d_orig = np.zeros((maxmp, maxmp), order='F', dtype=complex)
    d_orig[:p, :m] = (np.random.randn(p, m) + 1j * np.random.randn(p, m))

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = n - 1
    ku = n - 1

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)

    a_expected = P @ a_orig.T @ P
    b_expected = P @ c_orig[:p, :].T
    c_expected = b_orig[:, :m].T @ P
    d_expected = d_orig[:p, :m].T

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out[:, :p], b_expected, rtol=1e-14)
    np.testing.assert_allclose(c_out[:m, :], c_expected, rtol=1e-14)
    np.testing.assert_allclose(d_out[:m, :p], d_expected, rtol=1e-14)

def test_siso_system():
    """
    Test SISO system (M=1, P=1).

    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    n, m, p = 4, 1, 1

    a_orig = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
    b_orig = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c_orig = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')
    d_orig = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(complex, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = n - 1
    ku = n - 1

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)

    a_expected = P @ a_orig.T @ P
    b_expected = P @ c_orig.T
    c_expected = b_orig.T @ P
    d_expected = d_orig.T

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)
    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)
    np.testing.assert_allclose(d_out, d_expected, rtol=1e-14)


"""Error handling tests."""

def test_invalid_jobd():
    """Test invalid JOBD parameter."""
    n, m, p = 2, 1, 1
    a = np.zeros((n, n), order='F', dtype=complex)
    b = np.zeros((n, 1), order='F', dtype=complex)
    c = np.zeros((1, n), order='F', dtype=complex)
    d = np.zeros((1, 1), order='F', dtype=complex)

    with pytest.raises(ValueError):
        tb01xz('X', n, m, p, 1, 1, a, b, c, d)

def test_negative_n():
    """Test negative N parameter."""
    n, m, p = -1, 1, 1
    a = np.zeros((1, 1), order='F', dtype=complex)
    b = np.zeros((1, 1), order='F', dtype=complex)
    c = np.zeros((1, 1), order='F', dtype=complex)
    d = np.zeros((1, 1), order='F', dtype=complex)

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, 0, 0, a, b, c, d)
    assert info == -2

def test_negative_m():
    """Test negative M parameter."""
    n, m, p = 2, -1, 1
    a = np.zeros((n, n), order='F', dtype=complex)
    b = np.zeros((n, 1), order='F', dtype=complex)
    c = np.zeros((1, n), order='F', dtype=complex)
    d = np.zeros((1, 1), order='F', dtype=complex)

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, 1, 1, a, b, c, d)
    assert info == -3

def test_negative_p():
    """Test negative P parameter."""
    n, m, p = 2, 1, -1
    a = np.zeros((n, n), order='F', dtype=complex)
    b = np.zeros((n, 1), order='F', dtype=complex)
    c = np.zeros((1, n), order='F', dtype=complex)
    d = np.zeros((1, 1), order='F', dtype=complex)

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, 1, 1, a, b, c, d)
    assert info == -4

def test_invalid_kl():
    """Test invalid KL parameter (KL > N-1)."""
    n, m, p = 3, 1, 1
    a = np.zeros((n, n), order='F', dtype=complex)
    b = np.zeros((n, 1), order='F', dtype=complex)
    c = np.zeros((1, n), order='F', dtype=complex)
    d = np.zeros((1, 1), order='F', dtype=complex)

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, n, 1, a, b, c, d)
    assert info == -5

def test_invalid_ku():
    """Test invalid KU parameter (KU > N-1)."""
    n, m, p = 3, 1, 1
    a = np.zeros((n, n), order='F', dtype=complex)
    b = np.zeros((n, 1), order='F', dtype=complex)
    c = np.zeros((1, n), order='F', dtype=complex)
    d = np.zeros((1, 1), order='F', dtype=complex)

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, 1, n, a, b, c, d)
    assert info == -6


"""Complex-specific tests."""

def test_purely_imaginary_matrix():
    """
    Test with purely imaginary system matrices.

    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)
    n, m, p = 3, 2, 2

    a_orig = 1j * np.random.randn(n, n).astype(complex, order='F')
    b_orig = 1j * np.random.randn(n, m).astype(complex, order='F')
    c_orig = 1j * np.random.randn(p, n).astype(complex, order='F')
    d_orig = 1j * np.random.randn(p, m).astype(complex, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = n - 1
    ku = n - 1

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)

    a_expected = P @ a_orig.T @ P
    b_expected = P @ c_orig.T
    c_expected = b_orig.T @ P
    d_expected = d_orig.T

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out[:, :p], b_expected, rtol=1e-14)
    np.testing.assert_allclose(c_out[:m, :], c_expected, rtol=1e-14)
    np.testing.assert_allclose(d_out[:m, :p], d_expected, rtol=1e-14)

def test_hermitian_matrix():
    """
    Test with Hermitian A matrix.

    For Hermitian A: A = A^H (conjugate transpose).
    Pertransposition: P*A'*P (transpose, not conjugate).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m, p = 4, 1, 1

    a_rand = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    a_orig = ((a_rand + a_rand.conj().T) / 2).astype(complex, order='F')
    b_orig = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(complex, order='F')
    c_orig = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(complex, order='F')
    d_orig = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(complex, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = n - 1
    ku = n - 1

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)
    a_expected = P @ a_orig.T @ P

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)

def test_real_valued_complex_arrays():
    """
    Test with real-valued data stored in complex arrays.

    Results should match TB01XD behavior.
    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 3, 2, 2

    a_orig = np.random.randn(n, n).astype(complex, order='F')
    b_orig = np.random.randn(n, m).astype(complex, order='F')
    c_orig = np.random.randn(p, n).astype(complex, order='F')
    d_orig = np.random.randn(p, m).astype(complex, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    kl = n - 1
    ku = n - 1

    a_out, b_out, c_out, d_out, info = tb01xz('D', n, m, p, kl, ku, a, b, c, d)

    assert info == 0

    P = create_permutation_matrix(n)

    a_expected = P @ a_orig.T @ P
    b_expected = P @ c_orig.T
    c_expected = b_orig.T @ P
    d_expected = d_orig.T

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out[:, :p], b_expected, rtol=1e-14)
    np.testing.assert_allclose(c_out[:m, :], c_expected, rtol=1e-14)
    np.testing.assert_allclose(d_out[:m, :p], d_expected, rtol=1e-14)

    assert np.all(a_out.imag == 0)
    assert np.all(b_out[:, :p].imag == 0)
    assert np.all(c_out[:m, :].imag == 0)
    assert np.all(d_out[:m, :p].imag == 0)
