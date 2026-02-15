"""
Tests for MB03WD: Periodic Schur Decomposition

MB03WD computes the Schur decomposition and eigenvalues of a product of matrices
H = H_1*H_2*...*H_p, with H_1 upper Hessenberg and H_2...H_p upper triangular,
without evaluating the product.
"""

import numpy as np
import pytest

from ctrlsys import mb03vd, mb03wd


"""Basic functionality tests from SLICOT HTML documentation."""

def test_html_doc_example():
    """
    Test case from SLICOT HTML documentation.

    N=4, P=2, ILO=1, IHI=4, JOB='S', COMPZ='V'
    Two 4x4 matrices forming the product.

    The example first reduces to Hessenberg form using MB03VD,
    then applies MB03WD for periodic Schur decomposition.

    Expected eigenvalues (complex conjugate pairs and real):
        (6.449861, 7.817717)
        (6.449861, -7.817717)
        (0.091315, 0.0)
        (0.208964, 0.0)
    """
    n = 4
    p = 2
    ilo = 1
    ihi = 4
    iloz = 1
    ihiz = 4

    a = np.zeros((n, n, p), order='F', dtype=float)
    a[:, :, 0] = np.array([
        [1.5, -0.7, 3.5, -0.7],
        [1.0,  0.0, 2.0,  3.0],
        [1.5, -0.7, 2.5, -0.3],
        [1.0,  0.0, 2.0,  1.0]
    ], order='F', dtype=float)

    a[:, :, 1] = np.array([
        [1.5, -0.7, 3.5, -0.7],
        [1.0,  0.0, 2.0,  3.0],
        [1.5, -0.7, 2.5, -0.3],
        [1.0,  0.0, 2.0,  1.0]
    ], order='F', dtype=float)

    h, tau, info_vd = mb03vd(n, p, ilo, ihi, a)
    assert info_vd == 0

    h_result, z, wr, wi, info = mb03wd('S', 'I', n, p, ilo, ihi, iloz, ihiz, h)

    assert info == 0

    expected_wr = np.array([6.449861, 6.449861, 0.091315, 0.208964])
    expected_wi = np.array([7.817717, -7.817717, 0.0, 0.0])

    np.testing.assert_allclose(wr, expected_wr, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(wi, expected_wi, rtol=1e-5, atol=1e-10)

    t1 = h_result[:, :, 0]
    t2 = h_result[:, :, 1]

    assert np.abs(t1[2, 0]) < 1e-10
    assert np.abs(t1[3, 0]) < 1e-10
    assert np.abs(t1[3, 1]) < 1e-10

    for i in range(n):
        for j in range(i):
            assert np.abs(t2[i, j]) < 1e-10, f"t2[{i},{j}] = {t2[i,j]} not zero"


"""Test eigenvalue-only computation (JOB='E')."""

def test_eigenvalues_only_mode():
    """
    Test JOB='E' mode which computes eigenvalues without full Schur form.

    Uses pre-reduced Hessenberg matrices (H_1 upper Hessenberg, H_2 upper triangular).
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3
    p = 2
    ilo = 1
    ihi = n
    iloz = 1
    ihiz = n

    h1 = np.array([
        [2.0, 1.0, 0.5],
        [0.3, 1.5, 0.7],
        [0.0, 0.2, 1.0]
    ], order='F', dtype=float)

    h2 = np.triu(np.array([
        [1.0, 0.4, 0.2],
        [0.0, 0.8, 0.3],
        [0.0, 0.0, 0.5]
    ], order='F', dtype=float))

    h = np.zeros((n, n, p), order='F', dtype=float)
    h[:, :, 0] = h1
    h[:, :, 1] = h2

    product = h1 @ h2
    expected_eig = np.linalg.eigvals(product)
    expected_eig_sorted = np.sort(np.real(expected_eig))[::-1]

    h_copy = h.copy(order='F')

    h_result, z, wr, wi, info = mb03wd('E', 'N', n, p, ilo, ihi, iloz, ihiz, h_copy)

    assert info == 0
    assert wr.shape == (n,)
    assert wi.shape == (n,)

    computed_eig = []
    i = 0
    while i < n:
        if wi[i] == 0.0:
            computed_eig.append(wr[i])
            i += 1
        else:
            computed_eig.append(complex(wr[i], wi[i]))
            computed_eig.append(complex(wr[i+1], wi[i+1]))
            i += 2
    computed_eig_sorted = np.sort(np.real(computed_eig))[::-1]

    np.testing.assert_allclose(computed_eig_sorted, expected_eig_sorted, rtol=1e-3)


"""Mathematical property validation tests."""

def test_eigenvalue_preservation():
    """
    Validate eigenvalues of product are preserved by periodic Schur form.

    The eigenvalues of T_1*T_2*...*T_p should equal eigenvalues of H_1*H_2*...*H_p.

    Uses pre-reduced Hessenberg matrices (H_1 upper Hessenberg, H_2 upper triangular).
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    p = 2
    ilo = 1
    ihi = n
    iloz = 1
    ihiz = n

    h1 = np.array([
        [1.2, 0.5, -0.3, 0.1],
        [0.8, -0.4, 0.7, 0.2],
        [0.0, 0.6, 1.1, -0.2],
        [0.0, 0.0, 0.9, 0.8]
    ], order='F', dtype=float)

    h2 = np.triu(np.array([
        [0.9, 0.2, -0.1, 0.3],
        [0.0, 1.1, 0.4, -0.2],
        [0.0, 0.0, 0.7, 0.5],
        [0.0, 0.0, 0.0, 1.3]
    ], order='F', dtype=float))

    h = np.zeros((n, n, p), order='F', dtype=float)
    h[:, :, 0] = h1.copy()
    h[:, :, 1] = h2.copy()

    product_before = h1 @ h2
    eig_before = np.linalg.eigvals(product_before)

    h_result, z, wr, wi, info = mb03wd('S', 'I', n, p, ilo, ihi, iloz, ihiz, h)

    assert info == 0

    t1 = h_result[:, :, 0]
    t2 = h_result[:, :, 1]
    product_after = t1 @ t2
    eig_after = np.linalg.eigvals(product_after)

    eig_before_sorted = sorted(eig_before, key=lambda x: (x.real, x.imag))
    eig_after_sorted = sorted(eig_after, key=lambda x: (x.real, x.imag))

    np.testing.assert_allclose(
        [e.real for e in eig_before_sorted],
        [e.real for e in eig_after_sorted],
        rtol=1e-3
    )
    np.testing.assert_allclose(
        [e.imag for e in eig_before_sorted],
        [e.imag for e in eig_after_sorted],
        rtol=1e-3,
        atol=1e-9
    )

def test_transformation_orthogonality():
    """
    Validate transformation matrices Z_i are orthogonal.

    Z_i' * Z_i = I for all i = 1, ..., p.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4
    p = 3
    ilo = 1
    ihi = n
    iloz = 1
    ihiz = n

    h = np.zeros((n, n, p), order='F', dtype=float)

    h[:, :, 0] = np.array([
        [2.0, 1.0, 0.5, 0.2],
        [0.5, 1.5, 0.8, 0.3],
        [0.0, 0.4, 1.2, 0.6],
        [0.0, 0.0, 0.3, 0.9]
    ], order='F', dtype=float)

    for k in range(1, p):
        h[:, :, k] = np.triu(np.random.randn(n, n)).astype(float, order='F')
        for i in range(n):
            h[i, i, k] = np.abs(h[i, i, k]) + 0.5

    h_result, z, wr, wi, info = mb03wd('S', 'I', n, p, ilo, ihi, iloz, ihiz, h)

    assert info == 0
    assert z.shape == (n, n, p)

    identity = np.eye(n, dtype=float)
    for k in range(p):
        z_k = z[:, :, k]
        orthogonality = z_k.T @ z_k
        np.testing.assert_allclose(orthogonality, identity, rtol=1e-12, atol=1e-12)

def test_transformation_property():
    """
    Validate periodic Schur transformation property.

    Z_1' * H_1 * Z_2 = T_1 (real Schur form)
    Z_j' * H_j * Z_{j+1} = T_j (upper triangular) for j >= 2
    with Z_{p+1} = Z_1.

    Uses pre-reduced Hessenberg matrices (H_1 upper Hessenberg, H_2 upper triangular).
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4
    p = 2
    ilo = 1
    ihi = n
    iloz = 1
    ihiz = n

    h1_orig = np.array([
        [1.2, 0.5, -0.3, 0.1],
        [0.8, -0.4, 0.7, 0.2],
        [0.0, 0.6, 1.1, -0.2],
        [0.0, 0.0, 0.9, 0.8]
    ], order='F', dtype=float)

    h2_orig = np.triu(np.array([
        [0.9, 0.2, -0.1, 0.3],
        [0.0, 1.1, 0.4, -0.2],
        [0.0, 0.0, 0.7, 0.5],
        [0.0, 0.0, 0.0, 1.3]
    ], order='F', dtype=float))

    h = np.zeros((n, n, p), order='F', dtype=float)
    h[:, :, 0] = h1_orig.copy()
    h[:, :, 1] = h2_orig.copy()

    h_result, z, wr, wi, info = mb03wd('S', 'I', n, p, ilo, ihi, iloz, ihiz, h)

    assert info == 0

    z1 = z[:, :, 0]
    z2 = z[:, :, 1]
    t1 = h_result[:, :, 0]
    t2 = h_result[:, :, 1]

    computed_t1 = z1.T @ h1_orig @ z2
    np.testing.assert_allclose(computed_t1, t1, rtol=1e-4, atol=1e-4)

    computed_t2 = z2.T @ h2_orig @ z1
    np.testing.assert_allclose(computed_t2, t2, rtol=1e-4, atol=1e-4)


"""Edge case tests."""

def test_single_element():
    """Test with n=1 (single element matrices)."""
    n = 1
    p = 2
    ilo = 1
    ihi = 1
    iloz = 1
    ihiz = 1

    h = np.zeros((n, n, p), order='F', dtype=float)
    h[0, 0, 0] = 2.0
    h[0, 0, 1] = 3.0

    h_result, z, wr, wi, info = mb03wd('S', 'I', n, p, ilo, ihi, iloz, ihiz, h)

    assert info == 0
    np.testing.assert_allclose(wr[0], 6.0, rtol=1e-14)
    np.testing.assert_allclose(wi[0], 0.0, atol=1e-14)

def test_single_matrix_p1():
    """Test with p=1 (single matrix, reduces to standard Schur)."""
    n = 3
    p = 1
    ilo = 1
    ihi = n
    iloz = 1
    ihiz = n

    h = np.zeros((n, n, p), order='F', dtype=float)
    h[:, :, 0] = np.array([
        [4.0, 1.0, 2.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0]
    ], order='F', dtype=float)

    h_result, z, wr, wi, info = mb03wd('S', 'I', n, p, ilo, ihi, iloz, ihiz, h)

    assert info == 0

    expected_eig = np.linalg.eigvals(h[:, :, 0])
    expected_eig_real = np.sort(np.real(expected_eig))[::-1]

    computed_real = []
    i = 0
    while i < n:
        if wi[i] == 0.0:
            computed_real.append(wr[i])
            i += 1
        else:
            computed_real.append(wr[i])
            computed_real.append(wr[i+1])
            i += 2
    computed_real_sorted = np.sort(computed_real)[::-1]

    np.testing.assert_allclose(computed_real_sorted, expected_eig_real, rtol=1e-10)

def test_compz_v_accumulate():
    """
    Test COMPZ='V' mode which accumulates transformations.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 3
    p = 2
    ilo = 1
    ihi = n
    iloz = 1
    ihiz = n

    h = np.zeros((n, n, p), order='F', dtype=float)
    h[:, :, 0] = np.array([
        [2.0, 1.0, 0.5],
        [0.3, 1.5, 0.7],
        [0.0, 0.2, 1.0]
    ], order='F', dtype=float)
    h[:, :, 1] = np.triu(np.array([
        [1.0, 0.4, 0.2],
        [0.0, 0.8, 0.3],
        [0.0, 0.0, 0.5]
    ], order='F', dtype=float))

    z = np.zeros((n, n, p), order='F', dtype=float)
    for k in range(p):
        z[:, :, k] = np.eye(n, dtype=float, order='F')

    h_result, z_result, wr, wi, info = mb03wd('S', 'V', n, p, ilo, ihi, iloz, ihiz, h, z)

    assert info == 0

    for k in range(p):
        orthogonality = z_result[:, :, k].T @ z_result[:, :, k]
        np.testing.assert_allclose(orthogonality, np.eye(n), rtol=1e-12, atol=1e-12)


"""Error handling tests."""

def test_invalid_n():
    """Test with invalid n < 0."""
    n = -1
    p = 2
    ilo = 1
    ihi = 1
    iloz = 1
    ihiz = 1

    h = np.zeros((1, 1, p), order='F', dtype=float)

    with pytest.raises(ValueError):
        mb03wd('S', 'I', n, p, ilo, ihi, iloz, ihiz, h)

def test_invalid_p():
    """Test with invalid p < 1."""
    n = 3
    p = 0
    ilo = 1
    ihi = n
    iloz = 1
    ihiz = n

    h = np.zeros((n, n, 1), order='F', dtype=float)

    with pytest.raises(ValueError):
        mb03wd('S', 'I', n, p, ilo, ihi, iloz, ihiz, h)

def test_invalid_ilo_ihi():
    """Test with invalid ILO > IHI."""
    n = 4
    p = 2
    ilo = 3
    ihi = 2
    iloz = 1
    ihiz = n

    h = np.zeros((n, n, p), order='F', dtype=float)

    with pytest.raises(ValueError):
        mb03wd('S', 'I', n, p, ilo, ihi, iloz, ihiz, h)
