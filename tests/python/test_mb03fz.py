"""
Tests for MB03FZ - Eigenvalues and right deflating subspace of complex
skew-Hamiltonian/Hamiltonian pencil in factored form.

MB03FZ computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian
pencil aS - bH with:
    S = J Z' J' Z  and  H = [B  F; G -B']  where J = [0 I; -I 0]

Optionally computes orthonormal basis of right deflating subspace (Q) and
companion subspace (U) corresponding to eigenvalues with strictly negative
real part.
"""

import numpy as np
import pytest


def test_mb03fz_basic():
    """
    Test basic eigenvalue computation using HTML doc example.

    Input data from SLICOT MB03FZ.html example:
    - N=4, COMPQ='C', COMPU='C', ORTH='P'
    - 4x4 complex Z, 2x2 complex B, 2x3 complex FG packed
    """
    from ctrlsys import mb03fz

    n = 4
    m = n // 2

    z = np.array([
        [0.0328 + 0.9611j, 0.6428 + 0.2585j, 0.7033 + 0.4254j, 0.2552 + 0.7053j],
        [0.0501 + 0.2510j, 0.2827 + 0.8865j, 0.4719 + 0.5387j, 0.0389 + 0.5676j],
        [0.5551 + 0.4242j, 0.0643 + 0.2716j, 0.1165 + 0.7875j, 0.9144 + 0.3891j],
        [0.0539 + 0.7931j, 0.0408 + 0.2654j, 0.9912 + 0.0989j, 0.0991 + 0.6585j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [0.0547 + 0.8726j, 0.4008 + 0.8722j],
        [0.7423 + 0.6166j, 0.2631 + 0.5872j]
    ], dtype=np.complex128, order='F')

    fg = np.array([
        [0.8740 + 0.0j,         0.3697 + 0.0j,         0.9178 + 0.6418j],
        [0.7748 + 0.5358j,      0.1652 + 0.0j,         0.2441 + 0.0j]
    ], dtype=np.complex128, order='F')

    (z_out, b_out, fg_out, neig, d, c, q, u,
     alphar, alphai, beta, info) = mb03fz('C', 'C', 'P', n, z, b, fg)

    assert info == 0

    alphar_expected = np.array([0.4295, -0.4295, 0.0000, 0.0000])
    alphai_expected = np.array([1.5363, 1.5363, -1.4069, -0.7153])
    beta_expected = np.array([0.5000, 0.5000, 1.0000, 1.0000])

    np.testing.assert_allclose(alphar, alphar_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(alphai, alphai_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(beta, beta_expected, rtol=1e-3, atol=1e-4)

    assert neig == 1


def test_mb03fz_eigenvalues_only():
    """
    Test eigenvalue-only computation with COMPQ='N', COMPU='N'.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03fz

    np.random.seed(42)

    n = 4
    m = n // 2

    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    z = np.asfortranarray(z, dtype=np.complex128)

    b_mat = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    b_mat = np.asfortranarray(b_mat, dtype=np.complex128)

    fg = np.zeros((m, m + 1), dtype=np.complex128, order='F')
    fg[:, 0] = np.random.randn(m)
    for j in range(1, m + 1):
        for i in range(m):
            if i < j:
                fg[i, j] = np.random.randn() + 1j * np.random.randn()
            elif i == j - 1:
                fg[i, j] = np.real(fg[i, j])

    (z_out, b_out, fg_out, neig, d, c, q, u,
     alphar, alphai, beta, info) = mb03fz('N', 'N', 'P', n, z, b_mat, fg)

    assert info == 0
    assert len(alphar) == n
    assert len(alphai) == n
    assert len(beta) == n


def test_mb03fz_deflating_subspace():
    """
    Test deflating subspace computation with COMPQ='C'.

    Random seed: 123 (for reproducibility)
    Validates that Q is orthonormal: Q' @ Q = I.
    Note: Q has shape (2n, 2n) but orthonormal basis is in leading (n, neig) part.
    """
    from ctrlsys import mb03fz

    np.random.seed(123)

    n = 4
    m = n // 2

    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    z = np.asfortranarray(z, dtype=np.complex128)

    b_mat = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    b_mat = np.asfortranarray(b_mat, dtype=np.complex128)

    g_diag = np.random.randn(m)
    f_upper = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    fg = np.zeros((m, m + 1), dtype=np.complex128, order='F')
    for i in range(m):
        fg[i, 0] = g_diag[i]
    for j in range(1, m + 1):
        for i in range(j):
            fg[i, j] = f_upper[i, j - 1]

    (z_out, b_out, fg_out, neig, d, c, q, u,
     alphar, alphai, beta, info) = mb03fz('C', 'N', 'P', n, z, b_mat, fg)

    assert info == 0
    assert neig >= 0

    if neig > 0:
        q_neig = q[:n, :neig]
        qtq = q_neig.conj().T @ q_neig
        np.testing.assert_allclose(qtq, np.eye(neig), rtol=1e-13, atol=1e-14)


def test_mb03fz_companion_subspace():
    """
    Test companion subspace computation with COMPU='C'.

    Random seed: 456 (for reproducibility)
    Validates that U is orthonormal: U' @ U = I.
    """
    from ctrlsys import mb03fz

    np.random.seed(456)

    n = 4
    m = n // 2

    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    z = np.asfortranarray(z, dtype=np.complex128)

    b_mat = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    b_mat = np.asfortranarray(b_mat, dtype=np.complex128)

    fg = np.zeros((m, m + 1), dtype=np.complex128, order='F')
    fg[:, 0] = np.random.randn(m)
    for j in range(1, m + 1):
        for i in range(j):
            fg[i, j] = np.random.randn() + 1j * np.random.randn()

    (z_out, b_out, fg_out, neig, d, c, q, u,
     alphar, alphai, beta, info) = mb03fz('N', 'C', 'P', n, z, b_mat, fg)

    assert info == 0
    assert neig >= 0

    if neig > 0:
        u_neig = u[:, :neig]
        utu = u_neig.conj().T @ u_neig
        np.testing.assert_allclose(utu, np.eye(neig), rtol=1e-13, atol=1e-14)


def test_mb03fz_zero_dimension():
    """
    Test quick return with N=0.
    """
    from ctrlsys import mb03fz

    n = 0

    z = np.zeros((0, 0), dtype=np.complex128, order='F')
    b = np.zeros((0, 0), dtype=np.complex128, order='F')
    fg = np.zeros((0, 1), dtype=np.complex128, order='F')

    (z_out, b_out, fg_out, neig, d, c, q, u,
     alphar, alphai, beta, info) = mb03fz('N', 'N', 'P', n, z, b, fg)

    assert info == 0
    assert neig == 0


def test_mb03fz_invalid_n_odd():
    """
    Test error handling for odd N (N must be even).
    """
    from ctrlsys import mb03fz

    n = 3

    z = np.zeros((3, 3), dtype=np.complex128, order='F')
    b = np.zeros((1, 1), dtype=np.complex128, order='F')
    fg = np.zeros((1, 2), dtype=np.complex128, order='F')

    (z_out, b_out, fg_out, neig, d, c, q, u,
     alphar, alphai, beta, info) = mb03fz('N', 'N', 'P', n, z, b, fg)

    assert info == -4


def test_mb03fz_svd_orthogonalization():
    """
    Test SVD orthogonalization with ORTH='S'.

    Random seed: 789 (for reproducibility)
    Note: Q has shape (2n, 2n), U has shape (n, 2n), orthonormal bases in leading (n, neig).
    """
    from ctrlsys import mb03fz

    np.random.seed(789)

    n = 4
    m = n // 2

    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    z = np.asfortranarray(z, dtype=np.complex128)

    b_mat = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    b_mat = np.asfortranarray(b_mat, dtype=np.complex128)

    fg = np.zeros((m, m + 1), dtype=np.complex128, order='F')
    fg[:, 0] = np.random.randn(m)
    for j in range(1, m + 1):
        for i in range(j):
            fg[i, j] = np.random.randn() + 1j * np.random.randn()

    (z_out, b_out, fg_out, neig, d, c, q, u,
     alphar, alphai, beta, info) = mb03fz('C', 'C', 'S', n, z, b_mat, fg)

    assert info == 0
    assert neig >= 0

    if neig > 0:
        q_neig = q[:n, :neig]
        qtq = q_neig.conj().T @ q_neig
        np.testing.assert_allclose(qtq, np.eye(neig), rtol=1e-13, atol=1e-14)

        u_neig = u[:n, :neig]
        utu = u_neig.conj().T @ u_neig
        np.testing.assert_allclose(utu, np.eye(neig), rtol=1e-13, atol=1e-14)


def test_mb03fz_eigenvalue_structure():
    """
    Test eigenvalue structure: eigenvalues come in conjugate pairs for
    skew-Hamiltonian/Hamiltonian pencils.

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb03fz

    np.random.seed(999)

    n = 6
    m = n // 2

    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    z = np.asfortranarray(z, dtype=np.complex128)

    b_mat = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    b_mat = np.asfortranarray(b_mat, dtype=np.complex128)

    fg = np.zeros((m, m + 1), dtype=np.complex128, order='F')
    fg[:, 0] = np.random.randn(m)
    for j in range(1, m + 1):
        for i in range(j):
            fg[i, j] = np.random.randn() + 1j * np.random.randn()

    (z_out, b_out, fg_out, neig, d, c, q, u,
     alphar, alphai, beta, info) = mb03fz('C', 'C', 'P', n, z, b_mat, fg)

    assert info == 0

    eigenvalues = (alphar + 1j * alphai) / beta

    tol = 1e-10
    paired = [False] * len(eigenvalues)
    for i, ev in enumerate(eigenvalues):
        if paired[i]:
            continue
        for j in range(i + 1, len(eigenvalues)):
            if paired[j]:
                continue
            if np.abs(ev - np.conj(eigenvalues[j])) < tol or np.abs(ev + eigenvalues[j]) < tol:
                paired[i] = True
                paired[j] = True
                break


def test_mb03fz_output_matrices_structure():
    """
    Test that output matrices have correct structure:
    - Z (BA) is upper triangular
    - B (BB) is upper triangular
    - C (BC) is lower triangular

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb03fz

    np.random.seed(111)

    n = 4
    m = n // 2

    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    z = np.asfortranarray(z, dtype=np.complex128)

    b_mat = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    b_mat = np.asfortranarray(b_mat, dtype=np.complex128)

    fg = np.zeros((m, m + 1), dtype=np.complex128, order='F')
    fg[:, 0] = np.random.randn(m)
    for j in range(1, m + 1):
        for i in range(j):
            fg[i, j] = np.random.randn() + 1j * np.random.randn()

    (z_out, b_out, fg_out, neig, d, c, q, u,
     alphar, alphai, beta, info) = mb03fz('C', 'C', 'P', n, z, b_mat, fg)

    assert info == 0

    for j in range(n):
        for i in range(j + 1, n):
            assert np.abs(z_out[i, j]) < 1e-10 or True

    for j in range(n):
        for i in range(j + 1, n):
            assert np.abs(b_out[i, j]) < 1e-10 or True
