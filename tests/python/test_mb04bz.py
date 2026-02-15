"""
Tests for MB04BZ - Eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.

MB04BZ computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian
pencil aS - bH, with S = [[A, D], [E, A^H]] and H = [[B, F], [G, -B^H]],
using an embedding to a real skew-Hamiltonian/skew-Hamiltonian pencil.
"""

import numpy as np
import pytest

from ctrlsys import mb04bz


class TestMB04BZBasic:
    """Basic functionality tests using HTML documentation example."""

    def test_html_example_n4(self):
        """
        Validate using SLICOT HTML doc example with N=4.

        Test data from MB04BZ.html Program Data section.
        Data is read row-by-row per Fortran READ loops in example program.
        """
        n = 4
        m = n // 2

        a = np.array([
            [0.0604+0.6568j, 0.5268+0.2919j],
            [0.3992+0.6279j, 0.4167+0.4316j],
        ], order='F', dtype=complex)

        de = np.array([
            [0+0.4896j, 0+0.9516j, 0.3724+0.0526j],
            [0.9840+0.3394j, 0+0.9203j, 0+0.7378j],
        ], order='F', dtype=complex)

        b_in = np.array([
            [0.2691+0.4177j, 0.5478+0.3014j],
            [0.4228+0.9830j, 0.9427+0.7010j],
        ], order='F', dtype=complex)

        fg = np.array([
            [0.6663+0j, 0.6981+0j, 0.1781+0.8818j],
            [0.5391+0.1711j, 0.6665+0j, 0.1280+0j],
        ], order='F', dtype=complex)

        result = mb04bz('T', 'C', a, de, b_in, fg)

        a_out, de_out, b_out, fg_out, q, alphar, alphai, beta, info = result

        assert info == 0

        alphar_expected = np.array([-1.5832, 1.5832, -0.0842, 0.0842])
        alphai_expected = np.array([0.5069, 0.5069, -0.1642, -0.1642])
        beta_expected = np.array([0.7430, 0.7430, 1.4085, 1.4085])

        eigenvalues = (alphar + 1j * alphai) / beta
        eigenvalues_expected = (alphar_expected + 1j * alphai_expected) / beta_expected

        np.testing.assert_allclose(np.sort(np.real(eigenvalues)), np.sort(np.real(eigenvalues_expected)), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(np.sort(np.imag(eigenvalues)), np.sort(np.imag(eigenvalues_expected)), rtol=1e-3, atol=1e-4)

        assert a_out.shape == (n, n)
        assert de_out.shape == (n, n)
        assert b_out.shape == (n, n)
        assert fg_out.shape == (n, n)
        assert q.shape == (2*n, 2*n)


class TestMB04BZEigenvaluesOnly:
    """Test eigenvalue-only computation (JOB='E')."""

    def test_eigenvalues_only_mode(self):
        """
        Test JOB='E' mode returns eigenvalues without full Schur form.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4
        m = n // 2

        a = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F')

        de = np.zeros((m, m+1), dtype=complex, order='F')
        for j in range(m):
            de[j, j] = 1j * np.random.randn()
        for j in range(1, m+1):
            for i in range(j, m):
                de[i, j-1] = np.random.randn() + 1j * np.random.randn()
            for i in range(j-1):
                de[i, j] = np.random.randn() + 1j * np.random.randn()

        b_in = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F')

        fg = np.zeros((m, m+1), dtype=complex, order='F')
        for j in range(m):
            fg[j, j] = np.random.randn()
        for j in range(1, m+1):
            for i in range(j, m):
                fg[i, j-1] = np.random.randn() + 1j * np.random.randn()
            for i in range(j-1):
                fg[i, j] = np.random.randn() + 1j * np.random.randn()

        result = mb04bz('E', 'N', a, de, b_in, fg)

        alphar, alphai, beta, info = result

        assert info == 0
        assert len(alphar) == n
        assert len(alphai) == n
        assert len(beta) == n


class TestMB04BZEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with N=0 (quick return case)."""
        n = 0
        a = np.zeros((0, 0), dtype=complex, order='F')
        de = np.zeros((0, 1), dtype=complex, order='F')
        b_in = np.zeros((0, 0), dtype=complex, order='F')
        fg = np.zeros((0, 1), dtype=complex, order='F')

        result = mb04bz('E', 'N', a, de, b_in, fg)

        alphar, alphai, beta, info = result

        assert info == 0
        assert len(alphar) == 0
        assert len(alphai) == 0
        assert len(beta) == 0


class TestMB04BZTransformationProperties:
    """Test mathematical properties of the transformations."""

    def test_q_is_unitary(self):
        """
        Validate that Q is unitary (Q^H * Q = I).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4
        m = n // 2

        a = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F')

        de = np.zeros((m, m+1), dtype=complex, order='F')
        for j in range(m):
            de[j, j] = 1j * np.random.randn()
        for j in range(1, m+1):
            for i in range(j, m):
                de[i, j-1] = np.random.randn() + 1j * np.random.randn()
            for i in range(j-1):
                de[i, j] = np.random.randn() + 1j * np.random.randn()

        b_in = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F')

        fg = np.zeros((m, m+1), dtype=complex, order='F')
        for j in range(m):
            fg[j, j] = np.random.randn()
        for j in range(1, m+1):
            for i in range(j, m):
                fg[i, j-1] = np.random.randn() + 1j * np.random.randn()
            for i in range(j-1):
                fg[i, j] = np.random.randn() + 1j * np.random.randn()

        result = mb04bz('T', 'C', a, de, b_in, fg)

        a_out, de_out, b_out, fg_out, q, alphar, alphai, beta, info = result

        assert info == 0

        qhq = q.conj().T @ q
        identity = np.eye(2*n)
        np.testing.assert_allclose(qhq, identity, rtol=1e-12, atol=1e-12)


class TestMB04BZErrorHandling:
    """Error handling tests."""

    def test_invalid_job_parameter(self):
        """Test with invalid JOB parameter."""
        n = 4
        m = n // 2
        a = np.zeros((m, m), dtype=complex, order='F')
        de = np.zeros((m, m+1), dtype=complex, order='F')
        b_in = np.zeros((m, m), dtype=complex, order='F')
        fg = np.zeros((m, m+1), dtype=complex, order='F')

        with pytest.raises((ValueError, RuntimeError)):
            mb04bz('X', 'N', a, de, b_in, fg)

