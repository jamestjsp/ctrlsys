"""
Tests for MB04AZ - Eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.

MB04AZ computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian
pencil aS - bH, with S = J*Z^H*J^T*Z and H = [[B,F],[G,-B^H]].
"""

import numpy as np
import pytest

from ctrlsys import mb04az


class TestMB04AZBasic:
    """Basic functionality tests using HTML documentation example."""

    def test_html_example_n4(self):
        """
        Validate using SLICOT HTML doc example with N=4.

        Test data from MB04AZ.html Program Data section.
        """
        n = 4
        m = n // 2

        z = np.array([
            [0.4941+0.8054j, 0.8909+0.8865j, 0.0305+0.9786j, 0.9047+0.0596j],
            [0.7790+0.5767j, 0.3341+0.0286j, 0.7440+0.7126j, 0.6098+0.6819j],
            [0.7150+0.1829j, 0.6987+0.4899j, 0.5000+0.5004j, 0.6176+0.0424j],
            [0.9037+0.2399j, 0.1978+0.1679j, 0.4799+0.4710j, 0.8594+0.0714j],
        ], order='F', dtype=complex)

        b_in = np.array([
            [0.5216+0.7224j, 0.8181+0.6596j],
            [0.0967+0.1498j, 0.8175+0.5185j],
        ], order='F', dtype=complex)

        fg_in = np.array([
            [0.9729+0.0j, 0.8003+0.0j, 0.4323+0.8313j],
            [0.6489+0.1331j, 0.4537+0.0j, 0.8253+0.0j],
        ], order='F', dtype=complex)

        result = mb04az('T', 'C', 'C', z, b_in, fg_in)

        z_out, b_out, fg_out, d, c, q, u, alphar, alphai, beta, info = result

        assert info == 0

        alphar_expected = np.array([0.0, 0.0, 0.0, 0.0])
        alphai_expected = np.array([-1.4991, -1.3690, 1.0985, 0.9993])
        beta_expected = np.array([0.1250, 0.5000, 0.5000, 2.0000])

        np.testing.assert_allclose(alphar, alphar_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(alphai, alphai_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(beta, beta_expected, rtol=1e-3, atol=1e-4)

        assert z_out.shape == (n, n)
        assert b_out.shape == (n, n)
        assert fg_out.shape == (n, n)
        assert d.shape == (n, n)
        assert c.shape == (n, n)
        assert q.shape == (2*n, 2*n)
        assert u.shape == (n, 2*n)


class TestMB04AZEigenvaluesOnly:
    """Test eigenvalue-only computation (JOB='E')."""

    def test_eigenvalues_only_mode(self):
        """
        Test JOB='E' mode returns eigenvalues without full Schur form.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4
        m = n // 2

        z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
            complex, order='F')

        b_in = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F')

        fg_in = np.zeros((m, m+1), dtype=complex, order='F')
        fg_in[:, 0] = np.random.randn(m)
        for i in range(m):
            fg_in[i, 0] += 1j * np.random.randn()
        for j in range(1, m+1):
            for i in range(j):
                fg_in[i, j] = np.random.randn() + 1j * np.random.randn()
        fg_in[0, 1] = np.real(fg_in[0, 1])

        result = mb04az('E', 'N', 'N', z, b_in, fg_in)

        alphar, alphai, beta, info = result

        assert info == 0
        assert len(alphar) == n
        assert len(alphai) == n
        assert len(beta) == n


class TestMB04AZEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with N=0 (quick return case)."""
        n = 0
        z = np.zeros((0, 0), dtype=complex, order='F')
        b_in = np.zeros((0, 0), dtype=complex, order='F')
        fg_in = np.zeros((0, 1), dtype=complex, order='F')

        result = mb04az('E', 'N', 'N', z, b_in, fg_in)

        alphar, alphai, beta, info = result

        assert info == 0
        assert len(alphar) == 0
        assert len(alphai) == 0
        assert len(beta) == 0


class TestMB04AZTransformationProperties:
    """Test mathematical properties of the transformations."""

    def test_q_is_unitary(self):
        """
        Validate that Q is unitary (Q^H * Q = I).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4
        m = n // 2

        z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
            complex, order='F')

        b_in = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F')

        fg_in = np.zeros((m, m+1), dtype=complex, order='F')
        fg_in[0, 0] = 0.5
        fg_in[1, 0] = 0.3 + 0.1j
        fg_in[0, 1] = 0.4
        fg_in[1, 1] = 0.6
        fg_in[0, 2] = 0.2 + 0.3j
        fg_in[1, 2] = 0.7

        result = mb04az('T', 'C', 'N', z, b_in, fg_in)

        z_out, b_out, fg_out, d, c, q, alphar, alphai, beta, info = result

        assert info == 0

        qhq = q.conj().T @ q
        identity = np.eye(2*n)
        np.testing.assert_allclose(qhq, identity, rtol=1e-12, atol=1e-12)

    def test_u_is_unitary_symplectic(self):
        """
        Validate that U is unitary (first N rows of 2N-by-2N U satisfy U*U^H = I).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 4
        m = n // 2

        z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
            complex, order='F')

        b_in = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F')

        fg_in = np.zeros((m, m+1), dtype=complex, order='F')
        fg_in[0, 0] = 0.9
        fg_in[1, 0] = 0.5 + 0.2j
        fg_in[0, 1] = 0.3
        fg_in[1, 1] = 0.4
        fg_in[0, 2] = 0.1 + 0.2j
        fg_in[1, 2] = 0.6

        result = mb04az('T', 'N', 'C', z, b_in, fg_in)

        z_out, b_out, fg_out, d, c, u, alphar, alphai, beta, info = result

        assert info == 0
        assert u.shape == (n, 2*n)

        uuH = u @ u.conj().T
        identity = np.eye(n)
        np.testing.assert_allclose(uuH, identity, rtol=1e-12, atol=1e-12)


class TestMB04AZErrorHandling:
    """Error handling tests."""

    def test_invalid_job_parameter(self):
        """Test with invalid JOB parameter."""
        n = 4
        m = n // 2
        z = np.zeros((n, n), dtype=complex, order='F')
        b_in = np.zeros((m, m), dtype=complex, order='F')
        fg_in = np.zeros((m, m+1), dtype=complex, order='F')

        with pytest.raises((ValueError, RuntimeError)):
            mb04az('X', 'N', 'N', z, b_in, fg_in)

    def test_odd_n_parameter(self):
        """Test with odd N (should fail since N must be even)."""
        n = 3
        m = n // 2
        z = np.zeros((n, n), dtype=complex, order='F')
        b_in = np.zeros((m, m), dtype=complex, order='F')
        fg_in = np.zeros((m, m+1), dtype=complex, order='F')

        with pytest.raises((ValueError, RuntimeError)):
            mb04az('E', 'N', 'N', z, b_in, fg_in)
