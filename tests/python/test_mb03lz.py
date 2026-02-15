"""
Tests for MB03LZ: Eigenvalues and right deflating subspace of complex
skew-Hamiltonian/Hamiltonian pencil.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestMB03LZ:
    """Test cases for mb03lz function."""

    def test_html_doc_example(self):
        """
        Test case from SLICOT HTML documentation.

        COMPQ = 'C', ORTH = 'P', N = 4
        Tests eigenvalue computation and deflating subspace.
        """
        from ctrlsys import mb03lz

        n = 4
        m = n // 2

        # Input matrices from MB03LZ.dat (read row-by-row per Fortran READ)
        # A is N/2 x N/2
        a = np.array([
            [0.0604 + 0.6568j, 0.5268 + 0.2919j],
            [0.3992 + 0.6279j, 0.4167 + 0.4316j]
        ], dtype=np.complex128, order='F')

        # DE is N/2 x (N/2+1): lower triangle = E (skew-Hermitian),
        # columns 2 to N/2+1 upper triangle = D (skew-Hermitian)
        de = np.array([
            [0.0 + 0.4896j, 0.0 + 0.9516j, 0.3724 + 0.0526j],
            [0.9840 + 0.3394j, 0.0 + 0.9203j, 0.0 + 0.7378j]
        ], dtype=np.complex128, order='F')

        # B is N/2 x N/2
        b = np.array([
            [0.2691 + 0.4177j, 0.5478 + 0.3014j],
            [0.4228 + 0.9830j, 0.9427 + 0.7010j]
        ], dtype=np.complex128, order='F')

        # FG is N/2 x (N/2+1): lower triangle = G (Hermitian),
        # columns 2 to N/2+1 upper triangle = F (Hermitian)
        fg = np.array([
            [0.6663 + 0.0j, 0.6981 + 0.0j, 0.1781 + 0.8818j],
            [0.5391 + 0.1711j, 0.6665 + 0.0j, 0.1280 + 0.0j]
        ], dtype=np.complex128, order='F')

        # Call routine with COMPQ='C', ORTH='P'
        result = mb03lz('C', 'P', n, a, de, b, fg)

        a_out, de_out, b_out, fg_out, q, alphar, alphai, beta, neig, info = result

        assert info == 0, f"Expected info=0, got {info}"

        # Expected eigenvalue components from HTML doc (used to compute ratios)
        alphar_expected = np.array([-1.5832, 1.5832, -0.0842, 0.0842])
        alphai_expected = np.array([0.5069, 0.5069, -0.1642, -0.1642])
        beta_expected = np.array([0.7430, 0.7430, 1.4085, 1.4085])

        # Validate eigenvalues via ratios (alpha/beta) - normalization may differ
        actual_eig = (alphar + 1j * alphai) / beta
        expected_eig = (alphar_expected + 1j * alphai_expected) / beta_expected
        assert_allclose(actual_eig, expected_eig, rtol=1e-3, atol=1e-4)

        # Expected number of eigenvalues with negative real part
        assert neig == 2, f"Expected neig=2, got {neig}"

        # Validate deflating subspace orthonormality
        assert q.shape[0] == n
        assert q.shape[1] >= neig
        q_sub = q[:, :neig]
        qtq = np.conj(q_sub.T) @ q_sub
        assert_allclose(qtq, np.eye(neig), rtol=1e-10, atol=1e-12)

    def test_eigenvalues_only(self):
        """
        Test with COMPQ='N' - eigenvalues only, no deflating subspace.
        """
        from ctrlsys import mb03lz

        n = 4
        m = n // 2

        # Same input as HTML example
        a = np.array([
            [0.0604 + 0.6568j, 0.5268 + 0.2919j],
            [0.3992 + 0.6279j, 0.4167 + 0.4316j]
        ], dtype=np.complex128, order='F')

        de = np.array([
            [0.0 + 0.4896j, 0.0 + 0.9516j, 0.3724 + 0.0526j],
            [0.9840 + 0.3394j, 0.0 + 0.9203j, 0.0 + 0.7378j]
        ], dtype=np.complex128, order='F')

        b = np.array([
            [0.2691 + 0.4177j, 0.5478 + 0.3014j],
            [0.4228 + 0.9830j, 0.9427 + 0.7010j]
        ], dtype=np.complex128, order='F')

        fg = np.array([
            [0.6663 + 0.0j, 0.6981 + 0.0j, 0.1781 + 0.8818j],
            [0.5391 + 0.1711j, 0.6665 + 0.0j, 0.1280 + 0.0j]
        ], dtype=np.complex128, order='F')

        # Call routine with COMPQ='N' (eigenvalues only)
        result = mb03lz('N', 'P', n, a, de, b, fg)

        a_out, de_out, b_out, fg_out, q, alphar, alphai, beta, neig, info = result

        assert info == 0, f"Expected info=0, got {info}"

        # Expected eigenvalue components from HTML doc (used to compute ratios)
        alphar_expected = np.array([-1.5832, 1.5832, -0.0842, 0.0842])
        alphai_expected = np.array([0.5069, 0.5069, -0.1642, -0.1642])
        beta_expected = np.array([0.7430, 0.7430, 1.4085, 1.4085])

        # Validate eigenvalues via ratios (alpha/beta) - normalization may differ
        actual_eig = (alphar + 1j * alphai) / beta
        expected_eig = (alphar_expected + 1j * alphai_expected) / beta_expected
        assert_allclose(actual_eig, expected_eig, rtol=1e-3, atol=1e-4)

    def test_svd_orthogonalization(self):
        """
        Test with ORTH='S' (SVD-based orthogonalization).
        """
        from ctrlsys import mb03lz

        n = 4
        m = n // 2

        a = np.array([
            [0.0604 + 0.6568j, 0.5268 + 0.2919j],
            [0.3992 + 0.6279j, 0.4167 + 0.4316j]
        ], dtype=np.complex128, order='F')

        de = np.array([
            [0.0 + 0.4896j, 0.0 + 0.9516j, 0.3724 + 0.0526j],
            [0.9840 + 0.3394j, 0.0 + 0.9203j, 0.0 + 0.7378j]
        ], dtype=np.complex128, order='F')

        b = np.array([
            [0.2691 + 0.4177j, 0.5478 + 0.3014j],
            [0.4228 + 0.9830j, 0.9427 + 0.7010j]
        ], dtype=np.complex128, order='F')

        fg = np.array([
            [0.6663 + 0.0j, 0.6981 + 0.0j, 0.1781 + 0.8818j],
            [0.5391 + 0.1711j, 0.6665 + 0.0j, 0.1280 + 0.0j]
        ], dtype=np.complex128, order='F')

        result = mb03lz('C', 'S', n, a, de, b, fg)

        a_out, de_out, b_out, fg_out, q, alphar, alphai, beta, neig, info = result

        assert info == 0, f"Expected info=0, got {info}"

        # Expected eigenvalue components from HTML doc (used to compute ratios)
        alphar_expected = np.array([-1.5832, 1.5832, -0.0842, 0.0842])
        alphai_expected = np.array([0.5069, 0.5069, -0.1642, -0.1642])
        beta_expected = np.array([0.7430, 0.7430, 1.4085, 1.4085])

        # Validate eigenvalues via ratios (alpha/beta) - normalization may differ
        actual_eig = (alphar + 1j * alphai) / beta
        expected_eig = (alphar_expected + 1j * alphai_expected) / beta_expected
        assert_allclose(actual_eig, expected_eig, rtol=1e-3, atol=1e-4)

        # Deflating subspace: orthonormal property
        if neig > 0:
            qtq = np.conj(q[:, :neig].T) @ q[:, :neig]
            assert_allclose(qtq, np.eye(neig), rtol=1e-10, atol=1e-14)

    def test_zero_dimension(self):
        """
        Test with N=0 (quick return case).
        """
        from ctrlsys import mb03lz

        n = 0
        a = np.zeros((0, 0), dtype=np.complex128, order='F')
        de = np.zeros((0, 1), dtype=np.complex128, order='F')
        b = np.zeros((0, 0), dtype=np.complex128, order='F')
        fg = np.zeros((0, 1), dtype=np.complex128, order='F')

        result = mb03lz('N', 'P', n, a, de, b, fg)

        a_out, de_out, b_out, fg_out, q, alphar, alphai, beta, neig, info = result

        assert info == 0, f"Expected info=0, got {info}"
        assert neig == 0

    def test_deflating_subspace_orthonormality(self):
        """
        Mathematical property test: Deflating subspace should be orthonormal.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb03lz

        np.random.seed(42)
        n = 8
        m = n // 2

        # Generate random complex matrices with appropriate structure
        a_in = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            np.complex128, order='F'
        )

        # DE: E in lower triangle (skew-Hermitian), D in upper of cols 2:m+1
        de_in = np.zeros((m, m + 1), dtype=np.complex128, order='F')
        e_temp = np.random.randn(m, m) + 1j * np.random.randn(m, m)
        e_temp = (e_temp - np.conj(e_temp.T)) / 2
        d_temp = np.random.randn(m, m) + 1j * np.random.randn(m, m)
        d_temp = (d_temp - np.conj(d_temp.T)) / 2
        for i in range(m):
            de_in[i, 0] = e_temp[i, i]
            for j in range(i):
                de_in[i, j] = e_temp[i, j]
        for j in range(m):
            for i in range(j + 1):
                de_in[i, j + 1] = d_temp[i, j]

        b_in = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            np.complex128, order='F'
        )

        # FG: G in lower triangle (Hermitian), F in upper of cols 2:m+1
        fg_in = np.zeros((m, m + 1), dtype=np.complex128, order='F')
        g_temp = np.random.randn(m, m) + 1j * np.random.randn(m, m)
        g_temp = (g_temp + np.conj(g_temp.T)) / 2
        f_temp = np.random.randn(m, m) + 1j * np.random.randn(m, m)
        f_temp = (f_temp + np.conj(f_temp.T)) / 2
        for i in range(m):
            fg_in[i, 0] = g_temp[i, i]
            for j in range(i):
                fg_in[i, j] = g_temp[i, j]
        for j in range(m):
            for i in range(j + 1):
                fg_in[i, j + 1] = f_temp[i, j]

        result = mb03lz('C', 'P', n, a_in, de_in, b_in, fg_in)

        a_out, de_out, b_out, fg_out, q, alphar, alphai, beta, neig, info = result

        assert info == 0 or info == 4, f"Unexpected info={info}"

        # Validate orthonormality of deflating subspace
        if neig > 0:
            q_neig = q[:, :neig]
            qtq = np.conj(q_neig.T) @ q_neig
            assert_allclose(qtq, np.eye(neig), rtol=1e-10, atol=1e-14)

    def test_eigenvalue_pairing(self):
        """
        Mathematical property: Eigenvalues come in pairs due to structure.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb03lz

        np.random.seed(123)
        n = 6
        m = n // 2

        a_in = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            np.complex128, order='F'
        )

        de_in = np.zeros((m, m + 1), dtype=np.complex128, order='F')
        e_temp = np.random.randn(m, m) + 1j * np.random.randn(m, m)
        e_temp = (e_temp - np.conj(e_temp.T)) / 2
        d_temp = np.random.randn(m, m) + 1j * np.random.randn(m, m)
        d_temp = (d_temp - np.conj(d_temp.T)) / 2
        for i in range(m):
            de_in[i, 0] = e_temp[i, i]
            for j in range(i):
                de_in[i, j] = e_temp[i, j]
        for j in range(m):
            for i in range(j + 1):
                de_in[i, j + 1] = d_temp[i, j]

        b_in = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            np.complex128, order='F'
        )

        fg_in = np.zeros((m, m + 1), dtype=np.complex128, order='F')
        g_temp = np.random.randn(m, m) + 1j * np.random.randn(m, m)
        g_temp = (g_temp + np.conj(g_temp.T)) / 2
        f_temp = np.random.randn(m, m) + 1j * np.random.randn(m, m)
        f_temp = (f_temp + np.conj(f_temp.T)) / 2
        for i in range(m):
            fg_in[i, 0] = g_temp[i, i]
            for j in range(i):
                fg_in[i, j] = g_temp[i, j]
        for j in range(m):
            for i in range(j + 1):
                fg_in[i, j + 1] = f_temp[i, j]

        result = mb03lz('N', 'P', n, a_in, de_in, b_in, fg_in)

        a_out, de_out, b_out, fg_out, q, alphar, alphai, beta, neig, info = result

        assert info == 0 or info == 4, f"Unexpected info={info}"

        # Compute eigenvalues
        eig_real = alphar / (beta + 1e-300)
        eig_imag = alphai / (beta + 1e-300)

        # Sort eigenvalues
        idx_neg = np.where(eig_real < 0)[0]
        idx_pos = np.where(eig_real > 0)[0]

        # Due to skew-Hamiltonian/Hamiltonian structure, eigenvalues should
        # come in pairs with opposite real parts
        neg_reals = np.sort(np.abs(eig_real[idx_neg]))
        pos_reals = np.sort(np.abs(eig_real[idx_pos]))
        if len(neg_reals) == len(pos_reals):
            assert_allclose(neg_reals, pos_reals, rtol=1e-8)

    def test_invalid_n_odd(self):
        """
        Test that odd N is rejected.
        """
        from ctrlsys import mb03lz

        n = 3  # Invalid: must be even
        m = 1
        a = np.zeros((m, m), dtype=np.complex128, order='F')
        de = np.zeros((m, m + 1), dtype=np.complex128, order='F')
        b = np.zeros((m, m), dtype=np.complex128, order='F')
        fg = np.zeros((m, m + 1), dtype=np.complex128, order='F')

        result = mb03lz('N', 'P', n, a, de, b, fg)

        a_out, de_out, b_out, fg_out, q, alphar, alphai, beta, neig, info = result

        assert info == -3, f"Expected info=-3 for odd N, got {info}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
