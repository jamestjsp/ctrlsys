"""
Tests for MB3JZP - Reorder eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.

Moves eigenvalues with strictly negative real parts to the leading subpencil
while keeping triangular form. Panel version with blocked updates for large N.
"""

import numpy as np
import pytest

from ctrlsys import mb3jzp


class TestMB3JZPBasic:
    """Basic functionality tests for MB3JZP."""

    def test_basic_reordering(self):
        """
        Test basic functionality with a simple pencil.

        The full N×N pencil has structure that determines the eigenvalue count.
        We verify the routine maintains triangular structure and unitary Q.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m = 2  # n = 2*m = 4
        n = 2 * m

        # Upper triangular A and B
        a = np.array([[1.0 + 0.0j, 0.3 - 0.2j],
                      [0.0, 2.0 + 0.0j]], dtype=complex, order='F')
        b = np.array([[1.0 + 0.0j, 0.2 + 0.1j],
                      [0.0, 3.0 + 0.0j]], dtype=complex, order='F')

        # D is skew-Hermitian (D = -D^H)
        d = np.array([[0.5j, 0.1 + 0.2j],
                      [0.0, -0.3j]], dtype=complex, order='F')

        # F is Hermitian (F = F^H)
        f = np.array([[1.0, 0.2 - 0.1j],
                      [0.0, 0.8]], dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'I', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        assert info == 0
        assert 0 <= neig <= m  # neig in valid range
        # A and B remain upper triangular
        np.testing.assert_allclose(np.tril(a_out, -1), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.tril(b_out, -1), 0.0, atol=1e-14)
        # Q should be unitary
        np.testing.assert_allclose(q_out @ q_out.conj().T, np.eye(n, dtype=complex),
                                   rtol=1e-14, atol=1e-14)

    def test_mixed_eigenvalue_signs(self):
        """
        Test with a pencil having mixed eigenvalue signs in subpencil.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m = 2
        n = 2 * m

        # First eigenvalue positive in subpencil, second negative
        a = np.array([[1.0 + 0.0j, 0.5 - 0.3j],
                      [0.0, 1.0 + 0.0j]], dtype=complex, order='F')
        b = np.array([[1.0 + 0.0j, 0.2 + 0.1j],
                      [0.0, -1.0 + 0.0j]], dtype=complex, order='F')

        d = np.array([[0.2j, 0.1 + 0.05j],
                      [0.0, -0.1j]], dtype=complex, order='F')

        f = np.array([[0.5, 0.1 - 0.05j],
                      [0.0, 0.3]], dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'I', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        assert info == 0
        assert 0 <= neig <= m  # neig in valid range
        # A and B remain upper triangular
        np.testing.assert_allclose(np.tril(a_out, -1), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.tril(b_out, -1), 0.0, atol=1e-14)
        # Q should be unitary
        np.testing.assert_allclose(q_out @ q_out.conj().T, np.eye(n, dtype=complex),
                                   rtol=1e-14, atol=1e-14)

    def test_all_negative_eigenvalues(self):
        """
        Test with all eigenvalues having negative real parts.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m = 2
        n = 2 * m

        # Create pencil where both eigenvalues have negative real parts
        a = np.array([[1.0 + 1.0j, 0.3 - 0.2j],
                      [0.0, 0.5 + 0.5j]], dtype=complex, order='F')
        b = np.array([[-1.0 + 0.5j, 0.2 + 0.1j],
                      [0.0, -0.8 + 0.3j]], dtype=complex, order='F')

        d = np.array([[0.3j, 0.15 + 0.1j],
                      [0.0, -0.2j]], dtype=complex, order='F')

        f = np.array([[0.6, 0.2 - 0.1j],
                      [0.0, 0.4]], dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'I', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        assert info == 0
        assert neig == 2  # Both eigenvalues have negative real parts
        # A and B remain upper triangular
        np.testing.assert_allclose(np.tril(a_out, -1), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.tril(b_out, -1), 0.0, atol=1e-14)
        # Q should be unitary
        np.testing.assert_allclose(q_out @ q_out.conj().T, np.eye(n, dtype=complex),
                                   rtol=1e-14, atol=1e-14)


class TestMB3JZPMathematicalProperties:
    """Mathematical property validation tests."""

    def test_unitary_transformation_preserves_structure(self):
        """
        Validate that Q is unitary: Q @ Q^H = I.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m = 3
        n = 2 * m

        # Create random upper triangular matrices
        a = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        b = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')

        # D skew-Hermitian: upper triangular with purely imaginary diagonal
        d = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        np.fill_diagonal(d, 1j * np.random.randn(m))

        # F Hermitian: upper triangular with real diagonal
        f = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        np.fill_diagonal(f, np.random.randn(m))

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'I', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        assert info == 0
        # Q should be unitary
        np.testing.assert_allclose(q_out @ q_out.conj().T, np.eye(n, dtype=complex),
                                   rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(q_out.conj().T @ q_out, np.eye(n, dtype=complex),
                                   rtol=1e-13, atol=1e-14)

    def test_output_triangular_structure(self):
        """
        Validate that output matrices maintain triangular structure.

        Random seed: 321 (for reproducibility)
        """
        np.random.seed(321)
        m = 4
        n = 2 * m

        a = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        b = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        d = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        f = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'I', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        assert info == 0
        # A and B remain upper triangular
        np.testing.assert_allclose(np.tril(a_out, -1), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.tril(b_out, -1), 0.0, atol=1e-14)

    def test_eigenvalue_count_consistency(self):
        """
        Validate that neig correctly counts eigenvalues with negative real part.

        Random seed: 654 (for reproducibility)
        """
        np.random.seed(654)
        m = 3
        n = 2 * m

        # Create diagonal matrices to easily verify eigenvalues
        # Eigenvalue sign determined by Re(A[k,k] * conj(B[k,k]))
        a_diag = np.array([1.0 + 0.5j, 1.0 + 0.5j, 1.0 + 0.5j], dtype=complex)
        # First two have positive eigenvalue, third has negative
        b_diag = np.array([1.0 + 0.3j, 1.0 - 0.3j, -1.0 + 0.2j], dtype=complex)

        a = np.diag(a_diag).astype(complex, order='F')
        b = np.diag(b_diag).astype(complex, order='F')
        d = np.zeros((m, m), dtype=complex, order='F')
        np.fill_diagonal(d, 1j * np.random.randn(m))
        f = np.zeros((m, m), dtype=complex, order='F')
        np.fill_diagonal(f, np.random.randn(m))

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'I', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        assert info == 0
        # Count expected negative eigenvalues based on Re(a[k]*conj(b[k]))
        expected_neg = sum(1 for ak, bk in zip(a_diag, b_diag)
                          if (ak * bk.conjugate()).real < 0)
        assert neig == expected_neg

    def test_matches_mb03jz_for_small_n(self):
        """
        Verify MB3JZP matches MB03JZ results for small matrices.

        MB3JZP is a panel version of MB03JZ. They should produce equivalent results.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        from ctrlsys import mb03jz

        m = 3
        n = 2 * m

        a = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        b = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        d = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        np.fill_diagonal(d, 1j * np.random.randn(m))
        f = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        np.fill_diagonal(f, np.random.randn(m))

        # Run MB03JZ (non-panel version)
        a_jz, d_jz, b_jz, f_jz, q_jz, neig_jz, info_jz = mb03jz(
            'I', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        # Run MB3JZP (panel version)
        a_jzp, d_jzp, b_jzp, f_jzp, q_jzp, neig_jzp, info_jzp = mb3jzp(
            'I', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        assert info_jz == 0
        assert info_jzp == 0
        # Both should find the same number of negative eigenvalues
        assert neig_jz == neig_jzp

        # Both Q should be unitary
        np.testing.assert_allclose(q_jz @ q_jz.conj().T, np.eye(n, dtype=complex),
                                   rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(q_jzp @ q_jzp.conj().T, np.eye(n, dtype=complex),
                                   rtol=1e-13, atol=1e-14)


class TestMB3JZPEdgeCases:
    """Edge case tests."""

    def test_zero_dimension(self):
        """Test with n=0 (quick return)."""
        n = 0
        m = n // 2
        a = np.empty((m, m), dtype=complex, order='F')
        d = np.empty((m, m), dtype=complex, order='F')
        b = np.empty((m, m), dtype=complex, order='F')
        f = np.empty((m, m), dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'N', n, a, d, b, f
        )

        assert info == 0
        assert neig == 0

    def test_n_equals_2(self):
        """
        Test minimal case with n=2 (m=1, single eigenvalue).

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        m = 1
        n = 2

        a = np.array([[1.5 + 0.5j]], dtype=complex, order='F')
        b = np.array([[-1.0 + 0.2j]], dtype=complex, order='F')
        d = np.array([[0.3j]], dtype=complex, order='F')
        f = np.array([[0.5]], dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'I', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        assert info == 0
        # Check sign of eigenvalue: Re(a[0,0] * conj(b[0,0]))
        eig_sign = (a[0, 0] * b[0, 0].conjugate()).real
        expected_neig = 1 if eig_sign < 0 else 0
        assert neig == expected_neig


class TestMB3JZPCompqModes:
    """Tests for different COMPQ modes."""

    def test_compq_n_no_q_computed(self):
        """Test COMPQ='N' - Q not computed."""
        m = 2
        n = 2 * m

        a = np.array([[1.0 + 0.5j, 0.3 - 0.2j],
                      [0.0, 2.0 - 0.1j]], dtype=complex, order='F')
        b = np.array([[1.0 + 0.1j, 0.2 + 0.1j],
                      [0.0, 1.5 + 0.2j]], dtype=complex, order='F')
        d = np.array([[0.5j, 0.1 + 0.2j],
                      [0.0, -0.3j]], dtype=complex, order='F')
        f = np.array([[1.0, 0.2 - 0.1j],
                      [0.0, 0.8]], dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'N', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F')
        )

        assert info == 0

    def test_compq_u_update_q(self):
        """
        Test COMPQ='U' - update existing Q matrix.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        m = 2
        n = 2 * m

        a = np.array([[1.0 + 1.0j, 0.5 - 0.3j],
                      [0.0, 1.0 + 1.0j]], dtype=complex, order='F')
        b = np.array([[1.0 + 0.5j, 0.2 + 0.1j],
                      [0.0, -1.0 + 0.5j]], dtype=complex, order='F')
        d = np.array([[0.2j, 0.1 + 0.05j],
                      [0.0, -0.1j]], dtype=complex, order='F')
        f = np.array([[0.5, 0.1 - 0.05j],
                      [0.0, 0.3]], dtype=complex, order='F')

        # Start with identity matrix for Q0
        q0 = np.eye(n, dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'U', n, a.copy(order='F'), d.copy(order='F'),
            b.copy(order='F'), f.copy(order='F'), q=q0.copy(order='F')
        )

        assert info == 0
        # Result should be unitary since Q0 was identity
        np.testing.assert_allclose(q_out @ q_out.conj().T, np.eye(n, dtype=complex),
                                   rtol=1e-13, atol=1e-14)


class TestMB3JZPErrorHandling:
    """Error handling tests."""

    def test_invalid_compq(self):
        """Test invalid COMPQ parameter."""
        m = 2
        n = 2 * m
        a = np.zeros((m, m), dtype=complex, order='F')
        d = np.zeros((m, m), dtype=complex, order='F')
        b = np.zeros((m, m), dtype=complex, order='F')
        f = np.zeros((m, m), dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'X', n, a, d, b, f
        )

        assert info == -1

    def test_invalid_n_negative(self):
        """Test negative N."""
        m = 2
        a = np.zeros((m, m), dtype=complex, order='F')
        d = np.zeros((m, m), dtype=complex, order='F')
        b = np.zeros((m, m), dtype=complex, order='F')
        f = np.zeros((m, m), dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'N', -2, a, d, b, f
        )

        assert info == -2

    def test_invalid_n_odd(self):
        """Test odd N (must be even)."""
        m = 2
        a = np.zeros((m, m), dtype=complex, order='F')
        d = np.zeros((m, m), dtype=complex, order='F')
        b = np.zeros((m, m), dtype=complex, order='F')
        f = np.zeros((m, m), dtype=complex, order='F')

        a_out, d_out, b_out, f_out, q_out, neig, info = mb3jzp(
            'N', 3, a, d, b, f
        )

        assert info == -2
