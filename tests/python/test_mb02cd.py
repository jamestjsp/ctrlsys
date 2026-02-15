"""
Tests for MB02CD - Cholesky factorization of positive definite block Toeplitz matrix.

MB02CD computes the Cholesky factor and/or the generator of the inverse of a
symmetric positive definite block Toeplitz matrix T, defined by its first
block row or column.
"""

import numpy as np
import pytest
from ctrlsys import mb02cd


class TestMB02CDBasic:
    """Basic functionality tests using HTML documentation example."""

    def test_html_doc_example_job_a(self):
        """
        Test JOB='A' (all outputs) with HTML documentation example.

        Input: First block row of 6x6 s.p.d. block Toeplitz matrix
        K=2 (block size), N=3 (number of blocks)
        TYPET='R' (row-oriented)

        Expected outputs from HTML doc (4-decimal precision).
        """
        k = 2
        n = 3
        m = n * k

        # First block row of the block Toeplitz matrix (K x N*K)
        # Read row-wise: ((T(I,J), J=1,M), I=1,K)
        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('A', 'R', k, n, t)

        assert info == 0

        # Expected generator from HTML doc (after setting G(K+1:2*K, 1:K) = 0)
        g_expected = np.array([
            [-0.2355,  0.5231, -0.0642,  0.0077,  0.0187, -0.0265],
            [-0.5568, -0.0568,  0.0229,  0.0060,  0.0363,  0.0000],
            [ 0.0000,  0.0000, -0.0387,  0.0052,  0.0003, -0.0575],
            [ 0.0000,  0.0000,  0.0119, -0.0265, -0.0110,  0.0076]
        ], dtype=float, order='F')

        # Expected lower Cholesky factor of inverse
        l_expected = np.array([
            [ 0.5774,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [-0.1741,  0.5222,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0000, -0.0581,  0.5812,  0.0000,  0.0000,  0.0000],
            [-0.0142,  0.0080, -0.1747,  0.5224,  0.0000,  0.0000],
            [-0.0387,  0.0052,  0.0003, -0.0575,  0.5825,  0.0000],
            [ 0.0119, -0.0265, -0.0110,  0.0076, -0.1754,  0.5231]
        ], dtype=float, order='F')

        # Expected upper Cholesky factor of T
        r_expected = np.array([
            [1.7321, 0.5774, 0.0577, 0.0577, 0.1155, 0.0289],
            [0.0000, 1.9149, 0.1915, 0.0348, -0.0139, 0.0957],
            [0.0000, 0.0000, 1.7205, 0.5754, 0.0558, 0.0465],
            [0.0000, 0.0000, 0.0000, 1.9142, 0.1890, 0.0357],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.7169, 0.5759],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.9118]
        ], dtype=float, order='F')

        # Validate shapes
        assert g.shape == (2 * k, m)
        assert r.shape == (m, m)
        assert l.shape == (m, m)
        assert cs.shape[0] == 3 * (n - 1) * k

        # Validate numerical values (HTML doc precision ~4 decimals)
        # Set G(K+1:2*K, 1:K) = 0 to get actual generator
        g_copy = g.copy()
        g_copy[k:2*k, 0:k] = 0.0
        np.testing.assert_allclose(g_copy, g_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(l, l_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(r, r_expected, rtol=1e-3, atol=1e-4)

    def test_job_g_generator_only(self):
        """Test JOB='G' - compute only generator."""
        k = 2
        n = 3

        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('G', 'R', k, n, t)

        assert info == 0
        assert g.shape == (2 * k, n * k)

    def test_job_r_cholesky_of_t(self):
        """Test JOB='R' - compute generator and Cholesky factor of T."""
        k = 2
        n = 3

        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('R', 'R', k, n, t)

        assert info == 0
        assert g.shape == (2 * k, n * k)
        assert r.shape == (n * k, n * k)

    def test_job_l_cholesky_of_inverse(self):
        """Test JOB='L' - compute generator and Cholesky factor of inv(T)."""
        k = 2
        n = 3

        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('L', 'R', k, n, t)

        assert info == 0
        assert g.shape == (2 * k, n * k)
        assert l.shape == (n * k, n * k)

    def test_job_o_cholesky_only(self):
        """Test JOB='O' - compute only Cholesky factor of T."""
        k = 2
        n = 3

        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('O', 'R', k, n, t)

        assert info == 0
        assert r.shape == (n * k, n * k)


class TestMB02CDColumnOriented:
    """Tests for column-oriented (TYPET='C') algorithm."""

    def test_typet_c_basic(self):
        """
        Test TYPET='C' (column-oriented) with same data.

        For TYPET='C', T contains first block column (N*K x K).
        """
        k = 2
        n = 3
        m = n * k

        # First block column (N*K x K) - transpose of block row
        t = np.array([
            [3.0, 1.0],
            [1.0, 4.0],
            [0.1, 0.4],
            [0.1, 0.1],
            [0.2, 0.04],
            [0.05, 0.2]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('A', 'C', k, n, t)

        assert info == 0
        # For TYPET='C': G is N*K x 2*K, R is lower triangular, L is upper triangular
        assert g.shape == (m, 2 * k)
        assert r.shape == (m, m)
        assert l.shape == (m, m)


class TestMB02CDMathematicalProperties:
    """Tests validating mathematical correctness properties."""

    def test_cholesky_factorization_property(self):
        """
        Validate R'*R = T for TYPET='R'.

        Mathematical property: The upper Cholesky factor R satisfies R'*R = T.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k = 2
        n = 3
        m = n * k

        # Use the HTML example which is known to be s.p.d.
        t_row = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        # Construct the full symmetric block Toeplitz matrix T
        # For symmetric block Toeplitz defined by first row:
        # Upper triangle (i <= j): T[i*k:(i+1)*k, j*k:(j+1)*k] = T_{j-i}
        # Lower triangle (i > j): T[i*k:(i+1)*k, j*k:(j+1)*k] = T_{i-j}^T
        T_full = np.zeros((m, m), dtype=float, order='F')
        for i in range(n):
            for j in range(n):
                block_idx = abs(i - j)
                block = t_row[:, block_idx*k:(block_idx+1)*k]
                if i <= j:
                    T_full[i*k:(i+1)*k, j*k:(j+1)*k] = block
                else:
                    T_full[i*k:(i+1)*k, j*k:(j+1)*k] = block.T

        g, r, l, cs, info = mb02cd('R', 'R', k, n, t_row.copy())

        assert info == 0

        # Validate R'*R = T (upper triangular R)
        r_upper = np.triu(r)
        T_reconstructed = r_upper.T @ r_upper

        np.testing.assert_allclose(T_reconstructed, T_full, rtol=1e-10, atol=1e-12)

    def test_inverse_cholesky_property(self):
        """
        Validate L'*L = inv(T) for TYPET='R'.

        Mathematical property: The lower Cholesky factor L of inv(T) satisfies L'*L = inv(T).
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k = 2
        n = 3
        m = n * k

        t_row = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        # Construct the full symmetric block Toeplitz matrix T
        # Upper triangle (i <= j): T[i*k:(i+1)*k, j*k:(j+1)*k] = T_{j-i}
        # Lower triangle (i > j): T[i*k:(i+1)*k, j*k:(j+1)*k] = T_{i-j}^T
        T_full = np.zeros((m, m), dtype=float, order='F')
        for i in range(n):
            for j in range(n):
                block_idx = abs(i - j)
                block = t_row[:, block_idx*k:(block_idx+1)*k]
                if i <= j:
                    T_full[i*k:(i+1)*k, j*k:(j+1)*k] = block
                else:
                    T_full[i*k:(i+1)*k, j*k:(j+1)*k] = block.T

        g, r, l, cs, info = mb02cd('A', 'R', k, n, t_row.copy())

        assert info == 0

        # Compute inv(T)
        T_inv = np.linalg.inv(T_full)

        # Validate L'*L = inv(T) (L is lower triangular)
        l_lower = np.tril(l)
        T_inv_reconstructed = l_lower.T @ l_lower

        np.testing.assert_allclose(T_inv_reconstructed, T_inv, rtol=1e-10, atol=1e-12)

    def test_symmetry_preservation(self):
        """
        Validate that the reconstructed matrices are symmetric.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        k = 2
        n = 3
        m = n * k

        t_row = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('A', 'R', k, n, t_row.copy())

        assert info == 0

        # R'*R should be symmetric
        r_upper = np.triu(r)
        T_reconstructed = r_upper.T @ r_upper
        np.testing.assert_allclose(T_reconstructed, T_reconstructed.T, rtol=1e-14, atol=1e-15)

        # L'*L should be symmetric
        l_lower = np.tril(l)
        T_inv_reconstructed = l_lower.T @ l_lower
        np.testing.assert_allclose(T_inv_reconstructed, T_inv_reconstructed.T, rtol=1e-14, atol=1e-15)


class TestMB02CDEdgeCases:
    """Edge case tests."""

    def test_n_equals_1(self):
        """Test with single block (N=1)."""
        k = 2
        n = 1

        # First block is just the first KxK block
        t = np.array([
            [3.0, 1.0],
            [1.0, 4.0]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('A', 'R', k, n, t)

        assert info == 0
        assert r.shape == (k, k)
        assert l.shape == (k, k)

    def test_k_equals_1_scalar_blocks(self):
        """Test with scalar blocks (K=1)."""
        k = 1
        n = 3

        # First block row with scalar blocks
        t = np.array([
            [4.0, 1.0, 0.5]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('A', 'R', k, n, t)

        assert info == 0
        assert r.shape == (n, n)
        assert l.shape == (n, n)


class TestMB02CDErrorHandling:
    """Error handling tests."""

    def test_not_positive_definite(self):
        """Test error when matrix is not positive definite."""
        k = 2
        n = 2

        # Create a matrix that is NOT positive definite
        t = np.array([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 2.0]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('A', 'R', k, n, t)

        # Should return info=1 for non-positive definite matrix
        assert info == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
