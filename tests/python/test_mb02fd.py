"""
Tests for MB02FD - Incomplete Cholesky factor of positive definite block Toeplitz matrix.

MB02FD computes the incomplete Cholesky (ICC) factor of a symmetric
positive definite (s.p.d.) block Toeplitz matrix T, defined by
either its first block row, or its first block column.

By subsequent calls, further rows/columns of the Cholesky factor can be added.
"""

import numpy as np
import pytest
from ctrlsys import mb02fd


class TestMB02FDBasic:
    """Basic functionality tests using HTML documentation example."""

    def test_html_doc_example(self):
        """
        Test with HTML documentation example.

        Input: First block row of 8x8 s.p.d. block Toeplitz matrix
        K=2 (block size), N=4 (number of blocks)
        TYPET='R' (row-oriented)

        Expected: Upper trapezoidal ICC factor R

        The example computes ICC in 3 iterations with S=[0,1,1].
        Here we test computing all at once with S=N for P=0.
        """
        k = 2
        n = 4
        m = n * k

        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05, 0.2, 0.3],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2, 0.1, 0.2]
        ], dtype=float, order='F')

        r, info = mb02fd('R', k, n, 0, n, t.copy())

        assert info == 0

        r_expected = np.array([
            [1.7321, 0.5774, 0.0577, 0.0577, 0.1155, 0.0289, 0.1155, 0.1732],
            [0.0000, 1.9149, 0.1915, 0.0348, -0.0139, 0.0957, 0.0174, 0.0522],
            [0.0000, 0.0000, 1.7205, 0.5754, 0.0558, 0.0465, 0.1104, 0.0174],
            [0.0000, 0.0000, 0.0000, 1.9142, 0.1890, 0.0357, -0.0161, 0.0931]
        ], dtype=float, order='F')

        assert r.shape[0] == n * k
        assert r.shape[1] == n * k

        np.testing.assert_allclose(r[:4, :], r_expected, rtol=1e-3, atol=1e-4)

    def test_single_block_row_compute(self):
        """
        Test computing single block row at a time (S=1).

        This tests the incremental nature of the algorithm.
        """
        k = 2
        n = 3
        m = n * k

        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        r, info = mb02fd('R', k, n, 0, 1, t.copy())
        assert info == 0
        assert r.shape[0] == k
        assert r.shape[1] == m


class TestMB02FDColumnOriented:
    """Tests for column-oriented (TYPET='C') algorithm."""

    def test_typet_c_basic(self):
        """
        Test TYPET='C' (column-oriented) with basic example.

        For TYPET='C', T contains first block column (N*K x K).
        The ICC factor R is lower trapezoidal.
        """
        k = 2
        n = 3
        m = n * k

        t = np.array([
            [3.0, 1.0],
            [1.0, 4.0],
            [0.1, 0.4],
            [0.1, 0.1],
            [0.2, 0.04],
            [0.05, 0.2]
        ], dtype=float, order='F')

        r, info = mb02fd('C', k, n, 0, n, t.copy())

        assert info == 0
        assert r.shape == (m, m)


class TestMB02FDMathematicalProperties:
    """Tests validating mathematical correctness properties."""

    def test_cholesky_factorization_property(self):
        """
        Validate R'*R = T for TYPET='R'.

        Mathematical property: The upper ICC factor R should satisfy R'*R approx T.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k = 2
        n = 3
        m = n * k

        t_row = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        T_full = np.zeros((m, m), dtype=float, order='F')
        for i in range(n):
            for j in range(n):
                block_idx = abs(i - j)
                block = t_row[:, block_idx*k:(block_idx+1)*k]
                if i <= j:
                    T_full[i*k:(i+1)*k, j*k:(j+1)*k] = block
                else:
                    T_full[i*k:(i+1)*k, j*k:(j+1)*k] = block.T

        r, info = mb02fd('R', k, n, 0, n, t_row.copy())
        assert info == 0

        r_upper = np.triu(r)
        T_reconstructed = r_upper.T @ r_upper

        np.testing.assert_allclose(T_reconstructed, T_full, rtol=1e-10, atol=1e-12)

    def test_symmetry_of_reconstructed_matrix(self):
        """
        Validate that R'*R is symmetric.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k = 2
        n = 3

        t_row = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        r, info = mb02fd('R', k, n, 0, n, t_row.copy())
        assert info == 0

        r_upper = np.triu(r)
        T_reconstructed = r_upper.T @ r_upper
        np.testing.assert_allclose(T_reconstructed, T_reconstructed.T, rtol=1e-14, atol=1e-15)


class TestMB02FDEdgeCases:
    """Edge case tests."""

    def test_n_equals_1(self):
        """Test with single block (N=1)."""
        k = 2
        n = 1

        t = np.array([
            [3.0, 1.0],
            [1.0, 4.0]
        ], dtype=float, order='F')

        r, info = mb02fd('R', k, n, 0, 1, t.copy())

        assert info == 0
        assert r.shape == (k, k)

    def test_k_equals_1_scalar_blocks(self):
        """Test with scalar blocks (K=1)."""
        k = 1
        n = 3

        t = np.array([
            [4.0, 1.0, 0.5]
        ], dtype=float, order='F')

        r, info = mb02fd('R', k, n, 0, n, t.copy())

        assert info == 0
        assert r.shape == (n, n)

    def test_s_equals_zero(self):
        """Test with S=0 (quick return)."""
        k = 2
        n = 3

        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        r, info = mb02fd('R', k, n, 0, 0, t.copy())

        assert info == 0


class TestMB02FDErrorHandling:
    """Error handling tests."""

    def test_not_positive_definite(self):
        """Test error when matrix is not positive definite."""
        k = 2
        n = 2

        t = np.array([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 2.0]
        ], dtype=float, order='F')

        r, info = mb02fd('R', k, n, 0, n, t.copy())

        assert info == 1

    def test_invalid_p(self):
        """Test with invalid P > N."""
        k = 2
        n = 3

        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb02fd('R', k, n, 4, 1, t.copy())

    def test_invalid_s(self):
        """Test with invalid S > N-P."""
        k = 2
        n = 3

        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb02fd('R', k, n, 0, 4, t.copy())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
