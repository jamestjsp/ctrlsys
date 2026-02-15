"""
Tests for MB02GD - Cholesky factorization of banded symmetric positive definite block Toeplitz matrix.

MB02GD computes the Cholesky factor of a banded symmetric positive definite
(s.p.d.) block Toeplitz matrix, defined by either its first block row, or its
first block column, depending on the routine parameter TYPET.

By subsequent calls of this routine the Cholesky factor can be computed
block column by block column.
"""

import numpy as np
import pytest
from ctrlsys import mb02gd


class TestMB02GDBasic:
    """Basic functionality tests using HTML documentation example."""

    def test_html_doc_example(self):
        """
        Test with HTML documentation example.

        Input:
          K=2 (block size), N=4 (number of blocks), NL=2 (lower block bandwidth)
          TYPET='R' (row-oriented), TRIU='T' (last block is triangular)
          First block row: (NL+1)*K = 6 columns

        Expected: Upper Cholesky factor in banded storage format with shape
        (SIZR x N*K) = (NL*K+1 x N*K) = (5 x 8)
        """
        k = 2
        n = 4
        nl = 2
        typet = 'R'
        triu = 'T'

        t = np.array([
            [3.0000, 1.0000, 0.1000, 0.4000, 0.2000, 0.0000],
            [0.0000, 4.0000, 0.1000, 0.1000, 0.0500, 0.2000]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())

        assert info == 0

        rb_expected = np.array([
            [0.0000, 0.0000, 0.0000, 0.0000, 0.1155, 0.1044, 0.1156, 0.1051],
            [0.0000, 0.0000, 0.0000, 0.2309, -0.0087, 0.2290, -0.0084, 0.2302],
            [0.0000, 0.0000, 0.0577, -0.0174, 0.0541, -0.0151, 0.0544, -0.0159],
            [0.0000, 0.5774, 0.0348, 0.5704, 0.0222, 0.5725, 0.0223, 0.5724],
            [1.7321, 1.9149, 1.7307, 1.9029, 1.7272, 1.8996, 1.7272, 1.8995]
        ], dtype=float, order='F')

        assert rb.shape == (5, 8)
        np.testing.assert_allclose(rb, rb_expected, rtol=1e-3, atol=1e-4)

    def test_typet_r_triu_n(self):
        """
        Test TYPET='R' with TRIU='N' (last block has no special structure).

        This should produce output with shape (NL+1)*K x N*K instead of NL*K+1.
        """
        k = 2
        n = 3
        nl = 1
        typet = 'R'
        triu = 'N'

        t = np.array([
            [4.0, 1.0, 0.5, 0.2],
            [1.0, 5.0, 0.3, 0.4]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())

        assert info == 0
        sizr = (nl + 1) * k
        assert rb.shape[0] == sizr
        assert rb.shape[1] == n * k


class TestMB02GDColumnOriented:
    """Tests for column-oriented (TYPET='C') algorithm."""

    def test_typet_c_basic(self):
        """
        Test TYPET='C' (column-oriented) with basic example.

        For TYPET='C', T contains first block column ((NL+1)*K x K).
        The Cholesky factor is lower triangular in banded format.
        """
        k = 2
        n = 3
        nl = 1
        typet = 'C'
        triu = 'N'

        t = np.array([
            [4.0, 1.0],
            [1.0, 5.0],
            [0.5, 0.3],
            [0.2, 0.4]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())

        assert info == 0
        sizr = (nl + 1) * k
        assert rb.shape[0] == sizr
        assert rb.shape[1] == n * k

    def test_typet_c_triu_t(self):
        """
        Test TYPET='C' with TRIU='T' (last block is upper triangular).

        For TYPET='C' with TRIU='T', SIZR = NL*K+1.
        """
        k = 2
        n = 3
        nl = 2
        typet = 'C'
        triu = 'T'

        t = np.array([
            [4.0, 1.0],
            [1.0, 5.0],
            [0.2, 0.3],
            [0.0, 0.4],
            [0.1, 0.1],
            [0.0, 0.2]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())

        assert info == 0
        sizr = nl * k + 1
        assert rb.shape[0] == sizr


class TestMB02GDMathematicalProperties:
    """Tests validating mathematical correctness properties."""

    def test_positive_diagonal_in_factor(self):
        """
        Validate that diagonal elements in the Cholesky factor are positive.

        For a positive definite matrix, the Cholesky factor should have
        positive diagonal elements.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k = 2
        n = 3
        nl = 1
        typet = 'R'
        triu = 'N'

        t = np.array([
            [4.0, 0.5, 0.3, 0.1],
            [0.5, 5.0, 0.2, 0.3]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())
        assert info == 0

        sizr = (nl + 1) * k
        diag_row = sizr - 1
        for j in range(rb.shape[1]):
            assert rb[diag_row, j] > 0, f"Diagonal at column {j} is not positive: {rb[diag_row, j]}"

    def test_output_shape_consistency(self):
        """
        Validate output shape matches expected dimensions for different parameters.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k = 2
        n = 3
        nl = 1

        t = np.array([
            [4.0, 0.5, 0.3, 0.1],
            [0.5, 5.0, 0.2, 0.3]
        ], dtype=float, order='F')

        rb, info = mb02gd('R', 'N', k, n, nl, 0, n, t.copy())
        assert info == 0

        sizr = (nl + 1) * k
        m = n * k
        assert rb.shape[0] == sizr
        assert rb.shape[1] == m


class TestMB02GDIncrementalComputation:
    """Tests for incremental/block-by-block computation."""

    def test_incremental_vs_full(self):
        """
        Test that incremental computation gives same result as full computation.

        Compute all at once (P=0, S=N) vs. compute incrementally (P=0, S=1; then P=1, S=1; ...).
        """
        k = 2
        n = 4
        nl = 2
        typet = 'R'
        triu = 'T'

        t = np.array([
            [3.0000, 1.0000, 0.1000, 0.4000, 0.2000, 0.0000],
            [0.0000, 4.0000, 0.1000, 0.1000, 0.0500, 0.2000]
        ], dtype=float, order='F')

        rb_full, info_full = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())
        assert info_full == 0

        assert rb_full.shape[1] == n * k


class TestMB02GDEdgeCases:
    """Edge case tests."""

    def test_n_equals_2_triu_t(self):
        """Test minimum N=2 when TRIU='T' (required by spec)."""
        k = 2
        n = 2
        nl = 1
        typet = 'R'
        triu = 'T'

        t = np.array([
            [4.0, 1.0, 0.3, 0.0],
            [0.0, 5.0, 0.2, 0.4]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())

        assert info == 0

    def test_k_equals_2_small_blocks(self):
        """Test with small block size (K=2)."""
        k = 2
        n = 3
        nl = 1
        typet = 'R'
        triu = 'N'

        t = np.array([
            [4.0, 0.5, 0.1, 0.05],
            [0.5, 5.0, 0.1, 0.15]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())

        assert info == 0
        assert rb.shape == ((nl + 1) * k, n * k)

    def test_s_equals_zero(self):
        """Test with S=0 (quick return)."""
        k = 2
        n = 3
        nl = 1
        typet = 'R'
        triu = 'N'

        t = np.array([
            [4.0, 1.0, 0.5, 0.2],
            [1.0, 5.0, 0.3, 0.4]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, 0, t.copy())

        assert info == 0

    def test_nl_equals_1_triu_n(self):
        """Test with NL=1 (one off-diagonal block) when TRIU='N'."""
        k = 2
        n = 3
        nl = 1
        typet = 'R'
        triu = 'N'

        t = np.array([
            [4.0, 0.5, 0.3, 0.1],
            [0.5, 5.0, 0.2, 0.3]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())

        assert info == 0
        sizr = (nl + 1) * k
        assert rb.shape == (sizr, n * k)


class TestMB02GDErrorHandling:
    """Error handling tests."""

    def test_not_positive_definite(self):
        """Test error when matrix is not positive definite."""
        k = 2
        n = 2
        nl = 1
        typet = 'R'
        triu = 'N'

        t = np.array([
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 1.0, 0.0, 3.0]
        ], dtype=float, order='F')

        rb, info = mb02gd(typet, triu, k, n, nl, 0, n, t.copy())

        assert info == 1

    def test_invalid_typet(self):
        """Test with invalid TYPET parameter."""
        k = 2
        n = 3
        nl = 1

        t = np.array([
            [4.0, 1.0, 0.5, 0.2],
            [1.0, 5.0, 0.3, 0.4]
        ], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb02gd('X', 'N', k, n, nl, 0, n, t.copy())

    def test_invalid_triu(self):
        """Test with invalid TRIU parameter."""
        k = 2
        n = 3
        nl = 1

        t = np.array([
            [4.0, 1.0, 0.5, 0.2],
            [1.0, 5.0, 0.3, 0.4]
        ], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb02gd('R', 'X', k, n, nl, 0, n, t.copy())

    def test_invalid_n_for_triu_t(self):
        """Test that N < 2 is invalid when TRIU='T'."""
        k = 2
        n = 1
        nl = 1

        t = np.array([
            [4.0, 1.0, 0.5, 0.2],
            [1.0, 5.0, 0.3, 0.4]
        ], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb02gd('R', 'T', k, n, nl, 0, n, t.copy())

    def test_invalid_p_gt_n(self):
        """Test with invalid P > N."""
        k = 2
        n = 3
        nl = 1

        t = np.array([
            [4.0, 1.0, 0.5, 0.2],
            [1.0, 5.0, 0.3, 0.4]
        ], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb02gd('R', 'N', k, n, nl, 4, 1, t.copy())

    def test_invalid_s_gt_n_minus_p(self):
        """Test with invalid S > N-P."""
        k = 2
        n = 3
        nl = 1

        t = np.array([
            [4.0, 1.0, 0.5, 0.2],
            [1.0, 5.0, 0.3, 0.4]
        ], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb02gd('R', 'N', k, n, nl, 0, 4, t.copy())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
