"""
Tests for MB02DD - Updating Cholesky factorization of positive definite block Toeplitz matrix.

MB02DD updates the Cholesky factor and the generator and/or the Cholesky factor
of the inverse of a symmetric positive definite (s.p.d.) block Toeplitz matrix T,
given the information from a previous factorization (MB02CD) and additional blocks
in TA of its first block row or column.

Note: MB02DD returns only the NEW portions of R and L:
- R: Last M*K columns of the full (N+M)*K x (N+M)*K Cholesky factor
- L: Last M*K rows of the full (N+M)*K x (N+M)*K inverse Cholesky factor
"""

import numpy as np
import pytest
from ctrlsys import mb02cd, mb02dd


class TestMB02DDBasic:
    """Basic functionality tests using HTML documentation example."""

    def test_html_doc_example_job_a_typet_r(self):
        """
        Test JOB='A', TYPET='R' using HTML documentation example.

        Input: First block row of 10x10 s.p.d. block Toeplitz matrix
        K=2 (block size), N=3 (initial blocks), M=2 (additional blocks)
        TYPET='R' (row-oriented)

        Expected outputs from HTML doc (4-decimal precision).
        """
        k = 2
        n = 3
        m = 2
        s = (n + m) * k

        # First block row of the full block Toeplitz matrix (K x (N+M)*K)
        t_full = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05, 0.1, 0.04, 0.01, 0.02],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2, 0.03, 0.02, 0.03, 0.01]
        ], dtype=float, order='F')

        # Split into T (first N blocks) and TA (additional M blocks)
        t = t_full[:, :n*k].copy()
        ta = t_full[:, n*k:].copy()

        # First call MB02CD to get initial factorization
        # NOTE: t is modified in-place with transformation info!
        g, r, l, cs, info = mb02cd('A', 'R', k, n, t)
        assert info == 0

        # Now update with MB02DD
        # Need to copy last block column of R for input
        r_in = np.zeros((s, m * k), dtype=float, order='F')
        r_in[k:n*k+k, 0:k] = r[:n*k, (n-1)*k:n*k].copy()

        # Extend G for output
        g_in = np.zeros((2 * k, s), dtype=float, order='F')
        g_in[:, :n*k] = g.copy()

        # Extend cs for output
        cs_in = np.zeros(3 * (n + m - 1) * k, dtype=float, order='F')
        cs_in[:3*(n-1)*k] = cs.copy()

        ta_out, g_out, r_out, l_out, cs_out, info = mb02dd(
            'A', 'R', k, m, n, ta.copy(), t, g_in, r_in, cs_in
        )

        assert info == 0

        # Expected updated Cholesky factor R from HTML doc (full matrix)
        r_full_expected = np.array([
            [1.7321, 0.5774, 0.0577, 0.0577, 0.1155, 0.0289, 0.0577, 0.0231, 0.0058, 0.0115],
            [0.0000, 1.9149, 0.1915, 0.0348, -0.0139, 0.0957, -0.0017, 0.0035, 0.0139, 0.0017],
            [0.0000, 0.0000, 1.7205, 0.5754, 0.0558, 0.0465, 0.1145, 0.0279, 0.0564, 0.0227],
            [0.0000, 0.0000, 0.0000, 1.9142, 0.1890, 0.0357, -0.0152, 0.0953, -0.0017, 0.0033],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.7169, 0.5759, 0.0523, 0.0453, 0.1146, 0.0273],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.9118, 0.1902, 0.0357, -0.0157, 0.0955],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.7159, 0.5757, 0.0526, 0.0450],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.9118, 0.1901, 0.0357],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.7159, 0.5757],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.9117]
        ], dtype=float, order='F')

        # Expected updated inverse generator from HTML doc
        g_expected = np.array([
            [-0.5599, 0.3310, -0.0305, 0.0098, 0.0392, -0.0209, 0.0191, -0.0010, -0.0045, 0.0035],
            [-0.2289, -0.4091, 0.0612, -0.0012, 0.0125, 0.0182, 0.0042, 0.0017, 0.0014, 0.0000],
            [0.5828, 0.0000, 0.0027, -0.0029, -0.0195, 0.0072, -0.0393, 0.0057, 0.0016, -0.0580],
            [-0.1755, 0.5231, -0.0037, 0.0022, 0.0005, -0.0022, 0.0125, -0.0266, -0.0109, 0.0077]
        ], dtype=float, order='F')

        # Expected updated inverse Cholesky factor L from HTML doc (last M*K rows)
        l_full_expected = np.array([
            [0.5774, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [-0.1741, 0.5222, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, -0.0581, 0.5812, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [-0.0142, 0.0080, -0.1747, 0.5224, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [-0.0387, 0.0052, 0.0003, -0.0575, 0.5825, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0119, -0.0265, -0.0110, 0.0076, -0.1754, 0.5231, 0.0000, 0.0000, 0.0000, 0.0000],
            [-0.0199, 0.0073, -0.0391, 0.0056, 0.0017, -0.0580, 0.5828, 0.0000, 0.0000, 0.0000],
            [0.0007, -0.0023, 0.0122, -0.0265, -0.0110, 0.0077, -0.1755, 0.5231, 0.0000, 0.0000],
            [0.0027, -0.0029, -0.0195, 0.0072, -0.0393, 0.0057, 0.0016, -0.0580, 0.5828, 0.0000],
            [-0.0037, 0.0022, 0.0005, -0.0022, 0.0125, -0.0266, -0.0109, 0.0077, -0.1755, 0.5231]
        ], dtype=float, order='F')

        # MB02DD returns last M*K rows of L
        l_expected = l_full_expected[n*k:, :]

        # MB02DD returns last M*K columns of R
        r_expected = r_full_expected[:, n*k:]

        # Validate shapes: MB02DD outputs partial results
        # R: (N+M)*K x M*K (last M*K columns)
        # L: M*K x (N+M)*K (last M*K rows)
        # G: 2*K x (N+M)*K
        assert r_out.shape == (s, m * k), f"R shape: {r_out.shape}"
        assert l_out.shape == (m * k, s), f"L shape: {l_out.shape}"
        assert g_out.shape == (2 * k, s), f"G shape: {g_out.shape}"

        # Validate numerical values (HTML doc precision ~4 decimals)
        np.testing.assert_allclose(r_out, r_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(g_out, g_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(l_out, l_expected, rtol=1e-3, atol=1e-4)

    def test_job_r_update_cholesky_only(self):
        """Test JOB='R' - update generator and Cholesky factor of T."""
        k = 2
        n = 3
        m = 2
        s = (n + m) * k

        t_full = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05, 0.1, 0.04, 0.01, 0.02],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2, 0.03, 0.02, 0.03, 0.01]
        ], dtype=float, order='F')

        t = t_full[:, :n*k].copy()
        ta = t_full[:, n*k:].copy()

        g, r, l, cs, info = mb02cd('R', 'R', k, n, t)
        assert info == 0

        r_in = np.zeros((s, m * k), dtype=float, order='F')
        r_in[k:n*k+k, 0:k] = r[:n*k, (n-1)*k:n*k].copy()

        g_in = np.zeros((2 * k, s), dtype=float, order='F')
        g_in[:, :n*k] = g.copy()

        cs_in = np.zeros(3 * (n + m - 1) * k, dtype=float, order='F')
        cs_in[:3*(n-1)*k] = cs.copy()

        ta_out, g_out, r_out, l_out, cs_out, info = mb02dd(
            'R', 'R', k, m, n, ta.copy(), t, g_in, r_in, cs_in
        )

        assert info == 0
        # R: (N+M)*K x M*K (last M*K columns)
        assert r_out.shape == (s, m * k)
        assert g_out.shape == (2 * k, s)

    def test_job_o_cholesky_only(self):
        """Test JOB='O' - only compute new Cholesky factor of T."""
        k = 2
        n = 3
        m = 2
        s = (n + m) * k

        t_full = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05, 0.1, 0.04, 0.01, 0.02],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2, 0.03, 0.02, 0.03, 0.01]
        ], dtype=float, order='F')

        t = t_full[:, :n*k].copy()
        ta = t_full[:, n*k:].copy()

        g, r, l, cs, info = mb02cd('O', 'R', k, n, t)
        assert info == 0

        r_in = np.zeros((s, m * k), dtype=float, order='F')
        r_in[k:n*k+k, 0:k] = r[:n*k, (n-1)*k:n*k].copy()

        g_in = np.zeros((1, 1), dtype=float, order='F')

        cs_in = np.zeros(3 * (n + m - 1) * k, dtype=float, order='F')
        cs_in[:3*(n-1)*k] = cs.copy()

        ta_out, g_out, r_out, l_out, cs_out, info = mb02dd(
            'O', 'R', k, m, n, ta.copy(), t, g_in, r_in, cs_in
        )

        assert info == 0
        # R: (N+M)*K x M*K (last M*K columns)
        assert r_out.shape == (s, m * k)


class TestMB02DDColumnOriented:
    """Tests for column-oriented (TYPET='C') algorithm."""

    def test_typet_c_basic(self):
        """
        Test TYPET='C' (column-oriented) with the same data.

        For TYPET='C', T contains first block column ((N+M)*K x K).
        """
        k = 2
        n = 3
        m = 2
        s = (n + m) * k

        # First block column (transpose of block row)
        t_full = np.array([
            [3.0, 1.0],
            [1.0, 4.0],
            [0.1, 0.4],
            [0.1, 0.1],
            [0.2, 0.04],
            [0.05, 0.2],
            [0.1, 0.03],
            [0.04, 0.02],
            [0.01, 0.03],
            [0.02, 0.01]
        ], dtype=float, order='F')

        t = t_full[:n*k, :].copy()
        ta = t_full[n*k:, :].copy()

        g, r, l, cs, info = mb02cd('A', 'C', k, n, t)
        assert info == 0

        # For TYPET='C': G is (N+M)*K x 2*K, R is lower triangular
        r_in = np.zeros((m * k, s), dtype=float, order='F')
        r_in[0:k, k:n*k+k] = r[(n-1)*k:n*k, :n*k].copy()

        g_in = np.zeros((s, 2 * k), dtype=float, order='F')
        g_in[:n*k, :] = g.copy()

        cs_in = np.zeros(3 * (n + m - 1) * k, dtype=float, order='F')
        cs_in[:3*(n-1)*k] = cs.copy()

        ta_out, g_out, r_out, l_out, cs_out, info = mb02dd(
            'A', 'C', k, m, n, ta.copy(), t, g_in, r_in, cs_in
        )

        assert info == 0
        # For TYPET='C': R is M*K x (N+M)*K (last M*K rows of full R)
        assert r_out.shape == (m * k, s)
        # For TYPET='C', L output is (N+M)*K x M*K (last M*K columns of full L)
        assert l_out.shape == (s, m * k)
        assert g_out.shape == (s, 2 * k)


class TestMB02DDMathematicalProperties:
    """Tests validating mathematical correctness properties."""

    def test_cholesky_factorization_property(self):
        """
        Validate R'*R = T_updated for TYPET='R'.

        Mathematical property: The updated upper Cholesky factor R satisfies R'*R = T_updated.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k = 2
        n = 3
        m = 2
        s = (n + m) * k

        t_row = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05, 0.1, 0.04, 0.01, 0.02],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2, 0.03, 0.02, 0.03, 0.01]
        ], dtype=float, order='F')

        # Construct the full symmetric block Toeplitz matrix T_updated
        T_full = np.zeros((s, s), dtype=float, order='F')
        for i in range(n + m):
            for j in range(n + m):
                block_idx = abs(i - j)
                block = t_row[:, block_idx*k:(block_idx+1)*k]
                if i <= j:
                    T_full[i*k:(i+1)*k, j*k:(j+1)*k] = block
                else:
                    T_full[i*k:(i+1)*k, j*k:(j+1)*k] = block.T

        t = t_row[:, :n*k].copy()
        ta = t_row[:, n*k:].copy()

        g, r, l, cs, info = mb02cd('R', 'R', k, n, t)
        assert info == 0

        r_in = np.zeros((s, m * k), dtype=float, order='F')
        r_in[k:n*k+k, 0:k] = r[:n*k, (n-1)*k:n*k].copy()

        g_in = np.zeros((2 * k, s), dtype=float, order='F')
        g_in[:, :n*k] = g.copy()

        cs_in = np.zeros(3 * (n + m - 1) * k, dtype=float, order='F')
        cs_in[:3*(n-1)*k] = cs.copy()

        ta_out, g_out, r_out, l_out, cs_out, info = mb02dd(
            'R', 'R', k, m, n, ta.copy(), t, g_in, r_in, cs_in
        )

        assert info == 0

        # r_out contains last M*K columns of full R
        # Combine with initial R from MB02CD to get full R
        R_full = np.zeros((s, s), dtype=float, order='F')
        R_full[:n*k, :n*k] = r[:n*k, :n*k]
        R_full[:, n*k:] = r_out

        # Validate R'*R = T_updated (upper triangular R)
        r_upper = np.triu(R_full)
        T_reconstructed = r_upper.T @ r_upper

        np.testing.assert_allclose(T_reconstructed, T_full, rtol=1e-10, atol=1e-12)

    def test_inverse_cholesky_property(self):
        """
        Validate the L output matches the expected values from HTML doc.

        MB02DD returns only last M*K rows of the inverse Cholesky factor L.
        We validate these values against the expected output from the HTML documentation.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k = 2
        n = 3
        m = 2
        s = (n + m) * k

        t_row = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05, 0.1, 0.04, 0.01, 0.02],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2, 0.03, 0.02, 0.03, 0.01]
        ], dtype=float, order='F')

        t = t_row[:, :n*k].copy()
        ta = t_row[:, n*k:].copy()

        g, r, l_init, cs, info = mb02cd('A', 'R', k, n, t)
        assert info == 0

        r_in = np.zeros((s, m * k), dtype=float, order='F')
        r_in[k:n*k+k, 0:k] = r[:n*k, (n-1)*k:n*k].copy()

        g_in = np.zeros((2 * k, s), dtype=float, order='F')
        g_in[:, :n*k] = g.copy()

        cs_in = np.zeros(3 * (n + m - 1) * k, dtype=float, order='F')
        cs_in[:3*(n-1)*k] = cs.copy()

        ta_out, g_out, r_out, l_out, cs_out, info = mb02dd(
            'A', 'R', k, m, n, ta.copy(), t, g_in, r_in, cs_in
        )

        assert info == 0

        # l_out is M*K x (N+M)*K (last M*K rows of full L)
        assert l_out.shape == (m * k, s)

        # Expected L values from HTML doc (last M*K = 4 rows of full L)
        l_expected = np.array([
            [-0.0199, 0.0073, -0.0391, 0.0056, 0.0017, -0.0580, 0.5828, 0.0000, 0.0000, 0.0000],
            [0.0007, -0.0023, 0.0122, -0.0265, -0.0110, 0.0077, -0.1755, 0.5231, 0.0000, 0.0000],
            [0.0027, -0.0029, -0.0195, 0.0072, -0.0393, 0.0057, 0.0016, -0.0580, 0.5828, 0.0000],
            [-0.0037, 0.0022, 0.0005, -0.0022, 0.0125, -0.0266, -0.0109, 0.0077, -0.1755, 0.5231]
        ], dtype=float, order='F')

        np.testing.assert_allclose(l_out, l_expected, rtol=1e-3, atol=1e-4)


class TestMB02DDEdgeCases:
    """Edge case tests."""

    def test_m_equals_1_single_update(self):
        """Test with single update block (M=1)."""
        k = 2
        n = 3
        m = 1
        s = (n + m) * k

        t_full = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05, 0.1, 0.04],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2, 0.03, 0.02]
        ], dtype=float, order='F')

        t = t_full[:, :n*k].copy()
        ta = t_full[:, n*k:].copy()

        g, r, l, cs, info = mb02cd('A', 'R', k, n, t)
        assert info == 0

        r_in = np.zeros((s, m * k), dtype=float, order='F')
        r_in[k:n*k+k, 0:k] = r[:n*k, (n-1)*k:n*k].copy()

        g_in = np.zeros((2 * k, s), dtype=float, order='F')
        g_in[:, :n*k] = g.copy()

        cs_in = np.zeros(3 * (n + m - 1) * k, dtype=float, order='F')
        cs_in[:3*(n-1)*k] = cs.copy()

        ta_out, g_out, r_out, l_out, cs_out, info = mb02dd(
            'A', 'R', k, m, n, ta.copy(), t, g_in, r_in, cs_in
        )

        assert info == 0
        # R output is (N+M)*K x M*K for TYPET='R' (last M*K columns)
        assert r_out.shape == (s, m * k)
        # L output is M*K x (N+M)*K for TYPET='R', JOB='A'
        assert l_out.shape == (m * k, s)


class TestMB02DDErrorHandling:
    """Error handling tests."""

    def test_not_positive_definite(self):
        """Test error when extended matrix is not positive definite."""
        k = 2
        n = 3
        m = 2
        s = (n + m) * k

        # Use valid initial T but invalid TA that breaks positive definiteness
        t = np.array([
            [3.0, 1.0, 0.1, 0.1, 0.2, 0.05],
            [1.0, 4.0, 0.4, 0.1, 0.04, 0.2]
        ], dtype=float, order='F')

        # TA values that make the extended matrix non-positive-definite
        ta = np.array([
            [5.0, 0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0, 5.0]
        ], dtype=float, order='F')

        g, r, l, cs, info = mb02cd('A', 'R', k, n, t)
        assert info == 0

        r_in = np.zeros((s, m * k), dtype=float, order='F')
        r_in[k:n*k+k, 0:k] = r[:n*k, (n-1)*k:n*k].copy()

        g_in = np.zeros((2 * k, s), dtype=float, order='F')
        g_in[:, :n*k] = g.copy()

        cs_in = np.zeros(3 * (n + m - 1) * k, dtype=float, order='F')
        cs_in[:3*(n-1)*k] = cs.copy()

        ta_out, g_out, r_out, l_out, cs_out, info = mb02dd(
            'A', 'R', k, m, n, ta.copy(), t, g_in, r_in, cs_in
        )

        # Should return info=1 for non-positive definite matrix
        assert info == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
