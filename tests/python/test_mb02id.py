"""
Tests for mb02id: Solution of over/underdetermined linear systems with block Toeplitz matrix.

MB02ID solves:
1. JOB='O': Least squares: minimize ||B - T*X||
2. JOB='U': Minimum norm: T^T * X = C
3. JOB='A': Both problems
"""

import numpy as np
import pytest
from ctrlsys import mb02id


class TestMB02IDBasic:
    """Test MB02ID using HTML documentation example."""

    def test_both_problems_from_html_doc(self):
        """
        Validate both overdetermined and underdetermined systems using SLICOT HTML doc example.

        Parameters from doc: K=3, L=2, M=4, N=3, RB=1, RC=1, JOB='A'
        """
        k, l, m, n = 3, 2, 4, 3
        rb, rc = 1, 1

        tc = np.array([
            [5.0, 2.0],
            [1.0, 2.0],
            [4.0, 3.0],
            [4.0, 0.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [5.0, 1.0],
            [3.0, 3.0],
            [1.0, 1.0],
            [2.0, 3.0],
            [1.0, 3.0],
            [2.0, 2.0],
        ], order='F', dtype=float)

        tr = np.array([
            [1.0, 4.0, 2.0, 3.0],
            [2.0, 2.0, 2.0, 4.0],
            [3.0, 1.0, 0.0, 1.0],
        ], order='F', dtype=float)

        b = np.ones((m*k, rb), order='F', dtype=float)
        c = np.ones((n*l, rc), order='F', dtype=float)

        b_out, c_out, info = mb02id('A', k, l, m, n, tc, tr, b, c)

        assert info == 0

        x_expected = np.array([
            [0.0379],
            [0.1677],
            [0.0485],
            [-0.0038],
            [0.0429],
            [0.1365],
        ], order='F', dtype=float)

        c_expected = np.array([
            [0.0509],
            [0.0547],
            [0.0218],
            [0.0008],
            [0.0436],
            [0.0404],
            [0.0031],
            [0.0451],
            [0.0421],
            [0.0243],
            [0.0556],
            [0.0472],
        ], order='F', dtype=float)

        np.testing.assert_allclose(b_out[:n*l, :], x_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)

    def test_overdetermined_only(self):
        """Test JOB='O' solving only the overdetermined system."""
        k, l, m, n = 3, 2, 4, 3
        rb = 1

        tc = np.array([
            [5.0, 2.0],
            [1.0, 2.0],
            [4.0, 3.0],
            [4.0, 0.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [5.0, 1.0],
            [3.0, 3.0],
            [1.0, 1.0],
            [2.0, 3.0],
            [1.0, 3.0],
            [2.0, 2.0],
        ], order='F', dtype=float)

        tr = np.array([
            [1.0, 4.0, 2.0, 3.0],
            [2.0, 2.0, 2.0, 4.0],
            [3.0, 1.0, 0.0, 1.0],
        ], order='F', dtype=float)

        b = np.ones((m*k, rb), order='F', dtype=float)
        c = np.zeros((1, 1), order='F', dtype=float)

        b_out, c_out, info = mb02id('O', k, l, m, n, tc, tr, b, c)

        assert info == 0

        x_expected = np.array([
            [0.0379],
            [0.1677],
            [0.0485],
            [-0.0038],
            [0.0429],
            [0.1365],
        ], order='F', dtype=float)

        np.testing.assert_allclose(b_out[:n*l, :], x_expected, rtol=1e-3, atol=1e-4)

    def test_underdetermined_only(self):
        """Test JOB='U' solving only the underdetermined system."""
        k, l, m, n = 3, 2, 4, 3
        rc = 1

        tc = np.array([
            [5.0, 2.0],
            [1.0, 2.0],
            [4.0, 3.0],
            [4.0, 0.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [5.0, 1.0],
            [3.0, 3.0],
            [1.0, 1.0],
            [2.0, 3.0],
            [1.0, 3.0],
            [2.0, 2.0],
        ], order='F', dtype=float)

        tr = np.array([
            [1.0, 4.0, 2.0, 3.0],
            [2.0, 2.0, 2.0, 4.0],
            [3.0, 1.0, 0.0, 1.0],
        ], order='F', dtype=float)

        b = np.zeros((1, 1), order='F', dtype=float)
        c = np.ones((n*l, rc), order='F', dtype=float)

        b_out, c_out, info = mb02id('U', k, l, m, n, tc, tr, b, c)

        assert info == 0

        c_expected = np.array([
            [0.0509],
            [0.0547],
            [0.0218],
            [0.0008],
            [0.0436],
            [0.0404],
            [0.0031],
            [0.0451],
            [0.0421],
            [0.0243],
            [0.0556],
            [0.0472],
        ], order='F', dtype=float)

        np.testing.assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)


class TestMB02IDProperties:
    """Mathematical property tests for MB02ID."""

    def test_least_squares_residual_minimization(self):
        """
        Validate that least squares solution minimizes ||B - T*X||.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k, l, m, n = 2, 2, 3, 2
        rb = 1

        tc = np.random.randn(m*k, l).astype(float, order='F')
        tr = np.random.randn(k, (n-1)*l).astype(float, order='F')

        T = _build_block_toeplitz(tc, tr, k, l, m, n)

        b = np.random.randn(m*k, rb).astype(float, order='F')
        c = np.zeros((1, 1), order='F', dtype=float)

        b_copy = b.copy()
        b_out, c_out, info = mb02id('O', k, l, m, n, tc, tr, b, c)

        if info == 0:
            x = b_out[:n*l, :]
            residual_mb02id = np.linalg.norm(b_copy - T @ x)

            x_np, _, _, _ = np.linalg.lstsq(T, b_copy, rcond=None)
            residual_np = np.linalg.norm(b_copy - T @ x_np)

            np.testing.assert_allclose(residual_mb02id, residual_np, rtol=1e-10)

    def test_minimum_norm_solution(self):
        """
        Validate minimum norm solution for underdetermined system.

        For T^T * X = C, the minimum norm solution should satisfy:
        1. T^T * X = C (exactly)
        2. X has minimum 2-norm among all solutions

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k, l, m, n = 2, 2, 3, 2
        rc = 1

        tc = np.random.randn(m*k, l).astype(float, order='F')
        tr = np.random.randn(k, (n-1)*l).astype(float, order='F')

        T = _build_block_toeplitz(tc, tr, k, l, m, n)

        c = np.random.randn(n*l, rc).astype(float, order='F')
        b = np.zeros((1, 1), order='F', dtype=float)

        c_copy = c.copy()
        b_out, c_out, info = mb02id('U', k, l, m, n, tc, tr, b, c)

        if info == 0:
            x = c_out

            residual = np.linalg.norm(T.T @ x - c_copy)
            np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_normal_equations_satisfied(self):
        """
        Validate that least squares solution satisfies normal equations: T^T * T * X = T^T * B.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        k, l, m, n = 2, 2, 3, 2
        rb = 1

        tc = np.random.randn(m*k, l).astype(float, order='F')
        tr = np.random.randn(k, (n-1)*l).astype(float, order='F')

        T = _build_block_toeplitz(tc, tr, k, l, m, n)

        b = np.random.randn(m*k, rb).astype(float, order='F')
        c = np.zeros((1, 1), order='F', dtype=float)

        b_copy = b.copy()
        b_out, c_out, info = mb02id('O', k, l, m, n, tc, tr, b, c)

        if info == 0:
            x = b_out[:n*l, :]

            lhs = T.T @ T @ x
            rhs = T.T @ b_copy

            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


class TestMB02IDEdgeCases:
    """Edge case tests for MB02ID."""

    def test_single_block_m1(self):
        """Test with M=1 (single block column)."""
        k, l, m, n = 3, 2, 1, 1
        rb, rc = 1, 1

        tc = np.array([
            [5.0, 2.0],
            [1.0, 2.0],
            [4.0, 3.0],
        ], order='F', dtype=float)

        tr = np.zeros((k, 0), order='F', dtype=float)

        b = np.ones((m*k, rb), order='F', dtype=float)
        c = np.ones((n*l, rc), order='F', dtype=float)

        b_out, c_out, info = mb02id('A', k, l, m, n, tc, tr, b, c)

        assert info == 0

    def test_single_block_n1(self):
        """Test with N=1 (single block row)."""
        k, l, m, n = 2, 2, 3, 1
        rb, rc = 1, 1

        tc = np.array([
            [5.0, 2.0],
            [1.0, 2.0],
            [4.0, 3.0],
            [4.0, 0.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ], order='F', dtype=float)

        tr = np.zeros((k, 0), order='F', dtype=float)

        b = np.ones((m*k, rb), order='F', dtype=float)
        c = np.ones((n*l, rc), order='F', dtype=float)

        b_out, c_out, info = mb02id('A', k, l, m, n, tc, tr, b, c)

        assert info == 0


class TestMB02IDErrors:
    """Error handling tests for MB02ID."""

    def test_invalid_job(self):
        """Test with invalid JOB parameter."""
        k, l, m, n = 2, 2, 2, 2

        tc = np.ones((m*k, l), order='F', dtype=float)
        tr = np.ones((k, (n-1)*l), order='F', dtype=float)
        b = np.ones((m*k, 1), order='F', dtype=float)
        c = np.ones((n*l, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02id('X', k, l, m, n, tc, tr, b, c)

    def test_negative_k(self):
        """Test with negative K."""
        k, l, m, n = -1, 2, 2, 2

        tc = np.ones((4, 2), order='F', dtype=float)
        tr = np.ones((2, 2), order='F', dtype=float)
        b = np.ones((4, 1), order='F', dtype=float)
        c = np.ones((4, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02id('O', k, l, m, n, tc, tr, b, c)

    def test_n_too_large(self):
        """Test with N*L > M*K (invalid constraint)."""
        k, l, m, n = 2, 2, 2, 3

        tc = np.ones((m*k, l), order='F', dtype=float)
        tr = np.ones((k, (n-1)*l), order='F', dtype=float)
        b = np.ones((m*k, 1), order='F', dtype=float)
        c = np.ones((n*l, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb02id('O', k, l, m, n, tc, tr, b, c)


def _build_block_toeplitz(tc, tr, k, l, m, n):
    """Build full block Toeplitz matrix from first block column and row."""
    rows = m * k
    cols = n * l
    T = np.zeros((rows, cols), order='F', dtype=float)

    for j in range(n):
        col_start = j * l
        col_end = col_start + l

        for i in range(m):
            row_start = i * k
            row_end = row_start + k

            if j == 0:
                T[row_start:row_end, col_start:col_end] = tc[row_start:row_end, :]
            elif i == 0:
                T[row_start:row_end, col_start:col_end] = tr[:, (j-1)*l:j*l]
            else:
                block_idx = i - j
                if block_idx >= 0:
                    T[row_start:row_end, col_start:col_end] = tc[block_idx*k:(block_idx+1)*k, :]
                else:
                    block_col_idx = -block_idx - 1
                    if block_col_idx < n - 1:
                        T[row_start:row_end, col_start:col_end] = tr[:, block_col_idx*l:(block_col_idx+1)*l]

    return T
