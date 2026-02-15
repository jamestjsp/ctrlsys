"""
Tests for MB02QD - Linear least squares solution using complete orthogonal factorization.

MB02QD computes a solution, optionally corresponding to specified free elements,
to a real linear least squares problem: minimize || A * X - B || using a complete
orthogonal factorization of the M-by-N matrix A, which may be rank-deficient.
"""

import numpy as np
import pytest

from ctrlsys import mb02qd


class TestMB02QDBasic:
    """Test basic functionality using HTML documentation example."""

    def test_html_example_least_squares(self):
        """
        Validate basic functionality using SLICOT HTML doc example.

        Tests standard least squares solution (JOB='L') with rank-deficient matrix.
        """
        m, n, nrhs = 4, 3, 2
        rcond = 2.3e-16
        svlmax = 0.0

        # From HTML doc: READ (( A(I,J), J = 1,N ), I = 1,M) - row-wise read
        # Data:
        #   2.0  2.0 -3.0
        #   3.0  3.0 -1.0
        #   4.0  4.0 -5.0
        #  -1.0 -1.0 -2.0
        a = np.array([
            [2.0, 2.0, -3.0],
            [3.0, 3.0, -1.0],
            [4.0, 4.0, -5.0],
            [-1.0, -1.0, -2.0]
        ], order='F', dtype=float)

        # From HTML doc: READ (( B(I,J), J = 1,NRHS ), I = 1,M) - row-wise read
        # Data:
        #   1.0  0.0
        #   0.0  0.0
        #   0.0  0.0
        #   0.0  1.0
        b = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        # Expected outputs from HTML doc:
        # Effective rank = 2
        # SVAL = 7.8659, 2.6698, 0.0000
        # X = [[-0.0034, -0.1054],
        #      [-0.0034, -0.1054],
        #      [-0.0816, -0.1973]]
        expected_rank = 2
        expected_sval = np.array([7.8659, 2.6698, 0.0000], dtype=float)
        expected_x = np.array([
            [-0.0034, -0.1054],
            [-0.0034, -0.1054],
            [-0.0816, -0.1973]
        ], order='F', dtype=float)

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a, b)

        assert info == 0
        assert rank == expected_rank
        np.testing.assert_allclose(sval, expected_sval, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(x[:n, :], expected_x, rtol=1e-3, atol=1e-4)

    def test_full_rank_overdetermined(self):
        """
        Test overdetermined system with full rank matrix.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m, n, nrhs = 6, 3, 2
        rcond = 1e-14
        svlmax = 0.0

        # Create well-conditioned full rank matrix
        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, nrhs).astype(float, order='F')

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a.copy(order='F'), b.copy(order='F'))

        assert info == 0
        assert rank == n  # Full rank expected
        assert sval[0] > 0  # Largest singular value positive
        assert sval[1] > 0  # Smallest of R11 positive
        assert x.shape == (max(m, n), nrhs)

    def test_rank_deficient_matrix(self):
        """
        Test rank-deficient matrix detection.

        Creates matrix with rank 2 from 4x3 matrix.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m, n, nrhs = 4, 3, 1
        rcond = 1e-10
        svlmax = 0.0

        # Create rank-deficient matrix: column 2 = column 1
        a = np.random.randn(m, n).astype(float, order='F')
        a[:, 1] = a[:, 0]  # Make column 2 = column 1

        b = np.random.randn(m, nrhs).astype(float, order='F')

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a.copy(order='F'), b.copy(order='F'))

        assert info == 0
        assert rank == 2  # Rank should be 2 due to linear dependency

    def test_underdetermined_system(self):
        """
        Test underdetermined system (more columns than rows).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m, n, nrhs = 3, 5, 1
        rcond = 1e-14
        svlmax = 0.0

        a = np.random.randn(m, n).astype(float, order='F')
        # B must have at least max(m,n) rows for solution output
        b_full = np.zeros((max(m, n), nrhs), order='F', dtype=float)
        b_full[:m, :] = np.random.randn(m, nrhs)

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a.copy(order='F'), b_full.copy(order='F'))

        assert info == 0
        assert rank <= min(m, n)
        assert x.shape == (max(m, n), nrhs)


class TestMB02QDModeParameters:
    """Test different mode parameter combinations."""

    def test_job_f_with_free_elements(self):
        """
        Test JOB='F' with specified free elements.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m, n, nrhs = 4, 3, 1
        rcond = 1e-10
        svlmax = 0.0

        # Create rank-deficient matrix
        a = np.random.randn(m, n).astype(float, order='F')
        a[:, 2] = a[:, 0] + a[:, 1]  # Make rank 2

        b = np.random.randn(m, nrhs).astype(float, order='F')
        y = np.array([0.5], dtype=float)  # Free element for N-RANK=1 variable

        x, rank, sval, jpvt, info = mb02qd('F', 'N', m, n, nrhs, rcond, svlmax,
                                            a.copy(order='F'), b.copy(order='F'), y)

        assert info == 0
        assert rank == 2

    def test_iniper_p_initial_permutation(self):
        """
        Test INIPER='P' with initial column permutation.

        Random seed: 321 (for reproducibility)
        """
        np.random.seed(321)
        m, n, nrhs = 4, 3, 1
        rcond = 1e-14
        svlmax = 0.0

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, nrhs).astype(float, order='F')

        # Initial permutation: column 1 and 3 are initial, column 2 is free
        jpvt = np.array([1, 0, 1], dtype=np.int32)

        # Pass jpvt as positional argument (y=None before it)
        x, rank, sval, jpvt_out, info = mb02qd('L', 'P', m, n, nrhs, rcond, svlmax,
                                                a.copy(order='F'), b.copy(order='F'),
                                                None, jpvt)

        assert info == 0
        assert rank > 0


class TestMB02QDEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_dimensions(self):
        """Test with zero matrix dimensions."""
        m, n, nrhs = 0, 0, 0
        rcond = 1e-14
        svlmax = 0.0

        a = np.array([], dtype=float).reshape(0, 0)
        b = np.array([], dtype=float).reshape(0, 0)

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a, b)

        assert info == 0
        assert rank == 0

    def test_nrhs_zero(self):
        """Test with zero right-hand sides (just factorization)."""
        m, n, nrhs = 3, 3, 0
        rcond = 1e-14
        svlmax = 0.0

        np.random.seed(555)
        a = np.random.randn(m, n).astype(float, order='F')
        b = np.array([], dtype=float).reshape(max(m, n), 0)

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a.copy(order='F'), b)

        assert info == 0
        assert rank > 0  # Rank determined even without RHS

    def test_single_element_matrix(self):
        """Test with 1x1 matrix."""
        m, n, nrhs = 1, 1, 1
        rcond = 1e-14
        svlmax = 0.0

        a = np.array([[3.0]], order='F', dtype=float)
        b = np.array([[6.0]], order='F', dtype=float)

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a, b)

        assert info == 0
        assert rank == 1
        np.testing.assert_allclose(x[0, 0], 2.0, rtol=1e-14)  # x = 6/3 = 2

    def test_zero_matrix(self):
        """Test with all-zero matrix."""
        m, n, nrhs = 3, 3, 1
        rcond = 1e-14
        svlmax = 0.0

        a = np.zeros((m, n), order='F', dtype=float)
        b = np.ones((m, nrhs), order='F', dtype=float)

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a, b)

        assert info == 0
        assert rank == 0
        # Zero solution expected for zero matrix
        np.testing.assert_allclose(x[:n, :], np.zeros((n, nrhs)), atol=1e-14)


class TestMB02QDMathematicalProperties:
    """Test mathematical properties and invariants."""

    def test_residual_minimization(self):
        """
        Verify that solution minimizes ||A*x - b||.

        The least squares solution should produce smaller residual than
        any other solution.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        m, n, nrhs = 6, 3, 1
        rcond = 1e-14
        svlmax = 0.0

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, nrhs).astype(float, order='F')

        a_orig = a.copy()
        b_orig = b.copy()

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a, b)

        assert info == 0

        # Compute residual ||A*x - b||
        residual = np.linalg.norm(a_orig @ x[:n, :] - b_orig)

        # Compare with a random perturbation
        x_perturbed = x[:n, :] + 0.1 * np.random.randn(n, nrhs)
        residual_perturbed = np.linalg.norm(a_orig @ x_perturbed - b_orig)

        assert residual <= residual_perturbed

    def test_full_rank_exact_solution(self):
        """
        For full-rank square system, solution should satisfy A*x = b exactly.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n = 4
        m, nrhs = n, 1
        rcond = 1e-14
        svlmax = 0.0

        # Create well-conditioned square matrix
        a = np.random.randn(n, n).astype(float, order='F')
        a = a + 2.0 * np.eye(n)  # Make diagonally dominant

        x_true = np.random.randn(n, nrhs).astype(float, order='F')
        b = a @ x_true

        a_copy = a.copy()
        b_copy = b.copy()

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a_copy, b_copy)

        assert info == 0
        assert rank == n

        # Permute solution back according to jpvt
        x_perm = x[:n, :]

        # Verify A*x = b (should be exact for full-rank square system)
        np.testing.assert_allclose(a @ x_perm, b, rtol=1e-12, atol=1e-12)

    def test_singular_value_estimates(self):
        """
        Verify that SVAL estimates are reasonable.

        SVAL(1) >= SVAL(2) >= SVAL(3) >= 0

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        m, n, nrhs = 5, 4, 1
        rcond = 1e-14
        svlmax = 0.0

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, nrhs).astype(float, order='F')

        x, rank, sval, jpvt, info = mb02qd('L', 'N', m, n, nrhs, rcond, svlmax, a, b)

        assert info == 0
        # SVAL ordering: largest >= smallest of R11 >= smallest of R(1:rank+1)
        assert sval[0] >= sval[1] >= 0
        assert sval[0] >= sval[2] >= 0


class TestMB02QDErrorHandling:
    """Test error conditions and parameter validation."""

    def test_invalid_job(self):
        """Test with invalid JOB parameter."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)

        x, rank, sval, jpvt, info = mb02qd('X', 'N', 2, 2, 1, 1e-14, 0.0, a, b)
        assert info == -1

    def test_invalid_iniper(self):
        """Test with invalid INIPER parameter."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)

        x, rank, sval, jpvt, info = mb02qd('L', 'X', 2, 2, 1, 1e-14, 0.0, a, b)
        assert info == -2

    def test_negative_rcond(self):
        """Test with negative RCOND."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)

        x, rank, sval, jpvt, info = mb02qd('L', 'N', 2, 2, 1, -1.0, 0.0, a, b)
        assert info == -6

    def test_negative_svlmax(self):
        """Test with negative SVLMAX."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)

        x, rank, sval, jpvt, info = mb02qd('L', 'N', 2, 2, 1, 1e-14, -1.0, a, b)
        assert info == -7

    def test_rcond_out_of_range(self):
        """Test with RCOND > 1."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)

        x, rank, sval, jpvt, info = mb02qd('L', 'N', 2, 2, 1, 2.0, 0.0, a, b)
        assert info == -6
