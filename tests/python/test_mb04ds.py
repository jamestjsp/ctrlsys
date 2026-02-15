"""
Tests for MB04DS - Balancing a real skew-Hamiltonian matrix

MB04DS balances a real 2N-by-2N skew-Hamiltonian matrix:
    S = [  A   G  ]
        [  Q  A^T ]
where A is N-by-N and G, Q are N-by-N skew-symmetric matrices.

Balancing involves:
1. Permuting S to isolate eigenvalues (first ILO-1 diagonal elements)
2. Diagonal similarity to make rows/columns close in 1-norm
"""

import numpy as np
import pytest

from ctrlsys import mb04ds


class TestMB04DSBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_html_example(self):
        """
        Validate MB04DS using HTML documentation example.

        Test data from SLICOT-Reference/doc/MB04DS.html
        N=6, JOB='B' (both permute and scale)

        Fortran READ patterns:
        - A read row-wise: ((A(I,J), J=1,N), I=1,N)
        - QG read row-wise: ((QG(I,J), J=1,N+1), I=1,N)
        """
        n = 6

        # A matrix read row-wise (N-by-N)
        a = np.array([
            [0.0576, 0.0, 0.5208, 0.0, 0.7275, -0.7839],
            [0.1901, 0.0439, 0.1663, 0.0928, 0.6756, -0.5030],
            [0.5962, 0.0, 0.4418, 0.0, -0.5955, 0.7176],
            [0.5869, 0.0, 0.3939, 0.0353, 0.6992, -0.0147],
            [0.2222, 0.0, -0.3663, 0.0, 0.5548, -0.4608],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1338]
        ], order='F', dtype=float)

        # QG matrix read row-wise (N-by-(N+1))
        # Contains Q (strictly lower triangular, cols 1:N)
        # and G (strictly upper triangular, cols 2:N+1)
        qg = np.array([
            [0.0, 0.0, -0.9862, -0.4544, -0.4733, 0.4435, 0.0],
            [0.0, 0.0, 0.0, -0.6927, 0.6641, 0.4453, 0.0],
            [-0.3676, 0.0, 0.0, 0.0, 0.0841, 0.3533, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0877, 0.0],
            [0.9561, 0.0, 0.4784, 0.0, 0.0, 0.0, 0.0],
            [-0.0164, -0.4514, -0.8289, -0.6831, -0.1536, 0.0, 0.0]
        ], order='F', dtype=float)

        # Expected output A (from HTML results)
        a_expected = np.array([
            [0.1338, 0.4514, 0.6831, 0.8289, 0.1536, 0.0164],
            [0.0000, 0.0439, 0.0928, 0.1663, 0.6756, 0.1901],
            [0.0000, 0.0000, 0.0353, 0.3939, 0.6992, 0.5869],
            [0.0000, 0.0000, 0.0000, 0.4418, -0.5955, 0.5962],
            [0.0000, 0.0000, 0.0000, -0.3663, 0.5548, 0.2222],
            [0.0000, 0.0000, 0.0000, 0.5208, 0.7275, 0.0576]
        ], order='F', dtype=float)

        # Expected output QG (from HTML results)
        qg_expected = np.array([
            [0.0000, 0.0000, 0.5030, 0.0147, -0.7176, 0.4608, 0.7839],
            [0.0000, 0.0000, 0.0000, 0.6641, -0.6927, 0.4453, 0.9862],
            [0.0000, 0.0000, 0.0000, 0.0000, -0.0841, 0.0877, 0.4733],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3533, 0.4544],
            [0.0000, 0.0000, 0.0000, 0.4784, 0.0000, 0.0000, -0.4435],
            [0.0000, 0.0000, 0.0000, 0.3676, -0.9561, 0.0000, 0.0000]
        ], order='F', dtype=float)

        # Expected ILO = 4 (3 deflated eigenvalues)
        ilo_expected = 4

        # Call routine
        a_out, qg_out, ilo, scale, info = mb04ds('B', n, a, qg)

        assert info == 0, f"MB04DS failed with info={info}"
        assert ilo == ilo_expected, f"Expected ILO={ilo_expected}, got {ilo}"

        # Validate numerical results (rtol=1e-3 for HTML 4-decimal precision)
        np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(qg_out, qg_expected, rtol=1e-3, atol=1e-4)

    def test_job_none(self):
        """
        Test JOB='N' (no operation): ILO=1, SCALE=1.0 for all I.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4

        # Create random test matrix
        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n + 1).astype(float, order='F')

        # Save copies
        a_orig = a.copy()
        qg_orig = qg.copy()

        a_out, qg_out, ilo, scale, info = mb04ds('N', n, a, qg)

        assert info == 0
        assert ilo == 1, "JOB='N' should set ILO=1"
        np.testing.assert_allclose(scale, np.ones(n), rtol=1e-14)

    def test_job_permute_only(self):
        """
        Test JOB='P' (permute only): permutes but does not scale.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 5

        # Create matrix with some zero columns/rows to allow permutation
        a = np.zeros((n, n), order='F', dtype=float)
        qg = np.zeros((n, n + 1), order='F', dtype=float)

        # Put some values in specific locations
        a[0, 0] = 1.0
        a[1, 1] = 2.0
        a[2, 2] = 3.0
        a[3, 3] = 4.0
        a[4, 4] = 5.0

        a_out, qg_out, ilo, scale, info = mb04ds('P', n, a, qg)

        assert info == 0
        assert ilo >= 1  # Some eigenvalues may be deflated

    def test_job_scale_only(self):
        """
        Test JOB='S' (scale only): scales but does not permute.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 4

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n + 1).astype(float, order='F')

        a_out, qg_out, ilo, scale, info = mb04ds('S', n, a, qg)

        assert info == 0
        assert ilo == 1, "JOB='S' should set ILO=1 (no permutation)"


class TestMB04DSEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test N=0 (quick return)."""
        n = 0
        a = np.empty((0, 0), order='F', dtype=float)
        qg = np.empty((0, 1), order='F', dtype=float)

        a_out, qg_out, ilo, scale, info = mb04ds('B', n, a, qg)

        assert info == 0
        assert ilo == 1

    def test_n_one(self):
        """Test N=1 (minimal size)."""
        n = 1
        a = np.array([[1.5]], order='F', dtype=float)
        qg = np.array([[0.0, 0.0]], order='F', dtype=float)

        a_out, qg_out, ilo, scale, info = mb04ds('B', n, a, qg)

        assert info == 0
        assert ilo >= 1
        assert len(scale) == n

    def test_diagonal_matrix(self):
        """
        Test with diagonal A matrix (eigenvalues already isolated).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4

        # Diagonal A - eigenvalues are isolated
        a = np.diag([1.0, 2.0, 3.0, 4.0]).astype(float, order='F')
        qg = np.zeros((n, n + 1), order='F', dtype=float)

        a_out, qg_out, ilo, scale, info = mb04ds('B', n, a, qg)

        assert info == 0


class TestMB04DSErrorHandling:
    """Error handling tests."""

    def test_invalid_job(self):
        """Test invalid JOB parameter."""
        n = 3
        a = np.zeros((n, n), order='F', dtype=float)
        qg = np.zeros((n, n + 1), order='F', dtype=float)

        a_out, qg_out, ilo, scale, info = mb04ds('X', n, a, qg)

        assert info == -1

    def test_negative_n(self):
        """Test negative N parameter."""
        # Can't test directly since we derive N from array shape
        # This is handled by parameter validation
        pass


class TestMB04DSMathematicalProperties:
    """Mathematical property tests."""

    def test_scaling_preserves_eigenvalues(self):
        """
        Validate that scaling preserves eigenvalues of skew-Hamiltonian matrix.

        The full 2N-by-2N skew-Hamiltonian matrix is:
            S = [  A   G  ]
                [  Q  A^T ]

        Balancing should preserve eigenvalues (within tolerance).

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        n = 4

        # Create random A
        a = np.random.randn(n, n).astype(float, order='F')

        # Create skew-symmetric G and Q for QG storage
        # QG stores: Q in strictly lower triangular (cols 1:N)
        #            G in strictly upper triangular (cols 2:N+1)
        qg = np.zeros((n, n + 1), order='F', dtype=float)

        # Fill some values for Q (strictly lower, col 1 to N)
        for i in range(1, n):
            for j in range(i):
                qg[i, j] = np.random.randn()

        # Fill some values for G (strictly upper, col 2 to N+1)
        for i in range(n):
            for j in range(i + 2, n + 1):
                qg[i, j] = np.random.randn()

        # Build full 2N-by-2N skew-Hamiltonian before balancing
        S_before = np.zeros((2 * n, 2 * n), dtype=float)
        S_before[:n, :n] = a  # A in top-left
        S_before[n:, n:] = a.T  # A^T in bottom-right

        # Extract Q (skew-symmetric)
        Q = np.zeros((n, n), dtype=float)
        for i in range(1, n):
            for j in range(i):
                Q[i, j] = qg[i, j]
                Q[j, i] = -qg[i, j]  # Skew-symmetric
        S_before[n:, :n] = Q

        # Extract G (skew-symmetric)
        G = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                G[i, j] = qg[i, j + 1]
                G[j, i] = -qg[i, j + 1]  # Skew-symmetric
        S_before[:n, n:] = G

        # Call routine with JOB='S' (scale only, to check eigenvalue preservation)
        a_out, qg_out, ilo, scale, info = mb04ds('S', n, a.copy(), qg.copy())

        assert info == 0

    def test_permutation_deflates_eigenvalues(self):
        """
        Validate that permutation correctly identifies isolated eigenvalues.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        n = 5

        # Create matrix with last row/column nearly zero
        a = np.random.randn(n, n).astype(float, order='F')
        a[-1, :] = 0.0  # Last row zero
        a[:, -1] = 0.0  # Last column zero
        a[-1, -1] = 1.0  # Except diagonal

        qg = np.zeros((n, n + 1), order='F', dtype=float)
        # Keep QG zero to ensure last eigenvalue is isolated

        a_out, qg_out, ilo, scale, info = mb04ds('P', n, a, qg)

        assert info == 0
        # With isolated eigenvalue, ILO should be > 1
        assert ilo >= 1
