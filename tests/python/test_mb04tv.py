"""
Tests for MB04TV: Reduce submatrix A(k) to upper triangular form via column Givens rotations.

MB04TV reduces A(IFIRA:ma, IFICA:na) to upper triangular by applying column Givens
rotations. The same rotations are applied to E(1:IFIRA-1, IFICA:na).

Key relationships:
- A(k) = A(IFIRA:ma, IFICA:na) where ma = IFIRA-1+NRA, na = IFICA-1+NCA
- E(k) = E(1:IFIRA-1, IFICA:na) - different row range than A!
- If UPDATZ, Z accumulates column transformations

Algorithm: Bottom-up row processing with right-to-left column elimination.
"""

import numpy as np
import pytest
from ctrlsys import mb04tv


class TestMB04TVBasic:
    """Basic functionality tests."""

    def test_2x3_to_upper_triangular(self):
        """
        Test basic 2x3 submatrix reduction to upper triangular.

        A(k) is 2x3 starting at (1,1) (0-indexed):
            [[1, 2, 3],
             [4, 5, 6]]

        After reduction, last row should have zeros below diagonal.
        """
        n = 3
        nra = 2
        nca = 3
        ifira = 1  # 1-based row index
        ifica = 1  # 1-based column index

        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=float, order='F')

        e = np.zeros((0, 3), dtype=float, order='F')  # Empty, ifira-1=0 rows
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=True
        )

        assert info == 0

        # Result should be upper triangular in submatrix
        # Row 1 (index 1, 0-based) should have zeros in cols 0 (ifica-1)
        np.testing.assert_allclose(a_out[1, 0], 0.0, atol=1e-14)

        # Z should be orthogonal (use atol for machine precision zeros)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)

    def test_3x4_submatrix_reduction(self):
        """
        Test 3x4 submatrix starting at offset position.

        Full A is 5x6, submatrix A(k) = A(2:4, 2:5) (1-based) is 3x4.
        E(k) = E(1:1, 2:5) - just 1 row.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        n = 6
        nra = 3
        nca = 4
        ifira = 2  # 1-based
        ifica = 2  # 1-based

        a = np.random.randn(5, n).astype(float, order='F')
        e = np.random.randn(1, n).astype(float, order='F')  # ifira-1 = 1 row
        z = np.eye(n, dtype=float, order='F')

        a_orig = a.copy()

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=True
        )

        assert info == 0

        # Check upper triangular structure in A(k)
        # A(k) rows are ifira-1:ifira-1+nra = 1:4 (0-based), cols ifica-1:ifica-1+nca = 1:5
        # For upper triangular: A[i,j] = 0 for j < i (relative to submatrix start)
        # Row 2 (relative 1): j=0 (col 1) should be 0
        # Row 3 (relative 2): j=0,1 (cols 1,2) should be 0

        # Row ifira+nra-1 = row 4 (index 3), zeros at cols ifica-1, ifica (indices 1,2)
        np.testing.assert_allclose(a_out[3, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(a_out[3, 2], 0.0, atol=1e-14)

        # Row ifira+nra-2 = row 3 (index 2), zero at col ifica-1 (index 1)
        np.testing.assert_allclose(a_out[2, 1], 0.0, atol=1e-14)

        # Z orthogonal (use atol for machine precision zeros)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)


class TestMB04TVMathProperties:
    """Mathematical property validation tests."""

    def test_column_transformation_preserves_rank(self):
        """
        Verify column Givens rotations preserve matrix rank.

        The transformation A' = A @ Z should preserve rank since Z is orthogonal.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        n = 5
        nra = 3
        nca = 4
        ifira = 2
        ifica = 2

        a = np.random.randn(4, n).astype(float, order='F')
        e = np.random.randn(1, n).astype(float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_orig = a.copy()
        rank_before = np.linalg.matrix_rank(a_orig)

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=True
        )

        assert info == 0

        # Rank should be preserved
        rank_after = np.linalg.matrix_rank(a_out)
        assert rank_after == rank_before

        # Verify transformation: A_out = A_orig @ Z_out
        # But transformation only affects columns ifica-1:ifica-1+nca
        # Full column transformation: A_out = A_orig @ Z_out
        np.testing.assert_allclose(a_out, a_orig @ z_out, rtol=1e-13, atol=1e-14)

    def test_e_transformed_with_same_columns(self):
        """
        Verify E is transformed with same column rotations as A but different rows.

        E(k) = E(1:IFIRA-1, IFICA:na) gets same column transformations.
        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        n = 5
        nra = 2
        nca = 3
        ifira = 3  # So E has rows 1:2 (indices 0:2)
        ifica = 2

        a = np.random.randn(4, n).astype(float, order='F')
        e = np.random.randn(2, n).astype(float, order='F')  # ifira-1 = 2 rows
        z = np.eye(n, dtype=float, order='F')

        e_orig = e.copy()

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=True
        )

        assert info == 0

        # E should be transformed by same Z: E_out = E_orig @ Z_out
        np.testing.assert_allclose(e_out, e_orig @ z_out, rtol=1e-13)

    def test_z_accumulates_orthogonal_transformation(self):
        """
        Verify Z accumulates orthogonal column transformation.

        Starting from Z = I, final Z should be orthogonal (Z @ Z.T = I).
        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        n = 6
        nra = 4
        nca = 5
        ifira = 2
        ifica = 2

        a = np.random.randn(5, n).astype(float, order='F')
        e = np.random.randn(1, n).astype(float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=True
        )

        assert info == 0

        # Z should be orthogonal (use atol for machine precision zeros)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out.T @ z_out, np.eye(n), rtol=1e-14, atol=1e-14)

        # det(Z) should be +/- 1
        det_z = np.linalg.det(z_out)
        np.testing.assert_allclose(abs(det_z), 1.0, rtol=1e-14)


class TestMB04TVEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with n=0 returns immediately."""
        n = 0
        nra = 0
        nca = 0
        ifira = 1
        ifica = 1

        a = np.zeros((1, 1), dtype=float, order='F')
        e = np.zeros((0, 1), dtype=float, order='F')
        z = np.zeros((1, 1), dtype=float, order='F')

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=False
        )

        assert info == 0

    def test_nra_zero(self):
        """Test with nra=0 (no rows to transform)."""
        n = 3
        nra = 0
        nca = 2
        ifira = 1
        ifica = 1

        a = np.random.randn(2, n).astype(float, order='F')
        a_orig = a.copy()
        e = np.zeros((0, n), dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=True
        )

        assert info == 0
        # A should be unchanged
        np.testing.assert_array_equal(a_out, a_orig)
        # Z should be unchanged (identity)
        np.testing.assert_array_equal(z_out, np.eye(n))

    def test_nca_zero(self):
        """Test with nca=0 (no columns to transform)."""
        n = 3
        nra = 2
        nca = 0
        ifira = 1
        ifica = 1

        a = np.random.randn(2, n).astype(float, order='F')
        a_orig = a.copy()
        e = np.zeros((0, n), dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=True
        )

        assert info == 0
        np.testing.assert_array_equal(a_out, a_orig)
        np.testing.assert_array_equal(z_out, np.eye(n))

    def test_single_row(self):
        """Test with nra=1, nca=3: single row needs column elimination to make upper triangular."""
        n = 4
        nra = 1
        nca = 3
        ifira = 2
        ifica = 2

        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ], dtype=float, order='F')
        a_orig = a.copy()

        e = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float, order='F')  # 1 row
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=True
        )

        assert info == 0

        # nra=1 means only 1 row, but nca=3 means we eliminate cols ifica to ifica+nca-2
        # to make the single row "upper triangular" (zeros left of diagonal)
        # Row ifira (index 1), cols ifica-1 (index 1) and ifica (index 2) should be zero
        np.testing.assert_allclose(a_out[1, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(a_out[1, 2], 0.0, atol=1e-14)

        # Z should be orthogonal
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)

        # Verify transformation: A_out = A_orig @ Z_out
        np.testing.assert_allclose(a_out, a_orig @ z_out, rtol=1e-14, atol=1e-14)

    def test_updatz_false(self):
        """Test with updatz=False (Z not updated)."""
        np.random.seed(111)

        n = 4
        nra = 2
        nca = 3
        ifira = 1
        ifica = 1

        a = np.random.randn(2, n).astype(float, order='F')
        e = np.zeros((0, n), dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')
        z_orig = z.copy()

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=False
        )

        assert info == 0
        # Z should be unchanged when updatz=False
        np.testing.assert_array_equal(z_out, z_orig)


class TestMB04TVNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_ill_conditioned_submatrix(self):
        """
        Test with ill-conditioned submatrix to verify numerical stability.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)

        n = 5
        nra = 3
        nca = 4
        ifira = 2
        ifica = 2

        # Create ill-conditioned matrix
        u, _, vh = np.linalg.svd(np.random.randn(4, n))
        s = np.array([1.0, 1e-4, 1e-8, 1e-12, 1e-14])
        a = (u @ np.diag(s[:4]) @ vh[:4, :]).astype(float, order='F')

        e = np.random.randn(1, n).astype(float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_orig = a.copy()

        a_out, e_out, z_out, info = mb04tv(
            n, nra, nca, ifira, ifica, a, e, z, updatz=True
        )

        assert info == 0

        # Z should still be orthogonal (use atol for machine precision zeros)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-12, atol=1e-14)

        # Transformation should be consistent
        np.testing.assert_allclose(a_out, a_orig @ z_out, rtol=1e-10, atol=1e-14)
