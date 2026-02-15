"""
Tests for MB04TW: Reduce submatrix E(k) to upper triangular form via row Givens rotations.

MB04TW reduces E(IFIRE:me, IFICE:ne) to upper triangular by applying row Givens
rotations. The same rotations are applied to A(IFIRE:me, IFICA:N).

Key relationships:
- E(k) = E(IFIRE:me, IFICE:ne) where me = IFIRE-1+NRE, ne = IFICE-1+NCE
- A(k) = A(IFIRE:me, IFICA:N) - different column range than E!
- If UPDATQ, Q accumulates row transformations

Algorithm: Left-to-right column processing with top-down row elimination.
Difference from MB04TV: MB04TV uses column rotations, MB04TW uses row rotations.
"""

import numpy as np
import pytest
from ctrlsys import mb04tw


class TestMB04TWBasic:
    """Basic functionality tests."""

    def test_2x2_to_upper_triangular(self):
        """
        Test basic 2x2 submatrix reduction to upper triangular.

        E(k) is 2x2 starting at (1,1) (1-based):
            [[1, 2],
             [3, 4]]

        After reduction, E(k) should be upper triangular (zeros below diagonal).
        """
        m = 2
        n = 2
        nre = 2
        nce = 2
        ifire = 1  # 1-based
        ifice = 1  # 1-based
        ifica = 1  # 1-based

        e = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=float, order='F')

        a = np.array([
            [5.0, 6.0],
            [7.0, 8.0],
        ], dtype=float, order='F')

        q = np.eye(m, dtype=float, order='F')

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0

        # E(k) should be upper triangular: E[1,0] = 0 (below diagonal)
        np.testing.assert_allclose(e_out[1, 0], 0.0, atol=1e-14)

        # Q should be orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)

    def test_3x2_submatrix_reduction(self):
        """
        Test 3x2 submatrix E(k) reduction.

        E(k) = E(1:3, 1:2) is 3x2, requires eliminating E[1,0] and E[2,0], E[2,1].

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        m = 3
        n = 4
        nre = 3
        nce = 2
        ifire = 1
        ifice = 1
        ifica = 1

        e = np.random.randn(m, n).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')

        e_orig = e.copy()
        a_orig = a.copy()

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0

        # E(k) should be upper triangular in submatrix E(0:3, 0:2)
        # E[1,0], E[2,0], E[2,1] should be zero
        np.testing.assert_allclose(e_out[1, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(e_out[2, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(e_out[2, 1], 0.0, atol=1e-14)

        # Q orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)


class TestMB04TWMathProperties:
    """Mathematical property validation tests."""

    def test_row_transformation_preserves_rank(self):
        """
        Verify row Givens rotations preserve matrix rank.

        The transformation E' = Q @ E should preserve rank since Q is orthogonal.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        m = 5
        n = 4
        nre = 3
        nce = 3
        ifire = 2  # 1-based
        ifice = 1
        ifica = 1

        e = np.random.randn(m, n).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')

        e_orig = e.copy()
        rank_before = np.linalg.matrix_rank(e_orig)

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0

        rank_after = np.linalg.matrix_rank(e_out)
        assert rank_after == rank_before

        # Verify transformation: E_out = Q_out.T @ E_orig
        np.testing.assert_allclose(e_out, q_out.T @ e_orig, rtol=1e-13, atol=1e-14)

    def test_a_transformed_with_same_rows(self):
        """
        Verify A is transformed with same row rotations as E.

        When ifica=ifice (same starting column), the entire transformation
        is applied consistently to both A and E, so A_out = Q_out.T @ A_orig.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        m = 4
        n = 6
        nre = 2
        nce = 3
        ifire = 2  # So E(k) starts at row 2 (1-based), 1 (0-based)
        ifice = 1  # E columns 1:3 (1-based)
        ifica = 1  # A columns 1:6 (1-based), same start as E

        e = np.random.randn(m, n).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')

        a_orig = a.copy()

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0

        # A should be transformed by same Q: A_out = Q_out.T @ A_orig
        np.testing.assert_allclose(a_out, q_out.T @ a_orig, rtol=1e-13, atol=1e-14)

    def test_q_accumulates_orthogonal_transformation(self):
        """
        Verify Q accumulates orthogonal row transformation.

        Starting from Q = I, final Q should be orthogonal (Q @ Q.T = I).
        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        m = 6
        n = 5
        nre = 4
        nce = 3
        ifire = 2
        ifice = 1
        ifica = 1

        e = np.random.randn(m, n).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0

        # Q should be orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(q_out.T @ q_out, np.eye(m), rtol=1e-14, atol=1e-14)

        # det(Q) should be +/- 1
        det_q = np.linalg.det(q_out)
        np.testing.assert_allclose(abs(det_q), 1.0, rtol=1e-14)


class TestMB04TWEdgeCases:
    """Edge case tests."""

    def test_m_zero(self):
        """Test with m=0 returns immediately."""
        m = 0
        n = 0
        nre = 0
        nce = 0
        ifire = 1
        ifice = 1
        ifica = 1

        a = np.zeros((1, 1), dtype=float, order='F')
        e = np.zeros((1, 1), dtype=float, order='F')
        q = np.zeros((1, 1), dtype=float, order='F')

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=False
        )

        assert info == 0

    def test_nre_zero(self):
        """Test with nre=0 (no rows to transform)."""
        np.random.seed(111)

        m = 3
        n = 4
        nre = 0
        nce = 2
        ifire = 1
        ifice = 1
        ifica = 1

        e = np.random.randn(m, n).astype(float, order='F')
        e_orig = e.copy()
        a = np.random.randn(m, n).astype(float, order='F')
        a_orig = a.copy()
        q = np.eye(m, dtype=float, order='F')

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0
        np.testing.assert_array_equal(e_out, e_orig)
        np.testing.assert_array_equal(a_out, a_orig)
        np.testing.assert_array_equal(q_out, np.eye(m))

    def test_nce_zero(self):
        """Test with nce=0 (no columns to transform)."""
        np.random.seed(222)

        m = 3
        n = 4
        nre = 2
        nce = 0
        ifire = 1
        ifice = 1
        ifica = 1

        e = np.random.randn(m, n).astype(float, order='F')
        e_orig = e.copy()
        a = np.random.randn(m, n).astype(float, order='F')
        a_orig = a.copy()
        q = np.eye(m, dtype=float, order='F')

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0
        np.testing.assert_array_equal(e_out, e_orig)
        np.testing.assert_array_equal(a_out, a_orig)
        np.testing.assert_array_equal(q_out, np.eye(m))

    def test_single_column(self):
        """Test with nce=1: single column elimination."""
        m = 3
        n = 4
        nre = 3
        nce = 1
        ifire = 1
        ifice = 1
        ifica = 1

        e = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ], dtype=float, order='F')
        e_orig = e.copy()

        a = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ], dtype=float, order='F')

        q = np.eye(m, dtype=float, order='F')

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0

        # nce=1 means only first column. E[1,0] and E[2,0] should be zero
        np.testing.assert_allclose(e_out[1, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(e_out[2, 0], 0.0, atol=1e-14)

        # Q orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)

        # Verify transformation
        np.testing.assert_allclose(e_out, q_out.T @ e_orig, rtol=1e-14, atol=1e-14)

    def test_updatq_false(self):
        """Test with updatq=False (Q not updated)."""
        np.random.seed(333)

        m = 3
        n = 4
        nre = 2
        nce = 2
        ifire = 1
        ifice = 1
        ifica = 1

        e = np.random.randn(m, n).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        q_orig = q.copy()

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=False
        )

        assert info == 0
        # Q should be unchanged when updatq=False
        np.testing.assert_array_equal(q_out, q_orig)

    def test_offset_submatrix(self):
        """
        Test with offset submatrix: E(k) not starting at (1,1).

        E(k) = E(2:3, 2:4) (1-based), i.e., rows 1:2, cols 1:3 (0-based)

        Note: The Fortran routine only applies row rotations to columns from
        the current pivot column to N, so columns before ifice are NOT transformed.
        Therefore we can only verify the transformation on columns ifice:n.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)

        m = 4
        n = 5
        nre = 2
        nce = 3
        ifire = 2  # 1-based
        ifice = 2  # 1-based (0-based: column 1)
        ifica = 2  # 1-based, same as ifice for full A transformation

        e = np.random.randn(m, n).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')

        e_orig = e.copy()

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0

        # E(k) = E[1:3, 1:4] (0-based) should be upper triangular
        # E[2,1] should be zero (row 2, col 1, which is below diagonal in submatrix)
        np.testing.assert_allclose(e_out[2, 1], 0.0, atol=1e-14)

        # Q orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)

        # Columns before ifice (column 0) are not transformed
        np.testing.assert_array_equal(e_out[:, 0], e_orig[:, 0])


class TestMB04TWNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_large_values(self):
        """
        Test with large matrix values.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)

        m = 4
        n = 5
        nre = 3
        nce = 3
        ifire = 1
        ifice = 1
        ifica = 1

        e = (np.random.randn(m, n) * 1e8).astype(float, order='F')
        a = (np.random.randn(m, n) * 1e8).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')

        e_orig = e.copy()

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0

        # Q orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)

        # Transformation consistent
        np.testing.assert_allclose(e_out, q_out.T @ e_orig, rtol=1e-12, atol=1e-6)

    def test_small_values(self):
        """
        Test with small matrix values.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)

        m = 4
        n = 5
        nre = 3
        nce = 3
        ifire = 1
        ifice = 1
        ifica = 1

        e = (np.random.randn(m, n) * 1e-8).astype(float, order='F')
        a = (np.random.randn(m, n) * 1e-8).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')

        e_orig = e.copy()

        a_out, e_out, q_out, info = mb04tw(
            m, n, nre, nce, ifire, ifice, ifica, a, e, q, updatq=True
        )

        assert info == 0

        # Q orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)

        # Transformation consistent
        np.testing.assert_allclose(e_out, q_out.T @ e_orig, rtol=1e-12, atol=1e-20)
