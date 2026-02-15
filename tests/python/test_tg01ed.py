"""
Tests for TG01ED: Orthogonal reduction of descriptor system to SVD coordinate form.

TG01ED computes orthogonal transformation matrices Q and Z such that the
transformed system (Q'*A*Z - lambda Q'*E*Z, Q'*B, C*Z) is in SVD coordinate form
with:
           ( A11  A12 )             ( Er  0 )
  Q'*A*Z = (          ) ,  Q'*E*Z = (       )
           ( A21  A22 )             (  0  0 )

where Er is invertible diagonal with decreasingly ordered nonzero singular values of E.

Optionally (JOBA='R'), A22 can be further reduced to SVD form:
          ( Ar  0 )
    A22 = (       )
          (  0  0 )
"""

import numpy as np
import pytest
from ctrlsys import tg01ed


class TestTG01EDBasic:
    """Basic functionality tests from HTML doc example."""

    def test_html_example(self):
        """
        Test TG01ED with HTML documentation example.

        L=4, N=4, M=2, P=2, JOBA='R', TOL=0.0
        """
        l, n, m, p = 4, 4, 2, 2

        # Input matrices from HTML doc (row-wise read)
        a = np.array([
            [-1.0,  0.0,  0.0,  3.0],
            [ 0.0,  0.0,  1.0,  2.0],
            [ 1.0,  1.0,  0.0,  4.0],
            [ 0.0,  0.0,  0.0,  0.0]
        ], order='F', dtype=float)

        e = np.array([
            [ 1.0,  2.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  1.0],
            [ 3.0,  9.0,  6.0,  3.0],
            [ 0.0,  0.0,  2.0,  0.0]
        ], order='F', dtype=float)

        b = np.array([
            [ 1.0,  0.0],
            [ 0.0,  0.0],
            [ 0.0,  1.0],
            [ 1.0,  1.0]
        ], order='F', dtype=float)

        c = np.array([
            [-1.0,  0.0,  1.0,  0.0],
            [ 0.0,  1.0, -1.0,  1.0]
        ], order='F', dtype=float)

        # Expected outputs from HTML doc
        # Rank of matrix E = 3
        # Rank of matrix A22 = 1
        a_expected = np.array([
            [ 2.1882, -0.8664, -3.5097, -2.1353],
            [-0.4569, -0.2146,  1.9802,  0.3531],
            [-0.5717, -0.5245, -0.4591,  0.4696],
            [-0.4766, -0.5846,  2.1414,  0.3086]
        ], order='F', dtype=float)

        e_expected = np.array([
            [11.8494,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  2.1302,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  1.0270,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000]
        ], order='F', dtype=float)

        b_expected = np.array([
            [-0.2396, -1.0668],
            [-0.2656, -0.8393],
            [-0.7657, -0.1213],
            [ 1.1339,  0.3780]
        ], order='F', dtype=float)

        c_expected = np.array([
            [-0.2499, -1.0573,  0.3912, -0.8165],
            [-0.5225,  1.3958,  0.8825,  0.0000]
        ], order='F', dtype=float)

        q_expected = np.array([
            [-0.1534,  0.5377, -0.6049,  0.5669],
            [-0.0872,  0.2536,  0.7789,  0.5669],
            [-0.9805, -0.0360,  0.0395, -0.1890],
            [-0.0863, -0.8033, -0.1608,  0.5669]
        ], order='F', dtype=float)

        z_expected = np.array([
            [-0.2612,  0.2017, -0.4737,  0.8165],
            [-0.7780,  0.4718, -0.0738, -0.4082],
            [-0.5111, -0.8556, -0.0826,  0.0000],
            [-0.2556,  0.0684,  0.8737,  0.4082]
        ], order='F', dtype=float)

        # Call routine with JOBA='R' to reduce A22
        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0
        assert ranke == 3
        assert rnka22 == 1
        np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(e_out, e_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(np.abs(q_out), np.abs(q_expected), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(np.abs(z_out), np.abs(z_expected), rtol=1e-3, atol=1e-4)


class TestTG01EDOrthogonality:
    """Test mathematical properties - orthogonality of Q and Z."""

    def test_q_z_orthogonal(self):
        """
        Verify Q and Z are orthogonal: Q'*Q = I, Z'*Z = I

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        l, n, m, p = 5, 5, 2, 3

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0
        identity_l = np.eye(l, dtype=float, order='F')
        identity_n = np.eye(n, dtype=float, order='F')
        np.testing.assert_allclose(q_out.T @ q_out, identity_l, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out.T @ z_out, identity_n, rtol=1e-14, atol=1e-14)

    def test_transformation_consistency(self):
        """
        Verify transformations:
          A_out = Q'*A*Z, E_out = Q'*E*Z, B_out = Q'*B, C_out = C*Z

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        l, n, m, p = 4, 6, 2, 3

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_orig = a.copy()
        e_orig = e.copy()
        b_orig = b.copy()
        c_orig = c.copy()

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0

        # Check Q'*A*Z = A_out
        np.testing.assert_allclose(q_out.T @ a_orig @ z_out, a_out, rtol=1e-13, atol=1e-14)
        # Check Q'*E*Z = E_out
        np.testing.assert_allclose(q_out.T @ e_orig @ z_out, e_out, rtol=1e-13, atol=1e-14)
        # Check Q'*B = B_out
        np.testing.assert_allclose(q_out.T @ b_orig, b_out, rtol=1e-13, atol=1e-14)
        # Check C*Z = C_out
        np.testing.assert_allclose(c_orig @ z_out, c_out, rtol=1e-13, atol=1e-14)

    def test_e_diagonal_form(self):
        """
        Verify E is transformed to diagonal form with decreasingly ordered singular values.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        l, n, m, p = 5, 5, 2, 3

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0

        # E_out should be diagonal in the upper-left ranke x ranke block
        # and zero elsewhere
        for i in range(l):
            for j in range(n):
                if i == j and i < ranke:
                    # Diagonal elements should be positive and decreasing
                    assert e_out[i, j] > 0
                    if i > 0:
                        assert e_out[i, j] <= e_out[i-1, j-1] + 1e-14
                else:
                    # Off-diagonal should be zero
                    assert abs(e_out[i, j]) < 1e-13, f"E[{i},{j}] = {e_out[i,j]} should be 0"

    def test_singular_value_preservation(self):
        """
        Verify singular values of E are preserved.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        l, n, m, p = 5, 4, 2, 3

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        # Compute SVD of original E
        sv_original = np.linalg.svd(e, compute_uv=False)

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0

        # Extract diagonal of E_out (singular values)
        min_ln = min(l, n)
        sv_transformed = np.array([e_out[i, i] for i in range(min_ln)])

        # Singular values should match
        np.testing.assert_allclose(sv_transformed, sv_original[:min_ln], rtol=1e-13, atol=1e-14)


class TestTG01EDJobaModes:
    """Test different JOBA modes."""

    def test_joba_n_no_reduce(self):
        """
        Test JOBA='N': do not reduce A22.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        l, n, m, p = 4, 4, 2, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'N', a, e, b, c, 0.0
        )

        assert info == 0
        # rnka22 should be 0 or undefined when JOBA='N'
        # E should still be in SVD form

    def test_joba_r_reduce_a22(self):
        """
        Test JOBA='R': reduce A22 to SVD form.

        With JOBA='R', the A22 submatrix (lower-right (L-RANKE) x (N-RANKE) block)
        should have the form:
              ( Ar  0 )
        A22 = (       )
              (  0  0 )

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        l, n, m, p = 5, 5, 2, 3

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0

        # A22 block starts at (ranke, ranke)
        # It should be diagonal in the first rnka22 positions
        la22 = l - ranke
        na22 = n - ranke
        ln2 = min(la22, na22)

        for i in range(la22):
            for j in range(na22):
                row = ranke + i
                col = ranke + j
                if i == j and i < rnka22:
                    # Diagonal elements of Ar
                    assert a_out[row, col] > 0
                elif i >= rnka22 or j >= rnka22:
                    # Zero block
                    if i < ln2 and j < ln2:
                        # Within the reduced SVD block
                        if i != j:
                            assert abs(a_out[row, col]) < 1e-12, \
                                f"A22[{i},{j}] = {a_out[row,col]} should be 0"


class TestTG01EDEdgeCases:
    """Edge case tests."""

    def test_l_zero(self):
        """Test with L=0."""
        a = np.array([], dtype=float, order='F').reshape(0, 4)
        e = np.array([], dtype=float, order='F').reshape(0, 4)
        b = np.array([], dtype=float, order='F').reshape(0, 2)
        c = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float, order='F')

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0
        assert ranke == 0
        assert rnka22 == 0

    def test_n_zero(self):
        """Test with N=0."""
        a = np.array([], dtype=float, order='F').reshape(4, 0)
        e = np.array([], dtype=float, order='F').reshape(4, 0)
        b = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float, order='F')
        c = np.array([], dtype=float, order='F').reshape(2, 0)

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0
        assert ranke == 0
        assert rnka22 == 0

    def test_m_zero(self):
        """
        Test with M=0 (no B matrix columns).

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        l, n, p = 4, 4, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.array([], dtype=float, order='F').reshape(l, 0)
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0
        # Verify Q is orthogonal
        identity = np.eye(l, dtype=float, order='F')
        np.testing.assert_allclose(q_out.T @ q_out, identity, rtol=1e-14, atol=1e-14)

    def test_p_zero(self):
        """
        Test with P=0 (no C matrix rows).

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        l, n, m = 4, 4, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.array([], dtype=float, order='F').reshape(0, n)

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0
        # Verify Z is orthogonal
        identity = np.eye(n, dtype=float, order='F')
        np.testing.assert_allclose(z_out.T @ z_out, identity, rtol=1e-14, atol=1e-14)

    def test_rectangular_l_gt_n(self):
        """
        Test rectangular case L > N.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        l, n, m, p = 6, 4, 2, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_orig = a.copy()
        e_orig = e.copy()

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0

        # Verify transformation consistency
        np.testing.assert_allclose(q_out.T @ a_orig @ z_out, a_out, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(q_out.T @ e_orig @ z_out, e_out, rtol=1e-13, atol=1e-14)

    def test_rectangular_l_lt_n(self):
        """
        Test rectangular case L < N.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        l, n, m, p = 4, 6, 2, 2

        a = np.random.randn(l, n).astype(float, order='F')
        e = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_orig = a.copy()
        e_orig = e.copy()

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0

        # Verify transformation consistency
        np.testing.assert_allclose(q_out.T @ a_orig @ z_out, a_out, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(q_out.T @ e_orig @ z_out, e_out, rtol=1e-13, atol=1e-14)


class TestTG01EDRankDeficient:
    """Test rank-deficient cases."""

    def test_rank_deficient_e(self):
        """
        Test with rank-deficient E matrix.

        Random seed: 777 (for reproducibility)
        """
        np.random.seed(777)
        l, n, m, p = 5, 5, 2, 2

        # Create rank-deficient E (rank 3)
        u = np.random.randn(l, 3)
        v = np.random.randn(n, 3)
        e = (u @ v.T).astype(float, order='F')

        a = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0
        assert ranke <= 3  # Should detect rank deficiency

    def test_zero_e(self):
        """
        Test with zero E matrix.

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        l, n, m, p = 4, 4, 2, 2

        e = np.zeros((l, n), dtype=float, order='F')
        a = np.random.randn(l, n).astype(float, order='F')
        b = np.random.randn(l, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q_out, z_out, ranke, rnka22, info = tg01ed(
            'R', a, e, b, c, 0.0
        )

        assert info == 0
        assert ranke == 0  # E has zero rank


class TestTG01EDErrors:
    """Error handling tests."""

    def test_invalid_joba(self):
        """Test invalid JOBA parameter."""
        a = np.eye(4, dtype=float, order='F')
        e = np.eye(4, dtype=float, order='F')
        b = np.eye(4, 2, dtype=float, order='F')
        c = np.eye(2, 4, dtype=float, order='F')

        with pytest.raises((ValueError, RuntimeError)):
            tg01ed('X', a, e, b, c, 0.0)

    def test_invalid_tol(self):
        """Test TOL >= 1.0 is invalid."""
        a = np.eye(4, dtype=float, order='F')
        e = np.eye(4, dtype=float, order='F')
        b = np.eye(4, 2, dtype=float, order='F')
        c = np.eye(2, 4, dtype=float, order='F')

        with pytest.raises((ValueError, RuntimeError)):
            tg01ed('R', a, e, b, c, 1.0)
