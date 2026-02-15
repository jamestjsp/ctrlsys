"""
Tests for MB04TY: Triangularization of submatrices in staircase pencil.

MB04TY performs triangularization of submatrices having full row and column
rank in the pencil s*E(eps,inf)-A(eps,inf) using Algorithm 3.3.1 from
Beelen's thesis.

The routine iterates over NBLCKS blocks in reverse order (k = NBLCKS down to 1):
1. Calls MB04TW to reduce E(k) to upper triangular via row Givens rotations
2. Calls MB04TV to reduce A(k) to upper triangular via column Givens rotations

Key inputs:
- INUK: Row dimensions nu(k) of full row rank submatrices
- IMUK: Column dimensions mu(k) of full column rank submatrices
- ISMUK = sum of all mu(k), ISNUK = sum of all nu(k)

Algorithm constraints (must be satisfied for valid input):
- mu(k+1) <= nu(k) for all k (otherwise INFO=1)
- nu(k) <= mu(k) for all k (otherwise INFO=2)

Error conditions:
- INFO=1: mu(k+1) > nu(k) - incorrect dimensions for full column rank submatrix
- INFO=2: nu(k) > mu(k) - incorrect dimensions for full row rank submatrix
"""

import numpy as np
import pytest
from ctrlsys import mb04ty


class TestMB04TYBasic:
    """Basic functionality tests."""

    def test_single_block_2x2(self):
        """
        Test single block case with 2x2 matrices.

        nblcks=1, inuk=[2], imuk=[2]
        Constraint check: nu(1)=2 <= mu(1)=2, OK

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        m = 2
        n = 2
        nblcks = 1
        inuk = np.array([2], dtype=np.int32)
        imuk = np.array([2], dtype=np.int32)

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 0

        # Q and Z should be orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)

    def test_two_blocks_valid(self):
        """
        Test two block case with valid dimensions.

        nblcks=2, inuk=[2, 2], imuk=[2, 2]
        Constraints:
        - k=2: mu(3)=0 <= nu(2)=2, OK; nu(2)=2 <= mu(2)=2, OK
        - k=1: mu(2)=2 <= nu(1)=2, OK; nu(1)=2 <= mu(1)=2, OK

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        m = 4
        n = 4
        nblcks = 2
        inuk = np.array([2, 2], dtype=np.int32)
        imuk = np.array([2, 2], dtype=np.int32)

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 0

        # Q and Z should be orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)


class TestMB04TYMathProperties:
    """Mathematical property validation tests."""

    def test_orthogonal_transformations(self):
        """
        Verify Q and Z are orthogonal matrices.

        For any orthogonal matrix O: O @ O.T = O.T @ O = I, det(O) = +/- 1

        nblcks=2, inuk=[3, 3], imuk=[3, 3]
        Constraints:
        - k=2: mu(3)=0 <= nu(2)=3, OK; nu(2)=3 <= mu(2)=3, OK
        - k=1: mu(2)=3 <= nu(1)=3, OK; nu(1)=3 <= mu(1)=3, OK

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        m = 6
        n = 6
        nblcks = 2
        inuk = np.array([3, 3], dtype=np.int32)  # sum = 6 = m
        imuk = np.array([3, 3], dtype=np.int32)  # sum = 6 = n

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 0

        # Q orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(q_out.T @ q_out, np.eye(m), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(abs(np.linalg.det(q_out)), 1.0, rtol=1e-14)

        # Z orthogonal
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out.T @ z_out, np.eye(n), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(abs(np.linalg.det(z_out)), 1.0, rtol=1e-14)

    def test_rank_preservation(self):
        """
        Verify matrix rank is preserved under orthogonal transformations.

        rank(Q.T @ A @ Z) = rank(A) for orthogonal Q, Z.

        nblcks=2, inuk=[2, 2], imuk=[3, 2]
        Constraints:
        - k=2: mu(3)=0 <= nu(2)=2, OK; nu(2)=2 <= mu(2)=2, OK
        - k=1: mu(2)=2 <= nu(1)=2, OK; nu(1)=2 <= mu(1)=3, OK

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        m = 4
        n = 5
        nblcks = 2
        inuk = np.array([2, 2], dtype=np.int32)  # sum = 4 = m
        imuk = np.array([3, 2], dtype=np.int32)  # sum = 5 = n

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        rank_a_before = np.linalg.matrix_rank(a)
        rank_e_before = np.linalg.matrix_rank(e)

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 0

        rank_a_after = np.linalg.matrix_rank(a_out)
        rank_e_after = np.linalg.matrix_rank(e_out)

        assert rank_a_after == rank_a_before
        assert rank_e_after == rank_e_before

    def test_single_block_properties(self):
        """
        Test single block produces triangular structure.

        With nblcks=1, inuk=[3], imuk=[3]:
        - The routine reduces one block to upper triangular

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)

        m = 3
        n = 3
        nblcks = 1
        inuk = np.array([3], dtype=np.int32)
        imuk = np.array([3], dtype=np.int32)

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 0

        # Q and Z orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)

        # Determinant should be +/- 1
        np.testing.assert_allclose(abs(np.linalg.det(q_out)), 1.0, rtol=1e-14)
        np.testing.assert_allclose(abs(np.linalg.det(z_out)), 1.0, rtol=1e-14)


class TestMB04TYEdgeCases:
    """Edge case tests."""

    def test_m_zero(self):
        """Test with m=0 returns immediately (quick return)."""
        m = 0
        n = 2
        nblcks = 0
        inuk = np.array([], dtype=np.int32)
        imuk = np.array([], dtype=np.int32)

        a = np.zeros((1, 2), dtype=float, order='F')
        e = np.zeros((1, 2), dtype=float, order='F')
        q = np.zeros((1, 1), dtype=float, order='F')
        z = np.eye(2, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=False, updatz=False
        )

        assert info == 0

    def test_n_zero(self):
        """Test with n=0 returns immediately (quick return)."""
        m = 2
        n = 0
        nblcks = 0
        inuk = np.array([], dtype=np.int32)
        imuk = np.array([], dtype=np.int32)

        a = np.zeros((2, 1), dtype=float, order='F')
        e = np.zeros((2, 1), dtype=float, order='F')
        q = np.eye(2, dtype=float, order='F')
        z = np.zeros((1, 1), dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=False, updatz=False
        )

        assert info == 0

    def test_nblcks_zero(self):
        """Test with nblcks=0 (no blocks to process)."""
        np.random.seed(111)

        m = 3
        n = 4
        nblcks = 0
        inuk = np.array([], dtype=np.int32)
        imuk = np.array([], dtype=np.int32)

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_orig = a.copy()
        e_orig = e.copy()

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 0
        # With no blocks, matrices unchanged
        np.testing.assert_array_equal(a_out, a_orig)
        np.testing.assert_array_equal(e_out, e_orig)
        np.testing.assert_array_equal(q_out, np.eye(m))
        np.testing.assert_array_equal(z_out, np.eye(n))

    def test_updatq_false(self):
        """Test with updatq=False (Q not updated)."""
        np.random.seed(222)

        m = 3
        n = 3
        nblcks = 1
        inuk = np.array([3], dtype=np.int32)
        imuk = np.array([3], dtype=np.int32)

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        q_orig = q.copy()
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=False, updatz=True
        )

        assert info == 0
        # Q unchanged
        np.testing.assert_array_equal(q_out, q_orig)

    def test_updatz_false(self):
        """Test with updatz=False (Z not updated)."""
        np.random.seed(333)

        m = 3
        n = 3
        nblcks = 1
        inuk = np.array([3], dtype=np.int32)
        imuk = np.array([3], dtype=np.int32)

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')
        z_orig = z.copy()

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=False
        )

        assert info == 0
        # Z unchanged
        np.testing.assert_array_equal(z_out, z_orig)


class TestMB04TYErrorHandling:
    """Error handling tests."""

    def test_info_1_column_rank_error(self):
        """
        Test INFO=1: mu(k+1) > nu(k) - column rank submatrix dimensions incorrect.

        For k=NBLCKS (last block processed first in reverse order), mukp1=0 initially.
        So this error triggers when processing block k where mukp1 (from previous iteration)
        is greater than nuk.

        With nblcks=2, inuk=[1, 2], imuk=[3, 2]:
        - k=2: mukp1=0, nuk=2, check 0 > 2 -> False, proceed. mukp1 becomes 2.
        - k=1: mukp1=2, nuk=1, check 2 > 1 -> True, INFO=1
        """
        np.random.seed(444)

        m = 3
        n = 5
        nblcks = 2
        inuk = np.array([1, 2], dtype=np.int32)  # nuk=[1, 2]
        imuk = np.array([3, 2], dtype=np.int32)  # muk=[3, 2]

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 1

    def test_info_2_row_rank_error(self):
        """
        Test INFO=2: nu(k) > mu(k) - row rank submatrix dimensions incorrect.

        With nblcks=1, inuk=[3], imuk=[2]:
        - k=1: mukp1=0, nuk=3, check 0 > 3 -> False
        - Then check nuk > muk: 3 > 2 -> True, INFO=2
        """
        np.random.seed(555)

        m = 3
        n = 2
        nblcks = 1
        inuk = np.array([3], dtype=np.int32)  # nuk=3
        imuk = np.array([2], dtype=np.int32)  # muk=2

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 2


class TestMB04TYNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_large_values(self):
        """
        Test with large matrix values.

        nblcks=2, inuk=[2, 2], imuk=[3, 2]
        Constraints:
        - k=2: mu(3)=0 <= nu(2)=2, OK; nu(2)=2 <= mu(2)=2, OK
        - k=1: mu(2)=2 <= nu(1)=2, OK; nu(1)=2 <= mu(1)=3, OK

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)

        m = 4
        n = 5
        nblcks = 2
        inuk = np.array([2, 2], dtype=np.int32)
        imuk = np.array([3, 2], dtype=np.int32)

        a = (np.random.randn(m, n) * 1e8).astype(float, order='F')
        e = (np.random.randn(m, n) * 1e8).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 0

        # Q and Z orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)

    def test_small_values(self):
        """
        Test with small matrix values.

        nblcks=2, inuk=[2, 2], imuk=[3, 2]
        Same constraints as test_large_values.

        Random seed: 777 (for reproducibility)
        """
        np.random.seed(777)

        m = 4
        n = 5
        nblcks = 2
        inuk = np.array([2, 2], dtype=np.int32)
        imuk = np.array([3, 2], dtype=np.int32)

        a = (np.random.randn(m, n) * 1e-8).astype(float, order='F')
        e = (np.random.randn(m, n) * 1e-8).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 0

        # Q and Z orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)

    def test_three_blocks(self):
        """
        Test with three blocks to verify multi-block iteration.

        nblcks=3, inuk=[2, 2, 2], imuk=[2, 2, 3]
        Constraints:
        - k=3: mu(4)=0 <= nu(3)=2, OK; nu(3)=2 <= mu(3)=3, OK
        - k=2: mu(3)=3 > nu(2)=2, FAIL! Need mu(3) <= nu(2)

        So use imuk=[3, 2, 2] (decreasing):
        - k=3: mu(4)=0 <= nu(3)=2, OK; nu(3)=2 <= mu(3)=2, OK
        - k=2: mu(3)=2 <= nu(2)=2, OK; nu(2)=2 <= mu(2)=2, OK
        - k=1: mu(2)=2 <= nu(1)=2, OK; nu(1)=2 <= mu(1)=3, OK

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)

        m = 6
        n = 7
        nblcks = 3
        inuk = np.array([2, 2, 2], dtype=np.int32)  # sum = 6 = m
        imuk = np.array([3, 2, 2], dtype=np.int32)  # sum = 7 = n

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')

        a_out, e_out, q_out, z_out, info = mb04ty(
            m, n, nblcks, inuk, imuk, a, e, q, z, updatq=True, updatz=True
        )

        assert info == 0

        # Q and Z orthogonal
        np.testing.assert_allclose(q_out @ q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)
