"""
Tests for MB03DD - Exchanging eigenvalues of a real 2-by-2, 3-by-3 or 4-by-4
block upper triangular pencil.

MB03DD computes orthogonal matrices Q1 and Q2 for a real 2-by-2, 3-by-3, or
4-by-4 regular block upper triangular pencil such that the eigenvalues in
Spec(A11, B11) and Spec(A22, B22) are exchanged.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import mb03dd


class TestMB03DDBasic:
    """Basic functionality tests for mb03dd."""

    def test_2x2_upper_block_triangular(self):
        """
        Test 2x2 case with UPLO='U' - eigenvalue exchange.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n1, n2 = 1, 1
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 1.0],
                      [0.0, 3.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5],
                      [0.0, 2.0]], order='F', dtype=float)

        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('U', n1, n2, prec, a, b)

        assert info == 0
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)

        assert_allclose(q1 @ q1.T, np.eye(m), rtol=1e-14, atol=1e-14)
        assert_allclose(q2 @ q2.T, np.eye(m), rtol=1e-14, atol=1e-14)

        eig_orig = np.linalg.eigvals(a_orig @ np.linalg.inv(b_orig))
        eig_trans = np.linalg.eigvals(a_out @ np.linalg.inv(b_out))
        assert_allclose(sorted(eig_orig.real), sorted(eig_trans.real), rtol=1e-10)

    def test_2x2_lower_block_triangular(self):
        """
        Test 2x2 case with UPLO='L' - lower triangular to upper.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n1, n2 = 1, 1
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 0.0],
                      [0.5, 3.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.0],
                      [0.3, 2.0]], order='F', dtype=float)

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('L', n1, n2, prec, a, b)

        assert info == 0
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)

        assert_allclose(q1 @ q1.T, np.eye(m), rtol=1e-14, atol=1e-14)
        assert_allclose(q2 @ q2.T, np.eye(m), rtol=1e-14, atol=1e-14)

    def test_3x3_n1_1_n2_2_upper(self):
        """
        Test 3x3 case with n1=1, n2=2, UPLO='U'.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n1, n2 = 1, 2
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 1.0, 0.5],
                      [0.0, 3.0, 1.0],
                      [0.0, 0.2, 4.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5, 0.3],
                      [0.0, 2.0, 0.5],
                      [0.0, 0.1, 1.5]], order='F', dtype=float)

        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('U', n1, n2, prec, a, b)

        assert info == 0
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)

        assert_allclose(q1 @ q1.T, np.eye(m), rtol=1e-14, atol=1e-14)
        assert_allclose(q2 @ q2.T, np.eye(m), rtol=1e-14, atol=1e-14)

        eig_orig = np.linalg.eigvals(a_orig @ np.linalg.inv(b_orig))
        eig_trans = np.linalg.eigvals(a_out @ np.linalg.inv(b_out))
        assert_allclose(sorted(eig_orig.real), sorted(eig_trans.real), rtol=1e-10)

    def test_3x3_n1_2_n2_1_upper(self):
        """
        Test 3x3 case with n1=2, n2=1, UPLO='U'.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n1, n2 = 2, 1
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 1.0, 0.5],
                      [0.3, 3.0, 1.0],
                      [0.0, 0.0, 4.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5, 0.3],
                      [0.2, 2.0, 0.5],
                      [0.0, 0.0, 1.5]], order='F', dtype=float)

        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('U', n1, n2, prec, a, b)

        assert info == 0
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)

        assert_allclose(q1 @ q1.T, np.eye(m), rtol=1e-14, atol=1e-14)
        assert_allclose(q2 @ q2.T, np.eye(m), rtol=1e-14, atol=1e-14)

        eig_orig = np.linalg.eigvals(a_orig @ np.linalg.inv(b_orig))
        eig_trans = np.linalg.eigvals(a_out @ np.linalg.inv(b_out))
        assert_allclose(sorted(eig_orig.real), sorted(eig_trans.real), rtol=1e-10)

    def test_4x4_n1_2_n2_2_upper(self):
        """
        Test 4x4 case with n1=2, n2=2, UPLO='U'.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        n1, n2 = 2, 2
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 1.0, 0.5, 0.3],
                      [0.3, 3.0, 1.0, 0.5],
                      [0.0, 0.0, 4.0, 1.0],
                      [0.0, 0.0, 0.2, 5.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5, 0.3, 0.2],
                      [0.2, 2.0, 0.5, 0.3],
                      [0.0, 0.0, 1.5, 0.5],
                      [0.0, 0.0, 0.1, 2.5]], order='F', dtype=float)

        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('U', n1, n2, prec, a, b)

        assert info == 0
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)

        assert_allclose(q1 @ q1.T, np.eye(m), rtol=1e-14, atol=1e-14)
        assert_allclose(q2 @ q2.T, np.eye(m), rtol=1e-14, atol=1e-14)

        eig_orig = np.linalg.eigvals(a_orig @ np.linalg.inv(b_orig))
        eig_trans = np.linalg.eigvals(a_out @ np.linalg.inv(b_out))
        assert_allclose(sorted(eig_orig.real), sorted(eig_trans.real), rtol=1e-10)


class TestMB03DDTransformationProperty:
    """Test mathematical properties of the transformation."""

    def test_transformation_consistency_upper(self):
        """
        Verify Q2' * A * Q1 gives the transformed A for UPLO='U'.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n1, n2 = 2, 1
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 1.0, 0.5],
                      [0.3, 3.0, 1.0],
                      [0.0, 0.0, 4.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5, 0.3],
                      [0.2, 2.0, 0.5],
                      [0.0, 0.0, 1.5]], order='F', dtype=float)

        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('U', n1, n2, prec, a, b)

        assert info == 0

        a_transformed = q2.T @ a_orig @ q1
        b_transformed = q2.T @ b_orig @ q1

        assert_allclose(a_out, a_transformed, rtol=1e-12, atol=1e-12)
        assert_allclose(b_out, b_transformed, rtol=1e-12, atol=1e-12)

    def test_b_upper_triangular(self):
        """
        Verify B is upper triangular after transformation.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n1, n2 = 2, 2
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 1.0, 0.5, 0.3],
                      [0.3, 3.0, 1.0, 0.5],
                      [0.0, 0.0, 4.0, 1.0],
                      [0.0, 0.0, 0.2, 5.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5, 0.3, 0.2],
                      [0.2, 2.0, 0.5, 0.3],
                      [0.0, 0.0, 1.5, 0.5],
                      [0.0, 0.0, 0.1, 2.5]], order='F', dtype=float)

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('U', n1, n2, prec, a, b)

        assert info == 0

        b_lower = np.tril(b_out, -1)
        assert_allclose(b_lower, np.zeros((m, m)), atol=1e-12)


class TestMB03DDTriangularMode:
    """Test triangular mode (UPLO='T')."""

    def test_3x3_triangular_mode(self):
        """
        Test 3x3 case with UPLO='T' (B is already triangular).

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n1, n2 = 1, 2
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 1.0, 0.5],
                      [0.0, 3.0, 1.0],
                      [0.0, 0.2, 4.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5, 0.3],
                      [0.0, 2.0, 0.5],
                      [0.0, 0.0, 1.5]], order='F', dtype=float)

        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('T', n1, n2, prec, a, b)

        assert info == 0
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)

        assert_allclose(q1 @ q1.T, np.eye(m), rtol=1e-14, atol=1e-14)
        assert_allclose(q2 @ q2.T, np.eye(m), rtol=1e-14, atol=1e-14)

        eig_orig = np.linalg.eigvals(a_orig @ np.linalg.inv(b_orig))
        eig_trans = np.linalg.eigvals(a_out @ np.linalg.inv(b_out))
        assert_allclose(sorted(eig_orig.real), sorted(eig_trans.real), rtol=1e-10)


class TestMB03DDLowerMode:
    """Test lower triangular mode (UPLO='L')."""

    def test_3x3_lower_mode(self):
        """
        Test 3x3 case with UPLO='L' (lower block triangular).

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n1, n2 = 1, 2
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 0.0, 0.0],
                      [1.0, 3.0, 1.0],
                      [0.5, 0.2, 4.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.0, 0.0],
                      [0.5, 2.0, 0.5],
                      [0.3, 0.1, 1.5]], order='F', dtype=float)

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('L', n1, n2, prec, a, b)

        assert info == 0
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)

        assert_allclose(q1 @ q1.T, np.eye(m), rtol=1e-14, atol=1e-14)
        assert_allclose(q2 @ q2.T, np.eye(m), rtol=1e-14, atol=1e-14)

    def test_4x4_lower_mode(self):
        """
        Test 4x4 case with UPLO='L' (lower block triangular).

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n1, n2 = 2, 2
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 1.0, 0.0, 0.0],
                      [0.3, 3.0, 0.0, 0.0],
                      [0.5, 1.0, 4.0, 1.0],
                      [0.3, 0.5, 0.2, 5.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5, 0.0, 0.0],
                      [0.2, 2.0, 0.0, 0.0],
                      [0.3, 0.5, 1.5, 0.5],
                      [0.2, 0.3, 0.1, 2.5]], order='F', dtype=float)

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('L', n1, n2, prec, a, b)

        assert info == 0
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)

        assert_allclose(q1 @ q1.T, np.eye(m), rtol=1e-14, atol=1e-14)
        assert_allclose(q2 @ q2.T, np.eye(m), rtol=1e-14, atol=1e-14)


class TestMB03DDN1N2Exchange:
    """Test N1 and N2 exchange behavior."""

    def test_n1_n2_exchanged_for_uplo_u(self):
        """
        For UPLO='U' and INFO=0, N1 and N2 should be exchanged.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n1, n2 = 1, 2
        m = n1 + n2
        prec = np.finfo(float).eps

        a = np.array([[2.0, 1.0, 0.5],
                      [0.0, 3.0, 1.0],
                      [0.0, 0.2, 4.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5, 0.3],
                      [0.0, 2.0, 0.5],
                      [0.0, 0.1, 1.5]], order='F', dtype=float)

        a_out, b_out, q1, q2, n1_out, n2_out, info = mb03dd('U', n1, n2, prec, a, b)

        assert info == 0
        assert n1_out == n2
        assert n2_out == n1
