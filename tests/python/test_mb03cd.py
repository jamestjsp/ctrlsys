"""Tests for MB03CD - Eigenvalue exchange in block triangular pencils."""

import numpy as np
import pytest


class TestMB03CDBasic:
    """Basic functionality tests for MB03CD."""

    def test_2x2_upper_eigenvalue_exchange(self):
        """
        Test 2x2 upper block triangular pencil eigenvalue exchange.

        For a 2x2 pencil aAB - bD with N1=N2=1, we exchange scalar eigenvalues.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb03cd

        np.random.seed(42)

        n1, n2 = 1, 1
        m = n1 + n2

        a = np.array([
            [2.0, 0.5],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.2],
            [0.0, 1.5]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.1],
            [0.0, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='U', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0
        assert n1_out == n2
        assert n2_out == n1
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)
        assert q3.shape == (m, m)

    def test_3x3_upper_n1_1_n2_2(self):
        """
        Test 3x3 upper block triangular pencil with N1=1, N2=2.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb03cd

        np.random.seed(123)

        n1, n2 = 1, 2
        m = n1 + n2

        a = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 1.0, 0.4],
            [0.0, 0.2, 1.5]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.2, 0.1],
            [0.0, 2.0, 0.3],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.1, 0.2],
            [0.0, 1.5, 0.4],
            [0.0, 0.3, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='U', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0
        assert n1_out == n2
        assert n2_out == n1
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)
        assert q3.shape == (m, m)

    def test_4x4_upper_n1_2_n2_2(self):
        """
        Test 4x4 upper block triangular pencil with N1=2, N2=2.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import mb03cd

        np.random.seed(456)

        n1, n2 = 2, 2
        m = n1 + n2

        a = np.array([
            [2.0, 0.5, 0.3, 0.1],
            [0.1, 1.5, 0.2, 0.2],
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 0.0, 0.2, 0.8]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.2, 0.1, 0.1],
            [0.0, 1.5, 0.3, 0.2],
            [0.0, 0.0, 2.0, 0.4],
            [0.0, 0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.1, 0.2, 0.1],
            [0.0, 2.0, 0.4, 0.2],
            [0.0, 0.0, 1.5, 0.3],
            [0.0, 0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='U', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0
        assert n1_out == n2
        assert n2_out == n1
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)
        assert q3.shape == (m, m)

    def test_3x3_lower_triangular(self):
        """
        Test 3x3 lower block triangular pencil (UPLO='L').

        For lower triangular, eigenvalues are NOT exchanged.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import mb03cd

        np.random.seed(789)

        n1, n2 = 1, 2
        m = n1 + n2

        a = np.array([
            [2.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.3, 0.4, 1.5]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0, 0.0],
            [0.2, 2.0, 0.0],
            [0.1, 0.3, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0, 0.0],
            [0.1, 1.5, 0.0],
            [0.2, 0.4, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='L', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0
        assert q1.shape == (m, m)
        assert q2.shape == (m, m)
        assert q3.shape == (m, m)


class TestMB03CDMathematicalProperties:
    """Mathematical property tests for MB03CD."""

    def test_orthogonality_q1(self):
        """
        Validate Q1 is orthogonal: Q1^T Q1 = I.

        Random seed: 100 (for reproducibility)
        """
        from ctrlsys import mb03cd

        np.random.seed(100)

        n1, n2 = 1, 2
        m = n1 + n2

        a = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 1.0, 0.4],
            [0.0, 0.2, 1.5]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.2, 0.1],
            [0.0, 2.0, 0.3],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.1, 0.2],
            [0.0, 1.5, 0.4],
            [0.0, 0.3, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='U', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0

        q1tq1 = q1.T @ q1
        np.testing.assert_allclose(q1tq1, np.eye(m), rtol=1e-13, atol=1e-14)

    def test_orthogonality_q2(self):
        """
        Validate Q2 is orthogonal: Q2^T Q2 = I.

        Random seed: 101 (for reproducibility)
        """
        from ctrlsys import mb03cd

        np.random.seed(101)

        n1, n2 = 2, 1
        m = n1 + n2

        a = np.array([
            [2.0, 0.5, 0.3],
            [0.1, 1.5, 0.2],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.2, 0.1],
            [0.0, 2.0, 0.3],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.1, 0.2],
            [0.0, 1.5, 0.4],
            [0.0, 0.0, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='U', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0

        q2tq2 = q2.T @ q2
        np.testing.assert_allclose(q2tq2, np.eye(m), rtol=1e-13, atol=1e-14)

    def test_orthogonality_q3(self):
        """
        Validate Q3 is orthogonal: Q3^T Q3 = I.

        Random seed: 102 (for reproducibility)
        """
        from ctrlsys import mb03cd

        np.random.seed(102)

        n1, n2 = 1, 1
        m = n1 + n2

        a = np.array([
            [2.0, 0.5],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.2],
            [0.0, 1.5]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.1],
            [0.0, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='U', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0

        q3tq3 = q3.T @ q3
        np.testing.assert_allclose(q3tq3, np.eye(m), rtol=1e-13, atol=1e-14)

    def test_transformation_preserves_structure(self):
        """
        Test that transformation preserves pencil structure.

        For upper triangular: a(Q3' A Q2)(Q2' B Q1) - b(Q3' D Q1) stays block upper triangular.

        Random seed: 200 (for reproducibility)
        """
        from ctrlsys import mb03cd

        np.random.seed(200)

        n1, n2 = 1, 2
        m = n1 + n2

        a = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 1.0, 0.4],
            [0.0, 0.2, 1.5]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.2, 0.1],
            [0.0, 2.0, 0.3],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.1, 0.2],
            [0.0, 1.5, 0.4],
            [0.0, 0.3, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='U', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0

        for qi in [q1, q2, q3]:
            qtq = qi.T @ qi
            np.testing.assert_allclose(qtq, np.eye(m), rtol=1e-13, atol=1e-14)


class TestMB03CDEdgeCases:
    """Edge case tests for MB03CD."""

    def test_n1_0_returns_identity(self):
        """
        Test with N1=0 - should be a trivial case.

        If N1=0 and N2=2, the 'first block' is empty.
        """
        from ctrlsys import mb03cd

        n1, n2 = 0, 2
        m = n1 + n2

        a = np.array([
            [1.0, 0.4],
            [0.2, 1.5]
        ], order='F', dtype=float)

        b = np.array([
            [2.0, 0.3],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.5, 0.4],
            [0.3, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='U', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0

    def test_n2_0_returns_identity(self):
        """
        Test with N2=0 - should be a trivial case.

        If N1=2 and N2=0, the 'second block' is empty.
        """
        from ctrlsys import mb03cd

        n1, n2 = 2, 0
        m = n1 + n2

        a = np.array([
            [1.0, 0.4],
            [0.2, 1.5]
        ], order='F', dtype=float)

        b = np.array([
            [2.0, 0.3],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.5, 0.4],
            [0.3, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        a_out, b_out, d_out, q1, q2, q3, n1_out, n2_out, info = mb03cd(
            uplo='U', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d
        )

        assert info == 0


class TestMB03CDErrorHandling:
    """Error handling tests for MB03CD."""

    def test_invalid_uplo(self):
        """Test error for invalid UPLO parameter."""
        from ctrlsys import mb03cd

        n1, n2 = 1, 1
        m = n1 + n2

        a = np.array([
            [2.0, 0.5],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.2],
            [0.0, 1.5]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.1],
            [0.0, 2.0]
        ], order='F', dtype=float)

        prec = np.finfo(float).eps

        with pytest.raises(ValueError):
            mb03cd(uplo='X', n1=n1, n2=n2, prec=prec, a=a, b=b, d=d)
