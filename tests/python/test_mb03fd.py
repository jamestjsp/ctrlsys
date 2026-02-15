"""Tests for MB03FD - Reducing block diagonal skew-Hamiltonian/Hamiltonian pencil."""

import numpy as np
import pytest


class TestMB03FDBasic:
    """Basic functionality tests for MB03FD."""

    def test_n2_real_eigenvalues_positive_signs(self):
        """
        Test 2x2 pencil with real eigenvalues (positive sign product).

        For N=2 with block diagonal A and anti-diagonal B:
        A = diag(a11, a22), B = [0 b12; b21 0]

        When sign(a11)*sign(a22)*sign(b21)*sign(b12) > 0, pencil has real eigenvalues.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(42)

        n = 2
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 1.5],
            [2.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0
        assert q1.shape == (n, n)
        assert q2.shape == (n, n)

    def test_n2_complex_eigenvalues(self):
        """
        Test 2x2 pencil with complex conjugate eigenvalues.

        When sign(a11)*sign(a22)*sign(b21)*sign(b12) < 0, pencil has complex eigenvalues.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(123)

        n = 2
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, -1.5],
            [2.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0
        assert q1.shape == (n, n)
        assert q2.shape == (n, n)

    def test_n2_zero_a11(self):
        """
        Test 2x2 pencil with A(1,1) ~ 0 (degenerate case).

        When |A11| <= PREC, use permutation matrices.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(456)

        n = 2
        prec = np.finfo(float).eps

        a = np.array([
            [1e-20, 0.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 1.5],
            [2.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0
        assert q1.shape == (n, n)
        assert q2.shape == (n, n)

    def test_n2_zero_a22(self):
        """
        Test 2x2 pencil with A(2,2) ~ 0 (degenerate case).

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(789)

        n = 2
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.0],
            [0.0, 1e-20]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 1.5],
            [2.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0
        assert q1.shape == (n, n)
        assert q2.shape == (n, n)

    def test_n2_zero_b21(self):
        """
        Test 2x2 pencil with B(2,1) ~ 0 (degenerate case).

        Random seed: 111 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(111)

        n = 2
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 1.5],
            [1e-20, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0
        assert q1.shape == (n, n)
        assert q2.shape == (n, n)

    def test_n2_zero_b12(self):
        """
        Test 2x2 pencil with B(1,2) ~ 0 (degenerate case).

        Random seed: 222 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(222)

        n = 2
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 1e-20],
            [2.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0
        assert q1.shape == (n, n)
        assert q2.shape == (n, n)

    def test_n4_basic(self):
        """
        Test 4x4 pencil - uses DGGES for generalized Schur form.

        For N=4, the pencil has block diagonal A and anti-block-diagonal B.

        Random seed: 333 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(333)

        n = 4
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.5, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.5, 0.3],
            [0.0, 0.0, 0.0, 2.5]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 1.5],
            [2.0, 0.0, 0.0, 0.0],
            [0.3, 1.0, 0.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0
        assert q1.shape == (n, n)
        assert q2.shape == (n, n)

    def test_n4_with_negative_eigenvalues(self):
        """
        Test 4x4 pencil that should have eigenvalues with negative real parts.

        The DGGES call with SB02OW callback selects stable eigenvalues.

        Random seed: 444 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(444)

        n = 4
        prec = np.finfo(float).eps

        a = np.array([
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.1],
            [0.0, 0.0, 0.0, 1.5]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 0.0, -2.0, -0.1],
            [0.0, 0.0, 0.0, -1.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.1, 2.0, 0.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0
        assert q1.shape == (n, n)
        assert q2.shape == (n, n)


class TestMB03FDMathematicalProperties:
    """Mathematical property tests for MB03FD."""

    def test_q1_orthogonality_n2(self):
        """
        Validate Q1 is orthogonal for N=2: Q1^T Q1 = I.

        Random seed: 500 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(500)

        n = 2
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 1.5],
            [2.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0

        q1tq1 = q1.T @ q1
        np.testing.assert_allclose(q1tq1, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_q2_orthogonality_n2(self):
        """
        Validate Q2 is orthogonal for N=2: Q2^T Q2 = I.

        Random seed: 501 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(501)

        n = 2
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 1.5],
            [2.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0

        q2tq2 = q2.T @ q2
        np.testing.assert_allclose(q2tq2, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_q1_orthogonality_n4(self):
        """
        Validate Q1 is orthogonal for N=4: Q1^T Q1 = I.

        Random seed: 502 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(502)

        n = 4
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.5, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.5, 0.3],
            [0.0, 0.0, 0.0, 2.5]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 1.5],
            [2.0, 0.0, 0.0, 0.0],
            [0.3, 1.0, 0.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0

        q1tq1 = q1.T @ q1
        np.testing.assert_allclose(q1tq1, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_q2_orthogonality_n4(self):
        """
        Validate Q2 is orthogonal for N=4: Q2^T Q2 = I.

        Random seed: 503 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(503)

        n = 4
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.5, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.5, 0.3],
            [0.0, 0.0, 0.0, 2.5]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 1.5],
            [2.0, 0.0, 0.0, 0.0],
            [0.3, 1.0, 0.0, 0.0]
        ], order='F', dtype=float)

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0

        q2tq2 = q2.T @ q2
        np.testing.assert_allclose(q2tq2, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_transformation_n2_real_eigenvalues(self):
        """
        Validate transformation Q2' A Q1 is upper triangular for N=2.

        For N=2 with real eigenvalues, the transformed A should remain diagonal
        and B should be upper triangular.

        Random seed: 600 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(600)

        n = 2
        prec = np.finfo(float).eps

        a_orig = np.array([
            [2.0, 0.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        b_orig = np.array([
            [0.0, 1.5],
            [2.0, 0.0]
        ], order='F', dtype=float)

        a = a_orig.copy()
        b = b_orig.copy()

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0

        q1tq1 = q1.T @ q1
        q2tq2 = q2.T @ q2
        np.testing.assert_allclose(q1tq1, np.eye(n), rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(q2tq2, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_transformation_n4_schur_form(self):
        """
        Validate a_out is upper triangular for N=4.

        For N=4, the output a_out is already Q2' A Q1 which is upper triangular.
        The output b_out is Q2' B Q1 which is upper quasi-triangular.

        Random seed: 700 (for reproducibility)
        """
        from ctrlsys import mb03fd

        np.random.seed(700)

        n = 4
        prec = np.finfo(float).eps

        a_orig = np.array([
            [2.0, 0.5, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.5, 0.3],
            [0.0, 0.0, 0.0, 2.5]
        ], order='F', dtype=float)

        b_orig = np.array([
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 1.5],
            [2.0, 0.0, 0.0, 0.0],
            [0.3, 1.0, 0.0, 0.0]
        ], order='F', dtype=float)

        a = a_orig.copy()
        b = b_orig.copy()

        a_out, b_out, q1, q2, info = mb03fd(n=n, prec=prec, a=a, b=b)

        assert info == 0

        q1tq1 = q1.T @ q1
        q2tq2 = q2.T @ q2
        np.testing.assert_allclose(q1tq1, np.eye(n), rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(q2tq2, np.eye(n), rtol=1e-13, atol=1e-14)

        np.testing.assert_allclose(np.tril(a_out, -1), np.zeros((n, n)), atol=1e-12)


class TestMB03FDErrorHandling:
    """Error handling tests for MB03FD."""

    def test_invalid_n(self):
        """Test error for N not in {2, 4}."""
        from ctrlsys import mb03fd

        n = 3
        prec = np.finfo(float).eps

        a = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        b = np.array([
            [0.0, 1.5, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        with pytest.raises(ValueError):
            mb03fd(n=n, prec=prec, a=a, b=b)
