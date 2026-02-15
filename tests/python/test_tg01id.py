"""
Tests for TG01ID: Orthogonal reduction of descriptor system to observability staircase form.

TG01ID computes orthogonal transformation matrices Q and Z which reduce
the N-th order descriptor system (A-lambda*E,B,C) to the form:

Q'*A*Z = ( Ano  * )    Q'*E*Z = ( Eno  * )    Q'*B = ( Bno )    C*Z = ( 0  Co )
         ( 0   Ao )             ( 0   Eo )           ( Bo  )

where (Ao-lambda*Eo,Bo,Co) is finite and/or infinite observable.
"""

import numpy as np
import pytest


class TestTG01IDBasic:
    """Basic functionality tests using HTML doc example."""

    def test_html_doc_example(self):
        """
        Test using the example from TG01ID HTML documentation.

        System: N=7, M=2, P=3 with JOBOBS='O' (separate finite/infinite unobservable)
        Expected: NOBSV=3, NIUOBS=1, NLBLCK=2, CTAU=[2,1]
        """
        from ctrlsys import tg01id

        n, m, p = 7, 2, 3

        # A matrix (7x7) - row-wise from HTML doc
        a = np.array([
            [2, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [2, 0, 0, 2, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1],
            [-1, 1, 0, -1, 0, 1, 0],
            [3, 0, 0, 3, 0, 0, 0],
            [1, 0, 1, 1, 1, 0, 1]
        ], dtype=float, order='F')

        # E matrix (7x7) - row-wise from HTML doc
        e = np.array([
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 3],
            [1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 2],
            [0, 0, 0, 0, 0, -1, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0]
        ], dtype=float, order='F')

        # B matrix (7x2) - row-wise from HTML doc
        b = np.array([
            [1, 0],
            [0, -1],
            [0, 1],
            [1, 0],
            [0, -1],
            [0, 1],
            [1, 0]
        ], dtype=float, order='F')

        # C matrix (3x7) - row-wise from HTML doc
        c = np.array([
            [2, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 3]
        ], dtype=float, order='F')

        # Call routine with JOBOBS='O', COMPQ='I', COMPZ='I'
        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == 0
        assert nobsv == 3
        assert niuobs == 1
        assert nlblck == 2
        np.testing.assert_array_equal(ctau[:nlblck], [2, 1])

        # Verify orthogonality of Q: Q'*Q = I
        np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-14, atol=1e-14)

        # Verify orthogonality of Z: Z'*Z = I
        np.testing.assert_allclose(z.T @ z, np.eye(n), rtol=1e-14, atol=1e-14)

        # Expected results from HTML doc (rtol=1e-3 for displayed precision)
        # Verify structure: C*Z has zeros in first N-NOBSV columns
        np.testing.assert_allclose(c_out[:, :n-nobsv], np.zeros((p, n-nobsv)), atol=1e-10)


class TestTG01IDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_transformation_consistency(self):
        """
        Verify transformation is consistent: Q'*A*Z, Q'*E*Z, Q'*B, C*Z.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import tg01id

        np.random.seed(42)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        # Keep copies
        a_orig = a.copy()
        e_orig = e.copy()
        b_orig = b.copy()
        c_orig = c.copy()

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == 0

        # Verify: Q'*A_orig*Z = A_out
        np.testing.assert_allclose(q.T @ a_orig @ z, a_out, rtol=1e-13, atol=1e-14)

        # Verify: Q'*E_orig*Z = E_out
        np.testing.assert_allclose(q.T @ e_orig @ z, e_out, rtol=1e-13, atol=1e-14)

        # Verify: Q'*B_orig = B_out
        np.testing.assert_allclose(q.T @ b_orig, b_out, rtol=1e-13, atol=1e-14)

        # Verify: C_orig*Z = C_out
        np.testing.assert_allclose(c_orig @ z, c_out, rtol=1e-13, atol=1e-14)

    def test_staircase_structure(self):
        """
        Verify staircase structure: lower-left block of A and E are zeros.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import tg01id

        np.random.seed(123)
        n, m, p = 5, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == 0

        # Structure check: lower-left block of A should be zero
        # A has structure (Ano, *; 0, Ao) where Ano is (n-nobsv)x(n-nobsv)
        if nobsv > 0 and nobsv < n:
            lower_left_a = a_out[n-nobsv:, :n-nobsv]
            np.testing.assert_allclose(lower_left_a, np.zeros_like(lower_left_a), atol=1e-13)

            lower_left_e = e_out[n-nobsv:, :n-nobsv]
            np.testing.assert_allclose(lower_left_e, np.zeros_like(lower_left_e), atol=1e-13)

            # C*Z = (0, Co) structure
            c_left = c_out[:, :n-nobsv]
            np.testing.assert_allclose(c_left, np.zeros_like(c_left), atol=1e-13)

    def test_eigenvalue_preservation(self):
        """
        Verify eigenvalues are preserved by orthogonal transformation.

        The generalized eigenvalues of (A,E) should be preserved.
        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import tg01id

        np.random.seed(456)
        n, m, p = 4, 1, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.eye(n, dtype=float, order='F') + 0.1 * np.random.randn(n, n)

        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_orig = a.copy()
        e_orig = e.copy()

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'F', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == 0

        # Compute generalized eigenvalues before and after
        eig_before = np.linalg.eigvals(np.linalg.solve(e_orig, a_orig))
        eig_after = np.linalg.eigvals(np.linalg.solve(e_out, a_out))

        # Sort by real and imaginary parts for comparison
        eig_before_sorted = sorted(eig_before, key=lambda x: (x.real, x.imag))
        eig_after_sorted = sorted(eig_after, key=lambda x: (x.real, x.imag))

        np.testing.assert_allclose(
            np.array(eig_before_sorted),
            np.array(eig_after_sorted),
            rtol=1e-12, atol=1e-13
        )


class TestTG01IDModes:
    """Test different JOBOBS modes."""

    def test_jobobs_f_finite_only(self):
        """
        Test JOBOBS='F': separate only finite unobservable eigenvalues.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import tg01id

        np.random.seed(789)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.eye(n, dtype=float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'F', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == 0
        assert niuobs == 0  # For JOBOBS='F', NIUOBS is set to 0
        assert nobsv >= 0 and nobsv <= n

        # Q and Z should be orthogonal
        np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z.T @ z, np.eye(n), rtol=1e-14, atol=1e-14)

    def test_jobobs_i_infinite_only(self):
        """
        Test JOBOBS='I': separate nonzero finite and infinite unobservable eigenvalues.

        Random seed: 101 (for reproducibility)
        """
        from ctrlsys import tg01id

        np.random.seed(101)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'I', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == 0
        assert niuobs == 0  # For JOBOBS='I', NIUOBS is set to 0
        assert nobsv >= 0 and nobsv <= n

        # Q and Z should be orthogonal
        np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z.T @ z, np.eye(n), rtol=1e-14, atol=1e-14)


class TestTG01IDCompModes:
    """Test different COMPQ and COMPZ modes."""

    def test_compq_compz_n(self):
        """
        Test with COMPQ='N', COMPZ='N': do not compute Q and Z.

        Random seed: 202 (for reproducibility)
        """
        from ctrlsys import tg01id

        np.random.seed(202)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'N', 'N', a, e, b, c, 0.0
        )

        assert info == 0
        # Q and Z may be returned as None or 1x1 placeholder
        # The transformed matrices should still have the correct structure
        if nobsv > 0 and nobsv < n:
            c_left = c_out[:, :n-nobsv]
            np.testing.assert_allclose(c_left, np.zeros_like(c_left), atol=1e-13)


class TestTG01IDEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with N=0 (empty system)."""
        from ctrlsys import tg01id

        a = np.array([], dtype=float, order='F').reshape(0, 0)
        e = np.array([], dtype=float, order='F').reshape(0, 0)
        b = np.array([], dtype=float, order='F').reshape(0, 0)
        c = np.array([], dtype=float, order='F').reshape(0, 0)

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == 0
        assert nobsv == 0

    def test_p_zero(self):
        """
        Test with P=0 (no outputs).

        Random seed: 303 (for reproducibility)
        """
        from ctrlsys import tg01id

        np.random.seed(303)
        n, m, p = 3, 2, 0

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.array([], dtype=float, order='F').reshape(0, n)

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == 0
        # With no outputs, the system is completely unobservable
        assert nobsv == 0

    def test_m_zero(self):
        """
        Test with M=0 (no inputs).

        Random seed: 404 (for reproducibility)
        """
        from ctrlsys import tg01id

        np.random.seed(404)
        n, m, p = 3, 0, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        b = np.array([], dtype=float, order='F').reshape(n, 0)
        c = np.random.randn(p, n).astype(float, order='F')

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == 0
        # Q and Z should be orthogonal
        np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z.T @ z, np.eye(n), rtol=1e-14, atol=1e-14)


class TestTG01IDErrorHandling:
    """Error handling tests."""

    def test_invalid_jobobs(self):
        """Test with invalid JOBOBS parameter."""
        from ctrlsys import tg01id

        a = np.array([[1.0]], order='F')
        e = np.array([[1.0]], order='F')
        b = np.array([[1.0]], order='F')
        c = np.array([[1.0]], order='F')

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'X', 'I', 'I', a, e, b, c, 0.0
        )

        assert info == -1

    def test_invalid_compq(self):
        """Test with invalid COMPQ parameter."""
        from ctrlsys import tg01id

        a = np.array([[1.0]], order='F')
        e = np.array([[1.0]], order='F')
        b = np.array([[1.0]], order='F')
        c = np.array([[1.0]], order='F')

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'X', 'I', a, e, b, c, 0.0
        )

        assert info == -2

    def test_invalid_compz(self):
        """Test with invalid COMPZ parameter."""
        from ctrlsys import tg01id

        a = np.array([[1.0]], order='F')
        e = np.array([[1.0]], order='F')
        b = np.array([[1.0]], order='F')
        c = np.array([[1.0]], order='F')

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'I', 'X', a, e, b, c, 0.0
        )

        assert info == -3

    def test_invalid_tol(self):
        """Test with invalid TOL parameter (>= 1)."""
        from ctrlsys import tg01id

        a = np.array([[1.0]], order='F')
        e = np.array([[1.0]], order='F')
        b = np.array([[1.0]], order='F')
        c = np.array([[1.0]], order='F')

        a_out, e_out, b_out, c_out, q, z, nobsv, niuobs, nlblck, ctau, info = tg01id(
            'O', 'I', 'I', a, e, b, c, 1.5
        )

        assert info == -23
