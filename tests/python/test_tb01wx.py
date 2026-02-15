"""
Tests for TB01WX - Orthogonal similarity transformation to Hessenberg form.

TB01WX reduces A to upper Hessenberg form via orthogonal similarity
transformation and applies the transformation to B and C:
    A <- U'*A*U (Hessenberg)
    B <- U'*B
    C <- C*U

COMPU modes:
    'N' - do not compute U
    'I' - initialize U to identity, return orthogonal U
    'U' - update given U1 with product U1*U
"""

import numpy as np
import pytest

from ctrlsys import tb01wx


class TestTB01WXBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_basic_hessenberg_compu_i(self):
        """
        Test from SLICOT HTML documentation example.

        System: 5 states, 2 inputs, 3 outputs, COMPU='I'
        Validates numerical correctness against documented results.
        """
        n, m, p = 5, 2, 3

        a = np.array([
            [-0.04165,  4.9200, -4.9200,  0.0000,  0.0000],
            [-1.387944, -3.3300,  0.0000,  0.0000,  0.0000],
            [ 0.5450,   0.0000,  0.0000, -0.5450,  0.0000],
            [ 0.0000,   0.0000,  4.9200, -0.04165,  4.9200],
            [ 0.0000,   0.0000,  0.0000, -1.387944, -3.3300]
        ], order='F', dtype=float)

        b = np.array([
            [0.0000,  0.0000],
            [3.3300,  0.0000],
            [0.0000,  0.0000],
            [0.0000,  0.0000],
            [0.0000,  3.3300]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0]
        ], order='F', dtype=float)

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a, b, c)

        assert info == 0

        # Expected A in Hessenberg form from HTML doc
        a_expected = np.array([
            [-0.0416, -6.3778,  1.4826, -1.9856,  1.2630],
            [ 1.4911, -2.8851, -0.4353,  0.8984, -0.5714],
            [ 0.0000, -2.1254,  1.6804, -4.9686, -1.7731],
            [ 0.0000,  0.0000,  2.1880, -3.3545, -2.6069],
            [ 0.0000,  0.0000,  0.0000,  0.7554, -2.1424]
        ], order='F', dtype=float)

        # Expected B from HTML doc
        b_expected = np.array([
            [ 0.0000,  0.0000],
            [-3.0996,  0.0000],
            [-0.6488,  0.0000],
            [ 0.8689,  1.7872],
            [-0.5527,  2.8098]
        ], order='F', dtype=float)

        # Expected C from HTML doc
        c_expected = np.array([
            [1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [0.0000,  0.3655, -0.4962,  0.6645, -0.4227],
            [0.0000,  0.0000, -0.8461, -0.4498,  0.2861]
        ], order='F', dtype=float)

        # Expected U from HTML doc
        u_expected = np.array([
            [1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [0.0000, -0.9308, -0.1948,  0.2609, -0.1660],
            [0.0000,  0.3655, -0.4962,  0.6645, -0.4227],
            [0.0000,  0.0000, -0.8461, -0.4498,  0.2861],
            [0.0000,  0.0000,  0.0000,  0.5367,  0.8438]
        ], order='F', dtype=float)

        # HTML doc shows 4 decimal places - use appropriate tolerance
        np.testing.assert_allclose(a_hess, a_expected, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(b_trans, b_expected, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(c_trans, c_expected, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(u, u_expected, rtol=1e-3, atol=1e-3)


class TestTB01WXMathematical:
    """Mathematical property tests for numerical correctness."""

    def test_eigenvalue_preservation(self):
        """
        Validate eigenvalue preservation under similarity transformation.

        Hessenberg reduction is a similarity transformation, so eigenvalues
        must be preserved: lambda(A) = lambda(U'*A*U)

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 2, 3

        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        # Compute eigenvalues before transformation
        eig_before = np.linalg.eigvals(a.copy())

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a.copy(), b.copy(), c.copy())

        assert info == 0

        # Compute eigenvalues after transformation
        eig_after = np.linalg.eigvals(a_hess)

        # Sort by real part then imaginary part for comparison
        eig_before_sorted = sorted(eig_before, key=lambda x: (x.real, x.imag))
        eig_after_sorted = sorted(eig_after, key=lambda x: (x.real, x.imag))

        np.testing.assert_allclose(
            np.array(eig_before_sorted),
            np.array(eig_after_sorted),
            rtol=1e-13, atol=1e-14
        )

    def test_orthogonality_of_u(self):
        """
        Validate U is orthogonal: U'*U = I and U*U' = I.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 5, 2, 3

        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a, b, c)

        assert info == 0

        # U'*U should be identity
        utu = u.T @ u
        np.testing.assert_allclose(utu, np.eye(n), rtol=1e-14, atol=1e-14)

        # U*U' should be identity
        uut = u @ u.T
        np.testing.assert_allclose(uut, np.eye(n), rtol=1e-14, atol=1e-14)

    def test_similarity_transformation_relation(self):
        """
        Validate A_hess = U'*A*U (similarity transformation).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 4, 2, 2

        a_orig = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a_orig.copy(), b, c)

        assert info == 0

        # Verify A_hess = U'*A*U
        a_expected = u.T @ a_orig @ u
        np.testing.assert_allclose(a_hess, a_expected, rtol=1e-13, atol=1e-14)

    def test_b_transformation_relation(self):
        """
        Validate B_trans = U'*B.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 4, 3, 2

        a = np.random.randn(n, n).astype(float, order='F')
        b_orig = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a, b_orig.copy(), c)

        assert info == 0

        # Verify B_trans = U'*B
        b_expected = u.T @ b_orig
        np.testing.assert_allclose(b_trans, b_expected, rtol=1e-13, atol=1e-14)

    def test_c_transformation_relation(self):
        """
        Validate C_trans = C*U.

        Random seed: 101 (for reproducibility)
        """
        np.random.seed(101)
        n, m, p = 4, 2, 3

        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c_orig = np.random.randn(p, n).astype(float, order='F')

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a, b, c_orig.copy())

        assert info == 0

        # Verify C_trans = C*U
        c_expected = c_orig @ u
        np.testing.assert_allclose(c_trans, c_expected, rtol=1e-13, atol=1e-14)

    def test_hessenberg_structure(self):
        """
        Validate output A is upper Hessenberg (zeros below first subdiagonal).

        Random seed: 202 (for reproducibility)
        """
        np.random.seed(202)
        n, m, p = 6, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a, b, c)

        assert info == 0

        # Check below first subdiagonal is zero
        for i in range(2, n):
            for j in range(i - 1):
                assert abs(a_hess[i, j]) < 1e-14, f"a_hess[{i},{j}] = {a_hess[i,j]} should be zero"


class TestTB01WXCompuModes:
    """Test different COMPU modes."""

    def test_compu_n_no_u_computed(self):
        """
        Test COMPU='N' - U is not computed.

        Random seed: 303 (for reproducibility)
        """
        np.random.seed(303)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        # Compute eigenvalues before
        eig_before = np.linalg.eigvals(a.copy())

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('N', n, m, p, a.copy(), b.copy(), c.copy())

        assert info == 0

        # Eigenvalues should still be preserved
        eig_after = np.linalg.eigvals(a_hess)
        eig_before_sorted = sorted(eig_before, key=lambda x: (x.real, x.imag))
        eig_after_sorted = sorted(eig_after, key=lambda x: (x.real, x.imag))
        np.testing.assert_allclose(
            np.array(eig_before_sorted),
            np.array(eig_after_sorted),
            rtol=1e-13, atol=1e-14
        )

        # A should be Hessenberg
        for i in range(2, n):
            for j in range(i - 1):
                assert abs(a_hess[i, j]) < 1e-14

    def test_compu_u_update_given_matrix(self):
        """
        Test COMPU='U' - update given orthogonal U1 with product U1*U.

        Random seed: 404 (for reproducibility)
        """
        np.random.seed(404)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        # Create a random orthogonal matrix U1 via QR
        q, _ = np.linalg.qr(np.random.randn(n, n))
        u1 = np.asfortranarray(q)

        a_hess, b_trans, c_trans, u_out, dwork, info = tb01wx('U', n, m, p, a.copy(), b.copy(), c.copy(), u=u1.copy())

        assert info == 0

        # Output U should still be orthogonal (product of orthogonal matrices)
        np.testing.assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-13, atol=1e-14)

        # A should be Hessenberg
        for i in range(2, n):
            for j in range(i - 1):
                assert abs(a_hess[i, j]) < 1e-14


class TestTB01WXEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with n=0 (empty system)."""
        n, m, p = 0, 2, 3

        a = np.zeros((0, 0), order='F', dtype=float)
        b = np.zeros((0, m), order='F', dtype=float)
        c = np.zeros((p, 0), order='F', dtype=float)

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a, b, c)

        assert info == 0

    def test_m_zero_no_inputs(self):
        """
        Test with m=0 (no inputs).

        Random seed: 505 (for reproducibility)
        """
        np.random.seed(505)
        n, m, p = 4, 0, 2

        a = np.random.randn(n, n).astype(float, order='F')
        b = np.zeros((n, 0), order='F', dtype=float)
        c = np.random.randn(p, n).astype(float, order='F')

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a, b, c)

        assert info == 0
        assert a_hess.shape == (n, n)
        assert b_trans.shape == (n, 0)

        # Orthogonality
        np.testing.assert_allclose(u.T @ u, np.eye(n), rtol=1e-14, atol=1e-14)

    def test_p_zero_no_outputs(self):
        """
        Test with p=0 (no outputs).

        Random seed: 606 (for reproducibility)
        """
        np.random.seed(606)
        n, m, p = 4, 2, 0

        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.zeros((0, n), order='F', dtype=float)

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a, b, c)

        assert info == 0
        assert a_hess.shape == (n, n)
        assert c_trans.shape == (0, n)

        # Orthogonality
        np.testing.assert_allclose(u.T @ u, np.eye(n), rtol=1e-14, atol=1e-14)

    def test_n_one(self):
        """Test with n=1 (scalar system - already Hessenberg)."""
        n, m, p = 1, 2, 2

        a = np.array([[3.5]], order='F', dtype=float)
        b = np.array([[1.0, 2.0]], order='F', dtype=float)
        c = np.array([[0.5], [1.5]], order='F', dtype=float)

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a.copy(), b.copy(), c.copy())

        assert info == 0

        # U should be 1x1 identity (or -1)
        assert abs(abs(u[0, 0]) - 1.0) < 1e-14

        # A should be unchanged (scaled by u^2 = 1)
        np.testing.assert_allclose(a_hess[0, 0], a[0, 0], rtol=1e-14)


class TestTB01WXErrorHandling:
    """Error handling tests."""

    def test_invalid_compu(self):
        """Test invalid COMPU parameter."""
        n, m, p = 3, 2, 2

        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, m), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('X', n, m, p, a, b, c)

        assert info == -1  # COMPU is first parameter

    def test_negative_n(self):
        """Test negative n parameter."""
        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 1), order='F', dtype=float)
        c = np.zeros((1, 1), order='F', dtype=float)

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', -1, 1, 1, a, b, c)

        assert info == -2  # N is second parameter

    def test_negative_m(self):
        """Test negative m parameter."""
        n = 2
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, 1), order='F', dtype=float)
        c = np.zeros((1, n), order='F', dtype=float)

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, -1, 1, a, b, c)

        assert info == -3  # M is third parameter

    def test_negative_p(self):
        """Test negative p parameter."""
        n = 2
        a = np.zeros((n, n), order='F', dtype=float)
        b = np.zeros((n, 1), order='F', dtype=float)
        c = np.zeros((1, n), order='F', dtype=float)

        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, 1, -1, a, b, c)

        assert info == -4  # P is fourth parameter


class TestTB01WXWorkspaceQuery:
    """Workspace query tests."""

    def test_workspace_query(self):
        """
        Test workspace query with LDWORK=-1.

        Random seed: 707 (for reproducibility)
        """
        np.random.seed(707)
        n, m, p = 5, 2, 3

        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        # Query optimal workspace
        a_hess, b_trans, c_trans, u, dwork, info = tb01wx('I', n, m, p, a, b, c, ldwork=-1)

        assert info == 0
        assert dwork[0] >= n - 1 + max(n, m, p)  # Minimum workspace
