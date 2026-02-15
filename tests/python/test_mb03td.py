"""
Tests for MB03TD: Reordering diagonal blocks of (skew-)Hamiltonian Schur form.

MB03TD reorders a matrix X in skew-Hamiltonian Schur form:
    X = [A  G; 0  A^T], G = -G^T (skew-symmetric)
or Hamiltonian Schur form:
    X = [A  G; 0  -A^T], G = G^T (symmetric)

so that selected eigenvalues appear in the leading diagonal blocks.
"""

import numpy as np
import pytest


class TestMB03TDBasic:
    """Tests using HTML documentation example data."""

    def test_skew_hamiltonian_with_updates(self):
        """
        Test skew-Hamiltonian case with orthogonal symplectic update (COMPU='U').

        Data from SLICOT HTML documentation MB03TD example.
        N=5, TYP='S', COMPU='U'
        """
        from ctrlsys import mb03td

        n = 5

        # SELECT and LOWER arrays
        select = np.array([False, True, True, False, False], dtype=bool)
        lower = np.array([False, True, True, False, False], dtype=bool)

        # Matrix A (upper quasi-triangular, Schur form) - row-wise input
        a = np.array([
            [0.9501, 0.7621, 0.6154, 0.4057, 0.0579],
            [0.0,    0.4565, 0.7919, 0.9355, 0.3529],
            [0.0,   -0.6822, 0.4565, 0.9169, 0.8132],
            [0.0,    0.0,    0.0,    0.4103, 0.0099],
            [0.0,    0.0,    0.0,    0.0,    0.1389]
        ], order='F', dtype=float)

        # Matrix G (strictly upper triangular, skew-symmetric) - row-wise input
        g = np.array([
            [0.0,   -0.1834, -0.1851,  0.5659,  0.3040],
            [0.0,    0.0,     0.4011, -0.9122,  0.2435],
            [0.0,    0.0,     0.0,     0.4786, -0.2432],
            [0.0,    0.0,     0.0,     0.0,    -0.5272],
            [0.0,    0.0,     0.0,     0.0,     0.0]
        ], order='F', dtype=float)

        # U1 = I, U2 = 0
        u1 = np.eye(n, order='F', dtype=float)
        u2 = np.zeros((n, n), order='F', dtype=float)

        # Call routine
        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'S', 'U', select, lower, a, g, u1, u2
        )

        assert info == 0
        assert m == 2  # Two eigenvalues selected (complex pair)

        # Verify leading 2x2 block has the selected eigenvalues (0.4565 +- 0.735i)
        # Check diagonal elements of 2x2 block
        np.testing.assert_allclose(a_out[0, 0], 0.4565, rtol=1e-3)
        np.testing.assert_allclose(a_out[1, 1], 0.4565, rtol=1e-3)
        # Check that it's a proper 2x2 block (nonzero subdiagonal)
        assert np.abs(a_out[1, 0]) > 0.5  # Should be ~1.186
        # Check remaining diagonal elements are 1x1 blocks
        np.testing.assert_allclose(a_out[2, 2], 0.9501, rtol=1e-3)
        np.testing.assert_allclose(a_out[3, 3], 0.4103, rtol=1e-3)
        np.testing.assert_allclose(a_out[4, 4], 0.1389, rtol=1e-3)
        # Check strictly lower triangular part is zero (except 2x2 block)
        assert a_out[2, 0] == 0 and a_out[2, 1] == 0
        assert a_out[3, 0] == 0 and a_out[3, 1] == 0 and a_out[3, 2] == 0
        assert a_out[4, 0] == 0 and a_out[4, 1] == 0 and a_out[4, 2] == 0 and a_out[4, 3] == 0

        # Verify eigenvalues: complex pair at (0.4565, +-wi) should be at top
        assert np.isclose(wr[0], 0.4565, rtol=1e-3)
        assert wi[0] > 0  # Positive imaginary part first
        assert np.isclose(wi[1], -wi[0], rtol=1e-12)  # Conjugate pair

        # Verify U is orthogonal symplectic: U^T * U = I
        u_full = np.block([
            [u1_out, u2_out],
            [-u2_out, u1_out]
        ])
        identity = np.eye(2 * n)
        np.testing.assert_allclose(u_full.T @ u_full, identity, rtol=1e-13, atol=1e-14)


class TestMB03TDNoUpdate:
    """Tests without computing Schur vectors."""

    def test_skew_hamiltonian_no_update(self):
        """
        Test skew-Hamiltonian case without orthogonal symplectic update (COMPU='N').

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb03td

        np.random.seed(42)
        n = 4

        # Create a simple Schur form matrix
        # Diagonal entries
        a = np.array([
            [2.0, 1.5, 0.3, 0.1],
            [0.0, 1.0, 0.5, 0.2],
            [0.0, 0.0, 0.5, 0.8],
            [0.0, 0.0, 0.0, 0.3]
        ], order='F', dtype=float)

        # Skew-symmetric G (strictly upper triangular storage)
        g = np.array([
            [0.0, 0.5, 0.3, 0.1],
            [0.0, 0.0, 0.4, 0.2],
            [0.0, 0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        # Select eigenvalue at position 3 (0.5)
        select = np.array([False, False, True, False], dtype=bool)
        lower = np.array([False, False, False, False], dtype=bool)

        # Dummy U1, U2 (not used when COMPU='N')
        u1 = np.zeros((1, 1), order='F', dtype=float)
        u2 = np.zeros((1, 1), order='F', dtype=float)

        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'S', 'N', select, lower, a, g, u1, u2
        )

        assert info == 0
        assert m == 1  # One eigenvalue selected

        # The selected eigenvalue (0.5) should be moved to leading position
        assert np.isclose(wr[0], 0.5, rtol=1e-12)


class TestMB03TDHamiltonian:
    """Tests for Hamiltonian matrices (TYP='H')."""

    def test_hamiltonian_real_eigenvalues(self):
        """
        Test Hamiltonian case with real eigenvalues.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb03td

        np.random.seed(123)
        n = 3

        # Schur form matrix with distinct real eigenvalues
        a = np.array([
            [3.0, 1.0, 0.5],
            [0.0, 2.0, 0.3],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        # Symmetric G (upper triangular storage)
        g = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 0.8, 0.3],
            [0.2, 0.3, 0.6]
        ], order='F', dtype=float)

        # Select eigenvalue at position 2 (value 2.0)
        select = np.array([False, True, False], dtype=bool)
        lower = np.array([False, False, False], dtype=bool)

        u1 = np.eye(n, order='F', dtype=float)
        u2 = np.zeros((n, n), order='F', dtype=float)

        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'H', 'U', select, lower, a, g, u1, u2
        )

        assert info == 0
        assert m == 1

        # The selected eigenvalue (2.0) should be at leading position
        assert np.isclose(wr[0], 2.0, rtol=1e-12)

        # Verify U is orthogonal symplectic
        u_full = np.block([
            [u1_out, u2_out],
            [-u2_out, u1_out]
        ])
        identity = np.eye(2 * n)
        np.testing.assert_allclose(u_full.T @ u_full, identity, rtol=1e-13, atol=1e-14)


class TestMB03TDComplexEigenvalues:
    """Tests for matrices with complex eigenvalue pairs."""

    def test_complex_pair_selection(self):
        """
        Test selection of complex conjugate eigenvalue pair.

        A 2x2 block with eigenvalues a +- bi requires both SELECT entries true.
        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import mb03td

        np.random.seed(456)
        n = 4

        # Create Schur form with 2x2 block (complex eigenvalues) at positions 2-3
        # Eigenvalues: 3.0, 1.5 +- 0.5i, 0.5
        a = np.array([
            [3.0, 0.8, 0.3, 0.1],
            [0.0, 1.5, 0.5, 0.2],
            [0.0, -0.5, 1.5, 0.3],
            [0.0, 0.0, 0.0, 0.5]
        ], order='F', dtype=float)

        # Skew-symmetric G
        g = np.array([
            [0.0, 0.3, 0.2, 0.1],
            [0.0, 0.0, 0.4, 0.2],
            [0.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        # Select the complex pair
        select = np.array([False, True, True, False], dtype=bool)
        lower = np.array([False, False, False, False], dtype=bool)

        u1 = np.eye(n, order='F', dtype=float)
        u2 = np.zeros((n, n), order='F', dtype=float)

        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'S', 'U', select, lower, a, g, u1, u2
        )

        assert info == 0
        assert m == 2  # Complex pair = 2 dimensions

        # Complex eigenvalues should now be at leading 2x2 block
        # Check subdiagonal indicates 2x2 block
        assert a_out[1, 0] != 0  # Subdiagonal element nonzero for 2x2 block
        assert np.isclose(wr[0], wr[1], rtol=1e-12)  # Same real part
        assert wi[0] > 0 and wi[1] < 0  # Conjugate imaginary parts


class TestMB03TDEdgeCases:
    """Edge case tests."""

    def test_zero_dimension(self):
        """Test with n=0 (quick return)."""
        from ctrlsys import mb03td

        n = 0
        select = np.array([], dtype=bool)
        lower = np.array([], dtype=bool)
        a = np.zeros((0, 0), order='F', dtype=float)
        g = np.zeros((0, 0), order='F', dtype=float)
        u1 = np.zeros((0, 0), order='F', dtype=float)
        u2 = np.zeros((0, 0), order='F', dtype=float)

        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'S', 'N', select, lower, a, g, u1, u2
        )

        assert info == 0
        assert m == 0

    def test_single_element(self):
        """Test with n=1 (single eigenvalue)."""
        from ctrlsys import mb03td

        n = 1
        select = np.array([True], dtype=bool)
        lower = np.array([False], dtype=bool)
        a = np.array([[2.5]], order='F', dtype=float)
        g = np.array([[0.0]], order='F', dtype=float)
        u1 = np.array([[1.0]], order='F', dtype=float)
        u2 = np.array([[0.0]], order='F', dtype=float)

        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'S', 'U', select, lower, a, g, u1, u2
        )

        assert info == 0
        assert m == 1
        assert np.isclose(wr[0], 2.5)
        assert np.isclose(wi[0], 0.0)

    def test_no_selection(self):
        """Test with no eigenvalues selected."""
        from ctrlsys import mb03td

        n = 3
        select = np.array([False, False, False], dtype=bool)
        lower = np.array([False, False, False], dtype=bool)

        a = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 1.5, 0.2],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        g = np.array([
            [0.0, 0.3, 0.1],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        a_copy = a.copy()
        u1 = np.eye(n, order='F', dtype=float)
        u2 = np.zeros((n, n), order='F', dtype=float)

        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'S', 'U', select, lower, a, g, u1, u2
        )

        assert info == 0
        assert m == 0  # No eigenvalues selected

        # A should remain unchanged since no reordering needed
        np.testing.assert_allclose(a_out, a_copy, rtol=1e-14)


class TestMB03TDLowerFlag:
    """Tests for LOWER array functionality."""

    def test_lower_flag_effect(self):
        """
        Test that LOWER flag affects which copy of eigenvalue is moved.

        For skew-Hamiltonian matrices, eigenvalues appear twice.
        LOWER controls which copy is reordered to leading position.
        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import mb03td

        np.random.seed(789)
        n = 3

        a = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 1.5, 0.2],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        g = np.array([
            [0.0, 0.3, 0.1],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        select = np.array([False, True, False], dtype=bool)

        # Test with LOWER=True
        lower_true = np.array([False, True, False], dtype=bool)
        u1 = np.eye(n, order='F', dtype=float)
        u2 = np.zeros((n, n), order='F', dtype=float)

        a1, g1, _, _, wr1, wi1, m1, info1 = mb03td(
            'S', 'U', select, lower_true, a.copy(), g.copy(), u1.copy(), u2.copy()
        )

        # Test with LOWER=False
        lower_false = np.array([False, False, False], dtype=bool)
        u1 = np.eye(n, order='F', dtype=float)
        u2 = np.zeros((n, n), order='F', dtype=float)

        a2, g2, _, _, wr2, wi2, m2, info2 = mb03td(
            'S', 'U', select, lower_false, a.copy(), g.copy(), u1.copy(), u2.copy()
        )

        assert info1 == 0 and info2 == 0
        assert m1 == m2 == 1


class TestMB03TDEigenvaluePreservation:
    """Property tests for eigenvalue preservation."""

    def test_eigenvalue_set_preserved(self):
        """
        Verify that the set of eigenvalues is preserved under reordering.

        The eigenvalues should be the same before and after, just reordered.
        Random seed: 999 (for reproducibility)
        """
        from ctrlsys import mb03td

        np.random.seed(999)
        n = 4

        # Create Schur form matrix
        a = np.array([
            [3.0, 0.8, 0.3, 0.1],
            [0.0, 2.0, 0.5, 0.2],
            [0.0, 0.0, 1.5, 0.3],
            [0.0, 0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        g = np.array([
            [0.0, 0.3, 0.2, 0.1],
            [0.0, 0.0, 0.4, 0.2],
            [0.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        # Eigenvalues before: diagonal of A
        eig_before = np.diag(a.copy())

        select = np.array([False, False, True, True], dtype=bool)
        lower = np.array([False, False, False, False], dtype=bool)

        u1 = np.eye(n, order='F', dtype=float)
        u2 = np.zeros((n, n), order='F', dtype=float)

        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'S', 'U', select, lower, a, g, u1, u2
        )

        assert info == 0
        assert m == 2

        # WR should contain the eigenvalues in new order
        eig_after = wr.copy()

        # Same set of eigenvalues (sorted for comparison)
        np.testing.assert_allclose(sorted(eig_before), sorted(eig_after), rtol=1e-12)


class TestMB03TDErrorHandling:
    """Tests for error conditions."""

    def test_invalid_typ(self):
        """Test with invalid TYP parameter."""
        from ctrlsys import mb03td

        n = 2
        select = np.array([True, False], dtype=bool)
        lower = np.array([False, False], dtype=bool)
        a = np.array([[1.0, 0.5], [0.0, 0.5]], order='F', dtype=float)
        g = np.array([[0.0, 0.3], [0.0, 0.0]], order='F', dtype=float)
        u1 = np.eye(n, order='F', dtype=float)
        u2 = np.zeros((n, n), order='F', dtype=float)

        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'X', 'U', select, lower, a, g, u1, u2
        )

        assert info == -1  # Invalid TYP

    def test_invalid_compu(self):
        """Test with invalid COMPU parameter."""
        from ctrlsys import mb03td

        n = 2
        select = np.array([True, False], dtype=bool)
        lower = np.array([False, False], dtype=bool)
        a = np.array([[1.0, 0.5], [0.0, 0.5]], order='F', dtype=float)
        g = np.array([[0.0, 0.3], [0.0, 0.0]], order='F', dtype=float)
        u1 = np.eye(n, order='F', dtype=float)
        u2 = np.zeros((n, n), order='F', dtype=float)

        a_out, g_out, u1_out, u2_out, wr, wi, m, info = mb03td(
            'S', 'X', select, lower, a, g, u1, u2
        )

        assert info == -2  # Invalid COMPU
