"""
Tests for MB04WP: Generate orthogonal symplectic matrix from MB04PU output.

MB04WP generates an orthogonal symplectic matrix U defined as a product of
symplectic reflectors and Givens rotations (as returned by MB04PU).
The matrix U is returned in terms of its first N rows: U = [[U1, U2], [-U2, U1]].
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import mb04pu, mb04wp


class TestMB04WPBasic:
    """Basic functionality tests using HTML documentation example."""

    def test_html_example(self):
        """
        Test from SLICOT HTML documentation MB04PU example.

        This is the joint MB04PU + MB04WP example from the documentation.
        Input: 5x5 A matrix and 5x6 QG matrix
        Output: Orthogonal symplectic matrix U = [[U1, U2], [-U2, U1]]
        """
        n = 5
        ilo = 1

        a = np.array([
            [0.9501, 0.7621, 0.6154, 0.4057, 0.0579],
            [0.2311, 0.4565, 0.7919, 0.9355, 0.3529],
            [0.6068, 0.0185, 0.9218, 0.9169, 0.8132],
            [0.4860, 0.8214, 0.7382, 0.4103, 0.0099],
            [0.8913, 0.4447, 0.1763, 0.8936, 0.1389],
        ], dtype=float, order='F')

        qg = np.array([
            [0.4055, 0.3869, 1.3801, 0.7993, 1.2019, 0.8780],
            [0.2140, 1.4936, 0.7567, 1.7598, 1.1956, 0.9029],
            [1.0224, 1.2913, 1.0503, 1.6433, 0.9346, 1.6565],
            [1.1103, 0.9515, 0.8839, 0.7590, 0.6824, 1.1022],
            [0.7016, 1.1755, 1.1010, 1.1364, 0.3793, 0.7408],
        ], dtype=float, order='F')

        # Call MB04PU to reduce Hamiltonian to PVL form
        a_pvl, qg_pvl, cs, tau, info_pu = mb04pu(n, ilo, a, qg)
        assert info_pu == 0

        # Extract U1, U2 input from MB04PU output (lower triangular parts)
        u1_in = np.tril(a_pvl).copy(order='F')
        u2_in = np.tril(qg_pvl[:, :n]).copy(order='F')

        # Call MB04WP to generate U
        u1, u2, info = mb04wp(n, ilo, u1_in, u2_in, cs, tau)
        assert info == 0

        # Expected U from HTML doc (first 5 rows of 10x10 U matrix)
        # U = [[U1, U2], [-U2, U1]]
        # Note: orthogonal matrices have sign ambiguity - columns can be negated
        u1_expected = np.array([
            [1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [0.0000, -0.1119,  0.7763, -0.2005, -0.0001],
            [0.0000, -0.2937,  0.2320,  0.4014,  0.5541],
            [0.0000, -0.2352, -0.2243, -0.7056, -0.0500],
            [0.0000, -0.4314, -0.0354,  0.2658, -0.6061],
        ], dtype=float, order='F')

        u2_expected = np.array([
            [0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [0.0000,  0.1036, -0.2783, -0.2583,  0.4356],
            [0.0000,  0.4949,  0.1187, -0.0294, -0.3632],
            [0.0000,  0.5374,  0.3102, -0.0893,  0.0318],
            [0.0000,  0.3396, -0.3230,  0.3931,  0.0207],
        ], dtype=float, order='F')

        # Compare with sign normalization: for each column, if signs don't match,
        # negate the column before comparison
        for j in range(n):
            # Find first non-negligible element to determine sign
            for i in range(n):
                if abs(u1[i, j]) > 1e-6 and abs(u1_expected[i, j]) > 1e-6:
                    if np.sign(u1[i, j]) != np.sign(u1_expected[i, j]):
                        u1[:, j] *= -1
                        u2[:, j] *= -1
                    break

        assert_allclose(u1, u1_expected, rtol=1e-3, atol=1e-4)
        assert_allclose(u2, u2_expected, rtol=1e-3, atol=1e-4)


class TestMB04WPMathematical:
    """Mathematical property tests."""

    def test_orthogonality(self):
        """
        Verify U^T U = I (orthogonality of full symplectic matrix).

        The full U = [[U1, U2], [-U2, U1]] should satisfy U^T U = I.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        q_sym = np.random.randn(n, n)
        q_sym = (q_sym + q_sym.T) / 2
        g_sym = np.random.randn(n, n)
        g_sym = (g_sym + g_sym.T) / 2

        qg = np.zeros((n, n + 1), dtype=float, order='F')
        for i in range(n):
            for j in range(i + 1):
                qg[i, j] = q_sym[i, j]
        for i in range(n):
            for j in range(i, n):
                qg[i, j + 1] = g_sym[i, j]

        a_pvl, qg_pvl, cs, tau, info_pu = mb04pu(n, ilo, a, qg)
        assert info_pu == 0

        u1_in = np.tril(a_pvl).copy(order='F')
        u2_in = np.tril(qg_pvl[:, :n]).copy(order='F')

        u1, u2, info = mb04wp(n, ilo, u1_in, u2_in, cs, tau)
        assert info == 0

        # Build full U = [[U1, U2], [-U2, U1]]
        u_full_top = np.hstack([u1, u2])
        u_full_bot = np.hstack([-u2, u1])
        u_full = np.vstack([u_full_top, u_full_bot])

        # Check orthogonality
        utu = u_full.T @ u_full
        assert_allclose(utu, np.eye(2 * n), rtol=1e-13, atol=1e-14,
                       err_msg="U should be orthogonal: U^T U = I")

    def test_symplectic_structure(self):
        """
        Verify U^T J U = J (symplectic structure preservation).

        J = [[0, I], [-I, 0]] is the standard symplectic form.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 5
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        q_sym = np.random.randn(n, n)
        q_sym = (q_sym + q_sym.T) / 2
        g_sym = np.random.randn(n, n)
        g_sym = (g_sym + g_sym.T) / 2

        qg = np.zeros((n, n + 1), dtype=float, order='F')
        for i in range(n):
            for j in range(i + 1):
                qg[i, j] = q_sym[i, j]
        for i in range(n):
            for j in range(i, n):
                qg[i, j + 1] = g_sym[i, j]

        a_pvl, qg_pvl, cs, tau, info_pu = mb04pu(n, ilo, a, qg)
        assert info_pu == 0

        u1_in = np.tril(a_pvl).copy(order='F')
        u2_in = np.tril(qg_pvl[:, :n]).copy(order='F')

        u1, u2, info = mb04wp(n, ilo, u1_in, u2_in, cs, tau)
        assert info == 0

        # Build full U = [[U1, U2], [-U2, U1]]
        u_full_top = np.hstack([u1, u2])
        u_full_bot = np.hstack([-u2, u1])
        u_full = np.vstack([u_full_top, u_full_bot])

        # Standard symplectic form J
        j_top = np.hstack([np.zeros((n, n)), np.eye(n)])
        j_bot = np.hstack([-np.eye(n), np.zeros((n, n))])
        j_mat = np.vstack([j_top, j_bot])

        # Check U^T J U = J
        result = u_full.T @ j_mat @ u_full
        assert_allclose(result, j_mat, rtol=1e-13, atol=1e-14,
                       err_msg="U should preserve symplectic structure")

    def test_ilo_partial_transform(self):
        """
        Verify U is identity except in submatrix when ilo > 1.

        U is equal to the unit matrix except in the submatrix
        U([ilo+1:n n+ilo+1:2*n], [ilo+1:n n+ilo+1:2*n]).
        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 5
        ilo = 2

        a = np.random.randn(n, n).astype(float, order='F')
        # Make upper triangular in rows/cols 1:ilo-1
        for i in range(ilo - 1):
            for j in range(i):
                a[i, j] = 0.0

        q_sym = np.random.randn(n, n)
        q_sym = (q_sym + q_sym.T) / 2
        g_sym = np.random.randn(n, n)
        g_sym = (g_sym + g_sym.T) / 2
        # Zero Q in rows/cols 1:ilo-1
        for i in range(ilo - 1):
            for j in range(n):
                q_sym[i, j] = 0.0
                q_sym[j, i] = 0.0

        qg = np.zeros((n, n + 1), dtype=float, order='F')
        for i in range(n):
            for j in range(i + 1):
                qg[i, j] = q_sym[i, j]
        for i in range(n):
            for j in range(i, n):
                qg[i, j + 1] = g_sym[i, j]

        a_pvl, qg_pvl, cs, tau, info_pu = mb04pu(n, ilo, a, qg)
        assert info_pu == 0

        u1_in = np.tril(a_pvl).copy(order='F')
        u2_in = np.tril(qg_pvl[:, :n]).copy(order='F')

        u1, u2, info = mb04wp(n, ilo, u1_in, u2_in, cs, tau)
        assert info == 0

        # Check U1[0:ilo, 0:ilo] = identity
        assert_allclose(u1[:ilo, :ilo], np.eye(ilo), rtol=1e-14,
                       err_msg="U1[:ilo,:ilo] should be identity")
        # Check U2[0:ilo, :] = 0
        assert_allclose(u2[:ilo, :], np.zeros((ilo, n)), atol=1e-14,
                       err_msg="U2[:ilo,:] should be zero")
        # Check U1[:, 0:ilo] column ilo part
        assert_allclose(u1[ilo:, :ilo], np.zeros((n - ilo, ilo)), atol=1e-14,
                       err_msg="U1[ilo:,:ilo] should be zero")


class TestMB04WPEdgeCases:
    """Edge case tests."""

    def test_n_equals_0(self):
        """Test quick return for n=0."""
        n = 0
        ilo = 1

        u1 = np.empty((0, 0), dtype=float, order='F')
        u2 = np.empty((0, 0), dtype=float, order='F')
        cs = np.empty(0, dtype=float)
        tau = np.empty(0, dtype=float)

        u1_out, u2_out, info = mb04wp(n, ilo, u1, u2, cs, tau)

        assert info == 0

    def test_n_equals_1(self):
        """Test with minimal n=1."""
        n = 1
        ilo = 1

        a = np.array([[2.0]], dtype=float, order='F')
        qg = np.array([[1.0, 3.0]], dtype=float, order='F')

        a_pvl, qg_pvl, cs, tau, info_pu = mb04pu(n, ilo, a, qg)
        assert info_pu == 0

        u1_in = a_pvl.copy(order='F')
        u2_in = np.array([[qg_pvl[0, 0]]], dtype=float, order='F')

        u1, u2, info = mb04wp(n, ilo, u1_in, u2_in, cs, tau)

        assert info == 0
        # For n=1, U = I (identity)
        assert_allclose(u1, np.eye(1), rtol=1e-14)
        assert_allclose(u2, np.zeros((1, 1)), atol=1e-14)

    def test_ilo_equals_n(self):
        """Test with ilo=n (quick return, U=identity)."""
        n = 4
        ilo = 4

        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 8.0, 9.0],
            [0.0, 0.0, 0.0, 10.0],
        ], dtype=float, order='F')

        qg = np.zeros((n, n + 1), dtype=float, order='F')
        for i in range(n):
            qg[i, i + 1] = float(i + 1)

        a_pvl, qg_pvl, cs, tau, info_pu = mb04pu(n, ilo, a, qg)
        assert info_pu == 0

        u1_in = np.tril(a_pvl).copy(order='F')
        u2_in = np.tril(qg_pvl[:, :n]).copy(order='F')

        u1, u2, info = mb04wp(n, ilo, u1_in, u2_in, cs, tau)

        assert info == 0
        # When ilo=n, no transformation occurs, U = identity
        assert_allclose(u1, np.eye(n), rtol=1e-14)
        assert_allclose(u2, np.zeros((n, n)), atol=1e-14)


class TestMB04WPErrors:
    """Error handling tests."""

    def test_invalid_n_negative(self):
        """Test error for negative n."""
        n = -1
        ilo = 1

        u1 = np.array([[1.0]], dtype=float, order='F')
        u2 = np.array([[0.0]], dtype=float, order='F')
        cs = np.zeros(0, dtype=float)
        tau = np.zeros(0, dtype=float)

        with pytest.raises(ValueError):
            mb04wp(n, ilo, u1, u2, cs, tau)

    def test_invalid_ilo_low(self):
        """Test error for ilo < 1."""
        n = 3
        ilo = 0

        u1 = np.zeros((n, n), dtype=float, order='F')
        u2 = np.zeros((n, n), dtype=float, order='F')
        cs = np.zeros(2 * n - 2, dtype=float)
        tau = np.zeros(n - 1, dtype=float)

        with pytest.raises(ValueError):
            mb04wp(n, ilo, u1, u2, cs, tau)

    def test_invalid_ilo_high(self):
        """Test error for ilo > n."""
        n = 3
        ilo = 5

        u1 = np.zeros((n, n), dtype=float, order='F')
        u2 = np.zeros((n, n), dtype=float, order='F')
        cs = np.zeros(2 * n - 2, dtype=float)
        tau = np.zeros(n - 1, dtype=float)

        with pytest.raises(ValueError):
            mb04wp(n, ilo, u1, u2, cs, tau)
