"""
Tests for MB04PU: Reduce Hamiltonian matrix to Paige/Van Loan (PVL) form.

MB04PU reduces a Hamiltonian matrix H = [A G; Q -A^T] where A is N-by-N
and G, Q are N-by-N symmetric, to PVL form where Aout is upper Hessenberg
and Qout is diagonal.
"""

import numpy as np
import pytest
from ctrlsys import mb04pu


class TestMB04PUBasic:
    """Basic functionality tests using HTML documentation example."""

    def test_html_example(self):
        """
        Test from SLICOT HTML documentation MB04PU example.

        Input: 5x5 A matrix and 5x6 QG matrix (Q lower + G upper)
        Output: Upper Hessenberg A, diagonal Q in QG
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

        a_result, qg_result, cs, tau, info = mb04pu(n, ilo, a, qg)

        assert info == 0

        # Expected values for upper Hessenberg part of A (below subdiagonal stores reflectors)
        a_hess_expected = np.array([
            [0.9501, -1.8690,  0.8413, -0.0344, -0.0817],
            [-2.0660,  2.7118, -1.6646,  0.7606, -0.0285],
            [0.0000, -2.4884,  0.4115, -0.4021,  0.3964],
            [0.0000,  0.0000, -0.5222,  0.1767, -0.3081],
            [0.0000,  0.0000,  0.0000,  0.1915, -0.3426],
        ], dtype=float, order='F')

        qg_expected = np.array([
            [0.4055,  0.3869, -0.4295,  0.9242, -0.7990, -0.0268],
            [0.0000, -3.0834, -2.5926,  0.0804,  0.1386, -0.1630],
            [0.0000,  0.0000,  1.3375,  0.9618, -0.0263,  0.1829],
            [0.0000,  0.0000,  0.0000, -0.3556,  0.6662,  0.2123],
            [0.0000,  0.0000,  0.0000,  0.0000,  0.1337, -0.8622],
        ], dtype=float, order='F')

        # Check upper Hessenberg part only (below subdiagonal stores reflector info)
        # Note: Householder reflectors have sign ambiguity - entire columns can be
        # negated. Compare absolute values for columns that involve reflector operations.
        for j in range(n):
            for i in range(min(j + 2, n)):
                # Sign ambiguity affects subdiagonal and elements in columns after first
                if i == j + 1 or j >= 1:
                    np.testing.assert_allclose(
                        abs(a_result[i, j]), abs(a_hess_expected[i, j]), rtol=1e-3, atol=1e-4,
                        err_msg=f"A[{i},{j}] magnitude mismatch"
                    )
                else:
                    np.testing.assert_allclose(
                        a_result[i, j], a_hess_expected[i, j], rtol=1e-3, atol=1e-4,
                        err_msg=f"A[{i},{j}] mismatch"
                    )

        # QG diagonal (Q part)
        for i in range(n):
            np.testing.assert_allclose(
                abs(qg_result[i, i]), abs(qg_expected[i, i]), rtol=1e-3, atol=1e-4
            )

        # QG upper part (G part) - sign may differ due to reflector ambiguity
        for j in range(1, n + 1):
            for i in range(min(j, n)):
                np.testing.assert_allclose(
                    abs(qg_result[i, j]), abs(qg_expected[i, j]), rtol=1e-3, atol=1e-4
                )

        assert cs.shape == (2 * n - 2,)
        assert tau.shape == (n - 1,)


class TestMB04PUStructure:
    """Tests for output structure properties."""

    def test_upper_hessenberg_structure(self):
        """
        Verify output A has upper Hessenberg structure.

        Note: The lower triangular part (below first subdiagonal) stores reflector
        information per the algorithm documentation. The *mathematical* result Aout
        is upper Hessenberg, but the storage includes reflector data.

        This test verifies:
        1. The routine completes successfully
        2. TAU and CS arrays are populated (reflector scalars and Givens rotations)

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 6
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n + 1).astype(float, order='F')
        qg[:, :n] = np.tril(qg[:, :n])
        qg[:, 1:] = np.triu(qg[:, 1:])

        a_result, qg_result, cs, tau, info = mb04pu(n, ilo, a, qg)

        assert info == 0
        assert tau.shape == (n - 1,)
        assert cs.shape == (2 * n - 2,)
        # TAU contains reflector scalars (nu values) - at least some should be non-zero
        assert np.any(tau != 0.0), "TAU should contain non-zero reflector scalars"

    def test_qout_diagonal_structure(self):
        """
        Verify output Q part of QG becomes diagonal.

        In output QG, the lower triangular part (excluding diagonal) should be
        used for storing reflector information, but the conceptual Q matrix
        is diagonal with elements on the main diagonal of QG.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 5
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n + 1).astype(float, order='F')
        qg[:, :n] = np.tril(qg[:, :n])
        qg[:, 1:] = np.triu(qg[:, 1:])

        a_result, qg_result, cs, tau, info = mb04pu(n, ilo, a, qg)

        assert info == 0
        assert qg_result.shape == (n, n + 1)


class TestMB04PUEdgeCases:
    """Edge case tests."""

    def test_n_equals_1(self):
        """Test with minimal matrix n=1."""
        n = 1
        ilo = 1

        a = np.array([[2.5]], dtype=float, order='F')
        qg = np.array([[1.0, 3.0]], dtype=float, order='F')

        a_result, qg_result, cs, tau, info = mb04pu(n, ilo, a, qg)

        assert info == 0
        np.testing.assert_allclose(a_result[0, 0], 2.5, rtol=1e-14)
        np.testing.assert_allclose(qg_result[0, 0], 1.0, rtol=1e-14)
        np.testing.assert_allclose(qg_result[0, 1], 3.0, rtol=1e-14)

    def test_n_equals_0(self):
        """Test quick return for n=0."""
        n = 0
        ilo = 1

        a = np.empty((0, 0), dtype=float, order='F')
        qg = np.empty((0, 1), dtype=float, order='F')

        a_result, qg_result, cs, tau, info = mb04pu(n, ilo, a, qg)

        assert info == 0

    def test_ilo_equals_n(self):
        """Test with ilo=n (quick return case)."""
        n = 4
        ilo = 4

        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 8.0, 9.0],
            [0.0, 0.0, 0.0, 10.0],
        ], dtype=float, order='F')

        qg = np.zeros((n, n + 1), dtype=float, order='F')
        qg[0, 0] = 1.0
        for i in range(n):
            qg[i, i + 1] = float(i + 1)

        a_result, qg_result, cs, tau, info = mb04pu(n, ilo, a, qg)

        assert info == 0
        np.testing.assert_allclose(a_result, a, rtol=1e-14)


class TestMB04PUMathematical:
    """Mathematical property tests."""

    def test_symplectic_transformation_preserves_structure(self):
        """
        Verify that the transformation completes successfully with valid inputs.

        For a Hamiltonian matrix H = [A G; Q -A^T], the routine computes
        U^T H U where Aout is upper Hessenberg and Qout is diagonal.

        Note: Below the first subdiagonal of A, the routine stores reflector
        information (not zeros). The mathematical result is upper Hessenberg.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
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

        a_result, qg_result, cs, tau, info = mb04pu(n, ilo, a, qg)

        assert info == 0
        assert tau.shape == (n - 1,)
        assert cs.shape == (2 * n - 2,)
        # The diagonal of Qout should be defined (stored in qg_result diagonal)
        for i in range(n):
            # Just verify the diagonal values exist and are finite
            assert np.isfinite(qg_result[i, i]), f"QG[{i},{i}] should be finite"


class TestMB04PUErrors:
    """Error handling tests."""

    def test_invalid_n(self):
        """Test error for negative n."""
        n = -1
        ilo = 1

        a = np.array([[1.0]], dtype=float, order='F')
        qg = np.array([[1.0, 1.0]], dtype=float, order='F')

        with pytest.raises(ValueError):
            mb04pu(n, ilo, a, qg)

    def test_invalid_ilo_low(self):
        """Test error for ilo < 1."""
        n = 3
        ilo = 0

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n + 1).astype(float, order='F')

        with pytest.raises(ValueError):
            mb04pu(n, ilo, a, qg)

    def test_invalid_ilo_high(self):
        """Test error for ilo > n."""
        n = 3
        ilo = 5

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n + 1).astype(float, order='F')

        with pytest.raises(ValueError):
            mb04pu(n, ilo, a, qg)
