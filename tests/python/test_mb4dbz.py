"""
Tests for MB4DBZ - Inverse balancing transformation for complex matrices.

MB4DBZ applies from the left the inverse of a balancing transformation,
computed by MB4DPZ, to the complex matrix [[V1], [sgn*V2]].
"""

import numpy as np
import pytest

from ctrlsys import mb4dbz


class TestMB4DBZBasic:
    """Basic functionality tests."""

    def test_scaling_only(self):
        """
        Test inverse scaling transformation (JOB='S').

        Applies D^{-1} scaling to V1 and V2 rows where D = diag(lscale[i]) for V1
        and D = diag(rscale[i]) for V2.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3
        m = 2
        ilo = 1

        v1 = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            complex, order='F')
        v2 = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            complex, order='F')

        lscale = np.array([2.0, 0.5, 4.0], dtype=float)
        rscale = np.array([0.25, 8.0, 1.0], dtype=float)

        v1_orig = v1.copy()
        v2_orig = v2.copy()

        v1_out, v2_out, info = mb4dbz('S', 'P', n, ilo, lscale, rscale, v1, v2)

        assert info == 0

        for i in range(n):
            np.testing.assert_allclose(
                v1_out[i, :], v1_orig[i, :] / lscale[i], rtol=1e-14)
            np.testing.assert_allclose(
                v2_out[i, :], v2_orig[i, :] / rscale[i], rtol=1e-14)

    def test_permutation_only_no_swap(self):
        """
        Test inverse permutation transformation (JOB='P') when no permutation needed.

        When LSCALE[i-1] = i (1-based index equals current position), no swap occurs.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 3
        m = 2
        ilo = 4  # ILO > N means no active rows, permutation iterates from ILO-1 to 1

        v1 = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            complex, order='F')
        v2 = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            complex, order='F')

        lscale = np.array([1.0, 2.0, 3.0], dtype=float)
        rscale = np.array([1.0, 1.0, 1.0], dtype=float)

        v1_orig = v1.copy()
        v2_orig = v2.copy()

        v1_out, v2_out, info = mb4dbz('P', 'P', n, ilo, lscale, rscale, v1, v2)

        assert info == 0
        np.testing.assert_allclose(v1_out, v1_orig, rtol=1e-14)
        np.testing.assert_allclose(v2_out, v2_orig, rtol=1e-14)

    def test_permutation_row_exchange(self):
        """
        Test row exchange in permutation (JOB='P').

        LSCALE[i-1] contains the row index K (1-based) to exchange with row I.
        If K != I, rows I and K are swapped in both V1 and V2.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 3
        m = 2
        ilo = 2  # Process rows ILO-1 down to 1 (i.e., row 1 in 1-based = row 0 in 0-based)

        v1 = np.array([
            [1.0+1.0j, 2.0+2.0j],
            [3.0+3.0j, 4.0+4.0j],
            [5.0+5.0j, 6.0+6.0j],
        ], order='F', dtype=complex)
        v2 = np.array([
            [7.0+7.0j, 8.0+8.0j],
            [9.0+9.0j, 10.0+10.0j],
            [11.0+11.0j, 12.0+12.0j],
        ], order='F', dtype=complex)

        lscale = np.array([3.0, 2.0, 3.0], dtype=float)
        rscale = np.array([1.0, 1.0, 1.0], dtype=float)

        v1_out, v2_out, info = mb4dbz('P', 'P', n, ilo, lscale, rscale, v1, v2)

        assert info == 0
        np.testing.assert_allclose(v1_out[0, :], np.array([5.0+5.0j, 6.0+6.0j]), rtol=1e-14)
        np.testing.assert_allclose(v1_out[2, :], np.array([1.0+1.0j, 2.0+2.0j]), rtol=1e-14)
        np.testing.assert_allclose(v2_out[0, :], np.array([11.0+11.0j, 12.0+12.0j]), rtol=1e-14)
        np.testing.assert_allclose(v2_out[2, :], np.array([7.0+7.0j, 8.0+8.0j]), rtol=1e-14)


class TestMB4DBZSystemSwap:
    """Test system swap (V1 <-> V2 exchange) functionality."""

    def test_system_swap_with_positive_sign(self):
        """
        Test system swap when LSCALE[i] > N (SGN='P').

        When LSCALE[i] > N, it indicates system swap: V1[K,:] <-> V2[K,:]
        where K = LSCALE[i] - N. With SGN='P', V2[K,:] is negated after swap.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 2
        m = 2
        ilo = 2

        v1 = np.array([
            [1.0+1.0j, 2.0+2.0j],
            [3.0+3.0j, 4.0+4.0j],
        ], order='F', dtype=complex)
        v2 = np.array([
            [5.0+5.0j, 6.0+6.0j],
            [7.0+7.0j, 8.0+8.0j],
        ], order='F', dtype=complex)

        lscale = np.array([3.0, 2.0], dtype=float)
        rscale = np.array([1.0, 1.0], dtype=float)

        v1_out, v2_out, info = mb4dbz('P', 'P', n, ilo, lscale, rscale, v1, v2)

        assert info == 0
        np.testing.assert_allclose(v1_out[0, :], np.array([5.0+5.0j, 6.0+6.0j]), rtol=1e-14)
        np.testing.assert_allclose(v2_out[0, :], np.array([-1.0-1.0j, -2.0-2.0j]), rtol=1e-14)

    def test_system_swap_with_negative_sign(self):
        """
        Test system swap when LSCALE[i] > N (SGN='N').

        With SGN='N', V1[K,:] is negated after swap instead of V2.

        Random seed: 101 (for reproducibility)
        """
        np.random.seed(101)
        n = 2
        m = 2
        ilo = 2

        v1 = np.array([
            [1.0+1.0j, 2.0+2.0j],
            [3.0+3.0j, 4.0+4.0j],
        ], order='F', dtype=complex)
        v2 = np.array([
            [5.0+5.0j, 6.0+6.0j],
            [7.0+7.0j, 8.0+8.0j],
        ], order='F', dtype=complex)

        lscale = np.array([3.0, 2.0], dtype=float)
        rscale = np.array([1.0, 1.0], dtype=float)

        v1_out, v2_out, info = mb4dbz('P', 'N', n, ilo, lscale, rscale, v1, v2)

        assert info == 0
        np.testing.assert_allclose(v1_out[0, :], np.array([-5.0-5.0j, -6.0-6.0j]), rtol=1e-14)
        np.testing.assert_allclose(v2_out[0, :], np.array([1.0+1.0j, 2.0+2.0j]), rtol=1e-14)


class TestMB4DBZBothTransforms:
    """Test combined permutation and scaling (JOB='B')."""

    def test_both_scaling_and_permutation(self):
        """
        Test combined inverse transformation (JOB='B').

        Applies both scaling and permutation in sequence:
        1. First inverse scaling (rows ILO..N scaled by 1/LSCALE[i] for V1, 1/RSCALE[i] for V2)
        2. Then inverse permutation (rows ILO-1 down to 1)

        Random seed: 202 (for reproducibility)
        """
        np.random.seed(202)
        n = 3
        m = 2
        ilo = 2

        v1 = np.array([
            [2.0+2.0j, 4.0+4.0j],
            [6.0+6.0j, 8.0+8.0j],
            [10.0+10.0j, 12.0+12.0j],
        ], order='F', dtype=complex)
        v2 = np.array([
            [1.0+1.0j, 2.0+2.0j],
            [3.0+3.0j, 4.0+4.0j],
            [5.0+5.0j, 6.0+6.0j],
        ], order='F', dtype=complex)

        lscale = np.array([1.0, 2.0, 4.0], dtype=float)
        rscale = np.array([1.0, 0.5, 2.0], dtype=float)

        v1_out, v2_out, info = mb4dbz('B', 'P', n, ilo, lscale, rscale, v1, v2)

        assert info == 0

        v1_scaled_row1 = np.array([6.0+6.0j, 8.0+8.0j]) / 2.0
        v1_scaled_row2 = np.array([10.0+10.0j, 12.0+12.0j]) / 4.0
        v2_scaled_row1 = np.array([3.0+3.0j, 4.0+4.0j]) / 0.5
        v2_scaled_row2 = np.array([5.0+5.0j, 6.0+6.0j]) / 2.0

        np.testing.assert_allclose(v1_out[1, :], v1_scaled_row1, rtol=1e-14)
        np.testing.assert_allclose(v1_out[2, :], v1_scaled_row2, rtol=1e-14)
        np.testing.assert_allclose(v2_out[1, :], v2_scaled_row1, rtol=1e-14)
        np.testing.assert_allclose(v2_out[2, :], v2_scaled_row2, rtol=1e-14)


class TestMB4DBZEdgeCases:
    """Edge case tests."""

    def test_job_n_no_operation(self):
        """Test JOB='N' returns immediately with no changes."""
        n = 3
        m = 2
        ilo = 1

        v1 = np.array([
            [1.0+1.0j, 2.0+2.0j],
            [3.0+3.0j, 4.0+4.0j],
            [5.0+5.0j, 6.0+6.0j],
        ], order='F', dtype=complex)
        v2 = np.array([
            [7.0+7.0j, 8.0+8.0j],
            [9.0+9.0j, 10.0+10.0j],
            [11.0+11.0j, 12.0+12.0j],
        ], order='F', dtype=complex)

        v1_orig = v1.copy()
        v2_orig = v2.copy()
        lscale = np.array([1.0, 1.0, 1.0], dtype=float)
        rscale = np.array([1.0, 1.0, 1.0], dtype=float)

        v1_out, v2_out, info = mb4dbz('N', 'P', n, ilo, lscale, rscale, v1, v2)

        assert info == 0
        np.testing.assert_allclose(v1_out, v1_orig, rtol=1e-14)
        np.testing.assert_allclose(v2_out, v2_orig, rtol=1e-14)

    def test_n_zero(self):
        """Test with N=0 (quick return case)."""
        n = 0
        m = 2
        ilo = 1

        v1 = np.zeros((0, m), dtype=complex, order='F')
        v2 = np.zeros((0, m), dtype=complex, order='F')
        lscale = np.zeros(0, dtype=float)
        rscale = np.zeros(0, dtype=float)

        v1_out, v2_out, info = mb4dbz('B', 'P', n, ilo, lscale, rscale, v1, v2)

        assert info == 0
        assert v1_out.shape == (0, m)
        assert v2_out.shape == (0, m)

    def test_m_zero(self):
        """Test with M=0 (quick return case)."""
        n = 3
        m = 0
        ilo = 1

        v1 = np.zeros((n, 0), dtype=complex, order='F')
        v2 = np.zeros((n, 0), dtype=complex, order='F')
        lscale = np.array([1.0, 2.0, 3.0], dtype=float)
        rscale = np.array([1.0, 1.0, 1.0], dtype=float)

        v1_out, v2_out, info = mb4dbz('B', 'P', n, ilo, lscale, rscale, v1, v2)

        assert info == 0
        assert v1_out.shape == (n, 0)
        assert v2_out.shape == (n, 0)


class TestMB4DBZErrorHandling:
    """Error handling tests."""

    def test_invalid_job_parameter(self):
        """Test with invalid JOB parameter."""
        n = 3
        m = 2
        ilo = 1
        v1 = np.zeros((n, m), dtype=complex, order='F')
        v2 = np.zeros((n, m), dtype=complex, order='F')
        lscale = np.array([1.0, 1.0, 1.0], dtype=float)
        rscale = np.array([1.0, 1.0, 1.0], dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            mb4dbz('X', 'P', n, ilo, lscale, rscale, v1, v2)

    def test_invalid_sgn_parameter(self):
        """Test with invalid SGN parameter."""
        n = 3
        m = 2
        ilo = 1
        v1 = np.zeros((n, m), dtype=complex, order='F')
        v2 = np.zeros((n, m), dtype=complex, order='F')
        lscale = np.array([1.0, 1.0, 1.0], dtype=float)
        rscale = np.array([1.0, 1.0, 1.0], dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            mb4dbz('B', 'X', n, ilo, lscale, rscale, v1, v2)

    def test_negative_n(self):
        """Test with negative N."""
        with pytest.raises((ValueError, RuntimeError)):
            mb4dbz('B', 'P', -1, 1, np.array([]), np.array([]),
                   np.zeros((0, 2), dtype=complex, order='F'),
                   np.zeros((0, 2), dtype=complex, order='F'))

    def test_invalid_ilo(self):
        """Test with invalid ILO (outside valid range)."""
        n = 3
        m = 2
        v1 = np.zeros((n, m), dtype=complex, order='F')
        v2 = np.zeros((n, m), dtype=complex, order='F')
        lscale = np.array([1.0, 1.0, 1.0], dtype=float)
        rscale = np.array([1.0, 1.0, 1.0], dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            mb4dbz('B', 'P', n, 0, lscale, rscale, v1, v2)

        with pytest.raises((ValueError, RuntimeError)):
            mb4dbz('B', 'P', n, n + 2, lscale, rscale, v1, v2)


class TestMB4DBZInvolutionProperty:
    """Test mathematical involution property with MB4DPZ forward transform."""

    def test_inverse_is_true_inverse(self):
        """
        Validate that applying MB4DBZ (inverse) after MB4DPZ gives identity.

        This tests the mathematical property: inv(T) * T * x = x

        Since we don't have MB4DPZ, we verify inverse scaling works correctly.
        If V was scaled by D, then inverse scaling by D gives original V.

        Random seed: 303 (for reproducibility)
        """
        np.random.seed(303)
        n = 4
        m = 3
        ilo = 1

        v1_orig = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            complex, order='F')
        v2_orig = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            complex, order='F')

        lscale = np.abs(np.random.randn(n)) + 0.1
        rscale = np.abs(np.random.randn(n)) + 0.1

        v1_scaled = v1_orig.copy()
        v2_scaled = v2_orig.copy()
        for i in range(n):
            v1_scaled[i, :] *= lscale[i]
            v2_scaled[i, :] *= rscale[i]

        v1_recovered, v2_recovered, info = mb4dbz(
            'S', 'P', n, ilo, lscale, rscale, v1_scaled, v2_scaled)

        assert info == 0
        np.testing.assert_allclose(v1_recovered, v1_orig, rtol=1e-14)
        np.testing.assert_allclose(v2_recovered, v2_orig, rtol=1e-14)
