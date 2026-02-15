# SPDX-License-Identifier: BSD-3-Clause
"""Tests for mb4dlz - balance complex pencil (A,B)."""

import numpy as np
import pytest

from ctrlsys import mb4dlz


class TestMb4dlz:
    """Test suite for mb4dlz complex pencil balancing."""

    def test_basic_from_html_doc(self):
        """
        Test basic functionality using SLICOT HTML doc example.

        N=4, JOB='B' (both permute and scale), THRESH=-3.
        """
        n = 4
        # Input A (row-wise from HTML doc)
        a = np.array([
            [1+0.5j, 0, -1e-12, 0],
            [0, -2-1j, 0, 0],
            [1, -1-0.5j, -1+0.5j, 0],
            [-1+0.5j, -1, 0, 2-1j]
        ], order='F', dtype=np.complex128)

        # Input B (row-wise from HTML doc)
        b = np.array([
            [1+0.5j, 0, 0, 0],
            [0, 1+0.5j, 0, 0],
            [0, 0, 1-0.5j, 0],
            [0, 0, 0, 1-0.5j]
        ], order='F', dtype=np.complex128)

        # First test with simple job='N' to ensure basic functionality
        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb4dlz(
            'N', n, 0.0, a.copy(), b.copy()
        )
        assert info == 0, f"Job N failed with info={info}"
        assert ilo == 1
        assert ihi == n

        # Now test with job='B' and thresh=-3
        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb4dlz(
            'B', n, -3.0, a.copy(), b.copy()
        )

        assert info == 0, f"Job B failed with info={info}"
        assert iwarn == 0

        # Expected output A (from HTML doc)
        a_expected = np.array([
            [2-1j, -1+0.5j, 0, -1],
            [0, 1+0.5j, -1e-12, 0],
            [0, 1, -1+0.5j, -1-0.5j],
            [0, 0, 0, -2-1j]
        ], order='F', dtype=np.complex128)

        # Expected output B (from HTML doc)
        b_expected = np.array([
            [1-0.5j, 0, 0, 0],
            [0, 1+0.5j, 0, 0],
            [0, 0, 1-0.5j, 0],
            [0, 0, 0, 1+0.5j]
        ], order='F', dtype=np.complex128)

        # Expected ILO=2, IHI=3
        # Expected LSCALE = [2.0, 1.0, 1.0, 2.0]
        # Expected RSCALE = [2.0, 1.0, 1.0, 2.0]
        assert ilo == 2
        assert ihi == 3

        np.testing.assert_allclose(lscale, [2.0, 1.0, 1.0, 2.0], rtol=1e-3)
        np.testing.assert_allclose(rscale, [2.0, 1.0, 1.0, 2.0], rtol=1e-3)
        np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-10)
        np.testing.assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-10)

        # Check initial/final norms from dwork
        np.testing.assert_allclose(dwork[0], 2.118, rtol=1e-2)  # Initial norm A
        np.testing.assert_allclose(dwork[1], 1.118, rtol=1e-2)  # Initial norm B
        np.testing.assert_allclose(dwork[2], 2.118, rtol=1e-2)  # Final norm A
        np.testing.assert_allclose(dwork[3], 1.118, rtol=1e-2)  # Final norm B

    def test_permute_only(self):
        """
        Test permute only mode (JOB='P').

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3

        # Create a matrix with isolatable eigenvalue structure
        a = np.array([
            [1+1j, 0, 0],
            [2+0j, 3+2j, 0],
            [0, 4-1j, 5+0j]
        ], order='F', dtype=np.complex128)

        b = np.array([
            [1+0j, 0, 0],
            [0, 1+0j, 0],
            [0, 0, 1+0j]
        ], order='F', dtype=np.complex128)

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb4dlz(
            'P', n, 0.0, a, b
        )

        assert info == 0
        assert iwarn == 0
        # Permute only doesn't do scaling, so check lscale/rscale are permutation indices or 1s

    def test_scale_only(self):
        """
        Test scale only mode (JOB='S').

        Random seed: 123 (for reproducibility)
        """
        n = 3

        # Create a badly scaled matrix
        a = np.array([
            [1e6+1j, 1e-6, 0],
            [1e-6, 1e6+2j, 1e-6],
            [0, 1e-6, 1e6+3j]
        ], order='F', dtype=np.complex128)

        b = np.array([
            [1+0j, 0, 0],
            [0, 1+0j, 0],
            [0, 0, 1+0j]
        ], order='F', dtype=np.complex128)

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb4dlz(
            'S', n, 0.0, a, b
        )

        assert info == 0
        # JOB='S' means no permutation: ILO=1, IHI=N
        assert ilo == 1
        assert ihi == n

    def test_no_operation(self):
        """Test no operation mode (JOB='N')."""
        n = 3

        a = np.array([
            [1+1j, 2+0j, 3+0j],
            [4+0j, 5+1j, 6+0j],
            [7+0j, 8+0j, 9+1j]
        ], order='F', dtype=np.complex128)

        b = np.array([
            [1+0j, 0, 0],
            [0, 1+0j, 0],
            [0, 0, 1+0j]
        ], order='F', dtype=np.complex128)

        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb4dlz(
            'N', n, 0.0, a, b
        )

        assert info == 0
        assert ilo == 1
        assert ihi == n
        # JOB='N' should leave matrices unchanged and set scales to 1
        np.testing.assert_array_equal(lscale, np.ones(n))
        np.testing.assert_array_equal(rscale, np.ones(n))

    def test_n_equals_zero(self):
        """Test edge case N=0."""
        n = 0
        a = np.array([], dtype=np.complex128).reshape(0, 0, order='F')
        b = np.array([], dtype=np.complex128).reshape(0, 0, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb4dlz(
            'B', n, 0.0, a, b
        )

        assert info == 0

    def test_n_equals_one(self):
        """Test edge case N=1."""
        n = 1
        a = np.array([[2+1j]], order='F', dtype=np.complex128)
        b = np.array([[1+0j]], order='F', dtype=np.complex128)

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb4dlz(
            'B', n, 0.0, a, b
        )

        assert info == 0
        assert len(lscale) == 1
        assert len(rscale) == 1
        # For N=1, scaling should be 1
        np.testing.assert_allclose(lscale, [1.0], rtol=1e-14)
        np.testing.assert_allclose(rscale, [1.0], rtol=1e-14)

    def test_error_invalid_job(self):
        """Test error handling for invalid JOB parameter."""
        n = 2
        a = np.array([[1+0j, 0], [0, 1+0j]], order='F', dtype=np.complex128)
        b = np.array([[1+0j, 0], [0, 1+0j]], order='F', dtype=np.complex128)

        with pytest.raises(ValueError, match="argument 1"):
            mb4dlz('X', n, 0.0, a, b)

    def test_error_negative_n(self):
        """Test error handling for negative N."""
        n = -1
        a = np.array([[1+0j]], order='F', dtype=np.complex128)
        b = np.array([[1+0j]], order='F', dtype=np.complex128)

        with pytest.raises(ValueError, match="argument 2"):
            mb4dlz('B', n, 0.0, a, b)

    def test_balancing_property_preserves_eigenvalues(self):
        """
        Test mathematical property: balancing preserves generalized eigenvalues.

        For pencil (A,B), eigenvalues lambda satisfy det(A - lambda*B) = 0.
        After balancing (A',B') = (Dl*A*Dr, Dl*B*Dr), eigenvalues are preserved.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 4

        # Create random complex matrices
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
            np.complex128, order='F'
        )
        b = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
            np.complex128, order='F'
        )

        # Compute eigenvalues before balancing
        eig_before = np.linalg.eigvals(np.linalg.solve(b, a))

        a_bal, b_bal, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb4dlz(
            'B', n, 0.0, a, b
        )

        assert info == 0

        # Compute eigenvalues after balancing
        eig_after = np.linalg.eigvals(np.linalg.solve(b_bal, a_bal))

        # Sort by absolute value for comparison
        eig_before_sorted = np.sort(np.abs(eig_before))
        eig_after_sorted = np.sort(np.abs(eig_after))

        np.testing.assert_allclose(
            eig_before_sorted, eig_after_sorted, rtol=1e-10
        )

    def test_thresh_negative_one(self):
        """
        Test THRESH=-1: minimize norm ratio.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 3

        # Badly scaled matrices
        a = np.array([
            [1e4+1j, 1e-4, 0],
            [1, 1e4+0j, 1],
            [0, 1e-4, 1e4+2j]
        ], order='F', dtype=np.complex128)

        b = np.array([
            [1+0j, 0, 0],
            [0, 1e-4+0j, 0],
            [0, 0, 1e4+0j]
        ], order='F', dtype=np.complex128)

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb4dlz(
            'B', n, -1.0, a, b
        )

        assert info == 0
