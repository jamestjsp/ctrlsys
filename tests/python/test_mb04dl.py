"""
Tests for SLICOT MB04DL routine.

MB04DL balances a real matrix pencil (A, B) by:
1. Permuting to isolate eigenvalues
2. Diagonal equivalence transformation to equalize row/column 1-norms
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import mb04dl


class TestMB04DLBasic:
    """Test basic functionality from HTML documentation example."""

    def test_html_doc_example(self):
        """
        Test case from SLICOT HTML documentation.

        n=4, job='B' (both permute and scale), thresh=-3
        """
        n = 4
        a = np.array([
            [1.0, 0.0, -1e-12, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0, 2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'B', n, -3.0, a, b
        )

        assert info == 0
        assert iwarn == 0
        assert ilo == 2
        assert ihi == 3

        a_expected = np.array([
            [2.0, -1.0, 0.0, -1.0],
            [0.0, 1.0, -1e-12, 0.0],
            [0.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, -2.0]
        ], order='F', dtype=float)

        b_expected = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        lscale_expected = np.array([2.0, 1.0, 1.0, 2.0])
        rscale_expected = np.array([2.0, 1.0, 1.0, 2.0])

        assert_allclose(a_out, a_expected, rtol=1e-14, atol=1e-14)
        assert_allclose(b_out, b_expected, rtol=1e-14, atol=1e-14)
        assert_allclose(lscale, lscale_expected, rtol=1e-14, atol=1e-14)
        assert_allclose(rscale, rscale_expected, rtol=1e-14, atol=1e-14)

        assert_allclose(dwork[0], 2.0, rtol=1e-3)
        assert_allclose(dwork[1], 1.0, rtol=1e-3)
        assert_allclose(dwork[2], 2.0, rtol=1e-3)
        assert_allclose(dwork[3], 1.0, rtol=1e-3)


class TestMB04DLModes:
    """Test different job modes."""

    def test_job_none(self):
        """Test JOB='N': no balancing, just set scales to 1."""
        np.random.seed(42)
        n = 4
        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, n).astype(float, order='F')
        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'N', n, 0.0, a, b
        )

        assert info == 0
        assert ilo == 1
        assert ihi == n
        assert_allclose(lscale, np.ones(n), rtol=1e-14)
        assert_allclose(rscale, np.ones(n), rtol=1e-14)
        assert_allclose(a_out, a_orig, rtol=1e-14)
        assert_allclose(b_out, b_orig, rtol=1e-14)

    def test_job_permute_only(self):
        """Test JOB='P': permutation only, no scaling."""
        n = 3
        a = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 3.0]
        ], order='F', dtype=float)
        b = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'P', n, 0.0, a, b
        )

        assert info == 0
        assert ilo >= 1
        assert ihi <= n
        for i in range(ilo - 1, ihi):
            assert lscale[i] == 1.0
            assert rscale[i] == 1.0

    def test_job_scale_only(self):
        """Test JOB='S': scaling only, no permutation."""
        np.random.seed(123)
        n = 4
        a = np.diag([1.0, 1000.0, 0.001, 1.0]).astype(float, order='F')
        b = np.eye(n, dtype=float, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'S', n, 0.0, a, b
        )

        assert info == 0
        assert ilo == 1
        assert ihi == n

    def test_job_both(self):
        """Test JOB='B': both permutation and scaling."""
        n = 4
        a = np.array([
            [1.0, 0.0, -1e-12, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0, 2.0]
        ], order='F', dtype=float)
        b = np.eye(n, dtype=float, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'B', n, 0.0, a, b
        )

        assert info == 0


class TestMB04DLThresholds:
    """Test different THRESH values."""

    def test_thresh_zero(self):
        """Test THRESH=0: standard DGGBAL-like scaling."""
        np.random.seed(456)
        n = 4
        a = np.diag([1.0, 100.0, 0.01, 1.0]).astype(float, order='F')
        a[0, 1] = 10.0
        a[2, 3] = 0.1
        b = np.eye(n, dtype=float, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'S', n, 0.0, a, b
        )

        assert info == 0

    def test_thresh_negative_one(self):
        """Test THRESH=-1: minimize norm ratio."""
        np.random.seed(789)
        n = 4
        a = np.diag([1.0, 100.0, 0.01, 1.0]).astype(float, order='F')
        b = np.eye(n, dtype=float, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'S', n, -1.0, a, b
        )

        assert info == 0

    def test_thresh_negative_two(self):
        """Test THRESH=-2: norm ratio with safety check."""
        np.random.seed(101)
        n = 4
        a = np.diag([1.0, 100.0, 0.01, 1.0]).astype(float, order='F')
        b = np.eye(n, dtype=float, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'S', n, -2.0, a, b
        )

        assert info == 0

    def test_thresh_negative_three(self):
        """Test THRESH=-3: minimize norm product."""
        np.random.seed(202)
        n = 4
        a = np.diag([1.0, 100.0, 0.01, 1.0]).astype(float, order='F')
        b = np.eye(n, dtype=float, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'S', n, -3.0, a, b
        )

        assert info == 0


class TestMB04DLEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_n_zero(self):
        """Test n=0: empty matrices."""
        a = np.zeros((0, 0), order='F', dtype=float)
        b = np.zeros((0, 0), order='F', dtype=float)

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'B', 0, 0.0, a, b
        )

        assert info == 0

    def test_n_one(self):
        """Test n=1: scalar matrices."""
        a = np.array([[5.0]], order='F', dtype=float)
        b = np.array([[2.0]], order='F', dtype=float)

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'B', 1, 0.0, a, b
        )

        assert info == 0
        assert ilo == 1
        assert ihi == 1
        assert_allclose(lscale, [1.0], rtol=1e-14)
        assert_allclose(rscale, [1.0], rtol=1e-14)

    def test_identity_matrices(self):
        """Test with identity matrices."""
        n = 4
        a = np.eye(n, dtype=float, order='F')
        b = np.eye(n, dtype=float, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'B', n, 0.0, a, b
        )

        assert info == 0


class TestMB04DLMathematicalProperties:
    """Test mathematical properties of the balancing transformation."""

    def test_equivalence_transformation(self):
        """
        Verify that balancing is an equivalence transformation.

        For scaling: A_balanced = D_l * A * D_r and B_balanced = D_l * B * D_r
        where D_l = diag(lscale[ilo-1:ihi]) and D_r = diag(rscale[ilo-1:ihi])

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n = 5
        a = np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, n).astype(float, order='F')
        a[np.abs(a) < 0.1] = 0
        b[np.abs(b) < 0.1] = 0
        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'S', n, 0.0, a, b
        )

        assert info == 0

        dl = np.diag(lscale)
        dr = np.diag(rscale)
        a_reconstructed = dl @ a_orig @ dr
        b_reconstructed = dl @ b_orig @ dr

        assert_allclose(a_out, a_reconstructed, rtol=1e-14, atol=1e-14)
        assert_allclose(b_out, b_reconstructed, rtol=1e-14, atol=1e-14)

    def test_eigenvalue_preservation(self):
        """
        Verify that eigenvalues of the pencil (A, B) are preserved.

        Balancing does not change the eigenvalues of the generalized eigenproblem.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n = 4
        a = np.random.randn(n, n).astype(float, order='F')
        b = np.eye(n, dtype=float, order='F') + 0.1 * np.random.randn(n, n)
        b = b.astype(float, order='F')

        from numpy.linalg import eigvals

        try:
            from scipy.linalg import eigvals as geigvals
            eig_before = np.sort_complex(geigvals(a.copy(), b.copy()))
        except ImportError:
            eig_before = np.sort_complex(eigvals(a.copy() @ np.linalg.inv(b.copy())))

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'B', n, 0.0, a, b
        )

        assert info == 0

        try:
            from scipy.linalg import eigvals as geigvals
            eig_after = np.sort_complex(geigvals(a_out.copy(), b_out.copy()))
        except ImportError:
            eig_after = np.sort_complex(eigvals(a_out.copy() @ np.linalg.inv(b_out.copy())))

        assert_allclose(np.abs(eig_before), np.abs(eig_after), rtol=1e-10, atol=1e-12)


class TestMB04DLErrors:
    """Test error handling."""

    def test_invalid_job(self):
        """Test invalid JOB parameter."""
        n = 4
        a = np.eye(n, dtype=float, order='F')
        b = np.eye(n, dtype=float, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'X', n, 0.0, a, b
        )

        assert info == -1

    def test_negative_n(self):
        """Test negative n parameter."""
        a = np.eye(4, dtype=float, order='F')
        b = np.eye(4, dtype=float, order='F')

        a_out, b_out, ilo, ihi, lscale, rscale, dwork, iwarn, info = mb04dl(
            'B', -1, 0.0, a, b
        )

        assert info == -2
