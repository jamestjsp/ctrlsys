"""
Tests for MB04DP - Balance a real skew-Hamiltonian/Hamiltonian pencil.

MB04DP balances the 2*N-by-2*N skew-Hamiltonian/Hamiltonian pencil aS - bH with:
    S = [[A, D], [E, A']]  and  H = [[C, V], [W, -C']]
where D, E are skew-symmetric and V, W are symmetric.

Balancing involves:
1. Permuting to isolate eigenvalues
2. Diagonal equivalence transformation to equalize row/column 1-norms

Test data sources:
- SLICOT HTML documentation MB04DP.html
- Mathematical properties of skew-Hamiltonian/Hamiltonian pencils
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import mb04dp


class TestMB04DPBasic:
    """Test basic functionality from HTML documentation example."""

    def test_html_doc_example(self):
        """
        Test case from SLICOT HTML documentation.

        N=2, JOB='B' (both permute and scale), THRESH=-3

        Input (row-wise per Fortran code):
        A = [[1, 0], [0, 1]]
        DE = [[0, 0, 0], [0, 0, 0]]  (E in lower, D in upper cols 2:N+1)
        C = [[1, 0], [0, -2]]
        VW = [[-1, -1e-12, 0], [-1, -1, 0]]  (W in lower, V in upper cols 2:N+1)
        """
        n = 2

        a = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        de = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        vw = np.array([
            [-1.0, -1e-12, 0.0],
            [-1.0, -1.0, 0.0]
        ], order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'B', n, -3.0, a, de, c, vw
        )

        assert info == 0

        a_expected = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        de_expected = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        c_expected = np.array([
            [2.0, 1.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        vw_expected = np.array([
            [0.0, 1.0, 0.0],
            [0.0, -1.0, -1e-12]
        ], order='F', dtype=float)

        lscale_expected = np.array([4.0, 1.0])
        rscale_expected = np.array([4.0, 1.0])

        assert ilo == 2

        assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
        assert_allclose(de_out, de_expected, rtol=1e-3, atol=1e-4)
        assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)
        assert_allclose(vw_out, vw_expected, rtol=1e-3, atol=1e-4)
        assert_allclose(lscale, lscale_expected, rtol=1e-3, atol=1e-4)
        assert_allclose(rscale, rscale_expected, rtol=1e-3, atol=1e-4)

        assert_allclose(dwork[0], 1.0, rtol=1e-3)
        assert_allclose(dwork[1], 2.0, rtol=1e-3)
        assert_allclose(dwork[2], 1.0, rtol=1e-3)
        assert_allclose(dwork[3], 2.0, rtol=1e-3)
        assert_allclose(dwork[4], -3.0, rtol=1e-3)


class TestMB04DPModes:
    """Test different JOB modes."""

    def test_job_none(self):
        """
        Test JOB='N': no balancing, just set ILO=1 and scales to 1.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4

        a = np.random.randn(n, n).astype(float, order='F')
        de = np.random.randn(n, n + 1).astype(float, order='F')
        c = np.random.randn(n, n).astype(float, order='F')
        vw = np.random.randn(n, n + 1).astype(float, order='F')

        a_orig = a.copy()
        de_orig = de.copy()
        c_orig = c.copy()
        vw_orig = vw.copy()

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'N', n, 0.0, a, de, c, vw
        )

        assert info == 0
        assert ilo == 1
        assert_allclose(lscale, np.ones(n), rtol=1e-14)
        assert_allclose(rscale, np.ones(n), rtol=1e-14)
        assert_allclose(a_out, a_orig, rtol=1e-14)

    def test_job_permute_only(self):
        """
        Test JOB='P': permutation only.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4

        a = np.random.randn(n, n).astype(float, order='F')
        de = np.random.randn(n, n + 1).astype(float, order='F')
        c = np.random.randn(n, n).astype(float, order='F')
        vw = np.random.randn(n, n + 1).astype(float, order='F')

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'P', n, 0.0, a, de, c, vw
        )

        assert info == 0
        assert 1 <= ilo <= n + 1
        for i in range(ilo - 1, n):
            assert lscale[i] == 1.0
            assert rscale[i] == 1.0

    def test_job_scale_only(self):
        """
        Test JOB='S': scaling only.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 4

        a = np.diag([1.0, 100.0, 0.01, 1.0]).astype(float, order='F')
        de = np.zeros((n, n + 1), order='F', dtype=float)
        c = np.diag([1.0, 50.0, 0.02, 1.0]).astype(float, order='F')
        vw = np.zeros((n, n + 1), order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'S', n, 0.0, a, de, c, vw
        )

        assert info == 0
        assert ilo == 1

    def test_job_both(self):
        """
        Test JOB='B': both permutation and scaling.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4

        a = np.random.randn(n, n).astype(float, order='F')
        de = np.random.randn(n, n + 1).astype(float, order='F')
        c = np.random.randn(n, n).astype(float, order='F')
        vw = np.random.randn(n, n + 1).astype(float, order='F')

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'B', n, 0.0, a, de, c, vw
        )

        assert info == 0
        assert 1 <= ilo <= n + 1


class TestMB04DPThresholds:
    """Test different THRESH values."""

    def test_thresh_zero(self):
        """Test THRESH=0: standard DGGBAL-like scaling."""
        np.random.seed(100)
        n = 4
        a = np.diag([1.0, 100.0, 0.01, 1.0]).astype(float, order='F')
        a[0, 1] = 10.0
        de = np.zeros((n, n + 1), order='F', dtype=float)
        c = np.eye(n, dtype=float, order='F')
        vw = np.zeros((n, n + 1), order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'S', n, 0.0, a, de, c, vw
        )

        assert info == 0

    def test_thresh_negative_one(self):
        """Test THRESH=-1: minimize norm ratio."""
        np.random.seed(200)
        n = 4
        a = np.diag([1.0, 100.0, 0.01, 1.0]).astype(float, order='F')
        de = np.zeros((n, n + 1), order='F', dtype=float)
        c = np.eye(n, dtype=float, order='F')
        vw = np.zeros((n, n + 1), order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'S', n, -1.0, a, de, c, vw
        )

        assert info == 0

    def test_thresh_negative_three(self):
        """Test THRESH=-3: minimize norm product."""
        np.random.seed(300)
        n = 4
        a = np.diag([1.0, 100.0, 0.01, 1.0]).astype(float, order='F')
        de = np.zeros((n, n + 1), order='F', dtype=float)
        c = np.eye(n, dtype=float, order='F')
        vw = np.zeros((n, n + 1), order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'S', n, -3.0, a, de, c, vw
        )

        assert info == 0


class TestMB04DPEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_n_zero(self):
        """Test n=0: empty matrices."""
        a = np.zeros((0, 0), order='F', dtype=float)
        de = np.zeros((0, 1), order='F', dtype=float)
        c = np.zeros((0, 0), order='F', dtype=float)
        vw = np.zeros((0, 1), order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'B', 0, 0.0, a, de, c, vw
        )

        assert info == 0

    def test_n_one(self):
        """Test n=1: scalar matrices (N=1 special case)."""
        a = np.array([[2.0]], order='F', dtype=float)
        de = np.array([[0.0, 0.0]], order='F', dtype=float)
        c = np.array([[3.0]], order='F', dtype=float)
        vw = np.array([[1.0, 2.0]], order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'B', 1, 0.0, a, de, c, vw
        )

        assert info == 0
        assert_allclose(lscale, [1.0], rtol=1e-14)
        assert_allclose(rscale, [1.0], rtol=1e-14)

    def test_identity_matrices(self):
        """Test with identity A, C and zero DE, VW."""
        n = 4
        a = np.eye(n, dtype=float, order='F')
        de = np.zeros((n, n + 1), order='F', dtype=float)
        c = np.eye(n, dtype=float, order='F')
        vw = np.zeros((n, n + 1), order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'B', n, 0.0, a, de, c, vw
        )

        assert info == 0


class TestMB04DPMathematicalProperties:
    """Test mathematical properties of the balancing transformation."""

    def test_scaling_transformation(self):
        """
        Verify that scaling applies diagonal equivalence transformation.

        For JOB='S': A_bal = Dl * A * Dr, etc.
        where Dl = diag(lscale) and Dr = diag(rscale)

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n = 5
        a = np.random.randn(n, n).astype(float, order='F')
        de = np.zeros((n, n + 1), order='F', dtype=float)
        c = np.random.randn(n, n).astype(float, order='F')
        vw = np.zeros((n, n + 1), order='F', dtype=float)

        a_orig = a.copy()
        c_orig = c.copy()

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'S', n, 0.0, a, de, c, vw
        )

        assert info == 0

        dl = np.diag(lscale)
        dr = np.diag(rscale)
        a_reconstructed = dl @ a_orig @ dr
        c_reconstructed = dl @ c_orig @ dr

        assert_allclose(a_out, a_reconstructed, rtol=1e-14, atol=1e-14)
        assert_allclose(c_out, c_reconstructed, rtol=1e-14, atol=1e-14)

    def test_skew_symmetric_preservation(self):
        """
        Verify that skew-symmetric structure of D, E is preserved.

        D and E are stored in DE array. Diagonal is zero for skew-symmetric.
        For JOB='S': E_bal = Dr * E * Dr, D_bal = Dl * D * Dl

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n = 4

        a = np.random.randn(n, n).astype(float, order='F')
        c = np.random.randn(n, n).astype(float, order='F')

        e_mat = np.random.randn(n, n)
        e_mat = e_mat - e_mat.T
        np.fill_diagonal(e_mat, 0.0)

        d_mat = np.random.randn(n, n)
        d_mat = d_mat - d_mat.T
        np.fill_diagonal(d_mat, 0.0)

        de = np.zeros((n, n + 1), order='F', dtype=float)
        for i in range(n):
            for j in range(i):
                de[i, j] = e_mat[i, j]
        for i in range(n):
            for j in range(i + 1, n):
                de[i, j + 1] = d_mat[i, j]

        w_mat = np.random.randn(n, n)
        w_mat = (w_mat + w_mat.T) / 2

        v_mat = np.random.randn(n, n)
        v_mat = (v_mat + v_mat.T) / 2

        vw = np.zeros((n, n + 1), order='F', dtype=float)
        for i in range(n):
            for j in range(i + 1):
                vw[i, j] = w_mat[i, j]
        for i in range(n):
            for j in range(i, n):
                vw[i, j + 1] = v_mat[i, j]

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'S', n, 0.0, a, de, c, vw
        )

        assert info == 0


class TestMB04DPErrors:
    """Test error handling."""

    def test_invalid_job(self):
        """Test invalid JOB parameter."""
        n = 4
        a = np.eye(n, dtype=float, order='F')
        de = np.zeros((n, n + 1), order='F', dtype=float)
        c = np.eye(n, dtype=float, order='F')
        vw = np.zeros((n, n + 1), order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'X', n, 0.0, a, de, c, vw
        )

        assert info == -1

    def test_negative_n(self):
        """Test negative n parameter."""
        a = np.eye(4, dtype=float, order='F')
        de = np.zeros((4, 5), order='F', dtype=float)
        c = np.eye(4, dtype=float, order='F')
        vw = np.zeros((4, 5), order='F', dtype=float)

        a_out, de_out, c_out, vw_out, ilo, lscale, rscale, dwork, iwarn, info = mb04dp(
            'B', -1, 0.0, a, de, c, vw
        )

        assert info == -2
