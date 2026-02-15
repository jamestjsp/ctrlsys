"""
Tests for MB4DPZ: Balance a complex skew-Hamiltonian/Hamiltonian pencil.

The 2*N-by-2*N pencil aS - bH has structure:
    S = [A  D ]     H = [C  V ]   where A, C are N-by-N,
        [E  A']         [W -C']

D, E are skew-Hermitian; V, W are Hermitian; ' = conjugate transpose.
"""

import numpy as np
import pytest

from ctrlsys import mb4dpz


class TestMB4DPZBasic:
    """Basic functionality tests from SLICOT HTML doc example."""

    def test_html_example(self):
        """
        Test case from SLICOT HTML documentation.

        N=2, JOB='B', THRESH=-3
        Input matrices are stored in special format:
        - DE: lower tri of E in cols 1:N, upper tri of D in cols 2:N+1
        - VW: lower tri of W in cols 1:N, upper tri of V in cols 2:N+1
        """
        n = 2
        job = 'B'
        thresh = -3.0

        # A (N-by-N, row-by-row from HTML)
        a = np.array([
            [1.0+0.5j, 0.0+0.0j],
            [0.0+0.0j, 1.0+0.5j]
        ], dtype=complex, order='F')

        # DE (N-by-(N+1))
        # Row 1:  0, 0, 0
        # Row 2:  0, 0, 0
        de = np.array([
            [0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
            [0.0+0.0j, 0.0+0.0j, 0.0+0.0j]
        ], dtype=complex, order='F')

        # C (N-by-N)
        c = np.array([
            [1.0+0.5j, 0.0+0.0j],
            [0.0+0.0j, -2.0-1.0j]
        ], dtype=complex, order='F')

        # VW (N-by-(N+1))
        # Row 1:  1, -1e-12, 0
        # Row 2: (-1+0.5j), -1, 0
        vw = np.array([
            [1.0+0.0j, -1e-12+0.0j, 0.0+0.0j],
            [-1.0+0.5j, -1.0+0.0j, 0.0+0.0j]
        ], dtype=complex, order='F')

        ilo, lscale, rscale, dwork, iwarn, info = mb4dpz(job, n, thresh, a, de, c, vw)

        assert info == 0

        # Expected output from HTML doc
        # ILO = 2
        assert ilo == 2

        # LSCALE = [4.0, 1.0] (first is permutation index 4 = N+I where I=2)
        np.testing.assert_allclose(lscale, [4.0, 1.0], rtol=1e-3)

        # RSCALE = [4.0, 1.0]
        np.testing.assert_allclose(rscale, [4.0, 1.0], rtol=1e-3)

        # Expected balanced A from HTML:
        # 1.000 - 0.500i    0.000 + 0.000i
        # 0.000 + 0.000i    1.000 + 0.500i
        a_expected = np.array([
            [1.0-0.5j, 0.0+0.0j],
            [0.0+0.0j, 1.0+0.5j]
        ], dtype=complex, order='F')
        np.testing.assert_allclose(a, a_expected, rtol=1e-3, atol=1e-10)

        # Expected balanced C from HTML:
        # 2.000 - 1.000i    1.000 - 0.500i
        # 0.000 + 0.000i    1.000 + 0.500i
        c_expected = np.array([
            [2.0-1.0j, 1.0-0.5j],
            [0.0+0.0j, 1.0+0.5j]
        ], dtype=complex, order='F')
        np.testing.assert_allclose(c, c_expected, rtol=1e-3, atol=1e-10)

        # Check DWORK values
        np.testing.assert_allclose(dwork[0], 1.118, rtol=1e-2)  # Initial S norm
        np.testing.assert_allclose(dwork[1], 2.118, rtol=1e-2)  # Initial H norm
        np.testing.assert_allclose(dwork[2], 1.118, rtol=1e-2)  # Final S norm
        np.testing.assert_allclose(dwork[3], 2.118, rtol=1e-2)  # Final H norm


class TestMB4DPZJobModes:
    """Test different JOB parameter options."""

    def test_job_none(self):
        """JOB='N': no operation, just set ILO=1 and scales=1."""
        n = 3
        np.random.seed(42)
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        de = np.zeros((n, n+1), dtype=complex, order='F')
        c = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        vw = np.zeros((n, n+1), dtype=complex, order='F')

        a_orig = a.copy()
        c_orig = c.copy()

        ilo, lscale, rscale, dwork, iwarn, info = mb4dpz('N', n, 0.0, a, de, c, vw)

        assert info == 0
        assert ilo == 1
        np.testing.assert_array_equal(lscale, np.ones(n))
        np.testing.assert_array_equal(rscale, np.ones(n))
        # Matrices unchanged
        np.testing.assert_array_equal(a, a_orig)
        np.testing.assert_array_equal(c, c_orig)

    def test_job_permute_only(self):
        """JOB='P': permute only."""
        n = 2
        # Create a matrix with isolable eigenvalue
        a = np.array([
            [1.0+0.0j, 0.0+0.0j],
            [0.0+0.0j, 2.0+0.0j]
        ], dtype=complex, order='F')
        de = np.zeros((n, n+1), dtype=complex, order='F')
        c = np.array([
            [1.0+0.0j, 0.0+0.0j],
            [0.0+0.0j, 3.0+0.0j]
        ], dtype=complex, order='F')
        vw = np.zeros((n, n+1), dtype=complex, order='F')

        ilo, lscale, rscale, dwork, iwarn, info = mb4dpz('P', n, 0.0, a, de, c, vw)

        assert info == 0
        # Scaling factors should be 1 for permuted eigenvalues or scaling part
        assert ilo >= 1


class TestMB4DPZScaling:
    """Test scaling functionality."""

    def test_job_scale_only(self):
        """JOB='S': scale only, no permutation."""
        n = 2
        a = np.array([
            [1.0+0.0j, 1e-6+0.0j],
            [1e6+0.0j, 1.0+0.0j]
        ], dtype=complex, order='F')
        de = np.zeros((n, n+1), dtype=complex, order='F')
        c = np.array([
            [2.0+0.0j, 1e-6+0.0j],
            [1e6+0.0j, 2.0+0.0j]
        ], dtype=complex, order='F')
        vw = np.zeros((n, n+1), dtype=complex, order='F')

        ilo, lscale, rscale, dwork, iwarn, info = mb4dpz('S', n, 0.0, a, de, c, vw)

        assert info == 0
        assert ilo == 1  # No permutation, so ILO stays at 1

    def test_thresh_positive(self):
        """Positive THRESH value."""
        n = 2
        np.random.seed(123)
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        de = np.zeros((n, n+1), dtype=complex, order='F')
        c = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        vw = np.zeros((n, n+1), dtype=complex, order='F')

        ilo, lscale, rscale, dwork, iwarn, info = mb4dpz('B', n, 0.1, a, de, c, vw)

        assert info == 0


class TestMB4DPZEdgeCases:
    """Test edge cases."""

    def test_n_equals_zero(self):
        """N=0: quick return."""
        n = 0
        a = np.array([], dtype=complex, order='F').reshape(0, 0)
        de = np.array([], dtype=complex, order='F').reshape(0, 1)
        c = np.array([], dtype=complex, order='F').reshape(0, 0)
        vw = np.array([], dtype=complex, order='F').reshape(0, 1)

        ilo, lscale, rscale, dwork, iwarn, info = mb4dpz('B', n, 0.0, a, de, c, vw)

        assert info == 0
        assert ilo == 1

    def test_n_equals_one(self):
        """N=1: trivial case."""
        n = 1
        a = np.array([[2.0+1.0j]], dtype=complex, order='F')
        de = np.array([[0.0+0.0j, 0.0+0.0j]], dtype=complex, order='F')
        c = np.array([[3.0+0.5j]], dtype=complex, order='F')
        vw = np.array([[1.0+0.0j, 0.5+0.0j]], dtype=complex, order='F')

        ilo, lscale, rscale, dwork, iwarn, info = mb4dpz('B', n, 0.0, a, de, c, vw)

        assert info == 0
        assert lscale[0] == 1.0
        assert rscale[0] == 1.0


class TestMB4DPZErrors:
    """Test error handling."""

    def test_invalid_job(self):
        """Invalid JOB parameter."""
        n = 2
        a = np.zeros((n, n), dtype=complex, order='F')
        de = np.zeros((n, n+1), dtype=complex, order='F')
        c = np.zeros((n, n), dtype=complex, order='F')
        vw = np.zeros((n, n+1), dtype=complex, order='F')

        with pytest.raises(ValueError, match="argument 1"):
            mb4dpz('X', n, 0.0, a, de, c, vw)

    def test_negative_n(self):
        """Negative N."""
        a = np.zeros((1, 1), dtype=complex, order='F')
        de = np.zeros((1, 2), dtype=complex, order='F')
        c = np.zeros((1, 1), dtype=complex, order='F')
        vw = np.zeros((1, 2), dtype=complex, order='F')

        with pytest.raises(ValueError, match="argument 2"):
            mb4dpz('B', -1, 0.0, a, de, c, vw)


class TestMB4DPZMathematicalProperties:
    """
    Test mathematical properties of the balancing transformation.

    The transformation preserves eigenvalues and structure.
    """

    def test_eigenvalue_preservation_simple(self):
        """
        Test that balancing preserves the pencil eigenvalues.

        For a 2N x 2N pencil, we form the full matrices and check that
        eigenvalues of the balanced pencil match the original (sorted).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 2

        # Create simple diagonal-dominant matrices for predictable eigenvalues
        a = np.diag([1.0+0.1j, 2.0+0.2j]).astype(complex, order='F')
        c = np.diag([0.5+0.05j, 1.5+0.15j]).astype(complex, order='F')
        de = np.zeros((n, n+1), dtype=complex, order='F')
        vw = np.zeros((n, n+1), dtype=complex, order='F')

        # Store original for eigenvalue computation
        a_orig = a.copy()
        c_orig = c.copy()

        ilo, lscale, rscale, dwork, iwarn, info = mb4dpz('B', n, 0.0, a, de, c, vw)

        assert info == 0

        # For diagonal matrices with zero off-diagonal blocks,
        # eigenvalues are simply ratios of diagonal elements
        # Original: eig[i] = c[i,i] / a[i,i]
        eig_orig = np.sort(np.diag(c_orig) / np.diag(a_orig))
        eig_balanced = np.sort(np.diag(c) / np.diag(a))

        np.testing.assert_allclose(eig_balanced, eig_orig, rtol=1e-12)
