"""
Tests for MB03LD: Eigenvalues and right deflating subspace of a real
skew-Hamiltonian/Hamiltonian pencil.

Test data from SLICOT HTML documentation example.
"""

import numpy as np
import pytest
from ctrlsys import mb03ld


class TestMB03LDBasic:
    """Test MB03LD basic functionality using HTML doc example."""

    def test_compq_c_orth_p(self):
        """
        Validate MB03LD with COMPQ='C', ORTH='P' using HTML example.

        Test data from SLICOT-Reference/doc/MB03LD.html.
        N=8 (M=4), computes eigenvalues and deflating subspace.
        """
        n = 8
        m = n // 2

        a = np.array([
            [3.1472, 1.3236, 4.5751, 4.5717],
            [4.0579, -4.0246, 4.6489, -0.1462],
            [-3.7301, -2.2150, -3.4239, 3.0028],
            [4.1338, 0.4688, 4.7059, -3.5811]
        ], order='F', dtype=float)

        de = np.array([
            [0.0000, 0.0000, -1.5510, -4.5974, -2.5127],
            [3.5071, 0.0000, 0.0000, 1.5961, 2.4490],
            [-3.1428, 2.5648, 0.0000, 0.0000, -0.0596],
            [3.0340, 2.4892, -1.1604, 0.0000, 0.0000]
        ], order='F', dtype=float)

        b = np.array([
            [0.6882, -3.3782, -3.3435, 1.8921],
            [-0.3061, 2.9428, 1.0198, 2.4815],
            [-4.8810, -1.8878, -2.3703, -0.4946],
            [-1.6288, 0.2853, 1.5408, -4.1618]
        ], order='F', dtype=float)

        fg = np.array([
            [-2.4013, -2.7102, 0.3834, -3.9335, 3.1730],
            [-3.1815, -2.3620, 4.9613, 4.6190, 3.6869],
            [3.6929, 0.7970, 0.4986, -4.9537, -4.1556],
            [3.5303, 1.2206, -1.4905, 0.1325, -1.0022]
        ], order='F', dtype=float)

        a_out, de_out, b_out, fg_out, neig, q, alphar, alphai, beta, info = mb03ld(
            'C', 'P', n, a, de, b, fg
        )

        assert info == 0

        # Eigenvalues are platform-independent
        alphar_expected = np.array([0.8314, -0.8314, 0.8131, 0.0000])
        alphai_expected = np.array([0.4372, 0.4372, 0.0000, 0.9164])
        beta_expected = np.array([0.7071, 0.7071, 1.4142, 2.8284])

        np.testing.assert_allclose(alphar, alphar_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(alphai, alphai_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(beta, beta_expected, rtol=1e-3, atol=1e-4)

        # A_out diagonal absolute values (signs may vary by platform due to
        # orthogonal transformation choices in DGEQP3/DORGQR)
        diag_expected = np.array([4.7460, 6.4157, 7.4626, 8.8702])
        np.testing.assert_allclose(np.abs(np.diag(a_out)), diag_expected, rtol=1e-3, atol=1e-4)

        # A_out should be upper triangular
        for i in range(1, m):
            for j in range(i):
                assert abs(a_out[i, j]) < 1e-10, f"a_out[{i},{j}] = {a_out[i, j]} should be zero"

        assert neig == 3

        # Q columns should be orthonormal (column ordering may vary)
        q_sub = q[:n, :neig]
        qtq = q_sub.T @ q_sub
        np.testing.assert_allclose(qtq, np.eye(neig), rtol=1e-12, atol=1e-13)


class TestMB03LDNoComputation:
    """Test MB03LD with COMPQ='N' (no deflating subspace)."""

    def test_compq_n(self):
        """
        Validate MB03LD with COMPQ='N' (eigenvalues only).

        Uses same input data but skips deflating subspace computation.
        """
        n = 8
        m = n // 2

        a = np.array([
            [3.1472, 1.3236, 4.5751, 4.5717],
            [4.0579, -4.0246, 4.6489, -0.1462],
            [-3.7301, -2.2150, -3.4239, 3.0028],
            [4.1338, 0.4688, 4.7059, -3.5811]
        ], order='F', dtype=float)

        de = np.array([
            [0.0000, 0.0000, -1.5510, -4.5974, -2.5127],
            [3.5071, 0.0000, 0.0000, 1.5961, 2.4490],
            [-3.1428, 2.5648, 0.0000, 0.0000, -0.0596],
            [3.0340, 2.4892, -1.1604, 0.0000, 0.0000]
        ], order='F', dtype=float)

        b = np.array([
            [0.6882, -3.3782, -3.3435, 1.8921],
            [-0.3061, 2.9428, 1.0198, 2.4815],
            [-4.8810, -1.8878, -2.3703, -0.4946],
            [-1.6288, 0.2853, 1.5408, -4.1618]
        ], order='F', dtype=float)

        fg = np.array([
            [-2.4013, -2.7102, 0.3834, -3.9335, 3.1730],
            [-3.1815, -2.3620, 4.9613, 4.6190, 3.6869],
            [3.6929, 0.7970, 0.4986, -4.9537, -4.1556],
            [3.5303, 1.2206, -1.4905, 0.1325, -1.0022]
        ], order='F', dtype=float)

        a_out, de_out, b_out, fg_out, neig, q, alphar, alphai, beta, info = mb03ld(
            'N', 'P', n, a, de, b, fg
        )

        assert info == 0
        assert neig == 0

        alphar_expected = np.array([0.8314, -0.8314, 0.8131, 0.0000])
        alphai_expected = np.array([0.4372, 0.4372, 0.0000, 0.9164])
        beta_expected = np.array([0.7071, 0.7071, 1.4142, 2.8284])

        np.testing.assert_allclose(alphar, alphar_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(alphai, alphai_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(beta, beta_expected, rtol=1e-3, atol=1e-4)


class TestMB03LDOrthMethods:
    """Test different orthogonalization methods."""

    def test_orth_s_svd(self):
        """
        Validate MB03LD with ORTH='S' (SVD orthogonalization).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 8
        m = n // 2

        a = np.array([
            [3.1472, 1.3236, 4.5751, 4.5717],
            [4.0579, -4.0246, 4.6489, -0.1462],
            [-3.7301, -2.2150, -3.4239, 3.0028],
            [4.1338, 0.4688, 4.7059, -3.5811]
        ], order='F', dtype=float)

        de = np.array([
            [0.0000, 0.0000, -1.5510, -4.5974, -2.5127],
            [3.5071, 0.0000, 0.0000, 1.5961, 2.4490],
            [-3.1428, 2.5648, 0.0000, 0.0000, -0.0596],
            [3.0340, 2.4892, -1.1604, 0.0000, 0.0000]
        ], order='F', dtype=float)

        b = np.array([
            [0.6882, -3.3782, -3.3435, 1.8921],
            [-0.3061, 2.9428, 1.0198, 2.4815],
            [-4.8810, -1.8878, -2.3703, -0.4946],
            [-1.6288, 0.2853, 1.5408, -4.1618]
        ], order='F', dtype=float)

        fg = np.array([
            [-2.4013, -2.7102, 0.3834, -3.9335, 3.1730],
            [-3.1815, -2.3620, 4.9613, 4.6190, 3.6869],
            [3.6929, 0.7970, 0.4986, -4.9537, -4.1556],
            [3.5303, 1.2206, -1.4905, 0.1325, -1.0022]
        ], order='F', dtype=float)

        a_out, de_out, b_out, fg_out, neig, q, alphar, alphai, beta, info = mb03ld(
            'C', 'S', n, a, de, b, fg
        )

        assert info == 0 or info == 5

        alphar_expected = np.array([0.8314, -0.8314, 0.8131, 0.0000])
        alphai_expected = np.array([0.4372, 0.4372, 0.0000, 0.9164])
        beta_expected = np.array([0.7071, 0.7071, 1.4142, 2.8284])

        np.testing.assert_allclose(alphar, alphar_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(alphai, alphai_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(beta, beta_expected, rtol=1e-3, atol=1e-4)


class TestMB03LDMathematicalProperties:
    """Test mathematical properties of MB03LD output."""

    def test_q_orthogonality(self):
        """
        Validate that deflating subspace Q has orthonormal columns.

        Property: Q^T * Q = I (up to machine precision)
        """
        n = 8
        m = n // 2

        a = np.array([
            [3.1472, 1.3236, 4.5751, 4.5717],
            [4.0579, -4.0246, 4.6489, -0.1462],
            [-3.7301, -2.2150, -3.4239, 3.0028],
            [4.1338, 0.4688, 4.7059, -3.5811]
        ], order='F', dtype=float)

        de = np.array([
            [0.0000, 0.0000, -1.5510, -4.5974, -2.5127],
            [3.5071, 0.0000, 0.0000, 1.5961, 2.4490],
            [-3.1428, 2.5648, 0.0000, 0.0000, -0.0596],
            [3.0340, 2.4892, -1.1604, 0.0000, 0.0000]
        ], order='F', dtype=float)

        b = np.array([
            [0.6882, -3.3782, -3.3435, 1.8921],
            [-0.3061, 2.9428, 1.0198, 2.4815],
            [-4.8810, -1.8878, -2.3703, -0.4946],
            [-1.6288, 0.2853, 1.5408, -4.1618]
        ], order='F', dtype=float)

        fg = np.array([
            [-2.4013, -2.7102, 0.3834, -3.9335, 3.1730],
            [-3.1815, -2.3620, 4.9613, 4.6190, 3.6869],
            [3.6929, 0.7970, 0.4986, -4.9537, -4.1556],
            [3.5303, 1.2206, -1.4905, 0.1325, -1.0022]
        ], order='F', dtype=float)

        a_out, de_out, b_out, fg_out, neig, q, alphar, alphai, beta, info = mb03ld(
            'C', 'P', n, a, de, b, fg
        )

        assert info == 0
        assert neig > 0

        q_sub = q[:n, :neig]
        qtq = q_sub.T @ q_sub
        np.testing.assert_allclose(qtq, np.eye(neig), rtol=1e-13, atol=1e-14)

    def test_aout_upper_triangular(self):
        """
        Validate that Aout is upper triangular.
        """
        n = 8
        m = n // 2

        a = np.array([
            [3.1472, 1.3236, 4.5751, 4.5717],
            [4.0579, -4.0246, 4.6489, -0.1462],
            [-3.7301, -2.2150, -3.4239, 3.0028],
            [4.1338, 0.4688, 4.7059, -3.5811]
        ], order='F', dtype=float)

        de = np.array([
            [0.0000, 0.0000, -1.5510, -4.5974, -2.5127],
            [3.5071, 0.0000, 0.0000, 1.5961, 2.4490],
            [-3.1428, 2.5648, 0.0000, 0.0000, -0.0596],
            [3.0340, 2.4892, -1.1604, 0.0000, 0.0000]
        ], order='F', dtype=float)

        b = np.array([
            [0.6882, -3.3782, -3.3435, 1.8921],
            [-0.3061, 2.9428, 1.0198, 2.4815],
            [-4.8810, -1.8878, -2.3703, -0.4946],
            [-1.6288, 0.2853, 1.5408, -4.1618]
        ], order='F', dtype=float)

        fg = np.array([
            [-2.4013, -2.7102, 0.3834, -3.9335, 3.1730],
            [-3.1815, -2.3620, 4.9613, 4.6190, 3.6869],
            [3.6929, 0.7970, 0.4986, -4.9537, -4.1556],
            [3.5303, 1.2206, -1.4905, 0.1325, -1.0022]
        ], order='F', dtype=float)

        a_out, de_out, b_out, fg_out, neig, q, alphar, alphai, beta, info = mb03ld(
            'C', 'P', n, a, de, b, fg
        )

        assert info == 0

        for i in range(1, m):
            for j in range(i):
                assert abs(a_out[i, j]) < 1e-10, f"a_out[{i},{j}] = {a_out[i, j]} should be zero"


class TestMB03LDInterface:
    """Test MB03LD interface (no numerical validation)."""

    def test_compq_n_returns_without_crash(self):
        """
        Verify MB03LD with COMPQ='N' runs without crash.

        Numerical results not validated due to MB04BD dependency issues.
        """
        n = 8
        m = n // 2

        a = np.array([
            [3.1472, 1.3236, 4.5751, 4.5717],
            [4.0579, -4.0246, 4.6489, -0.1462],
            [-3.7301, -2.2150, -3.4239, 3.0028],
            [4.1338, 0.4688, 4.7059, -3.5811]
        ], order='F', dtype=float)

        de = np.array([
            [0.0000, 0.0000, -1.5510, -4.5974, -2.5127],
            [3.5071, 0.0000, 0.0000, 1.5961, 2.4490],
            [-3.1428, 2.5648, 0.0000, 0.0000, -0.0596],
            [3.0340, 2.4892, -1.1604, 0.0000, 0.0000]
        ], order='F', dtype=float)

        b = np.array([
            [0.6882, -3.3782, -3.3435, 1.8921],
            [-0.3061, 2.9428, 1.0198, 2.4815],
            [-4.8810, -1.8878, -2.3703, -0.4946],
            [-1.6288, 0.2853, 1.5408, -4.1618]
        ], order='F', dtype=float)

        fg = np.array([
            [-2.4013, -2.7102, 0.3834, -3.9335, 3.1730],
            [-3.1815, -2.3620, 4.9613, 4.6190, 3.6869],
            [3.6929, 0.7970, 0.4986, -4.9537, -4.1556],
            [3.5303, 1.2206, -1.4905, 0.1325, -1.0022]
        ], order='F', dtype=float)

        a_out, de_out, b_out, fg_out, neig, q, alphar, alphai, beta, info = mb03ld(
            'N', 'P', n, a, de, b, fg
        )

        assert info == 0 or info == 5  # 0=success, 5=warning about accuracy
        assert neig == 0  # COMPQ='N' should not compute neig
        assert alphar.shape == (m,)
        assert alphai.shape == (m,)
        assert beta.shape == (m,)

    def test_compq_c_returns_without_crash(self):
        """
        Verify MB03LD with COMPQ='C' runs without crash.

        Numerical results not validated due to MB04BD dependency issues.
        """
        n = 8
        m = n // 2

        a = np.array([
            [3.1472, 1.3236, 4.5751, 4.5717],
            [4.0579, -4.0246, 4.6489, -0.1462],
            [-3.7301, -2.2150, -3.4239, 3.0028],
            [4.1338, 0.4688, 4.7059, -3.5811]
        ], order='F', dtype=float)

        de = np.array([
            [0.0000, 0.0000, -1.5510, -4.5974, -2.5127],
            [3.5071, 0.0000, 0.0000, 1.5961, 2.4490],
            [-3.1428, 2.5648, 0.0000, 0.0000, -0.0596],
            [3.0340, 2.4892, -1.1604, 0.0000, 0.0000]
        ], order='F', dtype=float)

        b = np.array([
            [0.6882, -3.3782, -3.3435, 1.8921],
            [-0.3061, 2.9428, 1.0198, 2.4815],
            [-4.8810, -1.8878, -2.3703, -0.4946],
            [-1.6288, 0.2853, 1.5408, -4.1618]
        ], order='F', dtype=float)

        fg = np.array([
            [-2.4013, -2.7102, 0.3834, -3.9335, 3.1730],
            [-3.1815, -2.3620, 4.9613, 4.6190, 3.6869],
            [3.6929, 0.7970, 0.4986, -4.9537, -4.1556],
            [3.5303, 1.2206, -1.4905, 0.1325, -1.0022]
        ], order='F', dtype=float)

        a_out, de_out, b_out, fg_out, neig, q, alphar, alphai, beta, info = mb03ld(
            'C', 'P', n, a, de, b, fg
        )

        # Structural checks only (numerical accuracy not validated)
        assert info in [0, 1, 2, 5]  # Various possible outcomes
        assert alphar.shape == (m,)
        assert alphai.shape == (m,)
        assert beta.shape == (m,)
        assert q.shape[0] == 2 * n


class TestMB03LDEdgeCases:
    """Test edge cases for MB03LD."""

    def test_n_zero(self):
        """Test with N=0 (quick return)."""
        n = 0
        m = 0

        a = np.array([], order='F', dtype=float).reshape(0, 0)
        de = np.array([], order='F', dtype=float).reshape(0, 1)
        b = np.array([], order='F', dtype=float).reshape(0, 0)
        fg = np.array([], order='F', dtype=float).reshape(0, 1)

        a_out, de_out, b_out, fg_out, neig, q, alphar, alphai, beta, info = mb03ld(
            'N', 'P', n, a, de, b, fg
        )

        assert info == 0
        assert neig == 0


class TestMB03LDErrorHandling:
    """Test error handling for MB03LD."""

    def test_invalid_compq(self):
        """Test with invalid COMPQ parameter."""
        n = 8
        m = n // 2

        a = np.zeros((m, m), order='F', dtype=float)
        de = np.zeros((m, m + 1), order='F', dtype=float)
        b = np.zeros((m, m), order='F', dtype=float)
        fg = np.zeros((m, m + 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb03ld('X', 'P', n, a, de, b, fg)

    def test_invalid_orth(self):
        """Test with invalid ORTH parameter when COMPQ='C'."""
        n = 8
        m = n // 2

        a = np.zeros((m, m), order='F', dtype=float)
        de = np.zeros((m, m + 1), order='F', dtype=float)
        b = np.zeros((m, m), order='F', dtype=float)
        fg = np.zeros((m, m + 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb03ld('C', 'X', n, a, de, b, fg)

    def test_odd_n(self):
        """Test with odd N (should fail, N must be even)."""
        n = 7

        a = np.zeros((3, 3), order='F', dtype=float)
        de = np.zeros((3, 4), order='F', dtype=float)
        b = np.zeros((3, 3), order='F', dtype=float)
        fg = np.zeros((3, 4), order='F', dtype=float)

        with pytest.raises(ValueError):
            mb03ld('N', 'P', n, a, de, b, fg)
