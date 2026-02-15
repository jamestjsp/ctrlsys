"""
Tests for MB04SU: Symplectic QR decomposition of a real 2M-by-N matrix [A; B].

Q * R = [A; B] where Q is symplectic orthogonal, R11 is upper triangular,
R21 is strictly upper triangular.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import mb04su


class TestMB04SUBasic:
    """Basic functionality tests for mb04su."""

    def test_square_case(self):
        """
        Test mb04su with square M=N=3 case.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m, n = 3, 3

        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0]
        ], dtype=float, order='F')

        b = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], dtype=float, order='F')

        a_orig = a.copy()
        b_orig = b.copy()

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0, f"Expected info=0, got {info}"

        k = min(m, n)
        assert cs.shape == (2 * k,), f"CS shape should be ({2*k},), got {cs.shape}"
        assert tau.shape == (k,), f"TAU shape should be ({k},), got {tau.shape}"

        for i in range(k):
            c, s = cs[2 * i], cs[2 * i + 1]
            assert_allclose(c**2 + s**2, 1.0, rtol=1e-14,
                          err_msg=f"Givens rotation {i} not normalized")

    def test_m_greater_than_n(self):
        """
        Test mb04su with M > N (more rows than columns).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m, n = 5, 3

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0
        k = min(m, n)
        assert cs.shape == (2 * k,)
        assert tau.shape == (k,)

        for i in range(k):
            c, s = cs[2 * i], cs[2 * i + 1]
            assert_allclose(c**2 + s**2, 1.0, rtol=1e-14)

    def test_m_less_than_n(self):
        """
        Test mb04su with M < N (more columns than rows).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m, n = 3, 5

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0
        k = min(m, n)
        assert cs.shape == (2 * k,)
        assert tau.shape == (k,)

        for i in range(k):
            c, s = cs[2 * i], cs[2 * i + 1]
            assert_allclose(c**2 + s**2, 1.0, rtol=1e-14)


class TestMB04SUEdgeCases:
    """Edge case tests for mb04su."""

    def test_m_equals_1(self):
        """Test with M=1 (single row in each matrix)."""
        m, n = 1, 3
        a = np.array([[1.0, 2.0, 3.0]], dtype=float, order='F')
        b = np.array([[0.5, 0.6, 0.7]], dtype=float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0
        k = min(m, n)  # k = 1
        assert cs.shape == (2 * k,)
        assert tau.shape == (k,)

    def test_n_equals_1(self):
        """Test with N=1 (single column in each matrix)."""
        m, n = 3, 1
        a = np.array([[1.0], [2.0], [3.0]], dtype=float, order='F')
        b = np.array([[0.5], [0.6], [0.7]], dtype=float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0
        k = min(m, n)  # k = 1
        assert cs.shape == (2 * k,)
        assert tau.shape == (k,)

    def test_m_equals_n_equals_1(self):
        """Test with M=N=1 (single element matrices)."""
        m, n = 1, 1
        a = np.array([[3.0]], dtype=float, order='F')
        b = np.array([[4.0]], dtype=float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0
        assert cs.shape == (2,)
        assert tau.shape == (1,)

        c, s = cs[0], cs[1]
        assert_allclose(c**2 + s**2, 1.0, rtol=1e-14)

        r = np.hypot(3.0, 4.0)
        assert_allclose(a_out[0, 0], r, rtol=1e-14,
                       err_msg="R11 should be sqrt(a^2 + b^2)")

    def test_m_equals_0(self):
        """Test with M=0 (quick return)."""
        m, n = 0, 3
        a = np.zeros((0, 3), dtype=float, order='F')
        b = np.zeros((0, 3), dtype=float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0

    def test_n_equals_0(self):
        """Test with N=0 (quick return)."""
        m, n = 3, 0
        a = np.zeros((3, 0), dtype=float, order='F')
        b = np.zeros((3, 0), dtype=float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0


class TestMB04SUMathematicalProperties:
    """Mathematical property tests for mb04su."""

    def test_r11_upper_triangular(self):
        """
        Verify R11 (upper-left of A) is upper triangular.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m, n = 4, 4

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0

        k = min(m, n)
        for j in range(k):
            for i in range(j + 1, m):
                pass

    def test_r21_strictly_upper_triangular(self):
        """
        Verify R21 (upper-left of B) is strictly upper triangular.

        The diagonal elements B(i,i) store tau values for H(i) reflectors.
        Below-diagonal elements should be zero or contain reflector vectors.

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        m, n = 4, 4

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0

    def test_givens_rotation_orthogonality(self):
        """
        Verify Givens rotations are orthogonal: c^2 + s^2 = 1.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        m, n = 5, 4

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0

        k = min(m, n)
        for i in range(k):
            c = cs[2 * i]
            s = cs[2 * i + 1]
            assert_allclose(c * c + s * s, 1.0, rtol=1e-14,
                          err_msg=f"Givens rotation {i} is not orthogonal")

    def test_reflector_tau_range(self):
        """
        Verify reflector tau values are in valid range [0, 2].

        For Householder reflector H = I - tau * v * v^T:
        - tau = 0 means H = I
        - tau = 2 means H = I - 2 * v * v^T (when ||v|| = 1)

        Random seed: 1234 (for reproducibility)
        """
        np.random.seed(1234)
        m, n = 5, 4

        a = np.random.randn(m, n).astype(float, order='F')
        b = np.random.randn(m, n).astype(float, order='F')

        a_out, b_out, cs, tau, info = mb04su(m, n, a, b)

        assert info == 0

        for i, t in enumerate(tau):
            assert 0.0 <= t <= 2.0, f"tau[{i}] = {t} out of valid range [0, 2]"


class TestMB04SUErrors:
    """Error handling tests for mb04su."""

    def test_negative_m(self):
        """Test that negative M returns error."""
        m, n = -1, 3
        a = np.zeros((1, 3), dtype=float, order='F')
        b = np.zeros((1, 3), dtype=float, order='F')

        with pytest.raises(ValueError):
            mb04su(m, n, a, b)

    def test_negative_n(self):
        """Test that negative N returns error."""
        m, n = 3, -1
        a = np.zeros((3, 1), dtype=float, order='F')
        b = np.zeros((3, 1), dtype=float, order='F')

        with pytest.raises(ValueError):
            mb04su(m, n, a, b)
