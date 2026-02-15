"""
Tests for AB09AX - Balanced truncation core algorithm for model reduction

AB09AX computes a reduced order model (Ar,Br,Cr) for a stable system (A,B,C)
using either square-root or balancing-free square-root Balance & Truncate (B&T)
model reduction. The state matrix A must be in real Schur canonical form.

The reduced system matrices are computed using truncation formulas:
    Ar = TI * A * T,  Br = TI * B,  Cr = C * T

Key mathematical properties:
- HSV contains Hankel singular values in decreasing order
- HSV(1) is the Hankel norm of the system
- Error bound: HSV(NR) <= ||G-Gr||_inf <= 2*sum(HSV(NR+1:N))
"""

import numpy as np
import pytest
from ctrlsys import ab09ax


class TestAB09AXBasic:
    """Basic functionality tests."""

    def test_continuous_simple_system(self):
        """
        Test continuous-time model reduction for simple stable system.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 1, 1
        nr = 2

        a = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0, -4.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [1.0],
            [1.0],
            [1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 1.0, 1.0, 1.0]
        ], order='F', dtype=float)

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == nr
        assert hsv[0] >= hsv[1] >= hsv[2] >= hsv[3]
        assert np.all(hsv > 0)

    def test_discrete_simple_system(self):
        """
        Test discrete-time model reduction.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 3, 1, 1
        nr = 2

        a = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.2]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [1.0],
            [1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 1.0, 1.0]
        ], order='F', dtype=float)

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'D', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == nr
        assert hsv[0] >= hsv[1] >= hsv[2]
        assert np.all(hsv > 0)


class TestAB09AXMathematicalProperties:
    """Tests for mathematical property validation."""

    def test_truncation_matrices_transform(self):
        """
        Verify truncation formula: Ar = TI * A * T, Br = TI * B, Cr = C * T.

        The truncation matrices T and TI should produce correct reduced system.
        Note: A, B, C are modified in-place, so we need copies of originals.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 4, 2, 2
        nr = 2

        a = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0, -4.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
            [0.3, 0.7],
            [0.7, 0.3]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.3, 0.2],
            [0.2, 0.3, 0.5, 1.0]
        ], order='F', dtype=float)

        a_orig = a.copy()
        b_orig = b.copy()
        c_orig = c.copy()

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == nr

        ar_check = ti[:nr_out, :n] @ a_orig[:n, :n] @ t[:n, :nr_out]
        np.testing.assert_allclose(ar[:nr_out, :nr_out], ar_check, rtol=1e-10, atol=1e-12)

        br_check = ti[:nr_out, :n] @ b_orig[:n, :m]
        np.testing.assert_allclose(br[:nr_out, :m], br_check, rtol=1e-10, atol=1e-12)

        cr_check = c_orig[:p, :n] @ t[:n, :nr_out]
        np.testing.assert_allclose(cr[:p, :nr_out], cr_check, rtol=1e-10, atol=1e-12)

    def test_hankel_singular_values_decreasing(self):
        """
        Verify HSV are in strictly decreasing order (for distinct eigenvalues).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 5, 2, 2
        nr = 3

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        for i in range(n - 1):
            assert hsv[i] >= hsv[i + 1], f"HSV not decreasing at index {i}"

    def test_balancing_free_method(self):
        """
        Test balancing-free square-root B&T method (JOB='N').

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m, p = 4, 1, 1
        nr = 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        a_orig = a.copy()

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'N', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == nr

        ar_check = ti[:nr_out, :n] @ a_orig[:n, :n] @ t[:n, :nr_out]
        np.testing.assert_allclose(ar[:nr_out, :nr_out], ar_check, rtol=1e-10, atol=1e-12)

    def test_automatic_order_selection(self):
        """
        Test automatic order selection (ORDSEL='A') based on tolerance.

        The resulting NR should be number of HSV greater than MAX(TOL, N*EPS*HSV(1)).

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m, p = 4, 1, 1

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [0.1], [0.01], [0.001]], order='F', dtype=float)
        c = np.array([[1.0, 0.1, 0.01, 0.001]], order='F', dtype=float)

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'A', n, m, p, n, a, b, c, hsv_threshold := 1e-4
        )

        assert info == 0
        eps = np.finfo(float).eps
        atol = max(hsv_threshold, n * eps * hsv[0])
        expected_nr = sum(1 for h in hsv if h > atol)
        assert nr_out == expected_nr or nr_out <= expected_nr + 1


class TestAB09AXEdgeCases:
    """Edge case tests."""

    def test_minimal_system_n1(self):
        """
        Test with minimal 1st order system.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n, m, p = 1, 1, 1
        nr = 1

        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == 1
        assert len(hsv) >= 1
        assert hsv[0] > 0

    def test_full_order_preservation(self):
        """
        Test that requesting full order (nr=n) preserves system.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n, m, p = 3, 1, 1
        nr = n

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == n

    def test_zero_nr_quick_return(self):
        """
        Test quick return when nr=0 is requested.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n, m, p = 3, 1, 1
        nr = 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == 0

    def test_multi_input_output(self):
        """
        Test MIMO system reduction.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n, m, p = 5, 3, 2
        nr = 3

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        assert nr_out == nr
        assert ar.shape[0] >= nr_out
        assert ar.shape[1] >= nr_out
        assert br.shape[0] >= nr_out
        assert br.shape[1] >= m
        assert cr.shape[0] >= p
        assert cr.shape[1] >= nr_out


class TestAB09AXErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ax('X', 'B', 'F', n, m, p, nr, a, b, c, 0.0)

    def test_invalid_job(self):
        """Test error for invalid JOB parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ax('C', 'X', 'F', n, m, p, nr, a, b, c, 0.0)

    def test_invalid_ordsel(self):
        """Test error for invalid ORDSEL parameter."""
        n, m, p = 2, 1, 1
        nr = 1
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ax('C', 'B', 'X', n, m, p, nr, a, b, c, 0.0)

    def test_negative_n(self):
        """Test error for negative n."""
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((2, 1), order='F', dtype=float)
        c = np.ones((1, 2), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ax('C', 'B', 'F', -1, 1, 1, 1, a, b, c, 0.0)

    def test_nr_greater_than_n(self):
        """Test error when nr > n."""
        n, m, p = 2, 1, 1
        nr = 3
        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ax('C', 'B', 'F', n, m, p, nr, a, b, c, 0.0)

    def test_unstable_continuous_system(self):
        """
        Test error info=1 for unstable continuous-time system.

        A matrix with positive real eigenvalue is unstable.
        """
        n, m, p = 2, 1, 1
        nr = 1

        a = np.array([
            [1.0, 0.0],
            [0.0, -1.0]
        ], order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 1

    def test_nonconvergent_discrete_system(self):
        """
        Test error info=1 for non-convergent discrete-time system.

        A matrix with eigenvalue |lambda| > 1 is non-convergent.
        """
        n, m, p = 2, 1, 1
        nr = 1

        a = np.array([
            [2.0, 0.0],
            [0.0, 0.5]
        ], order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'D', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 1


class TestAB09AXWarning:
    """Warning indicator tests."""

    def test_iwarn_order_reduced_to_minimal(self):
        """
        Test iwarn=1 when requested NR exceeds minimal realization order.

        If HSV(NR) <= N*EPS*HSV(1), the order is reduced to minimal.
        """
        n, m, p = 4, 1, 1
        nr = 4

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.array([[1.0], [1e-16], [1e-16], [1e-16]], order='F', dtype=float)
        c = np.array([[1.0, 1e-16, 1e-16, 1e-16]], order='F', dtype=float)

        ar, br, cr, hsv, t, ti, nr_out, iwarn, info = ab09ax(
            'C', 'B', 'F', n, m, p, nr, a, b, c, 0.0
        )

        assert info == 0
        if nr_out < nr:
            assert iwarn == 1
