"""
Tests for sb08fd: Right coprime factorization with prescribed stability degree.

Constructs feedback matrix F and orthogonal transformation Z such that:
    Q = (Z'*(A+B*F)*Z, Z'*B, (C+D*F)*Z, D)
    R = (Z'*(A+B*F)*Z, Z'*B, F*Z, I)

provide a stable right coprime factorization G = Q * R^(-1).
"""

import numpy as np
import pytest
from ctrlsys import sb08fd


class TestSB08FDBasic:
    """Tests using HTML documentation example data."""

    def test_continuous_time_example(self):
        """
        Validate sb08fd using HTML doc example.

        7th order continuous-time system with ALPHA=-1.0.
        Validates dimensions, stability, and mathematical properties rather
        than exact values since Schur form is not unique (ordering/sign ambiguity).
        """
        n, m, p = 7, 2, 3
        alpha = np.array([-1.0, -1.0], dtype=float, order='F')
        tol = 1e-10

        a = np.array([
            [-0.04165,  0.0000,  4.9200,  0.4920,  0.0000,   0.0000,  0.0000],
            [-5.2100, -12.500,  0.0000,  0.0000,  0.0000,   0.0000,  0.0000],
            [ 0.0000,  3.3300, -3.3300,  0.0000,  0.0000,   0.0000,  0.0000],
            [ 0.5450,  0.0000,  0.0000,  0.0000,  0.0545,   0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000, -0.4920,  0.004165, 0.0000,  4.9200],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.5210, -12.500,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,   3.3300, -3.3300]
        ], dtype=float, order='F')

        b = np.array([
            [ 0.0000,  0.0000],
            [12.500,   0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [ 0.0000, 12.500],
            [ 0.0000,  0.0000]
        ], dtype=float, order='F')

        c = np.array([
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]
        ], dtype=float, order='F')

        d = np.array([
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000]
        ], dtype=float, order='F')

        aq, bq, cq, cr, dr, nq, nr, iwarn, info = sb08fd(
            'C', n, m, p, alpha, a, b, c, d, tol=tol
        )

        assert info == 0
        assert nq == 7
        assert nr == 2

        dr_expected = np.eye(m, dtype=float, order='F')
        np.testing.assert_allclose(dr, dr_expected, rtol=1e-14)

        eigs = np.linalg.eigvals(aq[:nq, :nq])
        assert np.all(eigs.real <= alpha[0] + 0.1), f"Eigenvalues not stable: {eigs}"

        if nq > 2:
            lower_part = np.tril(aq[:nq, :nq], k=-2)
            np.testing.assert_allclose(lower_part, 0.0, atol=1e-10)

        assert bq.shape == (n, m)
        assert cq.shape == (p, n)
        assert cr.shape == (m, n)


class TestSB08FDEdgeCases:
    """Edge case tests."""

    def test_zero_state_dimension(self):
        """System with n=0 should return immediately."""
        n, m, p = 0, 2, 2
        alpha = np.array([-1.0, -1.0], dtype=float, order='F')

        a = np.zeros((1, 1), dtype=float, order='F')
        b = np.zeros((1, m), dtype=float, order='F')
        c = np.zeros((p, 1), dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        aq, bq, cq, cr, dr, nq, nr, iwarn, info = sb08fd(
            'C', n, m, p, alpha, a, b, c, d
        )

        assert info == 0
        assert nq == 0
        assert nr == 0
        np.testing.assert_allclose(dr, np.eye(m), rtol=1e-14)

    def test_already_stable_system(self):
        """
        System with all eigenvalues already inside stability region.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 3, 2, 2
        alpha = np.array([-0.5, -0.5], dtype=float, order='F')

        a = np.array([
            [-2.0, 0.3, 0.1],
            [0.0, -1.5, 0.2],
            [0.0, 0.0, -1.0]
        ], dtype=float, order='F')

        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        aq, bq, cq, cr, dr, nq, nr, iwarn, info = sb08fd(
            'C', n, m, p, alpha, a, b, c, d
        )

        assert info == 0
        assert nr == 0
        assert nq == n


class TestSB08FDDiscreteTime:
    """Discrete-time system tests."""

    def test_discrete_time_basic(self):
        """
        Discrete-time system with eigenvalues outside unit circle.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 3, 1, 1
        alpha = np.array([0.5, 0.5], dtype=float, order='F')

        a = np.array([
            [1.5, 0.1, 0.0],
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.3]
        ], dtype=float, order='F')

        b = np.array([[1.0], [0.5], [0.2]], dtype=float, order='F')
        c = np.array([[1.0, 0.0, 0.0]], dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        aq, bq, cq, cr, dr, nq, nr, iwarn, info = sb08fd(
            'D', n, m, p, alpha, a, b, c, d
        )

        assert info == 0
        assert nq >= 0
        assert nr >= 0

        if nq > 0:
            eigs = np.linalg.eigvals(aq[:nq, :nq])
            assert np.all(np.abs(eigs) <= alpha[0] + 0.1)


class TestSB08FDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_schur_form_output(self):
        """
        Validate that output AQ is in real Schur form.

        A matrix in real Schur form is quasi-upper triangular:
        - Upper triangular except possibly 2x2 blocks on diagonal
        - Elements below first subdiagonal are zero

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 5, 2, 2
        alpha = np.array([-1.0, -1.0], dtype=float, order='F')

        a = np.random.randn(n, n).astype(float, order='F')
        a[np.diag_indices(n)] -= 2.0

        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        aq, bq, cq, cr, dr, nq, nr, iwarn, info = sb08fd(
            'C', n, m, p, alpha, a, b, c, d
        )

        assert info == 0

        if nq > 2:
            lower_part = np.tril(aq[:nq, :nq], k=-2)
            np.testing.assert_allclose(lower_part, 0.0, atol=1e-10)

    def test_dr_is_identity(self):
        """Validate DR output is identity matrix."""
        np.random.seed(789)
        n, m, p = 4, 2, 2
        alpha = np.array([-1.0, -1.0], dtype=float, order='F')

        a = np.random.randn(n, n).astype(float, order='F')
        a[np.diag_indices(n)] -= 3.0
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        aq, bq, cq, cr, dr, nq, nr, iwarn, info = sb08fd(
            'C', n, m, p, alpha, a, b, c, d
        )

        assert info == 0
        np.testing.assert_allclose(dr, np.eye(m), rtol=1e-14)


class TestSB08FDErrors:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Invalid DICO parameter should return info=-1."""
        n, m, p = 2, 1, 1
        alpha = np.array([-1.0, -1.0], dtype=float, order='F')
        a = np.zeros((n, n), dtype=float, order='F')
        b = np.zeros((n, m), dtype=float, order='F')
        c = np.zeros((p, n), dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        aq, bq, cq, cr, dr, nq, nr, iwarn, info = sb08fd(
            'X', n, m, p, alpha, a, b, c, d
        )

        assert info == -1

    def test_invalid_alpha_continuous(self):
        """
        Invalid ALPHA for continuous-time: ALPHA >= 0 should give info=-5.
        """
        n, m, p = 2, 1, 1
        alpha = np.array([0.5, 0.5], dtype=float, order='F')
        a = np.zeros((n, n), dtype=float, order='F')
        b = np.zeros((n, m), dtype=float, order='F')
        c = np.zeros((p, n), dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        aq, bq, cq, cr, dr, nq, nr, iwarn, info = sb08fd(
            'C', n, m, p, alpha, a, b, c, d
        )

        assert info == -5

    def test_invalid_alpha_discrete(self):
        """
        Invalid ALPHA for discrete-time: ALPHA < 0 or >= 1 should give info=-5.
        """
        n, m, p = 2, 1, 1
        alpha = np.array([-0.5, -0.5], dtype=float, order='F')
        a = np.zeros((n, n), dtype=float, order='F')
        b = np.zeros((n, m), dtype=float, order='F')
        c = np.zeros((p, n), dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        aq, bq, cq, cr, dr, nq, nr, iwarn, info = sb08fd(
            'D', n, m, p, alpha, a, b, c, d
        )

        assert info == -5
