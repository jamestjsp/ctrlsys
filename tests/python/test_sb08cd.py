"""
Tests for SB08CD - Left coprime factorization with inner denominator.

Tests derived from SLICOT HTML documentation example.
"""

import numpy as np
import pytest

from ctrlsys import sb08cd


class TestSB08CDBasic:
    """Basic functionality tests from HTML doc example."""

    def test_continuous_time_example(self):
        """
        Test continuous-time left coprime factorization from HTML doc.

        System: 7th order, 2 inputs, 3 outputs
        Expected: NQ=7, NR=2

        Validates mathematical properties rather than exact matrix values
        since Schur ordering can produce equivalent but not identical results.
        """
        n, m, p = 7, 2, 3
        tol = 1e-10

        a = np.array([
            [-0.04165,   0.0,    4.92,    0.492,   0.0,      0.0,    0.0],
            [-5.21,    -12.5,    0.0,     0.0,     0.0,      0.0,    0.0],
            [ 0.0,       3.33,  -3.33,    0.0,     0.0,      0.0,    0.0],
            [ 0.545,     0.0,    0.0,     0.0,     0.0545,   0.0,    0.0],
            [ 0.0,       0.0,    0.0,    -0.492,   0.004165, 0.0,    4.92],
            [ 0.0,       0.0,    0.0,     0.0,     0.521,  -12.5,    0.0],
            [ 0.0,       0.0,    0.0,     0.0,     0.0,      3.33,  -3.33],
        ], order='F', dtype=float)

        b = np.zeros((n, max(m, p)), order='F', dtype=float)
        b[1, 0] = 12.5
        b[5, 1] = 12.5

        c = np.zeros((max(m, p), n), order='F', dtype=float)
        c[0, 0] = 1.0
        c[1, 3] = 1.0
        c[2, 4] = 1.0

        d = np.zeros((max(m, p), max(m, p)), order='F', dtype=float)

        nq, nr, br, dr, iwarn, info = sb08cd('C', a, b, c, d, tol=tol)

        assert info == 0
        assert nq == 7
        assert nr == 2

        aq = a[:nq, :nq]
        eigvals = np.linalg.eigvals(aq)
        for ev in eigvals:
            assert ev.real < 0, f"Eigenvalue {ev} not stable (Re >= 0)"

        ar = a[:nr, :nr]
        ar_eigs = np.linalg.eigvals(ar)
        for ev in ar_eigs:
            assert ev.real < 0, f"Denominator eigenvalue {ev} not stable"

        ar_expected_eigs = sorted([-0.1605 + 0.1532j, -0.1605 - 0.1532j], key=lambda x: x.imag)
        ar_eigs_sorted = sorted(ar_eigs, key=lambda x: x.imag)
        for actual, expected in zip(ar_eigs_sorted, ar_expected_eigs):
            np.testing.assert_allclose(actual.real, expected.real, rtol=0.1)
            np.testing.assert_allclose(abs(actual.imag), abs(expected.imag), rtol=0.1)

        assert a[:nq, :nq][2:, :2].max() < 1e-10, "A should be quasi-upper triangular"

        np.testing.assert_allclose(d[:p, :m], np.zeros((p, m)), rtol=1e-14, atol=1e-14)


class TestSB08CDEdgeCases:
    """Edge case tests."""

    def test_empty_system(self):
        """Test with N=0 (empty state)."""
        n, m, p = 0, 2, 3

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, max(m, p)), order='F', dtype=float)
        c = np.zeros((max(m, p), 1), order='F', dtype=float)
        d = np.zeros((max(m, p), max(m, p)), order='F', dtype=float)

        nq, nr, br, dr, iwarn, info = sb08cd('C', a, b, c, d)

        assert info == 0
        assert nq == 0
        assert nr == 0
        np.testing.assert_allclose(dr, np.eye(p), rtol=1e-14)

    def test_zero_outputs(self):
        """Test with P=0 (no outputs)."""
        n, m, p = 0, 2, 0

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, max(m, 1)), order='F', dtype=float)
        c = np.zeros((max(m, 1), 1), order='F', dtype=float)
        d = np.zeros((max(m, 1), max(m, 1)), order='F', dtype=float)

        nq, nr, br, dr, iwarn, info = sb08cd('C', a, b, c, d)

        assert info == 0
        assert nq == 0
        assert nr == 0


class TestSB08CDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_dr_lower_triangular(self):
        """
        Validate DR is lower triangular as specified in documentation.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        n, m, p = 4, 2, 2
        a = np.array([
            [-2.0,  0.0,  0.0,  0.0],
            [ 1.0, -3.0,  0.0,  0.0],
            [ 0.0,  1.0, -4.0,  0.0],
            [ 0.0,  0.0,  1.0, -5.0],
        ], order='F', dtype=float)

        b = np.zeros((n, max(m, p)), order='F', dtype=float)
        b[0, 0] = 1.0
        b[1, 1] = 1.0

        c = np.zeros((max(m, p), n), order='F', dtype=float)
        c[0, 2] = 1.0
        c[1, 3] = 1.0

        d = np.zeros((max(m, p), max(m, p)), order='F', dtype=float)

        nq, nr, br, dr, iwarn, info = sb08cd('C', a, b, c, d)

        assert info == 0

        dr_p = dr[:p, :p]
        for i in range(p):
            for j in range(i + 1, p):
                assert abs(dr_p[i, j]) < 1e-14, f"DR[{i},{j}] = {dr_p[i, j]} should be 0"

    def test_eigenvalues_stable_continuous(self):
        """
        Validate eigenvalues of numerator state matrix are stable (Re < 0).

        For continuous-time systems, all eigenvalues should have negative real part.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        n, m, p = 3, 1, 2

        a = np.diag([-1.0, -2.0, 0.5])
        a[0, 1] = 0.5
        a = np.asfortranarray(a)

        b = np.zeros((n, max(m, p)), order='F', dtype=float)
        b[0, 0] = 1.0

        c = np.zeros((max(m, p), n), order='F', dtype=float)
        c[0, 0] = 1.0
        c[1, 2] = 1.0

        d = np.zeros((max(m, p), max(m, p)), order='F', dtype=float)

        nq, nr, br, dr, iwarn, info = sb08cd('C', a, b, c, d)

        assert info == 0

        if nq > 0:
            aq = a[:nq, :nq]
            eigvals = np.linalg.eigvals(aq)
            for ev in eigvals:
                assert ev.real < 1e-10, f"Eigenvalue {ev} has non-negative real part"


class TestSB08CDErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test with invalid DICO parameter."""
        n, m, p = 2, 1, 1

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.zeros((n, max(m, p)), order='F', dtype=float)
        b[0, 0] = 1.0
        c = np.zeros((max(m, p), n), order='F', dtype=float)
        c[0, 0] = 1.0
        d = np.zeros((max(m, p), max(m, p)), order='F', dtype=float)

        with pytest.raises(ValueError):
            sb08cd('X', a, b, c, d)
