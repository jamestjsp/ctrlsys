"""
Tests for AG07BD: Descriptor inverse of a state-space or descriptor representation.

AG07BD computes the inverse (Ai-lambda*Ei, Bi, Ci, Di) of a given
descriptor system (A-lambda*E, B, C, D) using the formulas:

    Ei = ( E  0 )    Ai = ( A  B )    Bi = (  0 )
         ( 0  0 )         ( C  D )         ( -I )

    Ci = ( 0  I )    Di = 0

The inverse system has order N+M where N is the original order and M is the
number of inputs/outputs.
"""

import numpy as np
import pytest
from ctrlsys import ag07bd


class TestAG07BDBasic:
    """Basic functionality tests."""

    def test_simple_siso_identity_e(self):
        """
        Test simple SISO system with E = I.

        System: A = [1, 2; 3, 4], B = [1; 0], C = [0, 1], D = [1]
        Expected inverse dimensions: (N+M) x (N+M) = 3x3 for Ai, Ei

        Mathematical property: Ai = [A, B; C, D] block structure
        """
        n, m = 2, 1

        A = np.array([[1.0, 2.0],
                      [3.0, 4.0]], order='F', dtype=float)
        B = np.array([[1.0],
                      [0.0]], order='F', dtype=float)
        C = np.array([[0.0, 1.0]], order='F', dtype=float)
        D = np.array([[1.0]], order='F', dtype=float)

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        assert ai.shape == (n + m, n + m)
        assert ei.shape == (n + m, n + m)
        assert bi.shape == (n + m, m)
        assert ci.shape == (m, n + m)
        assert di.shape == (m, m)

        np.testing.assert_allclose(ai[:n, :n], A, rtol=1e-14)
        np.testing.assert_allclose(ai[:n, n:], B, rtol=1e-14)
        np.testing.assert_allclose(ai[n:, :n], C, rtol=1e-14)
        np.testing.assert_allclose(ai[n:, n:], D, rtol=1e-14)

        ei_expected = np.zeros((n + m, n + m), order='F', dtype=float)
        ei_expected[:n, :n] = np.eye(n)
        np.testing.assert_allclose(ei, ei_expected, rtol=1e-14)

        bi_expected = np.zeros((n + m, m), order='F', dtype=float)
        bi_expected[n:, :] = -np.eye(m)
        np.testing.assert_allclose(bi, bi_expected, rtol=1e-14)

        ci_expected = np.zeros((m, n + m), order='F', dtype=float)
        ci_expected[:, n:] = np.eye(m)
        np.testing.assert_allclose(ci, ci_expected, rtol=1e-14)

        np.testing.assert_allclose(di, np.zeros((m, m)), rtol=1e-14)

    def test_simple_mimo_identity_e(self):
        """
        Test simple MIMO system with E = I.

        2-input, 2-output system.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m = 3, 2

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(m, n).astype(float, order='F')
        D = np.random.randn(m, m).astype(float, order='F')

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        assert ai.shape == (n + m, n + m)
        assert ei.shape == (n + m, n + m)
        assert bi.shape == (n + m, m)
        assert ci.shape == (m, n + m)
        assert di.shape == (m, m)

        np.testing.assert_allclose(ai[:n, :n], A, rtol=1e-14)
        np.testing.assert_allclose(ai[:n, n:], B, rtol=1e-14)
        np.testing.assert_allclose(ai[n:, :n], C, rtol=1e-14)
        np.testing.assert_allclose(ai[n:, n:], D, rtol=1e-14)


class TestAG07BDDescriptor:
    """Test with general descriptor matrix E."""

    def test_general_e_matrix(self):
        """
        Test with general (non-identity) E matrix.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m = 2, 2

        A = np.array([[1.0, 2.0],
                      [3.0, 4.0]], order='F', dtype=float)
        E = np.array([[2.0, 0.0],
                      [1.0, 3.0]], order='F', dtype=float)
        B = np.array([[1.0, 0.0],
                      [0.0, 1.0]], order='F', dtype=float)
        C = np.array([[1.0, 1.0],
                      [0.0, 1.0]], order='F', dtype=float)
        D = np.array([[0.5, 0.0],
                      [0.0, 0.5]], order='F', dtype=float)

        result = ag07bd('G', n, m, A, B, C, D, E)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        assert ai.shape == (n + m, n + m)
        assert ei.shape == (n + m, n + m)

        np.testing.assert_allclose(ai[:n, :n], A, rtol=1e-14)
        np.testing.assert_allclose(ai[:n, n:], B, rtol=1e-14)
        np.testing.assert_allclose(ai[n:, :n], C, rtol=1e-14)
        np.testing.assert_allclose(ai[n:, n:], D, rtol=1e-14)

        np.testing.assert_allclose(ei[:n, :n], E, rtol=1e-14)
        np.testing.assert_allclose(ei[:n, n:], np.zeros((n, m)), rtol=1e-14)
        np.testing.assert_allclose(ei[n:, :n], np.zeros((m, n)), rtol=1e-14)
        np.testing.assert_allclose(ei[n:, n:], np.zeros((m, m)), rtol=1e-14)


class TestAG07BDEdgeCases:
    """Edge case tests."""

    def test_zero_order_system(self):
        """
        Test with N=0 (feedthrough only, D matrix system).

        System has no state dynamics, just direct feedthrough.
        """
        n, m = 0, 2

        A = np.zeros((1, 1), order='F', dtype=float)
        B = np.zeros((1, m), order='F', dtype=float)
        C = np.zeros((m, 1), order='F', dtype=float)
        D = np.array([[1.0, 0.5],
                      [0.0, 2.0]], order='F', dtype=float)

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        assert ai.shape == (m, m)
        np.testing.assert_allclose(ai, D, rtol=1e-14)

    def test_zero_io_system(self):
        """Test with M=0 (no inputs/outputs) - quick return."""
        n, m = 3, 0

        A = np.array([[1.0, 2.0, 0.0],
                      [0.0, 1.0, 3.0],
                      [0.0, 0.0, 1.0]], order='F', dtype=float)
        B = np.zeros((n, 1), order='F', dtype=float)
        C = np.zeros((1, n), order='F', dtype=float)
        D = np.zeros((1, 1), order='F', dtype=float)

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0

    def test_single_state_single_io(self):
        """
        Test smallest non-trivial system: N=1, M=1.

        System: A = [2], B = [1], C = [1], D = [0]
        """
        n, m = 1, 1

        A = np.array([[2.0]], order='F', dtype=float)
        B = np.array([[1.0]], order='F', dtype=float)
        C = np.array([[1.0]], order='F', dtype=float)
        D = np.array([[0.0]], order='F', dtype=float)

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        ai_expected = np.array([[2.0, 1.0],
                                [1.0, 0.0]], order='F', dtype=float)
        np.testing.assert_allclose(ai, ai_expected, rtol=1e-14)

        ei_expected = np.array([[1.0, 0.0],
                                [0.0, 0.0]], order='F', dtype=float)
        np.testing.assert_allclose(ei, ei_expected, rtol=1e-14)

        bi_expected = np.array([[0.0],
                                [-1.0]], order='F', dtype=float)
        np.testing.assert_allclose(bi, bi_expected, rtol=1e-14)

        ci_expected = np.array([[0.0, 1.0]], order='F', dtype=float)
        np.testing.assert_allclose(ci, ci_expected, rtol=1e-14)


class TestAG07BDMathematicalProperties:
    """Test mathematical properties of the inverse system."""

    def test_inverse_block_structure_ai(self):
        """
        Validate Ai has correct block structure: [A, B; C, D].

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m = 4, 3

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(m, n).astype(float, order='F')
        D = np.random.randn(m, m).astype(float, order='F')

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        np.testing.assert_allclose(ai[:n, :n], A, rtol=1e-14,
                                   err_msg="Ai[0:n, 0:n] should equal A")
        np.testing.assert_allclose(ai[:n, n:n+m], B, rtol=1e-14,
                                   err_msg="Ai[0:n, n:n+m] should equal B")
        np.testing.assert_allclose(ai[n:n+m, :n], C, rtol=1e-14,
                                   err_msg="Ai[n:n+m, 0:n] should equal C")
        np.testing.assert_allclose(ai[n:n+m, n:n+m], D, rtol=1e-14,
                                   err_msg="Ai[n:n+m, n:n+m] should equal D")

    def test_inverse_block_structure_ei_identity(self):
        """
        Validate Ei has correct block structure when JOBE='I': [I, 0; 0, 0].

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m = 3, 2

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(m, n).astype(float, order='F')
        D = np.random.randn(m, m).astype(float, order='F')

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        np.testing.assert_allclose(ei[:n, :n], np.eye(n), rtol=1e-14,
                                   err_msg="Ei[0:n, 0:n] should be I")
        np.testing.assert_allclose(ei[:n, n:], np.zeros((n, m)), rtol=1e-14,
                                   err_msg="Ei[0:n, n:] should be 0")
        np.testing.assert_allclose(ei[n:, :n], np.zeros((m, n)), rtol=1e-14,
                                   err_msg="Ei[n:, 0:n] should be 0")
        np.testing.assert_allclose(ei[n:, n:], np.zeros((m, m)), rtol=1e-14,
                                   err_msg="Ei[n:, n:] should be 0")

    def test_inverse_block_structure_ei_general(self):
        """
        Validate Ei has correct block structure when JOBE='G': [E, 0; 0, 0].

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m = 3, 2

        A = np.random.randn(n, n).astype(float, order='F')
        E = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(m, n).astype(float, order='F')
        D = np.random.randn(m, m).astype(float, order='F')

        result = ag07bd('G', n, m, A, B, C, D, E)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        np.testing.assert_allclose(ei[:n, :n], E, rtol=1e-14,
                                   err_msg="Ei[0:n, 0:n] should equal E")
        np.testing.assert_allclose(ei[:n, n:], np.zeros((n, m)), rtol=1e-14)
        np.testing.assert_allclose(ei[n:, :n], np.zeros((m, n)), rtol=1e-14)
        np.testing.assert_allclose(ei[n:, n:], np.zeros((m, m)), rtol=1e-14)

    def test_bi_structure(self):
        """
        Validate Bi has correct structure: [0; -I].

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m = 4, 3

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(m, n).astype(float, order='F')
        D = np.random.randn(m, m).astype(float, order='F')

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        np.testing.assert_allclose(bi[:n, :], np.zeros((n, m)), rtol=1e-14,
                                   err_msg="Bi[0:n, :] should be 0")
        np.testing.assert_allclose(bi[n:, :], -np.eye(m), rtol=1e-14,
                                   err_msg="Bi[n:, :] should be -I")

    def test_ci_structure(self):
        """
        Validate Ci has correct structure: [0, I].

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n, m = 4, 3

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(m, n).astype(float, order='F')
        D = np.random.randn(m, m).astype(float, order='F')

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0

        np.testing.assert_allclose(ci[:, :n], np.zeros((m, n)), rtol=1e-14,
                                   err_msg="Ci[:, 0:n] should be 0")
        np.testing.assert_allclose(ci[:, n:], np.eye(m), rtol=1e-14,
                                   err_msg="Ci[:, n:] should be I")

    def test_di_is_zero(self):
        """
        Validate Di is always zero.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n, m = 5, 4

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(m, n).astype(float, order='F')
        D = np.random.randn(m, m).astype(float, order='F')

        result = ag07bd('I', n, m, A, B, C, D)
        ai, ei, bi, ci, di, info = result

        assert info == 0
        np.testing.assert_allclose(di, np.zeros((m, m)), rtol=1e-14,
                                   err_msg="Di should be zero")


class TestAG07BDErrorHandling:
    """Test error handling for invalid parameters."""

    def test_invalid_jobe(self):
        """Test that invalid JOBE parameter returns error info=-1."""
        n, m = 2, 1
        A = np.eye(n, dtype=float, order='F')
        B = np.ones((n, m), dtype=float, order='F')
        C = np.ones((m, n), dtype=float, order='F')
        D = np.ones((m, m), dtype=float, order='F')

        result = ag07bd('X', n, m, A, B, C, D)
        info = result[-1]
        assert info == -1

    def test_negative_n(self):
        """Test that negative N returns error info=-2."""
        m = 1
        A = np.eye(2, dtype=float, order='F')
        B = np.ones((2, m), dtype=float, order='F')
        C = np.ones((m, 2), dtype=float, order='F')
        D = np.ones((m, m), dtype=float, order='F')

        result = ag07bd('I', -1, m, A, B, C, D)
        info = result[-1]
        assert info == -2

    def test_negative_m(self):
        """Test that negative M returns error info=-3."""
        n = 2
        A = np.eye(n, dtype=float, order='F')
        B = np.ones((n, 1), dtype=float, order='F')
        C = np.ones((1, n), dtype=float, order='F')
        D = np.ones((1, 1), dtype=float, order='F')

        result = ag07bd('I', n, -1, A, B, C, D)
        info = result[-1]
        assert info == -3
