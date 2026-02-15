"""
Tests for AB09JW - State-space representation of projection of right weighted
transfer-function matrix.

AB09JW constructs a state-space representation (A,BS,CS,DS) of the projection
of G*W or G*conj(W) containing the poles of G, from the state-space
representations (A,B,C,D) and (AW-lambda*EW,BW,CW,DW) of the transfer-function
matrices G and W respectively.

Method:
For JOB='W', the stable projection of G*W is computed as:
    BS = B*DW + Y*BW, CS = C, DS = D*DW
where Y satisfies the generalized Sylvester equation:
    -A*Y*EW + Y*AW + B*CW = 0

For JOB='C', the stable projection of G*conj(W) is computed using:
- Continuous-time: BS = B*DW' + Y*CW', DS = D*DW'
  where A*Y*EW' + Y*AW' + B*BW' = 0
- Discrete-time: BS = B*DW' + A*Y*CW', DS = D*DW' + C*Y*CW'
  where Y*EW' - A*Y*AW' = B*BW'
"""

import numpy as np
import pytest
from ctrlsys import ab09jw


class TestAB09JWBasic:
    """Basic functionality tests for AB09JW."""

    def test_identity_weight_continuous_jobw(self):
        """
        Test with identity weight W = I for continuous-time system.

        When W = I (DW = I, AW empty, NW=0), the projection is just G*W = G.
        So BS = B*DW = B and DS = D*DW = D.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        n, m, p = 3, 2, 2
        nw, mw = 0, m

        a = np.array([
            [-1.0, 0.5, 0.0],
            [0.0, -2.0, 0.3],
            [0.0, 0.0, -3.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        ew = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, mw), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.eye(m, mw, order='F', dtype=float)

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'W', 'C', 'I', 'N', n, m, p, nw, mw,
            a, b, c, d, aw, ew, bw, cw, dw
        )

        assert info == 0
        np.testing.assert_allclose(b_out[:n, :mw], b[:n, :mw], rtol=1e-14)
        np.testing.assert_allclose(d_out[:p, :mw], d[:p, :mw], rtol=1e-14)

    def test_simple_continuous_jobw(self):
        """
        Test G*W projection with simple 2nd order systems, continuous-time.

        G: 2nd order stable system (Schur form)
        W: 2nd order antistable system

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        n, m, p = 2, 1, 1
        nw, mw = 2, 1

        a = np.array([
            [-1.0, 0.5],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        aw = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        ew = np.eye(2, order='F', dtype=float)

        bw = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0]
        ], order='F', dtype=float)

        b_orig = b.copy()
        d_orig = d.copy()

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'W', 'C', 'I', 'N', n, m, p, nw, mw,
            a, b_orig, c, d_orig, aw, ew, bw, cw, dw
        )

        assert info == 0
        assert b_out.shape[0] >= n
        assert b_out.shape[1] >= mw
        assert d_out.shape[0] >= p
        assert d_out.shape[1] >= mw
        np.testing.assert_allclose(d_out[:p, :mw], d[:p, :m] @ dw[:m, :mw], rtol=1e-14)

    def test_simple_discrete_jobw(self):
        """
        Test G*W projection for discrete-time system.

        G: 2nd order stable discrete system
        W: 2nd order antistable discrete system

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        n, m, p = 2, 1, 1
        nw, mw = 2, 1

        a = np.array([
            [0.5, 0.2],
            [0.0, 0.3]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        aw = np.array([
            [2.0, 0.0],
            [0.0, 1.5]
        ], order='F', dtype=float)

        ew = np.eye(2, order='F', dtype=float)

        bw = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0]
        ], order='F', dtype=float)

        b_orig = b.copy()
        d_orig = d.copy()

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'W', 'D', 'I', 'N', n, m, p, nw, mw,
            a, b_orig, c, d_orig, aw, ew, bw, cw, dw
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:p, :mw], d[:p, :m] @ dw[:m, :mw], rtol=1e-14)


class TestAB09JWConjugate:
    """Tests for G*conj(W) projection (JOB='C')."""

    def test_continuous_jobc_identity_ew(self):
        """
        Test G*conj(W) projection for continuous-time with EW=I.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        n, m, p = 2, 1, 1
        nw, mw = 2, 1

        a = np.array([
            [-1.0, 0.5],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        aw = np.array([
            [-3.0, 0.0],
            [0.0, -4.0]
        ], order='F', dtype=float)

        ew = np.eye(2, order='F', dtype=float)

        bw = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0]
        ], order='F', dtype=float)

        b_orig = b.copy()
        d_orig = d.copy()

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'C', 'C', 'I', 'N', n, m, p, nw, mw,
            a, b_orig, c, d_orig, aw, ew, bw, cw, dw
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:p, :mw], d[:p, :m] @ dw[:m, :mw].T, rtol=1e-14)

    def test_discrete_jobc_identity_ew(self):
        """
        Test G*conj(W) projection for discrete-time with EW=I.

        For discrete-time G*conj(W):
        DS = D*DW' + C*Y*CW'

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)

        n, m, p = 2, 1, 1
        nw, mw = 2, 1

        a = np.array([
            [0.5, 0.2],
            [0.0, 0.3]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        aw = np.array([
            [0.2, 0.0],
            [0.0, 0.3]
        ], order='F', dtype=float)

        ew = np.eye(2, order='F', dtype=float)

        bw = np.array([
            [0.0],
            [0.0]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0]
        ], order='F', dtype=float)

        b_orig = b.copy()
        d_orig = d.copy()

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'C', 'D', 'I', 'N', n, m, p, nw, mw,
            a, b_orig, c, d_orig, aw, ew, bw, cw, dw
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:p, :mw], d[:p, :m] @ dw[:m, :mw].T, rtol=1e-14)


class TestAB09JWGeneralEW:
    """Tests with general EW matrix (JOBEW='G')."""

    def test_general_ew_continuous_jobw(self):
        """
        Test with general EW matrix for continuous-time G*W.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)

        n, m, p = 2, 1, 1
        nw, mw = 2, 1

        a = np.array([
            [-1.0, 0.5],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        aw = np.array([
            [2.0, 0.1],
            [0.0, 3.0]
        ], order='F', dtype=float)

        ew = np.array([
            [1.0, 0.1],
            [0.0, 1.0]
        ], order='F', dtype=float)

        bw = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0]
        ], order='F', dtype=float)

        b_orig = b.copy()
        d_orig = d.copy()

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'W', 'C', 'G', 'N', n, m, p, nw, mw,
            a, b_orig, c, d_orig, aw, ew, bw, cw, dw
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:p, :mw], d[:p, :m] @ dw[:m, :mw], rtol=1e-14)

    def test_general_ew_discrete_jobc(self):
        """
        Test with general EW matrix for discrete-time G*conj(W).

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)

        n, m, p = 2, 1, 1
        nw, mw = 2, 1

        a = np.array([
            [0.5, 0.2],
            [0.0, 0.3]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        aw = np.array([
            [0.2, 0.1],
            [0.0, 0.3]
        ], order='F', dtype=float)

        ew = np.array([
            [1.0, 0.1],
            [0.0, 1.0]
        ], order='F', dtype=float)

        bw = np.array([
            [0.0],
            [0.0]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0]
        ], order='F', dtype=float)

        b_orig = b.copy()
        d_orig = d.copy()

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'C', 'D', 'G', 'N', n, m, p, nw, mw,
            a, b_orig, c, d_orig, aw, ew, bw, cw, dw
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:p, :mw], d[:p, :m] @ dw[:m, :mw].T, rtol=1e-14)


class TestAB09JWStabilityCheck:
    """Tests for stability checking (STBCHK='C')."""

    def test_stability_check_fails_for_stable_w_in_jobw(self):
        """
        Test that antistability check catches stable W when JOB='W'.

        For JOB='W', W must be antistable (eigenvalues in right half-plane
        for continuous-time).
        """
        n, m, p = 2, 1, 1
        nw, mw = 2, 1

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        aw = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        ew = np.eye(2, order='F', dtype=float)

        bw = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0]
        ], order='F', dtype=float)

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'W', 'C', 'I', 'C', n, m, p, nw, mw,
            a, b.copy(), c, d.copy(), aw, ew, bw, cw, dw
        )

        assert info == 4

    def test_stability_check_fails_for_unstable_w_in_jobc(self):
        """
        Test that stability check catches unstable W when JOB='C'.

        For JOB='C', W must be stable (eigenvalues in left half-plane for
        continuous-time).
        """
        n, m, p = 2, 1, 1
        nw, mw = 2, 1

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        aw = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        ew = np.eye(2, order='F', dtype=float)

        bw = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0]
        ], order='F', dtype=float)

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'C', 'C', 'I', 'C', n, m, p, nw, mw,
            a, b.copy(), c, d.copy(), aw, ew, bw, cw, dw
        )

        assert info == 4


class TestAB09JWEdgeCases:
    """Edge case tests for AB09JW."""

    def test_empty_systems(self):
        """Test with M=0 quick return."""
        n, m, p = 2, 0, 2
        nw, mw = 2, 1

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.zeros((2, 1), order='F', dtype=float)
        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        d = np.zeros((2, 1), order='F', dtype=float)
        aw = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)
        ew = np.eye(2, order='F', dtype=float)
        bw = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)
        cw = np.zeros((1, 2), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'W', 'C', 'I', 'N', n, m, p, nw, mw,
            a, b.copy(), c, d.copy(), aw, ew, bw, cw, dw
        )

        assert info == 0
        np.testing.assert_allclose(b_out[:n, :mw], np.zeros((n, mw)), rtol=1e-14)
        np.testing.assert_allclose(d_out[:p, :mw], np.zeros((p, mw)), rtol=1e-14)

    def test_single_state(self):
        """Test with N=1, NW=1."""
        n, m, p = 1, 1, 1
        nw, mw = 1, 1

        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        aw = np.array([[1.0]], order='F', dtype=float)
        ew = np.array([[1.0]], order='F', dtype=float)
        bw = np.array([[1.0]], order='F', dtype=float)
        cw = np.array([[1.0]], order='F', dtype=float)
        dw = np.array([[1.0]], order='F', dtype=float)

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'W', 'C', 'I', 'N', n, m, p, nw, mw,
            a, b.copy(), c, d.copy(), aw, ew, bw, cw, dw
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:p, :mw], d[:p, :m] @ dw[:m, :mw], rtol=1e-14)


class TestAB09JWMultipleInputsOutputs:
    """Tests with multiple inputs and outputs."""

    def test_mimo_system(self):
        """
        Test with MIMO system (M > 1, P > 1, MW > 1).

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)

        n, m, p = 3, 2, 2
        nw, mw = 2, 2

        a = np.array([
            [-1.0, 0.5, 0.0],
            [0.0, -2.0, 0.3],
            [0.0, 0.0, -3.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ], order='F', dtype=float)

        aw = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        ew = np.eye(2, order='F', dtype=float)

        bw = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        b_out, d_out, aw_out, ew_out, bw_out, cw_out, info = ab09jw(
            'W', 'C', 'I', 'N', n, m, p, nw, mw,
            a, b.copy(), c, d.copy(), aw, ew, bw, cw, dw
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:p, :mw], d[:p, :m] @ dw[:m, :mw], rtol=1e-14)


class TestAB09JWErrorHandling:
    """Error handling tests for AB09JW."""

    def test_invalid_job(self):
        """Test error for invalid JOB parameter."""
        n, m, p, nw, mw = 1, 1, 1, 1, 1
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        aw = np.array([[1.0]], order='F', dtype=float)
        ew = np.array([[1.0]], order='F', dtype=float)
        bw = np.array([[1.0]], order='F', dtype=float)
        cw = np.array([[1.0]], order='F', dtype=float)
        dw = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jw('X', 'C', 'I', 'N', n, m, p, nw, mw,
                   a, b, c, d, aw, ew, bw, cw, dw)

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        n, m, p, nw, mw = 1, 1, 1, 1, 1
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        aw = np.array([[1.0]], order='F', dtype=float)
        ew = np.array([[1.0]], order='F', dtype=float)
        bw = np.array([[1.0]], order='F', dtype=float)
        cw = np.array([[1.0]], order='F', dtype=float)
        dw = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jw('W', 'X', 'I', 'N', n, m, p, nw, mw,
                   a, b, c, d, aw, ew, bw, cw, dw)

    def test_invalid_jobew(self):
        """Test error for invalid JOBEW parameter."""
        n, m, p, nw, mw = 1, 1, 1, 1, 1
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        aw = np.array([[1.0]], order='F', dtype=float)
        ew = np.array([[1.0]], order='F', dtype=float)
        bw = np.array([[1.0]], order='F', dtype=float)
        cw = np.array([[1.0]], order='F', dtype=float)
        dw = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jw('W', 'C', 'X', 'N', n, m, p, nw, mw,
                   a, b, c, d, aw, ew, bw, cw, dw)

    def test_invalid_stbchk(self):
        """Test error for invalid STBCHK parameter."""
        n, m, p, nw, mw = 1, 1, 1, 1, 1
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        aw = np.array([[1.0]], order='F', dtype=float)
        ew = np.array([[1.0]], order='F', dtype=float)
        bw = np.array([[1.0]], order='F', dtype=float)
        cw = np.array([[1.0]], order='F', dtype=float)
        dw = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jw('W', 'C', 'I', 'X', n, m, p, nw, mw,
                   a, b, c, d, aw, ew, bw, cw, dw)

    def test_negative_n(self):
        """Test error for negative N."""
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        aw = np.array([[1.0]], order='F', dtype=float)
        ew = np.array([[1.0]], order='F', dtype=float)
        bw = np.array([[1.0]], order='F', dtype=float)
        cw = np.array([[1.0]], order='F', dtype=float)
        dw = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jw('W', 'C', 'I', 'N', -1, 1, 1, 1, 1,
                   a, b, c, d, aw, ew, bw, cw, dw)
