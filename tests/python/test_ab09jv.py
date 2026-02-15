"""
Tests for AB09JV - State-space representation of projection of left weighted
transfer-function matrix.

AB09JV constructs a state-space representation (A,BS,CS,DS) of the projection
of V*G or conj(V)*G containing the poles of G, from the state-space
representations (A,B,C,D) and (AV-lambda*EV,BV,CV,DV) of the transfer-function
matrices G and V respectively.

Method:
For JOB='V', the stable projection of V*G is computed as:
    BS = B, CS = CV*X + DV*C, DS = DV*D
where X satisfies the generalized Sylvester equation:
    AV*X - EV*X*A + BV*C = 0

For JOB='C', the stable projection of conj(V)*G is computed using:
- Continuous-time: CS = BV'*X + DV'*C, DS = DV'*D
  where AV'*X + EV'*X*A + CV'*C = 0
- Discrete-time: CS = BV'*X*A + DV'*C, DS = DV'*D + BV'*X*B
  where EV'*X - AV'*X*A = CV'*C
"""

import numpy as np
import pytest
from ctrlsys import ab09jv


class TestAB09JVBasic:
    """Basic functionality tests for AB09JV."""

    def test_identity_weight_continuous_jobv(self):
        """
        Test with identity weight V = I for continuous-time system.

        When V = I (DV = I, AV empty, NV=0), the projection is just V*G = G.
        So CS = DV*C = C and DS = DV*D = D.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        n, m, p = 3, 2, 2
        nv, pv = 0, p

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

        av = np.zeros((1, 1), order='F', dtype=float)
        ev = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((pv, 1), order='F', dtype=float)
        dv = np.eye(pv, p, order='F', dtype=float)

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'C', 'I', 'N', n, m, p, nv, pv,
            a, b, c, d, av, ev, bv, cv, dv
        )

        assert info == 0
        np.testing.assert_allclose(c_out[:pv, :n], c[:pv, :n], rtol=1e-14)
        np.testing.assert_allclose(d_out[:pv, :m], d[:pv, :m], rtol=1e-14)

    def test_simple_continuous_jobv(self):
        """
        Test V*G projection with simple 2nd order systems, continuous-time.

        G: 2nd order stable system (Schur form)
        V: 2nd order antistable system

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        n, m, p = 2, 1, 1
        nv, pv = 2, 1

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

        av = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        ev = np.eye(2, order='F', dtype=float)

        bv = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0]
        ], order='F', dtype=float)

        c_orig = c.copy()
        d_orig = d.copy()

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'C', 'I', 'N', n, m, p, nv, pv,
            a, b, c_orig, d_orig, av, ev, bv, cv, dv
        )

        assert info == 0
        assert c_out.shape[0] >= pv
        assert c_out.shape[1] >= n
        assert d_out.shape[0] >= pv
        assert d_out.shape[1] >= m
        np.testing.assert_allclose(d_out[:pv, :m], dv @ d[:p, :m], rtol=1e-14)

    def test_simple_discrete_jobv(self):
        """
        Test V*G projection for discrete-time system.

        G: 2nd order stable discrete system
        V: 2nd order antistable discrete system

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        n, m, p = 2, 1, 1
        nv, pv = 2, 1

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

        av = np.array([
            [2.0, 0.0],
            [0.0, 1.5]
        ], order='F', dtype=float)

        ev = np.eye(2, order='F', dtype=float)

        bv = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0]
        ], order='F', dtype=float)

        c_orig = c.copy()
        d_orig = d.copy()

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'D', 'I', 'N', n, m, p, nv, pv,
            a, b, c_orig, d_orig, av, ev, bv, cv, dv
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:pv, :m], dv @ d[:p, :m], rtol=1e-14)


class TestAB09JVConjugate:
    """Tests for conj(V)*G projection (JOB='C')."""

    def test_continuous_jobc_identity_ev(self):
        """
        Test conj(V)*G projection for continuous-time with EV=I.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        n, m, p = 2, 1, 1
        nv, pv = 2, 1

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

        av = np.array([
            [-3.0, 0.0],
            [0.0, -4.0]
        ], order='F', dtype=float)

        ev = np.eye(2, order='F', dtype=float)

        bv = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0]
        ], order='F', dtype=float)

        c_orig = c.copy()
        d_orig = d.copy()

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'C', 'C', 'I', 'N', n, m, p, nv, pv,
            a, b, c_orig, d_orig, av, ev, bv, cv, dv
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:pv, :m], dv.T @ d[:p, :m], rtol=1e-14)

    def test_discrete_jobc_identity_ev(self):
        """
        Test conj(V)*G projection for discrete-time with EV=I.

        For discrete-time conj(V)*G:
        DS = DV'*D + BV'*X*B

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)

        n, m, p = 2, 1, 1
        nv, pv = 2, 1

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

        av = np.array([
            [0.2, 0.0],
            [0.0, 0.3]
        ], order='F', dtype=float)

        ev = np.eye(2, order='F', dtype=float)

        bv = np.array([
            [0.0],
            [0.0]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0]
        ], order='F', dtype=float)

        c_orig = c.copy()
        d_orig = d.copy()

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'C', 'D', 'I', 'N', n, m, p, nv, pv,
            a, b, c_orig, d_orig, av, ev, bv, cv, dv
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:pv, :m], dv.T @ d[:p, :m], rtol=1e-14)


class TestAB09JVGeneralEV:
    """Tests with general EV matrix (JOBEV='G')."""

    def test_general_ev_continuous_jobv(self):
        """
        Test with general EV matrix for continuous-time V*G.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)

        n, m, p = 2, 1, 1
        nv, pv = 2, 1

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

        av = np.array([
            [2.0, 0.1],
            [0.0, 3.0]
        ], order='F', dtype=float)

        ev = np.array([
            [1.0, 0.1],
            [0.0, 1.0]
        ], order='F', dtype=float)

        bv = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0]
        ], order='F', dtype=float)

        c_orig = c.copy()
        d_orig = d.copy()

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'C', 'G', 'N', n, m, p, nv, pv,
            a, b, c_orig, d_orig, av, ev, bv, cv, dv
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:pv, :m], dv @ d[:p, :m], rtol=1e-14)

    def test_general_ev_discrete_jobc(self):
        """
        Test with general EV matrix for discrete-time conj(V)*G.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)

        n, m, p = 2, 1, 1
        nv, pv = 2, 1

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

        av = np.array([
            [0.2, 0.1],
            [0.0, 0.3]
        ], order='F', dtype=float)

        ev = np.array([
            [1.0, 0.1],
            [0.0, 1.0]
        ], order='F', dtype=float)

        bv = np.array([
            [0.0],
            [0.0]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0]
        ], order='F', dtype=float)

        c_orig = c.copy()
        d_orig = d.copy()

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'C', 'D', 'G', 'N', n, m, p, nv, pv,
            a, b, c_orig, d_orig, av, ev, bv, cv, dv
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:pv, :m], dv.T @ d[:p, :m], rtol=1e-14)


class TestAB09JVStabilityCheck:
    """Tests for stability checking (STBCHK='C')."""

    def test_stability_check_fails_for_unstable_v_in_jobc(self):
        """
        Test that stability check catches unstable V when JOB='C'.

        For JOB='C', V must be stable (eigenvalues in left half-plane for
        continuous-time).
        """
        n, m, p = 2, 1, 1
        nv, pv = 2, 1

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

        av = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        ev = np.eye(2, order='F', dtype=float)

        bv = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0]
        ], order='F', dtype=float)

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'C', 'C', 'I', 'C', n, m, p, nv, pv,
            a, b, c.copy(), d.copy(), av, ev, bv, cv, dv
        )

        assert info == 4

    def test_stability_check_fails_for_stable_v_in_jobv(self):
        """
        Test that antistability check catches stable V when JOB='V'.

        For JOB='V', V must be antistable (eigenvalues in right half-plane
        for continuous-time).
        """
        n, m, p = 2, 1, 1
        nv, pv = 2, 1

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

        av = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        ev = np.eye(2, order='F', dtype=float)

        bv = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0]
        ], order='F', dtype=float)

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'C', 'I', 'C', n, m, p, nv, pv,
            a, b, c.copy(), d.copy(), av, ev, bv, cv, dv
        )

        assert info == 4


class TestAB09JVEdgeCases:
    """Edge case tests for AB09JV."""

    def test_empty_systems(self):
        """Test with N=0 or NV=0."""
        n, m, p = 0, 1, 1
        nv, pv = 0, 1

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 1), order='F', dtype=float)
        c = np.zeros((1, 1), order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)
        av = np.zeros((1, 1), order='F', dtype=float)
        ev = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'C', 'I', 'N', n, m, p, nv, pv,
            a, b, c, d, av, ev, bv, cv, dv
        )

        assert info == 0

    def test_single_state(self):
        """Test with N=1, NV=1."""
        n, m, p = 1, 1, 1
        nv, pv = 1, 1

        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        av = np.array([[1.0]], order='F', dtype=float)
        ev = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'C', 'I', 'N', n, m, p, nv, pv,
            a, b, c.copy(), d.copy(), av, ev, bv, cv, dv
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:pv, :m], dv @ d[:p, :m], rtol=1e-14)

    def test_zero_p_returns_quick(self):
        """Test quick return when P=0."""
        n, m, p = 2, 1, 0
        nv, pv = 2, 1

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)
        c = np.zeros((1, 2), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        av = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)
        ev = np.eye(2, order='F', dtype=float)
        bv = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)
        cv = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'C', 'I', 'N', n, m, p, nv, pv,
            a, b, c, d, av, ev, bv, cv, dv
        )

        assert info == 0

    def test_zero_pv_returns_quick(self):
        """Test quick return when PV=0."""
        n, m, p = 2, 1, 1
        nv, pv = 2, 0

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
        d = np.zeros((1, 1), order='F', dtype=float)
        av = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)
        ev = np.eye(2, order='F', dtype=float)
        bv = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)
        cv = np.zeros((1, 2), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'C', 'I', 'N', n, m, p, nv, pv,
            a, b, c, d, av, ev, bv, cv, dv
        )

        assert info == 0


class TestAB09JVMultipleInputsOutputs:
    """Tests with multiple inputs and outputs."""

    def test_mimo_system(self):
        """
        Test with MIMO system (M > 1, P > 1, PV > 1).

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)

        n, m, p = 3, 2, 2
        nv, pv = 2, 2

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

        av = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        ev = np.eye(2, order='F', dtype=float)

        bv = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c_out, d_out, av_out, ev_out, bv_out, cv_out, info = ab09jv(
            'V', 'C', 'I', 'N', n, m, p, nv, pv,
            a, b, c.copy(), d.copy(), av, ev, bv, cv, dv
        )

        assert info == 0
        np.testing.assert_allclose(d_out[:pv, :m], dv @ d[:p, :m], rtol=1e-14)


class TestAB09JVErrorHandling:
    """Error handling tests for AB09JV."""

    def test_invalid_job(self):
        """Test error for invalid JOB parameter."""
        n, m, p, nv, pv = 1, 1, 1, 1, 1
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        av = np.array([[1.0]], order='F', dtype=float)
        ev = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jv('X', 'C', 'I', 'N', n, m, p, nv, pv,
                   a, b, c, d, av, ev, bv, cv, dv)

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        n, m, p, nv, pv = 1, 1, 1, 1, 1
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        av = np.array([[1.0]], order='F', dtype=float)
        ev = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jv('V', 'X', 'I', 'N', n, m, p, nv, pv,
                   a, b, c, d, av, ev, bv, cv, dv)

    def test_invalid_jobev(self):
        """Test error for invalid JOBEV parameter."""
        n, m, p, nv, pv = 1, 1, 1, 1, 1
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        av = np.array([[1.0]], order='F', dtype=float)
        ev = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jv('V', 'C', 'X', 'N', n, m, p, nv, pv,
                   a, b, c, d, av, ev, bv, cv, dv)

    def test_invalid_stbchk(self):
        """Test error for invalid STBCHK parameter."""
        n, m, p, nv, pv = 1, 1, 1, 1, 1
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        av = np.array([[1.0]], order='F', dtype=float)
        ev = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jv('V', 'C', 'I', 'X', n, m, p, nv, pv,
                   a, b, c, d, av, ev, bv, cv, dv)

    def test_negative_n(self):
        """Test error for negative N."""
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)
        av = np.array([[1.0]], order='F', dtype=float)
        ev = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jv('V', 'C', 'I', 'N', -1, 1, 1, 1, 1,
                   a, b, c, d, av, ev, bv, cv, dv)
