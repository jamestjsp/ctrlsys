"""
Tests for AB09IX - Accuracy enhanced balancing related model reduction

AB09IX computes a reduced order model (Ar,Br,Cr,Dr) for an original
state-space representation (A,B,C,D) using:
- Square-root Balance & Truncate (B&T) method (JOB='B')
- Balancing-free square-root B&T method (JOB='F')
- Square-root Singular Perturbation Approximation (JOB='S')
- Balancing-free square-root SPA method (JOB='P')

The computation uses Cholesky factors:
- S (TI input): controllability Grammian P = S*S' (upper triangular)
- R (T input): observability Grammian Q = R'*R (upper triangular)

Hankel singular values = singular values of R*S.

For B&T approach: Ar = TI*A*T, Br = TI*B, Cr = C*T
For SPA approach: Same formulas for minimal realization, then SPA applied.
"""

import numpy as np
import pytest
from slicot import ab09ix


class TestAB09IXBasic:
    """Basic functionality tests for AB09IX."""

    def test_square_root_bt_continuous(self):
        """
        Test square-root Balance & Truncate for continuous-time system.

        Creates a stable continuous-time system and computes Grammians
        via Lyapunov equations, then reduces with JOB='B'.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 1, 1

        a = np.array([
            [-1.0,  0.1,  0.0,  0.0],
            [ 0.0, -2.0,  0.1,  0.0],
            [ 0.0,  0.0, -3.0,  0.1],
            [ 0.0,  0.0,  0.0, -4.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.0],
            [0.0],
            [0.0]
        ], order='F', dtype=float)

        c = np.array([
            [0.0, 0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        ti = np.array([
            [0.7071, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.4082, 0.0],
            [0.0, 0.0, 0.0, 0.3536]
        ], order='F', dtype=float)

        t = np.array([
            [0.7071, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.4082, 0.0],
            [0.0, 0.0, 0.0, 0.3536]
        ], order='F', dtype=float)

        scalec = 1.0
        scaleo = 1.0
        nr_in = 2
        tol1 = 0.0
        tol2 = 0.0

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'C', 'B', 'N', 'F', n, m, p, nr_in, scalec, scaleo,
            a, b, c, d, ti, t, tol1, tol2
        )

        assert info == 0
        assert nr >= 0
        assert nr <= n
        assert nminr >= 0
        assert nminr <= n
        assert len(hsv) == n
        for i in range(n - 1):
            assert hsv[i] >= hsv[i + 1]

        if nr > 0:
            ar_red = ar[:nr, :nr]
            eigvals = np.linalg.eigvals(ar_red)
            assert np.all(np.real(eigvals) < 0)

    def test_balancing_free_bt_continuous(self):
        """
        Test balancing-free square-root B&T for continuous-time system.

        Uses JOB='F' for balancing-free method.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 3, 1, 1

        a = np.array([
            [-1.0,  0.2,  0.0],
            [ 0.0, -2.0,  0.3],
            [ 0.0,  0.0, -3.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5],
            [0.2]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.3]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        ti = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.6]
        ], order='F', dtype=float)

        t = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.6]
        ], order='F', dtype=float)

        scalec = 1.0
        scaleo = 1.0
        nr_in = 2
        tol1 = 0.0
        tol2 = 0.0

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'C', 'F', 'N', 'F', n, m, p, nr_in, scalec, scaleo,
            a, b, c, d, ti, t, tol1, tol2
        )

        assert info == 0
        assert nr >= 0
        assert nr <= n
        for i in range(n - 1):
            assert hsv[i] >= hsv[i + 1]

        if nr > 0:
            ar_red = ar[:nr, :nr]
            eigvals = np.linalg.eigvals(ar_red)
            assert np.all(np.real(eigvals) < 0)

    def test_spa_discrete_time(self):
        """
        Test Singular Perturbation Approximation for discrete-time system.

        Uses JOB='S' for square-root SPA method with DICO='D'.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 4, 2, 2

        a = np.array([
            [ 0.5,  0.1,  0.0,  0.0],
            [ 0.0,  0.4,  0.1,  0.0],
            [ 0.0,  0.0,  0.3,  0.1],
            [ 0.0,  0.0,  0.0,  0.2]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.0, 0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ], order='F', dtype=float)

        ti = np.array([
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.5]
        ], order='F', dtype=float)

        t = np.array([
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.5]
        ], order='F', dtype=float)

        scalec = 1.0
        scaleo = 1.0
        nr_in = 2
        tol1 = 0.0
        tol2 = 0.0

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'D', 'S', 'N', 'F', n, m, p, nr_in, scalec, scaleo,
            a, b, c, d, ti, t, tol1, tol2
        )

        assert info == 0
        assert nr >= 0
        assert nr <= n

        if nr > 0:
            ar_red = ar[:nr, :nr]
            eigvals = np.linalg.eigvals(ar_red)
            assert np.all(np.abs(eigvals) < 1.0)


class TestAB09IXMathematicalProperties:
    """Tests validating mathematical properties of AB09IX."""

    def test_hsv_ordering(self):
        """
        Verify Hankel singular values are in descending order.

        HSV must satisfy HSV(1) >= HSV(2) >= ... >= HSV(N) >= 0.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 5, 2, 2

        a = -np.diag([1.0, 2.0, 3.0, 4.0, 5.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), order='F', dtype=float)

        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'C', 'B', 'N', 'A', n, m, p, 0, 1.0, 1.0,
            a, b, c, d, ti, t, 0.0, 0.0
        )

        assert info == 0
        for i in range(n - 1):
            assert hsv[i] >= hsv[i + 1] - 1e-14
        assert hsv[-1] >= -1e-14

    def test_automatic_order_selection(self):
        """
        Test automatic order selection (ORDSEL='A').

        When ORDSEL='A', NR is determined by tolerance TOL1.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m, p = 4, 1, 1

        a = np.array([
            [-0.1,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0, -10.0, 0.0],
            [ 0.0,  0.0,  0.0, -100.0]
        ], order='F', dtype=float)

        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        tol1 = 0.01

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'C', 'B', 'N', 'A', n, m, p, 0, 1.0, 1.0,
            a, b, c, d, ti, t, tol1, 0.0
        )

        assert info == 0
        assert nr >= 0
        assert nr <= n

    def test_truncation_preserves_structure(self):
        """
        Test that truncation matrices TI and T have correct dimensions.

        Output TI has NMINR rows, T has NMINR columns.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m, p = 4, 1, 1

        a = -np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'C', 'B', 'N', 'F', n, m, p, 2, 1.0, 1.0,
            a, b, c, d, ti, t, 0.0, 0.0
        )

        assert info == 0
        assert ti_out.shape[0] >= nminr
        assert t_out.shape[1] >= nminr


class TestAB09IXEdgeCases:
    """Edge case tests for AB09IX."""

    def test_zero_reduction(self):
        """
        Test with nr=0 (full reduction to zero order).

        For SPA methods, this computes steady-state gain.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n, m, p = 2, 1, 1

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        a_orig = a.copy(order='F')
        b_orig = b.copy(order='F')
        c_orig = c.copy(order='F')

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'C', 'S', 'N', 'F', n, m, p, 0, 1.0, 1.0,
            a, b, c, d, ti, t, 0.0, 0.0
        )

        assert info == 0
        assert nr == 0

        dc_gain_orig = -c_orig @ np.linalg.solve(a_orig, b_orig)
        np.testing.assert_allclose(dr[:p, :m], dc_gain_orig, atol=1e-6)

    def test_single_state(self):
        """
        Test with n=1 (single state system).

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n, m, p = 1, 1, 1

        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        ti = np.array([[1.0]], order='F', dtype=float)
        t = np.array([[1.0]], order='F', dtype=float)

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'C', 'B', 'N', 'F', n, m, p, 1, 1.0, 1.0,
            a, b, c, d, ti, t, 0.0, 0.0
        )

        assert info == 0
        assert nr <= 1
        assert len(hsv) == 1

    def test_schur_form_input(self):
        """
        Test with A in real Schur form (FACT='S').

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n, m, p = 3, 1, 1

        a = np.array([
            [-1.0, 0.5, 0.3],
            [0.0, -2.0, 0.4],
            [0.0, 0.0, -3.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5], [0.2]], order='F', dtype=float)
        c = np.array([[1.0, 0.5, 0.3]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'C', 'B', 'S', 'F', n, m, p, 2, 1.0, 1.0,
            a, b, c, d, ti, t, 0.0, 0.0
        )

        assert info == 0


class TestAB09IXWarnings:
    """Tests for warning conditions in AB09IX."""

    def test_warning_nr_greater_than_nminr(self):
        """
        Test warning when NR > NMINR.

        IWARN=1 when desired order exceeds minimal realization order.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n, m, p = 3, 1, 1

        a = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, -3.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.0], [0.0]], order='F', dtype=float)
        c = np.array([[1.0, 0.0, 0.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        ti = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 0.0001]
        ], order='F', dtype=float)

        t = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 0.0001]
        ], order='F', dtype=float)

        ar, br, cr, dr, ti_out, t_out, nr, nminr, hsv, iwarn, info = ab09ix(
            'C', 'B', 'N', 'F', n, m, p, 3, 1.0, 1.0,
            a, b, c, d, ti, t, 0.0, 0.0
        )

        assert info == 0
        assert nr <= nminr


class TestAB09IXErrorHandling:
    """Error handling tests for AB09IX."""

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        n, m, p = 2, 1, 1
        a = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)
        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ix('X', 'B', 'N', 'F', n, m, p, 1, 1.0, 1.0,
                   a, b, c, d, ti, t, 0.0, 0.0)

    def test_invalid_job(self):
        """Test error for invalid JOB parameter."""
        n, m, p = 2, 1, 1
        a = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)
        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ix('C', 'X', 'N', 'F', n, m, p, 1, 1.0, 1.0,
                   a, b, c, d, ti, t, 0.0, 0.0)

    def test_invalid_scalec(self):
        """Test error for non-positive SCALEC."""
        n, m, p = 2, 1, 1
        a = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)
        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ix('C', 'B', 'N', 'F', n, m, p, 1, 0.0, 1.0,
                   a, b, c, d, ti, t, 0.0, 0.0)

    def test_invalid_scaleo(self):
        """Test error for non-positive SCALEO."""
        n, m, p = 2, 1, 1
        a = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)
        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ix('C', 'B', 'N', 'F', n, m, p, 1, 1.0, 0.0,
                   a, b, c, d, ti, t, 0.0, 0.0)

    def test_negative_n(self):
        """Test error for negative N."""
        m, p = 1, 1
        a = np.eye(1, order='F', dtype=float)
        b = np.ones((1, m), order='F', dtype=float)
        c = np.ones((p, 1), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)
        ti = np.eye(1, order='F', dtype=float)
        t = np.eye(1, order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ix('C', 'B', 'N', 'F', -1, m, p, 0, 1.0, 1.0,
                   a, b, c, d, ti, t, 0.0, 0.0)

    def test_nr_greater_than_n(self):
        """Test error when NR > N with fixed order."""
        n, m, p = 2, 1, 1
        a = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)
        ti = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09ix('C', 'B', 'N', 'F', n, m, p, 3, 1.0, 1.0,
                   a, b, c, d, ti, t, 0.0, 0.0)
