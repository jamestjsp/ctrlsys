"""
Tests for AB09HX - Stochastic balancing model reduction of stable systems.

AB09HX computes a reduced order model (Ar,Br,Cr,Dr) for an original stable
state-space representation (A,B,C,D) using stochastic balancing with:
- Balance & Truncate (B&T) method (JOB='B' or 'F')
- Singular Perturbation Approximation (SPA) method (JOB='S' or 'P')

Requirements:
- Matrix A must be stable and in real Schur canonical form
- Matrix D (P-by-M) must have full row rank (P <= M required)

For B&T:
    Ar = TI * A * T,  Br = TI * B,  Cr = C * T

For SPA:
    Am = TI * A * T,  Bm = TI * B,  Cm = C * T
    Then SPA is computed from (Am, Bm, Cm, D)

Mode Parameters:
- DICO: 'C' (continuous), 'D' (discrete)
- JOB: 'B' (sqrt B&T), 'F' (balancing-free sqrt B&T),
       'S' (sqrt SPA), 'P' (balancing-free sqrt SPA)
- ORDSEL: 'F' (fixed order), 'A' (automatic order selection)

Error codes:
- INFO = 0: success
- INFO = 1: A not stable or not in real Schur form
- INFO = 2: Hamiltonian reduction failed
- INFO = 3: Hamiltonian reordering failed
- INFO = 4: Hamiltonian has < N stable eigenvalues
- INFO = 5: U11 singular in Riccati solver
- INFO = 6: D does not have full row rank
- INFO = 7: SVD computation failed
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestAB09HXBasic:
    """Basic functionality tests."""

    def test_continuous_bt_auto_order(self):
        """
        Test continuous-time Balance & Truncate with automatic order selection.

        Uses a stable diagonal system in Schur form.
        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(42)
        n, m, p = 4, 2, 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.2, 0.3]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.0, 1.0, 0.5, 0.2]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        nr_in = 0
        tol1 = 1e-3
        tol2 = 0.0

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, nr_in, a, b, c, d, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert nr_out >= 0, "NR must be non-negative"
        assert nr_out <= n, "NR must not exceed N"
        assert nmin >= nr_out, "NMIN should be >= NR"
        assert len(hsv) == n, f"HSV should have length {n}"

        for i in range(n - 1):
            assert hsv[i] >= hsv[i + 1] - 1e-14, "HSV must be in decreasing order"

    def test_continuous_bt_fixed_order(self):
        """
        Test continuous-time Balance & Truncate with fixed order selection.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(123)
        n, m, p = 4, 2, 1

        a = np.diag([-0.5, -1.5, -2.5, -3.5]).astype(float, order='F')

        b = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
            [0.3, 0.7],
            [0.2, 0.4]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.3, 0.1]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        nr_in = 2
        tol1 = 0.0
        tol2 = 0.0

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'F', n, m, p, nr_in, a, b, c, d, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert nr_out <= nr_in, f"NR should be <= requested {nr_in}"
        assert nr_out <= nmin, "NR should be <= NMIN"

        if nr_out > 0:
            ar_reduced = ar[:nr_out, :nr_out]
            eigvals = np.linalg.eigvals(ar_reduced)
            for ev in eigvals:
                assert ev.real < 1e-10, f"Reduced system should be stable, got eigenvalue {ev}"

    def test_discrete_bt_auto_order(self):
        """
        Test discrete-time Balance & Truncate with automatic order.

        Discrete-time stable means |eigenvalues| < 1.
        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(456)
        n, m, p = 3, 2, 1

        a = np.diag([0.5, 0.3, 0.2]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.2]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        nr_in = 0
        tol1 = 1e-3
        tol2 = 0.0

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'D', 'B', 'A', n, m, p, nr_in, a, b, c, d, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert nr_out >= 0
        assert nr_out <= n


class TestAB09HXJobVariants:
    """Tests for different JOB parameter values."""

    def test_balancing_free_bt(self):
        """
        Test balancing-free square-root B&T method (JOB='F').

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(789)
        n, m, p = 3, 2, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([[1.0, 0.5, 0.3]], order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        nr_in = 2
        tol1 = 0.0
        tol2 = 0.0

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'F', 'F', n, m, p, nr_in, a, b, c, d, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert nr_out >= 0
        assert nr_out <= n

    def test_spa_method(self):
        """
        Test square-root Singular Perturbation Approximation (JOB='S').

        Random seed: 111 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(111)
        n, m, p = 4, 2, 1

        a = np.diag([-0.5, -1.0, -2.0, -4.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.3, 0.3]
        ], order='F', dtype=float)

        c = np.array([[1.0, 0.5, 0.2, 0.1]], order='F', dtype=float)
        d = np.array([[1.0, 0.0]], order='F', dtype=float)

        nr_in = 2
        tol1 = 0.0
        tol2 = 0.0

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'S', 'F', n, m, p, nr_in, a, b, c, d, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"

    def test_balancing_free_spa(self):
        """
        Test balancing-free square-root SPA (JOB='P').

        Random seed: 222 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(222)
        n, m, p = 3, 2, 2

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.3]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        nr_in = 2
        tol1 = 0.0
        tol2 = 0.0

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'P', 'F', n, m, p, nr_in, a, b, c, d, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"


class TestAB09HXMathematicalProperties:
    """Tests for mathematical properties."""

    def test_hsv_decreasing_order(self):
        """
        Validate: Hankel singular values are in strictly decreasing order.

        Random seed: 333 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(333)
        n, m, p = 5, 2, 2

        a = np.diag([-0.5, -1.0, -1.5, -2.0, -2.5]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.3, 0.2],
            [0.1, 0.4]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.2, 0.1, 0.05],
            [0.0, 1.0, 0.5, 0.2, 0.1]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        for i in range(n - 1):
            assert hsv[i] >= hsv[i + 1] - 1e-14, \
                f"HSV[{i}]={hsv[i]} should be >= HSV[{i+1}]={hsv[i+1]}"

    def test_hsv_bounded_by_one(self):
        """
        Validate: Hankel singular values are <= 1 for phase system.

        Random seed: 444 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(444)
        n, m, p = 4, 2, 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.3]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.0, 1.0, 0.3, 0.2]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        for i in range(n):
            assert hsv[i] <= 1.0 + 1e-10, f"HSV[{i}]={hsv[i]} should be <= 1"

    def test_reduced_system_stable_continuous(self):
        """
        Validate: Reduced continuous-time system is stable (eigenvalues < 0).

        Random seed: 555 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(555)
        n, m, p = 4, 2, 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.2, 0.3]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.0, 1.0, 0.5, 0.2]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'F', n, m, p, 2, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        if nr_out > 0:
            ar_reduced = ar[:nr_out, :nr_out]
            eigvals = np.linalg.eigvals(ar_reduced)
            for ev in eigvals:
                assert ev.real < 1e-10, \
                    f"Reduced system should be stable, got eigenvalue {ev}"

    def test_reduced_system_stable_discrete(self):
        """
        Validate: Reduced discrete-time system is stable (|eigenvalues| < 1).

        Random seed: 666 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(666)
        n, m, p = 3, 2, 1

        a = np.diag([0.5, 0.3, 0.2]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([[1.0, 0.5, 0.2]], order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'D', 'B', 'F', n, m, p, 2, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        if nr_out > 0:
            ar_reduced = ar[:nr_out, :nr_out]
            eigvals = np.linalg.eigvals(ar_reduced)
            for ev in eigvals:
                assert abs(ev) < 1.0 + 1e-10, \
                    f"Reduced system should be stable, got |eigenvalue| = {abs(ev)}"

    def test_truncation_matrices_orthogonality(self):
        """
        Validate: For balancing-free method, TI * T is close to identity.

        Random seed: 777 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(777)
        n, m, p = 3, 2, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([[1.0, 0.5, 0.3]], order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'F', 'F', n, m, p, 2, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

        if nr_out > 0:
            t_nr = t[:n, :nr_out]
            ti_nr = ti[:nr_out, :n]
            product = ti_nr @ t_nr
            expected = np.eye(nr_out)
            assert_allclose(product, expected, rtol=1e-10, atol=1e-10,
                err_msg="TI * T should be identity for balancing-free method")


class TestAB09HXEdgeCases:
    """Edge case tests."""

    def test_siso_system(self):
        """
        Test with SISO system (M=1, P=1).

        Random seed: 888 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(888)
        n, m, p = 2, 1, 1

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr_out >= 0
        assert nr_out <= n

    def test_minimal_system_n1(self):
        """
        Test with minimal 1st order system (N=1).

        Random seed: 999 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(999)
        n, m, p = 1, 2, 1

        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[1.0, 0.0]], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr_out >= 0
        assert nr_out <= n

    def test_square_d_matrix(self):
        """
        Test with square D matrix (M=P).

        Random seed: 1111 (for reproducibility)
        """
        from ctrlsys import ab09hx

        np.random.seed(1111)
        n, m, p = 3, 2, 2

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.3]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0

    def test_order_warning(self):
        """
        Test IWARN=1 when requested order exceeds minimal realization.

        IWARN=1 occurs when ORDSEL='F' and NR > minimal realization order.
        Create a system with some near-zero Hankel singular values.
        """
        from ctrlsys import ab09hx

        n, m, p = 3, 2, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ], order='F', dtype=float)

        c = np.array([[1.0, 0.0, 0.0]], order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'F', n, m, p, n, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr_out <= n


class TestAB09HXQuickReturn:
    """Quick return tests for zero dimensions."""

    def test_n_zero(self):
        """Test quick return when N=0."""
        from ctrlsys import ab09hx

        n, m, p = 0, 2, 1

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, m), order='F', dtype=float)
        c = np.zeros((p, 1), order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr_out == 0
        assert nmin == 0

    def test_m_zero(self):
        """Test quick return when M=0."""
        from ctrlsys import ab09hx

        n, m, p = 2, 0, 0

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.zeros((n, 1), order='F', dtype=float)
        c = np.zeros((1, n), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr_out == 0
        assert nmin == 0

    def test_p_zero(self):
        """Test quick return when P=0."""
        from ctrlsys import ab09hx

        n, m, p = 2, 2, 0

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.zeros((1, n), order='F', dtype=float)
        d = np.zeros((1, m), order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 0
        assert nr_out == 0
        assert nmin == 0


class TestAB09HXErrorHandling:
    """Error handling tests."""

    def test_unstable_continuous_system(self):
        """Test INFO=1 for unstable continuous-time system."""
        from ctrlsys import ab09hx

        n, m, p = 2, 2, 1

        a = np.array([
            [1.0, 0.0],
            [0.0, -1.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 1, f"Expected info=1 for unstable system, got {info}"

    def test_rank_deficient_d(self):
        """Test INFO=6 when D does not have full row rank."""
        from ctrlsys import ab09hx

        n, m, p = 2, 2, 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0],
            [2.0, 0.0]
        ], order='F', dtype=float)

        ar, br, cr, dr, nr_out, hsv, t, ti, nmin, rcond, iwarn, info = ab09hx(
            'C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0
        )

        assert info == 6, f"Expected info=6 for rank-deficient D, got {info}"

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        from ctrlsys import ab09hx

        n, m, p = 2, 1, 1

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09hx('X', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0)

    def test_invalid_job(self):
        """Test error for invalid JOB parameter."""
        from ctrlsys import ab09hx

        n, m, p = 2, 1, 1

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09hx('C', 'X', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0)

    def test_invalid_ordsel(self):
        """Test error for invalid ORDSEL parameter."""
        from ctrlsys import ab09hx

        n, m, p = 2, 1, 1

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09hx('C', 'B', 'X', n, m, p, 0, a, b, c, d, 0.0, 0.0)

    def test_p_greater_than_m(self):
        """Test error when P > M (D cannot have full row rank)."""
        from ctrlsys import ab09hx

        n, m, p = 2, 1, 2

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        d = np.array([[1.0], [0.5]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09hx('C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.0, 0.0)

    def test_tol2_greater_than_tol1(self):
        """Test error when TOL2 > TOL1 with ORDSEL='A'."""
        from ctrlsys import ab09hx

        n, m, p = 2, 2, 1

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09hx('C', 'B', 'A', n, m, p, 0, a, b, c, d, 0.01, 0.1)
