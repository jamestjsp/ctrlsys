"""
Tests for AB09ID - Frequency-weighted model reduction based on balancing techniques.

AB09ID computes a reduced order model (Ar,Br,Cr,Dr) for an original
state-space representation (A,B,C,D) using frequency weighted square-root
or balancing-free square-root Balance & Truncate (B&T) or Singular
Perturbation Approximation (SPA) model reduction methods.

The algorithm minimizes the norm of the frequency-weighted error:
    ||V*(G-Gr)*W||

where G and Gr are transfer-function matrices of original/reduced systems,
V and W are frequency-weighting transfer-function matrices.

Mode Parameters:
- DICO: 'C' (continuous-time), 'D' (discrete-time)
- JOBC: 'S' (standard combination), 'E' (stability enhanced)
- JOBO: 'S' (standard combination), 'E' (stability enhanced)
- JOB: 'B' (sqrt B&T), 'F' (bal-free sqrt B&T), 'S' (sqrt SPA), 'P' (bal-free SPA)
- WEIGHT: 'N' (no weight), 'L' (left V), 'R' (right W), 'B' (both)
- EQUIL: 'S' (scale), 'N' (no scaling)
- ORDSEL: 'F' (fixed order), 'A' (automatic order)

Warning codes (IWARN):
- 0: no warning
- 1: NR > NSMIN, NR set to NSMIN
- 2: repeated singular values, NR decreased
- 3: NR < NU (unstable order), NR set to NU
- 10+K: K stability violations in SB08CD/SB08DD

Error codes (INFO):
- 0: success
- 1: ordered Schur form of A failed
- 2: separation of stable/unstable blocks failed
- 3: Schur form of minimal realization of V failed
- 4: ordering of Schur form of V or coprime factorization failed
- 5: AV has observable eigenvalue on stability boundary
- 6: Schur form of minimal realization of W failed
- 7: ordering of Schur form of W or coprime factorization failed
- 8: AW has controllable eigenvalue on stability boundary
- 9: eigenvalue computation failed
- 10: Hankel singular value computation failed
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestAB09IDDocExample:
    """Tests based on HTML documentation example."""

    def test_continuous_left_weight_fixed_order(self):
        """
        Test from SLICOT HTML documentation AB09ID.html.

        3rd order continuous system with 6th order left weighting.
        Reduces to 2nd order with ORDSEL='F', NR=2.
        JOBC='S', JOBO='S' uses standard combination method.
        JOB='F' uses balancing-free square-root B&T method.
        """
        from ctrlsys import ab09id

        n, m, p = 3, 1, 1
        nv, pv = 6, 1
        nw, mw = 0, 0

        a = np.array([
            [-26.4000,  6.4023,  4.3868],
            [ 32.0000,  0.0,     0.0   ],
            [  0.0,     8.0000,  0.0   ]
        ], order='F', dtype=float)

        b = np.array([
            [16.0],
            [ 0.0],
            [ 0.0]
        ], order='F', dtype=float)

        c = np.array([
            [9.2994, 1.1624, 0.1090]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        av = np.array([
            [-1.0,  0.0,  4.0, -9.2994, -1.1624, -0.1090],
            [ 0.0,  2.0,  0.0, -9.2994, -1.1624, -0.1090],
            [ 0.0,  0.0, -3.0, -9.2994, -1.1624, -0.1090],
            [16.0, 16.0, 16.0, -26.4000, 6.4023,  4.3868],
            [ 0.0,  0.0,  0.0,  32.0000, 0.0,     0.0   ],
            [ 0.0,  0.0,  0.0,   0.0,    8.0000,  0.0   ]
        ], order='F', dtype=float)

        bv = np.array([
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        dv = np.array([
            [0.0]
        ], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        alphac = 0.0
        alphao = 0.0
        tol1 = 0.1
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, ns, hsv, nr_out, iwarn, info) = ab09id(
            'C', 'S', 'S', 'F', 'L', 'S', 'F',
            n, m, p, nv, pv, nw, mw, nr, alpha, alphac, alphao,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert nr_out == 2, f"Expected nr=2, got {nr_out}"

        hsv_expected = np.array([3.8253, 0.2005])
        assert_allclose(hsv[:ns], hsv_expected, rtol=1e-3, atol=1e-4,
                       err_msg="Hankel singular values mismatch")

        ar_expected = np.array([
            [ 9.1900, 0.0000],
            [ 0.0000, -34.5297]
        ], order='F', dtype=float)

        br_expected = np.array([
            [11.9593],
            [16.9329]
        ], order='F', dtype=float)

        cr_expected = np.array([
            [2.8955, 6.9152]
        ], order='F', dtype=float)

        dr_expected = np.array([
            [0.0]
        ], order='F', dtype=float)

        ar_actual = a_out[:nr_out, :nr_out].copy()
        br_actual = b_out[:nr_out, :m].copy()
        cr_actual = c_out[:p, :nr_out].copy()
        dr_actual = d_out[:p, :m].copy()

        assert_allclose(np.abs(ar_actual), np.abs(ar_expected), rtol=1e-3, atol=1e-3,
                       err_msg="Reduced A matrix mismatch")
        assert_allclose(np.abs(br_actual), np.abs(br_expected), rtol=1e-3, atol=1e-3,
                       err_msg="Reduced B matrix mismatch")
        assert_allclose(np.abs(cr_actual), np.abs(cr_expected), rtol=1e-3, atol=1e-3,
                       err_msg="Reduced C matrix mismatch")
        assert_allclose(dr_actual, dr_expected, rtol=1e-3, atol=1e-3,
                       err_msg="Reduced D matrix mismatch")


class TestAB09IDNoWeighting:
    """Tests with no frequency weighting (WEIGHT='N')."""

    def test_no_weight_continuous_stable(self):
        """
        Test with no frequency weighting (V=I, W=I).

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(42)
        n, m, p = 4, 1, 1
        nv, pv = 0, 0
        nw, mw = 0, 0

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        alphac = 0.0
        alphao = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, ns, hsv, nr_out, iwarn, info) = ab09id(
            'C', 'S', 'S', 'B', 'N', 'N', 'F',
            n, m, p, nv, pv, nw, mw, nr, alpha, alphac, alphao,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n, f"Expected ns={n}, got {ns}"
        assert nr_out == nr, f"Expected nr={nr}, got {nr_out}"

        for i in range(ns - 1):
            assert hsv[i] >= hsv[i + 1], "HSV should be decreasingly ordered"


class TestAB09IDLeftWeighting:
    """Tests with left frequency weighting only (WEIGHT='L')."""

    def test_left_weight_with_stable_v(self):
        """
        Test left weighting with stable V system.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(123)
        n, m, p = 3, 1, 1
        nv, pv = 2, 1
        nw, mw = 0, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)
        bv = np.array([[1.0], [1.0]], order='F', dtype=float)
        cv = np.array([[1.0, 1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        alphac = 0.0
        alphao = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, ns, hsv, nr_out, iwarn, info) = ab09id(
            'C', 'S', 'S', 'B', 'L', 'N', 'F',
            n, m, p, nv, pv, nw, mw, nr, alpha, alphac, alphao,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n, f"Expected ns={n}, got {ns}"


class TestAB09IDRightWeighting:
    """Tests with right frequency weighting only (WEIGHT='R')."""

    def test_right_weight_with_stable_w(self):
        """
        Test right weighting with stable W system.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(456)
        n, m, p = 3, 1, 1
        nv, pv = 0, 0
        nw, mw = 2, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)

        aw = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)
        bw = np.array([[1.0], [1.0]], order='F', dtype=float)
        cw = np.array([[1.0, 1.0]], order='F', dtype=float)
        dw = np.array([[1.0]], order='F', dtype=float)

        nr = 2
        alpha = 0.0
        alphac = 0.0
        alphao = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, ns, hsv, nr_out, iwarn, info) = ab09id(
            'C', 'S', 'S', 'B', 'R', 'N', 'F',
            n, m, p, nv, pv, nw, mw, nr, alpha, alphac, alphao,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n


class TestAB09IDJobModes:
    """Tests for different JOB modes."""

    def test_job_b_sqrt_bt(self):
        """
        Test JOB='B' square-root B&T method.

        Random seed: 101 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(101)
        n, m, p = 3, 1, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        (_, _, _, _, _, _, _, _, _, _, ns, hsv, nr_out, _, info) = ab09id(
            'C', 'S', 'S', 'B', 'N', 'N', 'F',
            n, m, p, 0, 0, 0, 0, 2, 0.0, 0.0, 0.0,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
        )

        assert info == 0

    def test_job_s_sqrt_spa(self):
        """
        Test JOB='S' square-root SPA method.

        Random seed: 102 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(102)
        n, m, p = 3, 1, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        (_, _, _, _, _, _, _, _, _, _, ns, hsv, nr_out, _, info) = ab09id(
            'C', 'S', 'S', 'S', 'N', 'N', 'F',
            n, m, p, 0, 0, 0, 0, 2, 0.0, 0.0, 0.0,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
        )

        assert info == 0

    def test_job_p_balfree_spa(self):
        """
        Test JOB='P' balancing-free SPA method.

        Random seed: 103 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(103)
        n, m, p = 3, 1, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        (_, _, _, _, _, _, _, _, _, _, ns, hsv, nr_out, _, info) = ab09id(
            'C', 'S', 'S', 'P', 'N', 'N', 'F',
            n, m, p, 0, 0, 0, 0, 2, 0.0, 0.0, 0.0,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
        )

        assert info == 0


class TestAB09IDDiscreteTime:
    """Tests for discrete-time systems."""

    def test_discrete_no_weight(self):
        """
        Test discrete-time system with no frequency weighting.

        Random seed: 201 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(201)
        n, m, p = 3, 1, 1

        a = np.diag([0.3, 0.5, 0.7]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        nr = 2
        alpha = 1.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, ns, hsv, nr_out, iwarn, info) = ab09id(
            'D', 'S', 'S', 'B', 'N', 'N', 'F',
            n, m, p, 0, 0, 0, 0, nr, alpha, 0.0, 0.0,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n


class TestAB09IDMathematicalProperties:
    """Tests for mathematical properties."""

    def test_hsv_decreasing_order(self):
        """
        Validate Hankel singular values are in decreasing order.

        Random seed: 301 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(301)
        n, m, p = 5, 1, 1

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        (_, _, _, _, _, _, _, _, _, _, ns, hsv, nr_out, _, info) = ab09id(
            'C', 'S', 'S', 'B', 'N', 'N', 'A',
            n, m, p, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 1e-10, 0.0
        )

        assert info == 0
        for i in range(ns - 1):
            assert hsv[i] >= hsv[i + 1], f"HSV not decreasing at index {i}"

    def test_reduced_system_stability_continuous(self):
        """
        Validate reduced system preserves stability (continuous-time).

        Eigenvalues of reduced system should have negative real parts.
        Random seed: 302 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(302)
        n, m, p = 4, 1, 1

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        nr = 2
        alpha = 0.0

        (a_out, b_out, c_out, d_out, _, _, _, _, _, _, ns, hsv, nr_out, _, info) = ab09id(
            'C', 'S', 'S', 'B', 'N', 'N', 'F',
            n, m, p, 0, 0, 0, 0, nr, alpha, 0.0, 0.0,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
        )

        assert info == 0

        if nr_out > 0:
            ar = a_out[:nr_out, :nr_out].copy()
            eigs = np.linalg.eigvals(ar)
            for eig in eigs:
                assert eig.real < 0, f"Eigenvalue {eig} has non-negative real part"


class TestAB09IDEdgeCases:
    """Edge case tests."""

    def test_quick_return_min_nmp_zero(self):
        """Test quick return when MIN(N,M,P)=0."""
        from ctrlsys import ab09id

        n, m, p = 0, 1, 1

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 1), order='F', dtype=float)
        c = np.zeros((1, 1), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        (_, _, _, _, _, _, _, _, _, _, ns, hsv, nr_out, _, info) = ab09id(
            'C', 'S', 'S', 'B', 'N', 'N', 'F',
            n, m, p, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
        )

        assert info == 0
        assert nr_out == 0
        assert ns == 0

    def test_automatic_order_selection(self):
        """
        Test ORDSEL='A' for automatic order selection.

        Random seed: 401 (for reproducibility)
        """
        from ctrlsys import ab09id

        np.random.seed(401)
        n, m, p = 4, 1, 1

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        tol1 = 0.05

        (_, _, _, _, _, _, _, _, _, _, ns, hsv, nr_out, _, info) = ab09id(
            'C', 'S', 'S', 'B', 'N', 'N', 'A',
            n, m, p, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, 0.0
        )

        assert info == 0
        assert nr_out <= n


class TestAB09IDErrors:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        from ctrlsys import ab09id

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((2, 1), order='F', dtype=float)
        c = np.ones((1, 2), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09id(
                'X', 'S', 'S', 'B', 'N', 'N', 'F',
                2, 1, 1, 0, 0, 0, 0, 1, 0.0, 0.0, 0.0,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
            )

    def test_invalid_job(self):
        """Test error for invalid JOB parameter."""
        from ctrlsys import ab09id

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((2, 1), order='F', dtype=float)
        c = np.ones((1, 2), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09id(
                'C', 'S', 'S', 'X', 'N', 'N', 'F',
                2, 1, 1, 0, 0, 0, 0, 1, 0.0, 0.0, 0.0,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
            )

    def test_invalid_alphac(self):
        """Test error for |ALPHAC| > 1."""
        from ctrlsys import ab09id

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((2, 1), order='F', dtype=float)
        c = np.ones((1, 2), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09id(
                'C', 'S', 'S', 'B', 'N', 'N', 'F',
                2, 1, 1, 0, 0, 0, 0, 1, 0.0, 1.5, 0.0,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
            )

    def test_negative_n(self):
        """Test error for negative N."""
        from ctrlsys import ab09id

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 1), order='F', dtype=float)
        c = np.zeros((1, 1), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09id(
                'C', 'S', 'S', 'B', 'N', 'N', 'F',
                -1, 1, 1, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
            )
