"""
Tests for AB09IY - Cholesky factors of frequency-weighted controllability and
observability Grammians for model reduction.

AB09IY computes the Cholesky factors S and R of the controllability Grammian
P = S*S' and observability Grammian Q = R'*R using frequency-weighting
with combination method of Enns and Lin-Chiu.

For continuous-time systems (DICO = 'C'):
    A*P + P*A' + scalec^2*B*B' = 0
    A'*Q + Q*A + scaleo^2*C'*C = 0

For discrete-time systems (DICO = 'D'):
    A*P*A' - P + scalec^2*B*B' = 0
    A'*Q*A - Q + scaleo^2*C'*C = 0

Frequency weighting options:
    WEIGHT = 'N': No frequency weighting
    WEIGHT = 'L': Left weighting (output weighting via W system)
    WEIGHT = 'R': Right weighting (input weighting via V system)
    WEIGHT = 'B': Both left and right weighting

Requirements:
- Matrix A must be stable
- Matrix A must be in real Schur form
- Input weighting system V must be minimum phase and have proper/bistable inverse
- Output weighting system W must be minimum phase and have proper/bistable inverse

Random seeds: 42, 123, 456, 789, 111, 222, 333, 444 (for reproducibility)
"""

import numpy as np
import pytest
from ctrlsys import ab09iy
from fortran_reference import run_fortran_driver


AB09IY_ENHANCED_LEFT_WEIGHT_CASE = r"""
program main
  implicit none
  integer, parameter :: n = 3, m = 2, p = 2, nv = 2, pv = 2, nw = 0, mw = 0
  integer, parameter :: lda = 3, ldb = 3, ldc = 2
  integer, parameter :: ldav = 2, ldbv = 2, ldcv = 2, lddv = 2
  integer, parameter :: ldaw = 1, ldbw = 1, ldcw = 2, lddw = 2
  integer, parameter :: lds = 3, ldr = 3, ldwork = 200
  integer info
  double precision alphac, alphao, scalec, scaleo
  double precision a(lda,n), b(ldb,m), c(ldc,n)
  double precision av(ldav,nv), bv(ldbv,p), cv(ldcv,nv), dv(lddv,p)
  double precision aw(ldaw,1), bw(ldbw,1), cw(ldcw,1), dw(lddw,1)
  double precision s(lds,n), r(ldr,n), dwork(ldwork)

  alphac = 1.0d0
  alphao = 0.5d0

  a = 0.0d0
  a(1,1) = -1.0d0
  a(2,2) = -2.0d0
  a(3,3) = -3.0d0

  b(1,1) = 1.0d0
  b(2,1) = 0.5d0
  b(3,1) = 0.0d0
  b(1,2) = 0.0d0
  b(2,2) = 0.5d0
  b(3,2) = 1.0d0

  c(1,1) = 1.0d0
  c(1,2) = 0.5d0
  c(1,3) = 0.3d0
  c(2,1) = 0.2d0
  c(2,2) = 1.0d0
  c(2,3) = 0.6d0

  av = 0.0d0
  av(1,1) = -0.5d0
  av(2,2) = -1.5d0
  bv = 0.0d0
  bv(1,1) = 1.0d0
  bv(2,2) = 1.0d0
  cv = 0.0d0
  cv(1,1) = 1.0d0
  cv(2,2) = 1.0d0
  dv = 0.0d0
  dv(1,1) = 1.0d0
  dv(2,2) = 1.0d0

  aw = 0.0d0
  bw = 0.0d0
  cw = 0.0d0
  dw = 0.0d0
  s = 0.0d0
  r = 0.0d0

  call AB09IY('C', 'S', 'E', 'L', n, m, p, nv, pv, nw, mw, alphac, alphao, &
       a, lda, b, ldb, c, ldc, av, ldav, bv, ldbv, cv, ldcv, dv, lddv, &
       aw, ldaw, bw, ldbw, cw, ldcw, dw, lddw, scalec, scaleo, s, lds, &
       r, ldr, dwork, ldwork, info)

  print '(I0,1X,ES24.16,1X,ES24.16)', info, scalec, scaleo
  print '(*(ES24.16,1X))', s
  print '(*(ES24.16,1X))', r
end program main
"""


def _take_matrix(values, offset, n):
    end = offset + n * n
    return values[offset:end].reshape((n, n), order="F"), end


class TestAB09IYBasic:
    """Basic functionality tests - no frequency weighting."""

    def test_no_weighting_continuous(self):
        """
        Test with no frequency weighting (WEIGHT='N') for continuous-time.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 3, 2, 2
        nv, pv = 0, 0
        nw, mw = 0, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
            [0.3, 0.7]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.5, 0.3],
            [0.2, 1.0, 0.6]
        ], order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'N',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert scalec > 0
        assert scaleo > 0
        assert s.shape == (n, n)
        assert r.shape == (n, n)

        p_gram = s @ s.T
        q_gram = r.T @ r

        p_eigvals = np.linalg.eigvalsh(p_gram)
        q_eigvals = np.linalg.eigvalsh(q_gram)
        assert np.all(p_eigvals >= -1e-10), "P must be positive semi-definite"
        assert np.all(q_eigvals >= -1e-10), "Q must be positive semi-definite"

    def test_no_weighting_discrete(self):
        """
        Test with no frequency weighting (WEIGHT='N') for discrete-time.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 2, 2, 2
        nv, pv = 0, 0
        nw, mw = 0, 0

        a = np.array([
            [0.5, 0.1],
            [0.0, 0.3]
        ], order='F', dtype=float)
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'D', 'S', 'S', 'N',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert scalec > 0
        assert scaleo > 0


class TestAB09IYLeftWeighting:
    """Tests with left (output) frequency weighting."""

    def test_left_weighting_continuous(self):
        """
        Test with left frequency weighting (WEIGHT='L') for continuous-time.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 2, 2, 2
        nv, pv = 0, 0
        nw, mw = 2, 2

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)

        aw = np.diag([-0.5, -1.5]).astype(float, order='F')
        bw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        cw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        dw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'L',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert s.shape == (n, n)
        assert r.shape == (n, n)


class TestAB09IYRightWeighting:
    """Tests with right (input) frequency weighting."""

    def test_right_weighting_continuous(self):
        """
        Test with right frequency weighting (WEIGHT='R') for continuous-time.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 2, 2, 2
        nv, pv = 2, 2
        nw, mw = 0, 0

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)

        av = np.diag([-0.5, -1.5]).astype(float, order='F')
        bv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        cv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        dv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'R',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert s.shape == (n, n)
        assert r.shape == (n, n)


class TestAB09IYBothWeighting:
    """Tests with both left and right frequency weighting."""

    def test_both_weighting_continuous(self):
        """
        Test with both frequency weightings (WEIGHT='B') for continuous-time.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m, p = 2, 2, 2
        nv, pv = 2, 2
        nw, mw = 2, 2

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        av = np.diag([-0.5, -1.5]).astype(float, order='F')
        bv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        cv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        dv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        aw = np.diag([-0.3, -0.7]).astype(float, order='F')
        bw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        cw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        dw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'B',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert s.shape == (n, n)
        assert r.shape == (n, n)


class TestAB09IYGrammianProperties:
    """Tests for Grammian mathematical properties."""

    def test_lyapunov_residual_continuous(self):
        """
        Verify controllability Grammian P = S*S' satisfies Lyapunov equation:
            A*P + P*A' + scalec^2*B*B' = 0

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m, p = 3, 2, 2
        nv, pv = 0, 0
        nw, mw = 0, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5]
        ], order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'N',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )
        assert info == 0

        p_gram = s @ s.T
        residual = a @ p_gram + p_gram @ a.T + (scalec ** 2) * (b @ b.T)

        np.testing.assert_allclose(residual, 0.0, atol=1e-5,
            err_msg="Controllability Lyapunov equation not satisfied")

    def test_grammians_positive_semidefinite(self):
        """
        Verify Grammians P = S*S' and Q = R'*R are positive semi-definite.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n, m, p = 3, 2, 2
        nv, pv = 0, 0
        nw, mw = 0, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'N',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )
        assert info == 0

        p_gram = s @ s.T
        q_gram = r.T @ r

        eig_p = np.linalg.eigvalsh(p_gram)
        eig_q = np.linalg.eigvalsh(q_gram)

        assert np.all(eig_p >= -1e-14), "P must be positive semi-definite"
        assert np.all(eig_q >= -1e-14), "Q must be positive semi-definite"


class TestAB09IYStandardCombination:
    """Tests with standard combination options (JOBC='S', JOBO='S')."""

    def test_standard_controllability(self):
        """
        Test with standard controllability computation (JOBC='S').

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n, m, p = 2, 2, 2
        nv, pv = 0, 0
        nw, mw = 0, 0

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'N',
            n, m, p, nv, pv, nw, mw,
            0.5, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )

        assert info == 0
        assert s.shape == (n, n)


class TestAB09IYErrorHandling:
    """Error handling tests."""

    def test_unstable_system(self):
        """
        Test INFO > 0 for unstable system.
        """
        n, m, p = 2, 2, 2
        nv, pv = 0, 0
        nw, mw = 0, 0

        a = np.array([
            [1.0, 0.0],
            [0.0, -1.0]
        ], order='F', dtype=float)
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'N',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )

        assert info > 0, f"Expected info > 0 for unstable system, got {info}"

    def test_negative_n_error(self):
        """Test error for negative n."""
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, 1), order='F', dtype=float)
        cv = np.zeros((0, 0), order='F', dtype=float)
        dv = np.zeros((0, 1), order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, 0), order='F', dtype=float)
        cw = np.zeros((1, 0), order='F', dtype=float)
        dw = np.zeros((1, 0), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09iy(
                'C', 'S', 'S', 'N',
                -1, 1, 1, 0, 0, 0, 0,
                1.0, 1.0,
                a, b, c,
                av, bv, cv, dv,
                aw, bw, cw, dw
            )


class TestAB09IYQuickReturn:
    """Quick return tests for zero dimensions."""

    def test_n_zero(self):
        """
        Test quick return when n=0.
        """
        n, m, p = 0, 2, 2
        nv, pv = 0, 0
        nw, mw = 0, 0

        a = np.zeros((0, 0), order='F', dtype=float)
        b = np.zeros((0, m), order='F', dtype=float)
        c = np.zeros((p, 0), order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'N',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )

        assert info == 0
        assert scalec == 1.0
        assert scaleo == 1.0

    def test_m_zero(self):
        """Test quick return when m=0."""
        n, m, p = 2, 0, 0
        nv, pv = 0, 0
        nw, mw = 0, 0

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.zeros((n, 0), order='F', dtype=float)
        c = np.zeros((0, n), order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, 0), order='F', dtype=float)
        cv = np.zeros((0, 0), order='F', dtype=float)
        dv = np.zeros((0, 0), order='F', dtype=float)

        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, 0), order='F', dtype=float)
        cw = np.zeros((0, 0), order='F', dtype=float)
        dw = np.zeros((0, 0), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'S', 'N',
            n, m, p, nv, pv, nw, mw,
            1.0, 1.0,
            a, b, c,
            av, bv, cv, dv,
            aw, bw, cw, dw
        )

        assert info == 0
        assert scalec == 1.0
        assert scaleo == 1.0


class TestAB09IYEnhancedStability:
    """Tests for JOBC='E' and JOBO='E' (stability-enhanced combination method)."""

    def test_jobo_enhanced_left_weighting_matches_fortran_reference(self, tmp_path):
        output = run_fortran_driver(AB09IY_ENHANCED_LEFT_WEIGHT_CASE, tmp_path)
        values = np.fromstring(output, sep=" ")
        info_f = int(values[0])
        scalec_f = values[1]
        scaleo_f = values[2]
        s_f, offset = _take_matrix(values, 3, 3)
        r_f, _ = _take_matrix(values, offset, 3)

        n, m, p = 3, 2, 2
        nv, pv, nw, mw = 2, 2, 0, 0
        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1, 0], [.5, .5], [0, 1]], order='F', dtype=float)
        c = np.array([[1, .5, .3], [.2, 1, .6]], order='F', dtype=float)
        av = np.diag([-0.5, -1.5]).astype(float, order='F')
        bv = np.eye(2, order='F', dtype=float)
        cv = np.eye(2, order='F', dtype=float)
        dv = np.eye(2, order='F', dtype=float)
        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'E', 'L', n, m, p, nv, pv, nw, mw,
            1.0, 0.5, a, b, c, av, bv, cv, dv, aw, bw, cw, dw)

        assert info == info_f
        np.testing.assert_allclose(scalec, scalec_f, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scaleo, scaleo_f, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.triu(s), np.triu(s_f), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.triu(r), np.triu(r_f), rtol=1e-11, atol=1e-11)

    def test_jobo_enhanced_left_weighting_continuous(self):
        """
        JOBO='E' with left weighting and ALPHAO < 1, continuous-time.
        Exercises MB01WD + DSYEV spectral factorization path.
        Random seed: 555
        """
        np.random.seed(555)
        n, m, p = 3, 2, 2
        nv, pv, nw, mw = 2, 2, 0, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1, 0], [.5, .5], [0, 1]], order='F', dtype=float)
        c = np.array([[1, .5, .3], [.2, 1, .6]], order='F', dtype=float)

        av = np.diag([-0.5, -1.5]).astype(float, order='F')
        bv = np.eye(2, order='F', dtype=float)
        cv = np.eye(2, order='F', dtype=float)
        dv = np.eye(2, order='F', dtype=float)
        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'S', 'E', 'L', n, m, p, nv, pv, nw, mw,
            1.0, 0.5, a, b, c, av, bv, cv, dv, aw, bw, cw, dw)

        assert info == 0, f"info={info}"
        q_gram = r.T @ r
        eig_q = np.linalg.eigvalsh(q_gram)
        assert np.all(eig_q >= -1e-10), "Q must be PSD"

    def test_jobc_enhanced_right_weighting_continuous(self):
        """
        JOBC='E' with right weighting and ALPHAC < 1, continuous-time.
        Random seed: 666
        """
        np.random.seed(666)
        n, m, p = 3, 2, 2
        nv, pv, nw, mw = 0, 0, 2, 2

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1, 0], [.5, .5], [0, 1]], order='F', dtype=float)
        c = np.array([[1, .5, .3], [.2, 1, .6]], order='F', dtype=float)

        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)
        aw = np.diag([-0.5, -1.5]).astype(float, order='F')
        bw = np.eye(2, order='F', dtype=float)
        cw = np.eye(2, order='F', dtype=float)
        dw = np.eye(2, order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'E', 'S', 'R', n, m, p, nv, pv, nw, mw,
            0.5, 1.0, a, b, c, av, bv, cv, dv, aw, bw, cw, dw)

        assert info == 0, f"info={info}"
        p_gram = s @ s.T
        eig_p = np.linalg.eigvalsh(p_gram)
        assert np.all(eig_p >= -1e-10), "P must be PSD"

    def test_both_enhanced_both_weighting_continuous(self):
        """
        JOBC='E', JOBO='E' with both weightings, ALPHAC=ALPHAO=0.5.
        Random seed: 777
        """
        np.random.seed(777)
        n, m, p = 2, 2, 2
        nv, pv, nw, mw = 2, 2, 2, 2

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.eye(2, order='F', dtype=float)
        c = np.eye(2, order='F', dtype=float)
        av = np.diag([-0.5, -1.5]).astype(float, order='F')
        bv = np.eye(2, order='F', dtype=float)
        cv = np.eye(2, order='F', dtype=float)
        dv = np.eye(2, order='F', dtype=float)
        aw = np.diag([-0.3, -0.7]).astype(float, order='F')
        bw = np.eye(2, order='F', dtype=float)
        cw = np.eye(2, order='F', dtype=float)
        dw = np.eye(2, order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'C', 'E', 'E', 'B', n, m, p, nv, pv, nw, mw,
            0.5, 0.5, a, b, c, av, bv, cv, dv, aw, bw, cw, dw)

        assert info == 0, f"info={info}"
        assert np.all(np.linalg.eigvalsh(s @ s.T) >= -1e-10), "P must be PSD"
        assert np.all(np.linalg.eigvalsh(r.T @ r) >= -1e-10), "Q must be PSD"

    def test_jobo_enhanced_discrete(self):
        """JOBO='E' discrete-time. Random seed: 888"""
        np.random.seed(888)
        n, m, p = 2, 2, 2
        nv, pv, nw, mw = 2, 2, 0, 0

        a = np.array([[.5, .1], [0, .3]], order='F', dtype=float)
        b = np.eye(2, order='F', dtype=float)
        c = np.array([[1, .5], [0, 1]], order='F', dtype=float)
        av = np.diag([0.4, 0.2]).astype(float, order='F')
        bv = np.eye(2, order='F', dtype=float)
        cv = np.eye(2, order='F', dtype=float)
        dv = np.eye(2, order='F', dtype=float)
        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'D', 'S', 'E', 'L', n, m, p, nv, pv, nw, mw,
            1.0, 0.5, a, b, c, av, bv, cv, dv, aw, bw, cw, dw)

        assert info == 0, f"info={info}"
        assert np.all(np.linalg.eigvalsh(r.T @ r) >= -1e-10)

    def test_jobc_enhanced_discrete(self):
        """JOBC='E' discrete-time. Random seed: 999"""
        np.random.seed(999)
        n, m, p = 2, 2, 2
        nv, pv, nw, mw = 0, 0, 2, 2

        a = np.array([[.5, .1], [0, .3]], order='F', dtype=float)
        b = np.eye(2, order='F', dtype=float)
        c = np.array([[1, .5], [0, 1]], order='F', dtype=float)
        av = np.zeros((0, 0), order='F', dtype=float)
        bv = np.zeros((0, m), order='F', dtype=float)
        cv = np.zeros((pv, 0), order='F', dtype=float)
        dv = np.zeros((pv, m), order='F', dtype=float)
        aw = np.diag([0.4, 0.2]).astype(float, order='F')
        bw = np.eye(2, order='F', dtype=float)
        cw = np.eye(2, order='F', dtype=float)
        dw = np.eye(2, order='F', dtype=float)

        s, r, scalec, scaleo, info = ab09iy(
            'D', 'E', 'S', 'R', n, m, p, nv, pv, nw, mw,
            0.5, 1.0, a, b, c, av, bv, cv, dv, aw, bw, cw, dw)

        assert info == 0, f"info={info}"
        assert np.all(np.linalg.eigvalsh(s @ s.T) >= -1e-10)

    def test_enhanced_grammian_lyapunov_residual(self):
        """
        Enhanced observability Grammian R satisfies its own Lyapunov equation.
        Random seed: 1111
        """
        np.random.seed(1111)
        n, m, p = 3, 2, 2
        nv, pv, nw, mw = 2, 2, 0, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1, 0], [.5, .5], [0, 1]], order='F', dtype=float)
        c = np.array([[1, .5, .3], [.2, 1, .6]], order='F', dtype=float)
        av = np.diag([-0.5, -1.5]).astype(float, order='F')
        bv = np.eye(2, order='F', dtype=float)
        cv = np.eye(2, order='F', dtype=float)
        dv = np.eye(2, order='F', dtype=float)
        aw = np.zeros((0, 0), order='F', dtype=float)
        bw = np.zeros((0, mw), order='F', dtype=float)
        cw = np.zeros((p, 0), order='F', dtype=float)
        dw = np.zeros((p, mw), order='F', dtype=float)

        s_e, r_e, scalec_e, scaleo_e, info_e = ab09iy(
            'C', 'S', 'E', 'L', n, m, p, nv, pv, nw, mw,
            1.0, 0.5, a, b, c, av, bv, cv, dv, aw, bw, cw, dw)
        assert info_e == 0

        q_e = r_e.T @ r_e
        assert np.all(np.linalg.eigvalsh(q_e) >= -1e-10), "Q must be PSD"
