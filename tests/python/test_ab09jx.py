"""
Tests for AB09JX - Check stability/antistability of finite eigenvalues

AB09JX checks whether all finite eigenvalues (or reciprocal eigenvalues)
lie within a specified stability domain defined by:
- DICO: 'C' (continuous-time) or 'D' (discrete-time)
- STDOM: 'S' (stability) or 'U' (instability)
- ALPHA: boundary value for real parts (continuous) or moduli (discrete)

Domain definitions:
- Continuous, stable: Re(lambda) < ALPHA
- Continuous, unstable: Re(lambda) > ALPHA
- Discrete, stable: |lambda| < ALPHA
- Discrete, unstable: |lambda| > ALPHA

For EVTYPE='R' (reciprocal), conditions apply to 1/lambda.
"""

import numpy as np
import pytest
from ctrlsys import ab09jx


class TestAB09JXBasic:
    """Basic functionality tests for AB09JX."""

    def test_continuous_stable_all_in_domain(self):
        """
        Test continuous-time system with all eigenvalues in stable domain.

        Eigenvalues: -1, -2, -3 all have Re(lambda) < 0.
        """
        n = 3
        er = np.array([-1.0, -2.0, -3.0], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_continuous_stable_outside_domain(self):
        """
        Test continuous-time system with eigenvalue outside stable domain.

        Eigenvalues: -1, 1, -3 where 1 > 0, so outside stable region.
        """
        n = 3
        er = np.array([-1.0, 1.0, -3.0], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 1

    def test_discrete_stable_all_in_domain(self):
        """
        Test discrete-time system with all eigenvalues in stable domain.

        Eigenvalues: 0.5, 0.3, -0.4 all have |lambda| < 1.
        """
        n = 3
        er = np.array([0.5, 0.3, -0.4], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 1.0
        tolinf = 1e-10

        info = ab09jx('D', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_discrete_stable_outside_domain(self):
        """
        Test discrete-time system with eigenvalue outside stable domain.

        Eigenvalues: 0.5, 1.5, -0.4 where |1.5| > 1.
        """
        n = 3
        er = np.array([0.5, 1.5, -0.4], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 1.0
        tolinf = 1e-10

        info = ab09jx('D', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 1


class TestAB09JXGeneralizedEigenvalues:
    """Tests for generalized eigenvalue problems (EVTYPE='G')."""

    def test_generalized_continuous_stable(self):
        """
        Test generalized eigenvalues for continuous-time stable domain.

        Eigenvalues: (-1+0i)/1, (-2+0i)/1, (-3+0i)/1.
        All have Re(lambda) < 0.
        """
        n = 3
        er = np.array([-1.0, -2.0, -3.0], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'G', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_generalized_with_scaling(self):
        """
        Test generalized eigenvalues with non-unit scaling.

        Eigenvalues: (-2+0i)/2 = -1, (-6+0i)/2 = -3.
        All have Re(lambda) < 0.
        """
        n = 2
        er = np.array([-2.0, -6.0], order='F', dtype=float)
        ei = np.array([0.0, 0.0], order='F', dtype=float)
        ed = np.array([2.0, 2.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'G', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_generalized_infinite_eigenvalue(self):
        """
        Test that infinite eigenvalues (ED=0) are skipped.

        Eigenvalues: -1, inf (ED=0), -3. Only finite ones checked.
        """
        n = 3
        er = np.array([-1.0, 1.0, -3.0], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 0.0, 1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'G', n, alpha, er, ei, ed, tolinf)
        assert info == 0


class TestAB09JXReciprocalEigenvalues:
    """Tests for reciprocal eigenvalue problems (EVTYPE='R')."""

    def test_reciprocal_continuous_stable(self):
        """
        Test reciprocal eigenvalues for continuous-time stable domain.

        For EVTYPE='R', the check is: ED/(ER+i*EI) must satisfy Re(reciprocal) < ALPHA.
        The Fortran code transforms to: ED < ALPHA*ER for continuous stability check.

        For ER=-1, ED=-1: reciprocal = (-1)/(-1) = 1 which has Re > 0, so fails.
        For ER=-1, ED=1: reciprocal = 1/(-1) = -1 which has Re < 0.
        But the check is: ED >= ALPHA*ER => 1 >= 0*(-1) = 0, so 1>=0 is TRUE, fails.

        For stable domain, we need ED < ALPHA*ER.
        With ER negative and ALPHA=0: 0*ER=0, so need ED < 0.
        """
        n = 3
        er = np.array([-1.0, -2.0, -3.0], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([-1.0, -1.0, -1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'R', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_reciprocal_discrete_stable(self):
        """
        Test reciprocal eigenvalues for discrete-time stable domain.

        For EVTYPE='R', check |1/lambda| < alpha.
        Eigenvalues: 0.5, 0.25, 0.2 (all < 1).
        Reciprocals: 2, 4, 5 (all > 1).
        So with alpha=1, checking |1/lambda| < 1 should FAIL.
        """
        n = 3
        er = np.array([0.5, 0.25, 0.2], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 1.0
        tolinf = 1e-10

        info = ab09jx('D', 'S', 'R', n, alpha, er, ei, ed, tolinf)
        assert info == 1


class TestAB09JXUnstableDomain:
    """Tests for instability domain (STDOM='U')."""

    def test_continuous_unstable_all_in_domain(self):
        """
        Test continuous-time system checking instability domain.

        For STDOM='U': Re(lambda) > ALPHA.
        Eigenvalues: 1, 2, 3 all have Re(lambda) > 0.
        """
        n = 3
        er = np.array([1.0, 2.0, 3.0], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'U', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_continuous_unstable_outside_domain(self):
        """
        Test continuous-time system with eigenvalue inside stable region.

        For STDOM='U': Re(lambda) > ALPHA.
        Eigenvalues: 1, -2, 3 where -2 < 0, so outside unstable region.
        """
        n = 3
        er = np.array([1.0, -2.0, 3.0], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'U', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 1

    def test_discrete_unstable_all_in_domain(self):
        """
        Test discrete-time system checking instability domain.

        For STDOM='U': |lambda| > ALPHA.
        Eigenvalues: 1.5, 2.0, 3.0 all have |lambda| > 1.
        """
        n = 3
        er = np.array([1.5, 2.0, 3.0], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 1.0
        tolinf = 1e-10

        info = ab09jx('D', 'U', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 0


class TestAB09JXComplexEigenvalues:
    """Tests for complex conjugate eigenvalues."""

    def test_complex_continuous_stable(self):
        """
        Test complex eigenvalues in continuous-time stable domain.

        Eigenvalues: -1+2i, -1-2i (complex conjugate pair), -3.
        All have Re(lambda) < 0.
        """
        n = 3
        er = np.array([-1.0, -1.0, -3.0], order='F', dtype=float)
        ei = np.array([2.0, -2.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_complex_discrete_stable(self):
        """
        Test complex eigenvalues in discrete-time stable domain.

        Eigenvalues: 0.3+0.4i, 0.3-0.4i (|lambda| = 0.5), -0.2.
        All have |lambda| < 1.
        """
        n = 3
        er = np.array([0.3, 0.3, -0.2], order='F', dtype=float)
        ei = np.array([0.4, -0.4, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 1.0
        tolinf = 1e-10

        info = ab09jx('D', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_complex_discrete_outside_domain(self):
        """
        Test complex eigenvalue outside discrete-time stable domain.

        Eigenvalues: 0.6+0.8i, 0.6-0.8i (|lambda| = 1.0 exactly), -0.2.
        With alpha=0.9, the complex pair has |lambda| >= alpha.
        """
        n = 3
        er = np.array([0.6, 0.6, -0.2], order='F', dtype=float)
        ei = np.array([0.8, -0.8, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = 0.9
        tolinf = 1e-10

        info = ab09jx('D', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 1


class TestAB09JXEdgeCases:
    """Edge case tests for AB09JX."""

    def test_empty_input(self):
        """Test with N=0 (empty eigenvalue list)."""
        n = 0
        er = np.array([], order='F', dtype=float)
        ei = np.array([], order='F', dtype=float)
        ed = np.array([], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_single_eigenvalue(self):
        """Test with N=1 (single eigenvalue)."""
        n = 1
        er = np.array([-1.0], order='F', dtype=float)
        ei = np.array([0.0], order='F', dtype=float)
        ed = np.array([1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 0

    def test_eigenvalue_on_boundary_stable(self):
        """
        Test eigenvalue exactly on boundary for stability domain.

        Eigenvalue at alpha exactly: Re(lambda) = alpha.
        For STDOM='S': Re(lambda) < alpha, so on boundary should FAIL.
        """
        n = 1
        er = np.array([0.0], order='F', dtype=float)
        ei = np.array([0.0], order='F', dtype=float)
        ed = np.array([1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 1

    def test_eigenvalue_on_boundary_unstable(self):
        """
        Test eigenvalue exactly on boundary for instability domain.

        Eigenvalue at alpha exactly: Re(lambda) = alpha.
        For STDOM='U': Re(lambda) > alpha, so on boundary should FAIL.
        """
        n = 1
        er = np.array([0.0], order='F', dtype=float)
        ei = np.array([0.0], order='F', dtype=float)
        ed = np.array([1.0], order='F', dtype=float)
        alpha = 0.0
        tolinf = 1e-10

        info = ab09jx('C', 'U', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 1

    def test_nondefault_alpha(self):
        """
        Test with non-zero ALPHA boundary.

        Eigenvalues: -0.5, -1.5, -2.5.
        With ALPHA=-1, check Re(lambda) < -1:
        -0.5 >= -1, so should FAIL.
        """
        n = 3
        er = np.array([-0.5, -1.5, -2.5], order='F', dtype=float)
        ei = np.array([0.0, 0.0, 0.0], order='F', dtype=float)
        ed = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
        alpha = -1.0
        tolinf = 1e-10

        info = ab09jx('C', 'S', 'S', n, alpha, er, ei, ed, tolinf)
        assert info == 1


class TestAB09JXErrorHandling:
    """Error handling tests for AB09JX."""

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        n = 1
        er = np.array([-1.0], order='F', dtype=float)
        ei = np.array([0.0], order='F', dtype=float)
        ed = np.array([1.0], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jx('X', 'S', 'S', n, 0.0, er, ei, ed, 1e-10)

    def test_invalid_stdom(self):
        """Test error for invalid STDOM parameter."""
        n = 1
        er = np.array([-1.0], order='F', dtype=float)
        ei = np.array([0.0], order='F', dtype=float)
        ed = np.array([1.0], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jx('C', 'X', 'S', n, 0.0, er, ei, ed, 1e-10)

    def test_invalid_evtype(self):
        """Test error for invalid EVTYPE parameter."""
        n = 1
        er = np.array([-1.0], order='F', dtype=float)
        ei = np.array([0.0], order='F', dtype=float)
        ed = np.array([1.0], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jx('C', 'S', 'X', n, 0.0, er, ei, ed, 1e-10)

    def test_negative_n(self):
        """Test error for negative N."""
        er = np.array([-1.0], order='F', dtype=float)
        ei = np.array([0.0], order='F', dtype=float)
        ed = np.array([1.0], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jx('C', 'S', 'S', -1, 0.0, er, ei, ed, 1e-10)

    def test_negative_alpha_discrete(self):
        """Test error for negative ALPHA in discrete-time."""
        n = 1
        er = np.array([0.5], order='F', dtype=float)
        ei = np.array([0.0], order='F', dtype=float)
        ed = np.array([1.0], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jx('D', 'S', 'S', n, -1.0, er, ei, ed, 1e-10)

    def test_tolinf_out_of_range(self):
        """Test error for TOLINF >= 1 or < 0."""
        n = 1
        er = np.array([-1.0], order='F', dtype=float)
        ei = np.array([0.0], order='F', dtype=float)
        ed = np.array([1.0], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jx('C', 'S', 'G', n, 0.0, er, ei, ed, 1.0)

        with pytest.raises((ValueError, RuntimeError)):
            ab09jx('C', 'S', 'G', n, 0.0, er, ei, ed, -0.1)


class TestAB09JXNumpyCrossCheck:
    """Cross-check AB09JX results against numpy eigenvalue classification."""

    @pytest.mark.parametrize("dico,stdom,alpha,er,ei,ed", [
        ('C', 'S', 0.0, [-1.0, -2.0, -3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ('C', 'S', 0.0, [-1.0, 1.0, -3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ('C', 'U', 0.0, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ('D', 'S', 1.0, [0.5, 0.3, -0.4], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ('D', 'S', 1.0, [0.5, 1.5, -0.4], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ('D', 'U', 1.0, [1.5, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ('C', 'S', -1.0, [-1.5, -2.5, -3.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ('C', 'S', 0.0, [-1.0, -1.0, -3.0], [2.0, -2.0, 0.0], [1.0, 1.0, 1.0]),
        ('D', 'S', 1.0, [0.3, 0.3, -0.2], [0.4, -0.4, 0.0], [1.0, 1.0, 1.0]),
    ])
    def test_eigenvalue_domain_matches_numpy(self, dico, stdom, alpha, er, ei, ed):
        er_arr = np.array(er, order='F', dtype=float)
        ei_arr = np.array(ei, order='F', dtype=float)
        ed_arr = np.array(ed, order='F', dtype=float)
        n = len(er)

        eigvals = (er_arr + 1j * ei_arr) / ed_arr

        if dico == 'C' and stdom == 'S':
            all_in_domain = np.all(np.real(eigvals) < alpha)
        elif dico == 'C' and stdom == 'U':
            all_in_domain = np.all(np.real(eigvals) > alpha)
        elif dico == 'D' and stdom == 'S':
            all_in_domain = np.all(np.abs(eigvals) < alpha)
        else:
            all_in_domain = np.all(np.abs(eigvals) > alpha)

        info = ab09jx(dico, stdom, 'S', n, alpha, er_arr, ei_arr, ed_arr, 1e-10)

        if all_in_domain:
            assert info == 0
        else:
            assert info == 1
