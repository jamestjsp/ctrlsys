"""
Tests for AB13AD - Hankel-norm of the ALPHA-stable projection.

AB13AD computes the Hankel-norm of the ALPHA-stable projection of the
transfer-function matrix G of the state-space system (A,B,C).

The procedure:
1. Decompose G = G1 + G2 where G1 has only ALPHA-stable poles
2. Compute Hankel-norm as max Hankel singular value of G1

Mode Parameters:
- DICO: 'C' (continuous), 'D' (discrete)
- EQUIL: 'S' (scale A,B,C), 'N' (no scaling)

ALPHA defines stability boundary:
- Continuous: ALPHA <= 0, eigenvalues with Re(lambda) < ALPHA are stable
- Discrete: 0 <= ALPHA <= 1, eigenvalues with |lambda| < ALPHA are stable

Error codes:
- INFO = 0: success
- INFO = 1: ordered Schur form computation failed
- INFO = 2: eigenvalue separation failed
- INFO = 3: ALPHA-stable part is marginally stable
- INFO = 4: Hankel singular value computation failed
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestAB13ADBasic:
    """Basic functionality tests from HTML doc example."""

    def test_continuous_hankel_norm_from_doc(self):
        """
        Test Hankel-norm computation from SLICOT HTML doc example.

        System: 7th order, 2 inputs, 3 outputs, continuous-time
        ALPHA = 0.0 (purely stable part)
        Expected Hankel-norm = 2.51388

        From AB13AD.html example data.
        """
        from slicot import ab13ad

        n, m, p = 7, 2, 3
        alpha = 0.0

        a = np.array([
            [-0.04165,  0.0000,  4.9200, -4.9200,  0.0000,  0.0000,  0.0000],
            [-5.2100, -12.500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  3.3300, -3.3300,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.5450,  0.0000,  0.0000,  0.0000, -0.5450,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  4.9200, -0.04165, 0.0000,  4.9200],
            [ 0.0000,  0.0000,  0.0000,  0.0000, -5.2100, -12.500,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  3.3300, -3.3300]
        ], order='F', dtype=float)

        b = np.array([
            [0.0000, 0.0000],
            [12.500, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 12.500],
            [0.0000, 0.0000]
        ], order='F', dtype=float)

        c = np.array([
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]
        ], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n, f"All eigenvalues should be stable, ns={ns}"
        assert_allclose(hankel_norm, 2.51388, rtol=1e-3, atol=1e-4,
                        err_msg="Hankel-norm should match doc example")

        expected_hsv = np.array([2.5139, 2.0846, 1.9178, 0.7666, 0.5473, 0.0253, 0.0246])
        assert_allclose(hsv[:ns], expected_hsv, rtol=1e-3, atol=1e-4,
                        err_msg="Hankel singular values should match doc example")

    def test_continuous_hankel_norm_stable_diagonal(self):
        """
        Test Hankel-norm for simple stable diagonal system.

        Random seed: 42 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(42)
        n, m, p = 4, 2, 2
        alpha = 0.0

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

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n, f"All eigenvalues should be stable, ns={ns}"
        assert hankel_norm >= 0, "Hankel-norm must be non-negative"
        assert hankel_norm == hsv[0], "Hankel-norm should equal max HSV"


class TestAB13ADDiscreteTime:
    """Tests for discrete-time systems."""

    def test_discrete_stable_system(self):
        """
        Test Hankel-norm for discrete-time stable system.

        Discrete stable: |eigenvalues| < ALPHA (here ALPHA=1).
        Random seed: 123 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(123)
        n, m, p = 3, 2, 2
        alpha = 1.0

        a = np.diag([0.5, 0.3, 0.2]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.2],
            [0.0, 1.0, 0.5]
        ], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('D', 'N', n, m, p, alpha, a, b, c)

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n, f"All eigenvalues should be in alpha-disk, ns={ns}"
        assert hankel_norm >= 0, "Hankel-norm must be non-negative"

    def test_discrete_partial_stable(self):
        """
        Test with ALPHA < 1 to select subset of eigenvalues.

        Only eigenvalues with |lambda| < 0.4 are considered stable.
        Random seed: 456 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(456)
        n, m, p = 4, 1, 1
        alpha = 0.4

        a = np.diag([0.1, 0.3, 0.6, 0.9]).astype(float, order='F')

        b = np.array([[1.0], [1.0], [1.0], [1.0]], order='F', dtype=float)

        c = np.array([[1.0, 1.0, 1.0, 1.0]], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('D', 'N', n, m, p, alpha, a, b, c)

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == 2, f"Two eigenvalues (0.1, 0.3) should be stable with alpha=0.4, got ns={ns}"


class TestAB13ADMathematicalProperties:
    """Tests for mathematical properties."""

    def test_hsv_decreasing_order(self):
        """
        Validate: Hankel singular values are in decreasing order.

        Random seed: 555 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(555)
        n, m, p = 5, 2, 2
        alpha = 0.0

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

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == n

        for i in range(ns - 1):
            assert hsv[i] >= hsv[i + 1] - 1e-14, \
                f"HSV[{i}]={hsv[i]} should be >= HSV[{i+1}]={hsv[i+1]}"

    def test_hankel_norm_is_max_hsv(self):
        """
        Validate: Hankel-norm equals maximum Hankel singular value.

        Random seed: 666 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(666)
        n, m, p = 4, 2, 3
        alpha = 0.0

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.2, 0.3]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.0, 1.0, 0.5, 0.2],
            [0.3, 0.0, 1.0, 0.4]
        ], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns > 0, "Should have at least one stable mode"
        assert_allclose(hankel_norm, hsv[0], rtol=1e-14,
                        err_msg="Hankel-norm should equal HSV[0]")

    def test_hankel_norm_nonnegative(self):
        """
        Validate: Hankel-norm is always non-negative.

        Random seed: 777 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(777)
        n, m, p = 3, 1, 1
        alpha = 0.0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [0.5], [0.2]], order='F', dtype=float)
        c = np.array([[1.0, 0.5, 0.2]], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert hankel_norm >= 0, "Hankel-norm must be non-negative"
        for i in range(ns):
            assert hsv[i] >= -1e-15, f"HSV[{i}]={hsv[i]} must be non-negative"


class TestAB13ADMIMO:
    """Larger MIMO Hankel-norm tests."""

    def test_larger_mimo_system(self):
        """
        Larger MIMO: 8-state, 3-input, 4-output system.
        Verify HSV ordering and Hankel-norm = max HSV.
        """
        from slicot import ab13ad

        n, m, p = 8, 3, 4
        alpha = 0.0

        a = np.diag([-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]).astype(float, order='F')

        np.random.seed(314)
        b = np.asfortranarray(np.random.randn(n, m))
        c = np.asfortranarray(np.random.randn(p, n))

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == n

        assert_allclose(hankel_norm, hsv[0], rtol=1e-14)

        for i in range(ns - 1):
            assert hsv[i] >= hsv[i + 1] - 1e-14, \
                f"HSV[{i}]={hsv[i]} < HSV[{i+1}]={hsv[i+1]}"


class TestAB13ADWithEquilibration:
    """Tests with equilibration (scaling)."""

    def test_with_scaling(self):
        """
        Test with equilibration enabled (EQUIL='S').

        Random seed: 888 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(888)
        n, m, p = 3, 2, 2
        alpha = 0.0

        a = np.diag([-0.001, -1000.0, -1.0]).astype(float, order='F')

        b = np.array([
            [100.0, 0.0],
            [0.001, 0.001],
            [1.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [0.01, 100.0, 1.0],
            [1.0, 0.001, 0.1]
        ], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'S', n, m, p, alpha, a, b, c)

        assert info == 0, f"Expected info=0, got {info}"
        assert hankel_norm >= 0

    def test_scaling_consistency(self):
        """
        Hankel-norm should be similar with and without scaling for well-conditioned systems.

        Random seed: 999 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(999)
        n, m, p = 3, 2, 2
        alpha = 0.0

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

        norm_no_scale, ns1, hsv1, info1 = ab13ad('C', 'N', n, m, p, alpha, a.copy(), b.copy(), c.copy())
        norm_scale, ns2, hsv2, info2 = ab13ad('C', 'S', n, m, p, alpha, a.copy(), b.copy(), c.copy())

        assert info1 == 0
        assert info2 == 0
        assert ns1 == ns2
        assert_allclose(norm_no_scale, norm_scale, rtol=1e-10,
                        err_msg="Hankel-norm should be similar with/without scaling")


class TestAB13ADEdgeCases:
    """Edge case tests."""

    def test_siso_system(self):
        """
        Test with SISO system (M=1, P=1).

        Random seed: 111 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(111)
        n, m, p = 2, 1, 1
        alpha = 0.0

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == n
        assert hankel_norm >= 0

    def test_minimal_system_n1(self):
        """
        Test with minimal 1st order system (N=1).

        Random seed: 222 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(222)
        n, m, p = 1, 2, 1
        alpha = 0.0

        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == 1
        assert hankel_norm >= 0

    def test_no_stable_eigenvalues(self):
        """
        Test when no eigenvalues are in the stable region.

        Continuous: all eigenvalues > ALPHA means NS=0, Hankel-norm=0.
        """
        from slicot import ab13ad

        n, m, p = 2, 1, 1
        alpha = 0.0

        a = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == 0, "No eigenvalues should be stable"
        assert hankel_norm == 0.0, "Hankel-norm should be 0 when NS=0"


class TestAB13ADQuickReturn:
    """Quick return tests for zero dimensions."""

    def test_n_zero(self):
        """Test quick return when N=0."""
        from slicot import ab13ad

        n, m, p = 0, 2, 1
        alpha = 0.0

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, m), order='F', dtype=float)
        c = np.zeros((p, 1), order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == 0
        assert hankel_norm == 0.0

    def test_m_zero(self):
        """Test quick return when M=0."""
        from slicot import ab13ad

        n, m, p = 2, 0, 1
        alpha = 0.0

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.zeros((n, 1), order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == 0
        assert hankel_norm == 0.0

    def test_p_zero(self):
        """Test quick return when P=0."""
        from slicot import ab13ad

        n, m, p = 2, 2, 0
        alpha = 0.0

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.zeros((1, n), order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == 0
        assert hankel_norm == 0.0


class TestAB13ADErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        from slicot import ab13ad

        n, m, p = 2, 1, 1
        alpha = 0.0

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)

        with pytest.raises(ValueError):
            ab13ad('X', 'N', n, m, p, alpha, a, b, c)

    def test_invalid_equil(self):
        """Test error for invalid EQUIL parameter."""
        from slicot import ab13ad

        n, m, p = 2, 1, 1
        alpha = 0.0

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)

        with pytest.raises(ValueError):
            ab13ad('C', 'X', n, m, p, alpha, a, b, c)

    def test_invalid_alpha_continuous(self):
        """Test error for invalid ALPHA with continuous-time (ALPHA must be <= 0)."""
        from slicot import ab13ad

        n, m, p = 2, 1, 1
        alpha = 0.5

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab13ad('C', 'N', n, m, p, alpha, a, b, c)

    def test_invalid_alpha_discrete_negative(self):
        """Test error for invalid ALPHA with discrete-time (ALPHA must be >= 0)."""
        from slicot import ab13ad

        n, m, p = 2, 1, 1
        alpha = -0.5

        a = np.array([[0.5, 0.0], [0.0, 0.3]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab13ad('D', 'N', n, m, p, alpha, a, b, c)

    def test_invalid_alpha_discrete_greater_than_one(self):
        """Test error for invalid ALPHA with discrete-time (ALPHA must be <= 1)."""
        from slicot import ab13ad

        n, m, p = 2, 1, 1
        alpha = 1.5

        a = np.array([[0.5, 0.0], [0.0, 0.3]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab13ad('D', 'N', n, m, p, alpha, a, b, c)


class TestAB13ADNumericalStability:
    """Numerical stability tests."""

    def test_well_separated_eigenvalues(self):
        """
        Test with well-separated eigenvalues for numerical stability.

        Random seed: 333 (for reproducibility)
        """
        from slicot import ab13ad

        np.random.seed(333)
        n, m, p = 4, 2, 2
        alpha = 0.0

        a = np.diag([-10.0, -1.0, -0.1, -0.01]).astype(float, order='F')

        b = np.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 1.0, 0.0, 0.5],
            [0.0, 1.0, 1.0, 0.5]
        ], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == n
        assert hankel_norm > 0

    def test_mixed_stable_unstable_continuous(self):
        """
        Test with mixed stable/unstable eigenvalues (continuous).

        Eigenvalues: -1.0, -0.5 (stable), +0.5, +1.0 (unstable)
        Only first 2 should be in stable projection.
        """
        from slicot import ab13ad

        n, m, p = 4, 1, 1
        alpha = 0.0

        a = np.diag([-1.0, -0.5, 0.5, 1.0]).astype(float, order='F')

        b = np.array([[1.0], [1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0, 1.0]], order='F', dtype=float)

        hankel_norm, ns, hsv, info = ab13ad('C', 'N', n, m, p, alpha, a, b, c)

        assert info == 0
        assert ns == 2, f"Only 2 eigenvalues should be stable, got ns={ns}"
        assert hankel_norm >= 0
