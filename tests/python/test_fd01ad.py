"""
FD01AD - Fast recursive least-squares filtering.

Solves the least-squares filtering problem recursively in time using
a fast QR-decomposition based approach. Each call implements one time
update of the solution.

Test data extracted from SLICOT HTML documentation example.
"""

import numpy as np
import pytest
from ctrlsys import fd01ad


class TestFD01ADBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_basic_both_mode(self):
        """
        Test basic FD01AD functionality with JP='B' (both prediction and filtering).

        Test data from SLICOT FD01AD.html documentation example.
        L=2, LAMBDA=0.99, 500 iterations.

        Input: XIN = sin(0.3*i)
        Reference: YIN = 0.5*sin(0.3*i) + 2.0*sin(0.3*(i-1))

        Expected outputs after 500 iterations (from HTML doc):
        XF[0] = 4.880088, XF[1] = -1.456881
        YQ[0] = 12.307615, YQ[1] = 2.914057
        EPSBCK[0] = -0.140367, EPSBCK[1] = -0.140367, EPSBCK[2] = 0.980099
        EFOR = 0.197e-02
        """
        l = 2
        lam = 0.99
        delta = 1.0e-2  # soft start
        n_iter = 500

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)
        efor = delta

        for i in range(1, n_iter + 1):
            xin = np.sin(0.3 * float(i))
            yin = 0.5 * np.sin(0.3 * float(i)) + 2.0 * np.sin(0.3 * float(i - 1))

            xf, epsbck, cteta, steta, yq, efor, epos, eout, salph, iwarn, info = fd01ad(
                'B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq
            )
            assert info == 0, f"fd01ad failed at iteration {i} with info={info}"

        xf_expected = np.array([4.880088, -1.456881], dtype=float)
        yq_expected = np.array([12.307615, 2.914057], dtype=float)
        epsbck_expected = np.array([-0.140367, -0.140367, 0.980099], dtype=float)
        efor_expected = 0.197e-02

        np.testing.assert_allclose(xf, xf_expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(yq, yq_expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(epsbck, epsbck_expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(efor, efor_expected, rtol=1e-2)

    def test_prediction_only_mode(self):
        """
        Test FD01AD with JP='P' (prediction only, no filtering).

        With JP='P', YIN and YQ are not used, EOUT is not computed.
        """
        l = 3
        lam = 0.99
        delta = 1.0e-2

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)
        efor = delta

        n_iter = 100
        for i in range(1, n_iter + 1):
            xin = np.sin(0.3 * float(i))
            yin = 0.0  # not used in 'P' mode

            xf, epsbck, cteta, steta, yq, efor, epos, eout, salph, iwarn, info = fd01ad(
                'P', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq
            )
            assert info == 0, f"fd01ad failed at iteration {i} with info={info}"

        assert np.all(np.isfinite(xf))
        assert np.all(np.isfinite(epsbck))
        assert np.all(np.isfinite(salph))
        assert efor > 0


class TestFD01ADEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_l1(self):
        """Test with minimal L=1 (edge case)."""
        l = 1
        lam = 0.99
        delta = 1.0e-2

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)
        efor = delta

        for i in range(1, 50):
            xin = np.sin(0.5 * float(i))
            yin = 0.7 * xin

            xf, epsbck, cteta, steta, yq, efor, epos, eout, salph, iwarn, info = fd01ad(
                'B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq
            )
            assert info == 0

        assert xf.shape == (1,)
        assert epsbck.shape == (2,)
        assert salph.shape == (1,)

    def test_large_l(self):
        """Test with larger L=10."""
        l = 10
        lam = 0.98
        delta = 1.0e-2

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)
        efor = delta

        np.random.seed(42)
        for i in range(1, 100):
            xin = np.random.randn()
            yin = 0.5 * xin + 0.3 * np.random.randn()

            xf, epsbck, cteta, steta, yq, efor, epos, eout, salph, iwarn, info = fd01ad(
                'B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq
            )
            assert info == 0

        assert xf.shape == (l,)
        assert salph.shape == (l,)
        assert np.all(np.isfinite(xf))

    def test_lambda_one_exact(self):
        """Test with LAMBDA=1.0 exactly (no forgetting)."""
        l = 2
        lam = 1.0  # no forgetting factor
        delta = 1.0e-2

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)
        efor = delta

        for i in range(1, 20):
            xin = np.sin(0.3 * float(i))
            yin = 0.5 * xin

            xf, epsbck, cteta, steta, yq, efor, epos, eout, salph, iwarn, info = fd01ad(
                'B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq
            )
            assert info == 0

        assert np.all(np.isfinite(xf))


class TestFD01ADErrorHandling:
    """Test parameter validation and error handling."""

    def test_invalid_jp(self):
        """Test with invalid JP parameter."""
        l = 2
        lam = 0.99
        efor = 0.01

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)

        xin, yin = 1.0, 0.5

        with pytest.raises((ValueError, RuntimeError)):
            fd01ad('X', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq)

    def test_invalid_l_zero(self):
        """Test with invalid L=0."""
        l = 0
        lam = 0.99
        efor = 0.01

        xf = np.zeros(1, dtype=float)
        epsbck = np.zeros(1, dtype=float)
        cteta = np.zeros(1, dtype=float)
        steta = np.zeros(1, dtype=float)
        yq = np.zeros(1, dtype=float)

        xin, yin = 1.0, 0.5

        with pytest.raises((ValueError, RuntimeError)):
            fd01ad('B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq)

    def test_invalid_lambda_zero(self):
        """Test with invalid LAMBDA=0."""
        l = 2
        lam = 0.0  # invalid: must be > 0
        efor = 0.01

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)

        xin, yin = 1.0, 0.5

        with pytest.raises((ValueError, RuntimeError)):
            fd01ad('B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq)

    def test_invalid_lambda_greater_than_one(self):
        """Test with invalid LAMBDA > 1."""
        l = 2
        lam = 1.5  # invalid: must be <= 1
        efor = 0.01

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)

        xin, yin = 1.0, 0.5

        with pytest.raises((ValueError, RuntimeError)):
            fd01ad('B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq)


class TestFD01ADNumericalProperties:
    """Test mathematical properties of the least-squares filter."""

    def test_energy_accumulation(self):
        """
        Verify exponentially weighted input signal energy relationship.

        The weighted input energy satisfies:
        sum_{k=1}^{n} [LAMBDA^{2(n-k)} * XIN(k)^2] = EFOR^2 + sum_{i=1}^{L} XF(i)^2

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        l = 3
        lam = 0.95
        delta = 1.0e-2

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)
        efor = delta

        n_iter = 50
        xin_history = []
        for i in range(1, n_iter + 1):
            xin = np.sin(0.3 * float(i)) + 0.1 * np.random.randn()
            xin_history.append(xin)
            yin = 0.5 * xin

            xf, epsbck, cteta, steta, yq, efor, epos, eout, salph, iwarn, info = fd01ad(
                'B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq
            )
            assert info == 0

        computed_energy = efor**2 + np.sum(xf**2)
        weighted_energy = sum(
            (lam ** (2 * (n_iter - k - 1))) * xin_history[k]**2
            for k in range(n_iter)
        )
        np.testing.assert_allclose(computed_energy, weighted_energy, rtol=0.15)

    def test_output_error_decreases(self):
        """
        Verify that output error decreases as filter converges.

        For a consistent signal, the filter should learn the relationship
        and reduce output error over time.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        l = 2
        lam = 0.99
        delta = 1.0e-2

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)
        efor = delta

        early_errors = []
        late_errors = []

        for i in range(1, 201):
            xin = np.sin(0.3 * float(i))
            yin = 0.5 * np.sin(0.3 * float(i)) + 2.0 * np.sin(0.3 * float(i - 1))

            xf, epsbck, cteta, steta, yq, efor, epos, eout, salph, iwarn, info = fd01ad(
                'B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq
            )
            assert info == 0

            if 10 <= i < 30:
                early_errors.append(abs(eout))
            elif 180 <= i <= 200:
                late_errors.append(abs(eout))

        avg_early = np.mean(early_errors)
        avg_late = np.mean(late_errors)
        assert avg_late < avg_early, "Output error should decrease as filter converges"

    def test_rotation_angles_normalized(self):
        """
        Verify rotation angles satisfy cos^2 + sin^2 = 1.

        This is a fundamental property of Givens rotations.
        """
        l = 3
        lam = 0.99
        delta = 1.0e-2

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)
        efor = delta

        for i in range(1, 100):
            xin = np.sin(0.3 * float(i))
            yin = 0.5 * xin

            xf, epsbck, cteta, steta, yq, efor, epos, eout, salph, iwarn, info = fd01ad(
                'B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq
            )
            assert info == 0

            for j in range(l):
                norm_sq = cteta[j]**2 + steta[j]**2
                np.testing.assert_allclose(norm_sq, 1.0, rtol=1e-12,
                    err_msg=f"Rotation angle {j} not normalized: cos^2+sin^2={norm_sq}")

    def test_conversion_factor_bounded(self):
        """
        Verify conversion factor EPSBCK[L] stays in valid range [0, 1].

        The conversion factor is the square root of a quantity bounded by 1.
        """
        l = 3
        lam = 0.99
        delta = 1.0e-2

        xf = np.zeros(l, dtype=float)
        epsbck = np.zeros(l + 1, dtype=float)
        epsbck[l] = 1.0
        cteta = np.ones(l, dtype=float)
        steta = np.zeros(l, dtype=float)
        yq = np.zeros(l, dtype=float)
        efor = delta

        for i in range(1, 100):
            xin = np.sin(0.3 * float(i))
            yin = 0.5 * xin

            xf, epsbck, cteta, steta, yq, efor, epos, eout, salph, iwarn, info = fd01ad(
                'B', l, lam, xin, yin, efor, xf, epsbck, cteta, steta, yq
            )
            assert info == 0

            conv_factor = epsbck[l]
            assert 0.0 <= conv_factor <= 1.0 + 1e-10, \
                f"Conversion factor {conv_factor} out of bounds [0, 1]"
