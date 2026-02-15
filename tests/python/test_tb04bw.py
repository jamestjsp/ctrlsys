"""
Tests for TB04BW: Sum of rational matrix and real matrix.

Computes G + D where G is a P-by-M rational matrix (polynomial ratios)
and D is a P-by-M real matrix.
"""

import numpy as np
import pytest
from ctrlsys import tb04bw


class TestTB04BWBasic:
    """Basic functionality tests."""

    def test_simple_scalar_increasing_order(self):
        """
        Test scalar case with increasing polynomial order.

        G(1,1) = (1 + 2s) / (1 + s) and D = 3
        Result: (1+2s+3*(1+s)) / (1+s) = (4+5s) / (1+s)

        Numerator changes from [1, 2] to [4, 5].
        """
        p, m, md = 1, 1, 2

        ign = np.array([[1]], dtype=np.int32, order='F')
        igd = np.array([[1]], dtype=np.int32, order='F')

        gn = np.array([1.0, 2.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 1.0], dtype=np.float64, order='F')
        d = np.array([[3.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == 0
        assert ign_out[0, 0] == 1
        np.testing.assert_allclose(gn_out[0], 4.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[1], 5.0, rtol=1e-14)

    def test_simple_scalar_decreasing_order(self):
        """
        Test scalar case with decreasing polynomial order.

        G(1,1) = (2s + 1) / (s + 1) = (2 + 1s^0) / (1 + 1s^0) in decreasing order
        With D = 3:
        Result: (2+3*1) + (1+3*1)s^0 = 5 + 4s^0
        Numerator [2, 1] → [5, 4]
        """
        p, m, md = 1, 1, 2

        ign = np.array([[1]], dtype=np.int32, order='F')
        igd = np.array([[1]], dtype=np.int32, order='F')

        gn = np.array([2.0, 1.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 1.0], dtype=np.float64, order='F')
        d = np.array([[3.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('D', p, m, md, ign, igd, gn, gd, d)

        assert info == 0
        assert ign_out[0, 0] == 1
        np.testing.assert_allclose(gn_out[0], 5.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[1], 4.0, rtol=1e-14)

    def test_zero_d_no_change(self):
        """
        Test that adding D=0 leaves numerator unchanged.

        G(1,1) = (1 + 2s) / (1 + s), D = 0
        Result should be unchanged.
        """
        p, m, md = 1, 1, 2

        ign = np.array([[1]], dtype=np.int32, order='F')
        igd = np.array([[1]], dtype=np.int32, order='F')

        gn = np.array([1.0, 2.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 1.0], dtype=np.float64, order='F')
        d = np.array([[0.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == 0
        assert ign_out[0, 0] == 1
        np.testing.assert_allclose(gn_out[0], 1.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[1], 2.0, rtol=1e-14)


class TestTB04BWDegreeIncrease:
    """Tests where numerator degree increases."""

    def test_degree_increase_increasing_order(self):
        """
        Test case where nn < nd causes degree increase (increasing order).

        G(1,1) = 1 / (1 + s + s^2), D = 2
        num = [1, 0, 0], den = [1, 1, 1], degree(num) = 0, degree(den) = 2

        Result: (1 + 2*(1+s+s^2)) / (1+s+s^2) = (3 + 2s + 2s^2) / (1+s+s^2)
        Numerator degree increases from 0 to 2.
        """
        p, m, md = 1, 1, 3

        ign = np.array([[0]], dtype=np.int32, order='F')
        igd = np.array([[2]], dtype=np.int32, order='F')

        gn = np.array([1.0, 0.0, 0.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 1.0, 1.0], dtype=np.float64, order='F')
        d = np.array([[2.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == 0
        assert ign_out[0, 0] == 2
        np.testing.assert_allclose(gn_out[0], 3.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[1], 2.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[2], 2.0, rtol=1e-14)

    def test_degree_increase_decreasing_order(self):
        """
        Test case where nn < nd causes degree increase (decreasing order).

        G(1,1) = 1 / (s^2 + s + 1), D = 2
        In decreasing order with degree nn=0: num = [1, *, *] (only position 0 used)
        den = [1, 1, 1] (s^2 + s + 1), degree(den) = 2

        Result: num + D*den = 1 + 2*(s^2+s+1) = 2s^2 + 2s + 3
        In decreasing order: [2, 2, 3]
        """
        p, m, md = 1, 1, 3

        ign = np.array([[0]], dtype=np.int32, order='F')
        igd = np.array([[2]], dtype=np.int32, order='F')

        gn = np.array([1.0, 0.0, 0.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 1.0, 1.0], dtype=np.float64, order='F')
        d = np.array([[2.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('D', p, m, md, ign, igd, gn, gd, d)

        assert info == 0
        assert ign_out[0, 0] == 2
        np.testing.assert_allclose(gn_out[0], 2.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[1], 2.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[2], 3.0, rtol=1e-14)


class TestTB04BWMultiDimensional:
    """Tests for multi-dimensional matrices."""

    def test_2x2_matrix_increasing_order(self):
        """
        Test 2x2 rational matrix with increasing order.

        Random seed: 42 (for reproducibility)

        G = [[1/(1+s), 0], [0, 2/(1+2s)]]
        D = [[1, 0.5], [0.5, 1]]

        G + D computed element-wise.
        """
        p, m, md = 2, 2, 2

        ign = np.array([[0, 0], [0, 0]], dtype=np.int32, order='F')
        igd = np.array([[1, 0], [0, 1]], dtype=np.int32, order='F')

        gn = np.zeros(p * m * md, dtype=np.float64, order='F')
        gd = np.zeros(p * m * md, dtype=np.float64, order='F')

        gn[0] = 1.0; gd[0] = 1.0; gd[1] = 1.0
        gn[2] = 0.0; gd[2] = 1.0
        gn[4] = 0.0; gd[4] = 1.0
        gn[6] = 2.0; gd[6] = 1.0; gd[7] = 2.0

        d = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == 0
        np.testing.assert_allclose(gn_out[0], 2.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[1], 1.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[2], 0.5, rtol=1e-14)
        np.testing.assert_allclose(gn_out[4], 0.5, rtol=1e-14)
        np.testing.assert_allclose(gn_out[6], 3.0, rtol=1e-14)
        np.testing.assert_allclose(gn_out[7], 2.0, rtol=1e-14)


class TestTB04BWSpecialCases:
    """Test special cases and edge conditions."""

    def test_zero_rational_with_nonzero_d(self):
        """
        Test adding D to zero rational (g(i,j) = 0).

        When g(i,j) = 0 (both nn=0, nd=0, and num[k]=0), adding D
        should give D in the numerator.
        """
        p, m, md = 1, 1, 2

        ign = np.array([[0]], dtype=np.int32, order='F')
        igd = np.array([[0]], dtype=np.int32, order='F')

        gn = np.array([0.0, 0.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 0.0], dtype=np.float64, order='F')
        d = np.array([[5.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == 0
        np.testing.assert_allclose(gn_out[0], 5.0, rtol=1e-14)

    def test_constant_ratio_with_d(self):
        """
        Test constant numerator and denominator (nn=0, nd=0, num!=0).

        G = 2 / 3 (constant), D = 1
        Result = 2/3 + 1 = 2/3 + 3/3 = 5/3
        But since den=3, we compute: (2 + 1*3) / 3 = 5/3
        So numerator becomes 5.
        """
        p, m, md = 1, 1, 2

        ign = np.array([[0]], dtype=np.int32, order='F')
        igd = np.array([[0]], dtype=np.int32, order='F')

        gn = np.array([2.0, 0.0], dtype=np.float64, order='F')
        gd = np.array([3.0, 0.0], dtype=np.float64, order='F')
        d = np.array([[1.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == 0
        np.testing.assert_allclose(gn_out[0], 5.0, rtol=1e-14)


class TestTB04BWMathematicalProperties:
    """Tests for mathematical property preservation."""

    def test_transfer_function_evaluation(self):
        """
        Validate G(s) + D = (num + D*den) / den at specific s values.

        Random seed: 123 (for reproducibility)

        Tests the mathematical identity:
        (G + D)(s) = (num(s) + D * den(s)) / den(s)
        """
        p, m, md = 1, 1, 3
        np.random.seed(123)

        ign = np.array([[2]], dtype=np.int32, order='F')
        igd = np.array([[2]], dtype=np.int32, order='F')

        num_coeffs = np.array([1.0, 2.0, 1.0])
        den_coeffs = np.array([1.0, 3.0, 2.0])
        d_val = 2.5

        gn = num_coeffs.astype(np.float64).copy(order='F')
        gd = den_coeffs.astype(np.float64).copy(order='F')
        d = np.array([[d_val]], dtype=np.float64, order='F')

        s_test = np.array([0.0, 1.0, 2.0, -0.5])

        g_before = np.zeros_like(s_test)
        for idx, s in enumerate(s_test):
            num_val = num_coeffs[0] + num_coeffs[1]*s + num_coeffs[2]*s**2
            den_val = den_coeffs[0] + den_coeffs[1]*s + den_coeffs[2]*s**2
            g_before[idx] = num_val / den_val

        g_plus_d_expected = g_before + d_val

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == 0

        g_after = np.zeros_like(s_test)
        for idx, s in enumerate(s_test):
            num_val = gn_out[0] + gn_out[1]*s + gn_out[2]*s**2
            den_val = den_coeffs[0] + den_coeffs[1]*s + den_coeffs[2]*s**2
            g_after[idx] = num_val / den_val

        np.testing.assert_allclose(g_after, g_plus_d_expected, rtol=1e-14)


class TestTB04BWErrorHandling:
    """Tests for error conditions."""

    def test_invalid_order(self):
        """Test invalid order parameter."""
        p, m, md = 1, 1, 2

        ign = np.array([[1]], dtype=np.int32, order='F')
        igd = np.array([[1]], dtype=np.int32, order='F')
        gn = np.array([1.0, 2.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 1.0], dtype=np.float64, order='F')
        d = np.array([[1.0]], dtype=np.float64, order='F')

        with pytest.raises(ValueError, match="order must be 'I' or 'D'"):
            tb04bw('X', p, m, md, ign, igd, gn, gd, d)

    def test_negative_p(self):
        """Test negative p parameter."""
        p, m, md = -1, 1, 2

        ign = np.array([[1]], dtype=np.int32, order='F')
        igd = np.array([[1]], dtype=np.int32, order='F')
        gn = np.array([1.0, 2.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 1.0], dtype=np.float64, order='F')
        d = np.array([[1.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == -2

    def test_negative_m(self):
        """Test negative m parameter."""
        p, m, md = 1, -1, 2

        ign = np.array([[1]], dtype=np.int32, order='F')
        igd = np.array([[1]], dtype=np.int32, order='F')
        gn = np.array([1.0, 2.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 1.0], dtype=np.float64, order='F')
        d = np.array([[1.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == -3

    def test_invalid_md(self):
        """Test invalid md parameter (< 1)."""
        p, m, md = 1, 1, 0

        ign = np.array([[1]], dtype=np.int32, order='F')
        igd = np.array([[1]], dtype=np.int32, order='F')
        gn = np.array([1.0, 2.0], dtype=np.float64, order='F')
        gd = np.array([1.0, 1.0], dtype=np.float64, order='F')
        d = np.array([[1.0]], dtype=np.float64, order='F')

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == -4


class TestTB04BWQuickReturn:
    """Test quick return conditions."""

    def test_zero_p(self):
        """Test p=0 quick return."""
        p, m, md = 0, 1, 2

        ign = np.array([[]], dtype=np.int32, order='F').reshape(0, 1)
        igd = np.array([[]], dtype=np.int32, order='F').reshape(0, 1)
        gn = np.array([], dtype=np.float64, order='F')
        gd = np.array([], dtype=np.float64, order='F')
        d = np.array([[]], dtype=np.float64, order='F').reshape(0, 1)

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == 0

    def test_zero_m(self):
        """Test m=0 quick return."""
        p, m, md = 1, 0, 2

        ign = np.array([[]], dtype=np.int32, order='F').reshape(1, 0)
        igd = np.array([[]], dtype=np.int32, order='F').reshape(1, 0)
        gn = np.array([], dtype=np.float64, order='F')
        gd = np.array([], dtype=np.float64, order='F')
        d = np.array([[]], dtype=np.float64, order='F').reshape(1, 0)

        ign_out, gn_out, info = tb04bw('I', p, m, md, ign, igd, gn, gd, d)

        assert info == 0
