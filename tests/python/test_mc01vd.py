"""
Tests for MC01VD: Roots of a quadratic equation with real coefficients.

MC01VD computes roots of: a*x^2 + b*x + c = 0
"""

import numpy as np
import pytest

from ctrlsys import mc01vd


class TestMC01VDBasic:
    """Basic functionality tests from HTML documentation."""

    def test_complex_roots_from_html_doc(self):
        """
        Test case from SLICOT HTML documentation.

        Equation: 0.5*x^2 - 1.0*x + 2.0 = 0
        Expected roots: 1.0 + 1.7321j and 1.0 - 1.7321j
        """
        a = 0.5
        b = -1.0
        c = 2.0

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        assert info == 0
        # z1 is largest in magnitude
        np.testing.assert_allclose(z1re, 1.0, rtol=1e-4)
        np.testing.assert_allclose(z1im, np.sqrt(3.0), rtol=1e-4)
        # z2 is complex conjugate
        np.testing.assert_allclose(z2re, 1.0, rtol=1e-4)
        np.testing.assert_allclose(z2im, -np.sqrt(3.0), rtol=1e-4)


class TestMC01VDRealRoots:
    """Tests for quadratic equations with real roots."""

    def test_distinct_real_roots(self):
        """
        Equation: x^2 - 5x + 6 = 0
        Roots: x = 3, x = 2

        Random seed: 42 (for reproducibility - not used, deterministic test)
        """
        a = 1.0
        b = -5.0
        c = 6.0

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        assert info == 0
        # Imaginary parts should be zero for real roots
        np.testing.assert_allclose(z1im, 0.0, atol=1e-14)
        np.testing.assert_allclose(z2im, 0.0, atol=1e-14)
        # z1 is largest in magnitude (3), z2 is smallest (2)
        np.testing.assert_allclose(z1re, 3.0, rtol=1e-14)
        np.testing.assert_allclose(z2re, 2.0, rtol=1e-14)

    def test_equal_real_roots(self):
        """
        Equation: x^2 - 4x + 4 = 0
        Roots: x = 2 (double root)
        """
        a = 1.0
        b = -4.0
        c = 4.0

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        assert info == 0
        np.testing.assert_allclose(z1im, 0.0, atol=1e-14)
        np.testing.assert_allclose(z2im, 0.0, atol=1e-14)
        np.testing.assert_allclose(z1re, 2.0, rtol=1e-14)
        np.testing.assert_allclose(z2re, 2.0, rtol=1e-14)

    def test_opposite_sign_roots(self):
        """
        Equation: x^2 - 4 = 0
        Roots: x = 2, x = -2
        """
        a = 1.0
        b = 0.0
        c = -4.0

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        assert info == 0
        np.testing.assert_allclose(z1im, 0.0, atol=1e-14)
        np.testing.assert_allclose(z2im, 0.0, atol=1e-14)
        # Both have same magnitude, order may vary
        roots = sorted([z1re, z2re])
        np.testing.assert_allclose(roots, [-2.0, 2.0], rtol=1e-14)


class TestMC01VDComplexRoots:
    """Tests for quadratic equations with complex conjugate roots."""

    def test_pure_imaginary_roots(self):
        """
        Equation: x^2 + 4 = 0
        Roots: x = 2j, x = -2j
        """
        a = 1.0
        b = 0.0
        c = 4.0

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        assert info == 0
        # Real parts should be zero
        np.testing.assert_allclose(z1re, 0.0, atol=1e-14)
        np.testing.assert_allclose(z2re, 0.0, atol=1e-14)
        # Imaginary parts are +/-2
        np.testing.assert_allclose(abs(z1im), 2.0, rtol=1e-14)
        np.testing.assert_allclose(abs(z2im), 2.0, rtol=1e-14)
        # Conjugate pair
        np.testing.assert_allclose(z1im, -z2im, rtol=1e-14)


class TestMC01VDSpecialCases:
    """Tests for special cases and edge conditions."""

    def test_linear_equation_a_zero(self):
        """
        When a=0: equation becomes bx + c = 0, root is -c/b.
        INFO should be 2 (a=0 case), z1re = BIG.
        """
        a = 0.0
        b = 2.0
        c = -6.0

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        assert info == 2
        # z2 should be -c/b = 3.0
        np.testing.assert_allclose(z2re, 3.0, rtol=1e-14)
        np.testing.assert_allclose(z2im, 0.0, atol=1e-14)
        # z1re should be BIG, z1im should be 0
        np.testing.assert_allclose(z1im, 0.0, atol=1e-14)

    def test_a_and_b_zero_error(self):
        """When a=0 and b=0, INFO should be 1."""
        a = 0.0
        b = 0.0
        c = 5.0

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        assert info == 1

    def test_c_zero_one_root_zero(self):
        """
        Equation: x^2 - 3x = 0
        Roots: x = 0, x = 3
        """
        a = 1.0
        b = -3.0
        c = 0.0

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        assert info == 0
        np.testing.assert_allclose(z1im, 0.0, atol=1e-14)
        np.testing.assert_allclose(z2im, 0.0, atol=1e-14)
        # z1 is largest (3), z2 is smallest (0)
        np.testing.assert_allclose(z1re, 3.0, rtol=1e-14)
        np.testing.assert_allclose(z2re, 0.0, atol=1e-14)


class TestMC01VDMathematicalProperties:
    """Validate mathematical properties of quadratic roots."""

    def test_vieta_sum_of_roots(self):
        """
        Vieta's formulas: sum of roots = -b/a

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        a = np.random.uniform(0.5, 2.0)
        b = np.random.uniform(-5.0, 5.0)
        c = np.random.uniform(-5.0, 5.0)

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        if info == 0:
            # Sum of roots (complex case: real parts sum)
            sum_roots = complex(z1re, z1im) + complex(z2re, z2im)
            expected_sum = -b / a
            np.testing.assert_allclose(sum_roots.real, expected_sum, rtol=1e-13)
            np.testing.assert_allclose(sum_roots.imag, 0.0, atol=1e-13)

    def test_vieta_product_of_roots(self):
        """
        Vieta's formulas: product of roots = c/a

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        a = np.random.uniform(0.5, 2.0)
        b = np.random.uniform(-5.0, 5.0)
        c = np.random.uniform(-5.0, 5.0)

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        if info == 0:
            # Product of roots
            z1 = complex(z1re, z1im)
            z2 = complex(z2re, z2im)
            product = z1 * z2
            expected_product = c / a
            np.testing.assert_allclose(product.real, expected_product, rtol=1e-13)
            np.testing.assert_allclose(product.imag, 0.0, atol=1e-13)

    def test_roots_satisfy_equation(self):
        """
        Each root z should satisfy a*z^2 + b*z + c = 0.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        a = np.random.uniform(0.5, 2.0)
        b = np.random.uniform(-5.0, 5.0)
        c = np.random.uniform(-5.0, 5.0)

        z1re, z1im, z2re, z2im, info = mc01vd(a, b, c)

        if info == 0:
            z1 = complex(z1re, z1im)
            z2 = complex(z2re, z2im)

            # Evaluate polynomial at each root
            p_z1 = a * z1 * z1 + b * z1 + c
            p_z2 = a * z2 * z2 + b * z2 + c

            np.testing.assert_allclose(abs(p_z1), 0.0, atol=1e-12)
            np.testing.assert_allclose(abs(p_z2), 0.0, atol=1e-12)
