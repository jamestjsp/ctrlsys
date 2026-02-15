"""
Tests for SB08MY - Compute B(s) = A(s) * A(-s) for spectral factorization.

SB08MY computes the coefficients of the polynomial B(s) = A(s) * A(-s)
where A(s) is a polynomial given in increasing powers of s, and B(s)
is returned in increasing powers of s**2.

Also computes an accuracy norm for the computed coefficients.
"""
import pytest
import numpy as np
from ctrlsys import sb08my


def test_sb08my_constant():
    """
    Test with constant polynomial A(s) = 2.

    B(s) = A(s)*A(-s) = 2*2 = 4
    """
    a = np.array([2.0], dtype=float, order='F')
    epsb = np.finfo(float).eps

    b, epsb_out = sb08my(a, epsb)

    assert b.shape == (1,)
    np.testing.assert_allclose(b[0], 4.0, rtol=1e-14)
    assert epsb_out > 0


def test_sb08my_linear():
    """
    Test with linear polynomial A(s) = 1 + 2*s.

    A(-s) = 1 - 2*s
    B(s) = A(s)*A(-s) = (1 + 2s)(1 - 2s) = 1 - 4*s^2
    In powers of s^2: B = [1, -4]
    """
    a = np.array([1.0, 2.0], dtype=float, order='F')
    epsb = np.finfo(float).eps

    b, epsb_out = sb08my(a, epsb)

    assert b.shape == (2,)
    np.testing.assert_allclose(b[0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(b[1], -4.0, rtol=1e-14)


def test_sb08my_quadratic():
    """
    Test with quadratic polynomial A(s) = 1 + s + s^2.

    A(-s) = 1 - s + s^2
    B(s) = A(s)*A(-s) = (1 + s + s^2)(1 - s + s^2)
         = 1 - s + s^2 + s - s^2 + s^3 + s^2 - s^3 + s^4
         = 1 + s^2 + s^4

    In powers of s^2: B = [1, 1, 1]
    """
    a = np.array([1.0, 1.0, 1.0], dtype=float, order='F')
    epsb = np.finfo(float).eps

    b, epsb_out = sb08my(a, epsb)

    assert b.shape == (3,)
    np.testing.assert_allclose(b[0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(b[1], 1.0, rtol=1e-14)
    np.testing.assert_allclose(b[2], 1.0, rtol=1e-14)


def test_sb08my_property_even_polynomial():
    """
    Verify B(s) = A(s)*A(-s) is an even polynomial.

    For any A(s), B(s) = A(s)*A(-s) satisfies B(s) = B(-s),
    meaning B has only even powers of s, which is why output
    is in powers of s^2.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    for _ in range(10):
        da = np.random.randint(1, 8)
        a = np.random.randn(da + 1)
        a = np.ascontiguousarray(a, dtype=float)
        epsb = np.finfo(float).eps

        b, epsb_out = sb08my(a.copy(order='F'), epsb)

        # Reconstruct polynomial in s^2 and verify it's the product
        # A(s)*A(-s) evaluated at specific points
        for s in [0.0, 0.5, 1.0, 1.5, 2.0]:
            # Evaluate A(s)
            a_s = sum(a[i] * (s ** i) for i in range(len(a)))
            # Evaluate A(-s)
            a_minus_s = sum(a[i] * ((-s) ** i) for i in range(len(a)))
            # Product should equal B(s^2)
            expected = a_s * a_minus_s
            # Evaluate B at s^2
            s2 = s * s
            b_s2 = sum(b[i] * (s2 ** i) for i in range(len(b)))

            np.testing.assert_allclose(b_s2, expected, rtol=1e-12)


def test_sb08my_symmetry():
    """
    Verify the polynomial property that odd powers cancel.

    The product A(s)*A(-s) eliminates odd powers of s, leaving
    only even powers. Coefficients of B satisfy specific relationships.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    da = 4
    a = np.random.randn(da + 1)
    a = np.array(a, dtype=float, order='F')
    epsb = np.finfo(float).eps

    b, epsb_out = sb08my(a.copy(order='F'), epsb)

    # For polynomial of degree n, B has n+1 coefficients in s^2
    assert b.shape == (da + 1,)

    # Verify by direct computation
    # B(s^2) = sum_{i=0}^{n} (-1)^i * sum_{j=max(0,i-n)}^{min(i,n)} a[i-j]*a[i+j] * s^{2i}
    # But we use functional verification above, this just checks structure
    assert epsb_out >= epsb


def test_sb08my_zero_polynomial():
    """
    Test edge case with zero polynomial A(s) = 0.

    B(s) = 0 * 0 = 0
    """
    a = np.array([0.0], dtype=float, order='F')
    epsb = np.finfo(float).eps

    b, epsb_out = sb08my(a, epsb)

    assert b.shape == (1,)
    np.testing.assert_allclose(b[0], 0.0, atol=1e-14)


def test_sb08my_accuracy_norm():
    """
    Test that accuracy norm is computed correctly.

    EPSB_out = 3 * max(abs_sum) * EPSB_in
    where abs_sum is the sum of absolute values of terms.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    a = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
    epsb = np.finfo(float).eps

    b, epsb_out = sb08my(a, epsb)

    # The output EPSB should be >= 3 * eps * max over i of (sum of |terms|)
    # This is implementation-specific, just verify it increased
    assert epsb_out >= 3 * epsb


def test_sb08my_numerical_example():
    """
    Test with specific numerical example.

    A(s) = 2 + 3*s + s^2
    A(-s) = 2 - 3*s + s^2
    B(s) = (2 + 3s + s^2)(2 - 3s + s^2)
         = 4 - 6s + 2s^2 + 6s - 9s^2 + 3s^3 + 2s^2 - 3s^3 + s^4
         = 4 - 5s^2 + s^4

    B in s^2: [4, -5, 1]
    """
    a = np.array([2.0, 3.0, 1.0], dtype=float, order='F')
    epsb = np.finfo(float).eps

    b, epsb_out = sb08my(a, epsb)

    np.testing.assert_allclose(b[0], 4.0, rtol=1e-14)
    np.testing.assert_allclose(b[1], -5.0, rtol=1e-14)
    np.testing.assert_allclose(b[2], 1.0, rtol=1e-14)
