"""
Tests for MC01QD - Polynomial division: quotient and remainder.

Computes Q(x) and R(x) such that A(x) = B(x) * Q(x) + R(x)
where degree(R) < degree(B).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mc01qd_basic_html_example():
    """
    Test using SLICOT HTML doc example.

    A(x) = 2 + 2*x - x^2 + 2*x^3 + x^4  (degree 4)
    B(x) = 1 - x + x^2  (degree 2)

    Expected results from HTML doc:
    Q(x) = 1 + 3*x + x^2  (coefficients: 1.0, 3.0, 1.0)
    R(x) = 1 + 0*x  (coefficients: 1.0, 0.0)

    RQ array layout: [R(x) coefficients | Q(x) coefficients]
    = [1.0, 0.0, 1.0, 3.0, 1.0]
    """
    from ctrlsys import mc01qd

    a = np.array([2.0, 2.0, -1.0, 2.0, 1.0], order='F', dtype=float)
    b = np.array([1.0, -1.0, 1.0], order='F', dtype=float)

    rq, db_out, iwarn, info = mc01qd(a, b)

    assert info == 0
    assert iwarn == 0
    assert db_out == 2

    expected_rq = np.array([1.0, 0.0, 1.0, 3.0, 1.0])
    assert_allclose(rq, expected_rq, rtol=1e-14)


def test_mc01qd_polynomial_division_property():
    """
    Validate polynomial division identity: A(x) = B(x) * Q(x) + R(x).

    Uses numpy.polynomial.polynomial to verify the relationship holds.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc01qd

    np.random.seed(42)

    a = np.array([6.0, 11.0, 6.0, 1.0], order='F', dtype=float)
    b = np.array([2.0, 3.0, 1.0], order='F', dtype=float)

    rq, db_out, iwarn, info = mc01qd(a, b)

    assert info == 0
    assert iwarn == 0

    da = len(a) - 1
    db = db_out

    r = rq[:db]
    q = rq[db:da+1]

    bq = np.polynomial.polynomial.polymul(b, q)
    bq_plus_r = np.polynomial.polynomial.polyadd(bq, r)

    bq_plus_r_padded = np.zeros(len(a))
    bq_plus_r_padded[:len(bq_plus_r)] = bq_plus_r

    assert_allclose(bq_plus_r_padded, a, rtol=1e-14)


def test_mc01qd_degree_a_less_than_b():
    """
    Test when degree(A) < degree(B): Q(x) = 0, R(x) = A(x).

    A(x) = 1 + 2*x (degree 1)
    B(x) = 1 + x + x^2 (degree 2)
    """
    from ctrlsys import mc01qd

    a = np.array([1.0, 2.0], order='F', dtype=float)
    b = np.array([1.0, 1.0, 1.0], order='F', dtype=float)

    rq, db_out, iwarn, info = mc01qd(a, b)

    assert info == 0
    assert iwarn == 0

    expected_rq = np.array([1.0, 2.0])
    assert_allclose(rq, expected_rq, rtol=1e-14)


def test_mc01qd_constant_divisor():
    """
    Test division by constant polynomial.

    A(x) = 2 + 4*x + 6*x^2  (degree 2)
    B(x) = 2  (degree 0, constant)

    Q(x) = A(x)/2 = 1 + 2*x + 3*x^2
    R(x) = 0 (empty, since degree < 0)
    """
    from ctrlsys import mc01qd

    a = np.array([2.0, 4.0, 6.0], order='F', dtype=float)
    b = np.array([2.0], order='F', dtype=float)

    rq, db_out, iwarn, info = mc01qd(a, b)

    assert info == 0
    assert iwarn == 0
    assert db_out == 0

    expected_q = np.array([1.0, 2.0, 3.0])
    assert_allclose(rq, expected_q, rtol=1e-14)


def test_mc01qd_leading_zeros_in_b():
    """
    Test when B has leading zeros (degree reduction triggers IWARN).

    B(x) = 1 + x + 0*x^2 (leading coeff is 0, actual degree is 1)
    A(x) = 1 + 3*x + 2*x^2

    IWARN should be 1 (degree reduced from 2 to 1).
    """
    from ctrlsys import mc01qd

    a = np.array([1.0, 3.0, 2.0], order='F', dtype=float)
    b = np.array([1.0, 1.0, 0.0], order='F', dtype=float)

    rq, db_out, iwarn, info = mc01qd(a, b)

    assert info == 0
    assert iwarn == 1
    assert db_out == 1


def test_mc01qd_zero_divisor_error():
    """
    Test error when B(x) is the zero polynomial.

    If all B coefficients are zero, INFO = 1.
    """
    from ctrlsys import mc01qd

    a = np.array([1.0, 2.0, 3.0], order='F', dtype=float)
    b = np.array([0.0, 0.0], order='F', dtype=float)

    rq, db_out, iwarn, info = mc01qd(a, b)

    assert info == 1


def test_mc01qd_zero_numerator():
    """
    Test with A(x) = 0 (zero polynomial, DA = -1).

    Q(x) = 0 and R(x) = 0.
    """
    from ctrlsys import mc01qd

    a = np.array([], order='F', dtype=float)
    b = np.array([1.0, 1.0], order='F', dtype=float)

    rq, db_out, iwarn, info = mc01qd(a, b)

    assert info == 0
    assert len(rq) == 0


def test_mc01qd_exact_division():
    """
    Test when A(x) is exactly divisible by B(x) (R(x) = 0).

    A(x) = x^3 - 1 = (x - 1)(x^2 + x + 1)
    B(x) = x - 1

    Q(x) = x^2 + x + 1 (coefficients: 1, 1, 1)
    R(x) = 0 (coefficient: 0)
    """
    from ctrlsys import mc01qd

    a = np.array([-1.0, 0.0, 0.0, 1.0], order='F', dtype=float)
    b = np.array([-1.0, 1.0], order='F', dtype=float)

    rq, db_out, iwarn, info = mc01qd(a, b)

    assert info == 0
    assert iwarn == 0
    assert db_out == 1

    r = rq[:db_out]
    q = rq[db_out:]

    expected_r = np.array([0.0])
    expected_q = np.array([1.0, 1.0, 1.0])

    assert_allclose(r, expected_r, atol=1e-14)
    assert_allclose(q, expected_q, rtol=1e-14)


def test_mc01qd_random_division_validation():
    """
    Validate polynomial division with random polynomials against NumPy.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mc01qd

    np.random.seed(123)

    da = 5
    db = 2

    a = np.random.randn(da + 1).astype(float, order='F')
    b = np.random.randn(db + 1).astype(float, order='F')
    while abs(b[-1]) < 1e-10:
        b[-1] = np.random.randn()

    rq, db_out, iwarn, info = mc01qd(a, b)

    assert info == 0

    q_numpy, r_numpy = np.polynomial.polynomial.polydiv(a, b)

    r = rq[:db_out]
    q = rq[db_out:da+1]

    assert_allclose(q, q_numpy, rtol=1e-13, atol=1e-14)

    r_numpy_padded = np.zeros(db_out)
    r_numpy_padded[:len(r_numpy)] = r_numpy
    assert_allclose(r, r_numpy_padded, rtol=1e-13, atol=1e-14)
