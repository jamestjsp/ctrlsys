"""
Tests for MB05ND: Matrix exponential and integral.

Computes:
  (a) F(delta) = exp(A*delta)
  (b) H(delta) = integral from 0 to delta of exp(A*s) ds
"""
import numpy as np
import pytest
from ctrlsys import mb05nd


"""Basic functionality tests using SLICOT HTML doc example."""

def test_html_doc_example():
    """
    Validate 5x5 example from SLICOT HTML documentation.

    Input: 5x5 matrix A, delta=0.1, tol=0.0001
    Expected output from MB05ND.html with 4-digit precision.
    """
    n = 5
    delta = 0.1
    tol = 0.0001

    a = np.array([
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 6.0, 0.0, 4.0, 3.0],
        [2.0, 0.0, 7.0, 6.0, 5.0],
        [1.0, 3.0, 1.0, 8.0, 7.0],
        [2.0, 5.0, 7.0, 1.0, 9.0],
    ], order='F', dtype=float)

    ex_expected = np.array([
        [1.8391, 0.9476, 0.7920, 0.8216, 0.7811],
        [0.3359, 2.2262, 0.4013, 1.0078, 1.0957],
        [0.6335, 0.6776, 2.6933, 1.6155, 1.8502],
        [0.4804, 1.1561, 0.9110, 2.7461, 2.0854],
        [0.7105, 1.4244, 1.8835, 1.0966, 3.4134],
    ], order='F', dtype=float)

    exint_expected = np.array([
        [0.1347, 0.0352, 0.0284, 0.0272, 0.0231],
        [0.0114, 0.1477, 0.0104, 0.0369, 0.0368],
        [0.0218, 0.0178, 0.1624, 0.0580, 0.0619],
        [0.0152, 0.0385, 0.0267, 0.1660, 0.0732],
        [0.0240, 0.0503, 0.0679, 0.0317, 0.1863],
    ], order='F', dtype=float)

    ex, exint, info = mb05nd(n, delta, a, tol)

    assert info == 0
    np.testing.assert_allclose(ex, ex_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(exint, exint_expected, rtol=1e-3, atol=1e-4)


"""Edge case tests for MB05ND."""

def test_n_equals_zero():
    """Test with n=0 (empty system)."""
    n = 0
    delta = 1.0
    tol = 1e-8
    a = np.array([], order='F', dtype=float).reshape(0, 0)

    ex, exint, info = mb05nd(n, delta, a, tol)

    assert info == 0
    assert ex.shape == (0, 0)
    assert exint.shape == (0, 0)

def test_delta_equals_zero():
    """
    Test with delta=0.

    F(0) = I (identity matrix)
    H(0) = 0 (zero matrix)
    """
    n = 3
    delta = 0.0
    tol = 1e-8

    np.random.seed(42)
    a = np.random.randn(n, n).astype(float, order='F')

    ex, exint, info = mb05nd(n, delta, a, tol)

    assert info == 0
    np.testing.assert_allclose(ex, np.eye(n), rtol=1e-14)
    np.testing.assert_allclose(exint, np.zeros((n, n)), rtol=1e-14, atol=1e-14)

def test_n_equals_one():
    """
    Test scalar case (n=1).

    For A = [a], F(delta) = exp(a*delta), H(delta) = (exp(a*delta)-1)/a
    """
    n = 1
    delta = 0.5
    tol = 1e-10
    a_val = 2.0
    a = np.array([[a_val]], order='F', dtype=float)

    ex_expected = np.exp(a_val * delta)
    exint_expected = (np.exp(a_val * delta) - 1.0) / a_val

    ex, exint, info = mb05nd(n, delta, a, tol)

    assert info == 0
    np.testing.assert_allclose(ex[0, 0], ex_expected, rtol=1e-14)
    np.testing.assert_allclose(exint[0, 0], exint_expected, rtol=1e-14)

def test_n_equals_one_zero_matrix():
    """
    Test scalar case with A=0.

    For A = [0], F(delta) = 1, H(delta) = delta
    """
    n = 1
    delta = 2.5
    tol = 1e-10
    a = np.array([[0.0]], order='F', dtype=float)

    ex, exint, info = mb05nd(n, delta, a, tol)

    assert info == 0
    np.testing.assert_allclose(ex[0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(exint[0, 0], delta, rtol=1e-14)


"""Mathematical property tests for numerical correctness."""

def test_exp_integral_relationship():
    """
    Validate relationship: A*H(delta) + I = F(delta).

    For matrix exponential and its integral:
    exp(A*delta) = A * integral_0^delta exp(A*s) ds + I

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    delta = 0.3
    tol = 1e-10

    a = np.random.randn(n, n).astype(float, order='F')

    ex, exint, info = mb05nd(n, delta, a, tol)

    assert info == 0
    lhs = a @ exint + np.eye(n)
    np.testing.assert_allclose(lhs, ex, rtol=1e-10, atol=1e-12)

def test_scaling_property():
    """
    Validate F(0) = I exactly.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3
    tol = 1e-10

    a = np.random.randn(n, n).astype(float, order='F')

    ex, exint, info = mb05nd(n, 0.0, a, tol)

    assert info == 0
    np.testing.assert_allclose(ex, np.eye(n), rtol=1e-14)
    np.testing.assert_allclose(exint, np.zeros((n, n)), atol=1e-15)

def test_nilpotent_matrix():
    """
    Test with nilpotent matrix (A^n = 0).

    For strictly upper triangular matrix:
    exp(A*delta) can be computed exactly via Taylor series.

    Random seed: 789 (for reproducibility)
    """
    n = 3
    delta = 1.0
    tol = 1e-10

    a = np.array([
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], order='F', dtype=float)

    ex_expected = np.array([
        [1.0, 1.0, 2.5],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ], order='F', dtype=float)

    ex, exint, info = mb05nd(n, delta, a, tol)

    assert info == 0
    np.testing.assert_allclose(ex, ex_expected, rtol=1e-14)

    lhs = a @ exint + np.eye(n)
    np.testing.assert_allclose(lhs, ex, rtol=1e-14)

def test_diagonal_matrix():
    """
    Test with diagonal matrix (exact closed form).

    For diagonal A = diag(a1, a2, ...):
    exp(A*delta) = diag(exp(a1*delta), exp(a2*delta), ...)
    H(delta)_ii = (exp(ai*delta) - 1) / ai

    Random seed: 999 (for reproducibility)
    """
    n = 4
    delta = 0.25
    tol = 1e-10

    np.random.seed(999)
    diag_vals = np.random.randn(n) * 2

    a = np.diag(diag_vals).astype(float, order='F')

    ex_expected = np.diag(np.exp(diag_vals * delta))
    exint_expected = np.diag((np.exp(diag_vals * delta) - 1.0) / diag_vals)

    ex, exint, info = mb05nd(n, delta, a, tol)

    assert info == 0
    np.testing.assert_allclose(ex, ex_expected, rtol=1e-12)
    np.testing.assert_allclose(exint, exint_expected, rtol=1e-11)


"""Error handling tests."""

def test_negative_n():
    """Test that negative n returns info=-1."""
    delta = 1.0
    tol = 1e-8
    a = np.array([[1.0]], order='F', dtype=float)

    ex, exint, info = mb05nd(-1, delta, a, tol)

    assert info == -1

def test_large_delta_overflow():
    """
    Test that excessively large delta returns info=n+1.

    When delta * ||A||_F > sqrt(BIG), routine should return error.
    """
    n = 2
    delta = 1e200
    tol = 1e-8
    a = np.array([[1e100, 0.0], [0.0, 1e100]], order='F', dtype=float)

    ex, exint, info = mb05nd(n, delta, a, tol)

    assert info == n + 1
