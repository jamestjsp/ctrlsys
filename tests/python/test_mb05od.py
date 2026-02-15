"""
Tests for MB05OD: Matrix exponential with accuracy estimate.

Computes exp(A*delta) where A is a real N-by-N matrix and delta is scalar.
Uses diagonal Pade approximation with scaling/squaring.
Returns minimal and 95%-confidence accurate digit counts.
"""
import numpy as np
import pytest
from ctrlsys import mb05od


def test_html_doc_example():
    """
    Validate 3x3 example from SLICOT HTML documentation.

    Input: 3x3 matrix A, delta=1.0, ndiag=9, balanc='S'
    Expected output from MB05OD.html.
    """
    n = 3
    delta = 1.0
    ndiag = 9
    balanc = 'S'

    a = np.array([
        [2.0, 1.0, 1.0],
        [0.0, 3.0, 2.0],
        [1.0, 0.0, 4.0],
    ], order='F', dtype=float)

    exp_a_expected = np.array([
        [22.5984, 17.2073, 53.8144],
        [24.4047, 27.6033, 83.2241],
        [29.4097, 12.2024, 81.4177],
    ], order='F', dtype=float)

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a, exp_a_expected, rtol=1e-3, atol=1e-3)
    assert mdig >= 10
    assert idig >= 10


def test_delta_zero():
    """
    Test with delta=0.

    exp(A*0) = I (identity matrix).
    """
    n = 3
    delta = 0.0
    ndiag = 9
    balanc = 'N'

    np.random.seed(42)
    a = np.random.randn(n, n).astype(float, order='F')

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a, np.eye(n), rtol=1e-14)


def test_n_equals_zero():
    """Test with n=0 (empty system)."""
    n = 0
    delta = 1.0
    ndiag = 9
    balanc = 'N'
    a = np.array([], order='F', dtype=float).reshape(0, 0)

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == 0
    assert exp_a.shape == (0, 0)


def test_n_equals_one():
    """
    Test scalar case (n=1).

    For A = [a], exp(A*delta) = exp(a*delta).
    """
    n = 1
    delta = 0.5
    ndiag = 9
    balanc = 'N'
    a_val = 2.0
    a = np.array([[a_val]], order='F', dtype=float)

    exp_expected = np.exp(a_val * delta)

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a[0, 0], exp_expected, rtol=1e-14)


def test_identity_matrix():
    """
    Test with identity matrix.

    exp(I*delta) = exp(delta) * I.
    """
    n = 3
    delta = 0.5
    ndiag = 9
    balanc = 'N'

    a = np.eye(n, order='F', dtype=float)
    exp_expected = np.exp(delta) * np.eye(n)

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a, exp_expected, rtol=1e-12)


def test_diagonal_matrix():
    """
    Test with diagonal matrix (exact closed form).

    For diagonal A = diag(a1, a2, ...):
    exp(A*delta) = diag(exp(a1*delta), exp(a2*delta), ...)

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    delta = 0.25
    ndiag = 9
    balanc = 'N'

    diag_vals = np.random.randn(n) * 2
    a = np.diag(diag_vals).astype(float, order='F')

    exp_expected = np.diag(np.exp(diag_vals * delta))

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a, exp_expected, rtol=1e-12)


def test_nilpotent_matrix():
    """
    Test with nilpotent matrix (A^n = 0).

    For strictly upper triangular matrix:
    exp(A*delta) can be computed exactly via Taylor series.
    """
    n = 3
    delta = 1.0
    ndiag = 9
    balanc = 'N'

    a = np.array([
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], order='F', dtype=float)

    exp_expected = np.array([
        [1.0, 1.0, 2.5],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ], order='F', dtype=float)

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a, exp_expected, rtol=1e-13)


def test_balancing_option():
    """
    Test balancing option ('S') vs no balancing ('N').

    Both should give similar results, balancing may improve accuracy.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4
    delta = 0.3
    ndiag = 9

    a = np.random.randn(n, n).astype(float, order='F')

    a_copy1 = a.copy(order='F')
    a_copy2 = a.copy(order='F')

    exp_a_n, mdig_n, idig_n, iwarn_n, info_n = mb05od('N', n, ndiag, delta, a_copy1)
    exp_a_s, mdig_s, idig_s, iwarn_s, info_s = mb05od('S', n, ndiag, delta, a_copy2)

    assert info_n == 0
    assert info_s == 0
    np.testing.assert_allclose(exp_a_n, exp_a_s, rtol=1e-10)


def test_exp_product_property():
    """
    Test exp(A*(t1+t2)) = exp(A*t1) * exp(A*t2).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3
    delta1 = 0.2
    delta2 = 0.3
    delta_sum = delta1 + delta2
    ndiag = 9
    balanc = 'N'

    a = np.random.randn(n, n).astype(float, order='F')

    a_copy1 = a.copy(order='F')
    a_copy2 = a.copy(order='F')
    a_copy3 = a.copy(order='F')

    exp_t1, _, _, _, info1 = mb05od(balanc, n, ndiag, delta1, a_copy1)
    exp_t2, _, _, _, info2 = mb05od(balanc, n, ndiag, delta2, a_copy2)
    exp_sum, _, _, _, info3 = mb05od(balanc, n, ndiag, delta_sum, a_copy3)

    assert info1 == 0
    assert info2 == 0
    assert info3 == 0

    exp_product = exp_t1 @ exp_t2
    np.testing.assert_allclose(exp_product, exp_sum, rtol=1e-12)


def test_determinant_property():
    """
    Test det(exp(A*delta)) = exp(trace(A)*delta).

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 4
    delta = 0.5
    ndiag = 9
    balanc = 'N'

    a = np.random.randn(n, n).astype(float, order='F')
    trace_a = np.trace(a)

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == 0

    det_exp = np.linalg.det(exp_a)
    exp_trace = np.exp(trace_a * delta)

    np.testing.assert_allclose(det_exp, exp_trace, rtol=1e-10)


def test_negative_n_error():
    """Test that negative n returns info=-2."""
    delta = 1.0
    ndiag = 9
    balanc = 'N'
    a = np.array([[1.0]], order='F', dtype=float)

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, -1, ndiag, delta, a)

    assert info == -2


def test_invalid_ndiag_error():
    """Test that ndiag < 1 returns info=-3."""
    n = 2
    delta = 1.0
    ndiag = 0
    balanc = 'N'
    a = np.eye(2, order='F', dtype=float)

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == -3


def test_invalid_balanc_error():
    """Test that invalid balanc returns info=-1."""
    n = 2
    delta = 1.0
    ndiag = 9
    balanc = 'X'
    a = np.eye(2, order='F', dtype=float)

    exp_a, mdig, idig, iwarn, info = mb05od(balanc, n, ndiag, delta, a)

    assert info == -1
