"""
Tests for SB02ND: Optimal state feedback matrix for optimal control problem.

SB02ND computes the optimal feedback matrix F:
  F = (R + B'XB)^(-1) (B'XA + L')  [discrete-time]
  F = R^(-1) (B'X + L')            [continuous-time]

Test data sources:
- HTML doc example: N=2, M=1, discrete-time with R=I, X=I
- Synthetic data with mathematical property validation
"""

import numpy as np
import pytest


"""Basic functionality tests from HTML documentation."""

def test_discrete_time_html_example():
    """
    Validate discrete-time case using SLICOT HTML doc example.

    From HTML doc: N=2, M=1, P=3, DICO='D', FACT='N', JOBL='Z', UPLO='U'
    A = [[2, -1], [1, 0]], B = [[1], [0]], R = [[0]] (but after SB02OD R is modified)
    X = [[1, 0], [0, 1]] (identity from SB02OD)
    Expected F = [[2.0, -1.0]]

    This tests the formula: F = (R + B'XB)^(-1) * B'XA
    With X=I, B=[1;0]: B'XB = 1, B'XA = [2, -1]
    So F = 1/(R+1) * [2, -1]
    With R=0: F = [2, -1]
    """
    from ctrlsys import sb02nd

    n, m = 2, 1
    p = 0

    a = np.array([[2.0, -1.0],
                  [1.0, 0.0]], order='F', dtype=float)
    b = np.array([[1.0],
                  [0.0]], order='F', dtype=float)
    r = np.zeros((m, m), order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)
    rnorm = 0.0

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'D', 'N', 'U', 'Z', n, m, p, a, b, r, ipiv, l, x, rnorm
    )

    assert info == 0, f"SB02ND failed with info={info}"

    f_expected = np.array([[2.0, -1.0]], order='F', dtype=float)
    np.testing.assert_allclose(f, f_expected, rtol=1e-4, atol=1e-4)

def test_continuous_time_basic():
    """
    Validate continuous-time case: F = R^(-1) * B'X

    Random seed: 42 (for reproducibility)

    Uses simple 2x2 system with known solution:
    R = I, X = I, B = [[1, 0], [0, 1]]
    Then F = I^(-1) * I * I = I
    """
    from ctrlsys import sb02nd

    np.random.seed(42)
    n, m = 2, 2
    p = 0

    a = np.eye(n, order='F', dtype=float)
    b = np.eye(n, m, order='F', dtype=float)
    r = np.eye(m, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)
    rnorm = 0.0

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'C', 'N', 'U', 'Z', n, m, p, a, b, r, ipiv, l, x, rnorm
    )

    assert info == 0, f"SB02ND failed with info={info}"

    f_expected = np.eye(m, n, order='F', dtype=float)
    np.testing.assert_allclose(f, f_expected, rtol=1e-14, atol=1e-14)


"""Mathematical property validation tests."""

def test_discrete_feedback_formula():
    """
    Validate discrete-time feedback formula: (R + B'XB) * F = B'XA + L'

    Random seed: 123 (for reproducibility)

    Tests that the linear system (R + B'XB) * F = B'XA + L' holds.
    """
    from ctrlsys import sb02nd

    np.random.seed(123)
    n, m = 3, 2
    p = 0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    x_rand = np.random.randn(n, n)
    x = (x_rand @ x_rand.T).astype(float, order='F')

    r_rand = np.random.randn(m, m)
    r = (np.eye(m) + r_rand @ r_rand.T).astype(float, order='F')
    r_copy = r.copy(order='F')

    l = np.random.randn(n, m).astype(float, order='F')
    ipiv = np.zeros(m, dtype=np.int32)
    rnorm = 0.0

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'D', 'N', 'U', 'N', n, m, p, a, b, r.copy(order='F'),
        ipiv, l.copy(order='F'), x.copy(order='F'), rnorm
    )

    assert info == 0, f"SB02ND failed with info={info}"

    btxb = b.T @ x @ b
    coef = r_copy + btxb
    rhs = b.T @ x @ a + l.T

    lhs = coef @ f
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)

def test_continuous_feedback_formula():
    """
    Validate continuous-time feedback formula: R * F = B'X + L'

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb02nd

    np.random.seed(456)
    n, m = 4, 2
    p = 0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    x_rand = np.random.randn(n, n)
    x = (x_rand @ x_rand.T).astype(float, order='F')

    r_rand = np.random.randn(m, m)
    r = (np.eye(m) + r_rand @ r_rand.T).astype(float, order='F')
    r_copy = r.copy(order='F')

    l = np.random.randn(n, m).astype(float, order='F')
    ipiv = np.zeros(m, dtype=np.int32)
    rnorm = 0.0

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'C', 'N', 'U', 'N', n, m, p, a, b, r.copy(order='F'),
        ipiv, l.copy(order='F'), x.copy(order='F'), rnorm
    )

    assert info == 0, f"SB02ND failed with info={info}"

    rhs = b.T @ x + l.T
    lhs = r_copy @ f

    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)

def test_zero_cross_term_l():
    """
    Validate JOBL='Z' mode: L is not used (treated as zero).

    Random seed: 789 (for reproducibility)

    For discrete-time with L=0: F = (R + B'XB)^(-1) * B'XA
    """
    from ctrlsys import sb02nd

    np.random.seed(789)
    n, m = 2, 2
    p = 0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    x_rand = np.random.randn(n, n)
    x = (x_rand @ x_rand.T + np.eye(n)).astype(float, order='F')

    r_rand = np.random.randn(m, m)
    r = (np.eye(m) + r_rand @ r_rand.T).astype(float, order='F')
    r_copy = r.copy(order='F')

    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)
    rnorm = 0.0

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'D', 'N', 'U', 'Z', n, m, p, a, b, r.copy(order='F'),
        ipiv, l, x.copy(order='F'), rnorm
    )

    assert info == 0, f"SB02ND failed with info={info}"

    coef = r_copy + b.T @ x @ b
    rhs = b.T @ x @ a
    f_expected = np.linalg.solve(coef, rhs)

    np.testing.assert_allclose(f, f_expected, rtol=1e-12, atol=1e-12)


"""Edge case and boundary condition tests."""

def test_single_input():
    """
    Test with single input (M=1).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import sb02nd

    np.random.seed(111)
    n, m = 3, 1
    p = 0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    x_rand = np.random.randn(n, n)
    x = (x_rand @ x_rand.T + np.eye(n)).astype(float, order='F')

    r = np.array([[2.0]], order='F', dtype=float)
    r_copy = r.copy(order='F')

    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)
    rnorm = 0.0

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'D', 'N', 'U', 'Z', n, m, p, a, b, r, ipiv, l, x.copy(order='F'), rnorm
    )

    assert info == 0, f"SB02ND failed with info={info}"

    coef = r_copy + b.T @ x @ b
    rhs = b.T @ x @ a
    f_expected = np.linalg.solve(coef, rhs)

    np.testing.assert_allclose(f, f_expected, rtol=1e-12, atol=1e-12)

def test_lower_triangular_uplo_l():
    """
    Test with UPLO='L' (lower triangular storage).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import sb02nd

    np.random.seed(222)
    n, m = 2, 2
    p = 0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    x_rand = np.random.randn(n, n)
    x = (x_rand @ x_rand.T + np.eye(n)).astype(float, order='F')

    r_rand = np.random.randn(m, m)
    r_full = np.eye(m) + r_rand @ r_rand.T
    r = np.tril(r_full).astype(float, order='F')
    r_copy = r_full.copy(order='F')

    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)
    rnorm = 0.0

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'D', 'N', 'L', 'Z', n, m, p, a, b, r, ipiv, l, x.copy(order='F'), rnorm
    )

    assert info == 0, f"SB02ND failed with info={info}"

    coef = r_copy + b.T @ x @ b
    rhs = b.T @ x @ a
    f_expected = np.linalg.solve(coef, rhs)

    np.testing.assert_allclose(f, f_expected, rtol=1e-12, atol=1e-12)


"""Tests for pre-factored R matrix (FACT='C', 'D')."""

def test_cholesky_factored_r_continuous():
    """
    Test with Cholesky-factored R (FACT='C') for continuous-time.

    Random seed: 333 (for reproducibility)

    R given as Cholesky factor: R_chol where R = R_chol' * R_chol
    """
    from ctrlsys import sb02nd

    np.random.seed(333)
    n, m = 3, 2
    p = 0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    x_rand = np.random.randn(n, n)
    x = (x_rand @ x_rand.T + np.eye(n)).astype(float, order='F')

    r_rand = np.random.randn(m, m)
    r_full = np.eye(m) + r_rand @ r_rand.T
    r_chol = np.linalg.cholesky(r_full).T
    r = np.triu(r_chol).astype(float, order='F')

    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)
    rnorm = 0.0

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'C', 'C', 'U', 'Z', n, m, p, a, b, r, ipiv, l, x.copy(order='F'), rnorm
    )

    assert info == 0, f"SB02ND failed with info={info}"

    rhs = b.T @ x
    f_expected = np.linalg.solve(r_full, rhs)

    np.testing.assert_allclose(f, f_expected, rtol=1e-12, atol=1e-12)

def test_d_factor_discrete():
    """
    Test with D factor (FACT='D') for discrete-time: R = D'D.

    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import sb02nd

    np.random.seed(444)
    n, m = 2, 2
    p = 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    x_rand = np.random.randn(n, n)
    x = (x_rand @ x_rand.T + np.eye(n)).astype(float, order='F')

    d = np.random.randn(p, m).astype(float, order='F')
    r_full = d.T @ d

    r = d.copy(order='F')

    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)
    rnorm = 0.0

    # Save originals - sb02nd modifies b in-place (DTRMM for B'XB update)
    b_orig = b.copy()
    x_orig = x.copy()

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'D', 'D', 'U', 'Z', n, m, p, a, b, r, ipiv, l, x.copy(order='F'), rnorm
    )

    assert info == 0, f"SB02ND failed with info={info}"

    coef = r_full + b_orig.T @ x_orig @ b_orig
    rhs = b_orig.T @ x_orig @ a
    f_expected = np.linalg.solve(coef, rhs)

    np.testing.assert_allclose(f, f_expected, rtol=1e-10, atol=1e-10)


"""Error handling tests."""

def test_invalid_dico():
    """Test invalid DICO parameter."""
    from ctrlsys import sb02nd

    n, m = 2, 1
    p = 0

    a = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    r = np.eye(m, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)

    with pytest.raises(ValueError):
        sb02nd('X', 'N', 'U', 'Z', n, m, p, a, b, r, ipiv, l, x, 0.0)

def test_invalid_uplo():
    """Test invalid UPLO parameter."""
    from ctrlsys import sb02nd

    n, m = 2, 1
    p = 0

    a = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    r = np.eye(m, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)

    with pytest.raises(ValueError):
        sb02nd('D', 'N', 'X', 'Z', n, m, p, a, b, r, ipiv, l, x, 0.0)

def test_singular_coefficient_matrix():
    """
    Test detection of singular coefficient matrix (info = M+1).

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import sb02nd

    np.random.seed(555)
    n, m = 2, 2
    p = 0

    a = np.eye(n, order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)
    r = np.zeros((m, m), order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'D', 'N', 'U', 'Z', n, m, p, a, b, r, ipiv, l, x, 0.0
    )

    assert info > 0, f"Expected positive info for singular matrix, got {info}"


"""Special case tests."""

def test_identity_matrices():
    """
    Test with identity matrices for all inputs.

    With A=I, B=I, R=I, X=I, L=0 (discrete):
    F = (I + I)^(-1) * I = 0.5 * I
    """
    from ctrlsys import sb02nd

    n, m = 2, 2
    p = 0

    a = np.eye(n, order='F', dtype=float)
    b = np.eye(n, m, order='F', dtype=float)
    r = np.eye(m, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'D', 'N', 'U', 'Z', n, m, p, a, b, r, ipiv, l, x, 0.0
    )

    assert info == 0, f"SB02ND failed with info={info}"

    f_expected = 0.5 * np.eye(m, n, order='F', dtype=float)
    np.testing.assert_allclose(f, f_expected, rtol=1e-14, atol=1e-14)

def test_larger_system():
    """
    Test with larger system (N=5, M=3).

    Random seed: 666 (for reproducibility)
    """
    from ctrlsys import sb02nd

    np.random.seed(666)
    n, m = 5, 3
    p = 0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    x_rand = np.random.randn(n, n)
    x = (x_rand @ x_rand.T + np.eye(n)).astype(float, order='F')

    r_rand = np.random.randn(m, m)
    r = (np.eye(m) + r_rand @ r_rand.T).astype(float, order='F')
    r_copy = r.copy(order='F')

    l = np.zeros((n, m), order='F', dtype=float)
    ipiv = np.zeros(m, dtype=np.int32)

    f, r_out, x_out, oufact, rcond, info = sb02nd(
        'D', 'N', 'U', 'Z', n, m, p, a, b, r, ipiv, l, x.copy(order='F'), 0.0
    )

    assert info == 0, f"SB02ND failed with info={info}"

    coef = r_copy + b.T @ x @ b
    rhs = b.T @ x @ a
    f_expected = np.linalg.solve(coef, rhs)

    np.testing.assert_allclose(f, f_expected, rtol=1e-11, atol=1e-11)
