"""
Tests for SB02MT - Riccati preprocessing routine.

Computes:
    G = B*R^(-1)*B'
    A_bar = A - B*R^(-1)*L'
    Q_bar = Q - L*R^(-1)*L'

for converting optimal problems with coupling weighting terms to standard form.
"""

import numpy as np
import pytest


def test_sb02mt_basic_cholesky():
    """
    Test SB02MT with positive definite R (Cholesky factorization path).

    Computes G = B*R^(-1)*B' for positive definite R.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n, m = 4, 2

    B = np.random.randn(n, m).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_inv = np.linalg.inv(R)
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mt

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mt(
        'G', 'Z', 'N', 'U',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == 0, f"sb02mt failed with info={info}"
    assert oufact == 1, f"Expected Cholesky factorization (oufact=1), got {oufact}"

    G_upper = np.triu(G_out)
    G_expected_upper = np.triu(G_expected)
    np.testing.assert_allclose(G_upper, G_expected_upper, rtol=1e-13, atol=1e-14)


def test_sb02mt_with_coupling_matrix_L():
    """
    Test SB02MT with nonzero coupling matrix L.

    Computes:
        A_bar = A - B*R^(-1)*L'
        Q_bar = Q - L*R^(-1)*L'
        G = B*R^(-1)*B'

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    L = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_inv = np.linalg.inv(R)
    A_bar_expected = A - B @ R_inv @ L.T
    Q_bar_expected = Q - L @ R_inv @ L.T
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mt

    A_copy = A.copy()
    B_copy = B.copy()
    Q_copy = Q.copy()
    R_copy = R.copy()
    L_copy = L.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    A_out, B_out, Q_out, L_out, G_out, oufact, info = sb02mt(
        'G', 'N', 'N', 'U',
        n, m, A_copy, B_copy, Q_copy, R_copy, L_copy, G
    )

    assert info == 0, f"sb02mt failed with info={info}"
    assert oufact == 1, f"Expected Cholesky factorization (oufact=1), got {oufact}"

    np.testing.assert_allclose(A_out, A_bar_expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.triu(Q_out), np.triu(Q_bar_expected), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.triu(G_out), np.triu(G_expected), rtol=1e-13, atol=1e-14)


def test_sb02mt_lower_triangle():
    """
    Test SB02MT with lower triangle storage (UPLO='L').

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n, m = 3, 2

    B = np.random.randn(n, m).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_inv = np.linalg.inv(R)
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mt

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mt(
        'G', 'Z', 'N', 'L',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == 0
    assert oufact == 1

    G_lower = np.tril(G_out)
    G_expected_lower = np.tril(G_expected)
    np.testing.assert_allclose(G_lower, G_expected_lower, rtol=1e-13, atol=1e-14)


def test_sb02mt_indefinite_R():
    """
    Test SB02MT with indefinite R (UdU' factorization path).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n, m = 3, 2

    B = np.random.randn(n, m).astype(float, order='F')

    R = np.array([[2.0, 0.5],
                  [0.5, -1.0]], dtype=float, order='F')

    R_inv = np.linalg.inv(R)
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mt

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mt(
        'G', 'Z', 'N', 'U',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == 0, f"sb02mt failed with info={info}"
    assert oufact == 2, f"Expected UdU' factorization (oufact=2), got {oufact}"

    G_upper = np.triu(G_out)
    G_expected_upper = np.triu(G_expected)
    np.testing.assert_allclose(G_upper, G_expected_upper, rtol=1e-12, atol=1e-13)


def test_sb02mt_precomputed_cholesky():
    """
    Test SB02MT with precomputed Cholesky factor (FACT='C').

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    n, m = 3, 2

    B = np.random.randn(n, m).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_chol = np.linalg.cholesky(R).T.astype(float, order='F')

    R_inv = np.linalg.inv(R)
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mt

    R_chol_copy = R_chol.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mt(
        'G', 'Z', 'C', 'U',
        n, m, None, B_copy, None, R_chol_copy, None, G
    )

    assert info == 0
    assert oufact == 1

    G_upper = np.triu(G_out)
    G_expected_upper = np.triu(G_expected)
    np.testing.assert_allclose(G_upper, G_expected_upper, rtol=1e-13, atol=1e-14)


def test_sb02mt_no_G_computation():
    """
    Test SB02MT with JOBG='N' (no G computation).

    Only computes A_bar and Q_bar.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    L = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_inv = np.linalg.inv(R)
    A_bar_expected = A - B @ R_inv @ L.T
    Q_bar_expected = Q - L @ R_inv @ L.T

    from ctrlsys import sb02mt

    A_copy = A.copy()
    B_copy = B.copy()
    Q_copy = Q.copy()
    R_copy = R.copy()
    L_copy = L.copy()

    A_out, B_out, Q_out, L_out, oufact, info = sb02mt(
        'N', 'N', 'N', 'U',
        n, m, A_copy, B_copy, Q_copy, R_copy, L_copy, None
    )

    assert info == 0
    assert oufact == 1

    np.testing.assert_allclose(A_out, A_bar_expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.triu(Q_out), np.triu(Q_bar_expected), rtol=1e-13, atol=1e-14)


def test_sb02mt_zero_M():
    """
    Test SB02MT with M=0 (quick return).
    """
    n, m = 3, 0

    B = np.zeros((n, 1), dtype=float, order='F')
    R = np.zeros((1, 1), dtype=float, order='F')
    G = np.zeros((n, n), dtype=float, order='F')

    from ctrlsys import sb02mt

    G_out, oufact, info = sb02mt(
        'G', 'Z', 'N', 'U',
        n, m, None, B, None, R, None, G
    )

    assert info == 0
    assert oufact == 0
    np.testing.assert_allclose(G_out, np.zeros((n, n)), rtol=1e-14)


def test_sb02mt_symmetric_property():
    """
    Validate mathematical property: G = B*R^(-1)*B' is symmetric.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)

    n, m = 5, 3

    B = np.random.randn(n, m).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    from ctrlsys import sb02mt

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mt(
        'G', 'Z', 'N', 'U',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == 0

    G_full = np.triu(G_out) + np.triu(G_out, 1).T

    np.testing.assert_allclose(G_full, G_full.T, rtol=1e-14, atol=1e-15)


def test_sb02mt_singular_R_error():
    """
    Test SB02MT error handling for numerically singular R.
    """
    n, m = 3, 2

    B = np.ones((n, m), dtype=float, order='F')

    R = np.array([[1.0, 2.0],
                  [2.0, 4.0]], dtype=float, order='F')

    from ctrlsys import sb02mt

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mt(
        'G', 'Z', 'N', 'U',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == m + 1 or info > 0


def test_sb02mt_error_invalid_jobg():
    """
    Test SB02MT error handling: invalid JOBG parameter.
    """
    n, m = 3, 2

    B = np.zeros((n, m), dtype=float, order='F')
    R = np.eye(m, dtype=float, order='F')
    G = np.zeros((n, n), dtype=float, order='F')

    from ctrlsys import sb02mt

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02mt('X', 'Z', 'N', 'U', n, m, None, B, None, R, None, G)


def test_sb02mt_error_negative_n():
    """
    Test SB02MT error handling: negative N.
    """
    from ctrlsys import sb02mt

    B = np.zeros((1, 1), dtype=float, order='F')
    R = np.eye(1, dtype=float, order='F')
    G = np.zeros((1, 1), dtype=float, order='F')

    with pytest.raises(ValueError, match="[Nn]"):
        sb02mt('G', 'Z', 'N', 'U', -1, 1, None, B, None, R, None, G)
