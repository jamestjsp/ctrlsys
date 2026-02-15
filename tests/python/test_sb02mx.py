"""
Tests for SB02MX - Extended Riccati preprocessing routine.

Computes:
    G = B*R^(-1)*B'
    A_bar = A +/- op(B*R^(-1)*L')
    Q_bar = Q +/- L*R^(-1)*L'

Extended version of SB02MT with TRANS, FLAG, and DEF parameters.
"""

import numpy as np
import pytest


def test_sb02mx_basic_compute_g_positive_r():
    """
    Test SB02MX computing G = B*R^(-1)*B' with positive definite R.

    Uses TRANS='N', FLAG='M' (minus sign, as in SB02MT).
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n, m = 4, 2

    B = np.random.randn(n, m).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_inv = np.linalg.inv(R)
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mx

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mx(
        'G', 'Z', 'N', 'U', 'N', 'M', 'D',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == 0, f"sb02mx failed with info={info}"
    assert oufact == 1, f"Expected Cholesky factorization (oufact=1), got {oufact}"

    G_upper = np.triu(G_out)
    G_expected_upper = np.triu(G_expected)
    np.testing.assert_allclose(G_upper, G_expected_upper, rtol=1e-13, atol=1e-14)


def test_sb02mx_with_coupling_matrix_trans_n():
    """
    Test SB02MX with nonzero coupling matrix L and TRANS='N'.

    Computes with TRANS='N', FLAG='M':
        A_bar = A - B*R^(-1)*L' (when TRANS='N', multiply on right)
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

    from ctrlsys import sb02mx

    A_copy = A.copy()
    B_copy = B.copy()
    Q_copy = Q.copy()
    R_copy = R.copy()
    L_copy = L.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    A_out, B_out, Q_out, L_out, G_out, oufact, info = sb02mx(
        'G', 'N', 'N', 'U', 'N', 'M', 'D',
        n, m, A_copy, B_copy, Q_copy, R_copy, L_copy, G
    )

    assert info == 0, f"sb02mx failed with info={info}"
    assert oufact == 1, f"Expected Cholesky factorization (oufact=1), got {oufact}"

    np.testing.assert_allclose(A_out, A_bar_expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.triu(Q_out), np.triu(Q_bar_expected), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.triu(G_out), np.triu(G_expected), rtol=1e-13, atol=1e-14)


def test_sb02mx_with_coupling_matrix_trans_t():
    """
    Test SB02MX with TRANS='T' (transpose operation).

    Computes with TRANS='T', FLAG='M':
        A_bar = A - L*R^(-1)*B' (when TRANS='T', multiply L first)
        Q_bar = Q - L*R^(-1)*L'
        G = B*R^(-1)*B'

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    L = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_inv = np.linalg.inv(R)
    A_bar_expected = A - L @ R_inv @ B.T
    Q_bar_expected = Q - L @ R_inv @ L.T
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mx

    A_copy = A.copy()
    B_copy = B.copy()
    Q_copy = Q.copy()
    R_copy = R.copy()
    L_copy = L.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    A_out, B_out, Q_out, L_out, G_out, oufact, info = sb02mx(
        'G', 'N', 'N', 'U', 'T', 'M', 'D',
        n, m, A_copy, B_copy, Q_copy, R_copy, L_copy, G
    )

    assert info == 0, f"sb02mx failed with info={info}"

    np.testing.assert_allclose(A_out, A_bar_expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.triu(Q_out), np.triu(Q_bar_expected), rtol=1e-13, atol=1e-14)


def test_sb02mx_flag_plus():
    """
    Test SB02MX with FLAG='P' (plus sign).

    Computes with TRANS='N', FLAG='P':
        A_bar = A + B*R^(-1)*L'
        Q_bar = Q + L*R^(-1)*L'

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    L = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_inv = np.linalg.inv(R)
    A_bar_expected = A + B @ R_inv @ L.T
    Q_bar_expected = Q + L @ R_inv @ L.T

    from ctrlsys import sb02mx

    A_copy = A.copy()
    B_copy = B.copy()
    Q_copy = Q.copy()
    R_copy = R.copy()
    L_copy = L.copy()

    A_out, B_out, Q_out, L_out, oufact, info = sb02mx(
        'N', 'N', 'N', 'U', 'N', 'P', 'D',
        n, m, A_copy, B_copy, Q_copy, R_copy, L_copy, None
    )

    assert info == 0, f"sb02mx failed with info={info}"

    np.testing.assert_allclose(A_out, A_bar_expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.triu(Q_out), np.triu(Q_bar_expected), rtol=1e-13, atol=1e-14)


def test_sb02mx_indefinite_r():
    """
    Test SB02MX with indefinite R (DEF='I', uses UdU' factorization).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    n, m = 3, 2

    B = np.random.randn(n, m).astype(float, order='F')

    R = np.array([[2.0, 0.5],
                  [0.5, -1.0]], dtype=float, order='F')

    R_inv = np.linalg.inv(R)
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mx

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mx(
        'G', 'Z', 'N', 'U', 'N', 'M', 'I',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == 0, f"sb02mx failed with info={info}"
    assert oufact == 2, f"Expected UdU' factorization (oufact=2), got {oufact}"

    G_upper = np.triu(G_out)
    G_expected_upper = np.triu(G_expected)
    np.testing.assert_allclose(G_upper, G_expected_upper, rtol=1e-12, atol=1e-13)


def test_sb02mx_precomputed_cholesky():
    """
    Test SB02MX with precomputed Cholesky factor (FACT='C').

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)

    n, m = 3, 2

    B = np.random.randn(n, m).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_chol = np.linalg.cholesky(R).T.astype(float, order='F')

    R_inv = np.linalg.inv(R)
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mx

    R_chol_copy = R_chol.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mx(
        'G', 'Z', 'C', 'U', 'N', 'M', 'D',
        n, m, None, B_copy, None, R_chol_copy, None, G
    )

    assert info == 0
    assert oufact == 1

    G_upper = np.triu(G_out)
    G_expected_upper = np.triu(G_expected)
    np.testing.assert_allclose(G_upper, G_expected_upper, rtol=1e-13, atol=1e-14)


def test_sb02mx_lower_triangle():
    """
    Test SB02MX with lower triangle storage (UPLO='L').

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)

    n, m = 3, 2

    B = np.random.randn(n, m).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    R_inv = np.linalg.inv(R)
    G_expected = B @ R_inv @ B.T

    from ctrlsys import sb02mx

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mx(
        'G', 'Z', 'N', 'L', 'N', 'M', 'D',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == 0
    assert oufact == 1

    G_lower = np.tril(G_out)
    G_expected_lower = np.tril(G_expected)
    np.testing.assert_allclose(G_lower, G_expected_lower, rtol=1e-13, atol=1e-14)


def test_sb02mx_zero_m():
    """
    Test SB02MX with M=0 (quick return).
    """
    n, m = 3, 0

    B = np.zeros((n, 1), dtype=float, order='F')
    R = np.zeros((1, 1), dtype=float, order='F')
    G = np.zeros((n, n), dtype=float, order='F')

    from ctrlsys import sb02mx

    G_out, oufact, info = sb02mx(
        'G', 'Z', 'N', 'U', 'N', 'M', 'D',
        n, m, None, B, None, R, None, G
    )

    assert info == 0
    assert oufact == 0
    np.testing.assert_allclose(G_out, np.zeros((n, n)), rtol=1e-14)


def test_sb02mx_symmetric_g_property():
    """
    Validate mathematical property: G = B*R^(-1)*B' is symmetric.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)

    n, m = 5, 3

    B = np.random.randn(n, m).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    from ctrlsys import sb02mx

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mx(
        'G', 'Z', 'N', 'U', 'N', 'M', 'D',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == 0

    G_full = np.triu(G_out) + np.triu(G_out, 1).T

    np.testing.assert_allclose(G_full, G_full.T, rtol=1e-14, atol=1e-15)


def test_sb02mx_singular_r_error():
    """
    Test SB02MX error handling for numerically singular R.
    """
    n, m = 3, 2

    B = np.ones((n, m), dtype=float, order='F')

    R = np.array([[1.0, 2.0],
                  [2.0, 4.0]], dtype=float, order='F')

    from ctrlsys import sb02mx

    R_copy = R.copy()
    B_copy = B.copy()
    G = np.zeros((n, n), dtype=float, order='F')

    G_out, oufact, info = sb02mx(
        'G', 'Z', 'N', 'U', 'N', 'M', 'D',
        n, m, None, B_copy, None, R_copy, None, G
    )

    assert info == m + 1 or info > 0


def test_sb02mx_error_invalid_jobg():
    """
    Test SB02MX error handling: invalid JOBG parameter.
    """
    n, m = 3, 2

    B = np.zeros((n, m), dtype=float, order='F')
    R = np.eye(m, dtype=float, order='F')
    G = np.zeros((n, n), dtype=float, order='F')

    from ctrlsys import sb02mx

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02mx('X', 'Z', 'N', 'U', 'N', 'M', 'D', n, m, None, B, None, R, None, G)


def test_sb02mx_error_invalid_trans():
    """
    Test SB02MX error handling: invalid TRANS parameter.
    """
    n, m = 3, 2

    B = np.zeros((n, m), dtype=float, order='F')
    R = np.eye(m, dtype=float, order='F')
    G = np.zeros((n, n), dtype=float, order='F')

    from ctrlsys import sb02mx

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02mx('G', 'Z', 'N', 'U', 'X', 'M', 'D', n, m, None, B, None, R, None, G)


def test_sb02mx_error_invalid_flag():
    """
    Test SB02MX error handling: invalid FLAG parameter.
    """
    n, m = 3, 2

    B = np.zeros((n, m), dtype=float, order='F')
    R = np.eye(m, dtype=float, order='F')
    G = np.zeros((n, n), dtype=float, order='F')

    from ctrlsys import sb02mx

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02mx('G', 'Z', 'N', 'U', 'N', 'X', 'D', n, m, None, B, None, R, None, G)


def test_sb02mx_error_negative_n():
    """
    Test SB02MX error handling: negative N.
    """
    from ctrlsys import sb02mx

    B = np.zeros((1, 1), dtype=float, order='F')
    R = np.eye(1, dtype=float, order='F')
    G = np.zeros((1, 1), dtype=float, order='F')

    with pytest.raises(ValueError, match="[Nn]"):
        sb02mx('G', 'Z', 'N', 'U', 'N', 'M', 'D', -1, 1, None, B, None, R, None, G)
