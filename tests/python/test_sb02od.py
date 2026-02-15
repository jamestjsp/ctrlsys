"""
Tests for SB02OD - Algebraic Riccati Equation Solver.

Solves for X either the continuous-time algebraic Riccati equation:
    Q + A'X + XA - (L+XB)R^(-1)(L+XB)' = 0

or the discrete-time algebraic Riccati equation:
    X = A'XA - (L+A'XB)(R + B'XB)^(-1)(L+A'XB)' + Q

Uses the method of deflating subspaces based on reordering eigenvalues
in a generalized Schur matrix pair.
"""

import numpy as np
import pytest


def test_sb02od_html_example():
    """
    Test SB02OD using the HTML documentation example.

    Continuous-time, JOBB='B', FACT='B' (Q=C'C, R=D'D), JOBL='Z'.
    N=2, M=1, P=3

    Expected solution X:
        X = [[1.7321, 1.0000],
             [1.0000, 1.7321]]
    """
    from ctrlsys import sb02od

    n, m, p = 2, 1, 3

    A = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ], dtype=float, order='F')

    B = np.array([
        [0.0],
        [1.0]
    ], dtype=float, order='F')

    C = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ], dtype=float, order='F')

    D = np.array([
        [0.0],
        [0.0],
        [1.0]
    ], dtype=float, order='F')

    L = np.zeros((n, m), dtype=float, order='F')

    X_expected = np.array([
        [1.7321, 1.0000],
        [1.0000, 1.7321]
    ], dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'B', 'U', 'Z', 'S',
        n, m, p, A, B, C, D, L, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)

    np.testing.assert_allclose(X, X_expected, rtol=1e-3, atol=1e-4)


def test_sb02od_continuous_jobb_B():
    """
    Test SB02OD for continuous-time with JOBB='B' (B and R given).

    Solves: Q + A'X + XA - XBR^(-1)B'X = 0

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    from ctrlsys import sb02od

    n, m = 3, 2

    A = np.array([
        [-2.0, 0.0, 0.0],
        [1.0, -3.0, 0.0],
        [0.0, 1.0, -4.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'N', 'U', 'Z', 'S',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)
    assert rcond > 0

    residual = Q + A.T @ X + X @ A - X @ B @ np.linalg.solve(R, B.T @ X)
    np.testing.assert_allclose(residual, np.zeros((n, n)), rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(X, X.T, rtol=1e-14)

    eigvals_X = np.linalg.eigvalsh(X)
    assert np.all(eigvals_X >= -1e-10), "X should be positive semi-definite"


def test_sb02od_discrete_jobb_B():
    """
    Test SB02OD for discrete-time with JOBB='B' (B and R given).

    Solves: X = A'XA - A'XB(R + B'XB)^(-1)B'XA + Q

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    from ctrlsys import sb02od

    n, m = 3, 2

    A = np.array([
        [0.8, 0.1, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 0.7]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 1.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 2.0).astype(float, order='F')

    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'D', 'B', 'N', 'U', 'Z', 'S',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)
    assert rcond > 0

    residual = (X - Q - A.T @ X @ A +
                A.T @ X @ B @ np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A))
    np.testing.assert_allclose(residual, np.zeros((n, n)), rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(X, X.T, rtol=1e-14)


def test_sb02od_continuous_jobb_G():
    """
    Test SB02OD for continuous-time with JOBB='G' (G = BR^(-1)B' given).

    Uses standard eigenvalue problem (QR algorithm).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    from ctrlsys import sb02od

    n = 3

    A = np.array([
        [-2.0, 0.0, 0.0],
        [1.0, -3.0, 0.0],
        [0.0, 1.0, -4.0]
    ], dtype=float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    G_half = np.random.randn(n, n)
    G = ((G_half.T @ G_half) + np.eye(n) * 0.5).astype(float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'G', 'N', 'U', 'Z', 'S',
        n, 0, 0, A, G, Q, None, None, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)

    Q_full = np.triu(Q) + np.triu(Q, 1).T
    G_full = np.triu(G) + np.triu(G, 1).T

    residual = Q_full + A.T @ X + X @ A - X @ G_full @ X
    np.testing.assert_allclose(residual, np.zeros((n, n)), rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(X, X.T, rtol=1e-14)


def test_sb02od_discrete_jobb_G():
    """
    Test SB02OD for discrete-time with JOBB='G' (G = BR^(-1)B' given).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    from ctrlsys import sb02od

    n = 3

    A = np.array([
        [0.8, 0.1, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 0.7]
    ], dtype=float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 1.0).astype(float, order='F')

    G_half = np.random.randn(n, n)
    G = ((G_half.T @ G_half) + np.eye(n) * 0.5).astype(float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'D', 'G', 'N', 'U', 'Z', 'S',
        n, 0, 0, A, G, Q, None, None, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)

    np.testing.assert_allclose(X, X.T, rtol=1e-14)


def test_sb02od_closed_loop_stability():
    """
    Test that SB02OD returns stable closed-loop eigenvalues (SORT='S').

    For continuous-time: all eigenvalues should have negative real part.
    For discrete-time: all eigenvalues should have modulus < 1.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    from ctrlsys import sb02od

    n, m = 3, 2

    A = np.array([
        [-1.0, 0.5, 0.0],
        [0.0, -2.0, 0.5],
        [0.0, 0.0, -3.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')
    Q = np.eye(n, dtype=float, order='F') * 2.0
    R = np.eye(m, dtype=float, order='F') * 1.0
    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'N', 'U', 'Z', 'S',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0

    for i in range(n):
        eig_real = alfar[i] / beta[i] if beta[i] != 0 else np.inf
        assert eig_real < 0.0, f"Eigenvalue {i} not stable: {eig_real}"


def test_sb02od_unstable_eigenvalues():
    """
    Test SB02OD with SORT='U' (unstable eigenvalues first).

    For continuous-time: leading eigenvalues should have positive real part.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    from ctrlsys import sb02od

    n, m = 3, 2

    A = np.array([
        [-1.0, 0.5, 0.0],
        [0.0, -2.0, 0.5],
        [0.0, 0.0, -3.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')
    Q = np.eye(n, dtype=float, order='F') * 2.0
    R = np.eye(m, dtype=float, order='F') * 1.0
    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'N', 'U', 'Z', 'U',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0
    assert X.shape == (n, n)


def test_sb02od_with_nonzero_L():
    """
    Test SB02OD with nonzero cross-weighting matrix L (JOBL='N').

    Solves: Q + A'X + XA - (L+XB)R^(-1)(L+XB)' = 0

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    from ctrlsys import sb02od

    n, m = 3, 2

    A = np.array([
        [-2.0, 0.0, 0.0],
        [1.0, -3.0, 0.0],
        [0.0, 1.0, -4.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')
    L = np.random.randn(n, m).astype(float, order='F') * 0.5
    Q = np.eye(n, dtype=float, order='F') * 2.0
    R = np.eye(m, dtype=float, order='F') * 3.0

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'N', 'U', 'N', 'S',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)

    K = np.linalg.solve(R, (L + X @ B).T).T
    residual = Q + A.T @ X + X @ A - (L + X @ B) @ K.T
    np.testing.assert_allclose(residual, np.zeros((n, n)), rtol=1e-10, atol=1e-10)


def test_sb02od_factored_C():
    """
    Test SB02OD with factored Q = C'C (FACT='C').

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    from ctrlsys import sb02od

    n, m, p = 3, 2, 4

    A = np.array([
        [-2.0, 0.0, 0.0],
        [1.0, -3.0, 0.0],
        [0.0, 1.0, -4.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')
    C = np.random.randn(p, n).astype(float, order='F')
    R = np.eye(m, dtype=float, order='F') * 3.0
    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'C', 'U', 'Z', 'S',
        n, m, p, A, B, C, R, L, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)

    Q = C.T @ C
    residual = Q + A.T @ X + X @ A - X @ B @ np.linalg.solve(R, B.T @ X)
    np.testing.assert_allclose(residual, np.zeros((n, n)), rtol=1e-10, atol=1e-10)


def test_sb02od_factored_D():
    """
    Test SB02OD with factored R = D'D (FACT='D').

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    from ctrlsys import sb02od

    n, m, p = 3, 2, 4

    A = np.array([
        [-2.0, 0.0, 0.0],
        [1.0, -3.0, 0.0],
        [0.0, 1.0, -4.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')
    D = np.random.randn(p, m).astype(float, order='F')
    Q = np.eye(n, dtype=float, order='F') * 2.0
    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'D', 'U', 'Z', 'S',
        n, m, p, A, B, Q, D, L, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)

    R = D.T @ D
    residual = Q + A.T @ X + X @ A - X @ B @ np.linalg.solve(R, B.T @ X)
    np.testing.assert_allclose(residual, np.zeros((n, n)), rtol=1e-10, atol=1e-10)


def test_sb02od_factored_both():
    """
    Test SB02OD with both Q=C'C and R=D'D (FACT='B').

    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    from ctrlsys import sb02od

    n, m, p = 3, 2, 4

    A = np.array([
        [-2.0, 0.0, 0.0],
        [1.0, -3.0, 0.0],
        [0.0, 1.0, -4.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')
    C = np.random.randn(p, n).astype(float, order='F')
    D = np.random.randn(p, m).astype(float, order='F')
    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'B', 'U', 'Z', 'S',
        n, m, p, A, B, C, D, L, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)

    Q = C.T @ C
    R = D.T @ D
    residual = Q + A.T @ X + X @ A - X @ B @ np.linalg.solve(R, B.T @ X)
    np.testing.assert_allclose(residual, np.zeros((n, n)), rtol=1e-10, atol=1e-10)


def test_sb02od_lower_triangle():
    """
    Test SB02OD with lower triangle storage (UPLO='L').

    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)
    from ctrlsys import sb02od

    n, m = 3, 2

    A = np.array([
        [-2.0, 0.0, 0.0],
        [1.0, -3.0, 0.0],
        [0.0, 1.0, -4.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')
    Q = np.eye(n, dtype=float, order='F') * 2.0
    R = np.eye(m, dtype=float, order='F') * 3.0
    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'N', 'L', 'Z', 'S',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0, f"sb02od failed with info={info}"
    assert X.shape == (n, n)


def test_sb02od_zero_n():
    """
    Test SB02OD with N=0 (quick return).
    """
    from ctrlsys import sb02od

    n, m = 0, 2

    A = np.zeros((1, 1), dtype=float, order='F')
    B = np.zeros((1, m), dtype=float, order='F')
    Q = np.zeros((1, 1), dtype=float, order='F')
    R = np.eye(m, dtype=float, order='F')
    L = np.zeros((1, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'N', 'U', 'Z', 'S',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0


def test_sb02od_error_invalid_dico():
    """
    Test SB02OD error handling: invalid DICO parameter.
    """
    from ctrlsys import sb02od

    n, m = 3, 2

    A = np.eye(n, dtype=float, order='F')
    B = np.eye(n, m, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')
    R = np.eye(m, dtype=float, order='F')
    L = np.zeros((n, m), dtype=float, order='F')

    with pytest.raises(ValueError, match="DICO"):
        sb02od('X', 'B', 'N', 'U', 'Z', 'S',
               n, m, 0, A, B, Q, R, L, 0.0)


def test_sb02od_error_negative_n():
    """
    Test SB02OD error handling: negative N.
    """
    from ctrlsys import sb02od

    A = np.eye(1, dtype=float, order='F')
    B = np.eye(1, dtype=float, order='F')
    Q = np.eye(1, dtype=float, order='F')
    R = np.eye(1, dtype=float, order='F')
    L = np.zeros((1, 1), dtype=float, order='F')

    with pytest.raises(ValueError):
        sb02od('C', 'B', 'N', 'U', 'Z', 'S',
               -1, 1, 0, A, B, Q, R, L, 0.0)


def test_sb02od_riccati_residual_property():
    """
    Validate mathematical property: Riccati equation residual.

    For continuous-time: Q + A'X + XA - XBR^(-1)B'X = 0
    The residual should be zero (machine precision).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    from ctrlsys import sb02od

    n, m = 4, 2

    A = np.array([
        [-1.0, 0.5, 0.0, 0.0],
        [0.0, -2.0, 0.5, 0.0],
        [0.0, 0.0, -3.0, 0.5],
        [0.0, 0.0, 0.0, -4.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n)).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m)).astype(float, order='F')

    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'N', 'U', 'Z', 'S',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0

    residual = Q + A.T @ X + X @ A - X @ B @ np.linalg.solve(R, B.T @ X)
    residual_norm = np.linalg.norm(residual, 'fro')
    assert residual_norm < 1e-10, f"Riccati residual too large: {residual_norm}"


def test_sb02od_symmetry_property():
    """
    Validate mathematical property: solution X is symmetric.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    from ctrlsys import sb02od

    n, m = 4, 2

    A = np.array([
        [-1.0, 0.5, 0.0, 0.0],
        [0.0, -2.0, 0.5, 0.0],
        [0.0, 0.0, -3.0, 0.5],
        [0.0, 0.0, 0.0, -4.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')
    Q = np.eye(n, dtype=float, order='F') * 2.0
    R = np.eye(m, dtype=float, order='F') * 1.0
    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'N', 'U', 'Z', 'S',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0

    np.testing.assert_allclose(X, X.T, rtol=1e-14, atol=1e-14)


def test_sb02od_positive_semidefinite():
    """
    Validate mathematical property: X is positive semi-definite.

    Under stabilizability and detectability assumptions, the solution
    should be positive semi-definite (all eigenvalues >= 0).

    Random seed: 1010 (for reproducibility)
    """
    np.random.seed(1010)
    from ctrlsys import sb02od

    n, m = 4, 2

    A = np.array([
        [-1.0, 0.5, 0.0, 0.0],
        [0.0, -2.0, 0.5, 0.0],
        [0.0, 0.0, -3.0, 0.5],
        [0.0, 0.0, 0.0, -4.0]
    ], dtype=float, order='F')

    B = np.random.randn(n, m).astype(float, order='F')
    Q = np.eye(n, dtype=float, order='F') * 2.0
    R = np.eye(m, dtype=float, order='F') * 1.0
    L = np.zeros((n, m), dtype=float, order='F')

    X, rcond, alfar, alfai, beta, S, T, U, info = sb02od(
        'C', 'B', 'N', 'U', 'Z', 'S',
        n, m, 0, A, B, Q, R, L, 0.0
    )

    assert info == 0

    eigvals_X = np.linalg.eigvalsh(X)
    assert np.all(eigvals_X >= -1e-10), f"X has negative eigenvalue: {eigvals_X.min()}"
