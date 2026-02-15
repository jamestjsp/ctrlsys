"""
Tests for SB02RU - Hamiltonian/Symplectic Matrix Construction.

Constructs 2n-by-2n Hamiltonian (continuous) or symplectic (discrete) matrix
for Riccati equation solvers.

For continuous-time (DICO='C'):
    S = [ op(A)   -G    ]
        [  -Q   -op(A)' ]

For discrete-time (DICO='D') with HINV='D':
    S = [  A^{-1}           A^{-1}*G       ]
        [ Q*A^{-1}     A' + Q*A^{-1}*G ]
"""

import numpy as np
import pytest


def test_sb02ru_continuous_basic():
    """
    Test SB02RU continuous-time Hamiltonian construction.

    Validates the basic structure:
    S = [ op(A)   -G    ]
        [  -Q   -op(A)' ]

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb02ru

    np.random.seed(42)
    n = 3

    A = np.random.randn(n, n).astype(float, order='F')

    G = np.eye(n, dtype=float, order='F') * 0.5
    Q = np.eye(n, dtype=float, order='F')

    S, rcond, pivotg, info = sb02ru('C', 'D', 'N', 'U', A, G, Q)

    assert info == 0, f"sb02ru failed with info={info}"
    assert S.shape == (2*n, 2*n)

    S11 = S[:n, :n]
    S12 = S[:n, n:]
    S21 = S[n:, :n]
    S22 = S[n:, n:]

    np.testing.assert_allclose(S11, A, rtol=1e-14)
    np.testing.assert_allclose(S12, -G, rtol=1e-14)
    np.testing.assert_allclose(S21, -Q, rtol=1e-14)
    np.testing.assert_allclose(S22, -A.T, rtol=1e-14)


def test_sb02ru_continuous_transpose():
    """
    Test SB02RU continuous-time with TRANA='T' (use A').

    Validates:
    S = [ A'   -G    ]
        [ -Q   -A    ]

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb02ru

    np.random.seed(123)
    n = 3

    A = np.random.randn(n, n).astype(float, order='F')
    G = np.random.randn(n, n)
    G = (G + G.T) / 2
    G = np.asfortranarray(G)

    Q = np.random.randn(n, n)
    Q = (Q + Q.T) / 2
    Q = np.asfortranarray(Q)

    S, rcond, pivotg, info = sb02ru('C', 'D', 'T', 'U', A, G, Q)

    assert info == 0

    S11 = S[:n, :n]
    S12 = S[:n, n:]
    S21 = S[n:, :n]
    S22 = S[n:, n:]

    np.testing.assert_allclose(S11, A.T, rtol=1e-14)
    np.testing.assert_allclose(S12, -G, rtol=1e-14)
    np.testing.assert_allclose(S21, -Q, rtol=1e-14)
    np.testing.assert_allclose(S22, -A, rtol=1e-14)


def test_sb02ru_continuous_hamiltonian_eigenvalue_property():
    """
    Validate Hamiltonian eigenvalue property: eigenvalues come in +/- pairs.

    For a Hamiltonian matrix H, if lambda is an eigenvalue,
    then -lambda is also an eigenvalue.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb02ru

    np.random.seed(456)
    n = 4

    A = np.random.randn(n, n).astype(float, order='F')

    G_raw = np.random.randn(n, n)
    G = (G_raw + G_raw.T) / 2
    G = np.asfortranarray(G)

    Q_raw = np.random.randn(n, n)
    Q = (Q_raw + Q_raw.T) / 2
    Q = np.asfortranarray(Q)

    S, rcond, pivotg, info = sb02ru('C', 'D', 'N', 'U', A, G, Q)

    assert info == 0

    eigvals = np.linalg.eigvals(S)

    eigvals_sorted = np.sort(eigvals.real)
    negated = np.sort(-eigvals.real)
    np.testing.assert_allclose(eigvals_sorted, negated, rtol=1e-10)


def test_sb02ru_discrete_basic():
    """
    Test SB02RU discrete-time symplectic construction with HINV='D'.

    For discrete-time with HINV='D':
    S = [  A^{-1}           A^{-1}*G       ]
        [ Q*A^{-1}     A' + Q*A^{-1}*G ]

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb02ru

    np.random.seed(789)
    n = 3

    A = np.eye(n) + 0.3 * np.random.randn(n, n)
    A = np.asfortranarray(A)

    G = np.eye(n, dtype=float, order='F') * 0.2
    Q = np.eye(n, dtype=float, order='F') * 0.5

    S, rcond, pivotg, info = sb02ru('D', 'D', 'N', 'U', A, G, Q)

    assert info == 0, f"sb02ru discrete failed with info={info}"
    assert S.shape == (2*n, 2*n)
    assert rcond > 0, "rcond should be positive for well-conditioned A"

    Ainv = np.linalg.inv(A)

    S11_expected = Ainv
    S12_expected = Ainv @ G
    S21_expected = Q @ Ainv
    S22_expected = A.T + Q @ Ainv @ G

    S11 = S[:n, :n]
    S12 = S[:n, n:]
    S21 = S[n:, :n]
    S22 = S[n:, n:]

    np.testing.assert_allclose(S11, S11_expected, rtol=1e-12)
    np.testing.assert_allclose(S12, S12_expected, rtol=1e-12)
    np.testing.assert_allclose(S21, S21_expected, rtol=1e-12)
    np.testing.assert_allclose(S22, S22_expected, rtol=1e-12)


def test_sb02ru_discrete_hinv_inverse():
    """
    Test SB02RU discrete-time with HINV='I' (inverse formula).

    For discrete-time with HINV='I':
    S = [ A + G*A'^{-1}*Q   -G*A'^{-1} ]
        [   -A'^{-1}*Q        A'^{-1}  ]

    Note: The original SLICOT SB02RU.f had a bug in the HINV='I' transpose loop
    (lines 466-474) where nested DO loops processed off-diagonal pairs twice.
    Our C implementation (sb02ru.c) fixes this by using correct loop bounds
    (i <= j - n) to process each pair exactly once.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import sb02ru

    np.random.seed(111)
    n = 3

    A = np.eye(n) + 0.3 * np.random.randn(n, n)
    A = np.asfortranarray(A)

    G = np.eye(n, dtype=float, order='F') * 0.2
    Q = np.eye(n, dtype=float, order='F') * 0.5

    S, rcond, pivotg, info = sb02ru('D', 'I', 'N', 'U', A, G, Q)

    assert info == 0, f"sb02ru discrete HINV=I failed with info={info}"

    ATinv = np.linalg.inv(A.T)

    S11_expected = A + G @ ATinv @ Q
    S12_expected = -G @ ATinv
    S21_expected = -ATinv @ Q
    S22_expected = ATinv

    S11 = S[:n, :n]
    S12 = S[:n, n:]
    S21 = S[n:, :n]
    S22 = S[n:, n:]

    np.testing.assert_allclose(S11, S11_expected, rtol=1e-12)
    np.testing.assert_allclose(S12, S12_expected, rtol=1e-12)
    np.testing.assert_allclose(S21, S21_expected, rtol=1e-12)
    np.testing.assert_allclose(S22, S22_expected, rtol=1e-12)


def test_sb02ru_discrete_symplectic_property():
    """
    Validate symplectic eigenvalue property: eigenvalues come in (lambda, 1/lambda) pairs.

    For a symplectic matrix S, if lambda is an eigenvalue,
    then 1/lambda is also an eigenvalue.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import sb02ru

    np.random.seed(222)
    n = 3

    A = np.eye(n) + 0.2 * np.random.randn(n, n)
    A = np.asfortranarray(A)

    G = np.eye(n, dtype=float, order='F') * 0.1
    Q = np.eye(n, dtype=float, order='F') * 0.1

    S, rcond, pivotg, info = sb02ru('D', 'D', 'N', 'U', A, G, Q)

    assert info == 0

    eigvals = np.linalg.eigvals(S)

    for ev in eigvals:
        reciprocal = 1.0 / ev
        dists = np.abs(eigvals - reciprocal)
        assert np.min(dists) < 1e-10, f"Eigenvalue {ev} lacks reciprocal pair"


def test_sb02ru_discrete_singular_a():
    """
    Test SB02RU discrete-time with singular A matrix.

    Should return info > 0 when A is singular.
    """
    from ctrlsys import sb02ru

    n = 3

    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0]], order='F', dtype=float)

    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    S, rcond, pivotg, info = sb02ru('D', 'D', 'N', 'U', A, G, Q)

    assert info > 0, "Should detect singular A"


def test_sb02ru_n_zero():
    """
    Test SB02RU edge case with n=0.
    """
    from ctrlsys import sb02ru

    A = np.zeros((0, 0), order='F', dtype=float)
    G = np.zeros((0, 0), order='F', dtype=float)
    Q = np.zeros((0, 0), order='F', dtype=float)

    S, rcond, pivotg, info = sb02ru('C', 'D', 'N', 'U', A, G, Q)

    assert info == 0
    assert S.shape == (0, 0)


def test_sb02ru_lower_triangular_storage():
    """
    Test SB02RU with UPLO='L' (lower triangular storage).

    Q and G are symmetric, stored in lower triangle.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import sb02ru

    np.random.seed(333)
    n = 3

    A = np.random.randn(n, n).astype(float, order='F')

    G_full = np.random.randn(n, n)
    G_full = (G_full + G_full.T) / 2
    G = np.tril(G_full).astype(float, order='F')

    Q_full = np.random.randn(n, n)
    Q_full = (Q_full + Q_full.T) / 2
    Q = np.tril(Q_full).astype(float, order='F')

    S, rcond, pivotg, info = sb02ru('C', 'D', 'N', 'L', A, G, Q)

    assert info == 0

    S11 = S[:n, :n]
    S12 = S[:n, n:]
    S21 = S[n:, :n]
    S22 = S[n:, n:]

    np.testing.assert_allclose(S11, A, rtol=1e-14)
    np.testing.assert_allclose(S12, -G_full, rtol=1e-14)
    np.testing.assert_allclose(S21, -Q_full, rtol=1e-14)
    np.testing.assert_allclose(S22, -A.T, rtol=1e-14)
