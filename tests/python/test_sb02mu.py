"""
Tests for SB02MU - Hamiltonian/Symplectic Matrix Construction.

Constructs the 2n-by-2n Hamiltonian or symplectic matrix S associated
to the linear-quadratic optimization problem.

Continuous-time (Hamiltonian):
    S = [  A   -G ]
        [ -Q   -A']

Discrete-time (Symplectic, HINV='D'):
    S = [  A^(-1)        A^(-1)*G     ]
        [ Q*A^(-1)   A' + Q*A^(-1)*G  ]

Discrete-time (Symplectic inverse, HINV='I'):
    S = [ A + G*A^(-T)*Q   -G*A^(-T) ]
        [    -A^(-T)*Q       A^(-T)  ]
"""

import numpy as np
import pytest


def test_sb02mu_continuous_time_hamiltonian():
    """
    Test SB02MU continuous-time Hamiltonian matrix construction.

    For DICO='C', constructs:
        S = [  A   -G ]
            [ -Q   -A']

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n = 3

    A = np.array([
        [0.5, 0.2, 0.1],
        [0.1, 0.4, 0.3],
        [0.2, 0.1, 0.6]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    S_expected = np.zeros((2*n, 2*n), dtype=float, order='F')
    S_expected[:n, :n] = A
    S_expected[:n, n:] = -G
    S_expected[n:, :n] = -Q
    S_expected[n:, n:] = -A.T

    from ctrlsys import sb02mu

    A_copy = A.copy()
    S, rcond, info = sb02mu('C', 'D', 'U', n, A_copy, G, Q)

    assert info == 0, f"sb02mu failed with info={info}"

    np.testing.assert_allclose(S, S_expected, rtol=1e-14, atol=1e-15)


def test_sb02mu_continuous_time_lower_triangle():
    """
    Test SB02MU with lower triangle storage (UPLO='L').

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n = 3

    A = np.random.randn(n, n).astype(float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    S_expected = np.zeros((2*n, 2*n), dtype=float, order='F')
    S_expected[:n, :n] = A
    S_expected[:n, n:] = -G
    S_expected[n:, :n] = -Q
    S_expected[n:, n:] = -A.T

    from ctrlsys import sb02mu

    A_copy = A.copy()
    S, rcond, info = sb02mu('C', 'D', 'L', n, A_copy, G, Q)

    assert info == 0
    np.testing.assert_allclose(S, S_expected, rtol=1e-14, atol=1e-15)


def test_sb02mu_discrete_time_symplectic_hinv_d():
    """
    Test SB02MU discrete-time symplectic matrix with HINV='D'.

    For DICO='D', HINV='D', constructs:
        S = [  A^(-1)        A^(-1)*G     ]
            [ Q*A^(-1)   A' + Q*A^(-1)*G  ]

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n = 3

    A = np.array([
        [2.0, 0.5, 0.3],
        [0.2, 1.5, 0.4],
        [0.1, 0.3, 2.5]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    A_inv = np.linalg.inv(A)
    S_expected = np.zeros((2*n, 2*n), dtype=float, order='F')
    S_expected[:n, :n] = A_inv
    S_expected[:n, n:] = A_inv @ G
    S_expected[n:, :n] = Q @ A_inv
    S_expected[n:, n:] = A.T + Q @ A_inv @ G

    from ctrlsys import sb02mu

    A_copy = A.copy()
    S, rcond, info = sb02mu('D', 'D', 'U', n, A_copy, G, Q)

    assert info == 0, f"sb02mu failed with info={info}"
    assert rcond > 0, f"Condition number should be positive, got {rcond}"

    np.testing.assert_allclose(S, S_expected, rtol=1e-13, atol=1e-14)

    np.testing.assert_allclose(A_copy, A_inv, rtol=1e-13, atol=1e-14)


def test_sb02mu_discrete_time_symplectic_hinv_i():
    """
    Test SB02MU discrete-time symplectic matrix with HINV='I' (inverse form).

    For DICO='D', HINV='I', constructs:
        S = [ A + G*A^(-T)*Q   -G*A^(-T) ]
            [    -A^(-T)*Q       A^(-T)  ]

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n = 3

    A = np.array([
        [1.8, 0.4, 0.2],
        [0.3, 2.2, 0.5],
        [0.1, 0.2, 1.9]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    A_inv_T = np.linalg.inv(A.T)
    S_expected = np.zeros((2*n, 2*n), dtype=float, order='F')
    S_expected[:n, :n] = A + G @ A_inv_T @ Q
    S_expected[:n, n:] = -G @ A_inv_T
    S_expected[n:, :n] = -A_inv_T @ Q
    S_expected[n:, n:] = A_inv_T

    from ctrlsys import sb02mu

    A_copy = A.copy()
    S, rcond, info = sb02mu('D', 'I', 'U', n, A_copy, G, Q)

    assert info == 0
    assert rcond > 0

    np.testing.assert_allclose(S, S_expected, rtol=1e-13, atol=1e-14)


def test_sb02mu_hamiltonian_eigenvalue_pairing():
    """
    Validate mathematical property: Hamiltonian eigenvalues come in +/- pairs.

    For Hamiltonian matrix H, if λ is eigenvalue then -λ is also eigenvalue.
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    n = 4

    A = np.array([
        [-0.5, 0.2, 0.1, 0.0],
        [0.1, -0.4, 0.2, 0.1],
        [0.0, 0.1, -0.6, 0.2],
        [0.1, 0.0, 0.1, -0.3]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + 0.5 * np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + 0.5 * np.eye(n)).astype(float, order='F')

    from ctrlsys import sb02mu

    A_copy = A.copy()
    S, rcond, info = sb02mu('C', 'D', 'U', n, A_copy, G, Q)

    assert info == 0

    eig = np.linalg.eigvals(S)

    for ev in eig:
        found_pair = any(np.abs(ev + other) < 1e-10 for other in eig)
        assert found_pair, f"Eigenvalue {ev} has no -λ pair"


def test_sb02mu_symplectic_determinant_unity():
    """
    Validate mathematical property: symplectic matrix has det(S) = 1.

    For symplectic matrices constructed from discrete-time Riccati problem,
    the determinant should be unity (up to sign).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)

    n = 3

    A = np.diag([2.0, 1.5, 1.8]).astype(float, order='F')
    A[0, 1] = 0.3
    A[1, 2] = 0.2

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    from ctrlsys import sb02mu

    A_copy = A.copy()
    S, rcond, info = sb02mu('D', 'D', 'U', n, A_copy, G, Q)

    assert info == 0

    det_S = np.linalg.det(S)
    np.testing.assert_allclose(np.abs(det_S), 1.0, rtol=1e-12)


def test_sb02mu_singular_matrix_error():
    """
    Test SB02MU error handling for singular A in discrete-time case.

    When A is singular, discrete-time case should return info > 0.
    """
    n = 3

    A = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 1.0, 1.0]
    ], dtype=float, order='F')

    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02mu

    A_copy = A.copy()
    S, rcond, info = sb02mu('D', 'D', 'U', n, A_copy, G, Q)

    assert info > 0, f"Expected info > 0 for singular A, got info={info}"


def test_sb02mu_nearly_singular_matrix():
    """
    Test SB02MU handling of nearly singular A (condition number warning).

    When A is nearly singular, should return info = N+1 and small rcond.
    """
    n = 3

    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1e-16]
    ], dtype=float, order='F')

    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02mu

    A_copy = A.copy()
    S, rcond, info = sb02mu('D', 'D', 'U', n, A_copy, G, Q)

    assert info == n + 1, f"Expected info={n+1} for nearly singular A, got {info}"


def test_sb02mu_n_zero():
    """
    Test SB02MU quick return for n=0.
    """
    n = 0

    A = np.zeros((1, 1), dtype=float, order='F')
    G = np.zeros((1, 1), dtype=float, order='F')
    Q = np.zeros((1, 1), dtype=float, order='F')

    from ctrlsys import sb02mu

    S, rcond, info = sb02mu('C', 'D', 'U', n, A, G, Q)

    assert info == 0
    assert S.shape == (0, 0) or S.size == 0


def test_sb02mu_error_invalid_dico():
    """
    Test SB02MU error handling: invalid DICO parameter.
    """
    n = 2

    A = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02mu

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02mu('X', 'D', 'U', n, A, G, Q)


def test_sb02mu_error_invalid_hinv():
    """
    Test SB02MU error handling: invalid HINV parameter.
    """
    n = 2

    A = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02mu

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02mu('D', 'X', 'U', n, A, G, Q)


def test_sb02mu_error_invalid_uplo():
    """
    Test SB02MU error handling: invalid UPLO parameter.
    """
    n = 2

    A = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02mu

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02mu('C', 'D', 'X', n, A, G, Q)


def test_sb02mu_error_negative_n():
    """
    Test SB02MU error handling: negative N.
    """
    from ctrlsys import sb02mu

    A = np.eye(1, dtype=float, order='F')
    G = np.eye(1, dtype=float, order='F')
    Q = np.eye(1, dtype=float, order='F')

    with pytest.raises(ValueError, match="[Nn]"):
        sb02mu('C', 'D', 'U', -1, A, G, Q)
