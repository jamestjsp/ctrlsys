"""
Tests for SB02MD - Continuous/Discrete-time Algebraic Riccati Equation Solver.

Solves for X the continuous-time algebraic Riccati equation:
    Q + A'*X + X*A - X*G*X = 0   (DICO='C')

Or the discrete-time algebraic Riccati equation:
    X = A'*X*A - A'*X*B*(R + B'*X*B)^-1 B'*X*A + Q   (DICO='D')

where G = B*R^-1*B' must be provided on input.

Returns the solution matrix X and the closed-loop spectrum.
Uses Laub's Schur vector method.
"""

import numpy as np
import pytest


def test_sb02md_html_doc_example():
    """
    Test SB02MD using HTML documentation example data.

    Problem: Continuous-time Riccati Q + A'*X + X*A - X*G*X = 0
    N=2, DICO='C', HINV='D', UPLO='U', SCAL='N', SORT='S'

    Input (row-wise from HTML):
        A = [0, 1; 0, 0]
        Q = [1, 0; 0, 2]
        G = [0, 0; 0, 1]

    Expected output (RCOND ~0.31):
        X = [2, 1; 1, 2]
    """
    from ctrlsys import sb02md

    n = 2

    A = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ], dtype=float, order='F')

    Q = np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ], dtype=float, order='F')

    G = np.array([
        [0.0, 0.0],
        [0.0, 1.0]
    ], dtype=float, order='F')

    X_expected = np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ], dtype=float, order='F')

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A, G, Q)

    assert info == 0, f"sb02md failed with info={info}"

    np.testing.assert_allclose(X, X_expected, rtol=1e-3, atol=1e-4)

    assert 0.3 < rcond < 0.35, f"Expected RCOND ~0.31, got {rcond}"

    assert wr.shape == (2 * n,)
    assert wi.shape == (2 * n,)
    for i in range(n):
        assert wr[i] < 0, f"Closed-loop eigenvalue {i} should be stable (Re < 0)"


def test_sb02md_continuous_riccati_residual():
    """
    Validate mathematical property: Riccati equation residual should be zero.

    For continuous-time: Q + A'*X + X*A - X*G*X = 0

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    from ctrlsys import sb02md

    n = 4

    A = np.array([
        [-0.5, 0.1, 0.0, 0.0],
        [0.0, -0.4, 0.2, 0.0],
        [0.1, 0.0, -0.6, 0.1],
        [0.0, 0.1, 0.0, -0.3]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + 0.5 * np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')
    Q_orig = Q.copy()

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A.copy(), G, Q.copy())

    assert info == 0, f"sb02md failed with info={info}"
    assert rcond > 0, "rcond should be positive"

    residual = Q_orig + A.T @ X + X @ A - X @ G @ X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb02md_solution_symmetry():
    """
    Validate mathematical property: X should be symmetric.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    from ctrlsys import sb02md

    n = 3

    A = np.array([
        [-0.8, 0.2, 0.1],
        [0.1, -0.5, 0.3],
        [0.0, 0.1, -0.6]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A.copy(), G, Q)

    assert info == 0
    np.testing.assert_allclose(X, X.T, rtol=1e-14, atol=1e-15)


def test_sb02md_solution_positive_semidefinite():
    """
    Validate mathematical property: X should be positive semidefinite.

    For stabilizable (A,B) and detectable (Q,A), solution is positive semidefinite.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    from ctrlsys import sb02md

    n = 4

    A = np.diag([-0.5, -0.4, -0.6, -0.3]).astype(float, order='F')
    A[0, 1] = 0.1
    A[1, 2] = 0.2
    A[2, 3] = 0.1

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A.copy(), G, Q)

    assert info == 0

    eigvals = np.linalg.eigvalsh(X)
    assert all(eigvals >= -1e-10), f"X should be positive semidefinite, min eigenvalue={min(eigvals)}"


def test_sb02md_closed_loop_eigenvalues():
    """
    Validate closed-loop spectrum: eigenvalues of (A - G*X) should match WR+j*WI.

    For continuous-time, closed-loop eigenvalues should have negative real parts.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    from ctrlsys import sb02md

    n = 3

    A = np.array([
        [-0.3, 0.2, 0.1],
        [0.1, -0.5, 0.2],
        [0.0, 0.1, -0.4]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A.copy(), G, Q)

    assert info == 0

    A_cl = A - G @ X
    eig_computed = np.linalg.eigvals(A_cl)

    eig_returned = wr[:n] + 1j * wi[:n]

    for eig_c in eig_computed:
        found = any(np.abs(eig_c - eig_r) < 1e-8 for eig_r in eig_returned)
        assert found, f"Computed eigenvalue {eig_c} not found in returned spectrum"

    for i in range(n):
        assert wr[i] < 0, f"Closed-loop eigenvalue {i} has Re={wr[i]} >= 0"


def test_sb02md_discrete_time_riccati():
    """
    Test discrete-time algebraic Riccati equation.

    For DICO='D', solves:
        X = A'*X*A - A'*X*B*(R + B'*X*B)^-1 B'*X*A + Q

    where G = B*R^-1*B' is provided.

    NOTE: For HINV='D', need SORT='U' to get stabilizing solution.
    For HINV='I', need SORT='S' to get stabilizing solution.
    (see HTML doc: "To obtain a stabilizing solution...")

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    from ctrlsys import sb02md

    n = 3

    A = np.array([
        [0.9, 0.1, 0.0],
        [0.0, 0.8, 0.2],
        [0.1, 0.0, 0.7]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (0.5 * G_half.T @ G_half + 0.1 * np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    A_copy = A.copy()
    X, rcond, wr, wi, S, U, info = sb02md('D', 'D', 'U', 'N', 'U', n, A_copy, G, Q.copy())

    assert info == 0, f"sb02md failed with info={info}"

    np.testing.assert_allclose(X, X.T, rtol=1e-14, atol=1e-15)

    closed_loop_eig = wr[:n] + 1j * wi[:n]
    for eig in closed_loop_eig:
        assert np.abs(eig) < 1.0 + 1e-10, f"Discrete closed-loop eigenvalue should have |λ|<1, got {np.abs(eig)}"


def test_sb02md_discrete_time_hinv_i():
    """
    Test discrete-time Riccati with HINV='I' (inverse symplectic form).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    from ctrlsys import sb02md

    n = 3

    A = np.array([
        [0.85, 0.15, 0.05],
        [0.05, 0.80, 0.15],
        [0.10, 0.05, 0.75]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (0.3 * G_half.T @ G_half + 0.05 * np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    X, rcond, wr, wi, S, U, info = sb02md('D', 'I', 'U', 'N', 'S', n, A.copy(), G, Q)

    assert info == 0
    np.testing.assert_allclose(X, X.T, rtol=1e-14, atol=1e-15)


def test_sb02md_scaling():
    """
    Test SB02MD with scaling (SCAL='G').

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    from ctrlsys import sb02md

    n = 3

    A = np.array([
        [-0.5, 0.2, 0.1],
        [0.1, -0.4, 0.2],
        [0.0, 0.1, -0.6]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + 10 * np.eye(n)).astype(float, order='F')
    Q_orig = Q.copy()

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'G', 'S', n, A.copy(), G, Q.copy())

    assert info == 0

    residual = Q_orig + A.T @ X + X @ A - X @ G @ X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-9)


def test_sb02md_lower_triangle():
    """
    Test SB02MD with lower triangle storage (UPLO='L').

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    from ctrlsys import sb02md

    n = 3

    A = np.array([
        [-0.4, 0.1, 0.0],
        [0.2, -0.5, 0.1],
        [0.0, 0.2, -0.3]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')
    Q_orig = Q.copy()

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'L', 'N', 'S', n, A.copy(), G, Q.copy())

    assert info == 0

    residual = Q_orig + A.T @ X + X @ A - X @ G @ X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb02md_unstable_sort():
    """
    Test SB02MD with unstable eigenvalue sorting (SORT='U').

    For continuous-time, this gives anti-stabilizing solution.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    from ctrlsys import sb02md

    n = 3

    A = np.array([
        [-0.5, 0.2, 0.0],
        [0.1, -0.4, 0.1],
        [0.0, 0.1, -0.6]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')
    Q_orig = Q.copy()

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'U', n, A.copy(), G, Q.copy())

    assert info == 0

    residual = Q_orig + A.T @ X + X @ A - X @ G @ X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb02md_n_zero():
    """
    Test SB02MD quick return for n=0.
    """
    from ctrlsys import sb02md

    n = 0

    A = np.zeros((1, 1), dtype=float, order='F')
    G = np.zeros((1, 1), dtype=float, order='F')
    Q = np.zeros((1, 1), dtype=float, order='F')

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A, G, Q)

    assert info == 0
    assert rcond == 1.0


def test_sb02md_error_singular_u11():
    """
    Test SB02MD info=4 or info=5: insufficient stable eigenvalues or singular U11.

    This occurs when the system doesn't have a unique stabilizing solution.
    For zero matrices, Hamiltonian has no stable eigenvalues (info=4).
    """
    from ctrlsys import sb02md

    n = 2

    A = np.zeros((n, n), dtype=float, order='F')
    G = np.zeros((n, n), dtype=float, order='F')
    Q = np.zeros((n, n), dtype=float, order='F')

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A.copy(), G, Q)

    assert info in [4, 5], f"Expected info=4 or 5, got {info}"


def test_sb02md_error_invalid_dico():
    """
    Test SB02MD error handling: invalid DICO parameter.
    """
    from ctrlsys import sb02md

    n = 2
    A = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02md('X', 'D', 'U', 'N', 'S', n, A, G, Q)


def test_sb02md_error_invalid_hinv():
    """
    Test SB02MD error handling: invalid HINV parameter.
    """
    from ctrlsys import sb02md

    n = 2
    A = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02md('D', 'X', 'U', 'N', 'S', n, A, G, Q)


def test_sb02md_error_invalid_uplo():
    """
    Test SB02MD error handling: invalid UPLO parameter.
    """
    from ctrlsys import sb02md

    n = 2
    A = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02md('C', 'D', 'X', 'N', 'S', n, A, G, Q)


def test_sb02md_error_invalid_scal():
    """
    Test SB02MD error handling: invalid SCAL parameter.
    """
    from ctrlsys import sb02md

    n = 2
    A = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02md('C', 'D', 'U', 'X', 'S', n, A, G, Q)


def test_sb02md_error_invalid_sort():
    """
    Test SB02MD error handling: invalid SORT parameter.
    """
    from ctrlsys import sb02md

    n = 2
    A = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')

    with pytest.raises(ValueError, match="[Pp]arameter"):
        sb02md('C', 'D', 'U', 'N', 'X', n, A, G, Q)


def test_sb02md_error_negative_n():
    """
    Test SB02MD error handling: negative N.
    """
    from ctrlsys import sb02md

    A = np.eye(1, dtype=float, order='F')
    G = np.eye(1, dtype=float, order='F')
    Q = np.eye(1, dtype=float, order='F')

    with pytest.raises(ValueError, match="[Nn]"):
        sb02md('C', 'D', 'U', 'N', 'S', -1, A, G, Q)


def test_sb02md_schur_form_structure():
    """
    Validate Schur form structure: S should be quasi-triangular.

    S = [ S11  S12 ]
        [  0   S22 ]

    where S11, S12, S22 are N-by-N blocks.

    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    from ctrlsys import sb02md

    n = 4

    A = np.array([
        [-0.5, 0.1, 0.0, 0.0],
        [0.1, -0.4, 0.2, 0.0],
        [0.0, 0.1, -0.6, 0.1],
        [0.0, 0.0, 0.1, -0.3]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A.copy(), G, Q)

    assert info == 0
    assert S.shape == (2 * n, 2 * n)
    assert U.shape == (2 * n, 2 * n)

    S21 = S[n:, :n]
    np.testing.assert_allclose(S21, np.zeros((n, n)), atol=1e-12)


def test_sb02md_orthogonal_transformation():
    """
    Validate U is orthogonal: U'*U = I.

    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)
    from ctrlsys import sb02md

    n = 3

    A = np.array([
        [-0.5, 0.2, 0.1],
        [0.1, -0.4, 0.2],
        [0.0, 0.1, -0.6]
    ], dtype=float, order='F')

    G_half = np.random.randn(n, n)
    G = (G_half.T @ G_half + np.eye(n)).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = (Q_half.T @ Q_half + np.eye(n)).astype(float, order='F')

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A.copy(), G, Q)

    assert info == 0

    I_2n = np.eye(2 * n)
    np.testing.assert_allclose(U.T @ U, I_2n, atol=1e-13)
    np.testing.assert_allclose(U @ U.T, I_2n, atol=1e-13)
