"""
Tests for SB02OY - Extended Hamiltonian/symplectic matrix pair construction.

Constructs extended matrix pairs for algebraic Riccati equations:
  - Discrete-time: |A 0 B|   |E 0 0|
                   |Q -E' L| - z|0 -A' 0|
                   |L' 0 R|    |0 -B' 0|

  - Continuous-time: |A 0 B|   |E 0 0|
                     |Q A' L| - s|0 -E' 0|
                     |L' B' R|    |0 0 0|

Then compresses to 2N-by-2N using QL factorization.

For JOBB='G', directly constructs 2N-by-2N pairs using G = B*R^(-1)*B'.
"""

import numpy as np
import pytest


def test_sb02oy_discrete_jobb_B():
    """
    Test SB02OY for discrete-time with JOBB='B' (B and R given).

    Constructs extended pencil and compresses to 2N-by-2N.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    L = np.zeros((n, m), dtype=float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'O', 'D', 'B', 'N', 'U', 'Z', 'I',
        n, m, 0, A, B, Q, R, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"

    assert AF.shape == (2*n, 2*n), f"AF shape mismatch: {AF.shape}"
    assert BF.shape == (2*n, 2*n), f"BF shape mismatch: {BF.shape}"

    assert rcond > 0, f"rcond should be positive, got {rcond}"


def test_sb02oy_continuous_jobb_B():
    """
    Test SB02OY for continuous-time with JOBB='B'.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    L = np.zeros((n, m), dtype=float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'O', 'C', 'B', 'N', 'U', 'Z', 'I',
        n, m, 0, A, B, Q, R, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)
    assert BF.shape == (2*n, 2*n)
    assert rcond > 0


def test_sb02oy_discrete_jobb_G():
    """
    Test SB02OY for discrete-time with JOBB='G' (G matrix given).

    Constructs 2N-by-2N pair directly without compression.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n = 3

    A = np.random.randn(n, n).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    G_half = np.random.randn(n, n)
    G = ((G_half.T @ G_half) + np.eye(n) * 0.5).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, info = sb02oy(
        'O', 'D', 'G', 'N', 'U', 'Z', 'I',
        n, 0, 0, A, G, Q, None, None, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)
    assert BF.shape == (2*n, 2*n)


def test_sb02oy_continuous_jobb_G_identity_E():
    """
    Test SB02OY for continuous-time with JOBB='G' and E=I.

    For this case, BF is not referenced (only AF returned).
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n = 3

    A = np.random.randn(n, n).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    G_half = np.random.randn(n, n)
    G = ((G_half.T @ G_half) + np.eye(n) * 0.5).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, info = sb02oy(
        'O', 'C', 'G', 'N', 'U', 'Z', 'I',
        n, 0, 0, A, G, Q, None, None, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)

    np.testing.assert_allclose(AF[:n, :n], A, rtol=1e-14)

    np.testing.assert_allclose(AF[:n, n:], -G, rtol=1e-14)

    np.testing.assert_allclose(AF[n:, n:], A.T, rtol=1e-14)


def test_sb02oy_with_nonzero_L():
    """
    Test SB02OY with nonzero cross-weighting matrix L (JOBL='N').

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    L = np.random.randn(n, m).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'O', 'D', 'B', 'N', 'U', 'N', 'I',
        n, m, 0, A, B, Q, R, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)
    assert BF.shape == (2*n, 2*n)
    assert rcond > 0


def test_sb02oy_with_general_E():
    """
    Test SB02OY with general (non-identity) E matrix (JOBE='N').

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    L = np.zeros((n, m), dtype=float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F') + 0.1 * np.random.randn(n, n).astype(float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'O', 'D', 'B', 'N', 'U', 'Z', 'N',
        n, m, 0, A, B, Q, R, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)
    assert BF.shape == (2*n, 2*n)
    assert rcond > 0


def test_sb02oy_factored_Q():
    """
    Test SB02OY with factored Q = C'C (FACT='C').

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)

    n, m, p = 3, 2, 4

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    C = np.random.randn(p, n).astype(float, order='F')
    L = np.zeros((n, m), dtype=float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'O', 'D', 'B', 'C', 'U', 'Z', 'I',
        n, m, p, A, B, C, R, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)
    assert BF.shape == (2*n, 2*n)


def test_sb02oy_factored_R():
    """
    Test SB02OY with factored R = D'D (FACT='D').

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)

    n, m, p = 3, 2, 4

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    D = np.random.randn(p, m).astype(float, order='F')
    L = np.zeros((n, m), dtype=float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'O', 'D', 'B', 'D', 'U', 'Z', 'I',
        n, m, p, A, B, Q, D, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)
    assert BF.shape == (2*n, 2*n)


def test_sb02oy_factored_both():
    """
    Test SB02OY with both Q=C'C and R=D'D (FACT='B').

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)

    n, m, p = 3, 2, 4

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    C = np.random.randn(p, n).astype(float, order='F')
    D = np.random.randn(p, m).astype(float, order='F')
    L = np.zeros((n, m), dtype=float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'O', 'D', 'B', 'B', 'U', 'Z', 'I',
        n, m, p, A, B, C, D, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)
    assert BF.shape == (2*n, 2*n)


def test_sb02oy_spectral_factorization():
    """
    Test SB02OY for spectral factorization (TYPE='S').

    For TYPE='S', the B parameter contains C' (N-by-P), not B.
    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)

    n, m, p = 3, 2, 2

    A = np.random.randn(n, n).astype(float, order='F')
    C = np.random.randn(p, n).astype(float, order='F')
    Ct = np.asfortranarray(C.T)  # C' is N-by-P for spectral factorization
    L = np.zeros((n, m), dtype=float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'S', 'D', 'B', 'N', 'U', 'Z', 'I',
        n, m, p, A, Ct, Q, R, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)
    assert BF.shape == (2*n, 2*n)


def test_sb02oy_lower_triangle():
    """
    Test SB02OY with lower triangle storage (UPLO='L').

    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    L = np.zeros((n, m), dtype=float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'O', 'D', 'B', 'N', 'L', 'Z', 'I',
        n, m, 0, A, B, Q, R, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    assert AF.shape == (2*n, 2*n)
    assert BF.shape == (2*n, 2*n)


def test_sb02oy_zero_n():
    """
    Test SB02OY with N=0 (quick return).
    """
    n, m = 0, 2

    A = np.zeros((1, 1), dtype=float, order='F')
    B = np.zeros((1, m), dtype=float, order='F')
    Q = np.zeros((1, 1), dtype=float, order='F')
    R = np.eye(m, dtype=float, order='F')
    L = np.zeros((1, m), dtype=float, order='F')
    E = np.zeros((1, 1), dtype=float, order='F')

    from ctrlsys import sb02oy

    result = sb02oy(
        'O', 'D', 'B', 'N', 'U', 'Z', 'I',
        n, m, 0, A, B, Q, R, L, E, 0.0
    )

    if len(result) == 4:
        AF, BF, rcond, info = result
    else:
        info = result[-1]

    assert info == 0


def test_sb02oy_rcond_returned():
    """
    Test SB02OY returns a valid reciprocal condition number.

    The rcond value indicates the conditioning of the compressed
    triangular factor after QL factorization.
    """
    np.random.seed(888)

    n, m = 3, 2

    A = np.random.randn(n, n).astype(float, order='F')
    B = np.random.randn(n, m).astype(float, order='F')
    L = np.zeros((n, m), dtype=float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    R_half = np.random.randn(m, m)
    R = (R_half.T @ R_half + np.eye(m) * 3.0).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, rcond, info = sb02oy(
        'O', 'D', 'B', 'N', 'U', 'Z', 'I',
        n, m, 0, A, B, Q, R, L, E, 0.0
    )

    assert info == 0, f"sb02oy failed with info={info}"
    # rcond should be positive and <= 1
    assert 0 < rcond <= 1, f"Expected 0 < rcond <= 1, got {rcond}"


def test_sb02oy_error_invalid_type():
    """
    Test SB02OY error handling: invalid TYPE parameter.
    """
    n, m = 3, 2

    A = np.eye(n, dtype=float, order='F')
    B = np.eye(n, m, dtype=float, order='F')
    Q = np.eye(n, dtype=float, order='F')
    R = np.eye(m, dtype=float, order='F')
    L = np.zeros((n, m), dtype=float, order='F')
    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    with pytest.raises(ValueError, match="TYPE"):
        sb02oy('X', 'D', 'B', 'N', 'U', 'Z', 'I',
               n, m, 0, A, B, Q, R, L, E, 0.0)


def test_sb02oy_error_negative_n():
    """
    Test SB02OY error handling: negative N.
    """
    from ctrlsys import sb02oy

    A = np.eye(1, dtype=float, order='F')
    B = np.eye(1, dtype=float, order='F')
    Q = np.eye(1, dtype=float, order='F')
    R = np.eye(1, dtype=float, order='F')
    L = np.zeros((1, 1), dtype=float, order='F')
    E = np.eye(1, dtype=float, order='F')

    with pytest.raises(ValueError):
        sb02oy('O', 'D', 'B', 'N', 'U', 'Z', 'I',
               -1, 1, 0, A, B, Q, R, L, E, 0.0)


def test_sb02oy_hamiltonian_structure():
    """
    Validate mathematical property: continuous-time Hamiltonian structure.

    For continuous-time with JOBB='G' and E=I, the matrix AF should have
    the Hamiltonian structure:
        AF = [A  -G]
             [Q  A']

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    n = 4

    A = np.random.randn(n, n).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    G_half = np.random.randn(n, n)
    G = ((G_half.T @ G_half) + np.eye(n) * 0.5).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, info = sb02oy(
        'O', 'C', 'G', 'N', 'U', 'Z', 'I',
        n, 0, 0, A, G, Q, None, None, E, 0.0
    )

    assert info == 0

    np.testing.assert_allclose(AF[:n, :n], A, rtol=1e-14)

    G_full = np.triu(G) + np.triu(G, 1).T
    np.testing.assert_allclose(AF[:n, n:], -G_full, rtol=1e-14)

    Q_full = np.triu(Q) + np.triu(Q, 1).T
    np.testing.assert_allclose(AF[n:, :n], Q_full, rtol=1e-14)

    np.testing.assert_allclose(AF[n:, n:], A.T, rtol=1e-14)


def test_sb02oy_symplectic_structure():
    """
    Validate mathematical property: discrete-time symplectic structure.

    For discrete-time with JOBB='G' and E=I, the matrix pair (AF, BF) should
    preserve the symplectic structure needed for computing eigenvalues.

    Random seed: 1010 (for reproducibility)
    """
    np.random.seed(1010)

    n = 3

    A = np.random.randn(n, n).astype(float, order='F')

    Q_half = np.random.randn(n, n)
    Q = ((Q_half.T @ Q_half) + np.eye(n) * 2.0).astype(float, order='F')

    G_half = np.random.randn(n, n)
    G = ((G_half.T @ G_half) + np.eye(n) * 0.5).astype(float, order='F')

    E = np.eye(n, dtype=float, order='F')

    from ctrlsys import sb02oy

    AF, BF, info = sb02oy(
        'O', 'D', 'G', 'N', 'U', 'Z', 'I',
        n, 0, 0, A, G, Q, None, None, E, 0.0
    )

    assert info == 0

    np.testing.assert_allclose(AF[:n, :n], A, rtol=1e-14)
    np.testing.assert_allclose(AF[:n, n:], np.zeros((n, n)), rtol=1e-14)

    Q_full = np.triu(Q) + np.triu(Q, 1).T
    np.testing.assert_allclose(AF[n:, :n], Q_full, rtol=1e-14)

    np.testing.assert_allclose(AF[n:, n:], -np.eye(n), rtol=1e-14)

    np.testing.assert_allclose(BF[:n, :n], np.eye(n), rtol=1e-14)

    G_full = np.triu(G) + np.triu(G, 1).T
    np.testing.assert_allclose(BF[:n, n:], G_full, rtol=1e-14)

    np.testing.assert_allclose(BF[n:, n:], -A.T, rtol=1e-14)
