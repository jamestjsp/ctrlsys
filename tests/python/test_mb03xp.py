"""
Tests for MB03XP: Periodic Schur decomposition of A*B product.

Computes Q' * A * Z = S and Z' * B * Q = T where:
- A is upper Hessenberg
- B is upper triangular
- S is real Schur form (quasi-triangular)
- T is upper triangular

Uses numpy only - no scipy.
"""

import numpy as np


def _make_hessenberg_triangular(n, rng_seed):
    """Generate upper Hessenberg A and upper triangular B."""
    np.random.seed(rng_seed)
    A = np.triu(np.random.randn(n, n), k=-1).astype(float, order='F')
    B = np.triu(np.random.randn(n, n)).astype(float, order='F')
    return A, B


def _verify_decomposition(A_orig, B_orig, S, T, Q, Z, n, rtol=1e-12):
    """Verify Q'*A*Z = S and Z'*B*Q = T, orthogonality, and Schur structure."""
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(Z.T @ Z, np.eye(n), rtol=1e-13, atol=1e-14)

    np.testing.assert_allclose(Q.T @ A_orig @ Z, S, rtol=rtol, atol=1e-13)
    np.testing.assert_allclose(Z.T @ B_orig @ Q, T, rtol=rtol, atol=1e-13)

    # T must be upper triangular
    np.testing.assert_allclose(np.tril(T, -1), 0, atol=1e-13)

    # S must be quasi-triangular (only 2x2 blocks on diagonal allowed)
    for j in range(n - 2):
        if abs(S[j + 1, j]) > 1e-13:
            assert abs(S[j + 2, j + 1]) < 1e-13, f"S has >2x2 block at row {j}"


def _verify_eigenvalues(A_orig, B_orig, alphar, alphai, beta, n, rtol=1e-10):
    """Verify eigenvalues match product A*B."""
    AB = A_orig @ B_orig
    eigs_ref = np.linalg.eigvals(AB)
    eigs_ref_sorted = np.sort_complex(eigs_ref)

    eigs_slicot = np.array(
        [beta[j] * (alphar[j] + 1j * alphai[j]) for j in range(n)]
    )
    eigs_slicot_sorted = np.sort_complex(eigs_slicot)

    np.testing.assert_allclose(
        np.abs(eigs_slicot_sorted),
        np.abs(eigs_ref_sorted),
        rtol=rtol,
        atol=1e-12,
    )


def test_mb03xp_n4_basic():
    """
    Test MB03XP with N=4 basic case. Verify transformations and eigenvalues.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03xp

    n = 4
    A, B = _make_hessenberg_triangular(n, 42)
    A_orig = A.copy()
    B_orig = B.copy()

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "S", "I", "I", n, 1, n, A, B
    )

    assert info == 0
    _verify_decomposition(A_orig, B_orig, S, T, Q, Z, n)
    _verify_eigenvalues(A_orig, B_orig, alphar, alphai, beta, n)


def test_mb03xp_n10_transformation_residual():
    """
    Test MB03XP with N=10. Verify Q'*A*Z = S and Z'*B*Q = T.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03xp

    n = 10
    A, B = _make_hessenberg_triangular(n, 123)
    A_orig = A.copy()
    B_orig = B.copy()

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "S", "I", "I", n, 1, n, A, B
    )

    assert info == 0
    _verify_decomposition(A_orig, B_orig, S, T, Q, Z, n)
    _verify_eigenvalues(A_orig, B_orig, alphar, alphai, beta, n)


def test_mb03xp_eigenvalue_only():
    """
    Test MB03XP with JOB='E' (eigenvalues only). Verify eigenvalue correctness.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03xp

    n = 4
    A, B = _make_hessenberg_triangular(n, 42)
    A_orig = A.copy()
    B_orig = B.copy()

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "E", "N", "N", n, 1, n, A, B
    )

    assert info == 0
    _verify_eigenvalues(A_orig, B_orig, alphar, alphai, beta, n)


def test_mb03xp_large_multishift():
    """
    Test MB03XP with N=60 to trigger multishift QZ path.
    UE01MD returns NS>2 and MAXB<NH for large enough matrices.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import mb03xp

    n = 60
    A, B = _make_hessenberg_triangular(n, 555)
    A_orig = A.copy()
    B_orig = B.copy()

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "S", "I", "I", n, 1, n, A, B
    )

    assert info == 0
    _verify_decomposition(A_orig, B_orig, S, T, Q, Z, n, rtol=1e-10)
    _verify_eigenvalues(A_orig, B_orig, alphar, alphai, beta, n, rtol=1e-8)


def test_mb03xp_large_multishift_eigenvalue_only():
    """
    Test MB03XP JOB='E' with N=60 to trigger multishift path.

    Random seed: 666 (for reproducibility)
    """
    from ctrlsys import mb03xp

    n = 60
    A, B = _make_hessenberg_triangular(n, 666)
    A_orig = A.copy()
    B_orig = B.copy()

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "E", "N", "N", n, 1, n, A, B
    )

    assert info == 0
    _verify_eigenvalues(A_orig, B_orig, alphar, alphai, beta, n, rtol=1e-8)


def test_mb03xp_partial_ilo_ihi():
    """
    Test MB03XP with partial ILO/IHI range (pre-deflated matrix).

    Random seed: 777 (for reproducibility)
    """
    from ctrlsys import mb03xp

    n = 8
    np.random.seed(777)

    A = np.zeros((n, n), order="F")
    B = np.zeros((n, n), order="F")

    ilo = 3
    ihi = 6

    for idx in range(ilo - 1):
        A[idx, idx] = np.random.randn()
        B[idx, idx] = abs(np.random.randn()) + 0.1
    for idx in range(ihi, n):
        A[idx, idx] = np.random.randn()
        B[idx, idx] = abs(np.random.randn()) + 0.1

    m = ihi - ilo + 1
    A_active = np.triu(np.random.randn(m, m), k=-1)
    B_active = np.triu(np.random.randn(m, m))
    A[ilo - 1 : ihi, ilo - 1 : ihi] = A_active
    B[ilo - 1 : ihi, ilo - 1 : ihi] = B_active

    for idx in range(ilo - 1):
        for jdx in range(ilo - 1, n):
            A[idx, jdx] = np.random.randn()
            B[idx, jdx] = np.random.randn()

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "S", "I", "I", n, ilo, ihi, A, B
    )

    assert info == 0
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(Z.T @ Z, np.eye(n), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.tril(T, -1), 0, atol=1e-13)


def test_mb03xp_n5_transformation():
    """
    Test MB03XP with N=5. Verify transformations and eigenvalues.

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import mb03xp

    n = 5
    A, B = _make_hessenberg_triangular(n, 888)
    A_orig = A.copy()
    B_orig = B.copy()

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "S", "I", "I", n, 1, n, A, B
    )

    assert info == 0
    _verify_decomposition(A_orig, B_orig, S, T, Q, Z, n)
    _verify_eigenvalues(A_orig, B_orig, alphar, alphai, beta, n)


def test_mb03xp_quick_return_n_zero():
    """Test quick return when N=0."""
    from ctrlsys import mb03xp

    n = 0
    A = np.zeros((1, 1), order="F")
    B = np.zeros((1, 1), order="F")

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "S", "I", "I", n, 1, 0, A, B
    )

    assert info == 0


def test_mb03xp_quick_return_ilo_eq_ihi_plus_1():
    """Test quick return when ILO=IHI+1 (no active block)."""
    from ctrlsys import mb03xp

    n = 4
    A = np.eye(n, order="F")
    B = np.eye(n, order="F")

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "S", "I", "I", n, 5, 4, A, B
    )

    assert info == 0


def test_mb03xp_negative_n():
    """Test error for negative N."""
    from ctrlsys import mb03xp

    A = np.zeros((1, 1), order="F")
    B = np.zeros((1, 1), order="F")

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "S", "I", "I", -1, 1, 0, A, B
    )

    assert info == -4


def test_mb03xp_invalid_job():
    """Test error for invalid JOB parameter."""
    from ctrlsys import mb03xp

    n = 4
    A = np.eye(n, order="F")
    B = np.eye(n, order="F")

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "X", "I", "I", n, 1, 4, A, B
    )

    assert info == -1


def test_mb03xp_invalid_ilo():
    """Test error for invalid ILO."""
    from ctrlsys import mb03xp

    n = 4
    A = np.eye(n, order="F")
    B = np.eye(n, order="F")

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        "S", "I", "I", n, 0, 4, A, B
    )

    assert info == -5
