"""
Tests for MB04FP: Eigenvalues and orthogonal decomposition of a real
skew-Hamiltonian/skew-Hamiltonian pencil (panel-based version).

MB04FP is a panel-based version of MB04FD that applies transformations
on panels of columns for better performance on large matrices.

For small matrices (M <= 32) or when panel size equals M, MB04FP
delegates to MB04FD.

Tests:
1. Basic case with larger matrix (N=64) to exercise panel-based code
2. Comparison with MB04FD for same input (should produce same eigenvalues)
3. Edge case: N=0 (quick return)
4. Mathematical property: orthogonality of Q matrix
5. Error handling: invalid N (odd)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def make_skew_symmetric_storage(m):
    """Create DE/FG matrices with proper skew-symmetric storage format.

    Random seed should be set before calling.

    DE/FG format: (m, m+1)
    - Column 0: strictly lower triangular of E/G (skew-symmetric)
    - Columns 1 to m: strictly upper triangular of D/F (skew-symmetric)
    """
    de = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            de[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            de[i, j] = np.random.randn()
    return de


def test_mb04fp_basic_panel():
    """
    Test MB04FP with N=64 to exercise panel-based code path.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04fp

    np.random.seed(42)
    n = 64
    m = n // 2

    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    de = make_skew_symmetric_storage(m)
    fg = make_skew_symmetric_storage(m)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fp(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04FP returned info={info}"

    assert len(alphar) == m
    assert len(alphai) == m
    assert len(beta) == m

    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb04fp_matches_mb04fd():
    """
    Validate MB04FP produces same eigenvalues as MB04FD.

    The eigenvalue ordering may differ, but sorted eigenvalues should match.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04fd, mb04fp

    np.random.seed(123)
    n = 64
    m = n // 2

    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    de = make_skew_symmetric_storage(m)
    fg = make_skew_symmetric_storage(m)

    _, _, _, _, _, alphar_fd, alphai_fd, beta_fd, _, info_fd = mb04fd(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )
    assert info_fd == 0

    _, _, _, _, _, alphar_fp, alphai_fp, beta_fp, _, info_fp = mb04fp(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )
    assert info_fp == 0

    eig_fd = np.sort(alphar_fd / np.where(beta_fd != 0, beta_fd, 1.0))
    eig_fp = np.sort(alphar_fp / np.where(beta_fp != 0, beta_fp, 1.0))

    mask_fd = np.isfinite(eig_fd)
    mask_fp = np.isfinite(eig_fp)

    assert_allclose(eig_fp[mask_fp], eig_fd[mask_fd], rtol=1e-10, atol=1e-12)


def test_mb04fp_small_delegates_to_mb04fd():
    """
    For small matrices (M <= 32), MB04FP delegates to MB04FD.

    Use N=8 (M=4) which should go through MB04FD path.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04fd, mb04fp

    np.random.seed(456)
    n = 8
    m = n // 2

    a = np.array([
        [0.8147, 0.6323, 0.9575, 0.9571],
        [0.9057, 0.0975, 0.9648, 0.4853],
        [0.1269, 0.2784, 0.1576, 0.8002],
        [0.9133, 0.5468, 0.9705, 0.1418]
    ], order='F', dtype=float)

    de = np.array([
        [0.4217, 0.6557, 0.6787, 0.6554, 0.2769],
        [0.9157, 0.0357, 0.7577, 0.1711, 0.0461],
        [0.7922, 0.8491, 0.7431, 0.7060, 0.0971],
        [0.9594, 0.9339, 0.3922, 0.0318, 0.8234]
    ], order='F', dtype=float)

    b = np.array([
        [0.6948, 0.4387, 0.1868, 0.7093],
        [0.3170, 0.3815, 0.4897, 0.7546],
        [0.9502, 0.7655, 0.4455, 0.2760],
        [0.0344, 0.7951, 0.6463, 0.6797]
    ], order='F', dtype=float)

    fg = np.array([
        [0.6550, 0.9597, 0.7512, 0.8909, 0.1492],
        [0.1626, 0.3403, 0.2550, 0.9592, 0.2575],
        [0.1189, 0.5852, 0.5059, 0.5472, 0.8407],
        [0.4983, 0.2238, 0.6990, 0.1386, 0.2542]
    ], order='F', dtype=float)

    _, _, _, _, _, alphar_fd, alphai_fd, beta_fd, _, info_fd = mb04fd(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )
    assert info_fd == 0

    _, _, _, _, _, alphar_fp, alphai_fp, beta_fp, _, info_fp = mb04fp(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )
    assert info_fp == 0

    eig_fd = np.sort(alphar_fd / np.where(beta_fd != 0, beta_fd, 1.0))
    eig_fp = np.sort(alphar_fp / np.where(beta_fp != 0, beta_fp, 1.0))
    assert_allclose(eig_fp, eig_fd, rtol=1e-10, atol=1e-12)


def test_mb04fp_q_orthogonality():
    """
    Validate mathematical property: Q should be orthogonal (Q'*Q = I).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04fp

    np.random.seed(789)
    n = 80
    m = n // 2

    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    de = make_skew_symmetric_storage(m)
    fg = make_skew_symmetric_storage(m)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fp(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04FP returned info={info}"

    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)

    qqt = q_out @ q_out.T
    assert_allclose(qqt, np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb04fp_eigenvalues_only():
    """
    Test JOB='E' mode: compute eigenvalues only.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import mb04fp

    np.random.seed(555)
    n = 64
    m = n // 2

    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    de = make_skew_symmetric_storage(m)
    fg = make_skew_symmetric_storage(m)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fp(
        'E', 'N', a.copy(), de.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04FP returned info={info}"
    assert len(alphar) == m
    assert len(alphai) == m
    assert len(beta) == m


def test_mb04fp_n_zero():
    """
    Edge case: N=0 should return immediately with info=0.
    """
    from ctrlsys import mb04fp

    n = 0
    a = np.array([], order='F', dtype=float).reshape(0, 0)
    de = np.array([], order='F', dtype=float).reshape(0, 1)
    b = np.array([], order='F', dtype=float).reshape(0, 0)
    fg = np.array([], order='F', dtype=float).reshape(0, 1)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fp(
        'T', 'I', a, de, b, fg
    )

    assert info == 0, f"MB04FP returned info={info}"
    assert len(alphar) == 0
    assert len(alphai) == 0
    assert len(beta) == 0


def test_mb04fp_invalid_n_odd():
    """
    Test error handling: N must be even. N=5 should return info=-3.
    """
    from ctrlsys import mb04fp

    np.random.seed(888)
    n = 5
    m = n // 2 + 1
    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    de = np.zeros((m, m + 1), order='F', dtype=float)
    fg = np.zeros((m, m + 1), order='F', dtype=float)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fp(
        'T', 'I', a, de, b, fg, n=n
    )

    assert info == -3, f"Expected info=-3 for odd N, got info={info}"


def test_mb04fp_compq_update():
    """
    Test COMPQ='U' mode: update existing Q0 matrix.

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb04fp

    np.random.seed(999)
    n = 64
    m = n // 2

    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    de = make_skew_symmetric_storage(m)
    fg = make_skew_symmetric_storage(m)

    q0 = np.eye(n, order='F', dtype=float)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fp(
        'T', 'U', a.copy(), de.copy(), b.copy(), fg.copy(), q=q0.copy()
    )

    assert info == 0, f"MB04FP returned info={info}"

    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb04fp_eigenvalue_conjugate_pairs():
    """
    Validate complex eigenvalues come in conjugate pairs.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04fp

    np.random.seed(111)
    n = 64
    m = n // 2

    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    de = make_skew_symmetric_storage(m)
    fg = make_skew_symmetric_storage(m)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fp(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04FP returned info={info}"

    j = 0
    while j < m:
        if alphai[j] != 0:
            assert j + 1 < m, "Complex eigenvalue pair incomplete"
            assert_allclose(alphar[j], alphar[j + 1], rtol=1e-14)
            assert_allclose(alphai[j], -alphai[j + 1], rtol=1e-14)
            j += 2
        else:
            j += 1
