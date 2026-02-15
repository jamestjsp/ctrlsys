"""
Tests for MB02NY - Separation of zero singular value of bidiagonal submatrix.

MB02NY separates a zero singular value of a bidiagonal submatrix by annihilating
superdiagonal elements using Givens rotations.
"""

import numpy as np
import pytest
from ctrlsys import mb02ny


def test_mb02ny_basic_annihilate_e_i():
    """
    Test MB02NY annihilating E(i) when I < K.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    m, n = 5, 5
    p = min(m, n)
    i_idx = 2  # Q(2) is negligible (1-based index)
    k = 4      # Consider submatrix up to index 4

    # Bidiagonal matrix diagonal entries
    q = np.array([2.0, 1.5, 0.8, 1.2, 0.9], dtype=np.float64, order='F')

    # Superdiagonal entries
    e = np.array([0.5, 0.3, 0.7, 0.4], dtype=np.float64, order='F')

    # Identity for U (m x p)
    u = np.eye(m, p, dtype=np.float64, order='F')

    # Identity for V (n x p)
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(True, True, m, n, i_idx, k, q, e, u, v)

    # Q(i) should be set to zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # E(i) should be annihilated if i < k (i_idx=2 < k=4)
    assert abs(e_out[i_idx - 1]) < 1e-14

    # U and V should be orthogonal (since we started with identity and applied Givens)
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-14, atol=1e-14)


def test_mb02ny_basic_annihilate_e_im1():
    """
    Test MB02NY annihilating E(i-1) when I > 1.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    m, n = 4, 4
    p = min(m, n)
    i_idx = 3  # Q(3) is negligible (1-based index)
    k = 3      # Consider submatrix up to index 3 (so i == k, only E(i-1) annihilated)

    q = np.array([1.5, 1.2, 0.9, 0.7], dtype=np.float64, order='F')
    e = np.array([0.4, 0.6, 0.3], dtype=np.float64, order='F')

    u = np.eye(m, p, dtype=np.float64, order='F')
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(True, True, m, n, i_idx, k, q, e, u, v)

    # Q(i) should be set to zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # E(i-1) should be annihilated since i > 1 (i_idx=3 > 1)
    assert abs(e_out[i_idx - 2]) < 1e-14

    # U and V should remain orthogonal
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-14, atol=1e-14)


def test_mb02ny_both_annihilations():
    """
    Test MB02NY annihilating both E(i-1) and E(i).

    When 1 < i < k, both E(i-1) and E(i) should be annihilated.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    m, n = 6, 6
    p = min(m, n)
    i_idx = 3  # Q(3) is negligible
    k = 5      # k > i, so both E(i-1) and E(i) are in scope

    q = np.array([2.5, 1.8, 1.3, 1.6, 1.1, 0.8], dtype=np.float64, order='F')
    e = np.array([0.7, 0.5, 0.9, 0.4, 0.6], dtype=np.float64, order='F')

    u = np.eye(m, p, dtype=np.float64, order='F')
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(True, True, m, n, i_idx, k, q, e, u, v)

    # Q(i) should be zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # E(i-1) should be annihilated (i > 1)
    assert abs(e_out[i_idx - 2]) < 1e-14

    # E(i) should be annihilated (i < k)
    assert abs(e_out[i_idx - 1]) < 1e-14

    # U and V should remain orthogonal
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-14, atol=1e-14)


def test_mb02ny_no_u_update():
    """
    Test MB02NY with UPDATU=False.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    m, n = 4, 4
    p = min(m, n)
    i_idx = 2
    k = 4

    q = np.array([1.5, 1.2, 0.9, 0.7], dtype=np.float64, order='F')
    e = np.array([0.4, 0.6, 0.3], dtype=np.float64, order='F')

    # U should not be modified, V should
    u = np.eye(m, p, dtype=np.float64, order='F')
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(False, True, m, n, i_idx, k, q, e, u, v)

    # Q(i) should be zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # Check V is orthogonal
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-14, atol=1e-14)


def test_mb02ny_no_v_update():
    """
    Test MB02NY with UPDATV=False.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    m, n = 4, 4
    p = min(m, n)
    i_idx = 2
    k = 4

    q = np.array([1.5, 1.2, 0.9, 0.7], dtype=np.float64, order='F')
    e = np.array([0.4, 0.6, 0.3], dtype=np.float64, order='F')

    u = np.eye(m, p, dtype=np.float64, order='F')
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(True, False, m, n, i_idx, k, q, e, u, v)

    # Q(i) should be zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # E(i) should be annihilated (i < k)
    assert abs(e_out[i_idx - 1]) < 1e-14

    # Check U is orthogonal
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-14, atol=1e-14)


def test_mb02ny_neither_update():
    """
    Test MB02NY with UPDATU=False and UPDATV=False.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    m, n = 4, 4
    p = min(m, n)
    i_idx = 2
    k = 4

    q = np.array([1.5, 1.2, 0.9, 0.7], dtype=np.float64, order='F')
    e = np.array([0.4, 0.6, 0.3], dtype=np.float64, order='F')

    u = np.eye(m, p, dtype=np.float64, order='F')
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(False, False, m, n, i_idx, k, q, e, u, v)

    # Q(i) should be zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # E(i) should be annihilated (i < k)
    assert abs(e_out[i_idx - 1]) < 1e-14

    # E(i-1) should be annihilated (i > 1)
    assert abs(e_out[i_idx - 2]) < 1e-14


def test_mb02ny_quick_return_m_zero():
    """
    Test MB02NY with M=0 (quick return).
    """
    m, n = 0, 4
    p = min(m, n) if m > 0 else 1  # Avoid zero-size arrays
    i_idx = 1
    k = 1

    q = np.array([1.0], dtype=np.float64, order='F')
    e = np.array([0.5], dtype=np.float64, order='F')
    u = np.zeros((1, 1), dtype=np.float64, order='F')
    v = np.eye(n, 1, dtype=np.float64, order='F')

    # Quick return - Q and E unchanged
    q_out, e_out, u_out, v_out = mb02ny(True, True, m, n, i_idx, k, q, e, u, v)

    # Arrays should be unchanged due to quick return
    np.testing.assert_allclose(q_out, q, rtol=1e-14)
    np.testing.assert_allclose(e_out, e, rtol=1e-14)


def test_mb02ny_quick_return_n_zero():
    """
    Test MB02NY with N=0 (quick return).
    """
    m, n = 4, 0
    p = 1  # Avoid zero-size arrays
    i_idx = 1
    k = 1

    q = np.array([1.0], dtype=np.float64, order='F')
    e = np.array([0.5], dtype=np.float64, order='F')
    u = np.eye(m, 1, dtype=np.float64, order='F')
    v = np.zeros((1, 1), dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(True, True, m, n, i_idx, k, q, e, u, v)

    # Arrays should be unchanged due to quick return
    np.testing.assert_allclose(q_out, q, rtol=1e-14)
    np.testing.assert_allclose(e_out, e, rtol=1e-14)


def test_mb02ny_i_equals_1():
    """
    Test MB02NY when I=1 (only E(i) annihilated, not E(i-1)).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)

    m, n = 4, 4
    p = min(m, n)
    i_idx = 1  # Q(1) is negligible
    k = 3

    q = np.array([1.5, 1.2, 0.9, 0.7], dtype=np.float64, order='F')
    e = np.array([0.4, 0.6, 0.3], dtype=np.float64, order='F')

    u = np.eye(m, p, dtype=np.float64, order='F')
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(True, True, m, n, i_idx, k, q, e, u, v)

    # Q(i) should be zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # E(i) should be annihilated (i < k)
    assert abs(e_out[i_idx - 1]) < 1e-14

    # E(i-1) = E(0) doesn't exist, so nothing else to check for annihilation

    # U and V should remain orthogonal
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-14, atol=1e-14)


def test_mb02ny_i_equals_k():
    """
    Test MB02NY when I=K (only E(i-1) annihilated, not E(i)).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)

    m, n = 4, 4
    p = min(m, n)
    i_idx = 4  # Q(4) is negligible, and K=4, so i == k
    k = 4

    q = np.array([1.5, 1.2, 0.9, 0.7], dtype=np.float64, order='F')
    e = np.array([0.4, 0.6, 0.3], dtype=np.float64, order='F')

    u = np.eye(m, p, dtype=np.float64, order='F')
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(True, True, m, n, i_idx, k, q, e, u, v)

    # Q(i) should be zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # E(i-1) = E(3) should be annihilated since i > 1
    assert abs(e_out[i_idx - 2]) < 1e-14

    # E(i) would be E(4) but i==k, so not annihilated (not in scope)

    # U and V should remain orthogonal
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-14, atol=1e-14)


def test_mb02ny_non_square():
    """
    Test MB02NY with non-square dimensions (M != N).

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)

    m, n = 6, 4
    p = min(m, n)
    i_idx = 2
    k = 3

    q = np.array([2.0, 1.5, 1.0, 0.8], dtype=np.float64, order='F')
    e = np.array([0.5, 0.4, 0.3], dtype=np.float64, order='F')

    u = np.eye(m, p, dtype=np.float64, order='F')
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(True, True, m, n, i_idx, k, q, e, u, v)

    # Q(i) should be zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # E(i-1) should be annihilated
    assert abs(e_out[i_idx - 2]) < 1e-14

    # E(i) should be annihilated (i < k)
    assert abs(e_out[i_idx - 1]) < 1e-14

    # U should be M x p orthonormal columns
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-14, atol=1e-14)

    # V should be N x p orthonormal columns
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-14, atol=1e-14)


def test_mb02ny_bidiagonal_structure_preservation():
    """
    Test that MB02NY preserves bidiagonal structure (zeros below diagonal).

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)

    m, n = 5, 5
    p = min(m, n)
    i_idx = 3
    k = 4

    q = np.array([3.0, 2.5, 2.0, 1.5, 1.0], dtype=np.float64, order='F')
    e = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float64, order='F')

    u = np.eye(m, p, dtype=np.float64, order='F')
    v = np.eye(n, p, dtype=np.float64, order='F')

    q_out, e_out, u_out, v_out = mb02ny(True, True, m, n, i_idx, k, q, e, u, v)

    # Construct full bidiagonal matrix from q_out and e_out
    j_out = np.diag(q_out) + np.diag(e_out, k=1)

    # Verify it remains bidiagonal (no elements below diagonal)
    lower = np.tril(j_out, k=-1)
    assert np.allclose(lower, 0.0, atol=1e-14)

    # Verify transformation matrices are orthogonal
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-14, atol=1e-14)

    # Verify Q(i) was set to zero
    assert abs(q_out[i_idx - 1]) < 1e-14

    # Verify E(i-1) and E(i) were annihilated
    assert abs(e_out[i_idx - 2]) < 1e-14  # E(i-1)
    assert abs(e_out[i_idx - 1]) < 1e-14  # E(i)
