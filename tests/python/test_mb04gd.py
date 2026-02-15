"""
Tests for mb04gd - RQ factorization with row pivoting.

MB04GD computes an RQ factorization with row pivoting of a real M-by-N
matrix A: P*A = R*Q, where P is a permutation matrix, R is upper triangular
(or trapezoidal), and Q is orthogonal.
"""

import numpy as np
import pytest
from ctrlsys import mb04gd


def test_mb04gd_html_example():
    """
    Test MB04GD using HTML documentation example.

    M=6, N=5 matrix with all free rows (JPVT all zeros).
    Expected row permutations: [2, 4, 6, 3, 1, 5]
    Expected R matrix (upper triangular part) from docs.
    """
    m = 6
    n = 5

    a = np.array([
        [1.0, 2.0, 6.0, 3.0, 5.0],
        [-2.0, -1.0, -1.0, 0.0, -2.0],
        [5.0, 5.0, 1.0, 5.0, 1.0],
        [-2.0, -1.0, -1.0, 0.0, -2.0],
        [4.0, 8.0, 4.0, 20.0, 4.0],
        [-2.0, -1.0, -1.0, 0.0, -2.0],
    ], dtype=float, order='F')

    jpvt = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)

    a_out, jpvt_out, tau, info = mb04gd(a, jpvt)

    assert info == 0, f"Expected info=0, got {info}"

    jpvt_expected = np.array([2, 4, 6, 3, 1, 5], dtype=np.int32)
    np.testing.assert_array_equal(jpvt_out, jpvt_expected)

    k = min(m, n)
    assert tau.shape == (k,)

    r_expected = np.array([
        [0.0000, -1.0517, -1.8646, -1.9712, 1.2374],
        [0.0000, -1.0517, -1.8646, -1.9712, 1.2374],
        [0.0000, -1.0517, -1.8646, -1.9712, 1.2374],
        [0.0000, 0.0000, 4.6768, 0.0466, -7.4246],
        [0.0000, 0.0000, 0.0000, 6.7059, -5.4801],
        [0.0000, 0.0000, 0.0000, 0.0000, -22.6274],
    ], dtype=float, order='F')

    for i in range(m):
        for j in range(n - m + i + 1, n):
            np.testing.assert_allclose(a_out[i, j], r_expected[i, j],
                                       rtol=1e-3, atol=1e-3)


def test_mb04gd_orthogonality():
    """
    Test that Q constructed from reflectors is orthogonal.

    Property: Q'*Q = I and Q*Q' = I
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m = 4
    n = 6

    a = np.random.randn(m, n).astype(float, order='F')
    jpvt = np.zeros(m, dtype=np.int32)

    a_out, jpvt_out, tau, info = mb04gd(a, jpvt)

    assert info == 0

    k = min(m, n)
    q = np.eye(n, dtype=float, order='F')

    for i in range(k - 1, -1, -1):
        v = np.zeros(n)
        nki = n - k + i
        v[nki] = 1.0
        v[:nki] = a_out[m - k + i, :nki]

        h = np.eye(n) - tau[i] * np.outer(v, v)
        q = h @ q

    np.testing.assert_allclose(q @ q.T, np.eye(n), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-12, atol=1e-12)


def test_mb04gd_factorization_correctness():
    """
    Test that P*A = R*Q holds.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m = 5
    n = 7

    a_orig = np.random.randn(m, n).astype(float, order='F')
    a = a_orig.copy()
    jpvt = np.zeros(m, dtype=np.int32)

    a_out, jpvt_out, tau, info = mb04gd(a, jpvt)

    assert info == 0

    k = min(m, n)
    q = np.eye(n, dtype=float, order='F')

    for i in range(k - 1, -1, -1):
        v = np.zeros(n)
        nki = n - k + i
        v[nki] = 1.0
        v[:nki] = a_out[m - k + i, :nki]

        h = np.eye(n) - tau[i] * np.outer(v, v)
        q = h @ q

    r = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(max(0, n - m + i), n):
            r[i, j] = a_out[i, j]

    p = np.zeros((m, m), dtype=float)
    for i in range(m):
        p[i, jpvt_out[i] - 1] = 1.0

    pa = p @ a_orig
    rq = r @ q

    np.testing.assert_allclose(pa, rq, rtol=1e-12, atol=1e-12)


def test_mb04gd_constrained_rows():
    """
    Test with non-free (constrained) rows via non-zero JPVT entries.

    When JPVT(i) != 0, row i is constrained to bottom of P*A.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m = 4
    n = 5

    a_orig = np.random.randn(m, n).astype(float, order='F')
    a = a_orig.copy()
    jpvt = np.array([0, 1, 0, 1], dtype=np.int32)

    a_out, jpvt_out, tau, info = mb04gd(a, jpvt)

    assert info == 0

    constrained_original_rows = {2, 4}
    bottom_original_rows = {jpvt_out[m-1], jpvt_out[m-2]}
    assert constrained_original_rows == bottom_original_rows, \
        f"Constrained rows {constrained_original_rows} not at bottom, got {bottom_original_rows}"


def test_mb04gd_square_matrix():
    """
    Test with square matrix (M == N).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5

    a_orig = np.random.randn(n, n).astype(float, order='F')
    a = a_orig.copy()
    jpvt = np.zeros(n, dtype=np.int32)

    a_out, jpvt_out, tau, info = mb04gd(a, jpvt)

    assert info == 0

    k = n
    q = np.eye(n, dtype=float, order='F')

    for i in range(k - 1, -1, -1):
        v = np.zeros(n)
        nki = n - k + i
        v[nki] = 1.0
        v[:nki] = a_out[n - k + i, :nki]

        h = np.eye(n) - tau[i] * np.outer(v, v)
        q = h @ q

    r = np.triu(a_out)

    p = np.zeros((n, n), dtype=float)
    for i in range(n):
        p[i, jpvt_out[i] - 1] = 1.0

    pa = p @ a_orig
    rq = r @ q

    np.testing.assert_allclose(pa, rq, rtol=1e-12, atol=1e-12)


def test_mb04gd_tall_matrix():
    """
    Test with tall matrix (M > N).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    m = 6
    n = 4

    a_orig = np.random.randn(m, n).astype(float, order='F')
    a = a_orig.copy()
    jpvt = np.zeros(m, dtype=np.int32)

    a_out, jpvt_out, tau, info = mb04gd(a, jpvt)

    assert info == 0

    k = min(m, n)
    assert tau.shape == (k,)

    q = np.eye(n, dtype=float, order='F')

    for i in range(k - 1, -1, -1):
        v = np.zeros(n)
        nki = n - k + i
        v[nki] = 1.0
        v[:nki] = a_out[m - k + i, :nki]

        h = np.eye(n) - tau[i] * np.outer(v, v)
        q = h @ q

    r = np.zeros((m, n), dtype=float)
    for i in range(m):
        start_j = max(0, i - (m - n))
        for j in range(start_j, n):
            r[i, j] = a_out[i, j]

    p = np.zeros((m, m), dtype=float)
    for i in range(m):
        p[i, jpvt_out[i] - 1] = 1.0

    pa = p @ a_orig
    rq = r @ q

    np.testing.assert_allclose(pa, rq, rtol=1e-12, atol=1e-12)


def test_mb04gd_zero_dimension():
    """
    Test with M=0 (quick return case).
    """
    m = 0
    n = 5

    a = np.array([], dtype=float, order='F').reshape(0, n)
    jpvt = np.array([], dtype=np.int32)

    a_out, jpvt_out, tau, info = mb04gd(a, jpvt)

    assert info == 0
    assert tau.shape == (0,)


def test_mb04gd_zero_columns():
    """
    Test with N=0 (quick return case).
    """
    m = 5
    n = 0

    a = np.array([], dtype=float, order='F').reshape(m, 0)
    jpvt = np.zeros(m, dtype=np.int32)

    a_out, jpvt_out, tau, info = mb04gd(a, jpvt)

    assert info == 0
    assert tau.shape == (0,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
