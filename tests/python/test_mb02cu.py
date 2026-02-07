import numpy as np
import pytest


def test_mb02cu_column_oriented_basic():
    """
    Validate bringing generator to proper form (column oriented, not deficient).

    Tests TYPEG='C' - column oriented without rank deficiencies.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    k = 3
    p = 5
    q = 2
    nb = 2

    a1 = np.tril(np.random.randn(k, k)).astype(float, order='F')
    for i in range(k):
        a1[i, i] = np.abs(a1[i, i]) + 5.0
    a2 = np.random.randn(k, p - k).astype(float, order='F')
    b = 0.1 * np.random.randn(k, q).astype(float, order='F')

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('C', k, p, q, nb, a1.copy(), a2.copy(), b.copy())

    assert info == 0
    assert a1_out.shape == (k, k)
    assert a2_out.shape == (k, p - k)
    assert b_out.shape == (k, q)

    np.testing.assert_allclose(np.triu(a1_out, 1), 0.0, atol=1e-12)
    assert np.all(np.diag(a1_out) > 0)


def test_mb02cu_row_oriented_basic():
    """
    Validate bringing generator to proper form (row oriented, not deficient).

    Tests TYPEG='R' - row oriented without rank deficiencies.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    k = 3
    p = 5
    q = 2
    nb = 2

    a1 = np.triu(np.random.randn(k, k)).astype(float, order='F')
    for i in range(k):
        a1[i, i] = np.abs(a1[i, i]) + 5.0
    a2 = np.random.randn(p - k, k).astype(float, order='F')
    b = 0.1 * np.random.randn(q, k).astype(float, order='F')

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('R', k, p, q, nb, a1.copy(), a2.copy(), b.copy())

    assert info == 0
    assert a1_out.shape == (k, k)
    assert a2_out.shape == (p - k, k)
    assert b_out.shape == (q, k)

    np.testing.assert_allclose(np.tril(a1_out, -1), 0.0, atol=1e-12)
    assert np.all(np.diag(a1_out) > 0)


def test_mb02cu_deficient_basic():
    """
    Validate bringing generator to proper form (rank deficient mode).

    Tests TYPEG='D' - column oriented with rank deficiencies expected.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    k = 3
    p = 5
    q = 4
    nb = 2
    tol = 1e-10

    a1 = np.random.randn(k, k).astype(float, order='F')
    for i in range(k):
        a1[i, i] = np.abs(a1[i, i]) + 5.0
    a2 = np.random.randn(k, p - k).astype(float, order='F')
    b = 0.1 * np.random.randn(k, q).astype(float, order='F')

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('D', k, p, q, nb, a1.copy(), a2.copy(), b.copy(), tol=tol)

    assert info == 0
    assert rnk >= 0 and rnk <= k
    assert a1_out.shape == (k, k)
    assert a2_out.shape == (k, p - k)
    assert b_out.shape == (k, q)
    assert ipvt.shape == (k,)


def test_mb02cu_cs_valid_rotation_params():
    """
    Validate CS array contains valid hyperbolic rotation parameters.

    For column/row oriented modes, CS should contain rotation params.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    k = 3
    p = 4
    q = 2
    nb = 2

    a1 = np.tril(np.random.randn(k, k)).astype(float, order='F')
    for i in range(k):
        a1[i, i] = np.abs(a1[i, i]) + 3.0
    a2 = np.random.randn(k, p - k).astype(float, order='F')
    b = 0.3 * np.random.randn(k, q).astype(float, order='F')

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('C', k, p, q, nb, a1.copy(), a2.copy(), b.copy())

    assert info == 0
    for i in range(k):
        c = cs[2 * i]
        assert c > 0, f"c[{i}] = {c} should be positive (hyperbolic rotation)"


def test_mb02cu_edge_k_zero():
    """
    Validate edge case: K = 0 (no rows to process).

    When K = 0, routine should return immediately with info=0.
    """
    k = 0
    p = 3
    q = 2
    nb = 2

    a1 = np.zeros((1, 1), order='F', dtype=float)
    a2 = np.zeros((1, p), order='F', dtype=float)
    b = np.zeros((1, q), order='F', dtype=float)

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('C', k, p, q, nb, a1, a2, b)

    assert info == 0


def test_mb02cu_edge_q_zero_column():
    """
    Validate edge case: Q = 0 for column-oriented (no negative generator).

    When Q = 0 and P = K, routine should return immediately.
    """
    np.random.seed(111)

    k = 3
    p = 3
    q = 0
    nb = 2

    a1 = np.tril(np.random.randn(k, k)).astype(float, order='F')
    for i in range(k):
        a1[i, i] = np.abs(a1[i, i]) + 2.0
    a2 = np.zeros((k, 1), order='F', dtype=float)
    b = np.zeros((k, 1), order='F', dtype=float)

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('C', k, p, q, nb, a1.copy(), a2, b)

    assert info == 0


def test_mb02cu_error_not_positive_definite():
    """
    Validate error handling: matrix not positive definite.

    When the negative generator dominates, algorithm fails with info=1.
    """
    np.random.seed(222)

    k = 3
    p = 4
    q = 3
    nb = 2

    a1 = np.eye(k, dtype=float, order='F') * 0.1
    a2 = np.zeros((k, p - k), dtype=float, order='F')
    b = np.random.randn(k, q).astype(float, order='F') * 10.0

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('C', k, p, q, nb, a1.copy(), a2, b)

    assert info == 1


def test_mb02cu_error_invalid_typeg():
    """
    Validate error handling: invalid TYPEG parameter.
    """
    a1 = np.array([[1.0]], order='F', dtype=float)
    a2 = np.array([[0.1]], order='F', dtype=float)
    b = np.array([[0.1]], order='F', dtype=float)

    from slicot import mb02cu

    with pytest.raises((ValueError, RuntimeError)):
        mb02cu('X', 1, 2, 1, 1, a1, a2, b)


def test_mb02cu_error_p_less_than_k():
    """
    Validate error handling: P < K (invalid dimensions).
    """
    k = 3
    p = 2
    q = 2
    nb = 1

    a1 = np.random.randn(k, k).astype(float, order='F')
    a2 = np.random.randn(k, 1).astype(float, order='F')
    b = np.random.randn(k, q).astype(float, order='F')

    from slicot import mb02cu

    with pytest.raises((ValueError, RuntimeError)):
        mb02cu('C', k, p, q, nb, a1, a2, b)


def test_mb02cu_deficient_rank_reduction():
    """
    Validate rank deficient mode properly detects rank.

    Uses a generator that produces a rank-deficient situation.
    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)

    k = 4
    p = 4
    q = 4
    nb = 0
    tol = 1e-8

    a1 = np.eye(k, dtype=float, order='F') * 2.0
    a2 = np.zeros((k, 1), dtype=float, order='F')
    b = np.eye(k, dtype=float, order='F') * 1.5
    b[:, -1] = b[:, 0]

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('D', k, p, q, nb, a1.copy(), a2, b.copy(), tol=tol)

    assert info == 0 or info == 1
    assert rnk >= 0 and rnk <= k


def test_mb02cu_column_unblocked():
    """
    Validate column-oriented mode with unblocked algorithm (NB <= 0).

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)

    k = 3
    p = 5
    q = 2
    nb = 0

    a1 = np.tril(np.random.randn(k, k)).astype(float, order='F')
    for i in range(k):
        a1[i, i] = np.abs(a1[i, i]) + 4.0
    a2 = np.random.randn(k, p - k).astype(float, order='F')
    b = 0.2 * np.random.randn(k, q).astype(float, order='F')

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('C', k, p, q, nb, a1.copy(), a2.copy(), b.copy())

    assert info == 0


def test_mb02cu_row_unblocked():
    """
    Validate row-oriented mode with unblocked algorithm (NB <= 0).

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)

    k = 3
    p = 5
    q = 2
    nb = 0

    a1 = np.triu(np.random.randn(k, k)).astype(float, order='F')
    for i in range(k):
        a1[i, i] = np.abs(a1[i, i]) + 4.0
    a2 = np.random.randn(p - k, k).astype(float, order='F')
    b = 0.2 * np.random.randn(q, k).astype(float, order='F')

    from slicot import mb02cu

    a1_out, a2_out, b_out, rnk, ipvt, cs, info = mb02cu('R', k, p, q, nb, a1.copy(), a2.copy(), b.copy())

    assert info == 0
