import numpy as np
import pytest


def test_mb02cx_basic_row():
    """
    Validate bringing first blocks of row-wise generator to proper form.

    Tests: QR decomposition of B, then Householder + hyperbolic rotation.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    p = 4
    q = 3
    k = 3

    a = np.triu(np.random.randn(p, k)).astype(float, order='F')
    for i in range(min(p, k)):
        a[i, i] = np.abs(a[i, i]) + 5.0
    b = 0.1 * np.random.randn(q, k).astype(float, order='F')

    from slicot import mb02cx

    a_orig = a.copy()
    a_out, b_out, cs, info = mb02cx('R', p, q, k, a.copy(), b.copy())

    assert info == 0
    assert a_out.shape == (p, k)
    assert b_out.shape == (q, k)
    assert cs.shape[0] >= 2 * k + min(k, q)

    np.testing.assert_allclose(np.tril(a_out[:k, :k], -1), 0.0, atol=1e-12)
    assert np.all(np.diag(a_out[:k, :k]) > 0)


def test_mb02cx_basic_column():
    """
    Validate bringing first blocks of column-wise generator to proper form.

    Tests: LQ decomposition of B, then Householder + hyperbolic rotation.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    p = 4
    q = 3
    k = 3

    a = np.tril(np.random.randn(k, p)).astype(float, order='F')
    for i in range(min(k, p)):
        a[i, i] = np.abs(a[i, i]) + 5.0
    b = 0.1 * np.random.randn(k, q).astype(float, order='F')

    from slicot import mb02cx

    a_out, b_out, cs, info = mb02cx('C', p, q, k, a.copy(), b.copy())

    assert info == 0
    assert a_out.shape == (k, p)
    assert b_out.shape == (k, q)
    assert cs.shape[0] >= 2 * k + min(k, q)

    np.testing.assert_allclose(np.triu(a_out[:k, :k], 1), 0.0, atol=1e-12)
    assert np.all(np.diag(a_out[:k, :k]) > 0)


def test_mb02cx_cs_contains_valid_rotation_params():
    """
    Validate CS contains valid hyperbolic rotation parameters.

    The CS array should contain c, s pairs where c^2 + s^2 = 1 for
    standard rotations, or c > 0 for hyperbolic rotations.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    p = 3
    q = 2
    k = 3

    a = np.triu(np.random.randn(p, k)).astype(float, order='F')
    for i in range(min(p, k)):
        a[i, i] = np.abs(a[i, i]) + 2.0
    b = 0.5 * np.random.randn(q, k).astype(float, order='F')

    from slicot import mb02cx

    a_out, b_out, cs, info = mb02cx('R', p, q, k, a.copy(), b.copy())

    assert info == 0

    for i in range(k):
        c = cs[2 * i]
        assert c > 0, f"c[{i}] = {c} should be positive"


def test_mb02cx_positive_definite_matrix():
    """
    Validate routine works for positive definite Toeplitz generator.

    Creates a generator for a known positive definite matrix.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    p = 3
    q = 2
    k = 3

    a = np.eye(p, k, order='F', dtype=float) * 5.0
    a[0, 1] = 1.0
    a[0, 2] = 0.5
    a[1, 2] = 1.0

    b = np.zeros((q, k), order='F', dtype=float)
    b[0, 0] = 0.1
    b[0, 1] = 0.2
    b[1, 1] = 0.1

    from slicot import mb02cx

    a_out, b_out, cs, info = mb02cx('R', p, q, k, a.copy(), b.copy())

    assert info == 0


def test_mb02cx_edge_case_q_zero():
    """
    Validate edge case: Q = 0 (no negative generator).

    When Q = 0, routine should return immediately with info=0.
    """
    np.random.seed(111)

    p = 3
    k = 3
    q = 0

    a = np.triu(np.random.randn(p, k)).astype(float, order='F')
    for i in range(min(p, k)):
        a[i, i] = np.abs(a[i, i]) + 1.0
    b = np.zeros((1, k), order='F', dtype=float)

    from slicot import mb02cx

    a_out, b_out, cs, info = mb02cx('R', p, q, k, a.copy(), b.copy())

    assert info == 0


def test_mb02cx_edge_case_k_zero():
    """
    Validate edge case: K = 0 (no columns to process).

    When K = 0, routine should return immediately with info=0.
    """
    p = 3
    q = 2
    k = 0

    a = np.zeros((p, 1), order='F', dtype=float)
    b = np.zeros((q, 1), order='F', dtype=float)

    from slicot import mb02cx

    a_out, b_out, cs, info = mb02cx('R', p, q, k, a.copy(), b.copy())

    assert info == 0


def test_mb02cx_error_not_positive_definite():
    """
    Validate error handling: matrix not positive definite.

    When the negative generator dominates, the algorithm fails with info=1.
    """
    np.random.seed(222)

    p = 3
    q = 3
    k = 3

    a = np.zeros((p, k), order='F', dtype=float)
    a[0, 0] = 0.1
    a[1, 1] = 0.1
    a[2, 2] = 0.1

    b = np.random.randn(q, k).astype(float, order='F') * 10.0

    from slicot import mb02cx

    a_out, b_out, cs, info = mb02cx('R', p, q, k, a.copy(), b.copy())

    assert info == 1


def test_mb02cx_error_invalid_typet():
    """
    Validate error handling: invalid TYPET parameter.
    """
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[0.1]], order='F', dtype=float)

    from slicot import mb02cx

    with pytest.raises((ValueError, RuntimeError)):
        mb02cx('X', 1, 1, 1, a, b)


def test_mb02cx_error_k_exceeds_p():
    """
    Validate error handling: K > P (invalid dimensions).
    """
    p = 2
    q = 2
    k = 3

    a = np.random.randn(p, k).astype(float, order='F')
    b = np.random.randn(q, k).astype(float, order='F')

    from slicot import mb02cx

    with pytest.raises((ValueError, RuntimeError)):
        mb02cx('R', p, q, k, a, b)
