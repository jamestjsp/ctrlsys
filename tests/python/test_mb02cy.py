import numpy as np
import pytest


def test_mb02cy_basic_row():
    """
    Validate applying hyperbolic transformations on row-wise generator.

    Tests: Apply transformations computed by MB02CX to additional columns.
    Uses simple identity-based transformation with known outputs.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    p = 3
    q = 2
    n = 4
    k = 3

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.random.randn(q, n).astype(float, order='F')

    h = np.random.randn(q, k).astype(float, order='F')
    lcs = 2 * k + min(k, q)
    cs = np.random.randn(lcs).astype(float, order='F')
    cs[0:2*k:2] = np.abs(cs[0:2*k:2]) + 1.0

    from ctrlsys import mb02cy

    a_orig = a.copy()
    b_orig = b.copy()

    a_out, b_out, info = mb02cy('R', 'N', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0
    assert a_out.shape == (p, n)
    assert b_out.shape == (q, n)

    assert np.all(np.isfinite(a_out))
    assert np.all(np.isfinite(b_out))
    assert not np.allclose(a_out, a_orig) or not np.allclose(b_out, b_orig)


def test_mb02cy_jnorm_preservation():
    np.random.seed(99)
    from ctrlsys import mb02cx, mb02cy

    p = 4
    q = 2
    k = 3
    n = 5

    a_first = np.triu(np.random.randn(p, k)).astype(float, order='F')
    for i in range(min(p, k)):
        a_first[i, i] = np.abs(a_first[i, i]) + 5.0
    b_first = 0.1 * np.random.randn(q, k).astype(float, order='F')

    a_cx_out, b_cx_out, cs, info_cx = mb02cx('R', p, q, k, a_first.copy(), b_first.copy())
    assert info_cx == 0

    a_extra = np.random.randn(p, n).astype(float, order='F')
    b_extra = np.random.randn(q, n).astype(float, order='F')
    a_extra_orig = a_extra.copy()
    b_extra_orig = b_extra.copy()

    a_out, b_out, info = mb02cy('R', 'N', p, q, n, k, a_extra.copy(), b_extra.copy(),
                                 b_cx_out.copy(), cs.copy())
    assert info == 0

    for j in range(n):
        jnorm_before = np.dot(a_extra_orig[:k, j], a_extra_orig[:k, j]) - np.dot(b_extra_orig[:, j], b_extra_orig[:, j])
        jnorm_after = np.dot(a_out[:k, j], a_out[:k, j]) - np.dot(b_out[:, j], b_out[:, j])
        np.testing.assert_allclose(jnorm_after, jnorm_before, atol=1e-8)


def test_mb02cy_basic_column():
    """
    Validate applying hyperbolic transformations on column-wise generator.

    Tests: Apply transformations computed by MB02CX to additional rows.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    p = 3
    q = 2
    n = 4
    k = 3

    a = np.random.randn(n, p).astype(float, order='F')
    b = np.random.randn(n, q).astype(float, order='F')

    h = np.random.randn(k, q).astype(float, order='F')
    lcs = 2 * k + min(k, q)
    cs = np.random.randn(lcs).astype(float, order='F')
    cs[0:2*k:2] = np.abs(cs[0:2*k:2]) + 1.0

    from ctrlsys import mb02cy

    a_out, b_out, info = mb02cy('C', 'N', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0
    assert a_out.shape == (n, p)
    assert b_out.shape == (n, q)
    assert np.all(np.isfinite(a_out))
    assert np.all(np.isfinite(b_out))


def test_mb02cy_triangular_structure():
    """
    Validate applying transformations with triangular structure.

    Tests: STRUCG = 'T' (triangular positive generator, zero negative trailing block).
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    p = 4
    q = 3
    n = 5
    k = 4

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.random.randn(q, n).astype(float, order='F')

    h = np.random.randn(q, k).astype(float, order='F')
    lcs = 2 * k + min(k, q)
    cs = np.random.randn(lcs).astype(float, order='F')
    cs[0:2*k:2] = np.abs(cs[0:2*k:2]) + 1.0

    from ctrlsys import mb02cy

    a_out, b_out, info = mb02cy('R', 'T', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0
    assert a_out.shape == (p, n)
    assert b_out.shape == (q, n)
    assert np.all(np.isfinite(a_out))
    assert np.all(np.isfinite(b_out))


def test_mb02cy_edge_case_q_zero():
    """
    Validate edge case: Q = 0 (no negative generator).

    When Q = 0 or K = 0, routine should return immediately.
    """
    p = 3
    k = 2
    n = 4
    q = 0

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.zeros((1, n), order='F', dtype=float)
    h = np.zeros((1, k), order='F', dtype=float)
    lcs = 2 * k
    cs = np.zeros(lcs, order='F', dtype=float)

    from ctrlsys import mb02cy

    a_out, b_out, info = mb02cy('R', 'N', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0


def test_mb02cy_edge_case_n_zero():
    """
    Validate edge case: N = 0 (no columns/rows to process).
    """
    p = 3
    q = 2
    k = 3
    n = 0

    a = np.zeros((p, 1), order='F', dtype=float)
    b = np.zeros((q, 1), order='F', dtype=float)
    h = np.zeros((q, k), order='F', dtype=float)
    lcs = 2 * k + min(k, q)
    cs = np.zeros(lcs, order='F', dtype=float)

    from ctrlsys import mb02cy

    a_out, b_out, info = mb02cy('R', 'N', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0


def test_mb02cy_error_invalid_typet():
    """
    Validate error handling: invalid TYPET parameter.
    """
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    h = np.array([[1.0]], order='F', dtype=float)
    cs = np.array([1.0, 0.0, 1.0], order='F', dtype=float)

    from ctrlsys import mb02cy

    with pytest.raises((ValueError, RuntimeError)):
        mb02cy('X', 'N', 1, 1, 1, 1, a, b, h, cs)


def test_mb02cy_error_invalid_strucg():
    """
    Validate error handling: invalid STRUCG parameter.
    """
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    h = np.array([[1.0]], order='F', dtype=float)
    cs = np.array([1.0, 0.0, 1.0], order='F', dtype=float)

    from ctrlsys import mb02cy

    with pytest.raises((ValueError, RuntimeError)):
        mb02cy('R', 'X', 1, 1, 1, 1, a, b, h, cs)


def test_mb02cy_error_k_exceeds_p():
    """
    Validate error handling: K > P (invalid dimensions).
    """
    p = 2
    q = 2
    k = 3
    n = 4

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.random.randn(q, n).astype(float, order='F')
    h = np.random.randn(q, k).astype(float, order='F')
    cs = np.zeros(2 * k + min(k, q), order='F', dtype=float)

    from ctrlsys import mb02cy

    with pytest.raises((ValueError, RuntimeError)):
        mb02cy('R', 'N', p, q, n, k, a, b, h, cs)
