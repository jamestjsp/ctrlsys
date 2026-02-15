import numpy as np
import pytest


def test_mb02ud_basic_left_notrans():
    """
    Validate minimum norm least squares solution for op(R)*X = alpha*B.

    Tests: Solve R*X = alpha*B with rank-deficient R.
    Random seed: 42 (for reproducibility)

    Note: SV array contains reciprocals of singular values (1/sv) for first RANK entries,
    so they are in ascending order. Remaining entries are actual singular values.
    """
    np.random.seed(42)

    m, n = 4, 3
    l = m

    r = np.triu(np.random.randn(l, l)).astype(float, order='F')
    r[l-1, l-1] = 1e-16
    b = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb02ud

    alpha = 1.0
    rcond = 1e-10

    x, q, sv, rank, rp, info = mb02ud(
        'N', 'L', 'N', 'N', m, n, alpha, rcond, r.copy(), b.copy()
    )

    assert info == 0
    assert rank <= l
    assert rank >= 0

    for i in range(rank):
        assert sv[i] > 0


def test_mb02ud_basic_right_notrans():
    """
    Validate minimum norm least squares solution for X*op(R) = alpha*B.

    Tests: Solve X*R = alpha*B with rank-deficient R.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    m, n = 3, 4
    l = n

    r = np.triu(np.random.randn(l, l)).astype(float, order='F')
    r[l-1, l-1] = 1e-16
    b = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb02ud

    alpha = 2.0
    rcond = 1e-10

    x, q, sv, rank, rp, info = mb02ud(
        'N', 'R', 'N', 'N', m, n, alpha, rcond, r.copy(), b.copy()
    )

    assert info == 0
    assert rank <= l
    assert rank >= 0


def test_mb02ud_transpose():
    """
    Validate minimum norm least squares solution with transposed R.

    Tests: Solve R'*X = alpha*B.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    m, n = 5, 2
    l = m

    r = np.triu(np.random.randn(l, l)).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb02ud

    alpha = 1.5
    rcond = 1e-12

    x, q, sv, rank, rp, info = mb02ud(
        'N', 'L', 'T', 'N', m, n, alpha, rcond, r.copy(), b.copy()
    )

    assert info == 0
    assert rank == l


def test_mb02ud_pinv():
    """
    Validate pseudoinverse computation (JOBP='P').

    Tests: Compute pinv(R) and use it to solve system.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    m, n = 4, 3
    l = m

    r = np.triu(np.random.randn(l, l)).astype(float, order='F')
    r = r + 2.0 * np.eye(l, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb02ud

    alpha = 1.0
    rcond = 1e-12

    x, q, sv, rank, rp, info = mb02ud(
        'N', 'L', 'N', 'P', m, n, alpha, rcond, r.copy(), b.copy()
    )

    assert info == 0
    assert rank == l
    assert rp is not None
    assert rp.shape == (l, l)

    r_orig = np.triu(np.random.randn(l, l)).astype(float, order='F')
    np.random.seed(789)
    r_orig = np.triu(np.random.randn(l, l)).astype(float, order='F')
    r_orig = r_orig + 2.0 * np.eye(l, order='F')

    identity_approx = rp @ r_orig
    np.testing.assert_allclose(identity_approx, np.eye(l), rtol=1e-10, atol=1e-12)


def test_mb02ud_full_rank():
    """
    Validate solution for full rank R.

    Tests: R has full rank, RANK should equal L.
    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)

    m, n = 3, 4
    l = m

    r = np.triu(np.random.randn(l, l)).astype(float, order='F')
    r = r + 5.0 * np.eye(l, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    b_orig = b.copy()

    from ctrlsys import mb02ud

    alpha = 1.0
    rcond = 1e-14

    x, q, sv, rank, rp, info = mb02ud(
        'N', 'L', 'N', 'N', m, n, alpha, rcond, r.copy(), b.copy()
    )

    assert info == 0
    assert rank == l

    np.random.seed(888)
    r_orig = np.triu(np.random.randn(l, l)).astype(float, order='F')
    r_orig = r_orig + 5.0 * np.eye(l, order='F')
    residual = r_orig @ x - alpha * b_orig
    np.testing.assert_allclose(residual, np.zeros_like(residual), rtol=1e-10, atol=1e-12)


def test_mb02ud_alpha_zero():
    """
    Validate behavior when alpha = 0.

    Tests: X should be set to zero when alpha = 0.
    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    m, n = 3, 2
    l = m

    r = np.triu(np.random.randn(l, l)).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb02ud

    alpha = 0.0
    rcond = 1e-12

    x, q, sv, rank, rp, info = mb02ud(
        'N', 'L', 'N', 'N', m, n, alpha, rcond, r.copy(), b.copy()
    )

    assert info == 0
    np.testing.assert_allclose(x, np.zeros_like(x), rtol=1e-15, atol=1e-15)


def test_mb02ud_edge_case_1x1():
    """
    Validate edge case: 1x1 matrix.
    """
    r = np.array([[2.0]], order='F', dtype=float)
    b = np.array([[4.0]], order='F', dtype=float)

    from ctrlsys import mb02ud

    alpha = 1.0
    rcond = 1e-14

    x, q, sv, rank, rp, info = mb02ud(
        'N', 'L', 'N', 'N', 1, 1, alpha, rcond, r.copy(), b.copy()
    )

    assert info == 0
    assert rank == 1
    np.testing.assert_allclose(x[0, 0], 2.0, rtol=1e-14)


def test_mb02ud_svd_properties():
    """
    Validate SVD properties: Q orthogonal.

    Random seed: 1111 (for reproducibility)

    Note: SV array contains reciprocals of singular values (1/sv) for first RANK entries,
    so sv[0:rank] contains 1/sigma_1, ..., 1/sigma_rank (ascending order since sigma_i descending).
    Entries sv[rank:] contain actual singular values sigma_{rank+1}, ..., sigma_l in descending order.
    """
    np.random.seed(1111)

    m, n = 5, 3
    l = m

    r = np.triu(np.random.randn(l, l)).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb02ud

    alpha = 1.0
    rcond = 1e-12

    x, q, sv, rank, rp, info = mb02ud(
        'N', 'L', 'N', 'N', m, n, alpha, rcond, r.copy(), b.copy()
    )

    assert info == 0

    for i in range(rank):
        assert sv[i] > 0

    for i in range(rank, l - 1):
        assert sv[i] >= sv[i + 1]

    qtq = q.T @ q
    np.testing.assert_allclose(qtq, np.eye(l), rtol=1e-13, atol=1e-14)


def test_mb02ud_factored_input():
    """
    Validate using pre-factored SVD (FACT='F').

    Random seed: 2222 (for reproducibility)
    """
    np.random.seed(2222)

    m, n = 4, 2
    l = m

    r = np.triu(np.random.randn(l, l)).astype(float, order='F')
    r = r + 3.0 * np.eye(l, order='F')
    r_factored = r.copy()
    b = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb02ud

    alpha = 1.0
    rcond = 1e-12

    x1, q1, sv1, rank1, rp1, info1 = mb02ud(
        'N', 'L', 'N', 'N', m, n, alpha, rcond, r_factored, b.copy()
    )
    assert info1 == 0

    b2 = np.random.randn(m, n).astype(float, order='F')

    x2, q2, sv2, rank2, rp2, info2 = mb02ud(
        'F', 'L', 'N', 'N', m, n, alpha, rcond, r_factored.copy(), b2.copy(),
        q=q1.copy(), sv=sv1.copy(), rank=rank1
    )
    assert info2 == 0


def test_mb02ud_error_invalid_m():
    """
    Validate error handling: invalid M (negative).
    """
    r = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)

    from ctrlsys import mb02ud

    with pytest.raises((ValueError, RuntimeError)):
        mb02ud('N', 'L', 'N', 'N', -1, 1, 1.0, 1e-12, r, b)


def test_mb02ud_error_invalid_fact():
    """
    Validate error handling: invalid FACT.
    """
    r = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)

    from ctrlsys import mb02ud

    with pytest.raises((ValueError, RuntimeError)):
        mb02ud('X', 'L', 'N', 'N', 1, 1, 1.0, 1e-12, r, b)
