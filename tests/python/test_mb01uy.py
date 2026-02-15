import pytest
import numpy as np
from ctrlsys import mb01uy


def _make_tri(n, uplo, seed):
    """Create a triangular matrix with given uplo."""
    np.random.seed(seed)
    a = np.random.randn(n, n).astype(np.float64, order='F')
    if uplo == 'U':
        return np.triu(a).astype(np.float64, order='F')
    else:
        return np.tril(a).astype(np.float64, order='F')


def _ref_result(side, uplo, trans, m, n, alpha, t_tri, a):
    """Compute reference result: alpha * op(T) * A or alpha * A * op(T)."""
    t_op = t_tri.T if (trans in ('T', 'C')) else t_tri.copy()
    if side == 'L':
        return alpha * t_op @ a
    else:
        return alpha * a @ t_op


@pytest.mark.parametrize("side,uplo,trans", [
    ('L', 'U', 'N'), ('L', 'U', 'T'),
    ('L', 'L', 'N'), ('L', 'L', 'T'),
    ('R', 'U', 'N'), ('R', 'U', 'T'),
    ('R', 'L', 'N'), ('R', 'L', 'T'),
])
def test_mb01uy_all_combos_small(side, uplo, trans):
    """Test all 8 SIDE/UPLO/TRANS combinations with small matrices.

    Random seed: 100 + combo_index (for reproducibility)
    """
    m, n = 4, 3
    alpha = 2.5
    k = m if side == 'L' else n
    seed = 100 + hash((side, uplo, trans)) % 100

    t = _make_tri(k, uplo, seed)
    np.random.seed(seed + 50)
    a = np.random.randn(m, n).astype(np.float64, order='F')

    expected = _ref_result(side, uplo, trans, m, n, alpha, t, a)
    result, info = mb01uy(side, uplo, trans, m, n, alpha, t, a)

    assert info == 0
    assert result.shape == (m, n)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("side,uplo,trans", [
    ('L', 'U', 'N'), ('L', 'U', 'T'),
    ('L', 'L', 'N'), ('L', 'L', 'T'),
    ('R', 'U', 'N'), ('R', 'U', 'T'),
    ('R', 'L', 'N'), ('R', 'L', 'T'),
])
def test_mb01uy_all_combos_larger(side, uplo, trans):
    """Test all 8 combos with larger matrices to exercise blocked paths.

    Random seed: 200 + combo_index (for reproducibility)
    """
    m, n = 10, 8
    alpha = -1.3
    k = m if side == 'L' else n
    seed = 200 + hash((side, uplo, trans)) % 100

    t = _make_tri(k, uplo, seed)
    np.random.seed(seed + 50)
    a = np.random.randn(m, n).astype(np.float64, order='F')

    expected = _ref_result(side, uplo, trans, m, n, alpha, t, a)
    result, info = mb01uy(side, uplo, trans, m, n, alpha, t, a)

    assert info == 0
    assert result.shape == (m, n)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_mb01uy_alpha_zero():
    """Test with alpha=0, should return zero matrix."""
    m, n = 4, 3
    np.random.seed(42)
    t = np.triu(np.random.randn(m, m)).astype(np.float64, order='F')
    a = np.random.randn(m, n).astype(np.float64, order='F')

    result, info = mb01uy('L', 'U', 'N', m, n, 0.0, t, a)

    assert info == 0
    np.testing.assert_allclose(result, np.zeros((m, n)), atol=1e-16)


def test_mb01uy_alpha_one_identity():
    """Test alpha=1 with identity triangular matrix gives back A.

    Mathematical property: I * A = A, A * I = A
    Random seed: 777
    """
    np.random.seed(777)
    m, n = 5, 4
    t = np.eye(m, dtype=np.float64, order='F')
    a = np.random.randn(m, n).astype(np.float64, order='F')

    result, info = mb01uy('L', 'U', 'N', m, n, 1.0, t, a)
    assert info == 0
    np.testing.assert_allclose(result, a, rtol=1e-14)


def test_mb01uy_scaling_property():
    """Test alpha scaling: alpha*(T*A) = (alpha*T)*A.

    Random seed: 888
    """
    np.random.seed(888)
    m, n = 6, 4
    alpha = 3.7
    t = np.triu(np.random.randn(m, m)).astype(np.float64, order='F')
    a = np.random.randn(m, n).astype(np.float64, order='F')

    result1, info1 = mb01uy('L', 'U', 'N', m, n, alpha, t, a)
    result2, info2 = mb01uy('L', 'U', 'N', m, n, 1.0, t, a)

    assert info1 == 0
    assert info2 == 0
    np.testing.assert_allclose(result1, alpha * result2, rtol=1e-14)


def test_mb01uy_trans_consistency():
    """Test op(T)*A with TRANS='T' matches explicit T^T*A.

    Random seed: 999
    """
    np.random.seed(999)
    m, n = 5, 3
    alpha = 2.0
    t_upper = np.triu(np.random.randn(m, m)).astype(np.float64, order='F')
    a = np.random.randn(m, n).astype(np.float64, order='F')

    result_trans, info1 = mb01uy('L', 'U', 'T', m, n, alpha, t_upper, a)
    assert info1 == 0

    expected = alpha * t_upper.T @ a
    np.testing.assert_allclose(result_trans, expected, rtol=1e-13)


def test_mb01uy_square_m_equals_n():
    """Test square case m == n.

    Random seed: 111
    """
    np.random.seed(111)
    n = 6
    alpha = 1.5
    t = np.tril(np.random.randn(n, n)).astype(np.float64, order='F')
    a = np.random.randn(n, n).astype(np.float64, order='F')

    expected = alpha * t @ a
    result, info = mb01uy('L', 'L', 'N', n, n, alpha, t, a)

    assert info == 0
    np.testing.assert_allclose(result, expected, rtol=1e-12)
