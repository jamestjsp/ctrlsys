import numpy as np
import pytest
from slicot import mb02yd


def test_mb02yd_basic_full_rank():
    """Test MB02YD with full rank system (COND='N')"""
    n = 3

    # Upper triangular R from QR factorization
    r = np.array([
        [3.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    # Permutation (identity for simplicity)
    ipvt = np.array([1, 2, 3], dtype=np.int32)

    # Diagonal constraint matrix D
    diag = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Q'*b
    qtb = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    # Expected: solve R*z = Q'*b, D*z = 0 (augmented system)
    r_orig = r.copy()
    x, rank, info = mb02yd('N', n, r, ipvt, diag, qtb, 0, 0.0)

    assert info == 0
    assert rank == n
    assert x.shape == (n,)
    assert np.all(np.isfinite(x))
    assert np.linalg.norm(x) < 100.0

    z = x[ipvt - 1]
    D_perm = np.diag(diag[ipvt - 1])
    A_aug = np.vstack([r_orig, D_perm])
    b_aug = np.concatenate([qtb, np.zeros(n)])
    z_ref, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    np.testing.assert_allclose(z, z_ref, atol=1e-10)


def test_mb02yd_with_condition_estimation():
    """Test MB02YD with condition estimation (COND='E')"""
    n = 4

    # Upper triangular R
    r = np.array([
        [4.0, 3.0, 2.0, 1.0],
        [0.0, 3.0, 2.0, 1.0],
        [0.0, 0.0, 2.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    # Permutation
    ipvt = np.array([1, 2, 3, 4], dtype=np.int32)

    # Diagonal matrix D
    diag = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)

    # Q'*b
    qtb = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # Use condition estimation with default tolerance
    x, rank, info = mb02yd('E', n, r, ipvt, diag, qtb, 0, 0.0)

    assert info == 0
    assert rank > 0 and rank <= n
    assert x.shape == (n,)
    assert np.all(np.isfinite(x))


def test_mb02yd_with_permutation():
    """Test MB02YD with non-trivial permutation"""
    n = 3

    # Upper triangular R
    r = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    # Non-identity permutation: columns [3, 1, 2]
    ipvt = np.array([3, 1, 2], dtype=np.int32)

    # Diagonal matrix D
    diag = np.array([0.05, 0.1, 0.15], dtype=np.float64)

    # Q'*b
    qtb = np.array([0.5, 1.0, 1.5], dtype=np.float64)

    x, rank, info = mb02yd('N', n, r, ipvt, diag, qtb, 0, 0.0)

    assert info == 0
    assert x.shape == (n,)
    assert np.all(np.isfinite(x))


def test_mb02yd_rank_deficient():
    """Test MB02YD with rank-deficient system"""
    n = 4

    # Upper triangular R with small diagonal element
    r = np.array([
        [1.0, 0.5, 0.3, 0.1],
        [0.0, 0.8, 0.2, 0.1],
        [0.0, 0.0, 1e-15, 0.05],  # Near-zero diagonal
        [0.0, 0.0, 0.0, 0.6]
    ], dtype=np.float64, order='F')

    ipvt = np.array([1, 2, 3, 4], dtype=np.int32)
    diag = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    qtb = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # Use condition estimation to detect rank deficiency
    x, rank, info = mb02yd('E', n, r, ipvt, diag, qtb, 0, 1e-10)

    assert info == 0
    # Rank may be full after augmentation with D, just check reasonable
    assert rank > 0 and rank <= n
    assert x.shape == (n,)
    assert np.all(np.isfinite(x))


def test_mb02yd_zero_dimension():
    """Test MB02YD with N=0 (quick return)"""
    n = 0
    r = np.zeros((1, 1), dtype=np.float64, order='F')
    ipvt = np.zeros(1, dtype=np.int32)
    diag = np.zeros(1, dtype=np.float64)
    qtb = np.zeros(1, dtype=np.float64)

    x, rank, info = mb02yd('N', n, r, ipvt, diag, qtb, 0, 0.0)

    assert info == 0
    assert rank == 0


def test_mb02yd_use_rank():
    """Test MB02YD with pre-computed rank (COND='U')"""
    n = 3

    r = np.array([
        [3.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    ipvt = np.array([1, 2, 3], dtype=np.int32)
    diag = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    qtb = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    # Use pre-computed rank
    rank_in = 3
    x, rank_out, info = mb02yd('U', n, r, ipvt, diag, qtb, rank_in, 0.0)

    assert info == 0
    assert rank_out == rank_in
    assert x.shape == (n,)
    assert np.all(np.isfinite(x))


def test_mb02yd_zero_diag_element():
    """Test MB02YD with some zero diagonal D elements"""
    n = 3

    r = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    ipvt = np.array([1, 2, 3], dtype=np.int32)

    # Second diagonal element is zero
    diag = np.array([0.1, 0.0, 0.15], dtype=np.float64)

    qtb = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    x, rank, info = mb02yd('N', n, r, ipvt, diag, qtb, 0, 0.0)

    assert info == 0
    assert x.shape == (n,)
    assert np.all(np.isfinite(x))


def test_mb02yd_invalid_parameters():
    """Test MB02YD parameter validation"""
    n = 3
    r = np.zeros((3, 3), dtype=np.float64, order='F')
    ipvt = np.array([1, 2, 3], dtype=np.int32)
    diag = np.zeros(3, dtype=np.float64)
    qtb = np.zeros(3, dtype=np.float64)

    # Invalid COND
    with pytest.raises((ValueError, RuntimeError)):
        mb02yd('X', n, r, ipvt, diag, qtb, 0, 0.0)

    # Invalid N
    with pytest.raises((ValueError, RuntimeError)):
        mb02yd('N', -1, r, ipvt, diag, qtb, 0, 0.0)

    # Invalid RANK for COND='U'
    with pytest.raises((ValueError, RuntimeError)):
        mb02yd('U', n, r, ipvt, diag, qtb, -1, 0.0)

    with pytest.raises((ValueError, RuntimeError)):
        mb02yd('U', n, r, ipvt, diag, qtb, n+1, 0.0)
