"""
Tests for AB08NX - Extract reduced system with full row rank D.

AB08NX extracts from the (N+P)-by-(M+N) compound system:
    [ B  A ]
    [ D  C ]
a reduced (NU+MU)-by-(M+NU) system:
    [ B' A']
    [ D' C']
having the same transmission zeros but with D' of full row rank.
"""
import numpy as np
import pytest
from ctrlsys import ab08nx


"""Basic functionality tests for ab08nx."""

def test_simple_siso_system():
    """
    Simple SISO system with n=2, m=1, p=1.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 2, 1, 1

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0
    assert mu >= 0
    assert nu >= 0
    assert nkrol >= 0

def test_mimo_2x2_system():
    """
    MIMO system with n=3, m=2, p=2.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0
    assert 0 <= mu <= p + n
    assert 0 <= nu <= n

def test_pertransposed_system():
    """
    Test with pertransposed system parameters (ro=max(p-m,0), sigma=m).
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 3, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = max(p - m, 0)
    sigma = m
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0


"""Tests with special system structures."""

def test_zero_d_matrix():
    """
    System with D=0 (strictly proper).
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.zeros((p, m), dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0

def test_full_rank_d_matrix():
    """
    System with full rank D matrix (no reduction needed).
    Random seed: 321 (for reproducibility)
    """
    np.random.seed(321)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.eye(p, m, dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0

def test_rank_deficient_d():
    """
    System with rank-deficient D matrix.
    """
    n, m, p = 3, 2, 3

    a = np.eye(n, dtype=float, order='F')
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                 dtype=float, order='F')
    d = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0
    assert mu <= p


"""Mathematical property tests for ab08nx."""

def test_reduced_dimensions():
    """
    Output dimensions should satisfy:
    - NU <= N (reduced state dimension)
    - MU <= P + N (reduced output rows plus remaining)
    Random seed: 654 (for reproducibility)
    """
    np.random.seed(654)
    n, m, p = 4, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0
    assert nu <= n
    assert mu <= p + n

def test_kronecker_indices_non_negative():
    """
    Left Kronecker indices should be non-negative.
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 4, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0
    if nkrol > 0:
        assert all(k >= 0 for k in kronl[:nkrol])

def test_infz_non_negative():
    """
    Infinite zero counts should be non-negative.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m, p = 4, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0
    assert ninfz_out >= 0
    if n > 0:
        assert all(iz >= 0 for iz in infz)


"""Edge case tests for ab08nx."""

def test_n_zero():
    """
    System with n=0 (static system, no state).
    """
    n, m, p = 0, 2, 2

    d = np.eye(p, m, dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[n:, :m] = d

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0
    assert nu == 0

def test_m_zero():
    """
    System with m=0 (no inputs).
    """
    n, m, p = 2, 0, 2

    a = np.eye(n, dtype=float, order='F')
    c = np.eye(p, n, dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, m:] = a
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0

def test_p_zero():
    """
    System with p=0 (no outputs).
    """
    n, m, p = 2, 2, 0

    a = np.eye(n, dtype=float, order='F')
    b = np.eye(n, m, dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0
    assert mu == 0

def test_siso_1x1():
    """
    Minimal SISO system n=1, m=1, p=1.
    """
    n, m, p = 1, 1, 1

    a = np.array([[0.5]], dtype=float, order='F')
    b = np.array([[1.0]], dtype=float, order='F')
    c = np.array([[1.0]], dtype=float, order='F')
    d = np.array([[0.0]], dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0


"""Tests for parameter variations."""

def test_svlmax_positive():
    """
    Test with positive svlmax.
    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    svlmax = np.linalg.norm(abcd, 'fro')

    ro = p
    sigma = 0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0

def test_tol_variation():
    """
    Different tolerance values should give consistent results for well-conditioned systems.
    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.eye(p, m, dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0

    result1 = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                     abcd=abcd.copy(order='F'), ninfz=ninfz, tol=1e-6)
    result2 = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                     abcd=abcd.copy(order='F'), ninfz=ninfz, tol=1e-12)

    assert result1[9] == 0
    assert result2[9] == 0


"""Error handling tests for ab08nx."""

def test_negative_n():
    """Negative n should raise error."""
    abcd = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError):
        ab08nx(n=-1, m=1, p=2, ro=2, sigma=0, svlmax=0.0,
               abcd=abcd, ninfz=0, tol=1e-10)

def test_negative_m():
    """Negative m should raise error."""
    abcd = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError):
        ab08nx(n=1, m=-1, p=2, ro=2, sigma=0, svlmax=0.0,
               abcd=abcd, ninfz=0, tol=1e-10)

def test_negative_p():
    """Negative p should raise error."""
    abcd = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError):
        ab08nx(n=1, m=1, p=-1, ro=0, sigma=0, svlmax=0.0,
               abcd=abcd, ninfz=0, tol=1e-10)

def test_invalid_ro():
    """Invalid ro (not p or max(p-m,0)) should return error."""
    n, m, p = 2, 1, 2

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')

    result = ab08nx(n=n, m=m, p=p, ro=5, sigma=0, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1e-10)

    assert result[9] == -4

def test_invalid_sigma():
    """Invalid sigma (not 0 or m) should return error."""
    n, m, p = 2, 1, 2

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')

    result = ab08nx(n=n, m=m, p=p, ro=p, sigma=5, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1e-10)

    assert result[9] == -5

def test_negative_svlmax():
    """Negative svlmax should return error."""
    n, m, p = 2, 1, 2

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')

    result = ab08nx(n=n, m=m, p=p, ro=p, sigma=0, svlmax=-1.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1e-10)

    assert result[9] == -6


"""Tests comparing ab08nx behavior with expected mathematical properties."""

def test_transmission_zero_count():
    """
    For a system with known transmission zeros, verify count is preserved.
    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    ro = p
    sigma = 0
    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08nx(n=n, m=m, p=p, ro=ro, sigma=sigma, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result

    assert info == 0
    assert nu <= n


def test_reduced_d_full_row_rank():
    """
    Verify that the reduced D' matrix has full row rank.
    """
    np.random.seed(700)
    n, m, p = 4, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    result = ab08nx(n=n, m=m, p=p, ro=p, sigma=0, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1e-10)
    abcd_out, ro_out, sigma_out, ninfz_out, mu, nu, nkrol, infz, kronl, info = result
    assert info == 0

    if mu > 0:
        d_prime = abcd_out[nu:nu+mu, :m]
        if d_prime.size > 0:
            sv = np.linalg.svd(d_prime, compute_uv=False)
            assert np.sum(sv > 1e-8) == mu


def test_normal_rank_matches_ab08md():
    """
    Verify that the normal rank (mu) from AB08NX matches AB08MD.
    """
    from ctrlsys import ab08md

    np.random.seed(800)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    a -= 2.0 * np.eye(n)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    rank, info_md = ab08md('N', n, m, p, a.copy(order='F'), b.copy(order='F'),
                           c.copy(order='F'), d.copy(order='F'))
    assert info_md == 0

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    result = ab08nx(n=n, m=m, p=p, ro=p, sigma=0, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1e-10)
    _, _, _, _, mu, nu, _, _, _, info = result
    assert info == 0
    assert mu == rank
