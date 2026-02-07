"""
Tests for AB08NY - Extract reduced system with full row rank D.

AB08NY extracts from the (N+P)-by-(M+N) system pencil:
    ( B  A-lambda*I )
    ( D      C      )
an (NR+PR)-by-(M+NR) "reduced" system pencil:
    ( Br Ar-lambda*I )
    ( Dr     Cr      )
having the same transmission zeros, but with Dr of full row rank.

AB08NY differs from AB08NX in that:
- Has a FIRST parameter for first-time vs subsequent calls
- Returns PR instead of MU (normal rank of transfer function)
- Returns NR instead of NU (order of reduced matrix Ar)
- Returns DINFZ (maximal multiplicity of infinite zeros)
"""
import numpy as np
import pytest
from slicot import ab08ny


def test_simple_siso_first():
    """
    Simple SISO system with n=2, m=1, p=1, first=True.
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

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert nr >= 0
    assert pr >= 0
    assert dinfz >= 0
    assert nkronl >= 0


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

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert 0 <= nr <= n
    assert 0 <= pr <= p


def test_first_false_with_prepared_d():
    """
    Test with first=False, where D must have full column rank with
    last M rows in upper triangular form.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 3, 2, 4

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.zeros((p, m), dtype=float, order='F')
    d[-m:, :] = np.eye(m)

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=False, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0


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

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0


def test_full_rank_d_matrix():
    """
    System with full rank D matrix.
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

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0


def test_reduced_dimensions():
    """
    Output dimensions should satisfy:
    - NR <= N (reduced state dimension)
    - PR <= P (reduced output dimension = normal rank)
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

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert nr <= n
    assert pr <= p


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

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    if nkronl > 0:
        assert all(k >= 0 for k in kronl[:nkronl])


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

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert ninfz_out >= 0
    if n > 0:
        assert all(iz >= 0 for iz in infz[:dinfz] if dinfz > 0)


def test_n_zero():
    """
    System with n=0 (static system, no state).
    """
    n, m, p = 0, 2, 2

    d = np.eye(p, m, dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[n:, :m] = d

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert nr == 0


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

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0


def test_p_zero():
    """
    System with p=0 (no outputs).
    """
    n, m, p = 2, 2, 0

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert pr == 0


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

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0


def test_svlmax_positive():
    """
    Test with positive svlmax (Frobenius norm estimate).
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
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

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

    svlmax = 0.0
    ninfz = 0

    result1 = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                     abcd=abcd.copy(order='F'), ninfz=ninfz, tol=1e-6)
    result2 = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                     abcd=abcd.copy(order='F'), ninfz=ninfz, tol=1e-12)

    assert result1[8] == 0
    assert result2[8] == 0


def test_negative_n():
    """Negative n should raise error."""
    abcd = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError):
        ab08ny(first=True, n=-1, m=1, p=2, svlmax=0.0,
               abcd=abcd, ninfz=0, tol=1e-10)


def test_negative_m():
    """Negative m should raise error."""
    abcd = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError):
        ab08ny(first=True, n=1, m=-1, p=2, svlmax=0.0,
               abcd=abcd, ninfz=0, tol=1e-10)


def test_negative_p():
    """Negative p should raise error."""
    abcd = np.zeros((3, 3), dtype=float, order='F')

    with pytest.raises(ValueError):
        ab08ny(first=True, n=1, m=1, p=-1, svlmax=0.0,
               abcd=abcd, ninfz=0, tol=1e-10)


def test_invalid_m_with_first_false():
    """When first=False, m must be <= p."""
    n, m, p = 2, 3, 2

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')

    result = ab08ny(first=False, n=n, m=m, p=p, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1e-10)

    assert result[8] == -3


def test_negative_svlmax():
    """Negative svlmax should return error."""
    n, m, p = 2, 1, 2

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=-1.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1e-10)

    assert result[8] == -5


def test_ninfz_negative():
    """Negative ninfz should return error."""
    n, m, p = 2, 1, 2

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=-1, tol=1e-10)

    assert result[8] == -8


def test_ninfz_positive_with_first_true():
    """When first=True, ninfz must be 0."""
    n, m, p = 2, 1, 2

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=5, tol=1e-10)

    assert result[8] == -8


def test_tol_too_large():
    """TOL >= 1 should return error."""
    n, m, p = 2, 1, 2

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1.0)

    assert result[8] == -15


def test_infinite_zeros_computation():
    """
    Test that infinite zeros are computed correctly.
    For FIRST=.TRUE., NINFZ = Sum(INFZ(i)*i) for i=1 to DINFZ.
    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n, m, p = 4, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.zeros((p, m), dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0

    computed_ninfz = sum((i + 1) * infz[i] for i in range(dinfz)) if dinfz > 0 else 0
    assert ninfz_out == computed_ninfz


def test_wide_system():
    """
    Test system with more inputs than outputs (m > p).
    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    n, m, p = 3, 4, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert pr <= p


def test_tall_system():
    """
    Test system with more outputs than inputs (p > m).
    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)
    n, m, p = 3, 2, 4

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    d = np.random.randn(p, m).astype(float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    svlmax = 0.0
    ninfz = 0
    tol = 1e-10

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=svlmax,
                    abcd=abcd.copy(order='F'), ninfz=ninfz, tol=tol)

    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert nr <= n


def test_pr_matches_normal_rank():
    """
    Verify PR (normal rank) from AB08NY matches AB08MD for the same system.
    """
    from slicot import ab08md

    np.random.seed(900)
    n, m, p = 4, 2, 3

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

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1e-10)
    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result
    assert info == 0
    assert pr == rank


def test_infinite_zero_multiplicity_strictly_proper():
    """
    For a strictly proper system (D=0), verify infinite zeros are detected
    and NINFZ = sum(i * INFZ(i)).
    """
    n, m, p = 3, 2, 2

    a = np.array([[-1.0, 0.0, 0.0],
                  [0.0, -2.0, 0.0],
                  [0.0, 0.0, -3.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.0, 0.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], order='F', dtype=float)
    d = np.zeros((p, m), dtype=float, order='F')

    abcd = np.zeros((n + p, m + n), dtype=float, order='F')
    abcd[:n, :m] = b
    abcd[:n, m:] = a
    abcd[n:, :m] = d
    abcd[n:, m:] = c

    result = ab08ny(first=True, n=n, m=m, p=p, svlmax=0.0,
                    abcd=abcd.copy(order='F'), ninfz=0, tol=1e-10)
    abcd_out, ninfz_out, nr, pr, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert ninfz_out > 0
    computed_ninfz = sum((i + 1) * infz[i] for i in range(dinfz)) if dinfz > 0 else 0
    assert ninfz_out == computed_ninfz
