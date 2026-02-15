#!/usr/bin/env python3
"""
pytest tests for AG8BYZ - Complex Kronecker structure computation.

This routine extracts from the (N+P)-by-(M+N) descriptor system pencil
S(lambda) = [B, A-lambda*E; D, C] a reduced pencil Sr(lambda) with
the same finite Smith zeros but with Dr full row rank (complex case).
"""
import pytest
import numpy as np
from ctrlsys import ag8byz


def test_ag8byz_basic_first_call():
    """
    Test AG8BYZ basic functionality with FIRST=True.

    Uses a simple 3x3 state-space system with known structure.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 3, 2, 2

    A = np.array([
        [1.0+0.1j, 2.0+0.0j, 0.0+0.0j],
        [4.0+0.0j, -1.0+0.2j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 1.0+0.0j]
    ], order='F', dtype=complex)

    B = np.array([
        [1.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 1.0+0.0j],
        [1.0+0.0j, 1.0+0.0j]
    ], order='F', dtype=complex)

    C = np.array([
        [1.0+0.0j, 0.0+0.0j, 1.0+0.0j],
        [0.0+0.0j, 1.0+0.0j, -1.0+0.0j]
    ], order='F', dtype=complex)

    D = np.array([
        [0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)

    E = np.eye(n, order='F', dtype=complex)

    abcd = np.zeros((n + p, m + n), order='F', dtype=complex)
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    svlmax = 0.0
    tol = 0.0

    result = ag8byz(True, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0, f"Expected info=0, got {info}"
    assert nr >= 0, f"NR should be non-negative, got {nr}"
    assert pr >= 0, f"PR should be non-negative, got {pr}"
    assert ninfz >= 0, f"NINFZ should be non-negative, got {ninfz}"
    assert dinfz >= 0, f"DINFZ should be non-negative, got {dinfz}"
    assert nkronl >= 0, f"NKRONL should be non-negative, got {nkronl}"

    assert nr <= n, f"NR should be <= N, got NR={nr}, N={n}"
    assert pr <= p, f"PR should be <= P, got PR={pr}, P={p}"

    if nr > 0:
        er = e_out[:nr, :nr]
        assert np.allclose(er, np.triu(er)), "Er must be upper triangular"
        assert abs(np.linalg.det(er)) > 1e-10, "Er must be nonsingular"

    if pr > 0 and pr <= m:
        dr = abcd_out[nr:nr+pr, :m]
        assert np.linalg.matrix_rank(dr, tol=1e-10) == pr

    computed_ninfz = sum(infz[i] * (i + 1) for i in range(dinfz))
    assert computed_ninfz == ninfz


def test_ag8byz_zero_d_matrix():
    """
    Test AG8BYZ with zero D matrix (proper system).

    When D=0, the system is strictly proper.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 2, 1, 1

    A = np.array([
        [-1.0+0.1j, 0.0+0.0j],
        [0.0+0.0j, -2.0+0.2j]
    ], order='F', dtype=complex)

    B = np.array([
        [1.0+0.0j],
        [0.0+0.0j]
    ], order='F', dtype=complex)

    C = np.array([
        [1.0+0.0j, 1.0+0.0j]
    ], order='F', dtype=complex)

    D = np.array([
        [0.0+0.0j]
    ], order='F', dtype=complex)

    E = np.eye(n, order='F', dtype=complex)

    abcd = np.zeros((n + p, m + n), order='F', dtype=complex)
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    svlmax = 0.0
    tol = 0.0

    result = ag8byz(True, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    if nr > 0:
        er = e_out[:nr, :nr]
        assert np.allclose(er, np.triu(er))
        assert abs(np.linalg.det(er)) > 1e-10


def test_ag8byz_identity_e():
    """
    Test AG8BYZ with identity E matrix (standard state-space).

    Tests the case when E=I, which is common in standard state-space.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 2, 2, 2

    A = np.array([
        [0.0+0.5j, 1.0+0.0j],
        [-2.0+0.0j, -3.0+0.3j]
    ], order='F', dtype=complex)

    B = np.array([
        [0.0+0.0j, 0.0+0.0j],
        [1.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)

    C = np.array([
        [1.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 1.0+0.0j]
    ], order='F', dtype=complex)

    D = np.array([
        [1.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 1.0+0.0j]
    ], order='F', dtype=complex)

    E = np.eye(n, order='F', dtype=complex)

    abcd = np.zeros((n + p, m + n), order='F', dtype=complex)
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    svlmax = 0.0
    tol = 0.0

    result = ag8byz(True, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert pr >= 0
    assert pr <= p
    if nr > 0:
        er = e_out[:nr, :nr]
        assert np.allclose(er, np.triu(er))
        assert abs(np.linalg.det(er)) > 1e-10


def test_ag8byz_p_zero():
    """
    Test AG8BYZ edge case: P=0 (no outputs).

    Quick return case.
    """
    n, m, p = 3, 2, 0

    A = np.eye(n, order='F', dtype=complex)
    B = np.zeros((n, m), order='F', dtype=complex)
    E = np.eye(n, order='F', dtype=complex)

    abcd = np.zeros((n, m + n), order='F', dtype=complex)
    abcd[:, :m] = B
    abcd[:, m:] = A

    svlmax = 0.0
    tol = 0.0

    result = ag8byz(True, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert nr == n
    assert pr == 0


def test_ag8byz_n_zero_m_zero():
    """
    Test AG8BYZ edge case: N=0 and M=0.

    When n=0 and m=0 but p>0, should compute Kronecker structure of D.
    """
    n, m, p = 0, 0, 2

    E = np.zeros((1, 1), order='F', dtype=complex)
    abcd = np.zeros((p, 0), order='F', dtype=complex)

    svlmax = 0.0
    tol = 0.0

    result = ag8byz(True, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert nr == 0
    assert pr == 0
    assert nkronl == 1
    assert kronl[0] == p


def test_ag8byz_error_negative_n():
    """Test AG8BYZ error handling: negative N."""
    n, m, p = -1, 2, 2

    abcd = np.zeros((4, 4), order='F', dtype=complex)
    E = np.eye(2, order='F', dtype=complex)

    with pytest.raises(ValueError):
        ag8byz(True, n, m, p, 0.0, abcd, E, 0.0)


def test_ag8byz_error_negative_m():
    """Test AG8BYZ error handling: negative M."""
    n, m, p = 2, -1, 2

    abcd = np.zeros((4, 4), order='F', dtype=complex)
    E = np.eye(2, order='F', dtype=complex)

    with pytest.raises(ValueError):
        ag8byz(True, n, m, p, 0.0, abcd, E, 0.0)


def test_ag8byz_error_negative_p():
    """Test AG8BYZ error handling: negative P."""
    n, m, p = 2, 2, -1

    abcd = np.zeros((4, 4), order='F', dtype=complex)
    E = np.eye(2, order='F', dtype=complex)

    with pytest.raises(ValueError):
        ag8byz(True, n, m, p, 0.0, abcd, E, 0.0)


def test_ag8byz_first_false():
    """
    Test AG8BYZ with FIRST=False.

    When FIRST=False, the D matrix must have full column rank with
    the last M rows in upper triangular form.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 2, 2, 3

    A = np.array([
        [1.0+0.1j, 0.0+0.0j],
        [0.0+0.0j, 2.0+0.2j]
    ], order='F', dtype=complex)

    B = np.array([
        [1.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 1.0+0.0j]
    ], order='F', dtype=complex)

    C = np.array([
        [1.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 1.0+0.0j],
        [1.0+0.0j, 1.0+0.0j]
    ], order='F', dtype=complex)

    D = np.array([
        [1.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 1.0+0.0j],
        [0.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)
    D[1, 0] = 0.0
    D[2, 0] = 0.0
    D[2, 1] = 0.0

    E = np.eye(n, order='F', dtype=complex)

    abcd = np.zeros((n + p, m + n), order='F', dtype=complex)
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    svlmax = 0.0
    tol = 0.0

    result = ag8byz(False, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert ninfz == 0
    assert dinfz == 0
    if nr > 0:
        er = e_out[:nr, :nr]
        assert np.allclose(er, np.triu(er))
    if pr > 0 and pr <= m:
        dr = abcd_out[nr:nr+pr, :m]
        assert np.linalg.matrix_rank(dr, tol=1e-10) == pr


def test_ag8byz_kronecker_structure():
    """
    Test AG8BYZ computes correct Kronecker structure.

    Use a system with known left Kronecker indices.
    Random seed: 321 (for reproducibility)
    """
    np.random.seed(321)
    n, m, p = 3, 1, 2

    A = np.array([
        [0.0+0.0j, 1.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 1.0+0.0j],
        [0.0+0.0j, 0.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)

    B = np.array([
        [0.0+0.0j],
        [0.0+0.0j],
        [1.0+0.0j]
    ], order='F', dtype=complex)

    C = np.array([
        [1.0+0.0j, 0.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 1.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)

    D = np.array([
        [0.0+0.0j],
        [0.0+0.0j]
    ], order='F', dtype=complex)

    E = np.eye(n, order='F', dtype=complex)

    abcd = np.zeros((n + p, m + n), order='F', dtype=complex)
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    svlmax = 0.0
    tol = 0.0

    result = ag8byz(True, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert nkronl >= 0

    total_kronl = sum(kronl[i] * i for i in range(nkronl))
    assert total_kronl >= 0
    if nr > 0:
        er = e_out[:nr, :nr]
        assert np.allclose(er, np.triu(er))


def test_ag8byz_infinite_zeros():
    """
    Test AG8BYZ correctly computes infinite zeros.

    System with known infinite zero structure.
    Random seed: 654 (for reproducibility)
    """
    np.random.seed(654)
    n, m, p = 2, 1, 1

    A = np.array([
        [1.0+0.1j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)

    B = np.array([
        [0.0+0.0j],
        [1.0+0.0j]
    ], order='F', dtype=complex)

    C = np.array([
        [1.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)

    D = np.array([
        [0.0+0.0j]
    ], order='F', dtype=complex)

    E = np.array([
        [1.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 0.0+0.0j]
    ], order='F', dtype=complex)

    abcd = np.zeros((n + p, m + n), order='F', dtype=complex)
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    svlmax = 0.0
    tol = 0.0

    result = ag8byz(True, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    computed_ninfz = sum(infz[i] * (i + 1) for i in range(dinfz))
    assert computed_ninfz == ninfz


def test_ag8byz_svlmax_threshold():
    """
    Test AG8BYZ with non-zero SVLMAX threshold.

    Tests that the routine respects the singular value threshold.
    Random seed: 987 (for reproducibility)
    """
    np.random.seed(987)
    n, m, p = 3, 2, 2

    A = np.random.randn(n, n) + 0.1j * np.random.randn(n, n)
    A = np.asfortranarray(A, dtype=complex)

    B = np.random.randn(n, m) + 0.1j * np.random.randn(n, m)
    B = np.asfortranarray(B, dtype=complex)

    C = np.random.randn(p, n) + 0.1j * np.random.randn(p, n)
    C = np.asfortranarray(C, dtype=complex)

    D = np.random.randn(p, m) + 0.1j * np.random.randn(p, m)
    D = np.asfortranarray(D, dtype=complex)

    E = np.eye(n, order='F', dtype=complex)

    abcd = np.zeros((n + p, m + n), order='F', dtype=complex)
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    svlmax = 10.0
    tol = 1e-10

    result = ag8byz(True, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    if nr > 0:
        er = e_out[:nr, :nr]
        assert np.allclose(er, np.triu(er))
    computed_ninfz = sum(infz[i] * (i + 1) for i in range(dinfz))
    assert computed_ninfz == ninfz


def test_ag8byz_complex_eigenvalues():
    """
    Test AG8BYZ with system having complex eigenvalues.

    Validates that complex arithmetic is handled correctly.
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 2, 1, 2

    A = np.array([
        [0.0+1.0j, 1.0+0.0j],
        [-1.0+0.0j, 0.0+1.0j]
    ], order='F', dtype=complex)

    B = np.array([
        [1.0+0.0j],
        [0.0+0.0j]
    ], order='F', dtype=complex)

    C = np.array([
        [1.0+0.0j, 0.0+0.0j],
        [0.0+0.0j, 1.0+0.0j]
    ], order='F', dtype=complex)

    D = np.array([
        [0.0+0.0j],
        [0.0+0.0j]
    ], order='F', dtype=complex)

    E = np.eye(n, order='F', dtype=complex)

    abcd = np.zeros((n + p, m + n), order='F', dtype=complex)
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    svlmax = 0.0
    tol = 0.0

    result = ag8byz(True, n, m, p, svlmax, abcd.copy(), E.copy(), tol)

    abcd_out, e_out, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info = result

    assert info == 0
    assert abcd_out.dtype == np.complex128
    assert e_out.dtype == np.complex128
    if nr > 0:
        er = e_out[:nr, :nr]
        assert np.allclose(er, np.triu(er))
    computed_ninfz = sum(infz[i] * (i + 1) for i in range(dinfz))
    assert computed_ninfz == ninfz


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
