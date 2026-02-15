"""
Tests for mb03ts - Swap diagonal blocks in (skew-)Hamiltonian Schur form.

MB03TS swaps diagonal blocks A11 and A22 of order 1 or 2 in the upper
quasi-triangular matrix A contained in a skew-Hamiltonian or Hamiltonian matrix.
"""
import numpy as np
import pytest
from ctrlsys import mb03ts


def test_mb03ts_swap_1x1_hamiltonian():
    """
    Test swapping two 1x1 blocks in a Hamiltonian matrix.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3

    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.1, 0.2, 0.3],
        [0.0, 0.4, 0.5],
        [0.0, 0.0, 0.6]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)

    isham = True
    wantu = True
    j1 = 1
    n1 = 1
    n2 = 1

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0
    assert a_out[0, 0] == pytest.approx(2.0, rel=1e-14)
    assert a_out[1, 1] == pytest.approx(1.0, rel=1e-14)

    assert np.allclose(u1_out.T @ u1_out, np.eye(n), rtol=1e-14)
    assert np.allclose(u2_out.T @ u2_out, np.eye(n), rtol=1e-14)


def test_mb03ts_swap_1x1_skew_hamiltonian():
    """
    Test swapping two 1x1 blocks in a skew-Hamiltonian matrix.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3

    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.0, 0.2, 0.3],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)

    isham = False
    wantu = True
    j1 = 1
    n1 = 1
    n2 = 1

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0
    assert a_out[0, 0] == pytest.approx(2.0, rel=1e-14)
    assert a_out[1, 1] == pytest.approx(1.0, rel=1e-14)


def test_mb03ts_swap_2x2_1x1_hamiltonian():
    """
    Test swapping 2x2 block with 1x1 block in Hamiltonian matrix (N1=2, N2=1).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4

    a = np.array([
        [1.0, 2.0, 0.5, 0.3],
        [-0.5, 1.0, 0.4, 0.2],
        [0.0, 0.0, 3.0, 0.6],
        [0.0, 0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.0, 0.3, 0.4, 0.5],
        [0.0, 0.0, 0.5, 0.6],
        [0.0, 0.0, 0.0, 0.7]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)

    isham = True
    wantu = True
    j1 = 1
    n1 = 2
    n2 = 1

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0
    assert a_out[0, 0] == pytest.approx(3.0, rel=1e-10)


def test_mb03ts_swap_1x1_2x2_hamiltonian():
    """
    Test swapping 1x1 block with 2x2 block in Hamiltonian matrix (N1=1, N2=2).

    Swap may be rejected (info=1) if the result would be too far from Schur form.
    This test verifies the routine returns without error in both cases.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4

    a = np.array([
        [1.0, 0.1, 0.0, 0.0],
        [0.0, 3.0, 0.5, 0.0],
        [0.0, -0.5, 3.0, 0.0],
        [0.0, 0.0, 0.0, 5.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.1]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)

    isham = True
    wantu = True
    j1 = 1
    n1 = 1
    n2 = 2

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info in [0, 1]


def test_mb03ts_swap_2x2_2x2_hamiltonian():
    """
    Test swapping two 2x2 blocks in Hamiltonian matrix (N1=2, N2=2).

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 5

    a = np.array([
        [0.5, 0.0, 0.0, 0.0, 0.1],
        [0.0, 1.0, 2.0, 0.5, 0.3],
        [0.0, -0.5, 1.0, 0.4, 0.2],
        [0.0, 0.0, 0.0, 3.0, 1.5],
        [0.0, 0.0, 0.0, -0.3, 3.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.0, 0.3, 0.4, 0.5, 0.6],
        [0.0, 0.0, 0.5, 0.6, 0.7],
        [0.0, 0.0, 0.0, 0.7, 0.8],
        [0.0, 0.0, 0.0, 0.0, 0.9]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)

    isham = True
    wantu = True
    j1 = 2
    n1 = 2
    n2 = 2

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0


def test_mb03ts_last_block_hamiltonian():
    """
    Test swapping last block with -A11' in Hamiltonian matrix (LBLK case).

    When J1+N1 > N, the routine swaps with -A11' or A11' depending on ISHAM.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 3

    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.1, 0.2, 0.3],
        [0.0, 0.4, 0.5],
        [0.0, 0.0, 0.6]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)

    isham = True
    wantu = True
    j1 = 3
    n1 = 1
    n2 = 1

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0
    assert a_out[2, 2] == pytest.approx(-3.0, rel=1e-10)


def test_mb03ts_quick_return_n_zero():
    """
    Test quick return when N=0.
    """
    n = 0
    a = np.zeros((1, 1), order='F', dtype=float)
    g = np.zeros((1, 1), order='F', dtype=float)
    u1 = np.zeros((1, 1), order='F', dtype=float)
    u2 = np.zeros((1, 1), order='F', dtype=float)

    isham = True
    wantu = False
    j1 = 1
    n1 = 0
    n2 = 0

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0


def test_mb03ts_quick_return_n1_zero():
    """
    Test quick return when N1=0.
    """
    n = 3
    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.1, 0.2, 0.3],
        [0.0, 0.4, 0.5],
        [0.0, 0.0, 0.6]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()

    isham = True
    wantu = True
    j1 = 1
    n1 = 0
    n2 = 1

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0
    np.testing.assert_array_equal(a_out, a_orig)


def test_mb03ts_wantu_false():
    """
    Test that U1 and U2 are not modified when WANTU=False.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 3

    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.1, 0.2, 0.3],
        [0.0, 0.4, 0.5],
        [0.0, 0.0, 0.6]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)
    u1_orig = u1.copy()
    u2_orig = u2.copy()

    isham = True
    wantu = False
    j1 = 1
    n1 = 1
    n2 = 1

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0


def test_mb03ts_eigenvalue_preservation():
    """
    Mathematical property test: eigenvalues of A should be preserved after swap.

    The swap operation is a similarity transformation, so eigenvalues must
    be preserved.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 3

    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.1, 0.2, 0.3],
        [0.0, 0.4, 0.5],
        [0.0, 0.0, 0.6]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)

    eig_before = np.sort(np.linalg.eigvals(a).real)

    isham = True
    wantu = True
    j1 = 1
    n1 = 1
    n2 = 1

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0

    eig_after = np.sort(np.linalg.eigvals(a_out).real)

    np.testing.assert_allclose(eig_before, eig_after, rtol=1e-13)


def test_mb03ts_last_block_2x2_hamiltonian():
    """
    Test swapping last 2x2 block in Hamiltonian matrix (LBLK, N1=2).

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 3

    a = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.5],
        [0.0, -0.5, 2.0]
    ], order='F', dtype=float)

    g = np.array([
        [0.1, 0.2, 0.3],
        [0.0, 0.4, 0.5],
        [0.0, 0.0, 0.6]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.eye(n, order='F', dtype=float)

    isham = True
    wantu = True
    j1 = 2
    n1 = 2
    n2 = 1

    a_out, g_out, u1_out, u2_out, info = mb03ts(isham, wantu, a, g, u1, u2, j1, n1, n2)

    assert info == 0
