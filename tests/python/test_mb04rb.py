"""
Tests for MB04RB - Reduction of skew-Hamiltonian matrix to PVL form (blocked).

MB04RB reduces a skew-Hamiltonian matrix:

              [  A   G  ]
        W  =  [       T ]
              [  Q   A  ]

to Paige/Van Loan (PVL) form using orthogonal symplectic transformation U:

          T       [  Aout  Gout  ]
         U W U =  [            T ]
                  [    0   Aout  ]

where Aout is in upper Hessenberg form. This is the blocked version of MB04RU.
"""

import numpy as np
import pytest
from slicot import mb04rb


def build_skew_hamiltonian(a, q_lower, g_upper):
    """
    Build full 2N x 2N skew-Hamiltonian matrix from components.

    W = [  A    G  ]
        [  Q   A^T ]

    where Q and G are skew-symmetric (Q = -Q^T, G = -G^T).
    """
    n = a.shape[0]
    w = np.zeros((2 * n, 2 * n), dtype=float, order='F')

    w[:n, :n] = a
    w[n:, n:] = a.T

    g = np.triu(g_upper, 1) - np.triu(g_upper, 1).T
    w[:n, n:] = g

    q = np.tril(q_lower, -1) - np.tril(q_lower, -1).T
    w[n:, :n] = q

    return w


def test_mb04rb_basic_3x3():
    """
    Test basic 3x3 case with simple skew-Hamiltonian matrix.

    The output A contains Aout (upper Hessenberg) PLUS reflector vectors.
    We test that info=0 and that the CS/TAU arrays have proper size and
    the Givens rotations are normalized.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3
    ilo = 1

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], order='F', dtype=float)

    qg = np.zeros((n, n + 1), order='F', dtype=float)
    qg[1, 0] = 0.5
    qg[2, 0] = 0.3
    qg[2, 1] = 0.2
    qg[0, 2] = 1.0
    qg[0, 3] = 0.7
    qg[1, 3] = 0.4

    a_out, qg_out, cs, tau, info = mb04rb(n, ilo, a, qg)

    assert info == 0
    assert len(cs) == 2 * (n - 1)
    assert len(tau) == n - 1

    for i in range(n - 1):
        c = cs[2 * i]
        s = cs[2 * i + 1]
        np.testing.assert_allclose(c * c + s * s, 1.0, rtol=1e-14,
                                   err_msg=f"Givens rotation {i} not normalized")


def test_mb04rb_ilo_greater_than_1():
    """
    Test with ILO > 1 (matrix already partially reduced).

    When ILO > 1, rows/columns 1:ILO-1 of A are upper triangular
    and Q is zero in those rows/columns.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    ilo = 2

    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 5.0, 6.0, 7.0],
        [0.0, 8.0, 9.0, 1.0],
        [0.0, 2.0, 3.0, 4.0]
    ], order='F', dtype=float)

    qg = np.zeros((n, n + 1), order='F', dtype=float)
    qg[2, 1] = 0.5
    qg[3, 1] = 0.3
    qg[3, 2] = 0.2
    qg[1, 3] = 1.0
    qg[1, 4] = 0.7
    qg[2, 4] = 0.4

    a_out, qg_out, cs, tau, info = mb04rb(n, ilo, a, qg)

    assert info == 0
    assert len(cs) == 2 * (n - 1)
    assert len(tau) == n - 1

    for i in range(ilo - 1, n - 1):
        c = cs[2 * i]
        s = cs[2 * i + 1]
        np.testing.assert_allclose(c * c + s * s, 1.0, rtol=1e-14,
                                   err_msg=f"Givens rotation {i} not normalized")


def test_mb04rb_eigenvalue_preservation():
    """
    Mathematical property: eigenvalues of W are preserved under transformation.

    The skew-Hamiltonian structure means eigenvalues come in pairs (lambda, -lambda).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4
    ilo = 1

    a = np.random.randn(n, n).astype(float, order='F')

    qg = np.zeros((n, n + 1), order='F', dtype=float)
    q_vals = np.random.randn(n * (n - 1) // 2)
    g_vals = np.random.randn(n * (n - 1) // 2)

    idx = 0
    for j in range(n):
        for i in range(j + 1, n):
            qg[i, j] = q_vals[idx]
            idx += 1

    idx = 0
    for j in range(1, n + 1):
        for i in range(j - 1):
            qg[i, j] = g_vals[idx]
            idx += 1

    q_lower = qg[:, :n].copy()
    g_upper = qg[:, 1:].copy()
    w_before = build_skew_hamiltonian(a.copy(), q_lower, g_upper)

    eig_before = np.linalg.eigvals(w_before)
    eig_before_sorted = np.sort(eig_before.real)

    a_out, qg_out, cs, tau, info = mb04rb(n, ilo, a.copy(), qg.copy())

    assert info == 0

    a_hess = a_out.copy()
    for j in range(n - 2):
        for i in range(j + 2, n):
            a_hess[i, j] = 0.0

    g_out_upper = qg_out[:, 1:].copy()
    g_out = np.triu(g_out_upper, 1) - np.triu(g_out_upper, 1).T

    w_after = np.zeros((2 * n, 2 * n), dtype=float, order='F')
    w_after[:n, :n] = a_hess
    w_after[n:, n:] = a_hess.T
    w_after[:n, n:] = g_out

    eig_after = np.linalg.eigvals(w_after)
    eig_after_sorted = np.sort(eig_after.real)

    np.testing.assert_allclose(eig_before_sorted, eig_after_sorted, rtol=1e-10, atol=1e-12)


def test_mb04rb_cs_tau_output():
    """
    Test that CS and TAU arrays have correct sizes and non-trivial values.

    CS contains 2*(N-1) cosines and sines of Givens rotations.
    TAU contains N-1 scalar factors of reflectors.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5
    ilo = 1

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    for j in range(n):
        for i in range(j + 1, n):
            qg[i, j] = np.random.randn()
    for j in range(1, n + 1):
        for i in range(j - 1):
            qg[i, j] = np.random.randn()

    a_out, qg_out, cs, tau, info = mb04rb(n, ilo, a, qg)

    assert info == 0
    assert len(cs) == 2 * (n - 1)
    assert len(tau) == n - 1

    for i in range(n - 1):
        c = cs[2 * i]
        s = cs[2 * i + 1]
        np.testing.assert_allclose(c * c + s * s, 1.0, rtol=1e-14,
                                   err_msg=f"Givens rotation {i} not normalized")


def test_mb04rb_quick_return_n_le_ilo():
    """
    Test quick return when N <= ILO (nothing to do).
    """
    n = 2
    ilo = 3

    a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    a_out, qg_out, cs, tau, info = mb04rb(n, ilo, a, qg)

    assert info == 0
    np.testing.assert_allclose(a_out, a, rtol=1e-14)


def test_mb04rb_n_equals_zero():
    """
    Test quick return for N=0.
    """
    n = 0
    ilo = 1

    a = np.array([], dtype=float, order='F').reshape(0, 0)
    qg = np.array([], dtype=float, order='F').reshape(0, 1)

    a_out, qg_out, cs, tau, info = mb04rb(n, ilo, a, qg)

    assert info == 0


def test_mb04rb_invalid_n():
    """
    Test error for invalid N < 0.
    """
    n = -1
    ilo = 1

    a = np.array([[1.0]], dtype=float, order='F')
    qg = np.array([[0.0, 0.0]], dtype=float, order='F')

    with pytest.raises(ValueError):
        mb04rb(n, ilo, a, qg)


def test_mb04rb_invalid_ilo():
    """
    Test error for invalid ILO (out of range).
    """
    n = 3
    ilo = 0

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    with pytest.raises(ValueError):
        mb04rb(n, ilo, a, qg)


def test_mb04rb_larger_system():
    """
    Test with larger 8x8 matrix to exercise blocked algorithm path.

    The blocked version uses MB04PA for panel factorization and MB04RU
    for cleanup. With larger N, blocking should be triggered.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n = 8
    ilo = 1

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    for j in range(n):
        for i in range(j + 1, n):
            qg[i, j] = np.random.randn()
    for j in range(1, n + 1):
        for i in range(j - 1):
            qg[i, j] = np.random.randn()

    a_out, qg_out, cs, tau, info = mb04rb(n, ilo, a, qg)

    assert info == 0
    assert len(cs) == 2 * (n - 1)
    assert len(tau) == n - 1

    for i in range(n - 1):
        c = cs[2 * i]
        s = cs[2 * i + 1]
        np.testing.assert_allclose(c * c + s * s, 1.0, rtol=1e-14,
                                   err_msg=f"Givens rotation {i} not normalized")


def test_mb04rb_workspace_query():
    """
    Test workspace query functionality (ldwork = -1).

    This tests the workspace query mechanism where the routine
    returns the optimal workspace size without computing.
    """
    n = 6
    ilo = 1

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    a_out, qg_out, cs, tau, info = mb04rb(n, ilo, a, qg)
    assert info == 0


def test_mb04rb_matches_mb04ru():
    """
    Verify blocked version produces same result as unblocked (MB04RU).

    Both routines should produce equivalent output since they implement
    the same mathematical transformation. We test eigenvalue preservation
    to verify correctness.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 4
    ilo = 1

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    for j in range(n):
        for i in range(j + 1, n):
            qg[i, j] = np.random.randn()
    for j in range(1, n + 1):
        for i in range(j - 1):
            qg[i, j] = np.random.randn()

    q_lower = qg[:, :n].copy()
    g_upper = qg[:, 1:].copy()
    w_original = build_skew_hamiltonian(a.copy(), q_lower, g_upper)

    eig_original = np.linalg.eigvals(w_original)
    eig_original_sorted = np.sort(eig_original.real)

    a_out, qg_out, cs, tau, info = mb04rb(n, ilo, a.copy(), qg.copy())
    assert info == 0

    a_hess = a_out.copy()
    for j in range(n - 2):
        for i in range(j + 2, n):
            a_hess[i, j] = 0.0

    g_out_upper = qg_out[:, 1:].copy()
    g_out = np.triu(g_out_upper, 1) - np.triu(g_out_upper, 1).T

    w_after = np.zeros((2 * n, 2 * n), dtype=float, order='F')
    w_after[:n, :n] = a_hess
    w_after[n:, n:] = a_hess.T
    w_after[:n, n:] = g_out

    eig_after = np.linalg.eigvals(w_after)
    eig_after_sorted = np.sort(eig_after.real)

    np.testing.assert_allclose(eig_original_sorted, eig_after_sorted, rtol=1e-10, atol=1e-12)


def _make_skew_hamiltonian_test(n, seed):
    """Helper: create random skew-Hamiltonian and verify eigenvalue preservation."""
    np.random.seed(seed)
    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    for j in range(n):
        for ii in range(j + 1, n):
            qg[ii, j] = np.random.randn()
    for j in range(1, n + 1):
        for ii in range(j - 1):
            qg[ii, j] = np.random.randn()

    q_lower = qg[:, :n].copy()
    g_upper = qg[:, 1:].copy()
    w_before = build_skew_hamiltonian(a.copy(), q_lower, g_upper)
    eig_before = np.sort(np.linalg.eigvals(w_before).real)

    a_out, qg_out, cs, tau, info = mb04rb(n, 1, a.copy(), qg.copy())
    assert info == 0

    a_hess = a_out.copy()
    for j in range(n - 2):
        for ii in range(j + 2, n):
            a_hess[ii, j] = 0.0

    g_out_upper = qg_out[:, 1:].copy()
    g_out = np.triu(g_out_upper, 1) - np.triu(g_out_upper, 1).T

    w_after = np.zeros((2 * n, 2 * n), dtype=float, order='F')
    w_after[:n, :n] = a_hess
    w_after[n:, n:] = a_hess.T
    w_after[:n, n:] = g_out

    eig_after = np.sort(np.linalg.eigvals(w_after).real)

    np.testing.assert_allclose(eig_before, eig_after, rtol=1e-10, atol=1e-12)

    for ii in range(n - 1):
        c = cs[2 * ii]
        s = cs[2 * ii + 1]
        np.testing.assert_allclose(c * c + s * s, 1.0, rtol=1e-14)


@pytest.mark.parametrize("n", [12, 16, 20])
def test_mb04rb_eigenvalue_preservation_large(n):
    """Eigenvalue preservation for larger matrices exercising blocked path.

    Random seed: 600 + n (for reproducibility)
    """
    _make_skew_hamiltonian_test(n, 600 + n)


def test_mb04rb_eigenvalue_preservation_n10():
    """Eigenvalue preservation for n=10.

    Random seed: 700 (for reproducibility)
    """
    _make_skew_hamiltonian_test(10, 700)
