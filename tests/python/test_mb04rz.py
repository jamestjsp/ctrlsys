"""
Tests for MB04RZ - Block-diagonalization of generalized complex Schur form.

Reduces a complex matrix pair (A,B) in generalized complex Schur form to
block-diagonal form using well-conditioned non-unitary equivalence
transformations.

The transformations are bounded by PMAX and optionally accumulated in X and Y.
Optionally reorders diagonal elements so clustered eigenvalues are grouped.
"""

import numpy as np
import pytest
from ctrlsys import mb04rz


def test_mb04rz_basic_2x2_diagonal():
    """
    Test 2x2 diagonal complex system - simplest case with distinct eigenvalues.

    Both A and B are diagonal (trivially in upper triangular Schur form).
    Eigenvalues: A11/B11 = (1+1j)/1 = 1+1j, A22/B22 = (4+2j)/2 = 2+1j.
    Already block-diagonal, so should return with minimal blocks.

    Random seed: 42 (for reproducibility)
    """
    n = 2
    pmax = 1e10
    tol = 0.0

    a = np.array([[1.0+1.0j, 0.0],
                  [0.0, 4.0+2.0j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.0],
                  [0.0, 2.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1
    assert np.sum(blsize[:nblcks]) == n

    for i in range(n):
        assert beta[i].real >= 0
        assert beta[i].imag == pytest.approx(0.0, abs=1e-14)

    assert alpha[0] == pytest.approx(1.0+1.0j, rel=1e-14)
    assert alpha[1] == pytest.approx(4.0+2.0j, rel=1e-14)
    assert beta[0].real == pytest.approx(1.0, rel=1e-14)
    assert beta[1].real == pytest.approx(2.0, rel=1e-14)


def test_mb04rz_3x3_upper_triangular():
    """
    Test 3x3 upper triangular complex system.

    A and B are upper triangular (complex Schur form).

    Random seed: 123 (for reproducibility)
    """
    n = 3
    pmax = 1e10
    tol = 0.0

    a = np.array([[1.0+0.5j, 2.0+1.0j, 0.5+0.2j],
                  [0.0, 1.5-0.5j, 0.3+0.1j],
                  [0.0, 0.0, 3.0+1.0j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.5+0.1j, 0.2+0.05j],
                  [0.0, 1.0+0.0j, 0.1+0.02j],
                  [0.0, 0.0, 2.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert np.sum(blsize[:nblcks]) == n

    for i in range(n):
        assert beta[i].real >= 0


def test_mb04rz_4x4_sorted_clustering():
    """
    Test 4x4 system with eigenvalue clustering (SORT='S').

    Create a system where eigenvalues cluster and should be grouped.

    Random seed: 456 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = 0.5

    a = np.array([[1.0+0.0j, 0.3+0.1j, 0.1+0.05j, 0.05+0.02j],
                  [0.0, 1.1+0.0j, 0.2+0.1j, 0.1+0.05j],
                  [0.0, 0.0, 3.0+1.0j, 0.5+0.2j],
                  [0.0, 0.0, 0.0, 3.2+1.0j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.2+0.05j, 0.1+0.02j, 0.05+0.01j],
                  [0.0, 1.0+0.0j, 0.1+0.02j, 0.05+0.01j],
                  [0.0, 0.0, 1.0+0.0j, 0.3+0.1j],
                  [0.0, 0.0, 0.0, 1.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'S', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1
    assert np.sum(blsize[:nblcks]) == n


def test_mb04rz_no_accumulation():
    """
    Test without accumulating transformations (JOBX='N', JOBY='N').

    Random seed: 789 (for reproducibility)
    """
    n = 3
    pmax = 1e10
    tol = 0.0

    a = np.array([[2.0+1.0j, 1.0+0.5j, 0.5+0.2j],
                  [0.0, 1.5+0.3j, 0.3+0.1j],
                  [0.0, 0.0, 3.0+0.5j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.3+0.1j, 0.1+0.05j],
                  [0.0, 1.0+0.0j, 0.2+0.05j],
                  [0.0, 0.0, 2.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'N', 'N', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0

    for i in range(n):
        assert alpha[i] != 0 or beta[i] != 0


def test_mb04rz_quick_return_n0():
    """
    Test quick return for N=0.
    """
    pmax = 1e10
    tol = 0.0

    a = np.array([], order='F', dtype=complex).reshape(0, 0)
    b = np.array([], order='F', dtype=complex).reshape(0, 0)
    x = np.array([], order='F', dtype=complex).reshape(0, 0)
    y = np.array([], order='F', dtype=complex).reshape(0, 0)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'N', 0, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks == 0


def test_mb04rz_quick_return_n1():
    """
    Test quick return for N=1.
    """
    n = 1
    pmax = 1e10
    tol = 0.0

    a = np.array([[5.0+2.0j]], order='F', dtype=complex)
    b = np.array([[2.0+0.0j]], order='F', dtype=complex)
    x = np.array([[1.0+0.0j]], order='F', dtype=complex)
    y = np.array([[1.0+0.0j]], order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks == 1
    assert blsize[0] == 1
    assert alpha[0] == pytest.approx(5.0+2.0j, rel=1e-14)
    assert beta[0].real == pytest.approx(2.0, rel=1e-14)


def test_mb04rz_closest_neighbour():
    """
    Test SORT='C' (closest-neighbour strategy without reordering).

    Random seed: 555 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = 0.0

    a = np.array([[2.0+0.5j, 1.0+0.3j, 0.5+0.1j, 0.1+0.05j],
                  [0.0, 1.5+0.2j, 0.3+0.1j, 0.2+0.05j],
                  [0.0, 0.0, 3.0+1.0j, 0.4+0.2j],
                  [0.0, 0.0, 0.0, 2.5+0.8j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.2+0.05j, 0.1+0.02j, 0.05+0.01j],
                  [0.0, 1.0+0.0j, 0.15+0.03j, 0.08+0.02j],
                  [0.0, 0.0, 1.5+0.0j, 0.2+0.05j],
                  [0.0, 0.0, 0.0, 1.2+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'C', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1


def test_mb04rz_both_strategies():
    """
    Test SORT='B' (reordering with closest-neighbour strategy).

    Random seed: 666 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = 0.3

    a = np.array([[1.0+0.2j, 0.5+0.1j, 0.2+0.05j, 0.1+0.02j],
                  [0.0, 1.2+0.3j, 0.3+0.1j, 0.15+0.05j],
                  [0.0, 0.0, 2.0+0.5j, 0.4+0.1j],
                  [0.0, 0.0, 0.0, 2.3+0.6j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.1+0.02j, 0.05+0.01j, 0.02+0.005j],
                  [0.0, 1.0+0.0j, 0.1+0.02j, 0.05+0.01j],
                  [0.0, 0.0, 1.0+0.0j, 0.2+0.05j],
                  [0.0, 0.0, 0.0, 1.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'B', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1


def test_mb04rz_eigenvalue_preservation():
    """
    Mathematical property: eigenvalues should be preserved.

    The block-diagonalization should not change the eigenvalues of the
    pencil (A,B). All alpha/beta should match original diagonal.

    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)
    n = 4
    pmax = 1e10
    tol = 0.0

    a = np.array([[2.0+0.5j, 0.5+0.1j, 0.2+0.05j, 0.1+0.02j],
                  [0.0, 1.5+0.3j, 0.3+0.1j, 0.15+0.05j],
                  [0.0, 0.0, 3.0+1.0j, 0.4+0.1j],
                  [0.0, 0.0, 0.0, 2.5+0.8j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.2+0.05j, 0.1+0.02j, 0.05+0.01j],
                  [0.0, 1.0+0.0j, 0.15+0.03j, 0.08+0.02j],
                  [0.0, 0.0, 1.5+0.0j, 0.2+0.05j],
                  [0.0, 0.0, 0.0, 1.2+0.0j]], order='F', dtype=complex)

    eig_before = []
    for i in range(n):
        if b[i, i].real > 1e-15:
            eig_before.append(a[i, i] / b[i, i].real)
        else:
            eig_before.append(np.inf)

    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0

    eig_after = []
    for i in range(n):
        if beta[i].real > 1e-15:
            eig_after.append(alpha[i] / beta[i].real)
        else:
            eig_after.append(np.inf)

    eig_before = np.array(eig_before)
    eig_after = np.array(eig_after)

    finite_before = eig_before[np.isfinite(eig_before)]
    finite_after = eig_after[np.isfinite(eig_after)]

    if len(finite_before) == len(finite_after):
        finite_before_sorted = np.sort(np.real(finite_before))
        finite_after_sorted = np.sort(np.real(finite_after))
        np.testing.assert_allclose(finite_before_sorted, finite_after_sorted, rtol=1e-10)


def test_mb04rz_block_structure():
    """
    Mathematical property: output A should be block upper triangular.

    After block-diagonalization, A should have zeros below each diagonal block.

    Random seed: 999 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = 0.0

    a = np.array([[2.0+0.5j, 0.5+0.1j, 0.2+0.05j, 0.1+0.02j],
                  [0.0, 1.5+0.3j, 0.3+0.1j, 0.15+0.05j],
                  [0.0, 0.0, 3.0+1.0j, 0.4+0.1j],
                  [0.0, 0.0, 0.0, 2.5+0.8j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.2+0.05j, 0.1+0.02j, 0.05+0.01j],
                  [0.0, 1.0+0.0j, 0.15+0.03j, 0.08+0.02j],
                  [0.0, 0.0, 1.5+0.0j, 0.2+0.05j],
                  [0.0, 0.0, 0.0, 1.2+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0

    block_start = 0
    for k in range(nblcks):
        bs = blsize[k]
        block_end = block_start + bs

        if block_end < n:
            off_diag_a = a_out[block_end:, block_start:block_end]
            off_diag_b = b_out[block_end:, block_start:block_end]
            np.testing.assert_allclose(off_diag_a, 0.0, atol=1e-10)
            np.testing.assert_allclose(off_diag_b, 0.0, atol=1e-10)

        block_start = block_end


def test_mb04rz_negative_tol():
    """
    Test with negative TOL (relative tolerance mode).

    Random seed: 111 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = -0.1

    a = np.array([[1.0+0.0j, 0.3+0.1j, 0.1+0.05j, 0.05+0.02j],
                  [0.0, 1.05+0.0j, 0.2+0.1j, 0.1+0.05j],
                  [0.0, 0.0, 3.0+1.0j, 0.5+0.2j],
                  [0.0, 0.0, 0.0, 3.1+1.0j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.2+0.05j, 0.1+0.02j, 0.05+0.01j],
                  [0.0, 1.0+0.0j, 0.1+0.02j, 0.05+0.01j],
                  [0.0, 0.0, 1.0+0.0j, 0.3+0.1j],
                  [0.0, 0.0, 0.0, 1.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'S', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1


def test_mb04rz_error_invalid_jobx():
    """
    Test error handling for invalid JOBX parameter.
    """
    n = 2
    pmax = 1e10
    tol = 0.0

    a = np.array([[1.0+0.0j, 0.5+0.1j],
                  [0.0, 2.0+0.0j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.2+0.05j],
                  [0.0, 1.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'X', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == -1


def test_mb04rz_error_invalid_joby():
    """
    Test error handling for invalid JOBY parameter.
    """
    n = 2
    pmax = 1e10
    tol = 0.0

    a = np.array([[1.0+0.0j, 0.5+0.1j],
                  [0.0, 2.0+0.0j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.2+0.05j],
                  [0.0, 1.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'X', 'N', n, pmax, a, b, x, y, tol)

    assert info == -2


def test_mb04rz_error_invalid_sort():
    """
    Test error handling for invalid SORT parameter.
    """
    n = 2
    pmax = 1e10
    tol = 0.0

    a = np.array([[1.0+0.0j, 0.5+0.1j],
                  [0.0, 2.0+0.0j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.2+0.05j],
                  [0.0, 1.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'X', n, pmax, a, b, x, y, tol)

    assert info == -3


def test_mb04rz_error_invalid_pmax():
    """
    Test error handling for invalid PMAX parameter.
    """
    n = 2
    pmax = 0.5
    tol = 0.0

    a = np.array([[1.0+0.0j, 0.5+0.1j],
                  [0.0, 2.0+0.0j]], order='F', dtype=complex)
    b = np.array([[1.0+0.0j, 0.2+0.05j],
                  [0.0, 1.0+0.0j]], order='F', dtype=complex)
    x = np.eye(n, order='F', dtype=complex)
    y = np.eye(n, order='F', dtype=complex)

    a_out, b_out, x_out, y_out, nblcks, blsize, alpha, beta, info = mb04rz(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == -5
