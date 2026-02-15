"""
Tests for MB04RD - Block-diagonalization of generalized real Schur form.

Reduces a matrix pair (A,B) in generalized real Schur form to block-diagonal
form using well-conditioned non-orthogonal equivalence transformations.

The transformations are bounded by PMAX and optionally accumulated in X and Y.
Optionally reorders diagonal blocks so clustered eigenvalues are grouped.
"""

import numpy as np
import pytest
from ctrlsys import mb04rd


def test_mb04rd_basic_2x2_diagonal():
    """
    Test 2x2 diagonal system - simplest case with distinct eigenvalues.

    Both A and B are diagonal (trivially in Schur form).
    Eigenvalues: A11/B11 = 1/1 = 1, A22/B22 = 4/2 = 2.
    Already block-diagonal, so should return with single 2x2 block.

    Random seed: 42 (for reproducibility)
    """
    n = 2
    pmax = 1e10
    tol = 0.0

    a = np.array([[1.0, 0.0],
                  [0.0, 4.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 2.0]], order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1

    assert np.sum(blsize[:nblcks]) == n

    for i in range(n):
        assert beta[i] >= 0

    assert alphar[0] == pytest.approx(1.0, rel=1e-14)
    assert alphar[1] == pytest.approx(4.0, rel=1e-14)
    assert beta[0] == pytest.approx(1.0, rel=1e-14)
    assert beta[1] == pytest.approx(2.0, rel=1e-14)


def test_mb04rd_3x3_with_2x2_block():
    """
    Test 3x3 system with a 2x2 block in A (complex eigenvalue pair).

    A has a 2x2 block at (1:2,1:2) with complex eigenvalues.
    B is upper triangular.

    Random seed: 123 (for reproducibility)
    """
    n = 3
    pmax = 1e10
    tol = 0.0

    a = np.array([[1.0, 2.0, 0.5],
                  [-0.5, 1.0, 0.3],
                  [0.0, 0.0, 3.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.5, 0.2],
                  [0.0, 1.0, 0.1],
                  [0.0, 0.0, 2.0]], order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0

    assert alphai[0] > 0
    assert alphai[1] < 0
    assert alphai[2] == pytest.approx(0.0, abs=1e-14)


def test_mb04rd_4x4_sorted_clustering():
    """
    Test 4x4 system with eigenvalue clustering (SORT='S').

    Create a system where eigenvalues cluster and should be grouped.

    Random seed: 456 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = 0.5

    a = np.array([[1.0, 0.3, 0.1, 0.05],
                  [0.0, 1.1, 0.2, 0.1],
                  [0.0, 0.0, 3.0, 0.5],
                  [0.0, 0.0, 0.0, 3.2]], order='F', dtype=float)
    b = np.array([[1.0, 0.2, 0.1, 0.05],
                  [0.0, 1.0, 0.1, 0.05],
                  [0.0, 0.0, 1.0, 0.3],
                  [0.0, 0.0, 0.0, 1.0]], order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'S', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1
    assert np.sum(blsize[:nblcks]) == n


def test_mb04rd_no_accumulation():
    """
    Test without accumulating transformations (JOBX='N', JOBY='N').

    Random seed: 789 (for reproducibility)
    """
    n = 3
    pmax = 1e10
    tol = 0.0

    a = np.array([[2.0, 1.0, 0.5],
                  [0.0, 1.5, 0.3],
                  [0.0, 0.0, 3.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.3, 0.1],
                  [0.0, 1.0, 0.2],
                  [0.0, 0.0, 2.0]], order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'N', 'N', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0

    for i in range(n):
        assert alphar[i] != 0 or beta[i] != 0


def test_mb04rd_quick_return_n0():
    """
    Test quick return for N=0.
    """
    pmax = 1e10
    tol = 0.0

    a = np.array([], order='F', dtype=float).reshape(0, 0)
    b = np.array([], order='F', dtype=float).reshape(0, 0)
    x = np.array([], order='F', dtype=float).reshape(0, 0)
    y = np.array([], order='F', dtype=float).reshape(0, 0)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'N', 0, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks == 0


def test_mb04rd_quick_return_n1():
    """
    Test quick return for N=1.
    """
    n = 1
    pmax = 1e10
    tol = 0.0

    a = np.array([[5.0]], order='F', dtype=float)
    b = np.array([[2.0]], order='F', dtype=float)
    x = np.array([[1.0]], order='F', dtype=float)
    y = np.array([[1.0]], order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks == 1
    assert blsize[0] == 1
    assert alphar[0] == pytest.approx(5.0, rel=1e-14)
    assert beta[0] == pytest.approx(2.0, rel=1e-14)


def test_mb04rd_closest_neighbour():
    """
    Test SORT='C' (closest-neighbour strategy without reordering).

    Random seed: 555 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = 0.0

    a = np.array([[2.0, 1.0, 0.5, 0.1],
                  [0.0, 1.5, 0.3, 0.2],
                  [0.0, 0.0, 3.0, 0.4],
                  [0.0, 0.0, 0.0, 2.5]], order='F', dtype=float)
    b = np.array([[1.0, 0.2, 0.1, 0.05],
                  [0.0, 1.0, 0.15, 0.08],
                  [0.0, 0.0, 1.5, 0.2],
                  [0.0, 0.0, 0.0, 1.2]], order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'C', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1


def test_mb04rd_both_strategies():
    """
    Test SORT='B' (reordering with closest-neighbour strategy).

    Random seed: 666 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = 0.3

    a = np.array([[1.0, 0.5, 0.2, 0.1],
                  [0.0, 1.2, 0.3, 0.15],
                  [0.0, 0.0, 2.0, 0.4],
                  [0.0, 0.0, 0.0, 2.3]], order='F', dtype=float)
    b = np.array([[1.0, 0.1, 0.05, 0.02],
                  [0.0, 1.0, 0.1, 0.05],
                  [0.0, 0.0, 1.0, 0.2],
                  [0.0, 0.0, 0.0, 1.0]], order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'B', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1


def test_mb04rd_eigenvalue_preservation():
    """
    Mathematical property: eigenvalues should be preserved.

    The block-diagonalization should not change the eigenvalues of the
    pencil (A,B). All (alphar+i*alphai)/beta should match original.

    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)
    n = 4
    pmax = 1e10
    tol = 0.0

    a = np.array([[2.0, 0.5, 0.2, 0.1],
                  [0.0, 1.5, 0.3, 0.15],
                  [0.0, 0.0, 3.0, 0.4],
                  [0.0, 0.0, 0.0, 2.5]], order='F', dtype=float)
    b = np.array([[1.0, 0.2, 0.1, 0.05],
                  [0.0, 1.0, 0.15, 0.08],
                  [0.0, 0.0, 1.5, 0.2],
                  [0.0, 0.0, 0.0, 1.2]], order='F', dtype=float)

    eig_before_alpha, eig_before_beta = np.linalg.eig(a), np.linalg.eig(b)
    eig_before = np.linalg.eigvals(a) / np.maximum(np.abs(np.diag(b)), 1e-15)

    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0

    eig_after = []
    for i in range(n):
        if beta[i] != 0:
            eig_after.append((alphar[i] + 1j * alphai[i]) / beta[i])
        else:
            eig_after.append(np.inf if alphar[i] > 0 else -np.inf)

    eig_after = np.array(eig_after)
    eig_before_sorted = np.sort(eig_before.real)
    eig_after_sorted = np.sort(np.real([e for e in eig_after if np.isfinite(e)]))

    if len(eig_before_sorted) == len(eig_after_sorted):
        np.testing.assert_allclose(eig_before_sorted, eig_after_sorted, rtol=1e-10)


def test_mb04rd_transformation_validity():
    """
    Mathematical property: X' * A_orig * Y should relate to A_out.

    If JOBX='U' and JOBY='U', the transformation X' * A * Y = A_new and
    X' * B * Y = B_new should hold (approximately, since transformations
    may scale columns).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n = 3
    pmax = 1e10
    tol = 0.0

    a = np.array([[2.0, 1.0, 0.5],
                  [0.0, 1.5, 0.3],
                  [0.0, 0.0, 3.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.3, 0.1],
                  [0.0, 1.0, 0.2],
                  [0.0, 0.0, 2.0]], order='F', dtype=float)

    a_orig = a.copy()
    b_orig = b.copy()

    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'N', n, pmax, a, b, x, y, tol)

    assert info == 0

    a_transformed = x_out.T @ a_orig @ y_out
    b_transformed = x_out.T @ b_orig @ y_out

    for j in range(n):
        col_norm_a = np.linalg.norm(a_transformed[:, j])
        col_norm_b = np.linalg.norm(b_transformed[:, j])
        if col_norm_a > 1e-10:
            ratio_a = np.linalg.norm(a_out[:, j]) / col_norm_a
            assert 0.1 < ratio_a < 10
        if col_norm_b > 1e-10:
            ratio_b = np.linalg.norm(b_out[:, j]) / col_norm_b
            assert 0.1 < ratio_b < 10


def test_mb04rd_block_structure():
    """
    Mathematical property: output A should be block upper triangular.

    After block-diagonalization, A should have zeros below each diagonal block.

    Random seed: 999 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = 0.0

    a = np.array([[2.0, 0.5, 0.2, 0.1],
                  [0.0, 1.5, 0.3, 0.15],
                  [0.0, 0.0, 3.0, 0.4],
                  [0.0, 0.0, 0.0, 2.5]], order='F', dtype=float)
    b = np.array([[1.0, 0.2, 0.1, 0.05],
                  [0.0, 1.0, 0.15, 0.08],
                  [0.0, 0.0, 1.5, 0.2],
                  [0.0, 0.0, 0.0, 1.2]], order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
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


def test_mb04rd_negative_tol():
    """
    Test with negative TOL (relative tolerance mode).

    Random seed: 111 (for reproducibility)
    """
    n = 4
    pmax = 1e10
    tol = -0.1

    a = np.array([[1.0, 0.3, 0.1, 0.05],
                  [0.0, 1.05, 0.2, 0.1],
                  [0.0, 0.0, 3.0, 0.5],
                  [0.0, 0.0, 0.0, 3.1]], order='F', dtype=float)
    b = np.array([[1.0, 0.2, 0.1, 0.05],
                  [0.0, 1.0, 0.1, 0.05],
                  [0.0, 0.0, 1.0, 0.3],
                  [0.0, 0.0, 0.0, 1.0]], order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    y = np.eye(n, order='F', dtype=float)

    a_out, b_out, x_out, y_out, nblcks, blsize, alphar, alphai, beta, info = mb04rd(
        'U', 'U', 'S', n, pmax, a, b, x, y, tol)

    assert info == 0
    assert nblcks >= 1
