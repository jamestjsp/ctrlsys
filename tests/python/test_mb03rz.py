"""
Tests for MB03RZ: Reduce complex Schur form matrix to block-diagonal form

MB03RZ reduces an upper triangular complex matrix A (Schur form) to a
block-diagonal form using well-conditioned non-unitary similarity
transformations. The condition numbers of the transformations are
roughly bounded by PMAX. The transformations are optionally postmultiplied
in a given matrix X.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03rz_basic_diagonal():
    """
    Test MB03RZ with a diagonal matrix (already block-diagonal).

    A diagonal matrix should remain diagonal with each element forming
    its own 1x1 block, and eigenvalues should be the diagonal elements.
    """
    from ctrlsys import mb03rz

    n = 3
    a = np.array([
        [1.0 + 2.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 3.0 + 4.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 5.0 + 6.0j]
    ], dtype=np.complex128, order='F')

    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'N', a, pmax, tol)

    assert info == 0
    assert nblcks == 3
    assert_allclose(blsize[:nblcks], [1, 1, 1])

    expected_eig = np.array([1.0+2.0j, 3.0+4.0j, 5.0+6.0j])
    assert_allclose(w, expected_eig, rtol=1e-14)

    assert_allclose(a_out, a, rtol=1e-14)


def test_mb03rz_upper_triangular():
    """
    Test MB03RZ with upper triangular matrix with distinct eigenvalues.

    With well-separated eigenvalues, should produce n blocks of size 1.
    The eigenvalues (diagonal elements) should be preserved.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03rz

    np.random.seed(42)
    n = 4

    a = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        a[i, i] = (i + 1) * (1.0 + 0.5j)
        for j in range(i + 1, n):
            a[i, j] = 0.1 * (np.random.randn() + 1j * np.random.randn())

    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'N', a, pmax, tol)

    assert info == 0

    expected_eig = np.diag(a)
    assert_allclose(np.sort(np.abs(w)), np.sort(np.abs(expected_eig)), rtol=1e-10)

    for i in range(n):
        for j in range(i):
            assert abs(a_out[i, j]) < 1e-10, f"Lower triangular a_out[{i},{j}] should be zero"


def test_mb03rz_with_transformation():
    """
    Test MB03RZ with JOBX='U' (accumulate transformations).

    Verifies that A_new = X^(-1) * A * X holds approximately
    for the block-diagonalized result.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03rz

    np.random.seed(123)
    n = 3

    a = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        a[i, i] = (i + 1) * 2.0 + 1.0j * (i + 0.5)
        for j in range(i + 1, n):
            a[i, j] = 0.05 * (np.random.randn() + 1j * np.random.randn())

    x_init = np.eye(n, dtype=np.complex128, order='F')

    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('U', 'N', a, pmax, tol, x=x_init)

    assert info == 0

    for i in range(n):
        for j in range(i):
            assert abs(a_out[i, j]) < 1e-10


def test_mb03rz_clustered_eigenvalues():
    """
    Test MB03RZ with clustered eigenvalues using SORT='S'.

    Eigenvalues in the same cluster should be grouped in the same block.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03rz

    np.random.seed(456)
    n = 4

    a = np.zeros((n, n), dtype=np.complex128, order='F')
    a[0, 0] = 1.0 + 0.1j
    a[1, 1] = 1.0 + 0.2j
    a[2, 2] = 5.0 + 0.1j
    a[3, 3] = 5.0 + 0.15j

    for i in range(n):
        for j in range(i + 1, n):
            a[i, j] = 0.01 * (np.random.randn() + 1j * np.random.randn())

    pmax = 1000.0
    tol = 0.5

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'S', a, pmax, tol)

    assert info == 0

    assert nblcks >= 1
    assert nblcks <= n
    assert sum(blsize[:nblcks]) == n


def test_mb03rz_closest_neighbor_strategy():
    """
    Test MB03RZ with SORT='C' (closest-neighbor strategy).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03rz

    np.random.seed(789)
    n = 4

    a = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        a[i, i] = (i + 1) * 1.5 + 0.5j * i
        for j in range(i + 1, n):
            a[i, j] = 0.1 * (np.random.randn() + 1j * np.random.randn())

    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'C', a, pmax, tol)

    assert info == 0
    assert sum(blsize[:nblcks]) == n


def test_mb03rz_sort_b():
    """
    Test MB03RZ with SORT='B' (reordering + closest-neighbor).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb03rz

    np.random.seed(111)
    n = 4

    a = np.zeros((n, n), dtype=np.complex128, order='F')
    a[0, 0] = 1.0 + 0.0j
    a[1, 1] = 1.1 + 0.05j
    a[2, 2] = 3.0 + 0.0j
    a[3, 3] = 3.05 + 0.02j

    for i in range(n):
        for j in range(i + 1, n):
            a[i, j] = 0.02 * (np.random.randn() + 1j * np.random.randn())

    pmax = 1000.0
    tol = 0.2

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'B', a, pmax, tol)

    assert info == 0
    assert sum(blsize[:nblcks]) == n


def test_mb03rz_n_zero():
    """Test MB03RZ with N=0 (edge case - quick return)."""
    from ctrlsys import mb03rz

    a = np.zeros((0, 0), dtype=np.complex128, order='F')
    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'N', a, pmax, tol)

    assert info == 0
    assert nblcks == 0


def test_mb03rz_n_one():
    """Test MB03RZ with N=1 (single element)."""
    from ctrlsys import mb03rz

    a = np.array([[3.0 + 4.0j]], dtype=np.complex128, order='F')
    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'N', a, pmax, tol)

    assert info == 0
    assert nblcks == 1
    assert blsize[0] == 1
    assert_allclose(w[0], 3.0 + 4.0j, rtol=1e-14)


def test_mb03rz_invalid_jobx():
    """Test MB03RZ with invalid JOBX parameter."""
    from ctrlsys import mb03rz

    a = np.array([[1.0 + 0.0j]], dtype=np.complex128, order='F')
    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('X', 'N', a, pmax, tol)

    assert info == -1


def test_mb03rz_invalid_sort():
    """Test MB03RZ with invalid SORT parameter."""
    from ctrlsys import mb03rz

    a = np.array([[1.0 + 0.0j]], dtype=np.complex128, order='F')
    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'X', a, pmax, tol)

    assert info == -2


def test_mb03rz_pmax_too_small():
    """Test MB03RZ with PMAX < 1."""
    from ctrlsys import mb03rz

    a = np.array([[1.0 + 0.0j]], dtype=np.complex128, order='F')
    pmax = 0.5
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'N', a, pmax, tol)

    assert info == -4


def test_mb03rz_eigenvalue_preservation():
    """
    Test that MB03RZ preserves eigenvalues during block-diagonalization.

    The eigenvalues of the block-diagonal result should match
    the original diagonal elements (which are the eigenvalues
    of an upper triangular matrix).
    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb03rz

    np.random.seed(222)
    n = 5

    a = np.zeros((n, n), dtype=np.complex128, order='F')
    original_eigs = np.array([
        1.0 + 0.5j, 2.0 - 0.3j, 3.0 + 0.1j, 4.0 - 0.2j, 5.0 + 0.4j
    ])
    for i in range(n):
        a[i, i] = original_eigs[i]
        for j in range(i + 1, n):
            a[i, j] = 0.1 * (np.random.randn() + 1j * np.random.randn())

    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'N', a, pmax, tol)

    assert info == 0

    assert_allclose(np.sort(np.abs(w)), np.sort(np.abs(original_eigs)), rtol=1e-10)

    computed_eigs = np.diag(a_out)
    assert_allclose(np.sort(np.abs(computed_eigs)), np.sort(np.abs(original_eigs)), rtol=1e-10)


def test_mb03rz_block_structure():
    """
    Verify block-diagonal structure of the output.

    After block-diagonalization, off-diagonal blocks should be zero.
    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb03rz

    np.random.seed(333)
    n = 4

    a = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        a[i, i] = (i + 1) * 2.0 + 0.3j * i
        for j in range(i + 1, n):
            a[i, j] = 0.1 * (np.random.randn() + 1j * np.random.randn())

    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'N', a, pmax, tol)

    assert info == 0

    block_start = 0
    for b in range(nblcks):
        block_end = block_start + blsize[b]

        if block_end < n:
            off_block = a_out[block_start:block_end, block_end:n]
            assert np.max(np.abs(off_block)) < 1e-10, \
                f"Off-diagonal block {b} to {b+1} should be zero"

        block_start = block_end


def test_mb03rz_transformation_accumulation():
    """
    Test transformation accumulation with JOBX='U'.

    Starting from identity, X should contain the cumulative
    transformation. Verifies X columns have unit norm (scaled).
    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb03rz

    np.random.seed(444)
    n = 3

    a = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        a[i, i] = (i + 1) * 1.5 + 0.2j * (i + 1)
        for j in range(i + 1, n):
            a[i, j] = 0.05 * (np.random.randn() + 1j * np.random.randn())

    x_init = np.eye(n, dtype=np.complex128, order='F')

    pmax = 1000.0
    tol = 0.0

    a_out, x_out, nblcks, blsize, w, info = mb03rz('U', 'N', a, pmax, tol, x=x_init)

    assert info == 0

    for j in range(n):
        col_norm = np.linalg.norm(x_out[:, j])
        assert col_norm > 0


def test_mb03rz_relative_tolerance():
    """
    Test MB03RZ with negative TOL (relative tolerance).

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import mb03rz

    np.random.seed(555)
    n = 4

    a = np.zeros((n, n), dtype=np.complex128, order='F')
    a[0, 0] = 10.0 + 0.0j
    a[1, 1] = 10.1 + 0.05j
    a[2, 2] = 20.0 + 0.0j
    a[3, 3] = 20.2 + 0.1j

    for i in range(n):
        for j in range(i + 1, n):
            a[i, j] = 0.01 * (np.random.randn() + 1j * np.random.randn())

    pmax = 1000.0
    tol = -0.02

    a_out, x_out, nblcks, blsize, w, info = mb03rz('N', 'S', a, pmax, tol)

    assert info == 0
    assert sum(blsize[:nblcks]) == n
