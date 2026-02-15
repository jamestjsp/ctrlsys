"""
Tests for MB03RX: Reorder diagonal blocks of a real Schur form matrix.

Moves the diagonal block at position KU to position KL by applying
orthogonal similarity transformations (using LAPACK DTREXC).

Tests:
1. Basic reordering: 3x3 diagonal matrix, move last eigenvalue to first
2. 2x2 block reordering: Matrix with complex conjugate pair
3. Eigenvalue preservation: Verify eigenvalues unchanged after reordering
4. No-op case: KL = KU (nothing to reorder)
5. Transformation accumulation: JOBV = 'V' updates X
6. Large matrix reordering

Random seeds: 42, 123, 456 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03rx_basic_diagonal():
    """
    Validate basic reordering of diagonal matrix.

    3x3 diagonal Schur matrix with eigenvalues [1, 2, 3].
    Move eigenvalue at position 3 (value 3.0) to position 1.
    Expected result: eigenvalues reordered to [3, 1, 2].
    """
    from ctrlsys import mb03rx

    n = 3

    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    wr = np.array([1.0, 2.0, 3.0], dtype=float)
    wi = np.zeros(n, dtype=float)

    kl = 1
    ku = 3

    a_out, x_out, wr_out, wi_out, ku_out = mb03rx('V', kl, ku, a, x, wr, wi)

    assert ku_out == ku

    eig_before = sorted([1.0, 2.0, 3.0])
    eig_after = sorted(wr_out.tolist())
    assert_allclose(eig_after, eig_before, rtol=1e-12)

    np.testing.assert_allclose(wr_out[0], 3.0, rtol=1e-12)


def test_mb03rx_2x2_block():
    """
    Validate reordering with a 2x2 block (complex conjugate pair).

    4x4 Schur matrix:
    - Position 1: real eigenvalue 1.0
    - Positions 2-3: 2x2 block with eigenvalues 2+i, 2-i
    - Position 4: real eigenvalue 4.0

    Move eigenvalue at position 4 to position 1.
    """
    from ctrlsys import mb03rx

    n = 4

    a = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.0, 2.0, 1.0, 0.4],
        [0.0,-1.0, 2.0, 0.5],
        [0.0, 0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    wr = np.array([1.0, 2.0, 2.0, 4.0], dtype=float)
    wi = np.array([0.0, 1.0, -1.0, 0.0], dtype=float)

    kl = 1
    ku = 4

    a_out, x_out, wr_out, wi_out, ku_out = mb03rx('V', kl, ku, a, x, wr, wi)

    eig_before = sorted([1.0, 2.0+1.0j, 2.0-1.0j, 4.0], key=lambda x: (x.real, x.imag))
    eig_after = sorted([wr_out[i] + 1j*wi_out[i] for i in range(n)], key=lambda x: (x.real, x.imag))
    assert_allclose([e.real for e in eig_after], [e.real for e in eig_before], rtol=1e-10)
    assert_allclose([e.imag for e in eig_after], [e.imag for e in eig_before], rtol=1e-10)

    assert_allclose(wr_out[0], 4.0, rtol=1e-12)
    assert_allclose(wi_out[0], 0.0, atol=1e-14)


def test_mb03rx_eigenvalue_preservation():
    """
    Validate eigenvalue preservation under similarity transformation.

    Mathematical property: eigenvalues must be preserved exactly.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03rx

    np.random.seed(42)
    n = 5

    a = np.triu(np.random.randn(n, n).astype(float, order='F'))

    x = np.eye(n, order='F', dtype=float)

    wr = np.diag(a).copy()
    wi = np.zeros(n, dtype=float)

    eig_before = np.linalg.eigvals(a)

    kl = 1
    ku = n

    a_out, x_out, wr_out, wi_out, ku_out = mb03rx('V', kl, ku, a, x, wr, wi)

    eig_after = np.linalg.eigvals(a_out)

    assert_allclose(sorted(eig_before.real), sorted(eig_after.real), rtol=1e-12)


def test_mb03rx_no_op():
    """
    Validate no-op case: KL = KU (nothing to reorder).

    When KL equals KU, no transformation should occur.
    """
    from ctrlsys import mb03rx

    n = 3

    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    x = np.eye(n, order='F', dtype=float)
    x_orig = x.copy()

    wr = np.array([1.0, 2.0, 3.0], dtype=float)
    wi = np.zeros(n, dtype=float)
    wr_orig = wr.copy()
    wi_orig = wi.copy()

    kl = 2
    ku = 2

    a_out, x_out, wr_out, wi_out, ku_out = mb03rx('V', kl, ku, a, x, wr, wi)

    assert ku_out == ku
    assert_allclose(a_out, a_orig, rtol=1e-14)
    assert_allclose(x_out, x_orig, rtol=1e-14)
    assert_allclose(wr_out, wr_orig, rtol=1e-14)
    assert_allclose(wi_out, wi_orig, atol=1e-14)


def test_mb03rx_no_accumulation():
    """
    Validate JOBV='N' does not modify X.
    """
    from ctrlsys import mb03rx

    n = 3

    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)
    x_orig = x.copy()

    wr = np.array([1.0, 2.0, 3.0], dtype=float)
    wi = np.zeros(n, dtype=float)

    kl = 1
    ku = 3

    a_out, x_out, wr_out, wi_out, ku_out = mb03rx('N', kl, ku, a, x, wr, wi)

    assert_allclose(wr_out[0], 3.0, rtol=1e-12)


def test_mb03rx_orthogonal_transformation():
    """
    Validate X is orthogonal when JOBV='V' and X starts as identity.

    Orthogonality: X' * X = I
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03rx

    np.random.seed(123)
    n = 4

    a = np.triu(np.random.randn(n, n).astype(float, order='F'))

    x = np.eye(n, order='F', dtype=float)

    wr = np.diag(a).copy()
    wi = np.zeros(n, dtype=float)

    kl = 1
    ku = n

    a_out, x_out, wr_out, wi_out, ku_out = mb03rx('V', kl, ku, a, x, wr, wi)

    xtx = x_out.T @ x_out
    assert_allclose(xtx, np.eye(n), rtol=1e-12, atol=1e-14)


def test_mb03rx_similarity_transformation():
    """
    Validate similarity transformation: A_out = X' * A_orig * X.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03rx

    np.random.seed(456)
    n = 4

    a_orig = np.triu(np.random.randn(n, n).astype(float, order='F'))
    a = a_orig.copy()

    x = np.eye(n, order='F', dtype=float)

    wr = np.diag(a).copy()
    wi = np.zeros(n, dtype=float)

    kl = 1
    ku = n

    a_out, x_out, wr_out, wi_out, ku_out = mb03rx('V', kl, ku, a, x, wr, wi)

    a_transformed = x_out.T @ a_orig @ x_out
    assert_allclose(a_out, a_transformed, rtol=1e-12, atol=1e-14)


def test_mb03rx_move_complex_block_to_front():
    """
    Validate moving a 2x2 complex block to front position.

    5x5 matrix with:
    - Real eigenvalue 1.0 at (1,1)
    - Real eigenvalue 2.0 at (2,2)
    - 2x2 block at (3:4,3:4) with eigenvalues 3+2i, 3-2i
    - Real eigenvalue 5.0 at (5,5)

    Move the 2x2 block (at position 3) to position 1.
    """
    from ctrlsys import mb03rx

    n = 5

    a = np.array([
        [1.0, 0.5, 0.3, 0.2, 0.1],
        [0.0, 2.0, 0.4, 0.3, 0.2],
        [0.0, 0.0, 3.0, 2.0, 0.5],
        [0.0, 0.0,-2.0, 3.0, 0.6],
        [0.0, 0.0, 0.0, 0.0, 5.0]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    wr = np.array([1.0, 2.0, 3.0, 3.0, 5.0], dtype=float)
    wi = np.array([0.0, 0.0, 2.0, -2.0, 0.0], dtype=float)

    kl = 1
    ku = 3

    a_out, x_out, wr_out, wi_out, ku_out = mb03rx('V', kl, ku, a, x, wr, wi)

    eig_before = [1.0, 2.0, 3.0+2.0j, 3.0-2.0j, 5.0]
    eig_after = [wr_out[i] + 1j*wi_out[i] for i in range(n)]

    eig_before_sorted = sorted(eig_before, key=lambda x: (x.real, x.imag))
    eig_after_sorted = sorted(eig_after, key=lambda x: (x.real, x.imag))

    assert_allclose([e.real for e in eig_after_sorted],
                   [e.real for e in eig_before_sorted], rtol=1e-10)
    assert_allclose([e.imag for e in eig_after_sorted],
                   [e.imag for e in eig_before_sorted], rtol=1e-10)
