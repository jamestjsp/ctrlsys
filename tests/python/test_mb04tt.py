"""
Tests for MB04TT: Row compression with column echelon form.

Transforms submatrices (AA, EE) of A and E such that Aj is row compressed
while keeping Ej in column echelon form. This is step j of Algorithm 3.2.1
from Beelen's thesis for computing Kronecker structure of matrix pencils.

Key concepts:
- ISTAIR array encodes column echelon form boundary elements
  - +j: E(i,j) is a corner point (staircase corner)
  - -j: E(i,j) is on boundary but not a corner
- 4 boundary types (ITYPE 1-4) based on sign patterns of adjacent ISTAIR entries
- DLAPMT with forwrd=.FALSE. means forward permutation (restore original order)
"""

import numpy as np


def test_mb04tt_basic():
    """
    Test MB04TT with a basic column echelon form example.

    Creates a simple system where E has proper staircase structure.
    ISTAIR encodes boundary positions: +j = corner, -j = boundary not corner.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04tt

    np.random.seed(42)
    m, n = 3, 5
    ifira, ifica, nca = 1, 1, 2

    A = np.array([
        [1.0, 2.0, 0.5, 0.2, 0.1],
        [4.0, 5.0, 0.6, 0.3, 0.2],
        [7.0, 8.0, 0.7, 0.4, 0.3]
    ], order='F', dtype=float)

    E = np.array([
        [0.0, 0.0, 1.0, 0.5, 0.2],
        [0.0, 0.0, 0.0, 2.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 1.5]
    ], order='F', dtype=float)

    istair = np.array([3, 4, 5], dtype=np.int32)

    Q = np.eye(m, order='F', dtype=float)
    Z = np.eye(n, order='F', dtype=float)

    A_out, E_out, Q_out, Z_out, istair_out, rank, info = mb04tt(
        True, True, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )

    assert info == 0
    assert rank >= 0
    assert rank <= min(m - ifira + 1, nca)
    assert A_out.shape == (m, n)
    assert E_out.shape == (m, n)
    assert Q_out.shape == (m, m)
    assert Z_out.shape == (n, n)

    np.testing.assert_allclose(Q_out @ Q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(Z_out @ Z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb04tt_rank_deficient():
    """
    Test MB04TT with rank-deficient Aj submatrix.

    When Aj columns are linearly dependent, rank < min(mj, nj).
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04tt

    np.random.seed(123)
    m, n = 3, 4
    ifira, ifica, nca = 1, 1, 2

    A = np.array([
        [1.0, 2.0, 0.5, 0.2],
        [2.0, 4.0, 0.6, 0.3],
        [3.0, 6.0, 0.7, 0.4]
    ], order='F', dtype=float)

    E = np.array([
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    istair = np.array([3, 4, -4], dtype=np.int32)

    Q = np.eye(m, order='F', dtype=float)
    Z = np.eye(n, order='F', dtype=float)

    A_out, E_out, Q_out, Z_out, istair_out, rank, info = mb04tt(
        True, True, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )

    assert info == 0
    assert rank == 1

    np.testing.assert_allclose(Q_out @ Q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)


def test_mb04tt_no_update_q_z():
    """
    Test MB04TT with UPDATQ=False, UPDATZ=False.

    Q and Z should not be modified.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04tt

    np.random.seed(456)
    m, n = 3, 4
    ifira, ifica, nca = 1, 1, 2

    A = np.random.randn(m, n).astype(float, order='F')
    E = np.zeros((m, n), order='F', dtype=float)
    E[0, 2] = 1.0
    E[1, 3] = 1.0

    istair = np.array([3, 4, -4], dtype=np.int32)

    Q = np.eye(m, order='F', dtype=float)
    Z = np.eye(n, order='F', dtype=float)
    Q_orig = Q.copy()
    Z_orig = Z.copy()

    A_out, E_out, Q_out, Z_out, istair_out, rank, info = mb04tt(
        False, False, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )

    assert info == 0
    assert rank >= 0
    np.testing.assert_array_equal(Q_out, Q_orig)
    np.testing.assert_array_equal(Z_out, Z_orig)


def test_mb04tt_full_rank_3x3():
    """
    Test MB04TT with full-rank 3x3 case.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04tt

    np.random.seed(789)
    m, n = 3, 5
    ifira, ifica, nca = 1, 1, 3

    A = np.random.randn(m, n).astype(float, order='F')
    A[:, :nca] += np.eye(m, nca)

    E = np.zeros((m, n), order='F', dtype=float)
    E[0, 3] = 1.0
    E[0, 4] = 0.5
    E[1, 4] = 2.0

    istair = np.array([4, 5, -5], dtype=np.int32)

    Q = np.eye(m, order='F', dtype=float)
    Z = np.eye(n, order='F', dtype=float)

    A_out, E_out, Q_out, Z_out, istair_out, rank, info = mb04tt(
        True, True, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )

    assert info == 0
    assert rank == 3

    np.testing.assert_allclose(Q_out @ Q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(Z_out @ Z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb04tt_boundary_type_changes():
    """
    Test MB04TT with boundary type transitions.

    When Givens rotations modify E, boundary types (ISTAIR signs) may change.
    Types 2 and 4 can cause ISTAIR updates based on element magnitudes.

    Random seed: 321 (for reproducibility)
    """
    from ctrlsys import mb04tt

    np.random.seed(321)
    m, n = 4, 5
    ifira, ifica, nca = 1, 1, 2

    A = np.array([
        [2.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.3, 0.3, 0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    E = np.array([
        [0.0, 0.0, 1.0, 0.5, 0.2],
        [0.0, 0.0, 0.0, 2.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    istair = np.array([3, 4, 5, -5], dtype=np.int32)

    Q = np.eye(m, order='F', dtype=float)
    Z = np.eye(n, order='F', dtype=float)

    A_out, E_out, Q_out, Z_out, istair_out, rank, info = mb04tt(
        True, True, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )

    assert info == 0
    assert rank >= 0

    for i in range(m):
        assert abs(istair_out[i]) >= 1, f"ISTAIR[{i}] invalid: {istair_out[i]}"
        assert abs(istair_out[i]) <= n, f"ISTAIR[{i}] out of range: {istair_out[i]}"


def test_mb04tt_zero_dimensions():
    """
    Test MB04TT with M=0 or N=0 edge cases.

    Should return immediately with rank=0.
    """
    from ctrlsys import mb04tt

    A = np.array([[]], order='F', dtype=float).reshape(0, 3)
    E = np.array([[]], order='F', dtype=float).reshape(0, 3)
    Q = np.array([[]], order='F', dtype=float).reshape(0, 0)
    Z = np.eye(3, order='F', dtype=float)
    istair = np.array([], dtype=np.int32)

    A_out, E_out, Q_out, Z_out, istair_out, rank, info = mb04tt(
        True, True, 0, 3, 1, 1, 1, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )

    assert info == 0
    assert rank == 0


def test_mb04tt_single_column_aj():
    """
    Test MB04TT with single column Aj (NCA=1).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04tt

    np.random.seed(111)
    m, n = 3, 4
    ifira, ifica, nca = 1, 1, 1

    A = np.array([
        [2.0, 0.5, 0.2, 0.1],
        [1.0, 0.6, 0.3, 0.2],
        [3.0, 0.7, 0.4, 0.3]
    ], order='F', dtype=float)

    E = np.zeros((m, n), order='F', dtype=float)
    E[0, 1] = 1.0
    E[0, 2] = 0.5
    E[1, 2] = 2.0
    E[1, 3] = 0.3
    E[2, 3] = 1.5

    istair = np.array([2, 3, 4], dtype=np.int32)

    Q = np.eye(m, order='F', dtype=float)
    Z = np.eye(n, order='F', dtype=float)

    A_out, E_out, Q_out, Z_out, istair_out, rank, info = mb04tt(
        True, True, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )

    assert info == 0
    assert rank == 1

    np.testing.assert_allclose(Q_out @ Q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)


def test_mb04tt_submatrix_offset():
    """
    Test MB04TT with IFIRA > 1 and IFICA > 1.

    Only the submatrix A(IFIRA:M, IFICA:IFICA+NCA-1) is processed.
    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb04tt

    np.random.seed(222)
    m, n = 5, 6
    ifira, ifica, nca = 2, 2, 2

    A = np.random.randn(m, n).astype(float, order='F')
    A[1:, 1:3] += 0.5 * np.eye(4, 2)

    E = np.zeros((m, n), order='F', dtype=float)
    E[0, 0] = 1.0
    E[1, 3] = 1.0
    E[1, 4] = 0.5
    E[2, 4] = 2.0
    E[2, 5] = 0.3
    E[3, 5] = 1.5

    istair = np.array([1, 4, 5, 6, -6], dtype=np.int32)

    Q = np.eye(m, order='F', dtype=float)
    Z = np.eye(n, order='F', dtype=float)

    A_out, E_out, Q_out, Z_out, istair_out, rank, info = mb04tt(
        True, True, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )

    assert info == 0
    assert rank >= 0
    assert rank <= min(m - ifira + 1, nca)

    np.testing.assert_allclose(Q_out @ Q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)


def test_mb04tt_orthogonal_transformation_property():
    """
    Test that Q and Z are proper orthogonal transformations.

    Q'*Q = I, Z'*Z = I, and transformations preserve matrix relationships.
    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb04tt

    np.random.seed(333)
    m, n = 4, 5
    ifira, ifica, nca = 1, 1, 2

    A = np.random.randn(m, n).astype(float, order='F')
    E = np.zeros((m, n), order='F', dtype=float)
    E[0, 2] = 1.0
    E[0, 3] = 0.5
    E[1, 3] = 2.0
    E[1, 4] = 0.3
    E[2, 4] = 1.5

    istair = np.array([3, 4, 5, -5], dtype=np.int32)

    Q = np.eye(m, order='F', dtype=float)
    Z = np.eye(n, order='F', dtype=float)

    A_orig = A.copy()

    A_out, E_out, Q_out, Z_out, istair_out, rank, info = mb04tt(
        True, True, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )

    assert info == 0

    np.testing.assert_allclose(Q_out.T @ Q_out, np.eye(m), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(Q_out @ Q_out.T, np.eye(m), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(Z_out.T @ Z_out, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(Z_out @ Z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)

    np.testing.assert_allclose(abs(np.linalg.det(Q_out)), 1.0, rtol=1e-14)
    np.testing.assert_allclose(abs(np.linalg.det(Z_out)), 1.0, rtol=1e-14)


def test_mb04tt_tolerance_effect():
    """
    Test that tolerance affects rank determination correctly.

    Elements <= TOL should be treated as zero.
    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb04tt

    np.random.seed(444)
    m, n = 3, 4
    ifira, ifica, nca = 1, 1, 2

    A = np.array([
        [1.0, 0.0, 0.5, 0.2],
        [0.0, 1e-12, 0.6, 0.3],
        [0.0, 0.0, 0.7, 0.4]
    ], order='F', dtype=float)

    E = np.zeros((m, n), order='F', dtype=float)
    E[0, 2] = 1.0
    E[1, 3] = 1.0

    istair = np.array([3, 4, -4], dtype=np.int32)

    Q = np.eye(m, order='F', dtype=float)
    Z = np.eye(n, order='F', dtype=float)

    A_out, E_out, Q_out, Z_out, istair_out, rank_tight, info = mb04tt(
        True, True, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-14
    )
    assert info == 0

    A_out2, E_out2, Q_out2, Z_out2, istair_out2, rank_loose, info2 = mb04tt(
        True, True, m, n, ifira, ifica, nca, A.copy(), E.copy(),
        Q.copy(), Z.copy(), istair.copy(), 1e-10
    )
    assert info2 == 0

    assert rank_loose <= rank_tight
