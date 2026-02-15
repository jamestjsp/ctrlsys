"""
Tests for AB01OD - Staircase form for multi-input systems.

AB01OD reduces matrices A and B using orthogonal state-space and input-space
transformations U and V such that Ac = U'*A*U, Bc = U'*B*V are in upper
staircase form, with Acont in upper block Hessenberg form with full row
rank subdiagonal blocks.
"""

import numpy as np
import pytest
from ctrlsys import ab01od


"""Basic functionality tests from HTML documentation example."""

def test_html_doc_example_forward_stage():
    """
    Test case from SLICOT HTML documentation with STAGES='F'.

    Input: N=5, M=2, TOL=0.0, STAGES='F', JOBU='N', JOBV='N'
    A matrix (column-wise from doc):
        17.0  24.0   1.0   8.0  15.0
        23.0   5.0   7.0  14.0  16.0
         4.0   6.0  13.0  20.0  22.0
        10.0  12.0  19.0  21.0   3.0
        11.0  18.0  25.0   2.0   9.0
    B matrix:
        -1.0  -4.0
         4.0   9.0
        -9.0 -16.0
        16.0  25.0
       -25.0 -36.0

    Expected results from HTML doc (4 decimal places):
    A transformed:
       12.8848   3.2345  11.8211   3.3758  -0.8982
        4.4741 -12.5544   5.3509   5.9403   1.4360
       14.4576   7.6855  23.1452  26.3872 -29.9557
        0.0000   1.4805  27.4668  22.6564  -0.0072
        0.0000   0.0000 -30.4822   0.6745  18.8680

    B transformed:
       31.1199  47.6865
        3.2480   0.0000
        0.0000   0.0000
        0.0000   0.0000
        0.0000   0.0000

    NCONT = 5 (fully controllable)
    INDCON = 3
    KSTAIR = [2, 2, 1]
    """
    n, m = 5, 2

    a = np.array([
        [17.0, 24.0,  1.0,  8.0, 15.0],
        [23.0,  5.0,  7.0, 14.0, 16.0],
        [ 4.0,  6.0, 13.0, 20.0, 22.0],
        [10.0, 12.0, 19.0, 21.0,  3.0],
        [11.0, 18.0, 25.0,  2.0,  9.0]
    ], dtype=float, order='F')

    b = np.array([
        [ -1.0,  -4.0],
        [  4.0,   9.0],
        [ -9.0, -16.0],
        [ 16.0,  25.0],
        [-25.0, -36.0]
    ], dtype=float, order='F')

    tol = 0.0

    a_copy = a.copy()
    b_copy = b.copy()

    a_out, b_out, u, v, ncont, indcon, kstair, info = ab01od(
        'F', 'I', 'N', a, b, tol
    )

    assert info == 0
    assert ncont == 5
    assert indcon == 3

    kstair_expected = [2, 2, 1]
    np.testing.assert_array_equal(kstair[:indcon], kstair_expected)

    b_zeros = b_out[kstair[0]:, :]
    np.testing.assert_allclose(b_zeros, np.zeros_like(b_zeros), rtol=0, atol=1e-14)

    utu = u.T @ u
    np.testing.assert_allclose(utu, np.eye(n), rtol=1e-14, atol=1e-14)

    a_transformed = u.T @ a_copy @ u
    np.testing.assert_allclose(a_out, a_transformed, rtol=1e-13, atol=1e-14)

    b_transformed = u.T @ b_copy
    np.testing.assert_allclose(b_out, b_transformed, rtol=1e-12, atol=1e-13)


"""Test all stages mode (forward + backward)."""

def test_all_stages_with_transformations():
    """
    Test STAGES='A' with JOBU='I', JOBV='I' to get both U and V matrices.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m = 4, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[0, 0] = 2.0

    a_copy = a.copy()
    b_copy = b.copy()

    a_out, b_out, u, v, ncont, indcon, kstair, info = ab01od(
        'A', 'I', 'I', a, b, 0.0
    )

    assert info == 0

    utu = u.T @ u
    np.testing.assert_allclose(utu, np.eye(n), rtol=1e-14, atol=1e-14)

    vtv = v.T @ v
    np.testing.assert_allclose(vtv, np.eye(m), rtol=1e-14, atol=1e-14)

    a_transformed = u.T @ a_copy @ u
    np.testing.assert_allclose(a_out, a_transformed, rtol=1e-13, atol=1e-14)

    b_transformed = u.T @ b_copy @ v
    np.testing.assert_allclose(b_out, b_transformed, rtol=1e-13, atol=1e-14)


"""Test orthogonality properties of transformation matrices."""

def test_u_is_orthogonal():
    """
    Validate U'*U = I (orthogonality property).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m = 5, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[0, 0] = 1.0

    _, _, u, _, ncont, _, _, info = ab01od('F', 'I', 'N', a, b, 0.0)

    assert info == 0

    utu = u.T @ u
    np.testing.assert_allclose(utu, np.eye(n), rtol=1e-14, atol=1e-14)

def test_v_is_orthogonal_all_stages():
    """
    Validate V'*V = I when STAGES='A' and JOBV='I'.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m = 4, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[0, 0] = 1.5

    _, _, _, v, _, _, _, info = ab01od('A', 'N', 'I', a, b, 0.0)

    assert info == 0

    vtv = v.T @ v
    np.testing.assert_allclose(vtv, np.eye(m), rtol=1e-14, atol=1e-14)


"""Test similarity transformation properties."""

def test_a_transformation_forward():
    """
    Validate A_out = U' * A_in * U for forward stage.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m = 5, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[0, 0] = 2.0

    a_copy = a.copy()

    a_out, _, u, _, ncont, _, _, info = ab01od('F', 'I', 'N', a, b, 0.0)

    assert info == 0

    a_transformed = u.T @ a_copy @ u
    np.testing.assert_allclose(a_out, a_transformed, rtol=1e-13, atol=1e-14)

def test_b_transformation_forward():
    """
    Validate B_out = U' * B_in for forward stage.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m = 4, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[0, 0] = 1.0

    b_copy = b.copy()

    _, b_out, u, _, ncont, _, _, info = ab01od('F', 'I', 'N', a, b, 0.0)

    assert info == 0

    b_transformed = u.T @ b_copy
    np.testing.assert_allclose(b_out, b_transformed, rtol=1e-13, atol=1e-14)


"""Test eigenvalue preservation under similarity transformation."""

def test_eigenvalues_preserved():
    """
    Validate eigenvalues of A are preserved by U'*A*U.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n, m = 5, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    eig_before = np.linalg.eigvals(a)

    a_out, _, _, _, ncont, _, _, info = ab01od('A', 'N', 'N', a, b, 0.0)

    assert info == 0

    eig_after = np.linalg.eigvals(a_out)

    eig_before_sorted = np.sort_complex(eig_before)
    eig_after_sorted = np.sort_complex(eig_after)

    np.testing.assert_allclose(eig_before_sorted, eig_after_sorted,
                               rtol=1e-12, atol=1e-13)


"""Test staircase/block Hessenberg structure of output."""

def test_zeros_below_subdiagonal_blocks():
    """
    Validate zeros below first block-subdiagonal in A.

    For upper block Hessenberg form, elements below first block
    subdiagonal should be zero.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m = 6, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[0, 0] = 3.0

    a_out, _, _, _, ncont, indcon, kstair, info = ab01od('A', 'N', 'N', a, b, 0.0)

    assert info == 0

    if indcon >= 2:
        block_ends = []
        pos = 0
        for i in range(indcon):
            pos += kstair[i]
            block_ends.append(pos)

        for block_idx in range(indcon - 2):
            row_start = block_ends[block_idx + 1] if block_idx + 1 < len(block_ends) else ncont
            col_end = block_ends[block_idx]

            for j in range(col_end):
                for i in range(row_start, ncont):
                    assert abs(a_out[i, j]) < 1e-12, \
                        f"Element ({i},{j}) should be zero: {a_out[i, j]}"

def test_b_zeros_below_first_block():
    """
    Validate all B rows except first block are zero.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n, m = 5, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[0, 0] = 2.5

    _, b_out, _, _, ncont, indcon, kstair, info = ab01od('F', 'N', 'N', a, b, 0.0)

    assert info == 0

    if indcon > 0:
        first_block_size = kstair[0]
        zeros_below = b_out[first_block_size:, :]
        np.testing.assert_allclose(zeros_below, np.zeros_like(zeros_below),
                                   rtol=0, atol=1e-14)


"""Test controllability detection."""

def test_partially_controllable_system():
    """
    Test system that is not fully controllable.
    """
    n, m = 4, 1

    a = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    b = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=float, order='F')

    _, _, _, _, ncont, indcon, kstair, info = ab01od('F', 'N', 'N', a, b, 0.0)

    assert info == 0
    assert ncont < n

def test_zero_b_matrix():
    """
    Test with zero B matrix - completely uncontrollable.
    """
    n, m = 3, 2

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=float, order='F')

    b = np.zeros((n, m), dtype=float, order='F')

    _, _, u, _, ncont, indcon, kstair, info = ab01od('F', 'I', 'N', a, b, 0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0
    np.testing.assert_allclose(u, np.eye(n), rtol=1e-14, atol=1e-14)


"""Test different STAGES modes."""

def test_forward_stage_only():
    """
    Test STAGES='F' mode (forward stage only).
    """
    n, m = 3, 2

    a = np.array([
        [-1.0,  0.0,  0.0],
        [-2.0, -2.0, -2.0],
        [-1.0,  0.0, -3.0]
    ], dtype=float, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 1.0]
    ], dtype=float, order='F')

    _, _, _, _, ncont, indcon, kstair, info = ab01od('F', 'N', 'N', a, b, 0.0)

    assert info == 0

def test_all_stages():
    """
    Test STAGES='A' mode (all stages).
    """
    n, m = 3, 2

    a = np.array([
        [-1.0,  0.0,  0.0],
        [-2.0, -2.0, -2.0],
        [-1.0,  0.0, -3.0]
    ], dtype=float, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 1.0]
    ], dtype=float, order='F')

    _, _, _, _, ncont, indcon, kstair, info = ab01od('A', 'N', 'N', a, b, 0.0)

    assert info == 0


"""Test edge cases and boundary conditions."""

def test_n_equals_1():
    """
    Test with N=1 (scalar system).
    """
    a = np.array([[2.0]], dtype=float, order='F')
    b = np.array([[3.0, 4.0]], dtype=float, order='F')

    a_out, b_out, u, v, ncont, indcon, kstair, info = ab01od(
        'A', 'I', 'I', a, b, 0.0
    )

    assert info == 0
    assert ncont == 1
    assert indcon == 1
    assert kstair[0] == 1

def test_n_equals_0():
    """
    Test with N=0 (empty system).
    """
    a = np.array([], dtype=float, order='F').reshape(0, 0)
    b = np.array([], dtype=float, order='F').reshape(0, 2)

    a_out, b_out, u, v, ncont, indcon, kstair, info = ab01od(
        'F', 'I', 'N', a, b, 0.0
    )

    assert info == 0
    assert ncont == 0
    assert indcon == 0

def test_m_equals_0():
    """
    Test with M=0 (no inputs).
    """
    n = 3
    a = np.eye(n, dtype=float, order='F')
    b = np.array([], dtype=float, order='F').reshape(n, 0)

    _, _, u, _, ncont, indcon, kstair, info = ab01od('F', 'I', 'N', a, b, 0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0
    np.testing.assert_allclose(u, np.eye(n), rtol=1e-14)


"""Test error handling."""

def test_invalid_stages():
    """
    Test invalid STAGES parameter.
    """
    n, m = 3, 2
    a = np.eye(n, dtype=float, order='F')
    b = np.ones((n, m), dtype=float, order='F')

    with pytest.raises(ValueError, match="[Ss]tages"):
        ab01od('X', 'N', 'N', a, b, 0.0)

def test_invalid_jobu():
    """
    Test invalid JOBU parameter.
    """
    n, m = 3, 2
    a = np.eye(n, dtype=float, order='F')
    b = np.ones((n, m), dtype=float, order='F')

    with pytest.raises(ValueError, match="[Jj]obu"):
        ab01od('F', 'X', 'N', a, b, 0.0)

def test_mismatched_dimensions():
    """
    Test mismatched A and B dimensions.
    """
    a = np.eye(3, dtype=float, order='F')
    b = np.ones((4, 2), dtype=float, order='F')

    with pytest.raises(ValueError):
        ab01od('F', 'N', 'N', a, b, 0.0)

def test_non_square_a():
    """
    Test non-square A matrix.
    """
    a = np.ones((3, 4), dtype=float, order='F')
    b = np.ones((3, 2), dtype=float, order='F')

    with pytest.raises(ValueError):
        ab01od('F', 'N', 'N', a, b, 0.0)
