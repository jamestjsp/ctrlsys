"""
Tests for AB01ND - Controllable realization for multi-input systems.

AB01ND finds a controllable realization for the linear time-invariant
multi-input system dX/dt = A * X + B * U, reducing (A,B) to orthogonal
canonical form where Acont is upper block Hessenberg with full row rank
subdiagonal blocks.
"""

import numpy as np
import pytest
from ctrlsys import ab01nd


"""Basic functionality tests from HTML documentation example."""

def test_html_doc_example():
    """
    Test case from SLICOT HTML documentation.

    Input: N=3, M=2, TOL=0.0, JOBZ='I'
    A read row-wise:
        -1.0   0.0   0.0
        -2.0  -2.0  -2.0
        -1.0   0.0  -3.0
    B read column-wise:
         1.0   0.0   0.0
         0.0   2.0   1.0

    Expected NCONT=2
    Expected INDCON=1
    Expected NBLK=[2]
    Expected Acont:
        -3.0000   2.2361
         0.0000  -1.0000
    Expected Bcont:
         0.0000  -2.2361
         1.0000   0.0000
    Expected Z:
         0.0000   1.0000   0.0000
        -0.8944   0.0000  -0.4472
        -0.4472   0.0000   0.8944
    """
    n, m = 3, 2

    # A read row-wise from HTML (standard row order)
    a = np.array([
        [-1.0,  0.0,  0.0],
        [-2.0, -2.0, -2.0],
        [-1.0,  0.0, -3.0]
    ], dtype=float, order='F')

    # B read column-wise: ((B(I,J), I=1,N), J=1,M)
    # First column: 1.0, 0.0, 0.0
    # Second column: 0.0, 2.0, 1.0
    b = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 1.0]
    ], dtype=float, order='F')

    tol = 0.0

    a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd('I', a, b, tol)

    assert info == 0
    assert ncont == 2
    assert indcon == 1

    # Expected nblk has indcon=1 element
    assert nblk[0] == 2

    # Expected Acont (NCONT x NCONT)
    a_expected = np.array([
        [-3.0000,  2.2361],
        [ 0.0000, -1.0000]
    ], dtype=float, order='F')

    # Expected Bcont (NCONT x M)
    b_expected = np.array([
        [0.0000, -2.2361],
        [1.0000,  0.0000]
    ], dtype=float, order='F')

    # Expected Z (N x N)
    z_expected = np.array([
        [ 0.0000,  1.0000,  0.0000],
        [-0.8944,  0.0000, -0.4472],
        [-0.4472,  0.0000,  0.8944]
    ], dtype=float, order='F')

    # Validate outputs - HTML doc shows 4 decimal places
    np.testing.assert_allclose(a_out[:ncont, :ncont], a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b_out[:ncont, :], b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z, z_expected, rtol=1e-3, atol=1e-4)


"""Test mathematical properties of orthogonal transformation."""

def test_z_is_orthogonal():
    """
    Validate Z'*Z = I (orthogonality property).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m = 4, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[0, 0] = 1.0  # Ensure non-zero for controllability

    _, _, ncont, indcon, nblk, z, _, info = ab01nd('I', a, b, 0.0)

    assert info == 0

    # Z should be orthogonal: Z'*Z = I
    ztz = z.T @ z
    np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14)

def test_similarity_transformation():
    """
    Validate A_out = Z' * A_in * Z (similarity transformation property).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m = 5, 2

    a_in = np.random.randn(n, n).astype(float, order='F')
    b_in = np.random.randn(n, m).astype(float, order='F')
    b_in[0, 0] = 2.0

    a_copy = a_in.copy()

    a_out, _, ncont, _, _, z, _, info = ab01nd('I', a_in, b_in, 0.0)

    assert info == 0

    # A_out = Z' * A_orig * Z
    a_transformed = z.T @ a_copy @ z
    np.testing.assert_allclose(a_out, a_transformed, rtol=1e-13, atol=1e-14)

def test_b_transformation():
    """
    Validate B_out = Z' * B_in (matrix transformation).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m = 4, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b_in = np.random.randn(n, m).astype(float, order='F')
    b_in[0, 0] = 1.5

    b_copy = b_in.copy()

    _, b_out, ncont, _, _, z, _, info = ab01nd('I', a, b_in, 0.0)

    assert info == 0

    # B_out = Z' * B_orig
    b_transformed = z.T @ b_copy
    np.testing.assert_allclose(b_out, b_transformed, rtol=1e-13, atol=1e-14)


"""Test upper block Hessenberg structure of output A."""

def test_output_is_block_hessenberg():
    """
    Validate output Acont has upper block Hessenberg structure.

    For block Hessenberg with blocks defined by NBLK, elements below
    the first block subdiagonal should be zero.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m = 6, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[0, 0] = 3.0

    a_out, _, ncont, indcon, nblk, _, _, info = ab01nd('I', a, b, 0.0)

    assert info == 0

    # Build block boundaries from nblk
    block_ends = []
    pos = 0
    for i in range(indcon):
        pos += nblk[i]
        block_ends.append(pos)

    # Check zeros below block subdiagonal
    # For each column block j, rows from block j+2 and beyond should be zero
    row_start = 0
    for block_idx in range(indcon):
        block_size = nblk[block_idx]
        col_end = block_ends[block_idx]

        # Rows from block (block_idx + 2) to ncont should be zero in these columns
        if block_idx + 2 <= indcon - 1:
            row_check_start = block_ends[block_idx + 1] if block_idx + 1 < len(block_ends) else ncont
            for j in range(row_start, col_end):
                for i in range(row_check_start, ncont):
                    assert abs(a_out[i, j]) < 1e-12, \
                        f"Element ({i},{j}) should be zero: {a_out[i, j]}"
        row_start = col_end

def test_subdiagonal_blocks_full_rank():
    """
    Validate subdiagonal blocks have full row rank.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m = 8, 4

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    b[:, 0] = np.array([1.0, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005])

    a_out, _, ncont, indcon, nblk, _, _, info = ab01nd('I', a, b, 0.0)

    assert info == 0

    if indcon > 1:
        # Check each subdiagonal block A_{i+1,i}
        row_start = 0
        col_start = 0
        for i in range(indcon - 1):
            block_rows = nblk[i + 1]
            block_cols = nblk[i]
            row_start_next = row_start + nblk[i]

            subblock = a_out[row_start_next:row_start_next + block_rows,
                             col_start:col_start + block_cols]

            # Full row rank means rank == number of rows
            rank = np.linalg.matrix_rank(subblock)
            assert rank == block_rows, \
                f"Subdiagonal block {i} has rank {rank}, expected {block_rows}"

            col_start += block_cols
            row_start = row_start_next


"""Test controllability detection."""

def test_partially_controllable_system():
    """
    Test system that is not fully controllable.

    Create a block diagonal A where B only affects some blocks.
    """
    n, m = 4, 1

    # Block diagonal A with two 2x2 blocks
    a = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    # B only affects first block
    b = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=float, order='F')

    _, _, ncont, indcon, nblk, _, _, info = ab01nd('I', a, b, 0.0)

    assert info == 0
    assert ncont < n  # Not fully controllable

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

    _, _, ncont, indcon, nblk, z, _, info = ab01nd('I', a, b, 0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0
    # For zero B with JOBZ='I', Z should be identity
    np.testing.assert_allclose(z, np.eye(n), rtol=1e-14, atol=1e-14)

def test_fully_controllable_random_system():
    """
    Test that random system is typically fully controllable.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m = 5, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    _, _, ncont, indcon, nblk, _, _, info = ab01nd('I', a, b, 0.0)

    assert info == 0
    assert ncont == n  # Fully controllable


"""Test different JOBZ modes."""

def test_jobz_n():
    """
    Test JOBZ='N' mode (no orthogonal transformation stored).
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

    a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd('N', a, b, 0.0)

    assert info == 0
    assert ncont == 2
    assert indcon == 1

def test_jobz_f():
    """
    Test JOBZ='F' mode (factored form storage).
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

    a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd('F', a, b, 0.0)

    assert info == 0
    assert ncont == 2
    assert indcon == 1


"""Test edge cases and boundary conditions."""

def test_n_equals_1_m_equals_1():
    """
    Test with N=1, M=1 (scalar system, single input).
    """
    a = np.array([[2.0]], dtype=float, order='F')
    b = np.array([[3.0]], dtype=float, order='F')

    a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd('I', a, b, 0.0)

    assert info == 0
    assert ncont == 1
    assert indcon == 1
    assert nblk[0] == 1

def test_n_equals_1_m_equals_2():
    """
    Test with N=1, M=2 (scalar system, two inputs).
    """
    a = np.array([[2.0]], dtype=float, order='F')
    b = np.array([[3.0, 4.0]], dtype=float, order='F')

    a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd('I', a, b, 0.0)

    assert info == 0
    assert ncont == 1
    assert indcon == 1

def test_n_equals_0():
    """
    Test with N=0 (empty system).
    """
    a = np.array([], dtype=float, order='F').reshape(0, 0)
    b = np.array([], dtype=float, order='F').reshape(0, 2)

    a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd('I', a, b, 0.0)

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

    a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd('I', a, b, 0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0
    # Z should be identity for JOBZ='I'
    np.testing.assert_allclose(z, np.eye(n), rtol=1e-14)

def test_large_system():
    """
    Test with larger system to verify scalability.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m = 20, 5

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    a_out, _, ncont, indcon, nblk, z, _, info = ab01nd('I', a, b, 0.0)

    assert info == 0

    # Z should be orthogonal
    ztz = z.T @ z
    np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-13, atol=1e-13)


"""Test tolerance handling."""

def test_explicit_tolerance():
    """
    Test with explicit positive tolerance.
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

    tol = 1e-10

    a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd('I', a, b, tol)

    assert info == 0
    assert ncont == 2
    assert indcon == 1

def test_negative_tolerance_uses_default():
    """
    Test that negative/zero tolerance uses default (N*N*EPS).
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

    # Zero tol means use default
    tol = 0.0

    a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd('I', a, b, tol)

    assert info == 0
    assert ncont == 2


"""Test error handling."""

def test_invalid_jobz():
    """
    Test invalid JOBZ parameter.
    """
    n, m = 3, 2
    a = np.eye(n, dtype=float, order='F')
    b = np.ones((n, m), dtype=float, order='F')

    with pytest.raises(ValueError, match="[Jj]obz"):
        ab01nd('X', a, b, 0.0)

def test_mismatched_dimensions():
    """
    Test mismatched A and B dimensions.
    """
    a = np.eye(3, dtype=float, order='F')
    b = np.ones((4, 2), dtype=float, order='F')  # Wrong number of rows

    with pytest.raises(ValueError):
        ab01nd('I', a, b, 0.0)


"""Test eigenvalue preservation under similarity transformation."""

def test_eigenvalues_preserved():
    """
    Validate eigenvalues are preserved by similarity transformation.

    Property: eigenvalues of A equal eigenvalues of Z'*A*Z.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n, m = 5, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    eig_before = np.linalg.eigvals(a)

    a_out, _, ncont, _, _, z, _, info = ab01nd('I', a, b, 0.0)

    assert info == 0

    eig_after = np.linalg.eigvals(a_out)

    # Sort by real then imaginary for comparison
    eig_before_sorted = np.sort_complex(eig_before)
    eig_after_sorted = np.sort_complex(eig_after)

    np.testing.assert_allclose(eig_before_sorted, eig_after_sorted,
                               rtol=1e-13, atol=1e-14)
