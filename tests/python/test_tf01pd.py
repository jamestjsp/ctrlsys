"""Tests for TF01PD - Block Toeplitz expansion of multivariable parameter sequence."""

import numpy as np
import pytest
from ctrlsys import tf01pd


def test_html_doc_example():
    """
    Test case from SLICOT HTML documentation.

    NH1=2, NH2=2, NR=3, NC=3
    H contains M(1),...,M(5) as 2x2 blocks stored horizontally.
    T is 6x6 block Toeplitz matrix.

    Toeplitz structure:
        T = | M(NC)     M(NC-1)   ...  M(1)      |
            | M(NC+1)   M(NC)     ...  M(2)      |
            | ...       ...       ...  ...       |
            | M(NR+NC-1) M(NR+NC-2) ... M(NR)    |

    Data read column-wise: ((H(I,J), I=1,NH1), J=1,(NR+NC-1)*NH2)
    """
    nh1, nh2, nr, nc = 2, 2, 3, 3

    h_flat = np.array([
        1.0647, -0.4282, -0.4922, -1.2072,
        -0.3043, 0.6883, -0.0926, 0.7167,
        -0.1844, -0.8507, 0.4441, -0.0478,
        0.7195, 0.0500, -0.3955, 0.5674,
        1.3387, -0.2801, 0.1073, -0.5315
    ], dtype=float)
    h = h_flat.reshape((nh1, (nr + nc - 1) * nh2), order='F')

    t_expected = np.array([
        [-0.1844, 0.4441, -0.3043, -0.0926, 1.0647, -0.4922],
        [-0.8507, -0.0478, 0.6883, 0.7167, -0.4282, -1.2072],
        [0.7195, -0.3955, -0.1844, 0.4441, -0.3043, -0.0926],
        [0.0500, 0.5674, -0.8507, -0.0478, 0.6883, 0.7167],
        [1.3387, 0.1073, 0.7195, -0.3955, -0.1844, 0.4441],
        [-0.2801, -0.5315, 0.0500, 0.5674, -0.8507, -0.0478]
    ], order='F', dtype=float)

    t, info = tf01pd(h, nr, nc)

    assert info == 0
    np.testing.assert_allclose(t, t_expected, rtol=1e-4, atol=1e-5)


def test_toeplitz_structure_property():
    """
    Validate Toeplitz structure: block (i,j) = M(NC + i - j).

    For block Toeplitz, block T(i,j) = M(NC + i - j) where i,j are 1-based.
    In 0-based indexing: T[i,j] = M(nc - 1 + i - j) = M(nc - 1 - (j - i))

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    nh1, nh2, nr, nc = 2, 3, 4, 3

    num_params = nr + nc - 1
    h = np.random.randn(nh1, nh2 * num_params).astype(float, order='F')

    t, info = tf01pd(h, nr, nc)

    assert info == 0

    for i in range(nr):
        for j in range(nc):
            k = nc - 1 + i - j
            m_k = h[:, k * nh2:(k + 1) * nh2]
            t_block = t[i * nh1:(i + 1) * nh1, j * nh2:(j + 1) * nh2]
            np.testing.assert_allclose(t_block, m_k, rtol=1e-14, atol=1e-15)


def test_single_parameter():
    """
    Test with NR=1, NC=1 (single parameter, T = M(1)).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    nh1, nh2, nr, nc = 3, 4, 1, 1

    m1 = np.random.randn(nh1, nh2).astype(float, order='F')

    t, info = tf01pd(m1, nr, nc)

    assert info == 0
    np.testing.assert_allclose(t, m1, rtol=1e-14, atol=1e-15)


def test_scalar_parameters():
    """
    Test with NH1=1, NH2=1 (scalar Toeplitz matrix).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    nh1, nh2, nr, nc = 1, 1, 4, 3

    num_params = nr + nc - 1
    h = np.random.randn(nh1, num_params).astype(float, order='F')

    t, info = tf01pd(h, nr, nc)

    assert info == 0

    for i in range(nr):
        for j in range(nc):
            k = nc - 1 + i - j
            np.testing.assert_allclose(t[i, j], h[0, k], rtol=1e-14, atol=1e-15)


def test_larger_blocks():
    """
    Test with larger block sizes.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    nh1, nh2, nr, nc = 4, 3, 3, 4

    num_params = nr + nc - 1
    h = np.random.randn(nh1, nh2 * num_params).astype(float, order='F')

    t, info = tf01pd(h, nr, nc)

    assert info == 0

    assert t.shape == (nh1 * nr, nh2 * nc)

    for i in range(nr):
        for j in range(nc):
            k = nc - 1 + i - j
            m_k = h[:, k * nh2:(k + 1) * nh2]
            t_block = t[i * nh1:(i + 1) * nh1, j * nh2:(j + 1) * nh2]
            np.testing.assert_allclose(t_block, m_k, rtol=1e-14, atol=1e-15)


def test_zero_nr():
    """Test with NR=0 (empty output - 0 rows)."""
    nh1, nh2 = 2, 2
    nr, nc = 0, 3

    num_params = nr + nc - 1
    h = np.ones((nh1, nh2 * num_params), order='F', dtype=float)

    t, info = tf01pd(h, nr, nc)

    assert info == 0
    assert t.shape == (0, nh2 * nc)


def test_zero_nc():
    """Test with NC=0 (empty output - 0 columns)."""
    nh1, nh2 = 2, 2
    nr, nc = 3, 0

    num_params = nr + nc - 1
    h = np.ones((nh1, nh2 * num_params), order='F', dtype=float)

    t, info = tf01pd(h, nr, nc)

    assert info == 0
    assert t.shape == (nh1 * nr, 0)


def test_last_column_preservation():
    """
    Validate that last block column of T equals M(1),...,M(NR).

    In Toeplitz, the last column contains M(1) at top, M(NR) at bottom.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    nh1, nh2, nr, nc = 2, 3, 5, 4

    num_params = nr + nc - 1
    h = np.random.randn(nh1, nh2 * num_params).astype(float, order='F')

    t, info = tf01pd(h, nr, nc)

    assert info == 0

    last_col_block = t[:, (nc - 1) * nh2:]
    expected_last_col = np.vstack([h[:, i * nh2:(i + 1) * nh2] for i in range(nr)])
    np.testing.assert_allclose(last_col_block, expected_last_col, rtol=1e-14, atol=1e-15)


def test_first_row_preservation():
    """
    Validate that first block row of T equals M(NC), M(NC-1), ..., M(1).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    nh1, nh2, nr, nc = 3, 2, 4, 5

    num_params = nr + nc - 1
    h = np.random.randn(nh1, nh2 * num_params).astype(float, order='F')

    t, info = tf01pd(h, nr, nc)

    assert info == 0

    first_row_block = t[:nh1, :]
    expected_first_row = np.hstack([h[:, (nc - 1 - j) * nh2:(nc - j) * nh2] for j in range(nc)])
    np.testing.assert_allclose(first_row_block, expected_first_row, rtol=1e-14, atol=1e-15)


def test_invalid_nr_negative():
    """Test error handling for negative NR."""
    h = np.ones((2, 6), order='F', dtype=float)

    with pytest.raises(ValueError):
        tf01pd(h, -1, 3)


def test_invalid_nc_negative():
    """Test error handling for negative NC."""
    h = np.ones((2, 6), order='F', dtype=float)

    with pytest.raises(ValueError):
        tf01pd(h, 3, -1)


def test_constant_diagonal_property():
    """
    Validate Toeplitz constant-diagonal property: elements on same diagonal are equal.

    For block Toeplitz, all blocks on the same block-diagonal are equal.

    Random seed: 777 (for reproducibility)
    """
    np.random.seed(777)
    nh1, nh2, nr, nc = 2, 2, 4, 5

    num_params = nr + nc - 1
    h = np.random.randn(nh1, nh2 * num_params).astype(float, order='F')

    t, info = tf01pd(h, nr, nc)

    assert info == 0

    for diag in range(-(nr - 1), nc):
        blocks_on_diag = []
        for i in range(nr):
            j = i - diag
            if 0 <= j < nc:
                t_block = t[i * nh1:(i + 1) * nh1, j * nh2:(j + 1) * nh2]
                blocks_on_diag.append(t_block.copy())

        for block in blocks_on_diag[1:]:
            np.testing.assert_allclose(block, blocks_on_diag[0], rtol=1e-14, atol=1e-15)
