"""Tests for TF01OD - Block Hankel expansion of multivariable parameter sequence."""

import numpy as np
import pytest
from ctrlsys import tf01od


def test_html_doc_example():
    """
    Test case from SLICOT HTML documentation.

    NH1=2, NH2=2, NR=3, NC=3
    H contains M(1),...,M(5) as 2x2 blocks stored horizontally.
    T is 6x6 block Hankel matrix.

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
        [1.0647, -0.4922, -0.3043, -0.0926, -0.1844, 0.4441],
        [-0.4282, -1.2072, 0.6883, 0.7167, -0.8507, -0.0478],
        [-0.3043, -0.0926, -0.1844, 0.4441, 0.7195, -0.3955],
        [0.6883, 0.7167, -0.8507, -0.0478, 0.0500, 0.5674],
        [-0.1844, 0.4441, 0.7195, -0.3955, 1.3387, 0.1073],
        [-0.8507, -0.0478, 0.0500, 0.5674, -0.2801, -0.5315]
    ], order='F', dtype=float)

    t, info = tf01od(h, nr, nc)

    assert info == 0
    np.testing.assert_allclose(t, t_expected, rtol=1e-4, atol=1e-5)


def test_hankel_structure_property():
    """
    Validate Hankel structure: T(i,j) depends only on i+j.

    For block Hankel, block (i,j) = M(i+j-1).
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    nh1, nh2, nr, nc = 2, 3, 4, 3

    num_params = nr + nc - 1
    h = np.random.randn(nh1, nh2 * num_params).astype(float, order='F')

    t, info = tf01od(h, nr, nc)

    assert info == 0

    for i in range(nr):
        for j in range(nc):
            k = i + j
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

    t, info = tf01od(m1, nr, nc)

    assert info == 0
    np.testing.assert_allclose(t, m1, rtol=1e-14, atol=1e-15)


def test_scalar_parameters():
    """
    Test with NH1=1, NH2=1 (scalar Hankel matrix).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    nh1, nh2, nr, nc = 1, 1, 4, 3

    num_params = nr + nc - 1
    h = np.random.randn(nh1, num_params).astype(float, order='F')

    t, info = tf01od(h, nr, nc)

    assert info == 0

    for i in range(nr):
        for j in range(nc):
            k = i + j
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

    t, info = tf01od(h, nr, nc)

    assert info == 0

    assert t.shape == (nh1 * nr, nh2 * nc)

    for i in range(nr):
        for j in range(nc):
            k = i + j
            m_k = h[:, k * nh2:(k + 1) * nh2]
            t_block = t[i * nh1:(i + 1) * nh1, j * nh2:(j + 1) * nh2]
            np.testing.assert_allclose(t_block, m_k, rtol=1e-14, atol=1e-15)


def test_zero_nr():
    """Test with NR=0 (empty output - 0 rows)."""
    nh1, nh2 = 2, 2
    nr, nc = 0, 3

    num_params = nr + nc - 1  # = 2 when NR=0, NC=3
    h = np.ones((nh1, nh2 * num_params), order='F', dtype=float)

    t, info = tf01od(h, nr, nc)

    assert info == 0
    assert t.shape == (0, nh2 * nc)  # 0 rows, 6 columns


def test_zero_nc():
    """Test with NC=0 (empty output - 0 columns)."""
    nh1, nh2 = 2, 2
    nr, nc = 3, 0

    num_params = nr + nc - 1  # = 2 when NR=3, NC=0
    h = np.ones((nh1, nh2 * num_params), order='F', dtype=float)

    t, info = tf01od(h, nr, nc)

    assert info == 0
    assert t.shape == (nh1 * nr, 0)  # 6 rows, 0 columns


def test_first_column_preservation():
    """
    Validate that first block column of T equals M(1),...,M(NR).

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    nh1, nh2, nr, nc = 2, 3, 5, 4

    num_params = nr + nc - 1
    h = np.random.randn(nh1, nh2 * num_params).astype(float, order='F')

    t, info = tf01od(h, nr, nc)

    assert info == 0

    first_col_block = t[:, :nh2]
    expected_first_col = np.vstack([h[:, i * nh2:(i + 1) * nh2] for i in range(nr)])
    np.testing.assert_allclose(first_col_block, expected_first_col, rtol=1e-14, atol=1e-15)


def test_invalid_nr_negative():
    """Test error handling for negative NR."""
    h = np.ones((2, 6), order='F', dtype=float)

    with pytest.raises(ValueError):
        tf01od(h, -1, 3)


def test_invalid_nc_negative():
    """Test error handling for negative NC."""
    h = np.ones((2, 6), order='F', dtype=float)

    with pytest.raises(ValueError):
        tf01od(h, 3, -1)
