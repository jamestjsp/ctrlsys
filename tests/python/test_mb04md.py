"""
Tests for mb04md - Balance a general real matrix.

MB04MD reduces the 1-norm of a general real matrix A by balancing using
diagonal similarity transformations applied iteratively.
"""

import numpy as np
import pytest
from ctrlsys import mb04md


def test_mb04md_html_example():
    """
    Test MB04MD using HTML documentation example (N=4).

    Input matrix A (row-wise in HTML):
      1.0   0.0   0.0   0.0
    300.0 400.0 500.0 600.0
      1.0   2.0   0.0   0.0
      1.0   1.0   1.0   1.0

    Expected balanced matrix:
      1.0000   0.0000   0.0000   0.0000
     30.0000 400.0000  50.0000  60.0000
      1.0000  20.0000   0.0000   0.0000
      1.0000  10.0000   1.0000   1.0000

    Expected SCALE:
      1.0000  10.0000   1.0000   1.0000
    """
    n = 4

    a = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [300.0, 400.0, 500.0, 600.0],
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=float, order='F')

    a_expected = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [30.0, 400.0, 50.0, 60.0],
        [1.0, 20.0, 0.0, 0.0],
        [1.0, 10.0, 1.0, 1.0],
    ], dtype=float, order='F')

    scale_expected = np.array([1.0, 10.0, 1.0, 1.0], dtype=float)

    a_balanced, scale, maxred_out, info = mb04md(a)

    assert info == 0, f"Expected info=0, got {info}"
    np.testing.assert_allclose(a_balanced, a_expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(scale, scale_expected, rtol=1e-10, atol=1e-10)
    assert maxred_out >= 1.0, f"Expected maxred >= 1 for norm reduction, got {maxred_out}"


def test_mb04md_zero_matrix():
    """
    Test MB04MD with zero matrix (quick return).

    When A is zero, SCALE should be ones and MAXRED unchanged.
    """
    n = 3
    a = np.zeros((n, n), dtype=float, order='F')

    a_balanced, scale, maxred_out, info = mb04md(a)

    assert info == 0, f"Expected info=0, got {info}"
    np.testing.assert_allclose(a_balanced, np.zeros((n, n)), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(scale, np.ones(n), rtol=1e-14, atol=1e-14)


def test_mb04md_identity_matrix():
    """
    Test MB04MD with identity matrix.

    Identity matrix is already balanced, SCALE should be ones.
    """
    n = 4
    a = np.eye(n, dtype=float, order='F')

    a_balanced, scale, maxred_out, info = mb04md(a)

    assert info == 0, f"Expected info=0, got {info}"
    np.testing.assert_allclose(a_balanced, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(scale, np.ones(n), rtol=1e-14, atol=1e-14)


def test_mb04md_diagonal_matrix():
    """
    Test MB04MD with diagonal matrix.

    Diagonal matrix is already balanced, SCALE should be ones.
    """
    n = 3
    a = np.diag([1.0, 2.0, 3.0]).astype(float, order='F')

    a_balanced, scale, maxred_out, info = mb04md(a)

    assert info == 0, f"Expected info=0, got {info}"
    np.testing.assert_allclose(a_balanced, a, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(scale, np.ones(n), rtol=1e-14, atol=1e-14)


def test_mb04md_upper_triangular():
    """
    Test MB04MD with upper triangular matrix.

    Upper triangular matrices often benefit from balancing.
    Random seed: 42 (for reproducibility)

    Balancing: A_balanced = D^{-1} * A * D
    So: A_orig = D * A_balanced * D^{-1}
    """
    np.random.seed(42)
    n = 4

    a = np.triu(np.random.randn(n, n) * 100).astype(float, order='F')
    a_orig = a.copy()

    a_balanced, scale, maxred_out, info = mb04md(a)

    assert info == 0, f"Expected info=0, got {info}"

    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)
    a_reconstructed = d @ a_balanced @ d_inv
    np.testing.assert_allclose(a_reconstructed, a_orig, rtol=1e-10, atol=1e-10)


def test_mb04md_similarity_transformation():
    """
    Validate mathematical property: A_balanced = D^{-1} * A * D.

    Balancing is a diagonal similarity transformation that preserves eigenvalues.
    So: A_orig = D * A_balanced * D^{-1}

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 5

    a = np.random.randn(n, n).astype(float, order='F')
    a[0, :] *= 1000
    a[:, 1] /= 100
    a_orig = a.copy()

    a_balanced, scale, maxred_out, info = mb04md(a)

    assert info == 0, f"Expected info=0, got {info}"

    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)
    a_reconstructed = d @ a_balanced @ d_inv
    np.testing.assert_allclose(a_reconstructed, a_orig, rtol=1e-10, atol=1e-10)

    eig_orig = np.linalg.eigvals(a_orig)
    eig_balanced = np.linalg.eigvals(a_balanced)
    np.testing.assert_allclose(sorted(eig_orig.real), sorted(eig_balanced.real), rtol=1e-10, atol=1e-10)


def test_mb04md_custom_maxred():
    """
    Test MB04MD with custom MAXRED parameter.

    MAXRED controls maximum allowed reduction when zero rows/columns encountered.
    """
    n = 4

    a = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [300.0, 400.0, 500.0, 600.0],
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=float, order='F')

    a_balanced, scale, maxred_out, info = mb04md(a, maxred=5.0)

    assert info == 0, f"Expected info=0, got {info}"
    assert maxred_out > 0, "Expected positive maxred output"


def test_mb04md_n_zero():
    """
    Test MB04MD with N=0 (empty matrix quick return).
    """
    n = 0
    a = np.array([], dtype=float, order='F').reshape(0, 0)

    a_balanced, scale, maxred_out, info = mb04md(a)

    assert info == 0, f"Expected info=0, got {info}"
    assert a_balanced.shape == (0, 0)
    assert scale.shape == (0,)


def test_mb04md_n_one():
    """
    Test MB04MD with N=1 (scalar case).
    """
    n = 1
    a = np.array([[5.0]], dtype=float, order='F')

    a_balanced, scale, maxred_out, info = mb04md(a)

    assert info == 0, f"Expected info=0, got {info}"
    np.testing.assert_allclose(a_balanced, a, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(scale, [1.0], rtol=1e-14, atol=1e-14)


def test_mb04md_invalid_maxred():
    """
    Test MB04MD error handling for invalid MAXRED.

    MAXRED > 0 but < 1 is invalid (must enable norm reduction).
    """
    n = 3
    a = np.eye(n, dtype=float, order='F')

    with pytest.raises(ValueError, match="illegal value in argument 2"):
        mb04md(a, maxred=0.5)


def test_mb04md_large_range():
    """
    Test MB04MD with elements spanning wide range of magnitudes.

    Balancing: A_balanced = D^{-1} * A * D
    So: A_orig = D * A_balanced * D^{-1}

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4

    a = np.array([
        [1e-6, 1e3, 0, 0],
        [1e-3, 1.0, 1e4, 0],
        [0, 1e-4, 1e2, 1e5],
        [0, 0, 1e-5, 1e-1],
    ], dtype=float, order='F')
    a_orig = a.copy()

    a_balanced, scale, maxred_out, info = mb04md(a)

    assert info == 0, f"Expected info=0, got {info}"

    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)
    a_reconstructed = d @ a_balanced @ d_inv
    np.testing.assert_allclose(a_reconstructed, a_orig, rtol=1e-8, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
