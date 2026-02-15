"""
Tests for MB01SD: Scale rows or columns of matrix by diagonal matrix.

Operations:
  JOBS='R': A := diag(R) * A  (row scaling)
  JOBS='C': A := A * diag(C)  (column scaling)
  JOBS='B': A := diag(R) * A * diag(C)  (both)
"""

import numpy as np
import pytest
from ctrlsys import mb01sd


"""Test row scaling: A := diag(R) * A"""

def test_row_scaling_basic():
    """
    Row scaling multiplies each row i by R[i].

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n = 3, 4
    a = np.random.randn(m, n).astype(float, order='F')
    r = np.array([2.0, 0.5, 3.0], dtype=float)
    c = np.empty(0, dtype=float)

    a_orig = a.copy()
    a_out = mb01sd('R', a, r, c)

    expected = np.diag(r) @ a_orig
    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_row_scaling_identity():
    """Row scaling with R = ones should leave A unchanged."""
    np.random.seed(123)
    m, n = 4, 3
    a = np.random.randn(m, n).astype(float, order='F')
    r = np.ones(m, dtype=float)
    c = np.empty(0, dtype=float)

    a_orig = a.copy()
    a_out = mb01sd('R', a, r, c)

    np.testing.assert_allclose(a_out, a_orig, rtol=1e-14)


"""Test column scaling: A := A * diag(C)"""

def test_column_scaling_basic():
    """
    Column scaling multiplies each column j by C[j].

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, n = 3, 4
    a = np.random.randn(m, n).astype(float, order='F')
    r = np.empty(0, dtype=float)
    c = np.array([2.0, 0.5, 3.0, 0.25], dtype=float)

    a_orig = a.copy()
    a_out = mb01sd('C', a, r, c)

    expected = a_orig @ np.diag(c)
    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_column_scaling_identity():
    """Column scaling with C = ones should leave A unchanged."""
    np.random.seed(789)
    m, n = 4, 3
    a = np.random.randn(m, n).astype(float, order='F')
    r = np.empty(0, dtype=float)
    c = np.ones(n, dtype=float)

    a_orig = a.copy()
    a_out = mb01sd('C', a, r, c)

    np.testing.assert_allclose(a_out, a_orig, rtol=1e-14)


"""Test both row and column scaling: A := diag(R) * A * diag(C)"""

def test_both_scaling_basic():
    """
    Both scaling applies row then column scaling.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m, n = 3, 4
    a = np.random.randn(m, n).astype(float, order='F')
    r = np.array([2.0, 0.5, 3.0], dtype=float)
    c = np.array([1.0, 2.0, 0.5, 4.0], dtype=float)

    a_orig = a.copy()
    a_out = mb01sd('B', a, r, c)

    expected = np.diag(r) @ a_orig @ np.diag(c)
    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_both_scaling_square():
    """
    Both scaling on square matrix.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 5
    a = np.random.randn(n, n).astype(float, order='F')
    r = np.array([2.0, 1.5, 0.5, 3.0, 0.25], dtype=float)
    c = np.array([1.0, 0.5, 2.0, 1.5, 3.0], dtype=float)

    a_orig = a.copy()
    a_out = mb01sd('B', a, r, c)

    expected = np.diag(r) @ a_orig @ np.diag(c)
    np.testing.assert_allclose(a_out, expected, rtol=1e-14)


"""Test edge cases and mathematical properties."""

def test_empty_matrix():
    """Empty matrix (M=0 or N=0) should return immediately."""
    a = np.empty((0, 3), dtype=float, order='F')
    r = np.empty(0, dtype=float)
    c = np.array([1.0, 2.0, 3.0], dtype=float)

    a_out = mb01sd('B', a, r, c)
    assert a_out.shape == (0, 3)

def test_single_element():
    """1x1 matrix scaling."""
    a = np.array([[5.0]], dtype=float, order='F')
    r = np.array([2.0], dtype=float)
    c = np.array([3.0], dtype=float)

    a_out = mb01sd('B', a, r, c)
    expected = np.array([[30.0]], dtype=float, order='F')
    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_scaling_commutativity():
    """
    Property: Row then column scaling equals column then row scaling.
    diag(R) * A * diag(C) gives same result regardless of order.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    m, n = 4, 5
    a = np.random.randn(m, n).astype(float, order='F')
    r = np.random.rand(m).astype(float) + 0.1
    c = np.random.rand(n).astype(float) + 0.1

    result_both = mb01sd('B', a.copy(), r, c)

    a_row_first = mb01sd('R', a.copy(), r, np.empty(0, dtype=float))
    result_sequential = mb01sd('C', a_row_first, np.empty(0, dtype=float), c)

    np.testing.assert_allclose(result_both, result_sequential, rtol=1e-14)

def test_involution_property():
    """
    Property: Scaling by R then by 1/R recovers original matrix.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    m, n = 3, 4
    a = np.random.randn(m, n).astype(float, order='F')
    r = np.array([2.0, 0.5, 4.0], dtype=float)

    a_orig = a.copy()
    a_scaled = mb01sd('R', a, r, np.empty(0, dtype=float))
    a_recovered = mb01sd('R', a_scaled, 1.0 / r, np.empty(0, dtype=float))

    np.testing.assert_allclose(a_recovered, a_orig, rtol=1e-14)

def test_lowercase_job():
    """Job parameter should be case-insensitive."""
    np.random.seed(555)
    m, n = 3, 4
    a = np.random.randn(m, n).astype(float, order='F')
    r = np.array([2.0, 0.5, 3.0], dtype=float)

    a1 = a.copy()
    a2 = a.copy()

    result_upper = mb01sd('R', a1, r, np.empty(0, dtype=float))
    result_lower = mb01sd('r', a2, r, np.empty(0, dtype=float))

    np.testing.assert_allclose(result_lower, result_upper, rtol=1e-14)


"""Test numerical accuracy with specific values."""

def test_exact_values_row_scaling():
    """Verify exact numerical values for row scaling."""
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=float, order='F')
    r = np.array([2.0, 3.0], dtype=float)

    a_out = mb01sd('R', a, r, np.empty(0, dtype=float))

    expected = np.array([
        [2.0, 4.0, 6.0],
        [12.0, 15.0, 18.0],
    ], dtype=float, order='F')
    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_exact_values_column_scaling():
    """Verify exact numerical values for column scaling."""
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=float, order='F')
    c = np.array([2.0, 3.0, 4.0], dtype=float)

    a_out = mb01sd('C', a, np.empty(0, dtype=float), c)

    expected = np.array([
        [2.0, 6.0, 12.0],
        [8.0, 15.0, 24.0],
    ], dtype=float, order='F')
    np.testing.assert_allclose(a_out, expected, rtol=1e-14)

def test_exact_values_both_scaling():
    """Verify exact numerical values for both row and column scaling."""
    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ], dtype=float, order='F')
    r = np.array([2.0, 3.0], dtype=float)
    c = np.array([4.0, 5.0], dtype=float)

    a_out = mb01sd('B', a, r, c)

    expected = np.array([
        [8.0, 20.0],
        [36.0, 60.0],
    ], dtype=float, order='F')
    np.testing.assert_allclose(a_out, expected, rtol=1e-14)
