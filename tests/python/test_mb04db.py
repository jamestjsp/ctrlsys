"""
Tests for MB04DB - Apply inverse balancing transformation.

MB04DB applies from the left the inverse of a balancing transformation
computed by MB04DP to the matrix [V1; sgn*V2].

Test data sources:
- Mathematical properties of inverse balancing transformations
- Synthetic test data (no HTML doc example available)
"""

import numpy as np
import pytest

from ctrlsys import mb04db


def test_mb04db_no_operation():
    """
    Test with JOB='N' - do nothing, return immediately.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4
    m = 3
    ilo = 1

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')
    lscale = np.ones(n, dtype=float)
    rscale = np.ones(n, dtype=float)

    v1_orig = v1.copy()
    v2_orig = v2.copy()

    v1_out, v2_out, info = mb04db('N', 'P', ilo, lscale, rscale, v1, v2)

    assert info == 0
    np.testing.assert_allclose(v1_out, v1_orig, rtol=1e-14)
    np.testing.assert_allclose(v2_out, v2_orig, rtol=1e-14)


def test_mb04db_scale_only():
    """
    Test with JOB='S' - scaling only.

    Inverse scaling divides each row by the scale factor.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    m = 3
    ilo = 1

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')

    lscale = np.array([2.0, 4.0, 0.5, 1.0], dtype=float)
    rscale = np.array([1.0, 2.0, 0.25, 8.0], dtype=float)

    v1_orig = v1.copy()
    v2_orig = v2.copy()

    v1_out, v2_out, info = mb04db('S', 'P', ilo, lscale, rscale, v1, v2)

    assert info == 0

    for i in range(n):
        np.testing.assert_allclose(v1_out[i, :], v1_orig[i, :] / lscale[i], rtol=1e-14)
        np.testing.assert_allclose(v2_out[i, :], v2_orig[i, :] / rscale[i], rtol=1e-14)


def test_mb04db_scale_with_ilo():
    """
    Test with JOB='S' and ILO > 1 - scaling starts from ILO.

    Rows 0..ILO-2 are NOT scaled (they were isolated by permutation).
    Only rows ILO-1..N-1 are scaled.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4
    m = 3
    ilo = 3

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')

    lscale = np.array([999.0, 999.0, 2.0, 4.0], dtype=float)
    rscale = np.array([999.0, 999.0, 0.5, 0.25], dtype=float)

    v1_orig = v1.copy()
    v2_orig = v2.copy()

    v1_out, v2_out, info = mb04db('S', 'P', ilo, lscale, rscale, v1, v2)

    assert info == 0

    np.testing.assert_allclose(v1_out[:2, :], v1_orig[:2, :], rtol=1e-14)
    np.testing.assert_allclose(v2_out[:2, :], v2_orig[:2, :], rtol=1e-14)

    for i in range(ilo - 1, n):
        np.testing.assert_allclose(v1_out[i, :], v1_orig[i, :] / lscale[i], rtol=1e-14)
        np.testing.assert_allclose(v2_out[i, :], v2_orig[i, :] / rscale[i], rtol=1e-14)


def test_mb04db_permute_simple():
    """
    Test with JOB='P' - permutation only with simple swap.

    The LSCALE array contains permutation indices (1-based Fortran convention).
    Random seed: 789 (for reproducibility)
    """
    n = 4
    m = 2
    ilo = 3

    v1 = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ], order='F', dtype=float)

    v2 = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
        [70.0, 80.0]
    ], order='F', dtype=float)

    lscale = np.array([2.0, 1.0, 999.0, 999.0], dtype=float)
    rscale = np.ones(n, dtype=float)

    v1_out, v2_out, info = mb04db('P', 'P', ilo, lscale, rscale, v1, v2)

    assert info == 0


def test_mb04db_both_permute_and_scale():
    """
    Test with JOB='B' - both permutation and scaling.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n = 4
    m = 3
    ilo = 2

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')

    lscale = np.array([3.0, 2.0, 4.0, 0.5], dtype=float)
    rscale = np.array([1.0, 0.5, 2.0, 0.25], dtype=float)

    v1_out, v2_out, info = mb04db('B', 'P', ilo, lscale, rscale, v1, v2)

    assert info == 0


def test_mb04db_sgn_negative():
    """
    Test with SGN='N' - negative sign for V2.

    The negative sign affects the symplectic swap operation.
    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 4
    m = 3
    ilo = 1

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')

    lscale = np.array([2.0, 4.0, 0.5, 1.0], dtype=float)
    rscale = np.array([1.0, 2.0, 0.25, 8.0], dtype=float)

    v1_out, v2_out, info = mb04db('S', 'N', ilo, lscale, rscale, v1, v2)

    assert info == 0


def test_mb04db_zero_columns():
    """
    Test with m=0 - zero columns.
    """
    n = 4
    m = 0
    ilo = 1

    v1 = np.zeros((n, 0), order='F', dtype=float)
    v2 = np.zeros((n, 0), order='F', dtype=float)
    lscale = np.ones(n, dtype=float)
    rscale = np.ones(n, dtype=float)

    v1_out, v2_out, info = mb04db('B', 'P', ilo, lscale, rscale, v1, v2)

    assert info == 0


def test_mb04db_zero_rows():
    """
    Test with n=0 - zero rows.
    """
    n = 0
    m = 3
    ilo = 1

    v1 = np.zeros((0, m), order='F', dtype=float)
    v2 = np.zeros((0, m), order='F', dtype=float)
    lscale = np.zeros(0, dtype=float)
    rscale = np.zeros(0, dtype=float)

    v1_out, v2_out, info = mb04db('B', 'P', ilo, lscale, rscale, v1, v2)

    assert info == 0


def test_mb04db_invalid_job():
    """
    Test with invalid JOB parameter.
    """
    n = 4
    m = 3
    ilo = 1

    v1 = np.ones((n, m), order='F', dtype=float)
    v2 = np.ones((n, m), order='F', dtype=float)
    lscale = np.ones(n, dtype=float)
    rscale = np.ones(n, dtype=float)

    v1_out, v2_out, info = mb04db('X', 'P', ilo, lscale, rscale, v1, v2)

    assert info == -1


def test_mb04db_invalid_sgn():
    """
    Test with invalid SGN parameter.
    """
    n = 4
    m = 3
    ilo = 1

    v1 = np.ones((n, m), order='F', dtype=float)
    v2 = np.ones((n, m), order='F', dtype=float)
    lscale = np.ones(n, dtype=float)
    rscale = np.ones(n, dtype=float)

    v1_out, v2_out, info = mb04db('B', 'X', ilo, lscale, rscale, v1, v2)

    assert info == -2


def test_mb04db_invalid_ilo():
    """
    Test with invalid ILO parameter.
    """
    n = 4
    m = 3

    v1 = np.ones((n, m), order='F', dtype=float)
    v2 = np.ones((n, m), order='F', dtype=float)
    lscale = np.ones(n, dtype=float)
    rscale = np.ones(n, dtype=float)

    v1_out, v2_out, info = mb04db('B', 'P', 0, lscale, rscale, v1, v2)
    assert info == -4

    v1_out, v2_out, info = mb04db('B', 'P', n + 2, lscale, rscale, v1, v2)
    assert info == -4
