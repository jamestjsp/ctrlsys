"""
Tests for MB04DI - Apply inverse balancing transformation to [V1; sgn*V2].

MB04DI applies the inverse of a balancing transformation computed by
MB04DD or MB04DS to a 2*N-by-M matrix [V1; sgn*V2] where sgn is +1 or -1.

Test data sources:
- Mathematical properties of inverse transformations
- Integration with MB04DD for round-trip verification
"""

import numpy as np
import pytest

from ctrlsys import mb04di, mb04dd


def test_mb04di_no_transform():
    """
    Test with JOB='N' - do nothing, return immediately.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4
    m = 3
    ilo = 2
    scale = np.array([5.0, 1.0, 2.0, 0.5], order='F', dtype=float)

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')
    v1_orig = v1.copy()
    v2_orig = v2.copy()

    v1_out, v2_out, info = mb04di('N', 'P', ilo, scale, v1, v2)

    assert info == 0
    np.testing.assert_allclose(v1_out, v1_orig, rtol=1e-14)
    np.testing.assert_allclose(v2_out, v2_orig, rtol=1e-14)


def test_mb04di_scale_only():
    """
    Test with JOB='S' - inverse scaling only.

    Inverse scaling: V1(i,:) *= scale(i), V2(i,:) /= scale(i) for i >= ilo.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    m = 2
    ilo = 2
    scale = np.array([5.0, 2.0, 4.0, 0.5], order='F', dtype=float)

    v1 = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0]
    ], order='F', dtype=float)
    v2 = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [4.0, 4.0],
        [8.0, 8.0]
    ], order='F', dtype=float)

    v1_out, v2_out, info = mb04di('S', 'P', ilo, scale, v1, v2)

    assert info == 0

    v1_expected = np.array([
        [1.0, 2.0],
        [4.0, 8.0],
        [12.0, 24.0],
        [2.0, 4.0]
    ], order='F', dtype=float)
    v2_expected = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [16.0, 16.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(v1_out, v1_expected, rtol=1e-14)
    np.testing.assert_allclose(v2_out, v2_expected, rtol=1e-14)


def test_mb04di_permute_only():
    """
    Test with JOB='P' - inverse permutation only.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3
    m = 2
    ilo = 3
    scale = np.array([2.0, 3.0, 1.0], order='F', dtype=float)

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')

    v1_out, v2_out, info = mb04di('P', 'P', ilo, scale, v1, v2)

    assert info == 0


def test_mb04di_both():
    """
    Test with JOB='B' - both inverse permutation and scaling.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4
    m = 3

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.random.randn(n, n + 1).astype(float, order='F')

    a_bal, qg_bal, ilo, scale, info = mb04dd('B', a, qg)
    assert info == 0

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')

    v1_out, v2_out, info = mb04di('B', 'P', ilo, scale, v1, v2)

    assert info == 0
    assert v1_out.shape == (n, m)
    assert v2_out.shape == (n, m)


def test_mb04di_zero_dimension():
    """
    Test with n=0 - empty matrices (quick return).
    """
    n = 0
    m = 0
    ilo = 1
    scale = np.zeros(0, order='F', dtype=float)
    v1 = np.zeros((0, 0), order='F', dtype=float)
    v2 = np.zeros((0, 0), order='F', dtype=float)

    v1_out, v2_out, info = mb04di('B', 'P', ilo, scale, v1, v2)

    assert info == 0


def test_mb04di_single_element():
    """
    Test with n=1, m=1 - minimal matrices.
    """
    n = 1
    m = 1
    ilo = 1
    scale = np.array([2.0], order='F', dtype=float)
    v1 = np.array([[4.0]], order='F', dtype=float)
    v2 = np.array([[8.0]], order='F', dtype=float)

    v1_out, v2_out, info = mb04di('S', 'P', ilo, scale, v1, v2)

    assert info == 0
    np.testing.assert_allclose(v1_out, np.array([[8.0]]), rtol=1e-14)
    np.testing.assert_allclose(v2_out, np.array([[4.0]]), rtol=1e-14)


def test_mb04di_sgn_negative():
    """
    Test with SGN='N' - negative sign for V2.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 3
    m = 2
    ilo = 2
    scale = np.array([2.0, 1.0, 0.5], order='F', dtype=float)

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')

    v1_out, v2_out, info = mb04di('S', 'N', ilo, scale, v1, v2)

    assert info == 0


def test_mb04di_error_invalid_job():
    """
    Test with invalid JOB parameter.
    """
    n = 2
    m = 2
    ilo = 1
    scale = np.array([1.0, 1.0], order='F', dtype=float)
    v1 = np.ones((n, m), order='F', dtype=float)
    v2 = np.ones((n, m), order='F', dtype=float)

    v1_out, v2_out, info = mb04di('X', 'P', ilo, scale, v1, v2)

    assert info == -1


def test_mb04di_error_invalid_sgn():
    """
    Test with invalid SGN parameter.
    """
    n = 2
    m = 2
    ilo = 1
    scale = np.array([1.0, 1.0], order='F', dtype=float)
    v1 = np.ones((n, m), order='F', dtype=float)
    v2 = np.ones((n, m), order='F', dtype=float)

    v1_out, v2_out, info = mb04di('B', 'X', ilo, scale, v1, v2)

    assert info == -2


def test_mb04di_error_invalid_ilo():
    """
    Test with invalid ILO parameter.
    """
    n = 3
    m = 2
    scale = np.array([1.0, 1.0, 1.0], order='F', dtype=float)
    v1 = np.ones((n, m), order='F', dtype=float)
    v2 = np.ones((n, m), order='F', dtype=float)

    v1_out, v2_out, info = mb04di('B', 'P', 0, scale, v1, v2)
    assert info == -4

    v1_out, v2_out, info = mb04di('B', 'P', n + 2, scale, v1, v2)
    assert info == -4


def test_mb04di_scaling_mathematical_property():
    """
    Test mathematical property: inverse scaling undoes scaling.

    For scale factor s, forward scaling does:
      V1(i,:) /= s, V2(i,:) *= s
    Inverse scaling does:
      V1(i,:) *= s, V2(i,:) /= s

    Property: V1_inv * V2_inv = V1 * V2 (element-wise product preserved)

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 4
    m = 3
    ilo = 2
    scale = np.array([1.0, 2.0, 0.5, 4.0], order='F', dtype=float)

    v1 = np.random.randn(n, m).astype(float, order='F')
    v2 = np.random.randn(n, m).astype(float, order='F')

    product_before = v1 * v2

    v1_out, v2_out, info = mb04di('S', 'P', ilo, scale, v1.copy(), v2.copy())

    assert info == 0

    product_after = v1_out * v2_out

    np.testing.assert_allclose(product_after, product_before, rtol=1e-14)
