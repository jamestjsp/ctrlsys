"""
Tests for MB05OY - restore matrix after balancing transformations.

MB05OY computes: A <- P * D * A * D^{-1} * P'
where P is permutation, D is diagonal scaling from DGEBAL.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb05oy_scaling_only():
    """
    Test scaling-only backward transformation (JOB='S').

    Scale: D = diag(1, 2, 4, 8)
    Balanced A_bal = D^{-1} * A_orig * D
    MB05OY should recover A_orig = D * A_bal * D^{-1}
    """
    from ctrlsys import mb05oy

    n = 4
    low = 1  # 1-based
    igh = 4  # 1-based

    scale = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)

    a_orig = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], order='F', dtype=float)

    D = np.diag(scale)
    D_inv = np.diag(1.0 / scale)
    a_balanced = D_inv @ a_orig @ D

    a_test = a_balanced.copy(order='F')

    a_out, info = mb05oy('S', n, low, igh, a_test, scale)

    assert info == 0
    assert_allclose(a_out, a_orig, rtol=1e-14)


def test_mb05oy_scaling_subset():
    """
    Test scaling with low=2, igh=3 (middle rows/cols scaled).

    Only rows/cols 2-3 (1-based) are scaled. Others have scale[i] = permutation index.
    For JOB='S', only scaling is applied (no permutation).
    """
    from ctrlsys import mb05oy

    n = 4
    low = 2  # 1-based
    igh = 3  # 1-based

    scale = np.array([1.0, 2.0, 4.0, 4.0], dtype=float)

    a_orig = np.eye(4, order='F', dtype=float)
    a_orig[1, 2] = 8.0
    a_orig[2, 1] = 0.5

    D = np.eye(4, dtype=float)
    D[1, 1] = scale[1]
    D[2, 2] = scale[2]
    D_inv = np.eye(4, dtype=float)
    D_inv[1, 1] = 1.0 / scale[1]
    D_inv[2, 2] = 1.0 / scale[2]

    a_balanced = D_inv @ a_orig @ D

    a_test = a_balanced.copy(order='F')

    a_out, info = mb05oy('S', n, low, igh, a_test, scale)

    assert info == 0
    assert_allclose(a_out, a_orig, rtol=1e-14)


def test_mb05oy_permutation_only():
    """
    Test permutation-only backward transformation (JOB='P').

    Test with LOW=1, IGH=N-1, so only row N needs permutation.
    SCALE(N) = K means row N was swapped with row K by DGEBAL.
    """
    from ctrlsys import mb05oy

    n = 4
    low = 1
    igh = 3  # Row 4 is "isolated"

    scale = np.array([1.0, 1.0, 1.0, 2.0], dtype=float)

    a_orig = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], order='F', dtype=float)

    P = np.eye(4, dtype=float)
    P[1, 1] = 0
    P[1, 3] = 1
    P[3, 1] = 1
    P[3, 3] = 0

    a_balanced = P.T @ a_orig @ P

    a_test = a_balanced.copy(order='F')

    a_out, info = mb05oy('P', n, low, igh, a_test, scale)

    assert info == 0
    assert_allclose(a_out, a_orig, rtol=1e-14)


def test_mb05oy_job_n():
    """
    Test JOB='N' (no operation).

    Matrix should be unchanged.
    """
    from ctrlsys import mb05oy

    n = 3
    a_orig = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], order='F', dtype=float)

    a_test = a_orig.copy(order='F')
    scale = np.array([1.0, 2.0, 3.0], dtype=float)

    a_out, info = mb05oy('N', n, 1, n, a_test, scale)

    assert info == 0
    assert_allclose(a_out, a_orig, rtol=1e-14)


def test_mb05oy_empty_matrix():
    """
    Test N=0 edge case.
    """
    from ctrlsys import mb05oy

    a = np.array([], dtype=float).reshape(0, 0, order='F')
    scale = np.array([], dtype=float)

    a_out, info = mb05oy('B', 0, 1, 0, a, scale)

    assert info == 0


def test_mb05oy_involution_property():
    """
    Test mathematical property: applying balancing twice returns identity.

    If A_bal = D^{-1} * A * D, and we apply MB05OY (which does D * A_bal * D^{-1}),
    we should get back A.
    """
    from ctrlsys import mb05oy

    np.random.seed(42)
    n = 5
    low = 1
    igh = n

    scale = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=float)

    a_orig = np.random.randn(n, n).astype(float, order='F')

    D = np.diag(scale)
    D_inv = np.diag(1.0 / scale)
    a_balanced = D_inv @ a_orig @ D

    a_test = a_balanced.copy(order='F')
    a_out, info = mb05oy('S', n, low, igh, a_test, scale)

    assert info == 0
    assert_allclose(a_out, a_orig, rtol=1e-14)


def test_mb05oy_eigenvalue_preservation():
    """
    Test mathematical property: eigenvalues preserved under transformation.

    Balancing is a similarity transformation, so eigenvalues must be preserved.
    """
    from ctrlsys import mb05oy

    np.random.seed(789)
    n = 4
    low = 1
    igh = n

    scale = np.array([1.0, 4.0, 16.0, 64.0], dtype=float)

    a_orig = np.array([
        [2.0, 500.0, 0.0, 0.0],
        [0.002, 3.0, 500.0, 0.0],
        [0.0, 0.002, 4.0, 500.0],
        [0.0, 0.0, 0.002, 5.0]
    ], order='F', dtype=float)

    eig_orig = np.linalg.eigvals(a_orig)

    D = np.diag(scale)
    D_inv = np.diag(1.0 / scale)
    a_balanced = D_inv @ a_orig @ D

    eig_balanced = np.linalg.eigvals(a_balanced)
    assert_allclose(sorted(eig_orig.real), sorted(eig_balanced.real), rtol=1e-12)

    a_test = a_balanced.copy(order='F')
    a_out, info = mb05oy('S', n, low, igh, a_test, scale)

    assert info == 0

    eig_restored = np.linalg.eigvals(a_out)
    assert_allclose(sorted(eig_orig.real), sorted(eig_restored.real), rtol=1e-14)


def test_mb05oy_invalid_job():
    """
    Test invalid JOB parameter.
    """
    from ctrlsys import mb05oy

    n = 3
    a = np.eye(n, order='F', dtype=float)
    scale = np.ones(n, dtype=float)

    a_out, info = mb05oy('X', n, 1, n, a, scale)

    assert info == -1


def test_mb05oy_invalid_n():
    """
    Test invalid N parameter (negative).
    """
    from ctrlsys import mb05oy

    a = np.eye(3, order='F', dtype=float)
    scale = np.ones(3, dtype=float)

    a_out, info = mb05oy('B', -1, 1, 3, a, scale)

    assert info == -2


def test_mb05oy_invalid_low():
    """
    Test invalid LOW parameter.
    """
    from ctrlsys import mb05oy

    n = 3
    a = np.eye(n, order='F', dtype=float)
    scale = np.ones(n, dtype=float)

    a_out, info = mb05oy('B', n, 0, n, a, scale)
    assert info == -3

    a_out, info = mb05oy('B', n, n + 2, n, a, scale)
    assert info == -3


def test_mb05oy_invalid_igh():
    """
    Test invalid IGH parameter.
    """
    from ctrlsys import mb05oy

    n = 3
    a = np.eye(n, order='F', dtype=float)
    scale = np.ones(n, dtype=float)

    a_out, info = mb05oy('B', n, 2, 1, a, scale)
    assert info == -4

    a_out, info = mb05oy('B', n, 1, n + 1, a, scale)
    assert info == -4


def test_mb05oy_combined_scale_permute():
    """
    Test combined scaling and permutation (JOB='B').

    Simple case: only row 4 is isolated (permuted), rows 1-3 are scaled.
    LOW=1, IGH=3, so:
    - SCALE(1..3) = scaling factors
    - SCALE(4) = permutation index (row 4 swapped with row 2)

    MB05OY computes: A <- P * D * A * D^{-1} * P'
    So if we have balanced = D^{-1} * P' * orig * P * D,
    then MB05OY(balanced) = P * D * D^{-1} * P' * orig * P * D * D^{-1} * P' = orig
    """
    from ctrlsys import mb05oy

    n = 4
    low = 1
    igh = 3

    scale = np.array([1.0, 2.0, 4.0, 2.0], dtype=float)

    a_orig = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], order='F', dtype=float)

    D = np.diag([scale[0], scale[1], scale[2], 1.0])
    D_inv = np.diag([1.0/scale[0], 1.0/scale[1], 1.0/scale[2], 1.0])

    P = np.eye(4, dtype=float)
    P[1, 1] = 0
    P[1, 3] = 1
    P[3, 1] = 1
    P[3, 3] = 0

    a_balanced = D_inv @ P.T @ a_orig @ P @ D

    a_test = a_balanced.copy(order='F')

    a_out, info = mb05oy('B', n, low, igh, a_test, scale)

    assert info == 0
    assert_allclose(a_out, a_orig, rtol=1e-14)
