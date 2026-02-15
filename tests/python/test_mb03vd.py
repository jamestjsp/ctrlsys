# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for MB03VD: Periodic Hessenberg form of product of p matrices.

Reduces A = A_1*A_2*...*A_p to upper Hessenberg form H = H_1*H_2*...*H_p
where H_1 is upper Hessenberg and H_2, ..., H_p are upper triangular,
using orthogonal similarity transformations.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ctrlsys import mb03vd


"""Basic functionality tests from HTML documentation example."""

def test_html_example():
    """
    Test MB03VD using example from SLICOT HTML documentation.

    Input: N=4, P=2, ILO=1, IHI=4
    Two 4x4 matrices (same data in this example).

    Expected output from SLICOT documentation:
    H_1 (upper Hessenberg):
        -2.3926   2.7042  -0.9598  -1.2335
         4.1417  -1.7046   1.3001  -1.3120
         0.0000  -1.6247  -0.2534   1.6453
         0.0000   0.0000  -0.0169  -0.4451

    H_2 (upper triangular):
        -2.5495   2.3402   4.7021   0.2329
         0.0000   1.9725  -0.2483  -2.3493
         0.0000   0.0000  -0.6290  -0.5975
         0.0000   0.0000   0.0000  -0.4426
    """
    n = 4
    p = 2
    ilo = 1
    ihi = 4

    a1 = np.array([
        [1.5, -0.7, 3.5, -0.7],
        [1.0,  0.0, 2.0,  3.0],
        [1.5, -0.7, 2.5, -0.3],
        [1.0,  0.0, 2.0,  1.0]
    ], dtype=np.float64, order='F')

    a2 = np.array([
        [1.5, -0.7, 3.5, -0.7],
        [1.0,  0.0, 2.0,  3.0],
        [1.5, -0.7, 2.5, -0.3],
        [1.0,  0.0, 2.0,  1.0]
    ], dtype=np.float64, order='F')

    a = np.zeros((n, n, p), dtype=np.float64, order='F')
    a[:, :, 0] = a1
    a[:, :, 1] = a2

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == 0

    h1_expected = np.array([
        [-2.3926,  2.7042, -0.9598, -1.2335],
        [ 4.1417, -1.7046,  1.3001, -1.3120],
        [ 0.0000, -1.6247, -0.2534,  1.6453],
        [ 0.0000,  0.0000, -0.0169, -0.4451]
    ], dtype=np.float64, order='F')

    h2_expected = np.array([
        [-2.5495,  2.3402,  4.7021,  0.2329],
        [ 0.0000,  1.9725, -0.2483, -2.3493],
        [ 0.0000,  0.0000, -0.6290, -0.5975],
        [ 0.0000,  0.0000,  0.0000, -0.4426]
    ], dtype=np.float64, order='F')

    h1 = np.triu(a_out[:, :, 0], -1)
    h2 = np.triu(a_out[:, :, 1])

    assert_allclose(h1, h1_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(h2, h2_expected, rtol=1e-3, atol=1e-4)

    assert tau.shape == (n - 1, p)


"""Mathematical property validation tests."""

def test_eigenvalue_preservation_p2():
    """
    Validate eigenvalues of product A_1*A_2 are preserved under transformation.

    The eigenvalues of H_1*H_2 should equal eigenvalues of A_1*A_2.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4
    p = 2
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, p).astype(np.float64, order='F')

    prod_before = a[:, :, 0] @ a[:, :, 1]
    eig_before = np.linalg.eigvals(prod_before)

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == 0

    h1 = np.triu(a_out[:, :, 0], -1)
    h2 = np.triu(a_out[:, :, 1])
    prod_after = h1 @ h2
    eig_after = np.linalg.eigvals(prod_after)

    eig_before_sorted = np.sort_complex(eig_before)
    eig_after_sorted = np.sort_complex(eig_after)

    assert_allclose(np.abs(eig_before_sorted), np.abs(eig_after_sorted), rtol=1e-12)
    assert_allclose(np.angle(eig_before_sorted), np.angle(eig_after_sorted), rtol=1e-12, atol=1e-12)

def test_eigenvalue_preservation_p3():
    """
    Validate eigenvalues of product A_1*A_2*A_3 are preserved.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3
    p = 3
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, p).astype(np.float64, order='F')

    prod_before = a[:, :, 0] @ a[:, :, 1] @ a[:, :, 2]
    eig_before = np.linalg.eigvals(prod_before)

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == 0

    h1 = np.triu(a_out[:, :, 0], -1)
    h2 = np.triu(a_out[:, :, 1])
    h3 = np.triu(a_out[:, :, 2])
    prod_after = h1 @ h2 @ h3
    eig_after = np.linalg.eigvals(prod_after)

    eig_before_sorted = np.sort_complex(eig_before)
    eig_after_sorted = np.sort_complex(eig_after)

    assert_allclose(np.abs(eig_before_sorted), np.abs(eig_after_sorted), rtol=1e-12)

def test_hessenberg_structure_h1():
    """
    Validate H_1 extraction has upper Hessenberg structure.

    The raw output contains Householder vectors below. Use np.triu(a, -1)
    to extract the upper Hessenberg part.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 5
    p = 2
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, p).astype(np.float64, order='F')

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == 0

    h1 = np.triu(a_out[:, :, 0], -1)
    for j in range(n - 2):
        for i in range(j + 2, n):
            assert abs(h1[i, j]) < 1e-14, f"H1[{i},{j}] = {h1[i,j]} should be zero"

def test_triangular_structure_h2_to_hp():
    """
    Validate H_2, ..., H_p extraction has upper triangular structure.

    The raw output contains Householder vectors below. Use np.triu(a)
    to extract the upper triangular part.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4
    p = 3
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, p).astype(np.float64, order='F')

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == 0

    for k in range(1, p):
        hk = np.triu(a_out[:, :, k])
        for j in range(n - 1):
            for i in range(j + 1, n):
                assert abs(hk[i, j]) < 1e-14, f"H{k+1}[{i},{j}] = {hk[i,j]} should be zero"


"""Edge case and boundary condition tests."""

def test_single_matrix_p1():
    """
    Test with P=1 (single matrix, should produce standard Hessenberg).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 4
    p = 1
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, p).astype(np.float64, order='F')
    a_copy = a.copy()

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == 0

    h1 = np.triu(a_out[:, :, 0], -1)
    for j in range(n - 2):
        for i in range(j + 2, n):
            assert abs(h1[i, j]) < 1e-14

    eig_before = np.linalg.eigvals(a_copy[:, :, 0])
    eig_after = np.linalg.eigvals(h1)
    assert_allclose(sorted(eig_before.real), sorted(eig_after.real), rtol=1e-12)

def test_n1_trivial():
    """
    Test with N=1 (trivial case, quick return).
    """
    n = 1
    p = 2
    ilo = 1
    ihi = 1

    a = np.array([[[1.5], [2.5]]]).reshape((1, 1, 2), order='F')

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == 0

def test_partial_ilo_ihi():
    """
    Test with ILO=2, IHI=3 (partial reduction).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 5
    p = 2
    ilo = 2
    ihi = 4

    a = np.zeros((n, n, p), dtype=np.float64, order='F')
    for k in range(p):
        a[0, 0, k] = 1.0
        a[n-1, n-1, k] = 2.0
        a[ilo-1:ihi, ilo-1:ihi, k] = np.random.randn(ihi - ilo + 1, ihi - ilo + 1)

    a_copy = a.copy()

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == 0

    assert_allclose(a_out[0, 0, 0], a_copy[0, 0, 0], rtol=1e-14)
    assert_allclose(a_out[n-1, n-1, 0], a_copy[n-1, n-1, 0], rtol=1e-14)

def test_identity_matrices():
    """
    Test with identity matrices (should remain identity in result).
    """
    n = 3
    p = 2
    ilo = 1
    ihi = n

    a = np.zeros((n, n, p), dtype=np.float64, order='F')
    for k in range(p):
        a[:, :, k] = np.eye(n)

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == 0

    for k in range(p):
        hk = np.triu(a_out[:, :, k], -1 if k == 0 else 0)
        assert_allclose(hk, np.eye(n), rtol=1e-14)


"""Error handling tests."""

def test_invalid_n_negative():
    """Test error for N < 0."""
    n = -1
    p = 2
    ilo = 1
    ihi = 1

    a = np.zeros((1, 1, p), dtype=np.float64, order='F')

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == -1

def test_invalid_p_zero():
    """Test error for P < 1."""
    n = 3
    p = 0
    ilo = 1
    ihi = 3

    a = np.zeros((n, n, 1), dtype=np.float64, order='F')

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == -2

def test_invalid_ilo():
    """Test error for ILO out of range."""
    n = 4
    p = 2
    ilo = 0
    ihi = 4

    a = np.zeros((n, n, p), dtype=np.float64, order='F')

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == -3

def test_invalid_ihi():
    """Test error for IHI < ILO."""
    n = 4
    p = 2
    ilo = 3
    ihi = 2

    a = np.zeros((n, n, p), dtype=np.float64, order='F')

    a_out, tau, info = mb03vd(n, p, ilo, ihi, a)
    assert info == -4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
