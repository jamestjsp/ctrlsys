# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for MB03VY: Generate orthogonal matrices from periodic Hessenberg reduction.

MB03VY generates the real orthogonal matrices Q_1, Q_2, ..., Q_p which are
defined as the product of ihi-ilo elementary reflectors of order n, as
returned by SLICOT routine MB03VD:

   Q_j = H_j(ilo) H_j(ilo+1) ... H_j(ihi-1)

This is a companion routine to MB03VD, using LAPACK's DORGHR and DORGQR.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ctrlsys import mb03vd, mb03vy


"""Basic functionality tests."""

def test_basic_p2_full_range():
    """
    Test MB03VY with P=2 matrices, full ilo=1 to ihi=n range.

    Uses MB03VD to compute Hessenberg form, then MB03VY to generate Q matrices.
    Validates Q matrices are orthogonal: Q @ Q.T = I.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4
    p = 2
    ilo = 1
    ihi = n

    a_orig = np.random.randn(n, n, p).astype(np.float64, order='F')

    a = a_orig.copy()
    a_out, tau, info_vd = mb03vd(n, p, ilo, ihi, a)
    assert info_vd == 0

    q, info_vy = mb03vy(n, p, ilo, ihi, a_out, tau)
    assert info_vy == 0

    assert q.shape == (n, n, p)

    for j in range(p):
        qj = q[:, :, j]
        qtq = qj.T @ qj
        qqt = qj @ qj.T
        assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14)
        assert_allclose(qqt, np.eye(n), rtol=1e-14, atol=1e-14)

def test_basic_p3():
    """
    Test MB03VY with P=3 matrices.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3
    p = 3
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, p).astype(np.float64, order='F')

    a_out, tau, info_vd = mb03vd(n, p, ilo, ihi, a)
    assert info_vd == 0

    q, info_vy = mb03vy(n, p, ilo, ihi, a_out, tau)
    assert info_vy == 0

    for j in range(p):
        qj = q[:, :, j]
        assert_allclose(qj.T @ qj, np.eye(n), rtol=1e-14, atol=1e-14)


"""Mathematical property validation tests."""

def test_orthogonality_property():
    """
    Validate Q matrices are orthogonal: det(Q) = +-1.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 5
    p = 2
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, p).astype(np.float64, order='F')

    a_out, tau, info_vd = mb03vd(n, p, ilo, ihi, a)
    assert info_vd == 0

    q, info_vy = mb03vy(n, p, ilo, ihi, a_out, tau)
    assert info_vy == 0

    for j in range(p):
        det = np.linalg.det(q[:, :, j])
        assert abs(abs(det) - 1.0) < 1e-14

def test_similarity_transformation():
    """
    Validate Q reconstructs original matrices via similarity transformation.

    For periodic Hessenberg reduction:
        A_j = Q_j @ H_j @ Q_{j+1}^T  (cyclic, j+1 wraps to 1)

    Where H_1 is upper Hessenberg and H_2, ..., H_p are upper triangular.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4
    p = 2
    ilo = 1
    ihi = n

    a_orig = np.random.randn(n, n, p).astype(np.float64, order='F')
    a_orig_copy = a_orig.copy()

    a_out, tau, info_vd = mb03vd(n, p, ilo, ihi, a_orig)
    assert info_vd == 0

    h1 = np.triu(a_out[:, :, 0], -1).copy()
    h2 = np.triu(a_out[:, :, 1]).copy()

    q, info_vy = mb03vy(n, p, ilo, ihi, a_out, tau)
    assert info_vy == 0

    a1_reconstructed = q[:, :, 0] @ h1 @ q[:, :, 1].T
    a2_reconstructed = q[:, :, 1] @ h2 @ q[:, :, 0].T

    assert_allclose(a1_reconstructed, a_orig_copy[:, :, 0], rtol=1e-13, atol=1e-14)
    assert_allclose(a2_reconstructed, a_orig_copy[:, :, 1], rtol=1e-13, atol=1e-14)

def test_q1_structure_dorghr():
    """
    Validate Q_1 matches DORGHR structure: identity outside [ilo:ihi, ilo:ihi].

    Q_1 is generated using DORGHR, which preserves identity structure
    outside the active range.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 6
    p = 2
    ilo = 2
    ihi = 5

    a = np.zeros((n, n, p), dtype=np.float64, order='F')
    for k in range(p):
        a[0, 0, k] = 1.0
        a[n-1, n-1, k] = 2.0
        a[ilo-1:ihi, ilo-1:ihi, k] = np.random.randn(ihi - ilo + 1, ihi - ilo + 1)

    a_out, tau, info_vd = mb03vd(n, p, ilo, ihi, a)
    assert info_vd == 0

    q, info_vy = mb03vy(n, p, ilo, ihi, a_out, tau)
    assert info_vy == 0

    q1 = q[:, :, 0]
    assert_allclose(q1[0, 0], 1.0, rtol=1e-14)
    assert_allclose(q1[n-1, n-1], 1.0, rtol=1e-14)
    assert_allclose(q1[0, 1:], 0.0, atol=1e-14)
    assert_allclose(q1[1:, 0], 0.0, atol=1e-14)
    assert_allclose(q1[n-1, :n-1], 0.0, atol=1e-14)
    assert_allclose(q1[:n-1, n-1], 0.0, atol=1e-14)


"""Edge case and boundary condition tests."""

def test_n1_trivial():
    """
    Test with N=1 (trivial case, Q should be identity).
    """
    n = 1
    p = 2
    ilo = 1
    ihi = 1

    a = np.array([[[1.5], [2.5]]]).reshape((1, 1, 2), order='F')

    a_out, tau, info_vd = mb03vd(n, p, ilo, ihi, a)
    assert info_vd == 0

    q, info_vy = mb03vy(n, p, ilo, ihi, a_out, tau)
    assert info_vy == 0

    assert q.shape == (1, 1, 2)
    assert_allclose(q[0, 0, 0], 1.0, rtol=1e-14)
    assert_allclose(q[0, 0, 1], 1.0, rtol=1e-14)

def test_n0_trivial():
    """
    Test with N=0 (empty case, quick return).
    """
    n = 0
    p = 2
    ilo = 1
    ihi = 0

    a = np.zeros((1, 1, p), dtype=np.float64, order='F')
    tau = np.zeros((1, p), dtype=np.float64, order='F')

    q, info = mb03vy(n, p, ilo, ihi, a, tau)
    assert info == 0

def test_single_matrix_p1():
    """
    Test with P=1 (single matrix).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 4
    p = 1
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, p).astype(np.float64, order='F')

    a_out, tau, info_vd = mb03vd(n, p, ilo, ihi, a)
    assert info_vd == 0

    q, info_vy = mb03vy(n, p, ilo, ihi, a_out, tau)
    assert info_vy == 0

    assert_allclose(q[:, :, 0].T @ q[:, :, 0], np.eye(n), rtol=1e-14, atol=1e-14)

def test_partial_ilo_ihi():
    """
    Test with partial ILO=2, IHI=3 range.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 5
    p = 2
    ilo = 2
    ihi = 4

    a = np.zeros((n, n, p), dtype=np.float64, order='F')
    for k in range(p):
        a[0, 0, k] = 1.0
        a[n-1, n-1, k] = 2.0
        a[ilo-1:ihi, ilo-1:ihi, k] = np.random.randn(ihi - ilo + 1, ihi - ilo + 1)

    a_out, tau, info_vd = mb03vd(n, p, ilo, ihi, a)
    assert info_vd == 0

    q, info_vy = mb03vy(n, p, ilo, ihi, a_out, tau)
    assert info_vy == 0

    for j in range(p):
        assert_allclose(q[:, :, j].T @ q[:, :, j], np.eye(n), rtol=1e-14, atol=1e-14)


"""Error handling tests."""

def test_invalid_n_negative():
    """Test error for N < 0."""
    n = -1
    p = 2
    ilo = 1
    ihi = 1

    a = np.zeros((1, 1, p), dtype=np.float64, order='F')
    tau = np.zeros((1, p), dtype=np.float64, order='F')

    q, info = mb03vy(n, p, ilo, ihi, a, tau)
    assert info == -1

def test_invalid_p_zero():
    """Test error for P < 1."""
    n = 3
    p = 0
    ilo = 1
    ihi = 3

    a = np.zeros((n, n, 1), dtype=np.float64, order='F')
    tau = np.zeros((max(1, n-1), 1), dtype=np.float64, order='F')

    q, info = mb03vy(n, p, ilo, ihi, a, tau)
    assert info == -2

def test_invalid_ilo_low():
    """Test error for ILO < 1."""
    n = 4
    p = 2
    ilo = 0
    ihi = 4

    a = np.zeros((n, n, p), dtype=np.float64, order='F')
    tau = np.zeros((n-1, p), dtype=np.float64, order='F')

    q, info = mb03vy(n, p, ilo, ihi, a, tau)
    assert info == -3

def test_invalid_ihi():
    """Test error for IHI < ILO."""
    n = 4
    p = 2
    ilo = 3
    ihi = 2

    a = np.zeros((n, n, p), dtype=np.float64, order='F')
    tau = np.zeros((n-1, p), dtype=np.float64, order='F')

    q, info = mb03vy(n, p, ilo, ihi, a, tau)
    assert info == -4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
