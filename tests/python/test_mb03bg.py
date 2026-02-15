"""
Tests for MB03BG - Eigenvalues of 2x2 trailing submatrix of matrix product.

MB03BG computes eigenvalues of the 2-by-2 trailing submatrix of a periodic
matrix product A(:,:,1)^S(1) * A(:,:,2)^S(2) * ... * A(:,:,K)^S(K), where
one factor is upper Hessenberg and the rest are upper triangular.
"""

import numpy as np
import pytest
from ctrlsys import mb03bg


def test_mb03bg_basic_2x2_single_factor():
    """
    Test with K=1 single 2x2 upper Hessenberg factor.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    k = 1
    n = 2
    amap = np.array([1], dtype=np.int32)
    s = np.array([1], dtype=np.int32)
    sinv = 1

    a = np.zeros((2, 2, 1), dtype=float, order='F')
    a[:, :, 0] = np.array([[3.0, 1.0],
                           [2.0, 4.0]], order='F')

    wr, wi = mb03bg(k, n, amap, s, sinv, a)

    assert len(wr) == 2
    assert len(wi) == 2

    eigs_expected = np.linalg.eigvals(a[:, :, 0])
    eigs_computed = wr + 1j * wi

    np.testing.assert_allclose(
        sorted(eigs_computed.real),
        sorted(eigs_expected.real),
        rtol=1e-14
    )
    np.testing.assert_allclose(
        sorted(np.abs(eigs_computed.imag)),
        sorted(np.abs(eigs_expected.imag)),
        rtol=1e-14
    )


def test_mb03bg_two_factors():
    """
    Test with K=2 factors - one triangular, one Hessenberg.

    The product is A(:,:,1)^S(1) * A(:,:,2)^S(2).
    AMAP determines which factor is Hessenberg (last one must be).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    k = 2
    n = 3
    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    a = np.zeros((3, 3, 2), dtype=float, order='F')

    a[:, :, 0] = np.array([[2.0, 0.5, 0.3],
                           [0.0, 3.0, 0.4],
                           [0.0, 0.0, 1.5]], order='F')

    a[:, :, 1] = np.array([[1.0, 0.2, 0.1],
                           [0.5, 2.0, 0.3],
                           [0.0, 0.4, 1.0]], order='F')

    wr, wi = mb03bg(k, n, amap, s, sinv, a)

    product = a[:, :, 0] @ a[:, :, 1]
    trailing_2x2 = product[n - 2:n, n - 2:n]
    eigs_expected = np.linalg.eigvals(trailing_2x2)

    eigs_computed = wr + 1j * wi

    np.testing.assert_allclose(
        sorted(eigs_computed.real),
        sorted(eigs_expected.real),
        rtol=1e-13
    )
    np.testing.assert_allclose(
        sorted(np.abs(eigs_computed.imag)),
        sorted(np.abs(eigs_expected.imag)),
        atol=1e-14
    )


def test_mb03bg_all_factors_same_sign():
    """
    Test with all factors having the same signature as SINV.

    Tests eigenvalue computation for product of upper triangular and Hessenberg.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    k = 2
    n = 3
    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    a = np.zeros((3, 3, 2), dtype=float, order='F')

    a[:, :, 0] = np.array([[2.0, 0.5, 0.3],
                           [0.0, 3.0, 0.4],
                           [0.0, 0.0, 1.5]], order='F')

    a[:, :, 1] = np.array([[1.0, 0.2, 0.1],
                           [0.5, 2.0, 0.3],
                           [0.0, 0.4, 1.0]], order='F')

    wr, wi = mb03bg(k, n, amap, s, sinv, a)

    product = a[:, :, 0] @ a[:, :, 1]
    trailing_2x2 = product[n - 2:n, n - 2:n]
    eigs_expected = np.linalg.eigvals(trailing_2x2)

    eigs_computed = wr + 1j * wi

    np.testing.assert_allclose(
        sorted(eigs_computed.real),
        sorted(eigs_expected.real),
        rtol=1e-13
    )
    np.testing.assert_allclose(
        sorted(np.abs(eigs_computed.imag)),
        sorted(np.abs(eigs_expected.imag)),
        atol=1e-13
    )


def test_mb03bg_complex_eigenvalues():
    """
    Test case that produces complex conjugate eigenvalues.

    Random seed: 789 (for reproducibility)
    """
    k = 1
    n = 2
    amap = np.array([1], dtype=np.int32)
    s = np.array([1], dtype=np.int32)
    sinv = 1

    a = np.zeros((2, 2, 1), dtype=float, order='F')
    a[:, :, 0] = np.array([[0.0, -1.0],
                           [1.0, 0.0]], order='F')

    wr, wi = mb03bg(k, n, amap, s, sinv, a)

    np.testing.assert_allclose(wr[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(wr[1], 0.0, atol=1e-14)
    assert abs(wi[0]) == pytest.approx(1.0, rel=1e-14)
    assert abs(wi[1]) == pytest.approx(1.0, rel=1e-14)
    assert wi[0] == -wi[1]


def test_mb03bg_three_factors():
    """
    Test with K=3 factors - all same signature.

    Product: A1 * A2 * A3 where all have S=SINV.

    Random seed: 101 (for reproducibility)
    """
    np.random.seed(101)

    k = 3
    n = 4
    amap = np.array([1, 2, 3], dtype=np.int32)
    s = np.array([1, 1, 1], dtype=np.int32)
    sinv = 1

    a = np.zeros((4, 4, 3), dtype=float, order='F')

    a[:, :, 0] = np.array([[2.0, 0.1, 0.2, 0.1],
                           [0.0, 1.5, 0.3, 0.2],
                           [0.0, 0.0, 3.0, 0.1],
                           [0.0, 0.0, 0.0, 2.5]], order='F')

    a[:, :, 1] = np.array([[1.0, 0.2, 0.1, 0.3],
                           [0.0, 2.0, 0.2, 0.1],
                           [0.0, 0.0, 1.5, 0.2],
                           [0.0, 0.0, 0.0, 1.0]], order='F')

    a[:, :, 2] = np.array([[1.5, 0.1, 0.2, 0.1],
                           [0.3, 2.0, 0.1, 0.2],
                           [0.0, 0.2, 1.0, 0.3],
                           [0.0, 0.0, 0.1, 0.5]], order='F')

    wr, wi = mb03bg(k, n, amap, s, sinv, a)

    product = a[:, :, 0] @ a[:, :, 1] @ a[:, :, 2]
    trailing_2x2 = product[n - 2:n, n - 2:n]
    eigs_expected = np.linalg.eigvals(trailing_2x2)

    eigs_computed = wr + 1j * wi

    np.testing.assert_allclose(
        sorted(eigs_computed.real),
        sorted(eigs_expected.real),
        rtol=1e-12
    )
    np.testing.assert_allclose(
        sorted(np.abs(eigs_computed.imag)),
        sorted(np.abs(eigs_expected.imag)),
        atol=1e-12
    )


def test_mb03bg_diagonal_factors():
    """
    Test with diagonal upper triangular factors (trivial case).

    Random seed: 202 (for reproducibility)
    """
    k = 2
    n = 3
    amap = np.array([1, 2], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    sinv = 1

    a = np.zeros((3, 3, 2), dtype=float, order='F')

    a[:, :, 0] = np.diag([2.0, 3.0, 4.0]).astype(float, order='F')

    a[:, :, 1] = np.array([[1.0, 0.0, 0.0],
                           [0.5, 2.0, 0.0],
                           [0.0, 0.3, 1.5]], order='F')

    wr, wi = mb03bg(k, n, amap, s, sinv, a)

    product = a[:, :, 0] @ a[:, :, 1]
    trailing_2x2 = product[n - 2:n, n - 2:n]
    eigs_expected = np.linalg.eigvals(trailing_2x2)

    eigs_computed = wr + 1j * wi

    np.testing.assert_allclose(
        sorted(eigs_computed.real),
        sorted(eigs_expected.real),
        rtol=1e-14
    )


def test_mb03bg_eigenvalue_property():
    """
    Validate eigenvalue properties: trace and determinant.

    For 2x2 matrix, eigenvalues satisfy:
    - lambda1 + lambda2 = trace
    - lambda1 * lambda2 = det

    Random seed: 303 (for reproducibility)
    """
    np.random.seed(303)

    k = 1
    n = 2
    amap = np.array([1], dtype=np.int32)
    s = np.array([1], dtype=np.int32)
    sinv = 1

    a = np.zeros((2, 2, 1), dtype=float, order='F')
    a[:, :, 0] = np.array([[5.0, 2.0],
                           [3.0, 4.0]], order='F')

    wr, wi = mb03bg(k, n, amap, s, sinv, a)

    eig1 = wr[0] + 1j * wi[0]
    eig2 = wr[1] + 1j * wi[1]

    trace_expected = np.trace(a[:, :, 0])
    det_expected = np.linalg.det(a[:, :, 0])

    np.testing.assert_allclose(
        (eig1 + eig2).real, trace_expected, rtol=1e-14
    )
    np.testing.assert_allclose(
        (eig1 * eig2).real, det_expected, rtol=1e-14
    )
