"""
Tests for MB03DZ - Eigenvalue exchange for complex 2x2 upper triangular pencils.

MB03DZ computes unitary matrices Q1 and Q2 for a complex 2-by-2 regular pencil
aA - bB, with A, B upper triangular, such that Q2' (aA - bB) Q1 is still upper
triangular but the eigenvalues are in reversed order.

The matrices Q1 and Q2 are represented by:
     (  CO1  SI1  )       (  CO2  SI2  )
Q1 = (            ), Q2 = (            ).
     ( -SI1' CO1  )       ( -SI2' CO2  )

where ' denotes conjugate transpose.
"""

import numpy as np
import pytest


def test_mb03dz_basic():
    """
    Test basic eigenvalue exchange with simple upper triangular matrices.

    Random seed: 42 (for reproducibility)
    Validates that:
    1. Output unitary matrices satisfy Q @ Q.conj().T = I
    2. Transformed pencil remains upper triangular
    3. Eigenvalues are exchanged
    """
    from ctrlsys import mb03dz

    np.random.seed(42)

    a = np.array([
        [1.0 + 0.5j, 0.3 - 0.2j],
        [0.0 + 0.0j, 2.0 - 1.0j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [0.5 + 0.1j, -0.4 + 0.3j],
        [0.0 + 0.0j, 1.5 - 0.5j]
    ], dtype=np.complex128, order='F')

    co1, si1, co2, si2 = mb03dz(a, b)

    q1 = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    q2 = np.array([
        [co2, si2],
        [-np.conj(si2), co2]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q1 @ q1.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        q2 @ q2.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )

    a_trans = q2.conj().T @ a @ q1
    b_trans = q2.conj().T @ b @ q1

    assert np.abs(a_trans[1, 0]) < 1e-12
    assert np.abs(b_trans[1, 0]) < 1e-12


def test_mb03dz_identity_matrices():
    """
    Test with identity-like upper triangular matrices.

    Validates edge case where diagonal elements are equal.
    """
    from ctrlsys import mb03dz

    a = np.array([
        [1.0 + 0.0j, 0.5 + 0.5j],
        [0.0 + 0.0j, 1.0 + 0.0j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [1.0 + 0.0j, 0.2 - 0.1j],
        [0.0 + 0.0j, 1.0 + 0.0j]
    ], dtype=np.complex128, order='F')

    co1, si1, co2, si2 = mb03dz(a, b)

    assert np.isfinite(co1)
    assert np.isfinite(co2)
    assert np.isfinite(si1)
    assert np.isfinite(si2)

    q1 = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q1 @ q1.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )


def test_mb03dz_eigenvalue_exchange():
    """
    Test that eigenvalues of the pencil are properly preserved.

    For pencil (A, B), eigenvalues are lambda where det(A - lambda*B) = 0.
    After transformation, eigenvalues should still be the same (just reordered).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03dz

    np.random.seed(123)

    a = np.array([
        [2.0 + 1.0j, 0.5 - 0.3j],
        [0.0 + 0.0j, 1.0 - 0.5j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [1.5 + 0.2j, -0.2 + 0.4j],
        [0.0 + 0.0j, 0.8 + 0.1j]
    ], dtype=np.complex128, order='F')

    eig_before = np.linalg.eigvals(np.linalg.solve(b, a))
    eig_before = np.sort_complex(eig_before)

    co1, si1, co2, si2 = mb03dz(a, b)

    q1 = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    q2 = np.array([
        [co2, si2],
        [-np.conj(si2), co2]
    ], dtype=np.complex128)

    a_new = q2.conj().T @ a @ q1
    b_new = q2.conj().T @ b @ q1

    eig_after = np.linalg.eigvals(np.linalg.solve(b_new, a_new))
    eig_after = np.sort_complex(eig_after)

    np.testing.assert_allclose(
        eig_before, eig_after, rtol=1e-12
    )


def test_mb03dz_real_valued():
    """
    Test with purely real upper triangular matrices.

    Validates routine handles real-valued complex input correctly.
    """
    from ctrlsys import mb03dz

    a = np.array([
        [3.0 + 0.0j, 1.0 + 0.0j],
        [0.0 + 0.0j, 2.0 + 0.0j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [1.0 + 0.0j, -0.5 + 0.0j],
        [0.0 + 0.0j, 1.5 + 0.0j]
    ], dtype=np.complex128, order='F')

    co1, si1, co2, si2 = mb03dz(a, b)

    assert isinstance(co1, float)
    assert isinstance(co2, float)
    assert isinstance(si1, complex)
    assert isinstance(si2, complex)

    q1 = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q1 @ q1.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )


def test_mb03dz_diagonal_exchange():
    """
    Test that diagonal elements are effectively exchanged.

    For 2x2 upper triangular pencil (A, B), the eigenvalues are:
    lambda_1 = A(1,1)/B(1,1)
    lambda_2 = A(2,2)/B(2,2)

    After transformation, these should be swapped.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03dz

    np.random.seed(456)

    a = np.array([
        [4.0 + 2.0j, 0.8 - 0.4j],
        [0.0 + 0.0j, 1.0 + 0.5j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [2.0 + 0.0j, 0.3 + 0.2j],
        [0.0 + 0.0j, 0.5 - 0.1j]
    ], dtype=np.complex128, order='F')

    lambda1_before = a[0, 0] / b[0, 0]
    lambda2_before = a[1, 1] / b[1, 1]

    co1, si1, co2, si2 = mb03dz(a, b)

    q1 = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    q2 = np.array([
        [co2, si2],
        [-np.conj(si2), co2]
    ], dtype=np.complex128)

    a_new = q2.conj().T @ a @ q1
    b_new = q2.conj().T @ b @ q1

    lambda1_after = a_new[0, 0] / b_new[0, 0]
    lambda2_after = a_new[1, 1] / b_new[1, 1]

    np.testing.assert_allclose(lambda1_before, lambda2_after, rtol=1e-12)
    np.testing.assert_allclose(lambda2_before, lambda1_after, rtol=1e-12)
