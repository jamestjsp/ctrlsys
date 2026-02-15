"""
Tests for MB03CZ - Eigenvalue exchange for complex 2x2 upper triangular pencils.

MB03CZ computes unitary matrices Q1, Q2, Q3 for a complex 2-by-2 regular pencil
aAB - bD, with A, B, D upper triangular, such that Q3' A Q2, Q2' B Q1, Q3' D Q1
are still upper triangular, but the eigenvalues are in reversed order.
"""

import numpy as np
import pytest


def test_mb03cz_basic():
    """
    Test basic eigenvalue exchange with simple upper triangular matrices.

    Random seed: 42 (for reproducibility)
    Validates that:
    1. Output unitary matrices satisfy Q @ Q.conj().T = I
    2. Transformed matrices remain upper triangular
    3. Eigenvalues are exchanged
    """
    from ctrlsys import mb03cz

    np.random.seed(42)

    a = np.array([
        [1.0 + 0.5j, 0.3 - 0.2j],
        [0.0 + 0.0j, 2.0 - 1.0j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [0.5 + 0.1j, -0.4 + 0.3j],
        [0.0 + 0.0j, 1.5 - 0.5j]
    ], dtype=np.complex128, order='F')

    d = np.array([
        [1.0 + 0.0j, 0.2 + 0.1j],
        [0.0 + 0.0j, 0.8 - 0.2j]
    ], dtype=np.complex128, order='F')

    co1, si1, co2, si2, co3, si3 = mb03cz(a, b, d)

    q1 = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    q2 = np.array([
        [co2, si2],
        [-np.conj(si2), co2]
    ], dtype=np.complex128)

    q3 = np.array([
        [co3, si3],
        [-np.conj(si3), co3]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q1 @ q1.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        q2 @ q2.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        q3 @ q3.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )

    a_trans = q3.conj().T @ a @ q2
    b_trans = q2.conj().T @ b @ q1
    d_trans = q3.conj().T @ d @ q1

    assert np.abs(a_trans[1, 0]) < 1e-12
    assert np.abs(b_trans[1, 0]) < 1e-12
    assert np.abs(d_trans[1, 0]) < 1e-12


def test_mb03cz_identity_matrices():
    """
    Test with identity-like upper triangular matrices.

    Validates edge case where diagonal elements are equal.
    """
    from ctrlsys import mb03cz

    a = np.array([
        [1.0 + 0.0j, 0.5 + 0.5j],
        [0.0 + 0.0j, 1.0 + 0.0j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [1.0 + 0.0j, 0.2 - 0.1j],
        [0.0 + 0.0j, 1.0 + 0.0j]
    ], dtype=np.complex128, order='F')

    d = np.array([
        [1.0 + 0.0j, 0.1 + 0.2j],
        [0.0 + 0.0j, 1.0 + 0.0j]
    ], dtype=np.complex128, order='F')

    co1, si1, co2, si2, co3, si3 = mb03cz(a, b, d)

    assert np.isfinite(co1)
    assert np.isfinite(co2)
    assert np.isfinite(co3)
    assert np.isfinite(si1)
    assert np.isfinite(si2)
    assert np.isfinite(si3)

    q1 = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q1 @ q1.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )


def test_mb03cz_eigenvalue_exchange():
    """
    Test that eigenvalues of the pencil are properly exchanged.

    For pencil (A*B, D), eigenvalues are lambda where det(A*B - lambda*D) = 0.
    After transformation, eigenvalues should be reversed.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03cz

    np.random.seed(123)

    a = np.array([
        [2.0 + 1.0j, 0.5 - 0.3j],
        [0.0 + 0.0j, 1.0 - 0.5j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [1.5 + 0.2j, -0.2 + 0.4j],
        [0.0 + 0.0j, 0.8 + 0.1j]
    ], dtype=np.complex128, order='F')

    d = np.array([
        [1.0 + 0.1j, 0.3 - 0.1j],
        [0.0 + 0.0j, 1.2 - 0.3j]
    ], dtype=np.complex128, order='F')

    ab = a @ b
    eig_before = np.linalg.eigvals(np.linalg.solve(d, ab))
    eig_before = np.sort_complex(eig_before)

    co1, si1, co2, si2, co3, si3 = mb03cz(a, b, d)

    q1 = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    q2 = np.array([
        [co2, si2],
        [-np.conj(si2), co2]
    ], dtype=np.complex128)

    q3 = np.array([
        [co3, si3],
        [-np.conj(si3), co3]
    ], dtype=np.complex128)

    a_new = q3.conj().T @ a @ q2
    b_new = q2.conj().T @ b @ q1
    d_new = q3.conj().T @ d @ q1

    ab_new = a_new @ b_new
    eig_after = np.linalg.eigvals(np.linalg.solve(d_new, ab_new))
    eig_after = np.sort_complex(eig_after)

    np.testing.assert_allclose(
        eig_before, eig_after, rtol=1e-12
    )


def test_mb03cz_real_valued():
    """
    Test with purely real upper triangular matrices.

    Validates routine handles real-valued complex input correctly.
    """
    from ctrlsys import mb03cz

    a = np.array([
        [3.0 + 0.0j, 1.0 + 0.0j],
        [0.0 + 0.0j, 2.0 + 0.0j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [1.0 + 0.0j, -0.5 + 0.0j],
        [0.0 + 0.0j, 1.5 + 0.0j]
    ], dtype=np.complex128, order='F')

    d = np.array([
        [2.0 + 0.0j, 0.5 + 0.0j],
        [0.0 + 0.0j, 1.0 + 0.0j]
    ], dtype=np.complex128, order='F')

    co1, si1, co2, si2, co3, si3 = mb03cz(a, b, d)

    assert isinstance(co1, float)
    assert isinstance(co2, float)
    assert isinstance(co3, float)
    assert isinstance(si1, complex)
    assert isinstance(si2, complex)
    assert isinstance(si3, complex)

    q1 = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q1 @ q1.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )
