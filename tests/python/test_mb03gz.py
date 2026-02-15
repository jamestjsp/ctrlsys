"""
Tests for MB03GZ - Eigenvalue exchange for complex 2x2 skew-Hamiltonian/Hamiltonian pencil.

MB03GZ computes a unitary matrix Q and a unitary symplectic matrix U for a
complex regular 2-by-2 skew-Hamiltonian/Hamiltonian pencil aS - bH with
S = J Z' J' Z, where Z and H are upper triangular.

The matrices Q and U are represented by:
    Q = [[CO1, SI1], [-SI1', CO1]]
    U = [[CO2, SI2], [-SI2', CO2]]

Such that U' Z Q and (J Q J')' H Q are both upper triangular, but the
eigenvalues are in reversed order.
"""

import numpy as np
import pytest


def test_mb03gz_basic():
    """
    Test basic eigenvalue exchange with simple upper triangular matrices.

    Random seed: 42 (for reproducibility)
    Validates that:
    1. Output unitary matrices satisfy Q @ Q.conj().T = I
    2. Output unitary symplectic matrices satisfy U @ U.conj().T = I
    """
    from ctrlsys import mb03gz

    np.random.seed(42)

    z11 = 1.0 + 0.5j
    z12 = 0.3 - 0.2j
    z22 = 2.0 - 1.0j
    h11 = 0.5 + 0.1j
    h12 = -0.4 + 0.3j

    co1, si1, co2, si2 = mb03gz(z11, z12, z22, h11, h12)

    q = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    u = np.array([
        [co2, si2],
        [-np.conj(si2), co2]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q @ q.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        u @ u.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )


def test_mb03gz_identity_like():
    """
    Test with identity-like input matrices.

    Validates edge case where diagonal elements are close to identity.
    """
    from ctrlsys import mb03gz

    z11 = 1.0 + 0.0j
    z12 = 0.1 + 0.05j
    z22 = 1.0 + 0.0j
    h11 = 1.0 + 0.0j
    h12 = 0.1 - 0.05j

    co1, si1, co2, si2 = mb03gz(z11, z12, z22, h11, h12)

    assert np.isfinite(co1)
    assert np.isfinite(co2)
    assert np.isfinite(si1)
    assert np.isfinite(si2)

    assert isinstance(co1, float)
    assert isinstance(co2, float)
    assert isinstance(si1, complex)
    assert isinstance(si2, complex)


def test_mb03gz_real_valued():
    """
    Test with purely real input values.

    Validates routine handles real-valued complex input correctly.
    """
    from ctrlsys import mb03gz

    z11 = 2.0 + 0.0j
    z12 = 1.0 + 0.0j
    z22 = 3.0 + 0.0j
    h11 = 1.5 + 0.0j
    h12 = -0.5 + 0.0j

    co1, si1, co2, si2 = mb03gz(z11, z12, z22, h11, h12)

    q = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    u = np.array([
        [co2, si2],
        [-np.conj(si2), co2]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q @ q.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        u @ u.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )


def test_mb03gz_transformation_structure():
    """
    Test that transformations preserve upper triangular structure.

    Random seed: 123 (for reproducibility)
    Validates:
    1. U' Z Q is upper triangular (lower-left element is zero)
    """
    from ctrlsys import mb03gz

    np.random.seed(123)

    z11 = 2.0 + 1.0j
    z12 = 0.5 - 0.3j
    z22 = 1.0 - 0.5j
    h11 = 1.5 + 0.2j
    h12 = -0.2 + 0.4j

    z = np.array([
        [z11, z12],
        [0.0 + 0.0j, z22]
    ], dtype=np.complex128)

    co1, si1, co2, si2 = mb03gz(z11, z12, z22, h11, h12)

    q = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    u = np.array([
        [co2, si2],
        [-np.conj(si2), co2]
    ], dtype=np.complex128)

    z_transformed = u.conj().T @ z @ q

    assert np.abs(z_transformed[1, 0]) < 1e-12


def test_mb03gz_determinant_unity():
    """
    Test that Q and U have determinant of magnitude 1.

    Unitary matrices have |det| = 1.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03gz

    np.random.seed(456)

    z11 = 1.2 + 0.8j
    z12 = 0.4 - 0.1j
    z22 = 0.9 - 0.6j
    h11 = 0.7 + 0.3j
    h12 = -0.3 + 0.2j

    co1, si1, co2, si2 = mb03gz(z11, z12, z22, h11, h12)

    q = np.array([
        [co1, si1],
        [-np.conj(si1), co1]
    ], dtype=np.complex128)

    u = np.array([
        [co2, si2],
        [-np.conj(si2), co2]
    ], dtype=np.complex128)

    det_q = np.linalg.det(q)
    det_u = np.linalg.det(u)

    np.testing.assert_allclose(np.abs(det_q), 1.0, rtol=1e-14)
    np.testing.assert_allclose(np.abs(det_u), 1.0, rtol=1e-14)
