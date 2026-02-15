"""
Tests for MB03HZ - Eigenvalue exchange for complex 2x2 skew-Hamiltonian/Hamiltonian pencil.

MB03HZ computes a unitary matrix Q for a complex regular 2-by-2
skew-Hamiltonian/Hamiltonian pencil aS - bH with
    (  S11  S12  )        (  H11  H12  )
S = (            ),   H = (            ),
    (   0   S11' )        (   0  -H11' )

such that J Q' J' (aS - bH) Q is upper triangular but the eigenvalues
are in reversed order. The matrix Q is represented by
    (  CO  SI  )
Q = (          ).
    ( -SI' CO  )

The notation M' denotes the conjugate transpose of the matrix M.
"""

import numpy as np
import pytest


def test_mb03hz_basic():
    """
    Test basic eigenvalue exchange with simple complex input values.

    Random seed: 42 (for reproducibility)
    Validates that:
    1. Output CO is a real scalar
    2. Output SI is a complex scalar
    3. The unitary matrix Q satisfies Q @ Q.conj().T = I
    """
    from ctrlsys import mb03hz

    np.random.seed(42)

    s11 = 1.0 + 0.5j
    s12 = 0.3 - 0.2j
    h11 = 0.5 + 0.1j
    h12 = -0.4 + 0.3j

    co, si = mb03hz(s11, s12, h11, h12)

    q = np.array([
        [co, si],
        [-np.conj(si), co]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q @ q.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )


def test_mb03hz_identity_like():
    """
    Test with identity-like input matrices.

    Validates edge case where diagonal elements are close to identity.
    """
    from ctrlsys import mb03hz

    s11 = 1.0 + 0.0j
    s12 = 0.1 + 0.05j
    h11 = 1.0 + 0.0j
    h12 = 0.1 - 0.05j

    co, si = mb03hz(s11, s12, h11, h12)

    assert np.isfinite(co)
    assert np.isfinite(si)

    assert isinstance(co, float)
    assert isinstance(si, complex)


def test_mb03hz_real_valued():
    """
    Test with purely real input values.

    Validates routine handles real-valued complex input correctly.
    """
    from ctrlsys import mb03hz

    s11 = 2.0 + 0.0j
    s12 = 1.0 + 0.0j
    h11 = 1.5 + 0.0j
    h12 = -0.5 + 0.0j

    co, si = mb03hz(s11, s12, h11, h12)

    q = np.array([
        [co, si],
        [-np.conj(si), co]
    ], dtype=np.complex128)

    np.testing.assert_allclose(
        q @ q.conj().T, np.eye(2), rtol=1e-14, atol=1e-14
    )


def test_mb03hz_determinant_unity():
    """
    Test that Q has determinant of magnitude 1.

    Unitary matrices have |det| = 1.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03hz

    np.random.seed(456)

    s11 = 1.2 + 0.8j
    s12 = 0.4 - 0.1j
    h11 = 0.7 + 0.3j
    h12 = -0.3 + 0.2j

    co, si = mb03hz(s11, s12, h11, h12)

    q = np.array([
        [co, si],
        [-np.conj(si), co]
    ], dtype=np.complex128)

    det_q = np.linalg.det(q)

    np.testing.assert_allclose(np.abs(det_q), 1.0, rtol=1e-14)


def test_mb03hz_orthogonality_constraint():
    """
    Test that CO^2 + |SI|^2 = 1 (unitary constraint).

    For the matrix Q to be unitary with structure [[CO, SI], [-conj(SI), CO]],
    we need CO^2 + |SI|^2 = 1 where CO is real.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03hz

    np.random.seed(789)

    s11 = 2.5 - 1.3j
    s12 = 0.7 + 0.4j
    h11 = 1.1 - 0.6j
    h12 = -0.8 + 0.5j

    co, si = mb03hz(s11, s12, h11, h12)

    np.testing.assert_allclose(
        co * co + np.abs(si) ** 2, 1.0, rtol=1e-14
    )
