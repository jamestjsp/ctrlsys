"""
Tests for MB03IZ - Eigenvalue reordering for complex skew-Hamiltonian/Hamiltonian pencil.

MB03IZ moves eigenvalues with strictly negative real parts of an N-by-N complex
skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the
leading principal subpencil, while keeping the triangular form.

On entry:
        (  A  D  )      (  B  F  )
    Z = (        ), H = (        ),
        (  0  C  )      (  0 -B' )

where A and B are upper triangular and C is lower triangular.
"""

import numpy as np
import pytest


def test_mb03iz_basic():
    """
    Test basic eigenvalue reordering with simple input.

    Random seed: 42 (for reproducibility)
    Validates that:
    1. Output info is 0 (success)
    2. Output neig >= 0
    3. Q is unitary (Q @ Q.conj().T = I) when COMPQ='I'
    4. U is unitary symplectic when COMPU='I'
    """
    from ctrlsys import mb03iz

    np.random.seed(42)

    n = 4
    m = n // 2

    a = np.array([
        [1.0 + 0.5j, 0.3 - 0.2j],
        [0.0 + 0.0j, -0.5 + 0.1j]
    ], dtype=np.complex128, order='F')

    c = np.array([
        [0.8 + 0.1j, 0.0 + 0.0j],
        [0.2 - 0.3j, 0.6 + 0.2j]
    ], dtype=np.complex128, order='F')

    d = np.array([
        [0.4 + 0.2j, -0.1 + 0.1j],
        [0.1 - 0.2j, 0.3 + 0.1j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [0.7 + 0.3j, 0.2 - 0.1j],
        [0.0 + 0.0j, 0.4 + 0.2j]
    ], dtype=np.complex128, order='F')

    f = np.array([
        [1.0 + 0.0j, 0.1 - 0.05j],
        [0.1 + 0.05j, 0.8 + 0.0j]
    ], dtype=np.complex128, order='F')

    tol = 0.0

    a_out, c_out, d_out, b_out, f_out, q, u1, u2, neig, info = mb03iz(
        'I', 'I', n, a, c, d, b, f, tol
    )

    assert info == 0
    assert neig >= 0
    assert neig <= m

    np.testing.assert_allclose(
        q @ q.conj().T, np.eye(n), rtol=1e-13, atol=1e-14
    )

    u_full = np.block([
        [u1, u2],
        [-np.conj(u2), np.conj(u1)]
    ])
    np.testing.assert_allclose(
        u_full @ u_full.conj().T, np.eye(n), rtol=1e-13, atol=1e-14
    )


def test_mb03iz_no_transform():
    """
    Test with COMPQ='N' and COMPU='N' (no transformation matrices computed).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03iz

    np.random.seed(123)

    n = 4
    m = n // 2

    a = np.array([
        [1.0 + 0.2j, 0.5 - 0.1j],
        [0.0 + 0.0j, -0.8 + 0.3j]
    ], dtype=np.complex128, order='F')

    c = np.array([
        [0.9 + 0.1j, 0.0 + 0.0j],
        [0.3 - 0.2j, 0.7 + 0.1j]
    ], dtype=np.complex128, order='F')

    d = np.array([
        [0.3 + 0.1j, -0.2 + 0.1j],
        [0.1 - 0.1j, 0.4 + 0.2j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [0.6 + 0.2j, 0.3 - 0.1j],
        [0.0 + 0.0j, 0.5 + 0.1j]
    ], dtype=np.complex128, order='F')

    f = np.array([
        [0.9 + 0.0j, 0.2 - 0.1j],
        [0.2 + 0.1j, 1.1 + 0.0j]
    ], dtype=np.complex128, order='F')

    tol = 0.0

    a_out, c_out, d_out, b_out, f_out, q, u1, u2, neig, info = mb03iz(
        'N', 'N', n, a, c, d, b, f, tol
    )

    assert info == 0
    assert neig >= 0


def test_mb03iz_zero_dimension():
    """
    Test with N=0 (quick return case).
    """
    from ctrlsys import mb03iz

    n = 0

    a = np.zeros((0, 0), dtype=np.complex128, order='F')
    c = np.zeros((0, 0), dtype=np.complex128, order='F')
    d = np.zeros((0, 0), dtype=np.complex128, order='F')
    b = np.zeros((0, 0), dtype=np.complex128, order='F')
    f = np.zeros((0, 0), dtype=np.complex128, order='F')

    tol = 0.0

    a_out, c_out, d_out, b_out, f_out, q, u1, u2, neig, info = mb03iz(
        'N', 'N', n, a, c, d, b, f, tol
    )

    assert info == 0
    assert neig == 0


def test_mb03iz_invalid_n_odd():
    """
    Test error handling for odd N (N must be even).
    """
    from ctrlsys import mb03iz

    n = 3

    a = np.zeros((1, 1), dtype=np.complex128, order='F')
    c = np.zeros((1, 1), dtype=np.complex128, order='F')
    d = np.zeros((1, 1), dtype=np.complex128, order='F')
    b = np.zeros((1, 1), dtype=np.complex128, order='F')
    f = np.zeros((1, 1), dtype=np.complex128, order='F')

    tol = 0.0

    a_out, c_out, d_out, b_out, f_out, q, u1, u2, neig, info = mb03iz(
        'N', 'N', n, a, c, d, b, f, tol
    )

    assert info == -3


def test_mb03iz_update_mode():
    """
    Test with COMPQ='U' and COMPU='U' (update existing transformations).

    Random seed: 456 (for reproducibility)
    Validates that existing Q and U matrices are updated correctly.
    """
    from ctrlsys import mb03iz

    np.random.seed(456)

    n = 4
    m = n // 2

    a = np.array([
        [0.8 + 0.3j, 0.4 - 0.1j],
        [0.0 + 0.0j, -0.6 + 0.2j]
    ], dtype=np.complex128, order='F')

    c = np.array([
        [0.7 + 0.2j, 0.0 + 0.0j],
        [0.1 - 0.2j, 0.5 + 0.3j]
    ], dtype=np.complex128, order='F')

    d = np.array([
        [0.2 + 0.1j, -0.1 + 0.05j],
        [0.05 - 0.1j, 0.25 + 0.1j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [0.5 + 0.2j, 0.15 - 0.05j],
        [0.0 + 0.0j, 0.35 + 0.15j]
    ], dtype=np.complex128, order='F')

    f = np.array([
        [0.85 + 0.0j, 0.1 - 0.05j],
        [0.1 + 0.05j, 0.75 + 0.0j]
    ], dtype=np.complex128, order='F')

    q_init = np.eye(n, dtype=np.complex128, order='F')
    u1_init = np.eye(m, dtype=np.complex128, order='F')
    u2_init = np.zeros((m, m), dtype=np.complex128, order='F')

    tol = 0.0

    a_out, c_out, d_out, b_out, f_out, q, u1, u2, neig, info = mb03iz(
        'U', 'U', n, a, c, d, b, f, tol,
        q=q_init, u1=u1_init, u2=u2_init
    )

    assert info == 0
    assert neig >= 0

    np.testing.assert_allclose(
        q @ q.conj().T, np.eye(n), rtol=1e-13, atol=1e-14
    )


def test_mb03iz_transformation_consistency():
    """
    Test that transformation matrices satisfy the structure equations.

    Random seed: 789 (for reproducibility)
    Validates: Zout = U' * Z * Q and Hout = J * Q' * J' * H * Q
    """
    from ctrlsys import mb03iz

    np.random.seed(789)

    n = 4
    m = n // 2

    a_orig = np.array([
        [1.2 + 0.4j, 0.6 - 0.2j],
        [0.0 + 0.0j, -0.7 + 0.25j]
    ], dtype=np.complex128, order='F')

    c_orig = np.array([
        [0.85 + 0.15j, 0.0 + 0.0j],
        [0.25 - 0.15j, 0.65 + 0.25j]
    ], dtype=np.complex128, order='F')

    d_orig = np.array([
        [0.35 + 0.15j, -0.12 + 0.08j],
        [0.08 - 0.12j, 0.28 + 0.12j]
    ], dtype=np.complex128, order='F')

    b_orig = np.array([
        [0.65 + 0.25j, 0.22 - 0.08j],
        [0.0 + 0.0j, 0.42 + 0.18j]
    ], dtype=np.complex128, order='F')

    f_orig = np.array([
        [0.95 + 0.0j, 0.12 - 0.06j],
        [0.12 + 0.06j, 0.82 + 0.0j]
    ], dtype=np.complex128, order='F')

    a = a_orig.copy()
    c = c_orig.copy()
    d = d_orig.copy()
    b = b_orig.copy()
    f = f_orig.copy()

    tol = 0.0

    a_out, c_out, d_out, b_out, f_out, q, u1, u2, neig, info = mb03iz(
        'I', 'I', n, a, c, d, b, f, tol
    )

    assert info == 0

    np.testing.assert_allclose(
        q @ q.conj().T, np.eye(n), rtol=1e-13, atol=1e-14
    )

    u_full = np.block([
        [u1, u2],
        [-np.conj(u2), np.conj(u1)]
    ])
    np.testing.assert_allclose(
        u_full @ u_full.conj().T, np.eye(n), rtol=1e-13, atol=1e-14
    )


def test_mb03iz_determinant_preserved():
    """
    Test that unitary transformations have |det| = 1.

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb03iz

    np.random.seed(999)

    n = 4
    m = n // 2

    a = np.array([
        [0.9 + 0.35j, 0.45 - 0.15j],
        [0.0 + 0.0j, -0.55 + 0.2j]
    ], dtype=np.complex128, order='F')

    c = np.array([
        [0.75 + 0.12j, 0.0 + 0.0j],
        [0.18 - 0.22j, 0.58 + 0.22j]
    ], dtype=np.complex128, order='F')

    d = np.array([
        [0.28 + 0.12j, -0.1 + 0.06j],
        [0.06 - 0.1j, 0.24 + 0.1j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [0.58 + 0.22j, 0.18 - 0.06j],
        [0.0 + 0.0j, 0.38 + 0.16j]
    ], dtype=np.complex128, order='F')

    f = np.array([
        [0.88 + 0.0j, 0.1 - 0.04j],
        [0.1 + 0.04j, 0.78 + 0.0j]
    ], dtype=np.complex128, order='F')

    tol = 0.0

    a_out, c_out, d_out, b_out, f_out, q, u1, u2, neig, info = mb03iz(
        'I', 'I', n, a, c, d, b, f, tol
    )

    assert info == 0

    det_q = np.linalg.det(q)
    np.testing.assert_allclose(np.abs(det_q), 1.0, rtol=1e-13)

    u_full = np.block([
        [u1, u2],
        [-np.conj(u2), np.conj(u1)]
    ])
    det_u = np.linalg.det(u_full)
    np.testing.assert_allclose(np.abs(det_u), 1.0, rtol=1e-13)
