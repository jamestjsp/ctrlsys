"""
Tests for SG03BZ - Full generalized Lyapunov solver (complex).

Computes Cholesky factor U of X = op(U)^H * op(U) for:

Continuous-time (DICO='C'):
    op(A)^H * X * op(E) + op(E)^H * X * op(A) = -SCALE^2 * op(B)^H * op(B)

Discrete-time (DICO='D'):
    op(A)^H * X * op(A) - op(E)^H * X * op(E) = -SCALE^2 * op(B)^H * op(B)

where op(K) = K (TRANS='N') or op(K) = K^H (TRANS='C').

Unlike SG03BS/SG03BT which work on triangular Schur form, SG03BZ:
1. Performs QZ factorization if FACT='N'
2. Transforms B appropriately
3. Calls SG03BS (discrete) or SG03BT (continuous) on reduced problem
4. Transforms solution back

Key test cases:
- FACT='N' (compute Schur factorization)
- FACT='F' (factorization supplied)
- DICO='C' (continuous-time, c-stable: eigenvalues with Re < 0)
- DICO='D' (discrete-time, d-stable: eigenvalue modulus < 1)
- TRANS='N' and TRANS='C'
- Identity Q/Z detection
- Various M vs N relationships (M < N, M >= N)
"""

import numpy as np
from ctrlsys import sg03bz


def test_sg03bz_continuous_trans_n_fact_n():
    """
    Test SG03BZ with DICO='C', TRANS='N', FACT='N'.

    Continuous-time equation with no pre-factorization:
        A^H * X * E + E^H * X * A = -SCALE^2 * B^H * B
    where X = U^H * U.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3

    a = np.array([
        [-0.5+0.1j, 0.2+0.05j, 0.1+0.02j],
        [0.1-0.05j, -0.4-0.1j, 0.15+0.03j],
        [0.05+0.02j, 0.08-0.04j, -0.3+0.05j]
    ], dtype=complex, order='F')

    e = np.array([
        [1.0+0j, 0.1+0.02j, 0.05+0.01j],
        [0.0, 1.0+0j, 0.08+0.02j],
        [0.0, 0.0, 1.0+0j]
    ], dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j, 0.1+0.05j],
        [0.0, 0.3, 0.15+0.08j]
    ], dtype=complex, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'N', a, e, q, z, b)

    assert info == 0, f"SG03BZ failed with INFO={info}"
    assert 0.0 < scale <= 1.0, f"Invalid scale: {scale}"

    np.testing.assert_allclose(np.tril(u, -1), 0, atol=1e-14)

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ e_orig + e_orig.conj().T @ x @ a_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-15)
    assert residual < 1e-9, f"Lyapunov residual too large: {residual}"

    assert np.all(np.diag(u).real >= -1e-14), "U diagonal should be non-negative real"


def test_sg03bz_continuous_trans_c_fact_n():
    """
    Test SG03BZ with DICO='C', TRANS='C', FACT='N'.

    Continuous-time conjugate transpose equation:
        A * X * E^H + E * X * A^H = -SCALE^2 * B * B^H
    where X = U * U^H.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3
    m = 2

    a = np.array([
        [-0.4+0.08j, 0.15+0.04j, 0.08+0.02j],
        [0.1-0.04j, -0.35-0.08j, 0.12+0.03j],
        [0.04+0.02j, 0.06-0.03j, -0.25+0.04j]
    ], dtype=complex, order='F')

    e = np.array([
        [1.0+0j, 0.12+0.03j, 0.06+0.02j],
        [0.0, 1.0+0j, 0.1+0.02j],
        [0.0, 0.0, 1.0+0j]
    ], dtype=complex, order='F')

    b = np.array([
        [0.4+0j, 0.0+0j, 0.0+0j],
        [0.18+0.08j, 0.35+0j, 0.0+0j],
        [0.09+0.04j, 0.12+0.06j, 0.2+0j]
    ], dtype=complex, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'C', a, e, q, z, b)

    assert info == 0, f"SG03BZ failed with INFO={info}"
    assert 0.0 < scale <= 1.0, f"Invalid scale: {scale}"

    np.testing.assert_allclose(np.tril(u, -1), 0, atol=1e-14)

    x = u @ u.conj().T
    lhs = a_orig @ x @ e_orig.conj().T + e_orig @ x @ a_orig.conj().T
    rhs = -scale**2 * (b_orig @ b_orig.conj().T)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-15)
    assert residual < 1e-9, f"Lyapunov residual too large: {residual}"


def test_sg03bz_discrete_trans_n_fact_n():
    """
    Test SG03BZ with DICO='D', TRANS='N', FACT='N'.

    Discrete-time equation:
        A^H * X * A - E^H * X * E = -SCALE^2 * B^H * B
    where X = U^H * U.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3
    m = 2

    a = np.array([
        [0.3+0.1j, 0.1+0.05j, 0.05+0.02j],
        [0.05-0.02j, 0.2-0.08j, 0.08+0.03j],
        [0.02+0.01j, 0.04-0.02j, 0.15+0.05j]
    ], dtype=complex, order='F')

    e = np.array([
        [1.0+0j, 0.1+0.02j, 0.05+0.01j],
        [0.0, 1.0+0j, 0.08+0.02j],
        [0.0, 0.0, 1.0+0j]
    ], dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j, 0.1+0.05j],
        [0.0, 0.3, 0.15+0.08j]
    ], dtype=complex, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('D', 'N', 'N', a, e, q, z, b)

    assert info == 0, f"SG03BZ failed with INFO={info}"
    assert 0.0 < scale <= 1.0, f"Invalid scale: {scale}"

    np.testing.assert_allclose(np.tril(u, -1), 0, atol=1e-14)

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ a_orig - e_orig.conj().T @ x @ e_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-15)
    assert residual < 1e-9, f"Lyapunov residual too large: {residual}"


def test_sg03bz_discrete_trans_c_fact_n():
    """
    Test SG03BZ with DICO='D', TRANS='C', FACT='N'.

    Discrete-time conjugate transpose equation:
        A * X * A^H - E * X * E^H = -SCALE^2 * B * B^H
    where X = U * U^H.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3
    m = 2

    a = np.array([
        [0.25+0.08j, 0.12+0.04j, 0.06+0.02j],
        [0.04-0.02j, 0.18-0.06j, 0.09+0.03j],
        [0.02+0.01j, 0.03-0.02j, 0.12+0.04j]
    ], dtype=complex, order='F')

    e = np.array([
        [1.0+0j, 0.15+0.03j, 0.08+0.02j],
        [0.0, 1.0+0j, 0.1+0.02j],
        [0.0, 0.0, 1.0+0j]
    ], dtype=complex, order='F')

    b = np.array([
        [0.4+0j, 0.0+0j, 0.0+0j],
        [0.18+0.08j, 0.35+0j, 0.0+0j],
        [0.09+0.04j, 0.12+0.06j, 0.2+0j]
    ], dtype=complex, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('D', 'N', 'C', a, e, q, z, b)

    assert info == 0, f"SG03BZ failed with INFO={info}"
    assert 0.0 < scale <= 1.0, f"Invalid scale: {scale}"

    np.testing.assert_allclose(np.tril(u, -1), 0, atol=1e-14)

    x = u @ u.conj().T
    lhs = a_orig @ x @ a_orig.conj().T - e_orig @ x @ e_orig.conj().T
    rhs = -scale**2 * (b_orig @ b_orig.conj().T)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-15)
    assert residual < 1e-9, f"Lyapunov residual too large: {residual}"


def test_sg03bz_continuous_fact_f():
    """
    Test SG03BZ with FACT='F' (factorization supplied).

    When FACT='F', A and E must already be in upper triangular (Schur) form,
    and Q, Z must be the unitary transformation matrices.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 3
    m = 2

    a = np.array([
        [-0.5+0.1j, 0.1+0.05j, 0.05+0.02j],
        [0.0, -0.3-0.1j, 0.08+0.03j],
        [0.0, 0.0, -0.2+0.05j]
    ], dtype=complex, order='F')

    e = np.array([
        [1.0+0j, 0.1+0.02j, 0.05+0.01j],
        [0.0, 1.0+0j, 0.08+0.02j],
        [0.0, 0.0, 1.0+0j]
    ], dtype=complex, order='F')

    q = np.eye(n, dtype=complex, order='F')
    z = np.eye(n, dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j, 0.1+0.05j],
        [0.0, 0.3, 0.15+0.08j]
    ], dtype=complex, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    u, scale, _alpha, _beta, info = sg03bz('C', 'F', 'N', a, e, q, z, b)

    assert info == 0, f"SG03BZ failed with INFO={info}"
    assert 0.0 < scale <= 1.0, f"Invalid scale: {scale}"

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ e_orig + e_orig.conj().T @ x @ a_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-15)
    assert residual < 1e-9, f"Lyapunov residual too large: {residual}"


def test_sg03bz_m_equals_n():
    """
    Test SG03BZ when M equals N.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 3
    m = 3

    a = np.array([
        [-0.4+0.1j, 0.15+0.04j, 0.08+0.02j],
        [0.1-0.03j, -0.35-0.08j, 0.12+0.03j],
        [0.05+0.02j, 0.06-0.03j, -0.25+0.04j]
    ], dtype=complex, order='F')

    e = np.eye(n, dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j, 0.1+0.05j],
        [0.0, 0.4, 0.15+0.08j],
        [0.0, 0.0, 0.3]
    ], dtype=complex, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'N', a, e, q, z, b)

    assert info == 0, f"SG03BZ failed with INFO={info}"

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ e_orig + e_orig.conj().T @ x @ a_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-15)
    assert residual < 1e-9, f"Lyapunov residual too large: {residual}"


def test_sg03bz_m_greater_n():
    """
    Test SG03BZ when M > N.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 3
    m = 5

    a = np.array([
        [-0.4+0.1j, 0.15+0.04j, 0.08+0.02j],
        [0.1-0.03j, -0.35-0.08j, 0.12+0.03j],
        [0.05+0.02j, 0.06-0.03j, -0.25+0.04j]
    ], dtype=complex, order='F')

    e = np.eye(n, dtype=complex, order='F')

    b = np.zeros((m, n), dtype=complex, order='F')
    for i in range(m):
        for j in range(n):
            b[i, j] = 0.1 * (np.random.randn() + 1j * np.random.randn())
    b = np.asfortranarray(b)

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'N', a, e, q, z, b)

    assert info == 0, f"SG03BZ failed with INFO={info}"

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ e_orig + e_orig.conj().T @ x @ a_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-15)
    assert residual < 1e-9, f"Lyapunov residual too large: {residual}"


def test_sg03bz_n0():
    """Test SG03BZ with N=0 (quick return)."""
    n = 0
    m = 0

    a = np.zeros((0, 0), dtype=complex, order='F')
    e = np.zeros((0, 0), dtype=complex, order='F')
    q = np.zeros((0, 0), dtype=complex, order='F')
    z = np.zeros((0, 0), dtype=complex, order='F')
    b = np.zeros((0, 0), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'N', a, e, q, z, b)

    assert info == 0
    assert scale == 1.0


def test_sg03bz_m0():
    """Test SG03BZ with M=0 (U should be zero)."""
    n = 3
    m = 0

    a = np.array([
        [-0.5+0.1j, 0.1+0.05j, 0.05+0.02j],
        [0.0, -0.3-0.1j, 0.08+0.03j],
        [0.0, 0.0, -0.2+0.05j]
    ], dtype=complex, order='F')

    e = np.eye(n, dtype=complex, order='F')

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')
    b = np.zeros((0, n), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'N', a, e, q, z, b)

    assert info == 0
    np.testing.assert_allclose(u, 0, atol=1e-14)


def test_sg03bz_continuous_unstable():
    """Test SG03BZ returns INFO=5 for c-unstable pencil (DICO='C')."""
    n = 2
    m = 2

    a = np.array([
        [0.5+0j, 0.1+0.05j],
        [0.0, 0.3-0.1j]
    ], dtype=complex, order='F')

    e = np.eye(n, dtype=complex, order='F')

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j],
        [0.0, 0.3]
    ], dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'N', a, e, q, z, b)

    assert info == 5, f"Expected INFO=5 for c-unstable system, got {info}"


def test_sg03bz_discrete_unstable():
    """Test SG03BZ returns INFO=6 for d-unstable pencil (DICO='D')."""
    n = 2
    m = 2

    a = np.array([
        [1.5+0j, 0.1+0.05j],
        [0.0, 1.2-0.1j]
    ], dtype=complex, order='F')

    e = np.eye(n, dtype=complex, order='F')

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j],
        [0.0, 0.3]
    ], dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('D', 'N', 'N', a, e, q, z, b)

    assert info == 6, f"Expected INFO=6 for d-unstable system, got {info}"


def test_sg03bz_invalid_dico():
    """Test SG03BZ returns INFO=-1 for invalid DICO."""
    n = 2
    m = 2

    a = np.array([
        [-0.5+0.1j, 0.1+0.05j],
        [0.0, -0.3-0.1j]
    ], dtype=complex, order='F')

    e = np.eye(n, dtype=complex, order='F')
    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j],
        [0.0, 0.3]
    ], dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('X', 'N', 'N', a, e, q, z, b)

    assert info == -1, f"Expected INFO=-1 for invalid DICO, got {info}"


def test_sg03bz_invalid_fact():
    """Test SG03BZ returns INFO=-2 for invalid FACT."""
    n = 2
    m = 2

    a = np.array([
        [-0.5+0.1j, 0.1+0.05j],
        [0.0, -0.3-0.1j]
    ], dtype=complex, order='F')

    e = np.eye(n, dtype=complex, order='F')
    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j],
        [0.0, 0.3]
    ], dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'X', 'N', a, e, q, z, b)

    assert info == -2, f"Expected INFO=-2 for invalid FACT, got {info}"


def test_sg03bz_invalid_trans():
    """Test SG03BZ returns INFO=-3 for invalid TRANS."""
    n = 2
    m = 2

    a = np.array([
        [-0.5+0.1j, 0.1+0.05j],
        [0.0, -0.3-0.1j]
    ], dtype=complex, order='F')

    e = np.eye(n, dtype=complex, order='F')
    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j],
        [0.0, 0.3]
    ], dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'X', a, e, q, z, b)

    assert info == -3, f"Expected INFO=-3 for invalid TRANS, got {info}"


def test_sg03bz_eigenvalue_output():
    """
    Test that alpha/beta eigenvalue outputs are correct.

    For triangular A, E (FACT='F'), eigenvalues should match diagonal entries.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n = 3
    m = 2

    a = np.array([
        [-0.5+0.1j, 0.1+0.05j, 0.05+0.02j],
        [0.0, -0.3-0.1j, 0.08+0.03j],
        [0.0, 0.0, -0.2+0.05j]
    ], dtype=complex, order='F')

    e = np.array([
        [1.0+0j, 0.1+0.02j, 0.05+0.01j],
        [0.0, 0.8+0j, 0.08+0.02j],
        [0.0, 0.0, 0.6+0j]
    ], dtype=complex, order='F')

    q = np.eye(n, dtype=complex, order='F')
    z = np.eye(n, dtype=complex, order='F')

    b = np.array([
        [0.5, 0.2+0.1j, 0.1+0.05j],
        [0.0, 0.3, 0.15+0.08j]
    ], dtype=complex, order='F')

    u, scale, alpha, beta, info = sg03bz('C', 'F', 'N', a, e, q, z, b)

    assert info == 0, f"SG03BZ failed with INFO={info}"

    np.testing.assert_allclose(alpha, np.diag(a), rtol=1e-14)
    np.testing.assert_allclose(beta, np.diag(e), rtol=1e-14)


def test_sg03bz_large_system():
    """
    Test SG03BZ with larger system.

    Random seed: 2024 (for reproducibility)
    """
    np.random.seed(2024)
    n = 8
    m = 4

    diag_a = -0.3 - 0.2 * np.random.rand(n) + 0.1j * np.random.randn(n)
    a = np.diag(diag_a).astype(complex, order='F')
    for i in range(n):
        for j in range(i+1, n):
            a[i, j] = 0.05 * (np.random.randn() + 1j * np.random.randn())
    for i in range(1, n):
        for j in range(i):
            a[i, j] = 0.02 * (np.random.randn() + 1j * np.random.randn())
    a = np.asfortranarray(a)

    e = np.eye(n, dtype=complex, order='F')
    for i in range(n):
        for j in range(i+1, n):
            e[i, j] = 0.02 * (np.random.randn() + 1j * np.random.randn())
    e = np.asfortranarray(e)

    b = 0.1 * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
    b = np.asfortranarray(b.astype(complex))

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'N', a, e, q, z, b)

    assert info == 0, f"SG03BZ failed with INFO={info}"
    assert 0.0 < scale <= 1.0

    x = u.conj().T @ u
    lhs = a_orig.conj().T @ x @ e_orig + e_orig.conj().T @ x @ a_orig
    rhs = -scale**2 * (b_orig.conj().T @ b_orig)
    residual = np.linalg.norm(lhs - rhs, 'fro') / max(np.linalg.norm(lhs, 'fro'), 1e-15)
    assert residual < 1e-8, f"Lyapunov residual too large: {residual}"


def test_sg03bz_positive_semidefinite_solution():
    """
    Validate X = U^H * U is positive semidefinite.

    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    n = 4
    m = 3

    diag_a = -0.3 - 0.2 * np.random.rand(n) + 0.1j * np.random.randn(n)
    a = np.diag(diag_a).astype(complex, order='F')
    for i in range(n):
        for j in range(i+1, n):
            a[i, j] = 0.05 * (np.random.randn() + 1j * np.random.randn())
    a = np.asfortranarray(a)

    e = np.eye(n, dtype=complex, order='F')

    b = 0.2 * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
    b = np.asfortranarray(b.astype(complex))

    q = np.zeros((n, n), dtype=complex, order='F')
    z = np.zeros((n, n), dtype=complex, order='F')

    u, scale, _alpha, _beta, info = sg03bz('C', 'N', 'N', a.copy(), e.copy(), q, z, b.copy())

    assert info == 0, f"SG03BZ failed with INFO={info}"

    np.testing.assert_allclose(np.tril(u, -1), 0, atol=1e-14)

    x = u.conj().T @ u

    eig_x = np.linalg.eigvalsh(x)
    assert np.all(eig_x >= -1e-10), f"X should be positive semidefinite, min eig={eig_x.min()}"
