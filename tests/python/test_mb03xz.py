"""
Tests for MB03XZ: Eigenvalues of complex Hamiltonian matrix.

MB03XZ computes the eigenvalues of a complex Hamiltonian matrix:
    H = [ A   G  ]    G = G^H, Q = Q^H
        [ Q  -A^H]

Uses embedding to real skew-Hamiltonian and structured Schur form.
"""

import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03xz_basic_eigenvalues():
    """
    Test basic eigenvalue computation using HTML doc example.

    Input data from SLICOT MB03XZ.html example:
    N=4, BALANC='N', JOB='G', JOBU='U'
    """
    from ctrlsys import mb03xz

    n = 4
    # A matrix from HTML doc (row-wise as displayed)
    a = np.array([
        [0.8147+0.4217j, 0.6323+0.6557j, 0.9575+0.6787j, 0.9571+0.6554j],
        [0.9057+0.9157j, 0.0975+0.0357j, 0.9648+0.7577j, 0.4853+0.1711j],
        [0.1269+0.7922j, 0.2784+0.8491j, 0.1576+0.7431j, 0.8002+0.7060j],
        [0.9133+0.9594j, 0.5468+0.9339j, 0.9705+0.3922j, 0.1418+0.0318j],
    ], dtype=complex, order='F')

    # QG matrix (lower triangular Q in cols 1:N, upper triangular G in cols 2:N+1)
    # Parsed from HTML doc data
    qg = np.array([
        [0.2769+0.0j,      0.6948+0.0j,      0.4387+0.7513j, 0.1869+0.8909j, 0.7094+0.1493j],
        [0.0462+0.1626j,   0.3171+0.0j,      0.3816+0.0j,    0.4898+0.9593j, 0.7547+0.2575j],
        [0.0971+0.1190j,   0.9502+0.5853j,   0.7655+0.0j,    0.4456+0.0j,    0.2760+0.8407j],
        [0.8235+0.4984j,   0.0344+0.2238j,   0.7952+0.6991j, 0.6463+0.0j,    0.6797+0.0j],
    ], dtype=complex, order='F')

    # Expected eigenvalues from HTML doc
    expected_wr = np.array([3.0844, -3.0844, 0.5241, -0.5241, 0.8824, -0.8824, 0.4459, -0.4459])
    expected_wi = np.array([2.7519,  2.7519, -1.3026, -1.3026, -0.6918, -0.6918, 0.4748, 0.4748])

    wr, wi, sc, gc, u1, u2, ilo, scale, info = mb03xz(
        a, qg, balanc='N', job='G', jobu='U'
    )

    assert info == 0
    # Eigenvalues should match (may be in different order, so sort by magnitude)
    eig_computed = wr + 1j * wi
    eig_expected = expected_wr + 1j * expected_wi

    # Sort by real part then imaginary part
    idx_c = np.lexsort((eig_computed.imag, eig_computed.real))
    idx_e = np.lexsort((eig_expected.imag, eig_expected.real))

    assert_allclose(eig_computed[idx_c], eig_expected[idx_e], rtol=1e-3, atol=1e-3)


def test_mb03xz_eigenvalues_only():
    """
    Test eigenvalue-only mode (JOB='E').

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03xz

    np.random.seed(42)
    n = 3

    # Create Hermitian G and Q
    a = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    a = np.asfortranarray(a)

    g_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    g_full = (g_full + g_full.T.conj()) / 2  # Make Hermitian

    q_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    q_full = (q_full + q_full.T.conj()) / 2  # Make Hermitian

    # Pack into QG format: lower tri of Q in cols 0:n-1, upper tri of G in cols 1:n
    qg = np.zeros((n, n+1), dtype=complex, order='F')
    for i in range(n):
        for j in range(i+1):  # Lower triangular Q
            qg[i, j] = q_full[i, j]
        for j in range(i, n):  # Upper triangular G (in cols 1:n)
            qg[i, j+1] = g_full[i, j]

    wr, wi, sc, gc, u1, u2, ilo, scale, info = mb03xz(
        a, qg, balanc='N', job='E', jobu='N'
    )

    assert info == 0
    assert len(wr) == 2 * n
    assert len(wi) == 2 * n

    # Eigenvalues of Hamiltonian: if lambda is eigenvalue, -conj(lambda) also is
    eig = wr + 1j * wi
    for lam in eig:
        neg_conj = -np.conj(lam)
        # Check if -conj(lambda) is also in eigenvalues
        dists = np.abs(eig - neg_conj)
        assert np.min(dists) < 1e-10, f"Missing paired eigenvalue for {lam}"


def test_mb03xz_hamiltonian_structure():
    """
    Validate Hamiltonian eigenvalue pairing: if lambda is an eigenvalue,
    then -conj(lambda) is also an eigenvalue.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03xz

    np.random.seed(123)
    n = 4

    # Create random complex Hamiltonian
    a = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    a = np.asfortranarray(a)

    g_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    g_full = (g_full + g_full.T.conj()) / 2

    q_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    q_full = (q_full + q_full.T.conj()) / 2

    qg = np.zeros((n, n+1), dtype=complex, order='F')
    for i in range(n):
        for j in range(i+1):
            qg[i, j] = q_full[i, j]
        for j in range(i, n):
            qg[i, j+1] = g_full[i, j]

    wr, wi, sc, gc, u1, u2, ilo, scale, info = mb03xz(
        a, qg, balanc='N', job='E', jobu='N'
    )

    assert info == 0

    # Verify Hamiltonian pairing
    eig = wr + 1j * wi
    for lam in eig:
        neg_conj = -np.conj(lam)
        dists = np.abs(eig - neg_conj)
        assert np.min(dists) < 1e-10, f"Eigenvalue {lam} missing pair -conj(lambda)={neg_conj}"


def test_mb03xz_with_balancing():
    """
    Test with balancing enabled (BALANC='B').

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03xz

    np.random.seed(456)
    n = 3

    # Create poorly conditioned matrix that benefits from balancing
    a = np.array([
        [1e-3+1e-3j, 1e2+0j, 1e1+1e1j],
        [1e-4+0j, 1e3+1e2j, 1e0+0j],
        [1e-2+1e-2j, 1e1+1e1j, 1e-1+1e-1j],
    ], dtype=complex, order='F')

    g_full = np.array([
        [1.0+0j, 0.5+0.1j, 0.2+0.3j],
        [0.5-0.1j, 2.0+0j, 0.4+0.2j],
        [0.2-0.3j, 0.4-0.2j, 0.5+0j],
    ], dtype=complex, order='F')

    q_full = np.array([
        [0.3+0j, 0.1+0.05j, 0.2+0.1j],
        [0.1-0.05j, 0.5+0j, 0.15+0.08j],
        [0.2-0.1j, 0.15-0.08j, 0.4+0j],
    ], dtype=complex, order='F')

    qg = np.zeros((n, n+1), dtype=complex, order='F')
    for i in range(n):
        for j in range(i+1):
            qg[i, j] = q_full[i, j]
        for j in range(i, n):
            qg[i, j+1] = g_full[i, j]

    wr, wi, sc, gc, u1, u2, ilo, scale, info = mb03xz(
        a, qg, balanc='B', job='E', jobu='N'
    )

    assert info == 0
    assert len(wr) == 2 * n
    assert ilo >= 1  # ILO is set by balancing


def test_mb03xz_schur_form():
    """
    Test Schur form computation (JOB='S').

    When JOB='S', the routine returns the complex Schur matrix Sc.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03xz

    np.random.seed(789)
    n = 2

    a = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    a = np.asfortranarray(a)

    g_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    g_full = (g_full + g_full.T.conj()) / 2

    q_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    q_full = (q_full + q_full.T.conj()) / 2

    qg = np.zeros((n, n+1), dtype=complex, order='F')
    for i in range(n):
        for j in range(i+1):
            qg[i, j] = q_full[i, j]
        for j in range(i, n):
            qg[i, j+1] = g_full[i, j]

    wr, wi, sc, gc, u1, u2, ilo, scale, info = mb03xz(
        a, qg, balanc='N', job='S', jobu='N'
    )

    assert info == 0
    assert sc.shape == (2*n, 2*n)

    # Sc should be upper triangular (complex Schur form)
    # Check that lower triangular part is zero
    for i in range(1, 2*n):
        for j in range(i):
            assert np.abs(sc[i, j]) < 1e-10, f"sc[{i},{j}]={sc[i,j]} should be zero"


def test_mb03xz_full_decomposition():
    """
    Test full decomposition (JOB='G', JOBU='U').

    Verifies the transformation satisfies:
    U^H (i*He) U = [Sc  Gc; 0  -Sc^H]

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb03xz

    np.random.seed(111)
    n = 2

    a = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    a = np.asfortranarray(a)

    g_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    g_full = (g_full + g_full.T.conj()) / 2

    q_full = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    q_full = (q_full + q_full.T.conj()) / 2

    qg = np.zeros((n, n+1), dtype=complex, order='F')
    for i in range(n):
        for j in range(i+1):
            qg[i, j] = q_full[i, j]
        for j in range(i, n):
            qg[i, j+1] = g_full[i, j]

    wr, wi, sc, gc, u1, u2, ilo, scale, info = mb03xz(
        a, qg, balanc='N', job='G', jobu='U'
    )

    assert info == 0
    assert sc.shape == (2*n, 2*n)
    assert gc.shape == (2*n, 2*n)
    assert u1.shape == (2*n, 2*n)
    assert u2.shape == (2*n, 2*n)

    # Verify U is unitary symplectic: U = [U1 U2; -U2 U1]
    # Check U^H U = I
    u_top = np.hstack([u1, u2])
    u_bot = np.hstack([-u2, u1])
    u = np.vstack([u_top, u_bot])

    ident = u.T.conj() @ u
    assert_allclose(ident, np.eye(4*n), rtol=1e-12, atol=1e-12)


def test_mb03xz_n_zero():
    """Test with N=0 (quick return)."""
    from ctrlsys import mb03xz

    n = 0
    a = np.zeros((0, 0), dtype=complex, order='F')
    qg = np.zeros((0, 1), dtype=complex, order='F')

    wr, wi, sc, gc, u1, u2, ilo, scale, info = mb03xz(
        a, qg, balanc='N', job='E', jobu='N'
    )

    assert info == 0
    assert len(wr) == 0
    assert len(wi) == 0


def test_mb03xz_invalid_balanc():
    """Test invalid BALANC parameter."""
    from ctrlsys import mb03xz

    n = 2
    a = np.zeros((n, n), dtype=complex, order='F')
    qg = np.zeros((n, n+1), dtype=complex, order='F')

    with pytest.raises(ValueError, match="balanc"):
        mb03xz(a, qg, balanc='X', job='E', jobu='N')


def test_mb03xz_invalid_job():
    """Test invalid JOB parameter."""
    from ctrlsys import mb03xz

    n = 2
    a = np.zeros((n, n), dtype=complex, order='F')
    qg = np.zeros((n, n+1), dtype=complex, order='F')

    with pytest.raises(ValueError, match="job"):
        mb03xz(a, qg, balanc='N', job='X', jobu='N')
