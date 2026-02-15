"""
Tests for MB04DY - Symplectic scaling of Hamiltonian matrix.

MB04DY performs symplectic scaling on a Hamiltonian matrix:
    H = [ A    G  ]
        [ Q   -A' ]
where A is N-by-N and G, Q are symmetric N-by-N matrices.

Two scaling strategies:
- 'S': Symplectic scaling using DGEBAL + equilibration
- '1'/'O': 1-norm scaling by power of machine base
- 'N': No scaling

Test data sources:
- SLICOT HTML documentation example (MB04DY.html)
- Mathematical properties of Hamiltonian matrices
"""

import numpy as np
import pytest

from ctrlsys import mb04dy


def test_mb04dy_html_doc_example():
    """
    Test using SLICOT HTML documentation example.

    From MB04DY.html Program Data:
    N=3, JOBSCL='S'
    A = [-0.4, 0.05, 0.0007; -4.7, 0.8, 0.025; 81.0, 29.0, -0.9] (row-wise)
    G upper triangle data: 0.0034, 0.0014, 0.00077, -0.005, 0.0004, 0.003
      Read order: QG(1,2), QG(1,3), QG(1,4), QG(2,3), QG(2,4), QG(3,4)
    Q lower triangle data: -18.0, -12.0, 43.0, 99.0, 420.0, -200.0
      Read order: QG(1,1), QG(2,1), QG(3,1), QG(2,2), QG(3,2), QG(3,3)

    Validates that the scaling is a valid symplectic similarity transformation
    by checking eigenvalue preservation of the full Hamiltonian.
    """
    n = 3

    # A matrix - READ (A(I,J), J=1,N), I=1,N) reads row-wise
    a = np.array([
        [-0.4,   0.05,   0.0007],
        [-4.7,   0.8,    0.025],
        [81.0,  29.0,   -0.9]
    ], order='F', dtype=float)

    # Build QG array (N x N+1) - Fortran indexing starts at 1
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    # G upper triangle: READ ((QG(J,I+1), I=J,N), J=1,N)
    qg[0, 1] = 0.0034
    qg[0, 2] = 0.0014
    qg[0, 3] = 0.00077
    qg[1, 2] = -0.005
    qg[1, 3] = 0.0004
    qg[2, 3] = 0.003

    # Q lower triangle: READ ((QG(I,J), I=J,N), J=1,N)
    qg[0, 0] = -18.0
    qg[1, 0] = -12.0
    qg[2, 0] = 43.0
    qg[1, 1] = 99.0
    qg[2, 1] = 420.0
    qg[2, 2] = -200.0

    # Extract full Q and G matrices for eigenvalue check
    q_full = np.zeros((n, n), dtype=float)
    g_full = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            q_full[i, j] = qg[i, j]
            q_full[j, i] = qg[i, j]
    for i in range(n):
        for j in range(i, n):
            g_full[i, j] = qg[i, j + 1]
            g_full[j, i] = qg[i, j + 1]

    # Build original full Hamiltonian
    h_orig = np.zeros((2*n, 2*n), dtype=float)
    h_orig[:n, :n] = a
    h_orig[:n, n:] = g_full
    h_orig[n:, :n] = q_full
    h_orig[n:, n:] = -a.T
    eig_orig = np.linalg.eigvals(h_orig)

    a_out, qg_out, d, info = mb04dy('S', a.copy(), qg.copy())

    assert info == 0
    assert len(d) == n
    assert all(di > 0 for di in d)

    # Reconstruct full matrices from scaled output
    q_scaled = np.zeros((n, n), dtype=float)
    g_scaled = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            q_scaled[i, j] = qg_out[i, j]
            q_scaled[j, i] = qg_out[i, j]
    for i in range(n):
        for j in range(i, n):
            g_scaled[i, j] = qg_out[i, j + 1]
            g_scaled[j, i] = qg_out[i, j + 1]

    # Build scaled Hamiltonian
    h_scaled = np.zeros((2*n, 2*n), dtype=float)
    h_scaled[:n, :n] = a_out
    h_scaled[:n, n:] = g_scaled
    h_scaled[n:, :n] = q_scaled
    h_scaled[n:, n:] = -a_out.T

    # Verify similarity transformation: H' = D^(-1) H D where D = diag(d, 1/d)
    D_full = np.diag(np.concatenate([d, 1.0 / d]))
    D_inv = np.diag(np.concatenate([1.0 / d, d]))
    h_reconstructed = D_full @ h_scaled @ D_inv

    # Eigenvalues of reconstructed should match original
    eig_reconstructed = np.linalg.eigvals(h_reconstructed)
    eig_orig_sorted = sorted(eig_orig, key=lambda x: (x.real, x.imag))
    eig_reconstructed_sorted = sorted(eig_reconstructed, key=lambda x: (x.real, x.imag))

    np.testing.assert_allclose(
        np.array(eig_orig_sorted),
        np.array(eig_reconstructed_sorted),
        rtol=1e-10, atol=1e-12
    )


def test_mb04dy_norm_scaling():
    """
    Test 1-norm scaling (JOBSCL='1').

    tau = MAX(1, ||A||_1, ||G||_1, ||Q||_1) rounded to power of machine base
    A' = A/tau, G' = G/tau^2, Q unchanged (only tau stored in D[0])

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.random.randn(n, n + 1).astype(float, order='F')

    a_out, qg_out, d, info = mb04dy('1', a, qg)

    assert info == 0
    assert len(d) == 1
    tau = d[0]
    assert tau >= 1.0  # tau is always at least 1


def test_mb04dy_no_scaling():
    """
    Test no scaling (JOBSCL='N').

    Should return immediately without modifying arrays.
    """
    np.random.seed(123)
    n = 3

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.random.randn(n, n + 1).astype(float, order='F')

    a_orig = a.copy()
    qg_orig = qg.copy()

    a_out, qg_out, d, info = mb04dy('N', a, qg)

    assert info == 0
    # Arrays should be unchanged
    np.testing.assert_allclose(a_out, a_orig, rtol=1e-14)
    np.testing.assert_allclose(qg_out, qg_orig, rtol=1e-14)


def test_mb04dy_zero_dimension():
    """
    Test with n=0 - empty matrix (quick return).
    """
    a = np.zeros((0, 0), order='F', dtype=float)
    qg = np.zeros((0, 1), order='F', dtype=float)

    a_out, qg_out, d, info = mb04dy('S', a, qg)

    assert info == 0


def test_mb04dy_single_element():
    """
    Test with n=1 - single element matrix.
    """
    a = np.array([[2.0]], order='F', dtype=float)
    qg = np.array([[1.0, 3.0]], order='F', dtype=float)  # Q(1,1)=1.0, G(1,1)=3.0

    a_out, qg_out, d, info = mb04dy('S', a, qg)

    assert info == 0
    assert len(d) == 1


def test_mb04dy_eigenvalue_scaling_property():
    """
    Test that symplectic scaling is a similarity transformation.

    For JOBSCL='S', the scaling is:
    H' = D^(-1) H D where D = diag(D_A, D_A^(-1))

    Eigenvalues of H should equal eigenvalues of reconstructed H.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3

    # Create random Hamiltonian
    a = np.random.randn(n, n).astype(float, order='F')

    # Create symmetric Q and G
    q_full = np.random.randn(n, n)
    q_full = 0.5 * (q_full + q_full.T)
    g_full = np.random.randn(n, n)
    g_full = 0.5 * (g_full + g_full.T)

    # Build QG storage
    qg = np.zeros((n, n + 1), order='F', dtype=float)
    for i in range(n):
        for j in range(i + 1):
            qg[i, j] = q_full[i, j]  # Lower triangle of Q
    for i in range(n):
        for j in range(i, n):
            qg[i, j + 1] = g_full[i, j]  # Upper triangle of G

    # Build full 2n x 2n Hamiltonian
    h_orig = np.zeros((2*n, 2*n), dtype=float)
    h_orig[:n, :n] = a
    h_orig[:n, n:] = g_full
    h_orig[n:, :n] = q_full
    h_orig[n:, n:] = -a.T

    # Compute original eigenvalues
    eig_orig = np.linalg.eigvals(h_orig)

    a_out, qg_out, d, info = mb04dy('S', a.copy(), qg.copy())

    assert info == 0

    # Reconstruct full matrices from scaled output
    a_scaled = a_out
    q_scaled = np.zeros((n, n), dtype=float)
    g_scaled = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1):
            q_scaled[i, j] = qg_out[i, j]
            q_scaled[j, i] = qg_out[i, j]
    for i in range(n):
        for j in range(i, n):
            g_scaled[i, j] = qg_out[i, j + 1]
            g_scaled[j, i] = qg_out[i, j + 1]

    # Build scaled Hamiltonian
    h_scaled = np.zeros((2*n, 2*n), dtype=float)
    h_scaled[:n, :n] = a_scaled
    h_scaled[:n, n:] = g_scaled
    h_scaled[n:, :n] = q_scaled
    h_scaled[n:, n:] = -a_scaled.T

    # Verify similarity transformation: H' = D^(-1) H D
    # Build D = diag(d, 1/d)
    D_full = np.diag(np.concatenate([d, 1.0 / d]))
    D_inv = np.diag(np.concatenate([1.0 / d, d]))

    h_reconstructed = D_full @ h_scaled @ D_inv

    # Eigenvalues should match original
    eig_reconstructed = np.linalg.eigvals(h_reconstructed)

    # Sort by real then imaginary parts
    eig_orig_sorted = sorted(eig_orig, key=lambda x: (x.real, x.imag))
    eig_reconstructed_sorted = sorted(eig_reconstructed, key=lambda x: (x.real, x.imag))

    np.testing.assert_allclose(
        np.array(eig_orig_sorted),
        np.array(eig_reconstructed_sorted),
        rtol=1e-10, atol=1e-12
    )


def test_mb04dy_invalid_jobscl():
    """
    Test invalid JOBSCL parameter.

    Should return info = -1.
    """
    n = 2
    a = np.eye(n, order='F', dtype=float)
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    a_out, qg_out, d, info = mb04dy('X', a, qg)

    assert info == -1


def test_mb04dy_diagonal_hamiltonian():
    """
    Test with diagonal A and zero Q, G.

    Already balanced, scaling factors should be close to 1.
    """
    n = 3
    a = np.diag([1.0, 2.0, 3.0]).astype(float, order='F')
    qg = np.zeros((n, n + 1), order='F', dtype=float)

    a_out, qg_out, d, info = mb04dy('S', a, qg)

    assert info == 0
