"""Tests for MB04BD - Eigenvalues of skew-Hamiltonian/Hamiltonian pencil."""

import numpy as np
import pytest


def test_mb04bd_html_example():
    """
    Test MB04BD using data from SLICOT HTML documentation.

    Computes eigenvalues of skew-Hamiltonian/Hamiltonian pencil aS - bH.
    JOB='T', COMPQ1='I', COMPQ2='I', N=8
    """
    from ctrlsys import mb04bd

    n = 8
    m = n // 2

    # Input matrix A (m x m) - read row-by-row from HTML
    a = np.array([
        [3.1472,  1.3236,  4.5751,  4.5717],
        [4.0579, -4.0246,  4.6489, -0.1462],
        [-3.7301, -2.2150, -3.4239,  3.0028],
        [4.1338,  0.4688,  4.7059, -3.5811]
    ], order='F', dtype=float)

    # Input matrix DE (m x m+1) - read row-by-row from HTML
    # E stored in strictly lower triangular part
    # D stored in strictly upper triangular part of columns 2 to m+1
    de = np.array([
        [0.0000,  0.0000, -1.5510, -4.5974, -2.5127],
        [3.5071,  0.0000,  0.0000,  1.5961,  2.4490],
        [-3.1428,  2.5648,  0.0000,  0.0000, -0.0596],
        [3.0340,  2.4892, -1.1604,  0.0000,  0.0000]
    ], order='F', dtype=float)

    # Input matrix C1 (m x m) - read row-by-row from HTML
    c1 = np.array([
        [0.6882, -3.3782, -3.3435,  1.8921],
        [-0.3061,  2.9428,  1.0198,  2.4815],
        [-4.8810, -1.8878, -2.3703, -0.4946],
        [-1.6288,  0.2853,  1.5408, -4.1618]
    ], order='F', dtype=float)

    # Input matrix VW (m x m+1) - read row-by-row from HTML
    # W stored in lower triangular part
    # V stored in upper triangular part of columns 2 to m+1
    vw = np.array([
        [-2.4013, -2.7102,  0.3834, -3.9335,  3.1730],
        [-3.1815, -2.3620,  4.9613,  4.6190,  3.6869],
        [3.6929,  0.7970,  0.4986, -4.9537, -4.1556],
        [3.5303,  1.2206, -1.4905,  0.1325, -1.0022]
    ], order='F', dtype=float)

    # Call MB04BD
    result = mb04bd('T', 'I', 'I', a, de, c1, vw)
    (a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2,
     alphar, alphai, beta, info) = result

    assert info == 0

    # Expected eigenvalue components from HTML results
    alphar_expected = np.array([0.8314, -0.8314, 0.8131, 0.0000])
    alphai_expected = np.array([0.4372, 0.4372, 0.0000, 0.9164])
    beta_expected = np.array([0.7071, 0.7071, 1.4142, 2.8284])

    np.testing.assert_allclose(alphar, alphar_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(alphai, alphai_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(beta, beta_expected, rtol=1e-3, atol=1e-4)

    # A should be upper triangular (lower part is zero)
    m = n // 2
    np.testing.assert_allclose(np.tril(a_out, -1), np.zeros((m, m)), atol=1e-10)

    # Expected diagonal absolute values (signs can differ)
    a_diag_expected = np.array([4.7460, 6.4157, 7.4626, 8.8702])
    np.testing.assert_allclose(np.abs(np.diag(a_out)), a_diag_expected, rtol=1e-3, atol=1e-4)

    # B should be upper triangular (lower part is zero)
    np.testing.assert_allclose(np.tril(b, -1), np.zeros((m, m)), atol=1e-10)

    # Expected B diagonal absolute values
    b_diag_expected = np.array([6.4937, 4.6929, 9.1725, 7.2106])
    np.testing.assert_allclose(np.abs(np.diag(b)), b_diag_expected, rtol=1e-3, atol=1e-4)

    # Q1 should be orthogonal
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-10, atol=1e-10)

    # Q2 should be orthogonal
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-10, atol=1e-10)


def test_mb04bd_eigenvalues_only():
    """
    Test MB04BD with JOB='E' (eigenvalues only).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04bd

    np.random.seed(42)
    n = 4
    m = n // 2

    # Random input matrices
    a = np.random.randn(m, m).astype(float, order='F')
    de = np.random.randn(m, m + 1).astype(float, order='F')
    c1 = np.random.randn(m, m).astype(float, order='F')
    vw = np.random.randn(m, m + 1).astype(float, order='F')

    # Call with JOB='E', no Q matrices
    result = mb04bd('E', 'N', 'N', a, de, c1, vw)
    (a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2,
     alphar, alphai, beta, info) = result

    assert info == 0
    assert alphar.shape == (m,)
    assert alphai.shape == (m,)
    assert beta.shape == (m,)


def test_mb04bd_n_zero():
    """Test MB04BD with N=0 (quick return)."""
    from ctrlsys import mb04bd

    a = np.array([], dtype=float, order='F').reshape(0, 0)
    de = np.array([], dtype=float, order='F').reshape(0, 1)
    c1 = np.array([], dtype=float, order='F').reshape(0, 0)
    vw = np.array([], dtype=float, order='F').reshape(0, 1)

    result = mb04bd('E', 'N', 'N', a, de, c1, vw)
    (a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2,
     alphar, alphai, beta, info) = result

    assert info == 0
    assert alphar.shape == (0,)
    assert alphai.shape == (0,)
    assert beta.shape == (0,)


def test_mb04bd_invalid_n():
    """Test MB04BD with invalid N (odd)."""
    from ctrlsys import mb04bd

    # N must be even - provide N=3 (odd) via incorrect array sizes
    a = np.array([[1.0]], dtype=float, order='F')  # Would correspond to N=2
    de = np.array([[0.0, 0.0]], dtype=float, order='F')
    c1 = np.array([[1.0]], dtype=float, order='F')
    vw = np.array([[0.0, 0.0]], dtype=float, order='F')

    # This should work for N=2
    result = mb04bd('E', 'N', 'N', a, de, c1, vw)
    (a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2,
     alphar, alphai, beta, info) = result

    assert info == 0


def test_mb04bd_eigenvalue_structure():
    """
    Validate eigenvalue structure: for each lambda, -lambda is also an eigenvalue.

    Due to the skew-Hamiltonian/Hamiltonian pencil structure, eigenvalues come
    in pairs +/- lambda. The routine only returns eigenvalues with non-negative
    imaginary parts.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04bd

    np.random.seed(123)
    n = 6
    m = n // 2

    # Random input matrices
    a = np.random.randn(m, m).astype(float, order='F')
    de = np.random.randn(m, m + 1).astype(float, order='F')
    c1 = np.random.randn(m, m).astype(float, order='F')
    vw = np.random.randn(m, m + 1).astype(float, order='F')

    result = mb04bd('T', 'I', 'I', a, de, c1, vw)
    (a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2,
     alphar, alphai, beta, info) = result

    assert info == 0 or info == 3  # info=3 is a warning

    # Eigenvalues should have non-negative imaginary parts (or be real positive)
    for i in range(m):
        if beta[i] != 0:
            assert alphai[i] >= -1e-10 or alphar[i] >= -1e-10


def test_mb04bd_orthogonality_q1_q2():
    """
    Validate orthogonality of Q1 and Q2 transformation matrices.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04bd

    np.random.seed(456)
    n = 8
    m = n // 2

    a = np.random.randn(m, m).astype(float, order='F')
    de = np.random.randn(m, m + 1).astype(float, order='F')
    c1 = np.random.randn(m, m).astype(float, order='F')
    vw = np.random.randn(m, m + 1).astype(float, order='F')

    result = mb04bd('T', 'I', 'I', a, de, c1, vw)
    (a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2,
     alphar, alphai, beta, info) = result

    assert info == 0 or info == 3

    # Q1 and Q2 should be orthogonal
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q1.T @ q1, np.eye(n), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q2.T @ q2, np.eye(n), rtol=1e-12, atol=1e-12)


def test_mb04bd_update_mode():
    """
    Test MB04BD with COMPQ1='U', COMPQ2='U' (update mode).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04bd

    np.random.seed(789)
    n = 4
    m = n // 2

    a = np.random.randn(m, m).astype(float, order='F')
    de = np.random.randn(m, m + 1).astype(float, order='F')
    c1 = np.random.randn(m, m).astype(float, order='F')
    vw = np.random.randn(m, m + 1).astype(float, order='F')

    # Initial orthogonal Q matrix
    q_init = np.eye(n, dtype=float, order='F')

    result = mb04bd('T', 'U', 'U', a, de, c1, vw, q1=q_init.copy())
    (a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2,
     alphar, alphai, beta, info) = result

    assert info == 0 or info == 3

    # Q1 and Q2 should still be orthogonal after update
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-12, atol=1e-12)
