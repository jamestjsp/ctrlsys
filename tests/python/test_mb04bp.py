"""
Tests for MB04BP: Eigenvalues of skew-Hamiltonian/Hamiltonian pencil (block algorithm)

MB04BP is a block algorithm variant of MB04BD that applies transformations on
panels of columns for better performance on large matrices.

Note: For small N (<= 250), MB04BP delegates to MB04BD. Since MB04BD has known
numerical issues (see test_mb04bd.py), we skip numerical validation tests here
and focus on interface and structural tests.
"""

import numpy as np
import pytest


def test_mb04bp_basic():
    """
    Basic test using MB04BD example from HTML docs.

    Tests JOB='T' (Schur form), COMPQ1='I', COMPQ2='I' for N=8.
    Skipped because MB04BD has numerical issues.
    """
    from ctrlsys import mb04bp

    a = np.array([
        [3.1472,  1.3236,  4.5751,  4.5717],
        [4.0579, -4.0246,  4.6489, -0.1462],
        [-3.7301, -2.2150, -3.4239,  3.0028],
        [4.1338,  0.4688,  4.7059, -3.5811]
    ], dtype=float, order='F')

    de = np.array([
        [0.0000,  0.0000, -1.5510, -4.5974, -2.5127],
        [3.5071,  0.0000,  0.0000,  1.5961,  2.4490],
        [-3.1428,  2.5648,  0.0000,  0.0000, -0.0596],
        [3.0340,  2.4892, -1.1604,  0.0000,  0.0000]
    ], dtype=float, order='F')

    c1 = np.array([
        [0.6882, -3.3782, -3.3435,  1.8921],
        [-0.3061,  2.9428,  1.0198,  2.4815],
        [-4.8810, -1.8878, -2.3703, -0.4946],
        [-1.6288,  0.2853,  1.5408, -4.1618]
    ], dtype=float, order='F')

    vw = np.array([
        [-2.4013, -2.7102,  0.3834, -3.9335,  3.1730],
        [-3.1815, -2.3620,  4.9613,  4.6190,  3.6869],
        [3.6929,  0.7970,  0.4986, -4.9537, -4.1556],
        [3.5303,  1.2206, -1.4905,  0.1325, -1.0022]
    ], dtype=float, order='F')

    result = mb04bp('T', 'I', 'I', a, de, c1, vw)
    a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2, alphar, alphai, beta, info = result

    assert info == 0

    m = 4
    n = 8

    assert a_out.shape == (m, m)
    assert b.shape == (m, m)
    assert c1_out.shape == (m, m)
    assert c2.shape == (m, m)
    assert q1.shape == (n, n)
    assert q2.shape == (n, n)
    assert alphar.shape == (m,)
    assert alphai.shape == (m,)
    assert beta.shape == (m,)

    alphar_expected = np.array([0.8314, -0.8314, 0.8131, 0.0000], dtype=float)
    alphai_expected = np.array([0.4372,  0.4372, 0.0000, 0.9164], dtype=float)
    beta_expected = np.array([0.7071,  0.7071, 1.4142, 2.8284], dtype=float)

    np.testing.assert_allclose(alphar, alphar_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(alphai, alphai_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(beta, beta_expected, rtol=1e-3, atol=1e-4)

    # A, B, C1 should be upper triangular (structural property)
    np.testing.assert_allclose(np.tril(a_out, -1), np.zeros((m, m)), atol=1e-10)
    np.testing.assert_allclose(np.tril(b, -1), np.zeros((m, m)), atol=1e-10)
    np.testing.assert_allclose(np.tril(c1_out, -1), np.zeros((m, m)), atol=1e-10)

    # Check diagonal absolute values match expected
    a_diag_expected = np.array([4.7460, 6.4157, 7.4626, 8.8702])
    b_diag_expected = np.array([6.4937, 4.6929, 9.1725, 7.2106])
    c1_diag_expected = np.array([6.9525, 8.5009, 4.6650, 1.5124])
    np.testing.assert_allclose(np.abs(np.diag(a_out)), a_diag_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.abs(np.diag(b)), b_diag_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.abs(np.diag(c1_out)), c1_diag_expected, rtol=1e-3, atol=1e-4)


def test_mb04bp_eigenvalues_only():
    """
    Test JOB='E' mode (eigenvalues only, no transformation matrices).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04bp

    np.random.seed(42)
    m = 3
    n = 2 * m

    a = np.random.randn(m, m).astype(float, order='F')
    de = np.random.randn(m, m + 1).astype(float, order='F')
    c1 = np.random.randn(m, m).astype(float, order='F')
    vw = np.random.randn(m, m + 1).astype(float, order='F')

    result = mb04bp('E', 'N', 'N', a, de, c1, vw)
    a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2, alphar, alphai, beta, info = result

    assert info == 0

    assert alphar.shape == (m,)
    assert alphai.shape == (m,)
    assert beta.shape == (m,)
    assert b.shape == (m, m)
    assert f.shape == (m, m)
    assert c2.shape == (m, m)

    assert q1 is None
    assert q2 is None


def test_mb04bp_orthogonality():
    """
    Validate Q1 and Q2 orthogonality: Q'Q = I.

    Random seed: 123 (for reproducibility)
    Skipped because MB04BD has numerical issues.
    """
    from ctrlsys import mb04bp

    np.random.seed(123)
    m = 4
    n = 2 * m

    a = np.random.randn(m, m).astype(float, order='F')
    de = np.zeros((m, m + 1), dtype=float, order='F')
    for i in range(1, m):
        de[i, 0:i] = np.random.randn(i)
    for j in range(2, m + 1):
        de[0:j-1, j] = np.random.randn(j - 1)
    c1 = np.random.randn(m, m).astype(float, order='F')
    vw = np.random.randn(m, m + 1).astype(float, order='F')

    result = mb04bp('T', 'I', 'I', a, de, c1, vw)
    a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2, alphar, alphai, beta, info = result

    assert info == 0 or info == 3

    q1_orth = q1.T @ q1
    q2_orth = q2.T @ q2
    eye_n = np.eye(n)

    np.testing.assert_allclose(q1_orth, eye_n, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(q2_orth, eye_n, rtol=1e-13, atol=1e-14)


def test_mb04bp_empty():
    """Test N=0 case returns successfully."""
    from ctrlsys import mb04bp

    a = np.zeros((0, 0), dtype=float, order='F')
    de = np.zeros((0, 1), dtype=float, order='F')
    c1 = np.zeros((0, 0), dtype=float, order='F')
    vw = np.zeros((0, 1), dtype=float, order='F')

    result = mb04bp('E', 'N', 'N', a, de, c1, vw)
    a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2, alphar, alphai, beta, info = result

    assert info == 0
    assert alphar.shape == (0,)


def test_mb04bp_matches_mb04bd():
    """
    Verify MB04BP produces same results as MB04BD for small N.

    MB04BP calls MB04BD directly for N <= 250 when INFO=0 on entry.
    This test verifies the delegation is correct (same output from both).
    """
    from ctrlsys import mb04bd, mb04bp

    np.random.seed(456)
    m = 3
    n = 2 * m

    a = np.random.randn(m, m).astype(float, order='F')
    de = np.random.randn(m, m + 1).astype(float, order='F')
    c1 = np.random.randn(m, m).astype(float, order='F')
    vw = np.random.randn(m, m + 1).astype(float, order='F')

    a_bd = a.copy(order='F')
    de_bd = de.copy(order='F')
    c1_bd = c1.copy(order='F')
    vw_bd = vw.copy(order='F')

    a_bp = a.copy(order='F')
    de_bp = de.copy(order='F')
    c1_bp = c1.copy(order='F')
    vw_bp = vw.copy(order='F')

    result_bd = mb04bd('E', 'N', 'N', a_bd, de_bd, c1_bd, vw_bd)
    result_bp = mb04bp('E', 'N', 'N', a_bp, de_bp, c1_bp, vw_bp)

    _, _, _, _, q1_bd, q2_bd, b_bd, f_bd, c2_bd, alphar_bd, alphai_bd, beta_bd, info_bd = result_bd
    _, _, _, _, q1_bp, q2_bp, b_bp, f_bp, c2_bp, alphar_bp, alphai_bp, beta_bp, info_bp = result_bp

    assert info_bd == info_bp or (info_bd in [0, 3] and info_bp in [0, 3])

    np.testing.assert_allclose(alphar_bp, alphar_bd, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(alphai_bp, alphai_bd, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(beta_bp, beta_bd, rtol=1e-13, atol=1e-14)


def test_mb04bp_upper_triangular_output():
    """
    Validate A, B, C1 are upper triangular after JOB='T'.

    Random seed: 789 (for reproducibility)
    Skipped because MB04BD has numerical issues.
    """
    from ctrlsys import mb04bp

    np.random.seed(789)
    m = 4

    a = np.random.randn(m, m).astype(float, order='F')
    de = np.random.randn(m, m + 1).astype(float, order='F')
    c1 = np.random.randn(m, m).astype(float, order='F')
    vw = np.random.randn(m, m + 1).astype(float, order='F')

    result = mb04bp('T', 'I', 'I', a, de, c1, vw)
    a_out, de_out, c1_out, vw_out, q1, q2, b, f, c2, alphar, alphai, beta, info = result

    assert info == 0 or info == 3

    for i in range(1, m):
        for j in range(i):
            assert abs(a_out[i, j]) < 1e-10, f"A[{i},{j}] = {a_out[i, j]} should be zero"
            assert abs(b[i, j]) < 1e-10, f"B[{i},{j}] = {b[i, j]} should be zero"
            assert abs(c1_out[i, j]) < 1e-10, f"C1[{i},{j}] = {c1_out[i, j]} should be zero"
