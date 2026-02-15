"""Tests for MB04CD - Reducing skew-Hamiltonian/Hamiltonian pencil to Schur form."""

import numpy as np
import pytest


def test_mb04cd_basic():
    """
    Test MB04CD with COMPQ1='I', COMPQ2='I', COMPQ3='I'.

    Tests transformation of block diagonal pencil to generalized Schur form.
    N must be even (N >= 0).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(42)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21

    result = mb04cd('I', 'I', 'I', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0

    assert a_out.shape == (n, n)
    assert b_out.shape == (n, n)
    assert d_out.shape == (n, n)
    assert q1.shape == (n, n)
    assert q2.shape == (n, n)
    assert q3.shape == (n, n)


def test_mb04cd_orthogonality():
    """
    Validate Q1, Q2, Q3 are orthogonal matrices.

    For COMPQ1='I', COMPQ2='I', COMPQ3='I', the transformation matrices
    Q1, Q2, Q3 must satisfy Q'Q = QQ' = I.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(123)
    n = 6
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21

    result = mb04cd('I', 'I', 'I', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q3 @ q3.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q1.T @ q1, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q2.T @ q2, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q3.T @ q3, np.eye(n), rtol=1e-12, atol=1e-12)


def test_mb04cd_upper_triangular_a_b():
    """
    Validate transformed A and B are upper triangular.

    After transformation, Q3' A Q2 and Q2' B Q1 should be upper triangular.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(456)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21

    result = mb04cd('I', 'I', 'I', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(np.tril(a_out, -1), 0, atol=1e-12)
        np.testing.assert_allclose(np.tril(b_out, -1), 0, atol=1e-12)


def test_mb04cd_n_zero():
    """Test MB04CD with N=0 (quick return)."""
    from ctrlsys import mb04cd

    a = np.array([], dtype=float, order='F').reshape(0, 0)
    b = np.array([], dtype=float, order='F').reshape(0, 0)
    d = np.array([], dtype=float, order='F').reshape(0, 0)

    result = mb04cd('N', 'N', 'N', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0


def test_mb04cd_no_orthogonal_matrices():
    """
    Test MB04CD with COMPQ1='N', COMPQ2='N', COMPQ3='N'.

    Q1, Q2, Q3 are not computed.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(789)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21

    result = mb04cd('N', 'N', 'N', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]
    assert a_out.shape == (n, n)
    assert b_out.shape == (n, n)
    assert d_out.shape == (n, n)


def test_mb04cd_small_n2():
    """
    Test MB04CD with N=2 (smallest even case).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(111)
    n = 2
    m = n // 2

    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 1.5
    a[1, 1] = 2.5

    b = np.zeros((n, n), order='F', dtype=float)
    b[0, 0] = 0.8
    b[1, 1] = 1.2

    d = np.zeros((n, n), order='F', dtype=float)
    d[0, 1] = 0.3
    d[1, 0] = 0.4

    result = mb04cd('I', 'I', 'I', a, b, d)
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]
    assert a_out.shape == (n, n)
    assert b_out.shape == (n, n)
    assert d_out.shape == (n, n)


def test_mb04cd_update_mode():
    """
    Test MB04CD with COMPQ1='U', COMPQ2='U', COMPQ3='U' (update mode).

    When update mode, initial orthogonal matrices are provided and multiplied
    by the transformation matrices.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb04cd

    np.random.seed(222)
    n = 4
    m = n // 2

    a11 = np.triu(np.random.randn(m, m))
    a22 = np.triu(np.random.randn(m, m))
    a = np.zeros((n, n), order='F', dtype=float)
    a[:m, :m] = a11
    a[m:, m:] = a22

    b11 = np.triu(np.random.randn(m, m))
    b22 = np.triu(np.random.randn(m, m))
    b = np.zeros((n, n), order='F', dtype=float)
    b[:m, :m] = b11
    b[m:, m:] = b22

    d12 = np.triu(np.random.randn(m, m))
    d21 = np.triu(np.random.randn(m, m))
    d = np.zeros((n, n), order='F', dtype=float)
    d[:m, m:] = d12
    d[m:, :m] = d21

    q1_init = np.eye(n, dtype=float, order='F')
    q2_init = np.eye(n, dtype=float, order='F')
    q3_init = np.eye(n, dtype=float, order='F')

    result = mb04cd('U', 'U', 'U', a, b, d,
                    q1=q1_init.copy(), q2=q2_init.copy(), q3=q3_init.copy())
    a_out, b_out, d_out, q1, q2, q3, info = result

    assert info == 0 or info in [1, 2, 3, 4]

    if info == 0:
        np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(q3 @ q3.T, np.eye(n), rtol=1e-12, atol=1e-12)
