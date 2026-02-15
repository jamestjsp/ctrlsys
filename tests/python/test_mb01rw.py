"""
Tests for MB01RW: Symmetric matrix transformation.

Computes A := op(Z)*A*op(Z)' where A is symmetric.
BLAS 2 version of MB01RD.
"""

import numpy as np
import pytest


def test_mb01rw_basic_upper_notrans():
    """
    Validate basic functionality with UPLO='U', TRANS='N'.

    A := Z*A*Z'
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb01rw

    np.random.seed(42)
    m, n = 3, 4

    a = np.array([
        [4.0, 2.0, 1.0, 0.5],
        [0.0, 3.0, 2.0, 1.0],
        [0.0, 0.0, 5.0, 0.5],
        [0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    z = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    a_full = np.array([
        [4.0, 2.0, 1.0, 0.5],
        [2.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 5.0, 0.5],
        [0.5, 1.0, 0.5, 2.0]
    ], order='F', dtype=float)

    expected = z @ a_full @ z.T

    a_out, info = mb01rw('U', 'N', m, n, a, z)

    assert info == 0
    np.testing.assert_allclose(np.triu(a_out[:m, :m]), np.triu(expected), rtol=1e-14)


def test_mb01rw_basic_lower_notrans():
    """
    Validate basic functionality with UPLO='L', TRANS='N'.

    A := Z*A*Z'
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb01rw

    np.random.seed(123)
    m, n = 3, 4

    a = np.array([
        [4.0, 0.0, 0.0, 0.0],
        [2.0, 3.0, 0.0, 0.0],
        [1.0, 2.0, 5.0, 0.0],
        [0.5, 1.0, 0.5, 2.0]
    ], order='F', dtype=float)

    z = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    a_full = np.array([
        [4.0, 2.0, 1.0, 0.5],
        [2.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 5.0, 0.5],
        [0.5, 1.0, 0.5, 2.0]
    ], order='F', dtype=float)

    expected = z @ a_full @ z.T

    a_out, info = mb01rw('L', 'N', m, n, a, z)

    assert info == 0
    np.testing.assert_allclose(np.tril(a_out[:m, :m]), np.tril(expected), rtol=1e-14)


def test_mb01rw_transpose():
    """
    Validate TRANS='T' mode: A := Z'*A*Z.

    For TRANS='T': Z is N-by-M, op(Z)=Z', result is M-by-M.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb01rw

    np.random.seed(456)
    m, n = 4, 3

    a_input = np.zeros((max(m, n), max(m, n)), order='F', dtype=float)
    a_input[:n, :n] = np.array([
        [4.0, 2.0, 1.0],
        [0.0, 3.0, 2.0],
        [0.0, 0.0, 5.0]
    ], dtype=float)

    z = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    a_full = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 3.0, 2.0],
        [1.0, 2.0, 5.0]
    ], order='F', dtype=float)

    expected = z.T @ a_full @ z

    a_out, info = mb01rw('U', 'T', m, n, a_input, z)

    assert info == 0
    np.testing.assert_allclose(np.triu(a_out[:m, :m]), np.triu(expected), rtol=1e-14)


def test_mb01rw_property_symmetry():
    """
    Validate that result is symmetric (upper part equals lower part transpose).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb01rw

    np.random.seed(789)
    m, n = 3, 4

    a = np.array([
        [4.0, 2.0, 1.0, 0.5],
        [0.0, 3.0, 2.0, 1.0],
        [0.0, 0.0, 5.0, 0.5],
        [0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    z = np.random.randn(m, n).astype(float, order='F')

    a_out, info = mb01rw('U', 'N', m, n, a, z)

    assert info == 0
    result_full = np.triu(a_out[:m, :m]) + np.triu(a_out[:m, :m], 1).T
    np.testing.assert_allclose(result_full, result_full.T, rtol=1e-14)


def test_mb01rw_property_identity_transform():
    """
    Validate mathematical property: With Z=I, result equals A.

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import mb01rw

    np.random.seed(888)
    n = 4
    m = n

    a = np.array([
        [4.0, 2.0, 1.0, 0.5],
        [0.0, 3.0, 2.0, 1.0],
        [0.0, 0.0, 5.0, 0.5],
        [0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    z = np.eye(n, order='F', dtype=float)

    a_out, info = mb01rw('U', 'N', m, n, a.copy(), z)

    assert info == 0
    np.testing.assert_allclose(np.triu(a_out), np.triu(a), rtol=1e-14)


def test_mb01rw_property_congruence():
    """
    Validate congruence transformation: A := Z*A*Z'.

    For square Z, we verify the formula holds.

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb01rw

    np.random.seed(999)
    n = 4
    m = n

    a = np.array([
        [4.0, 2.0, 1.0, 0.5],
        [0.0, 3.0, 2.0, 1.0],
        [0.0, 0.0, 5.0, 0.5],
        [0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    a_full = np.array([
        [4.0, 2.0, 1.0, 0.5],
        [2.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 5.0, 0.5],
        [0.5, 1.0, 0.5, 2.0]
    ], order='F', dtype=float)

    z = np.eye(n, order='F', dtype=float) * 2.0

    expected = z @ a_full @ z.T

    a_input = a.copy(order='F')
    a_out, info = mb01rw('U', 'N', m, n, a_input, z)

    assert info == 0

    np.testing.assert_allclose(np.triu(a_out), np.triu(expected), rtol=1e-14)


def test_mb01rw_error_invalid_uplo():
    """Test error handling: invalid UPLO parameter."""
    from ctrlsys import mb01rw

    m, n = 3, 4
    a = np.eye(max(m, n), order='F', dtype=float)
    z = np.eye(m, n, order='F', dtype=float)

    with pytest.raises(ValueError, match="Parameter 1"):
        mb01rw('X', 'N', m, n, a, z)


def test_mb01rw_error_invalid_trans():
    """Test error handling: invalid TRANS parameter."""
    from ctrlsys import mb01rw

    m, n = 3, 4
    a = np.eye(max(m, n), order='F', dtype=float)
    z = np.eye(m, n, order='F', dtype=float)

    with pytest.raises(ValueError, match="Parameter 2"):
        mb01rw('U', 'X', m, n, a, z)


def test_mb01rw_error_m_negative():
    """Test error handling: M < 0."""
    from ctrlsys import mb01rw

    a = np.eye(4, order='F', dtype=float)
    z = np.eye(3, 4, order='F', dtype=float)

    with pytest.raises(ValueError, match="Parameter 3"):
        mb01rw('U', 'N', -1, 4, a, z)


def test_mb01rw_error_n_negative():
    """Test error handling: N < 0."""
    from ctrlsys import mb01rw

    a = np.eye(4, order='F', dtype=float)
    z = np.eye(3, 4, order='F', dtype=float)

    with pytest.raises(ValueError, match="Parameter 4"):
        mb01rw('U', 'N', 3, -1, a, z)


def test_mb01rw_edge_case_m_zero():
    """Test edge case: M=0."""
    from ctrlsys import mb01rw

    m, n = 0, 4
    a = np.eye(n, order='F', dtype=float)
    z = np.zeros((1, n), order='F', dtype=float)

    a_out, info = mb01rw('U', 'N', m, n, a, z)
    assert info == 0


def test_mb01rw_edge_case_n_zero():
    """Test edge case: N=0."""
    from ctrlsys import mb01rw

    m, n = 3, 0
    a = np.eye(m, order='F', dtype=float)
    z = np.zeros((m, 1), order='F', dtype=float)

    a_out, info = mb01rw('U', 'N', m, n, a, z)
    assert info == 0
