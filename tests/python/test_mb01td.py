"""
Tests for MB01TD: Product of upper quasi-triangular matrices A*B.

MB01TD computes A*B where A and B are upper quasi-triangular matrices
(block upper triangular with 1x1 or 2x2 diagonal blocks) with the same structure.
Result is returned in B.
"""
import numpy as np
import pytest
from ctrlsys import mb01td


def test_mb01td_html_example():
    """
    Test using HTML documentation example.

    N=5, A and B upper quasi-triangular with 2x2 block at (4,4)-(5,5).
    """
    a = np.array([
        [1., 2., 6., 3., 5.],
        [-2., -1., -1., 0., -2.],
        [0., 0., 1., 5., 1.],
        [0., 0., 0., 0., -4.],
        [0., 0., 0., 20., 4.]
    ], order='F', dtype=float)

    b = np.array([
        [5., 5., 1., 5., 1.],
        [-2., 1., 3., 0., -4.],
        [0., 0., 4., 20., 4.],
        [0., 0., 0., 3., 5.],
        [0., 0., 0., 1., -2.]
    ], order='F', dtype=float)

    expected = np.array([
        [1., 7., 31., 139., 22.],
        [-8., -11., -9., -32., 2.],
        [0., 0., 4., 36., 27.],
        [0., 0., 0., -4., 8.],
        [0., 0., 0., 64., 92.]
    ], order='F', dtype=float)

    result, info = mb01td(a, b)

    assert info == 0
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_mb01td_n1():
    """
    Test with N=1 (scalar multiplication).
    """
    a = np.array([[3.0]], order='F', dtype=float)
    b = np.array([[4.0]], order='F', dtype=float)

    result, info = mb01td(a, b)

    assert info == 0
    np.testing.assert_allclose(result, np.array([[12.0]], order='F'), rtol=1e-14)


def test_mb01td_n0():
    """
    Test with N=0 (empty matrices).
    """
    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, 0), order='F', dtype=float)

    result, info = mb01td(a, b)

    assert info == 0


def test_mb01td_upper_triangular():
    """
    Test with purely upper triangular matrices (no 2x2 blocks).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    a = np.triu(np.random.randn(n, n)).astype(float, order='F')
    b = np.triu(np.random.randn(n, n)).astype(float, order='F')

    expected = a @ b

    result, info = mb01td(a.copy(order='F'), b.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_mb01td_associativity():
    """
    Property test: (A*B)*C = A*(B*C) for upper triangular.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3

    a = np.triu(np.random.randn(n, n)).astype(float, order='F')
    b = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.triu(np.random.randn(n, n)).astype(float, order='F')

    # (A*B)*C
    ab = a.copy(order='F')
    b1 = b.copy(order='F')
    ab_result, info1 = mb01td(ab, b1)
    assert info1 == 0

    abc1 = ab_result.copy(order='F')
    c1 = c.copy(order='F')
    abc1_result, info2 = mb01td(abc1, c1)
    assert info2 == 0

    # A*(B*C)
    bc = b.copy(order='F')
    c2 = c.copy(order='F')
    bc_result, info3 = mb01td(bc, c2)
    assert info3 == 0

    a2 = a.copy(order='F')
    abc2_result, info4 = mb01td(a2, bc_result.copy(order='F'))
    assert info4 == 0

    np.testing.assert_allclose(abc1_result, abc2_result, rtol=1e-13)


def test_mb01td_structure_mismatch():
    """
    Test error handling when A and B have different structures.

    A has NO 2x2 block (subdiag=0), but B has nonzero subdiag entry.
    This is a structure mismatch.
    """
    a = np.array([
        [1., 2., 3.],
        [0., 4., 5.],
        [0., 0., 6.]
    ], order='F', dtype=float)

    b = np.array([
        [1., 2., 3.],
        [0.5, 4., 5.],
        [0., 0., 6.]
    ], order='F', dtype=float)

    result, info = mb01td(a, b)

    assert info == 1


def test_mb01td_with_2x2_blocks():
    """
    Test with quasi-triangular structure (2x2 diagonal blocks).

    Random seed: 456 (for reproducibility)
    Matrix has 2x2 block at positions (0,1) and (2,3).
    """
    np.random.seed(456)

    a = np.array([
        [1.5, 2.0, 0.5, 0.3],
        [-0.5, 1.8, 0.2, 0.1],
        [0.0, 0.0, 2.0, 1.0],
        [0.0, 0.0, -0.3, 1.5]
    ], order='F', dtype=float)

    b = np.array([
        [2.0, 1.0, 0.4, 0.2],
        [-0.5, 2.5, 0.3, 0.1],
        [0.0, 0.0, 1.5, 0.8],
        [0.0, 0.0, -0.3, 2.0]
    ], order='F', dtype=float)

    expected = a @ b

    result, info = mb01td(a.copy(order='F'), b.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_mb01td_schur_power():
    """
    Test use case: computing power of Schur form matrix.

    A^2 = A*A for upper quasi-triangular A.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4

    a = np.triu(np.random.randn(n, n)).astype(float, order='F')

    expected = a @ a

    result, info = mb01td(a.copy(order='F'), a.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(result, expected, rtol=1e-14)
