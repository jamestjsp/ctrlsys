"""
Tests for MB03RW: Solve complex Sylvester equation -AX + XB = C.

MB03RW solves the Sylvester equation -AX + XB = C where A (M-by-M) and B (N-by-N)
are complex upper triangular matrices in Schur form. The routine aborts if any
element of X exceeds PMAX in absolute value.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03rw_basic_sylvester():
    """
    Basic test: solve -AX + XB = C with well-conditioned small matrices.

    Mathematical verification: The solution X must satisfy -A @ X + X @ B = C
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03rw

    np.random.seed(42)
    m, n = 3, 3

    a = np.array([
        [1.0 + 0.5j, 0.2 + 0.1j, 0.1 - 0.2j],
        [0.0 + 0.0j, 2.0 - 0.3j, 0.3 + 0.1j],
        [0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.2j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [4.0 + 0.1j, 0.1 + 0.2j, 0.2 - 0.1j],
        [0.0 + 0.0j, 5.0 - 0.2j, 0.1 + 0.3j],
        [0.0 + 0.0j, 0.0 + 0.0j, 6.0 + 0.4j]
    ], dtype=np.complex128, order='F')

    c_orig = np.array([
        [1.0 + 1.0j, 2.0 - 0.5j, 3.0 + 0.3j],
        [0.5 - 0.2j, 1.5 + 0.8j, 2.5 - 0.1j],
        [0.3 + 0.4j, 0.8 - 0.3j, 1.2 + 0.6j]
    ], dtype=np.complex128, order='F')

    c = c_orig.copy(order='F')
    pmax = 1e6

    x, info = mb03rw(m, n, pmax, a, b, c)

    assert info == 0 or info == 2

    residual = -a @ x + x @ b - c_orig
    assert_allclose(residual, np.zeros((m, n), dtype=np.complex128),
                    atol=1e-10, rtol=1e-10)


def test_mb03rw_scalar_case():
    """
    Test 1x1 case: -A*X + X*B = C => X = C / (B - A).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03rw

    m, n = 1, 1

    a = np.array([[1.0 + 0.5j]], dtype=np.complex128, order='F')
    b = np.array([[3.0 - 0.2j]], dtype=np.complex128, order='F')
    c = np.array([[2.0 + 1.0j]], dtype=np.complex128, order='F')

    c_orig = c.copy()
    pmax = 1e6

    x, info = mb03rw(m, n, pmax, a, b, c)

    assert info == 0 or info == 2

    x_expected = c_orig[0, 0] / (b[0, 0] - a[0, 0])
    assert_allclose(x[0, 0], x_expected, rtol=1e-14)


def test_mb03rw_different_sizes():
    """
    Test with M != N (non-square C matrix).

    Mathematical verification: -A @ X + X @ B = C
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03rw

    np.random.seed(456)
    m, n = 2, 3

    a = np.array([
        [1.0 + 0.2j, 0.3 - 0.1j],
        [0.0 + 0.0j, 2.0 + 0.4j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [3.0 - 0.1j, 0.2 + 0.1j, 0.1 - 0.2j],
        [0.0 + 0.0j, 4.0 + 0.3j, 0.3 - 0.1j],
        [0.0 + 0.0j, 0.0 + 0.0j, 5.0 - 0.2j]
    ], dtype=np.complex128, order='F')

    c_orig = np.array([
        [1.0 + 0.5j, 2.0 - 0.3j, 1.5 + 0.2j],
        [0.8 - 0.1j, 1.2 + 0.4j, 0.9 - 0.5j]
    ], dtype=np.complex128, order='F')

    c = c_orig.copy(order='F')
    pmax = 1e6

    x, info = mb03rw(m, n, pmax, a, b, c)

    assert info == 0 or info == 2

    residual = -a @ x + x @ b - c_orig
    assert_allclose(residual, np.zeros((m, n), dtype=np.complex128),
                    atol=1e-10, rtol=1e-10)


def test_mb03rw_pmax_exceeded():
    """
    Test that INFO=1 when solution element exceeds PMAX.

    Use nearly identical eigenvalues in A and B to cause large X.
    """
    from ctrlsys import mb03rw

    m, n = 2, 2

    a = np.array([
        [1.0 + 0.0j, 0.1 + 0.0j],
        [0.0 + 0.0j, 1.0 + 1e-12j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [1.0 + 0.0j, 0.1 + 0.0j],
        [0.0 + 0.0j, 1.0 + 1e-12j]
    ], dtype=np.complex128, order='F')

    c = np.array([
        [1.0 + 0.0j, 1.0 + 0.0j],
        [1.0 + 0.0j, 1.0 + 0.0j]
    ], dtype=np.complex128, order='F')

    pmax = 1e-3

    x, info = mb03rw(m, n, pmax, a, b, c)

    assert info == 1


def test_mb03rw_m_zero():
    """
    Test quick return when M=0.
    """
    from ctrlsys import mb03rw

    m, n = 0, 3

    a = np.zeros((1, 1), dtype=np.complex128, order='F')
    b = np.array([
        [1.0 + 0.0j, 0.1 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 2.0 + 0.0j, 0.2 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j]
    ], dtype=np.complex128, order='F')
    c = np.zeros((1, n), dtype=np.complex128, order='F')

    pmax = 1e6

    x, info = mb03rw(m, n, pmax, a, b, c)

    assert info == 0


def test_mb03rw_n_zero():
    """
    Test quick return when N=0.
    """
    from ctrlsys import mb03rw

    m, n = 3, 0

    a = np.array([
        [1.0 + 0.0j, 0.1 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 2.0 + 0.0j, 0.2 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j]
    ], dtype=np.complex128, order='F')
    b = np.zeros((1, 1), dtype=np.complex128, order='F')
    c = np.zeros((m, 1), dtype=np.complex128, order='F')

    pmax = 1e6

    x, info = mb03rw(m, n, pmax, a, b, c)

    assert info == 0


def test_mb03rw_well_conditioned():
    """
    Test well-conditioned case with distinct eigenvalues.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03rw

    np.random.seed(456)
    m, n = 3, 3

    a = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
        np.complex128, order='F')
    for i in range(m):
        a[i, i] = (i + 1.0) + (i + 0.5) * 1j

    b = np.triu(np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F')
    for j in range(n):
        b[j, j] = -(j + 2.0) + (j + 1.5) * 1j

    c = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
        np.complex128, order='F')

    pmax = 1e10

    x, info = mb03rw(m, n, pmax, a, b, c)

    assert info == 0


def test_mb03rw_sylvester_identity():
    """
    Property test: Verify -AX + XB = C holds for random systems.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03rw

    np.random.seed(789)
    m, n = 4, 5

    a = np.triu(np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
        np.complex128, order='F')
    np.fill_diagonal(a, np.abs(np.diag(a)) + 1.0)

    b = np.triu(np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F')
    np.fill_diagonal(b, np.abs(np.diag(b)) + m + 1.0)

    c_orig = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
        np.complex128, order='F')
    c = c_orig.copy(order='F')
    pmax = 1e8

    x, info = mb03rw(m, n, pmax, a, b, c)

    assert info == 0 or info == 2

    residual = -a @ x + x @ b - c_orig
    assert_allclose(residual, np.zeros((m, n), dtype=np.complex128),
                    atol=1e-10, rtol=1e-10)
