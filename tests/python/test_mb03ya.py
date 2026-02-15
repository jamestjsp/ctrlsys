"""
Tests for MB03YA - Annihilate subdiagonal entries of Hessenberg matrix A.

MB03YA annihilates one or two entries on the subdiagonal of the Hessenberg
matrix A for dealing with zero elements on the diagonal of the triangular
matrix B. This is an auxiliary routine for MB03XP and MB03YD.

Test data sources:
- Mathematical properties of QR/RQ steps
- Validation against known transformations
"""

import numpy as np
import pytest

from ctrlsys import mb03ya


def test_mb03ya_basic():
    """
    Test basic functionality with small matrix where B has zero diagonal.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4
    ilo, ihi = 1, 4
    iloq, ihiq = 1, 4
    pos = 2

    a = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(i, n):
            a[i, j] = np.random.randn()
        if i < n - 1:
            a[i + 1, i] = np.random.randn()

    b = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(i, n):
            if i != pos - 1 or j != pos - 1:
                b[i, j] = np.random.randn()
    b[pos - 1, pos - 1] = 0.0

    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03ya(
        True, True, True, ilo, ihi, iloq, ihiq, pos, a, b, q, z
    )

    assert info == 0
    if pos > ilo:
        assert abs(a_out[pos - 1, pos - 2]) < 1e-14
    if pos < ihi:
        assert abs(a_out[pos, pos - 1]) < 1e-14


def test_mb03ya_no_accumulate():
    """
    Test without accumulating Q and Z.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3
    ilo, ihi = 1, 3
    iloq, ihiq = 1, 3
    pos = 2

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 5.0]
    ], order='F', dtype=float)

    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03ya(
        True, False, False, ilo, ihi, iloq, ihiq, pos, a, b, q, z
    )

    assert info == 0


def test_mb03ya_boundary_pos():
    """
    Test with pos at the boundary (first position).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3
    ilo, ihi = 1, 3
    iloq, ihiq = 1, 3
    pos = 1

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03ya(
        True, True, True, ilo, ihi, iloq, ihiq, pos, a, b, q, z
    )

    assert info == 0


def test_mb03ya_invalid_pos():
    """Test error handling for invalid pos parameter."""
    n = 3
    ilo, ihi = 1, 3
    iloq, ihiq = 1, 3
    pos = 0

    a = np.eye(n, order='F', dtype=float)
    b = np.eye(n, order='F', dtype=float)
    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03ya(
        True, True, True, ilo, ihi, iloq, ihiq, pos, a, b, q, z
    )

    assert info == -9


def test_mb03ya_orthogonality():
    """
    Test that Q and Z remain orthogonal after transformation.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4
    ilo, ihi = 1, 4
    iloq, ihiq = 1, 4
    pos = 2

    a = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(i, n):
            a[i, j] = np.random.randn()
        if i < n - 1:
            a[i + 1, i] = np.random.randn()

    b = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(i, n):
            if i != pos - 1 or j != pos - 1:
                b[i, j] = np.random.randn()
    b[pos - 1, pos - 1] = 0.0

    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03ya(
        True, True, True, ilo, ihi, iloq, ihiq, pos, a, b, q, z
    )

    assert info == 0
    np.testing.assert_allclose(q_out @ q_out.T, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z_out @ z_out.T, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03ya_upper_triangular_b():
    """
    Test that B remains upper triangular after transformation.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 4
    ilo, ihi = 1, 4
    iloq, ihiq = 1, 4
    pos = 3

    a = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(i, n):
            a[i, j] = np.random.randn()
        if i < n - 1:
            a[i + 1, i] = np.random.randn()

    b = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(i, n):
            if i != pos - 1 or j != pos - 1:
                b[i, j] = np.random.randn()
    b[pos - 1, pos - 1] = 0.0

    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03ya(
        True, True, True, ilo, ihi, iloq, ihiq, pos, a, b, q, z
    )

    assert info == 0
    for i in range(1, n):
        for j in range(i):
            assert abs(b_out[i, j]) < 1e-14, f"B[{i},{j}] = {b_out[i,j]} not zero"
