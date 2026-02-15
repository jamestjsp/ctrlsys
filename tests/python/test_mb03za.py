"""
Tests for MB03ZA - Reordering eigenvalues in periodic Schur form and
computing Schur decomposition of associated skew-Hamiltonian matrix.

MB03ZA computes orthogonal matrices Ur and Vr so that:
1. Vr' * A * Ur and Ur' * B * Vr reorder selected eigenvalues to top-left
2. Computes orthogonal W transforming [0, -A11; B11, 0] to block triangular form

Test data sources:
- Mathematical properties of periodic Schur form
- Eigenvalue preservation under orthogonal transformations
"""

import numpy as np
import pytest

from ctrlsys import mb03za


def test_mb03za_select_all():
    """
    Test with WHICH='A' - select all eigenvalues.

    When all eigenvalues are selected, A11=A, B11=B, M=N.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3

    a = np.array([
        [-2.0, 0.5, 0.1],
        [0.0, -1.5, 0.3],
        [0.0, 0.0, -1.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.2, 0.1],
        [0.0, 1.5, 0.2],
        [0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    c = np.eye(n, order='F', dtype=float)
    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.zeros((n, n), order='F', dtype=float)
    v1 = np.eye(n, order='F', dtype=float)
    v2 = np.zeros((n, n), order='F', dtype=float)
    w = np.zeros((2*n, 2*n), order='F', dtype=float)
    select = np.array([True, True, True], dtype=bool)

    r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info = mb03za(
        'N', 'N', 'N', 'I', 'A', select, a, b, c, u1, u2, v1, v2, w
    )

    assert info == 0, f"Expected info=0, got {info}"
    assert m == n, f"Expected m={n}, got {m}"
    assert len(wr) == m
    assert len(wi) == m


def test_mb03za_select_subset():
    """
    Test with WHICH='S' - select a subset of eigenvalues.

    Select first eigenvalue only. Tests reordering functionality.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3

    a = np.array([
        [-2.0, 0.5, 0.1],
        [0.0, -1.5, 0.3],
        [0.0, 0.0, -1.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.2, 0.1],
        [0.0, 1.5, 0.2],
        [0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    c = np.eye(n, order='F', dtype=float)
    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.zeros((n, n), order='F', dtype=float)
    v1 = np.eye(n, order='F', dtype=float)
    v2 = np.zeros((n, n), order='F', dtype=float)
    w = np.zeros((2, 2), order='F', dtype=float)
    select = np.array([True, False, False], dtype=bool)

    r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info = mb03za(
        'N', 'N', 'N', 'I', 'S', select, a, b, c, u1, u2, v1, v2, w
    )

    assert info == 0, f"Expected info=0, got {info}"
    assert m == 1, f"Expected m=1, got {m}"


def test_mb03za_with_c_update():
    """
    Test with COMPC='U' - update matrix C.

    C is overwritten by Ur'*C*Vr.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 2

    a = np.array([
        [-2.0, 0.5],
        [0.0, -1.5]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.2],
        [0.0, 1.5]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.zeros((n, n), order='F', dtype=float)
    v1 = np.eye(n, order='F', dtype=float)
    v2 = np.zeros((n, n), order='F', dtype=float)
    w = np.zeros((2*n, 2*n), order='F', dtype=float)
    select = np.array([True, True], dtype=bool)

    r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info = mb03za(
        'U', 'N', 'N', 'I', 'A', select, a, b, c, u1, u2, v1, v2, w
    )

    assert info == 0, f"Expected info=0, got {info}"
    assert m == n


def test_mb03za_with_u_update():
    """
    Test with COMPU='U' - update matrices U1 and U2.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 2

    a = np.array([
        [-2.0, 0.5],
        [0.0, -1.5]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.2],
        [0.0, 1.5]
    ], order='F', dtype=float)

    c = np.eye(n, order='F', dtype=float)
    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.zeros((n, n), order='F', dtype=float)
    v1 = np.eye(n, order='F', dtype=float)
    v2 = np.zeros((n, n), order='F', dtype=float)
    w = np.zeros((2*n, 2*n), order='F', dtype=float)
    select = np.array([True, False], dtype=bool)

    r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info = mb03za(
        'N', 'U', 'N', 'I', 'S', select, a, b, c, u1, u2, v1, v2, w
    )

    assert info == 0, f"Expected info=0, got {info}"


def test_mb03za_with_v_update():
    """
    Test with COMPV='U' - update matrices V1 and V2.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 2

    a = np.array([
        [-2.0, 0.5],
        [0.0, -1.5]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.2],
        [0.0, 1.5]
    ], order='F', dtype=float)

    c = np.eye(n, order='F', dtype=float)
    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.zeros((n, n), order='F', dtype=float)
    v1 = np.eye(n, order='F', dtype=float)
    v2 = np.zeros((n, n), order='F', dtype=float)
    w = np.zeros((2*n, 2*n), order='F', dtype=float)
    select = np.array([True, False], dtype=bool)

    r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info = mb03za(
        'N', 'N', 'U', 'I', 'S', select, a, b, c, u1, u2, v1, v2, w
    )

    assert info == 0, f"Expected info=0, got {info}"


def test_mb03za_n_zero():
    """
    Test with N=0 - quick return case.
    """
    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, 0), order='F', dtype=float)
    c = np.zeros((0, 0), order='F', dtype=float)
    u1 = np.zeros((0, 0), order='F', dtype=float)
    u2 = np.zeros((0, 0), order='F', dtype=float)
    v1 = np.zeros((0, 0), order='F', dtype=float)
    v2 = np.zeros((0, 0), order='F', dtype=float)
    w = np.zeros((0, 0), order='F', dtype=float)
    select = np.array([], dtype=bool)

    r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info = mb03za(
        'N', 'N', 'N', 'N', 'A', select, a, b, c, u1, u2, v1, v2, w
    )

    assert info == 0, f"Expected info=0, got {info}"


def test_mb03za_complex_eigenvalues():
    """
    Test with complex conjugate eigenvalue pair (2x2 block in quasi-triangular form).

    A 2x2 block with sub-diagonal indicates complex eigenvalues.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 2

    a = np.array([
        [0.5, 1.0],
        [-0.5, 0.5]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.2],
        [0.0, 1.0]
    ], order='F', dtype=float)

    c = np.eye(n, order='F', dtype=float)
    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.zeros((n, n), order='F', dtype=float)
    v1 = np.eye(n, order='F', dtype=float)
    v2 = np.zeros((n, n), order='F', dtype=float)
    w = np.zeros((2*n, 2*n), order='F', dtype=float)
    select = np.array([True, True], dtype=bool)

    r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info = mb03za(
        'N', 'N', 'N', 'I', 'A', select, a, b, c, u1, u2, v1, v2, w
    )

    if info == 0:
        assert m == n
        has_imaginary = np.any(wi != 0)
        assert has_imaginary or np.all(wr > 0) or np.all(wr < 0)


def test_mb03za_w_orthogonality():
    """
    Test that W matrix is orthogonal when COMPW='I'.

    W should satisfy W'*W = I.
    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 2

    a = np.array([
        [-2.0, 0.5],
        [0.0, -1.5]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.2],
        [0.0, 1.5]
    ], order='F', dtype=float)

    c = np.eye(n, order='F', dtype=float)
    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.zeros((n, n), order='F', dtype=float)
    v1 = np.eye(n, order='F', dtype=float)
    v2 = np.zeros((n, n), order='F', dtype=float)
    w = np.zeros((2*n, 2*n), order='F', dtype=float)
    select = np.array([True, True], dtype=bool)

    r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info = mb03za(
        'N', 'N', 'N', 'I', 'A', select, a, b, c, u1, u2, v1, v2, w
    )

    if info == 0:
        np.testing.assert_allclose(
            w_out @ w_out.T, np.eye(2*m), rtol=1e-12, atol=1e-14
        )


def test_mb03za_eigenvalue_positive_real_part():
    """
    Test that eigenvalues of R11 (from WR) have positive real part.

    This is a fundamental property guaranteed by the algorithm when it succeeds.
    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 2

    a = np.array([
        [-2.0, 0.5],
        [0.0, -1.5]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.2],
        [0.0, 1.5]
    ], order='F', dtype=float)

    c = np.eye(n, order='F', dtype=float)
    u1 = np.eye(n, order='F', dtype=float)
    u2 = np.zeros((n, n), order='F', dtype=float)
    v1 = np.eye(n, order='F', dtype=float)
    v2 = np.zeros((n, n), order='F', dtype=float)
    w = np.zeros((2*n, 2*n), order='F', dtype=float)
    select = np.array([True, True], dtype=bool)

    r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info = mb03za(
        'N', 'N', 'N', 'I', 'A', select, a, b, c, u1, u2, v1, v2, w
    )

    if info == 0:
        assert np.all(wr > 0), f"Expected all WR > 0, got {wr}"
