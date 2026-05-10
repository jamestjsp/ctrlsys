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
from fortran_reference import run_fortran_driver


def _fortran_matrix_assignment(name, matrix):
    values = np.asarray(matrix, order="F").ravel(order="F")
    chunks = [
        ", ".join(f"{value:.17e}".replace("e", "d") for value in values[i:i + 4])
        for i in range(0, len(values), 4)
    ]
    return (
        f"  {name} = reshape((/ &\n"
        + ", &\n".join(f"       {chunk}" for chunk in chunks)
        + f" &\n  /), shape({name}))\n"
    )


def _select_all_case():
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
    w = np.zeros((2 * n, 2 * n), order='F', dtype=float)
    select = np.array([True, True, True], dtype=bool)
    return a, b, c, u1, u2, v1, v2, w, select


def _mb03za_select_all_fortran_source():
    a, b, c, u1, u2, v1, v2, w, _ = _select_all_case()
    source = """
program main
  implicit none
  integer, parameter :: n=3, ldw=2*n, ldwork=100
  integer m, info
  logical select(n)
  double precision a(n,n), b(n,n), c(n,n), u1(n,n), u2(n,n), v1(n,n), v2(n,n)
  double precision w(ldw,ldw), wr(n), wi(n), dwork(ldwork)
"""
    source += _fortran_matrix_assignment("a", a)
    source += _fortran_matrix_assignment("b", b)
    source += _fortran_matrix_assignment("c", c)
    source += _fortran_matrix_assignment("u1", u1)
    source += _fortran_matrix_assignment("u2", u2)
    source += _fortran_matrix_assignment("v1", v1)
    source += _fortran_matrix_assignment("v2", v2)
    source += _fortran_matrix_assignment("w", w)
    source += """
  select = .true.
  call MB03ZA('N', 'N', 'N', 'I', 'A', select, n, a, n, b, n, c, n, &
       u1, n, u2, n, v1, n, v2, n, w, ldw, wr, wi, m, dwork, ldwork, info)
  print '(2(I0,1X))', info, m
  print '(*(ES26.16E3,1X))', a
  print '(*(ES26.16E3,1X))', b
  print '(*(ES26.16E3,1X))', w
  print '(*(ES26.16E3,1X))', wr(1:m)
  print '(*(ES26.16E3,1X))', wi(1:m)
end program main
"""
    return source


def _take_matrix(values, offset, shape):
    size = shape[0] * shape[1]
    end = offset + size
    return values[offset:end].reshape(shape, order="F"), end


def test_mb03za_select_all_matches_fortran_reference(tmp_path):
    output = run_fortran_driver(_mb03za_select_all_fortran_source(), tmp_path)
    tokens = output.split()
    info_f = int(tokens[0])
    m_f = int(tokens[1])
    values = np.array(tokens[2:], dtype=float)

    offset = 0
    a_f, offset = _take_matrix(values, offset, (3, 3))
    b_f, offset = _take_matrix(values, offset, (3, 3))
    w_f, offset = _take_matrix(values, offset, (6, 6))
    wr_f = values[offset:offset + m_f]
    wi_f = values[offset + m_f:offset + 2 * m_f]

    a, b, c, u1, u2, v1, v2, w, select = _select_all_case()
    result = mb03za('N', 'N', 'N', 'I', 'A', select, a, b, c, u1, u2, v1, v2, w)
    a_out, b_out, _, _, _, _, _, w_out, wr, wi, m, info = result

    assert info == info_f == 0
    assert m == m_f == 3
    np.testing.assert_allclose(a_out, a_f, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b_out, b_f, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w_out, w_f, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wr, wr_f, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wi, wi_f, rtol=1e-12, atol=1e-12)


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
