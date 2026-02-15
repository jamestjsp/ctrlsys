"""
Tests for BB02AD - Benchmark examples for discrete-time algebraic Riccati equations.

BB02AD generates benchmark examples for the numerical solution of discrete-time
algebraic Riccati equations (DAREs) of the form:

    0 = A^T X A - X - (A^T X B + S)(R + B^T X B)^{-1} (B^T X A + S^T) + Q

corresponding to the DAREX collection. This routine is similar to BB01AD but
for discrete-time systems.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_bb02ad_example_2_3():
    """
    Test Example 2.3 from SLICOT HTML documentation.

    "increasingly bad scaled system as eps -> oo"

    Uses NR=(2,3), DEF='N', DPAR(1)=0.1234.
    Expected output from HTML doc:
      A = [[0.0, 0.1234], [0.0, 0.0]]
      B = [[0.0], [1.0]]
      Q = I_2
      R = 1.0
      X = [[1.0, 0.0], [0.0, 1.0152]]
    """
    from ctrlsys import bb02ad

    nr = np.array([2, 3], dtype=np.int32)
    dpar = np.array([0.1234, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.array([0, 0, 0], dtype=np.int32)
    bpar = np.array([True, True, True, False, False, True, True], dtype=bool)

    result = bb02ad('N', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0, f"BB02AD failed with info={info}"
    assert n == 2
    assert m == 1
    assert p == 2

    a_expected = np.array([
        [0.0000, 0.1234],
        [0.0000, 0.0000],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-3, atol=1e-4)

    b_expected = np.array([
        [0.0],
        [1.0],
    ], order='F', dtype=np.float64)
    assert_allclose(b[:n, :m], b_expected, rtol=1e-3, atol=1e-4)

    q_expected = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], order='F', dtype=np.float64)
    assert_allclose(q[:n, :n], q_expected, rtol=1e-3, atol=1e-4)

    r_expected = np.array([[1.0]], order='F', dtype=np.float64)
    assert_allclose(r[:m, :m], r_expected, rtol=1e-3, atol=1e-4)

    x_expected = np.array([
        [1.0000, 0.0000],
        [0.0000, 1.0152],
    ], order='F', dtype=np.float64)
    assert_allclose(x[:n, :n], x_expected, rtol=1e-3, atol=1e-4)

    assert vec[0] == True
    assert vec[1] == True
    assert vec[2] == True
    assert vec[3] == True
    assert vec[4] == True
    assert vec[5] == False
    assert vec[8] == True
    assert vec[9] == True


def test_bb02ad_example_1_1_default():
    """
    Test Example 1.1: Van Dooren 1981, Ex. II (singular R matrix).

    Uses default parameters (DEF='D').
    This is a 2x2 problem with known exact solution X = I (identity).

    Note: Example 1.1 has R=0 (singular), so we use BPAR[3]=False to get
    factored form (B, R) instead of trying to compute G = B*R^{-1}*B^T.
    """
    from ctrlsys import bb02ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, False, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0, f"BB02AD failed with info={info}"
    assert n == 2
    assert m == 1
    assert p == 1

    x_expected = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], order='F', dtype=np.float64)
    assert_allclose(x[:n, :n], x_expected, rtol=1e-14)

    assert vec[9] == True


def test_bb02ad_example_1_3():
    """
    Test Example 1.3: Jonckheere 1981.

    (A,B) controllable, no solution X <= 0.
    Expected X = [[1, 2], [2, 2+sqrt(5)]]
    """
    from ctrlsys import bb02ad

    nr = np.array([1, 3], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0
    assert n == 2
    assert m == 1
    assert p == 2

    a_expected = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-14)

    assert vec[9] == True
    x_expected = np.array([
        [1.0, 2.0],
        [2.0, 2.0 + np.sqrt(5.0)],
    ], order='F', dtype=np.float64)
    assert_allclose(x[:n, :n], x_expected, rtol=1e-14)


def test_bb02ad_example_2_1():
    """
    Test Example 2.1: Laub 1979, Ex. 2 (uncontrollable-unobservable data).

    Uses DPAR(1) as the scalar R matrix value.
    """
    from ctrlsys import bb02ad

    nr = np.array([2, 1], dtype=np.int32)
    dpar = np.array([1e7, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0
    assert n == 2
    assert m == 1
    assert p == 2

    assert vec[9] == True


def test_bb02ad_example_4_1():
    """
    Test Example 4.1: Pappas et al. 1980, Ex. 3 (scalable size).

    Uses IPAR(1) to specify problem size.
    """
    from ctrlsys import bb02ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.array([5, 0, 0], dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True, False], dtype=bool)

    result = bb02ad('N', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0
    assert n == 5
    assert m == 1

    assert vec[9] == True

    for i in range(n):
        assert_allclose(x[i, i], float(i + 1), rtol=1e-14)


def test_bb02ad_with_s_matrix():
    """
    Test Example 1.2 which has non-zero S matrix.

    Ionescu/Weiss 1992: singular R matrix, nonzero S matrix.
    Note: R is not singular in this example, but we use BPAR[3]=False
    to get factored form (B, R) and access S.
    """
    from ctrlsys import bb02ad

    nr = np.array([1, 2], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, False, True, True, True], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0
    assert n == 2
    assert m == 2

    assert vec[8] == True


def test_bb02ad_factored_q():
    """
    Test requesting factored form for Q (BPAR(1)=False).

    When BPAR(1)=False, the routine returns C and Q0 from Q = C^T Q0 C.
    """
    from ctrlsys import bb02ad

    nr = np.array([1, 3], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([False, True, True, True, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0
    assert vec[5] == True


def test_bb02ad_factored_g():
    """
    Test requesting factored form for G (BPAR(4)=False).

    When BPAR(4)=False, the routine returns B and R from G = B R^{-1} B^T.
    """
    from ctrlsys import bb02ad

    nr = np.array([1, 3], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, False, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0
    assert vec[4] == True


def test_bb02ad_invalid_nr_group():
    """
    Test error handling for invalid NR(1) parameter.
    """
    from ctrlsys import bb02ad

    nr = np.array([5, 1], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == -2


def test_bb02ad_invalid_example_number():
    """
    Test error handling for invalid example number in group.
    """
    from ctrlsys import bb02ad

    nr = np.array([1, 20], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == -2


def test_bb02ad_division_by_zero():
    """
    Test error handling when division by zero would occur.

    Example 2.2 with DPAR(1)=0 causes division by zero.
    """
    from ctrlsys import bb02ad

    nr = np.array([2, 2], dtype=np.int32)
    dpar = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True, False], dtype=bool)

    result = bb02ad('N', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 2


def test_bb02ad_dare_residual():
    """
    Validate mathematical property: DARE residual for example with known X.

    For the DARE (when S=0):
        0 = A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q

    The solution X should satisfy this equation.
    """
    from ctrlsys import bb02ad

    nr = np.array([1, 3], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, False, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0
    assert vec[9] == True

    a_out = a[:n, :n]
    b_out = b[:n, :m]
    q_out = q[:n, :n]
    r_out = r[:m, :m]
    x_out = x[:n, :n]

    atxa = a_out.T @ x_out @ a_out
    btxb = b_out.T @ x_out @ b_out
    r_plus_btxb = r_out + btxb
    atxb = a_out.T @ x_out @ b_out
    btxa = b_out.T @ x_out @ a_out

    feedback_term = atxb @ np.linalg.solve(r_plus_btxb, btxa)
    residual = atxa - x_out - feedback_term + q_out

    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_bb02ad_solution_symmetry():
    """
    Validate mathematical property: solution X is symmetric.

    The solution to a DARE should be symmetric: X = X^T.
    """
    from ctrlsys import bb02ad

    nr = np.array([2, 1], dtype=np.int32)
    dpar = np.array([1e6, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0
    assert vec[9] == True

    x_out = x[:n, :n]
    assert_allclose(x_out, x_out.T, rtol=1e-14)


def test_bb02ad_q_symmetry():
    """
    Validate mathematical property: Q matrix is symmetric.

    Q is always symmetric in the DARE formulation.
    """
    from ctrlsys import bb02ad

    nr = np.array([1, 3], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0

    q_out = q[:n, :n]
    assert_allclose(q_out, q_out.T, rtol=1e-14)


def test_bb02ad_r_symmetry():
    """
    Validate mathematical property: R matrix is symmetric.

    R is always symmetric in the DARE formulation.
    """
    from ctrlsys import bb02ad

    nr = np.array([1, 2], dtype=np.int32)
    dpar = np.zeros(4, dtype=np.float64)
    ipar = np.zeros(3, dtype=np.int32)
    bpar = np.array([True, True, True, False, True, True, False], dtype=bool)

    result = bb02ad('D', nr, dpar, ipar, bpar)
    vec, n, m, p, a, b, c, q, r, s, x, info = result

    assert info == 0

    r_out = r[:m, :m]
    assert_allclose(r_out, r_out.T, rtol=1e-14)
