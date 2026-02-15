"""
Tests for BB01AD - Benchmark examples for continuous-time algebraic Riccati equations.

BB01AD generates benchmark examples for the numerical solution of continuous-time
algebraic Riccati equations (CAREs) of the form:

    0 = Q + A'X + XA - XGX

corresponding to the Hamiltonian matrix. This routine is part of the CAREX
(Continuous-time Algebraic Riccati Equation eXamples) collection.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_bb01ad_example_2_3():
    """
    Test Example 2.3 from SLICOT HTML documentation.

    Kenney/Laub/Wette 1989, Ex.2: ARE ill conditioned for EPS -> oo

    Uses NR=(2,3), DEF='N', DPAR(1)=0.1234, BPAR=(T,T,T,F,F,T).
    Returns factored form for Q (BPAR(4)=False), full storage for G (BPAR(2)=True).
    """
    from ctrlsys import bb01ad

    nr = [2, 3]
    dpar = np.array([0.1234, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.array([0, 0, 0, 0], dtype=np.int32)
    bpar = np.array([True, True, True, False, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('N', nr, dpar, ipar, bpar)

    assert info == 0, f"BB01AD failed with info={info}"
    assert n == 2, f"Expected n=2, got {n}"
    assert m == 1, f"Expected m=1, got {m}"
    assert p == 2, f"Expected p=2, got {p}"

    a_expected = np.array([
        [0.0000, 0.1234],
        [0.0000, 0.0000],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-3, atol=1e-4)

    c_expected = np.array([
        [1.0000, 0.0000],
        [0.0000, 1.0000],
    ], order='F', dtype=np.float64)
    assert_allclose(c[:p, :n], c_expected, rtol=1e-3, atol=1e-4)

    g_expected = np.array([
        [0.0000, 0.0000],
        [0.0000, 1.0000],
    ], order='F', dtype=np.float64)
    assert_allclose(g[:n, :n], g_expected, rtol=1e-3, atol=1e-4)

    w_expected = np.array([
        [1.0000, 0.0000],
        [0.0000, 1.0000],
    ], order='F', dtype=np.float64)
    assert_allclose(q[:p, :p], w_expected, rtol=1e-3, atol=1e-4)

    x_expected = np.array([
        [9.0486, 1.0000],
        [1.0000, 1.1166],
    ], order='F', dtype=np.float64)
    assert_allclose(x[:n, :n], x_expected, rtol=1e-3, atol=1e-4)

    assert vec[0] == True
    assert vec[1] == True
    assert vec[2] == True
    assert vec[3] == True
    assert vec[4] == False
    assert vec[5] == True
    assert vec[6] == True
    assert vec[7] == True
    assert vec[8] == True


def test_bb01ad_example_1_1_default():
    """
    Test Example 1.1: Laub 1979, Ex.1 (parameter-free, fixed size).

    Uses default parameters (DEF='D').
    This is a simple 2x2 example with known exact solution.
    """
    from ctrlsys import bb01ad

    nr = [1, 1]
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('D', nr, dpar, ipar, bpar)

    assert info == 0, f"BB01AD failed with info={info}"
    assert n == 2
    assert m == 1
    assert p == 2

    a_expected = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-14)

    assert vec[8] == True

    x_expected = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ], order='F', dtype=np.float64)
    assert_allclose(x[:n, :n], x_expected, rtol=1e-14)


def test_bb01ad_example_1_2():
    """
    Test Example 1.2: Laub 1979, Ex.2 (uncontrollable-unobservable data).

    Uses default parameters.
    """
    from ctrlsys import bb01ad

    nr = [1, 2]
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('D', nr, dpar, ipar, bpar)

    assert info == 0, f"BB01AD failed with info={info}"
    assert n == 2
    assert vec[8] == True


def test_bb01ad_example_2_1():
    """
    Test Example 2.1: Arnold/Laub 1984, Ex.1.

    (A,B) becomes unstabilizable as EPS -> 0.
    Uses DPAR(1) as epsilon parameter.
    """
    from ctrlsys import bb01ad

    nr = [2, 1]
    dpar = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('D', nr, dpar, ipar, bpar)

    assert info == 0, f"BB01AD failed with info={info}"
    assert n == 2
    assert vec[8] == True


def test_bb01ad_factored_form():
    """
    Test requesting factored form for G (BPAR(1)=False).

    When BPAR(1)=False, the routine returns B and R from G = B R^{-1} B^T.
    """
    from ctrlsys import bb01ad

    nr = [1, 1]
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([False, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('D', nr, dpar, ipar, bpar)

    assert info == 0, f"BB01AD failed with info={info}"
    assert vec[4] == True


def test_bb01ad_packed_storage():
    """
    Test packed storage mode for G (BPAR(2)=False).

    When BPAR(2)=False, symmetric matrix is stored in packed mode.
    """
    from ctrlsys import bb01ad

    nr = [2, 3]
    dpar = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, False, True, True, False, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('N', nr, dpar, ipar, bpar)

    assert info == 0, f"BB01AD failed with info={info}"
    assert n == 2


def test_bb01ad_invalid_nr():
    """
    Test error handling for invalid NR parameter.
    """
    from ctrlsys import bb01ad

    nr = [5, 1]
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('D', nr, dpar, ipar, bpar)

    assert info == -2, f"Expected info=-2, got {info}"


def test_bb01ad_invalid_example_number():
    """
    Test error handling for invalid example number in group.
    """
    from ctrlsys import bb01ad

    nr = [1, 10]
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('D', nr, dpar, ipar, bpar)

    assert info == -2, f"Expected info=-2, got {info}"


def test_bb01ad_care_residual():
    """
    Validate mathematical property: CARE residual for example with known X.

    For the CARE: 0 = Q + A'X + XA - XGX
    The solution X should satisfy this equation (residual should be small).

    Random seed: N/A (uses fixed example data)
    """
    from ctrlsys import bb01ad

    nr = [1, 1]
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('D', nr, dpar, ipar, bpar)

    assert info == 0
    assert vec[8] == True

    a_out = a[:n, :n]
    g_out = g[:n, :n]
    q_out = q[:n, :n]
    x_out = x[:n, :n]

    residual = q_out + a_out.T @ x_out + x_out @ a_out - x_out @ g_out @ x_out
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_bb01ad_solution_symmetry():
    """
    Validate mathematical property: solution X is symmetric.

    The solution to a CARE should be symmetric: X = X^T.

    Random seed: N/A (uses fixed example data)
    """
    from ctrlsys import bb01ad

    nr = [1, 2]
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('D', nr, dpar, ipar, bpar)

    assert info == 0
    assert vec[8] == True

    x_out = x[:n, :n]
    assert_allclose(x_out, x_out.T, rtol=1e-14)


def test_bb01ad_g_symmetry():
    """
    Validate mathematical property: G matrix is symmetric.

    G is always symmetric (either given or computed from B R^{-1} B^T).

    Random seed: N/A (uses fixed example data)
    """
    from ctrlsys import bb01ad

    nr = [2, 3]
    dpar = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('N', nr, dpar, ipar, bpar)

    assert info == 0

    g_out = g[:n, :n]
    assert_allclose(g_out, g_out.T, rtol=1e-14)


def test_bb01ad_q_symmetry():
    """
    Validate mathematical property: Q matrix is symmetric.

    Q is always symmetric (either given or computed from C^T W C).

    Random seed: N/A (uses fixed example data)
    """
    from ctrlsys import bb01ad

    nr = [1, 1]
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(4, dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    vec, n, m, p, a, b, c, g, q, x, info = bb01ad('D', nr, dpar, ipar, bpar)

    assert info == 0

    q_out = q[:n, :n]
    assert_allclose(q_out, q_out.T, rtol=1e-14)
