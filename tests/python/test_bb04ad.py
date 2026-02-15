"""
Tests for BB04AD - Benchmark examples for (generalized) discrete-time Lyapunov equations.

BB04AD generates benchmark examples for numerical solution of (generalized)
discrete-time Lyapunov equations:

    A^T X A - E^T X E = Y

In some examples, the right hand side has the form Y = -B^T B
and the solution can be represented as X = U^T U.

This implements the DTLEX benchmark collection (Kressner/Mehrmann/Penzl 1999).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_bb04ad_example_4_1():
    """
    Test Example 4.1 from SLICOT HTML documentation.

    NR=(4,1), DEF='N', DPAR=(1.5, 1.5), IPAR(1)=5.
    This is a parameter-dependent problem of scalable size.

    Expected outputs from HTML doc:
      N=5, M=1, E=identity
      A, B, Y, X matrices as shown in Program Results
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0, f"BB04AD failed with info={info}"
    assert n == 5
    assert m == 1

    assert vec[0] == True
    assert vec[1] == True
    assert vec[2] == False
    assert vec[3] == True
    assert vec[4] == True
    assert vec[5] == True
    assert vec[6] == True
    assert vec[7] == False

    a_expected = np.array([
        [0.4562, 0.0308, 0.1990, 0.0861, 0.0217],
        [0.0637, 0.5142, -0.1828, 0.0096, -0.1148],
        [0.3139, 0.1287, 0.3484, 0.1653, -0.1975],
        [0.1500, 0.0053, -0.1838, 0.2501, -0.0687],
        [0.0568, -0.1006, -0.3735, -0.0202, 0.2285],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-3, atol=1e-4)

    b_expected = np.array([
        [0.3086, 0.0247, -0.4691, 0.1728, -0.3704],
    ], order='F', dtype=np.float64)
    assert_allclose(b[:m, :n], b_expected, rtol=1e-3, atol=1e-4)

    y_expected = np.array([
        [-0.0953, -0.0076, 0.1448, -0.0533, 0.1143],
        [-0.0076, -0.0006, 0.0116, -0.0043, 0.0091],
        [0.1448, 0.0116, -0.2201, 0.0811, -0.1738],
        [-0.0533, -0.0043, 0.0811, -0.0299, 0.0640],
        [0.1143, 0.0091, -0.1738, 0.0640, -0.1372],
    ], order='F', dtype=np.float64)
    assert_allclose(y[:n, :n], y_expected, rtol=1e-3, atol=1e-4)

    x_expected = np.array([
        [0.0953, 0.0076, -0.1448, 0.0533, -0.1143],
        [0.0076, 0.0006, -0.0116, 0.0043, -0.0091],
        [-0.1448, -0.0116, 0.2201, -0.0811, 0.1738],
        [0.0533, 0.0043, -0.0811, 0.0299, -0.0640],
        [-0.1143, -0.0091, 0.1738, -0.0640, 0.1372],
    ], order='F', dtype=np.float64)
    assert_allclose(x[:n, :n], x_expected, rtol=1e-3, atol=1e-4)


def test_bb04ad_example_4_1_default():
    """
    Test Example 4.1 with default parameters (DEF='D').

    Default: IPAR(1)=10, DPAR(1)=1.5, DPAR(2)=1.5.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([0.0, 0.0], dtype=np.float64)
    ipar = np.array([0], dtype=np.int32)

    result = bb04ad('D', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0
    assert n == 10
    assert m == 1
    assert_allclose(dpar[0], 1.5, rtol=1e-14)
    assert_allclose(dpar[1], 1.5, rtol=1e-14)

    assert vec[5] == True
    assert vec[6] == True


def test_bb04ad_example_4_2():
    """
    Test Example 4.2: parameter-dependent scalable problem.

    Uses DPAR(1)=lambda in (-1,1), DPAR(2)=s>1 as parameters.
    Default: lambda=-0.5, s=1.5, n=10.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 2], dtype=np.int32)
    dpar = np.array([-0.5, 1.5], dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0
    assert n == 5
    assert m == 1

    assert vec[5] == True
    assert vec[6] == False


def test_bb04ad_example_4_3():
    """
    Test Example 4.3: generalized Lyapunov equation (E not identity).

    Default: IPAR(1)=10, DPAR(1)=10.
    Has E matrix, solution X provided.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 3], dtype=np.int32)
    dpar = np.array([10.0, 0.0], dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0
    assert n == 5
    assert m == 0

    assert vec[2] == True
    assert vec[5] == False
    assert vec[6] == True


def test_bb04ad_example_4_4():
    """
    Test Example 4.4: generalized Lyapunov with E matrix.

    Uses IPAR(1)=q to define n=3*q.
    Default: q=10, t=1.5.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 4], dtype=np.int32)
    dpar = np.array([1.5, 0.0], dtype=np.float64)
    ipar = np.array([3], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0
    assert n == 9
    assert m == 1

    assert vec[2] == True
    assert vec[5] == True
    assert vec[6] == False


def test_bb04ad_lyapunov_residual():
    """
    Validate mathematical property: discrete-time Lyapunov equation residual.

    For discrete-time Lyapunov equation:
        A^T X A - E^T X E = Y

    The solution X should satisfy this equation.
    Uses Example 4.1 which has known solution.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([4], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0
    assert vec[6] == True

    a_out = a[:n, :n]
    e_out = e[:n, :n]
    y_out = y[:n, :n]
    x_out = x[:n, :n]

    residual = a_out.T @ x_out @ a_out - e_out.T @ x_out @ e_out - y_out

    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_bb04ad_y_from_b():
    """
    Validate mathematical property: Y = -B^T B when B is provided.

    For examples with vec[5]=True (B provided), the RHS Y = -B^T B.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([4], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0
    assert vec[5] == True

    b_out = b[:m, :n]
    y_out = y[:n, :n]

    y_from_b = -b_out.T @ b_out

    assert_allclose(y_out, y_from_b, rtol=1e-13)


def test_bb04ad_solution_symmetry():
    """
    Validate mathematical property: solution X is symmetric.

    For Lyapunov equations, the solution matrix X should be symmetric.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0
    assert vec[6] == True

    x_out = x[:n, :n]
    assert_allclose(x_out, x_out.T, rtol=1e-13, atol=1e-15)


def test_bb04ad_y_symmetry():
    """
    Validate mathematical property: Y matrix is symmetric.

    The right-hand side Y should be symmetric.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0

    y_out = y[:n, :n]
    assert_allclose(y_out, y_out.T, rtol=1e-14)


def test_bb04ad_invalid_group():
    """
    Test error handling for invalid NR(1) parameter.

    Only NR(1)=4 is supported in BB04AD.
    """
    from ctrlsys import bb04ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.array([0.0, 0.0], dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bb04ad('D', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == -2


def test_bb04ad_invalid_example():
    """
    Test error handling for invalid NR(2) parameter.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 10], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == -2


def test_bb04ad_invalid_dpar_4_1():
    """
    Test error handling for invalid DPAR values in Example 4.1.

    r > 1 and s > 1 are required.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([0.5, 1.5], dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == -3


def test_bb04ad_invalid_ipar():
    """
    Test error handling for invalid IPAR values.

    IPAR(1) must be >= 2 for Examples 4.1, 4.2, 4.3.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([1], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == -4


def test_bb04ad_example_4_3_e_structure():
    """
    Test Example 4.3 E matrix structure.

    E should be lower triangular with specific structure:
    E(i,j) = 2^(-t) for i > j, E(i,i) = 1.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 3], dtype=np.int32)
    dpar = np.array([10.0, 0.0], dtype=np.float64)
    ipar = np.array([3], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0
    assert vec[2] == True

    e_out = e[:n, :n]
    for i in range(n):
        assert_allclose(e_out[i, i], 1.0, rtol=1e-14)


def test_bb04ad_note_string():
    """
    Test that NOTE string is returned correctly.
    """
    from ctrlsys import bb04ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bb04ad('N', nr, dpar, ipar)
    vec, n, m, e, a, y, b, x, u, note, info = result

    assert info == 0
    assert "4.1" in note or "DTLEX" in note
