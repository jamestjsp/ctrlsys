"""
Tests for BD01AD - Benchmark examples for continuous-time dynamical systems.

BD01AD generates benchmark examples for time-invariant, continuous-time
dynamical systems (E, A, B, C, D):

    E x'(t) = A x(t) + B u(t)
      y(t)  = C x(t) + D u(t)

This implements the CTDSX (Continuous-Time Dynamical Systems eXamples)
benchmark library from SLICOT Working Note 1998-9.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_bd01ad_example_1_1():
    """
    Test Example 1.1 from SLICOT HTML documentation: Laub 1979, Ex.1.

    Uses NR=(1,1), DEF='D'.
    Expected output:
      N=2, M=1, P=2
      E = I (identity)
      A = [[0, 1], [0, 0]]
      B = [[0], [1]]
      C = I (identity)
      D = 0 (zeros)
    """
    from ctrlsys import bd01ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0, f"BD01AD failed with info={info}"
    assert n == 2
    assert m == 1
    assert p == 2

    a_expected = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-14)

    b_expected = np.array([
        [0.0],
        [1.0],
    ], order='F', dtype=np.float64)
    assert_allclose(b[:n, :m], b_expected, rtol=1e-14)

    c_expected = np.eye(2, order='F', dtype=np.float64)
    assert_allclose(c[:p, :n], c_expected, rtol=1e-14)

    assert vec[0] == True
    assert vec[1] == True
    assert vec[2] == True
    assert vec[3] == False
    assert vec[4] == True
    assert vec[5] == True
    assert vec[6] == True
    assert vec[7] == False


def test_bd01ad_example_1_2():
    """
    Test Example 1.2: Laub 1979, Ex.2 - uncontrollable-unobservable data.

    Uses NR=(1,2), DEF='D'.
    Expected output:
      N=2, M=1, P=1
      A = [[4, 3], [-4.5, -3.5]]
      B = [[1], [-1]]
      C = [[3, 2]]
    """
    from ctrlsys import bd01ad

    nr = np.array([1, 2], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 2
    assert m == 1
    assert p == 1

    a_expected = np.array([
        [4.0, 3.0],
        [-4.5, -3.5],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-14)

    b_expected = np.array([
        [1.0],
        [-1.0],
    ], order='F', dtype=np.float64)
    assert_allclose(b[:n, :m], b_expected, rtol=1e-14)

    c_expected = np.array([[3.0, 2.0]], order='F', dtype=np.float64)
    assert_allclose(c[:p, :n], c_expected, rtol=1e-14)


def test_bd01ad_example_2_1():
    """
    Test Example 2.1: Chow/Kokotovic 1976 - magnetic tape control system.

    Uses NR=(2,1), DEF='D' with default epsilon=1e-6.
    Expected:
      N=4, M=1, P=2
    """
    from ctrlsys import bd01ad

    nr = np.array([2, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 4
    assert m == 1
    assert p == 2

    eps = 1e-6
    a_out = a[:n, :n]
    assert_allclose(a_out[0, 1], 0.4, rtol=1e-14)
    assert_allclose(a_out[1, 2], 0.345, rtol=1e-14)
    assert_allclose(a_out[2, 1], -0.524 / eps, rtol=1e-14)
    assert_allclose(a_out[2, 2], -0.465 / eps, rtol=1e-14)
    assert_allclose(a_out[2, 3], 0.262 / eps, rtol=1e-14)
    assert_allclose(a_out[3, 3], -1.0 / eps, rtol=1e-14)

    b_out = b[:n, :m]
    assert_allclose(b_out[3, 0], 1.0 / eps, rtol=1e-14)


def test_bd01ad_example_2_1_custom_param():
    """
    Test Example 2.1 with custom epsilon parameter.

    Uses NR=(2,1), DEF='N', DPAR(1)=0.01.
    """
    from ctrlsys import bd01ad

    nr = np.array([2, 1], dtype=np.int32)
    dpar = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('N', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 4
    assert m == 1
    assert p == 2

    eps = 0.01
    a_out = a[:n, :n]
    assert_allclose(a_out[2, 1], -0.524 / eps, rtol=1e-14)
    assert_allclose(a_out[3, 3], -1.0 / eps, rtol=1e-14)


def test_bd01ad_example_2_2():
    """
    Test Example 2.2: Arnold/Laub 1984.

    Uses NR=(2,2), DEF='D' with default epsilon=1e-6.
    """
    from ctrlsys import bd01ad

    nr = np.array([2, 2], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 4
    assert m == 1
    assert p == 1

    eps = 1e-6
    a_out = a[:n, :n]
    assert_allclose(a_out[0, 0], -eps, rtol=1e-14)
    assert_allclose(a_out[0, 1], 1.0, rtol=1e-14)
    assert_allclose(a_out[1, 0], -1.0, rtol=1e-14)
    assert_allclose(a_out[1, 1], -eps, rtol=1e-14)

    b_out = b[:n, :m]
    assert_allclose(b_out, np.ones((4, 1)), rtol=1e-14)

    c_out = c[:p, :n]
    assert_allclose(c_out, np.ones((1, 4)), rtol=1e-14)


def test_bd01ad_example_2_7():
    """
    Test Example 2.7: Ackermann 1989 - track-guided bus.

    Uses NR=(2,7), DEF='D' with mu=15, nu=10.
    """
    from ctrlsys import bd01ad

    nr = np.array([2, 7], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 5
    assert m == 1
    assert p == 1

    mu = 15.0
    nu = 10.0

    a_out = a[:n, :n]
    assert_allclose(a_out[0, 0], -668.0 / (mu * nu), rtol=1e-14)
    assert_allclose(a_out[0, 1], -1.0 + 180.4 / (mu * nu * nu), rtol=1e-14)

    b_out = b[:n, :m]
    assert_allclose(b_out[4, 0], 1.0, rtol=1e-14)

    c_out = c[:p, :n]
    assert_allclose(c_out[0, 2], 1.0, rtol=1e-14)
    assert_allclose(c_out[0, 3], 6.12, rtol=1e-14)


def test_bd01ad_example_3_1():
    """
    Test Example 3.1: Laub 1979, Ex.4 - string of high speed vehicles.

    Uses NR=(3,1), DEF='D' with default q=20.
    This produces N=2*q-1=39, M=q=20, P=q-1=19.
    """
    from ctrlsys import bd01ad

    nr = np.array([3, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 39
    assert m == 20
    assert p == 19

    a_out = a[:n, :n]
    for i in range(n):
        if (i + 1) % 2 == 1:
            assert_allclose(a_out[i, i], -1.0, rtol=1e-14)
        else:
            assert_allclose(a_out[i, i - 1], 1.0, rtol=1e-14)
            assert_allclose(a_out[i, i + 1], -1.0, rtol=1e-14)

    b_out = b[:n, :m]
    for i in range(n):
        if (i + 1) % 2 == 1:
            col = i // 2
            assert_allclose(b_out[i, col], 1.0, rtol=1e-14)


def test_bd01ad_example_3_1_custom_size():
    """
    Test Example 3.1 with custom size parameter.

    Uses NR=(3,1), DEF='N', IPAR(1)=5.
    This produces N=2*5-1=9, M=5, P=4.
    """
    from ctrlsys import bd01ad

    nr = np.array([3, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.array([5], dtype=np.int32)

    result = bd01ad('N', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 9
    assert m == 5
    assert p == 4


def test_bd01ad_example_3_2():
    """
    Test Example 3.2: Hodel et al. 1996 - heat flow in thin rod.

    Uses NR=(3,2), DEF='D' with default n=100.
    """
    from ctrlsys import bd01ad

    nr = np.array([3, 2], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 100
    assert m == 1
    assert p == 100

    temp = float(n + 1)

    a_out = a[:n, :n]
    assert_allclose(a_out[0, 0], -temp, rtol=1e-14)

    for i in range(1, n):
        assert_allclose(a_out[i, i], -2.0 * temp, rtol=1e-14)

    for i in range(n - 1):
        assert_allclose(a_out[i, i + 1], temp, rtol=1e-14)
        assert_allclose(a_out[i + 1, i], temp, rtol=1e-14)

    b_out = b[:n, :m]
    assert_allclose(b_out[n - 1, 0], temp, rtol=1e-14)


def test_bd01ad_example_3_3():
    """
    Test Example 3.3: Laub 1979, Ex.6 - integrator chain.

    Uses NR=(3,3), DEF='D' with default n=21.
    """
    from ctrlsys import bd01ad

    nr = np.array([3, 3], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 21
    assert m == 1
    assert p == 1

    a_out = a[:n, :n]
    for i in range(n - 1):
        assert_allclose(a_out[i, i + 1], 1.0, rtol=1e-14)

    b_out = b[:n, :m]
    assert_allclose(b_out[n - 1, 0], 1.0, rtol=1e-14)

    c_out = c[:p, :n]
    assert_allclose(c_out[0, 0], 1.0, rtol=1e-14)


def test_bd01ad_example_4_1():
    """
    Test Example 4.1: Rosen/Wang 1995 - 1D heat flow control.

    Uses NR=(4,1), DEF='D' with default n=100.
    VEC(4)=True (E is NOT identity), this is a descriptor system.
    """
    from ctrlsys import bd01ad

    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 100
    assert m == 1
    assert p == 1

    assert vec[3] == True

    e_out = e[:n, :n]
    assert not np.allclose(e_out, np.eye(n))

    for i in range(n):
        assert e_out[i, i] != 0.0


def test_bd01ad_example_4_2():
    """
    Test Example 4.2: Hench et al. 1995 - coupled springs, dashpots, masses.

    Uses NR=(4,2), DEF='D' with default l=30.
    VEC(4)=True (E is NOT identity).
    """
    from ctrlsys import bd01ad

    nr = np.array([4, 2], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 60
    assert m == 2
    assert p == 60

    assert vec[3] == True

    e_out = e[:n, :n]
    assert not np.allclose(e_out, np.eye(n))


def test_bd01ad_invalid_example():
    """
    Test error handling for invalid example number.
    """
    from ctrlsys import bd01ad

    nr = np.array([1, 100], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == -2


def test_bd01ad_invalid_group():
    """
    Test error handling for invalid group number.
    """
    from ctrlsys import bd01ad

    nr = np.array([5, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == -2


def test_bd01ad_invalid_def():
    """
    Test error handling for invalid DEF parameter in parameter-dependent examples.
    """
    from ctrlsys import bd01ad

    nr = np.array([2, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('X', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == -1


def test_bd01ad_example_2_1_zero_epsilon():
    """
    Test error handling when epsilon=0 (division by zero).
    """
    from ctrlsys import bd01ad

    nr = np.array([2, 1], dtype=np.int32)
    dpar = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('N', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == -3


def test_bd01ad_file_based_example_returns_error():
    """
    Test that file-based examples (1.3-1.10) return info=1.

    These examples require external data files which we don't support.
    """
    from ctrlsys import bd01ad

    for ex in [3, 4, 5, 6, 7, 8, 9, 10]:
        nr = np.array([1, ex], dtype=np.int32)
        dpar = np.zeros(7, dtype=np.float64)
        ipar = np.zeros(1, dtype=np.int32)

        result = bd01ad('D', nr, dpar, ipar)
        vec, n, m, p, e, a, b, c, d, note, info = result

        assert info == 1, f"Example 1.{ex} should return info=1"


def test_bd01ad_state_space_structure():
    """
    Validate mathematical property: state-space structure is consistent.

    For Ex.1.1, verify dimensions match E*x' = A*x + B*u, y = C*x + D*u.
    """
    from ctrlsys import bd01ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0

    a_out = a[:n, :n]
    b_out = b[:n, :m]
    c_out = c[:p, :n]
    d_out = d[:p, :m]

    assert a_out.shape == (n, n)
    assert b_out.shape == (n, m)
    assert c_out.shape == (p, n)
    assert d_out.shape == (p, m)

    if not vec[3]:
        e_out = e[:n, :n]
        assert_allclose(e_out, np.eye(n), rtol=1e-14)

    if not vec[7]:
        assert_allclose(d_out, np.zeros((p, m)), rtol=1e-14)


def test_bd01ad_note_string():
    """
    Test that NOTE string contains meaningful description.
    """
    from ctrlsys import bd01ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert len(note) > 0
    assert "Laub" in note or "1979" in note


def test_bd01ad_eigenvalue_stability():
    """
    Validate mathematical property: eigenvalue structure of A matrix.

    For Example 1.1 (double integrator), eigenvalues should be at origin.

    Random seed: Not applicable (deterministic problem).
    """
    from ctrlsys import bd01ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0

    a_out = a[:n, :n]
    eigenvalues = np.linalg.eigvals(a_out)

    assert_allclose(eigenvalues, np.zeros(n), atol=1e-14)


def test_bd01ad_markov_parameters():
    """
    Validate mathematical property: Markov parameters for state-space system.

    For a state-space system (A, B, C, D), Markov parameters are:
    h(0) = D, h(1) = C*B, h(2) = C*A*B, h(3) = C*A^2*B, ...

    Test with Example 1.1.
    """
    from ctrlsys import bd01ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd01ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0

    a_out = a[:n, :n]
    b_out = b[:n, :m]
    c_out = c[:p, :n]
    d_out = d[:p, :m]

    h0 = d_out
    h1 = c_out @ b_out
    h2 = c_out @ a_out @ b_out

    assert_allclose(h0, np.zeros((p, m)), atol=1e-14)
    expected_h1 = np.array([[0.0], [1.0]], order='F', dtype=np.float64)
    assert_allclose(h1, expected_h1, rtol=1e-14)
