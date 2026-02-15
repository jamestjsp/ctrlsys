"""
Tests for BD02AD - Benchmark examples for discrete-time dynamical systems.

BD02AD generates benchmark examples for time-invariant, discrete-time
dynamical systems (E, A, B, C, D):

    E x_{k+1} = A x_k + B u_k
          y_k = C x_k + D u_k

This implements the DTDSX (Discrete-Time Dynamical Systems eXamples)
benchmark library from SLICOT Working Note 1998-10.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_bd02ad_example_1_1():
    """
    Test Example 1.1 from SLICOT HTML documentation: Laub 1979, Ex.2.

    Uses NR=(1,1), DEF='D'.
    Expected output from HTML doc:
      N=2, M=1, P=1
      E = I (identity)
      A = [[4, 3], [-4.5, -3.5]]
      B = [[1], [-1]]
      C = [[3, 2]]
      D = 0 (zeros)
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0, f"BD02AD failed with info={info}"
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

    assert vec[0] == True
    assert vec[1] == True
    assert vec[2] == True
    assert vec[3] == False
    assert vec[4] == True
    assert vec[5] == True
    assert vec[6] == True
    assert vec[7] == False


def test_bd02ad_example_1_2():
    """
    Test Example 1.2: Laub 1979, Ex.3.

    Uses NR=(1,2), DEF='D'.
    Expected:
      N=2, M=2, P=2
      A = diag([0.9512, 0.9048])
      B = [[4.877, 4.877], [-1.1895, 3.569]]
      C = I (identity)
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 2], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 2
    assert m == 2
    assert p == 2

    a_expected = np.array([
        [0.9512, 0.0],
        [0.0, 0.9048],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-14)

    b_expected = np.array([
        [4.877, 4.877],
        [-1.1895, 3.569],
    ], order='F', dtype=np.float64)
    assert_allclose(b[:n, :m], b_expected, rtol=1e-14)


def test_bd02ad_example_1_3():
    """
    Test Example 1.3: Van Dooren 1981, Ex.II.

    Uses NR=(1,3), DEF='D'.
    Expected:
      N=2, M=1, P=1
      A = [[2, -1], [1, 0]]
      B = [[1], [0]]
      C = [[0, 1]]
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 3], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 2
    assert m == 1
    assert p == 1

    a_expected = np.array([
        [2.0, -1.0],
        [1.0, 0.0],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-14)

    b_expected = np.array([
        [1.0],
        [0.0],
    ], order='F', dtype=np.float64)
    assert_allclose(b[:n, :m], b_expected, rtol=1e-14)


def test_bd02ad_example_1_4():
    """
    Test Example 1.4: Ionescu/Weiss 1992.

    Uses NR=(1,4), DEF='D'.
    Expected:
      N=2, M=2, P=2
      A = [[0, 1], [0, -1]]
      B = [[1, 1], [2, 1]] (first column has 2 at row 2)
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 4], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 2
    assert m == 2
    assert p == 2


def test_bd02ad_example_1_5():
    """
    Test Example 1.5: Jonckheere 1981.

    Uses NR=(1,5), DEF='D'.
    Expected:
      N=2, M=1, P=2
      A = [[0, 1], [0, 0]]
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 5], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 2
    assert m == 1
    assert p == 2

    a_expected = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ], order='F', dtype=np.float64)
    assert_allclose(a[:n, :n], a_expected, rtol=1e-14)


def test_bd02ad_example_1_10():
    """
    Test Example 1.10: Davison/Wang 1974.

    Uses NR=(1,10), DEF='D'.
    This example has VEC(8)=True (D is NOT zero matrix).
    Expected:
      N=6, M=2, P=2
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 10], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 6
    assert m == 2
    assert p == 2

    assert vec[7] == True

    a_out = a[:n, :n]
    assert_allclose(a_out[0, 1], 1.0, rtol=1e-14)
    assert_allclose(a_out[1, 2], 1.0, rtol=1e-14)
    assert_allclose(a_out[3, 4], 1.0, rtol=1e-14)
    assert_allclose(a_out[4, 5], 1.0, rtol=1e-14)

    b_out = b[:n, :m]
    assert_allclose(b_out[2, 0], 1.0, rtol=1e-14)
    assert_allclose(b_out[5, 1], 1.0, rtol=1e-14)


def test_bd02ad_example_2_1():
    """
    Test Example 2.1: Pappas et al. 1980 - process control of paper machine.

    Uses NR=(2,1), DEF='D' with default parameters.
    Expected:
      N=4, M=1, P=1
    """
    from ctrlsys import bd02ad

    nr = np.array([2, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 4
    assert m == 1
    assert p == 1

    c_out = c[:p, :n]
    assert_allclose(c_out[0, 3], 1.0, rtol=1e-14)


def test_bd02ad_example_2_1_custom_param():
    """
    Test Example 2.1 with custom parameters.

    Uses NR=(2,1), DEF='N'.
    """
    from ctrlsys import bd02ad

    nr = np.array([2, 1], dtype=np.int32)
    tau = 1e6
    delta = 2.0
    K = 1.5
    dpar = np.array([tau, delta, K, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('N', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 4

    temp = delta / tau
    a_out = a[:n, :n]
    assert_allclose(a_out[0, 0], 1.0 - temp, rtol=1e-14)

    b_out = b[:n, :m]
    assert_allclose(b_out[0, 0], K * temp, rtol=1e-14)


def test_bd02ad_example_3_1():
    """
    Test Example 3.1: Pappas et al. 1980, Ex.3 (scalable size).

    Uses NR=(3,1), DEF='D' with default n=100.
    """
    from ctrlsys import bd02ad

    nr = np.array([3, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 100
    assert m == 1
    assert p == 100

    a_out = a[:n, :n]
    nm1 = n - 1
    for i in range(nm1):
        assert_allclose(a_out[i, i + 1], 1.0, rtol=1e-14)

    b_out = b[:n, :m]
    assert_allclose(b_out[n - 1, 0], 1.0, rtol=1e-14)


def test_bd02ad_example_3_1_custom_size():
    """
    Test Example 3.1 with custom size parameter.

    Uses NR=(3,1), DEF='N', IPAR(1)=10.
    """
    from ctrlsys import bd02ad

    nr = np.array([3, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.array([10], dtype=np.int32)

    result = bd02ad('N', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert n == 10
    assert m == 1
    assert p == 10


def test_bd02ad_invalid_example():
    """
    Test error handling for invalid example number.
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 100], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == -2


def test_bd02ad_invalid_group():
    """
    Test error handling for invalid group number.
    """
    from ctrlsys import bd02ad

    nr = np.array([5, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == -2


def test_bd02ad_invalid_def():
    """
    Test error handling for invalid DEF parameter in parameter-dependent examples.
    """
    from ctrlsys import bd02ad

    nr = np.array([2, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('X', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == -1


def test_bd02ad_example_2_1_zero_tau():
    """
    Test error handling when tau=0 (division by zero).
    """
    from ctrlsys import bd02ad

    nr = np.array([2, 1], dtype=np.int32)
    dpar = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('N', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == -3


def test_bd02ad_file_based_example_returns_error():
    """
    Test that file-based examples (1.6-1.9, 1.11, 1.12) return info=1.

    These examples require external data files which we don't support.
    """
    from ctrlsys import bd02ad

    for ex in [6, 7, 8, 9, 11, 12]:
        nr = np.array([1, ex], dtype=np.int32)
        dpar = np.zeros(7, dtype=np.float64)
        ipar = np.zeros(1, dtype=np.int32)

        result = bd02ad('D', nr, dpar, ipar)
        vec, n, m, p, e, a, b, c, d, note, info = result

        assert info == 1, f"Example 1.{ex} should return info=1"


def test_bd02ad_state_space_structure():
    """
    Validate mathematical property: state-space structure is consistent.

    For Ex.1.1, verify dimensions match E*x_{k+1} = A*x_k + B*u_k, y_k = C*x_k + D*u_k.
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
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


def test_bd02ad_note_string():
    """
    Test that NOTE string contains meaningful description.
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0
    assert len(note) > 0
    assert "Laub" in note or "1979" in note


def test_bd02ad_discrete_time_evolution():
    """
    Validate mathematical property: discrete-time state evolution.

    For a discrete-time system (A, B, C, D) with E=I:
    x_{k+1} = A*x_k + B*u_k
    y_k = C*x_k + D*u_k

    Verify single-step evolution.
    Random seed: Not applicable (deterministic problem).
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == 0

    a_out = a[:n, :n]
    b_out = b[:n, :m]
    c_out = c[:p, :n]
    d_out = d[:p, :m]

    x0 = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    u0 = np.array([[1.0]], order='F', dtype=np.float64)

    x1_expected = a_out @ x0 + b_out @ u0
    y0_expected = c_out @ x0 + d_out @ u0

    assert x1_expected.shape == (n, 1)
    assert y0_expected.shape == (p, 1)

    assert_allclose(x1_expected, np.array([[5.0], [-5.5]], order='F'), rtol=1e-14)


def test_bd02ad_markov_parameters():
    """
    Validate mathematical property: Markov parameters for discrete-time system.

    For a state-space system (A, B, C, D), Markov parameters are:
    h(0) = D, h(1) = C*B, h(2) = C*A*B, h(3) = C*A^2*B, ...

    Test with Example 1.1.
    """
    from ctrlsys import bd02ad

    nr = np.array([1, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.zeros(1, dtype=np.int32)

    result = bd02ad('D', nr, dpar, ipar)
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
    expected_h1 = np.array([[1.0]], order='F', dtype=np.float64)
    assert_allclose(h1, expected_h1, rtol=1e-14)


def test_bd02ad_example_3_1_ipar_validation():
    """
    Test error handling for invalid IPAR value (n < 2) in Example 3.1.
    """
    from ctrlsys import bd02ad

    nr = np.array([3, 1], dtype=np.int32)
    dpar = np.zeros(7, dtype=np.float64)
    ipar = np.array([1], dtype=np.int32)

    result = bd02ad('N', nr, dpar, ipar)
    vec, n, m, p, e, a, b, c, d, note, info = result

    assert info == -4
