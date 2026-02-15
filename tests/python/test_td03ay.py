"""
Tests for TD03AY - Helper for polynomial to state-space conversion.

TD03AY calculates a state-space representation for a (PWORK x MWORK)
transfer matrix given as polynomial row vectors over common denominators.

T(s) = inv(D(s)) * U(s)

where D(s) is diagonal with (I,I)-th element of degree INDEX(I).
The output is in observable companion form with order N = sum(INDEX).
"""

import numpy as np
from numpy.testing import assert_allclose


def test_td03ay_basic_siso():
    """
    Test basic SISO (single-input, single-output) system.

    Transfer function: H(s) = (2s + 1) / (s + 3)

    DCOEFF(1,:) = [1, 3]  (denominator: s + 3)
    UCOEFF(1,1,:) = [2, 1]  (numerator: 2s + 1)
    INDEX = [1]  -> N = 1

    Expected state-space:
    A = [-3], B = [-5], C = [1], D = [2]

    Verification: H(s) = C(sI-A)^{-1}B + D
                      = 1/(s+3)*(-5) + 2
                      = -5/(s+3) + 2(s+3)/(s+3)
                      = (-5 + 2s + 6)/(s+3)
                      = (2s + 1)/(s+3)
    """
    from ctrlsys import td03ay

    mwork, pwork = 1, 1
    n = 1
    index = np.array([1], dtype=np.int32)

    dcoeff = np.array([[1.0, 3.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [2.0, 1.0]

    a, b, c, d, info = td03ay(mwork, pwork, index, dcoeff, ucoeff, n)

    assert info == 0

    assert_allclose(a, np.array([[-3.0]], order='F'), rtol=1e-14)
    assert_allclose(b, np.array([[-5.0]], order='F'), rtol=1e-14)
    assert_allclose(c, np.array([[1.0]], order='F'), rtol=1e-14)
    assert_allclose(d, np.array([[2.0]], order='F'), rtol=1e-14)


def test_td03ay_siso_second_order():
    """
    Test SISO system with second-order denominator.

    Transfer function: H(s) = (s + 2) / (s^2 + 3s + 2)

    DCOEFF(1,:) = [1, 3, 2]  (denominator: s^2 + 3s + 2)
    UCOEFF(1,1,:) = [0, 1, 2]  (numerator: s + 2)
    INDEX = [2]  -> N = 2

    Observable companion form:
    A = [0  -2; 1  -3], B computed from numerator, C = [0, 1], D = [0]
    """
    from ctrlsys import td03ay

    mwork, pwork = 1, 1
    n = 2
    index = np.array([2], dtype=np.int32)

    dcoeff = np.array([[1.0, 3.0, 2.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 3), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 1.0, 2.0]

    a, b, c, d, info = td03ay(mwork, pwork, index, dcoeff, ucoeff, n)

    assert info == 0

    a_expected = np.array([
        [0.0, -2.0],
        [1.0, -3.0]
    ], order='F', dtype=float)
    assert_allclose(a, a_expected, rtol=1e-14, atol=1e-15)

    assert_allclose(c, np.array([[0.0, 1.0]], order='F'), rtol=1e-14)
    assert_allclose(d, np.array([[0.0]], order='F'), rtol=1e-14)


def test_td03ay_transfer_function_equivalence():
    """
    Verify state-space produces identical transfer function.

    Mathematical property test:
    T(s) = C * inv(sI - A) * B + D = inv(D(s)) * U(s)

    Uses simple SISO system for exact validation.
    """
    from ctrlsys import td03ay

    mwork, pwork = 1, 1
    n = 2
    index = np.array([2], dtype=np.int32)

    dcoeff = np.array([[1.0, 2.0, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 3), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 1.0, 0.0]

    a, b, c, d, info = td03ay(mwork, pwork, index, dcoeff, ucoeff, n)
    assert info == 0

    for s in [0.5j, 1.0j, 2.0j, 1.0 + 0.5j]:
        D_s = dcoeff[0, 0] * s**2 + dcoeff[0, 1] * s + dcoeff[0, 2]
        U_s = ucoeff[0, 0, 0] * s**2 + ucoeff[0, 0, 1] * s + ucoeff[0, 0, 2]
        T_poly = U_s / D_s

        I_n = np.eye(n, dtype=complex)
        T_ss = (c @ np.linalg.solve(s * I_n - a, b) + d)[0, 0]

        assert_allclose(T_ss, T_poly, rtol=1e-12, atol=1e-14)


def test_td03ay_mimo_2x2():
    """
    Test 2x2 MIMO system.

    Two outputs, two inputs, each with separate denominator polynomial.
    INDEX = [1, 1] -> N = 2
    """
    from ctrlsys import td03ay

    mwork, pwork = 2, 2
    n = 2
    index = np.array([1, 1], dtype=np.int32)

    dcoeff = np.array([
        [1.0, 2.0],
        [1.0, 3.0]
    ], order='F', dtype=float)

    ucoeff = np.zeros((2, 2, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 0.0]
    ucoeff[0, 1, :] = [0.0, 1.0]
    ucoeff[1, 0, :] = [0.0, 0.5]
    ucoeff[1, 1, :] = [1.0, 0.0]

    a, b, c, d, info = td03ay(mwork, pwork, index, dcoeff, ucoeff, n)

    assert info == 0

    assert a.shape == (n, n)
    assert b.shape == (n, mwork)
    assert c.shape == (pwork, n)
    assert d.shape == (pwork, mwork)

    assert_allclose(c[0, 0], 1.0, rtol=1e-14)
    assert_allclose(c[1, 1], 1.0, rtol=1e-14)


def test_td03ay_zero_leading_coefficient_error():
    """
    Test error when leading denominator coefficient is near zero.

    INFO = I when row I has leading coefficient too small.
    """
    from ctrlsys import td03ay

    mwork, pwork = 1, 1
    n = 1
    index = np.array([1], dtype=np.int32)

    dcoeff = np.array([[1e-320, 1.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 0.0]

    a, b, c, d, info = td03ay(mwork, pwork, index, dcoeff, ucoeff, n)

    assert info == 1


def test_td03ay_mixed_orders():
    """
    Test system with different orders per output.

    INDEX = [2, 1] -> N = 3
    Row 1: second-order denominator
    Row 2: first-order denominator
    """
    from ctrlsys import td03ay

    mwork, pwork = 1, 2
    n = 3
    index = np.array([2, 1], dtype=np.int32)

    dcoeff = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 3.0, 0.0]
    ], order='F', dtype=float)

    ucoeff = np.zeros((2, 1, 3), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 1.0, 0.0]
    ucoeff[1, 0, :] = [0.0, 0.0, 1.0]

    a, b, c, d, info = td03ay(mwork, pwork, index, dcoeff, ucoeff, n)

    assert info == 0
    assert a.shape == (n, n)
    assert b.shape == (n, mwork)
    assert c.shape == (pwork, n)
    assert d.shape == (pwork, mwork)


def test_td03ay_zero_index():
    """
    Test when one output has zero index (direct feedthrough only).

    INDEX = [1, 0] -> N = 1
    Row 2 contributes only to D matrix.
    """
    from ctrlsys import td03ay

    mwork, pwork = 1, 2
    n = 1
    index = np.array([1, 0], dtype=np.int32)

    dcoeff = np.array([
        [1.0, 2.0],
        [2.0, 0.0]
    ], order='F', dtype=float)

    ucoeff = np.zeros((2, 1, 2), order='F', dtype=float)
    ucoeff[0, 0, :] = [1.0, 1.0]
    ucoeff[1, 0, :] = [3.0, 0.0]

    a, b, c, d, info = td03ay(mwork, pwork, index, dcoeff, ucoeff, n)

    assert info == 0

    assert_allclose(d[1, 0], 3.0 / 2.0, rtol=1e-14)


def test_td03ay_observable_companion_form():
    """
    Validate observable companion form structure.

    For diagonal D(s), the A matrix should have:
    - Zeros in upper triangle (except non-trivial columns)
    - Ones on subdiagonal (except at block boundaries)
    - Block structure determined by INDEX
    """
    from ctrlsys import td03ay

    mwork, pwork = 1, 1
    n = 3
    index = np.array([3], dtype=np.int32)

    dcoeff = np.array([[1.0, 6.0, 11.0, 6.0]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 4), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 0.0, 1.0, 2.0]

    a, b, c, d, info = td03ay(mwork, pwork, index, dcoeff, ucoeff, n)

    assert info == 0

    assert_allclose(a[1, 0], 1.0, rtol=1e-14)
    assert_allclose(a[2, 1], 1.0, rtol=1e-14)

    assert_allclose(a[0, 2], -6.0, rtol=1e-14)
    assert_allclose(a[1, 2], -11.0, rtol=1e-14)
    assert_allclose(a[2, 2], -6.0, rtol=1e-14)

    assert_allclose(c[0, 2], 1.0, rtol=1e-14)


def test_td03ay_state_space_equations():
    """
    Validate state equations hold for discrete simulation.

    x(k+1) = A*x(k) + B*u(k)
    y(k) = C*x(k) + D*u(k)

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import td03ay

    np.random.seed(42)

    mwork, pwork = 1, 1
    n = 2
    index = np.array([2], dtype=np.int32)

    dcoeff = np.array([[1.0, 0.5, 0.1]], order='F', dtype=float)
    ucoeff = np.zeros((1, 1, 3), order='F', dtype=float)
    ucoeff[0, 0, :] = [0.0, 1.0, 0.5]

    a, b, c, d, info = td03ay(mwork, pwork, index, dcoeff, ucoeff, n)
    assert info == 0

    x = np.random.randn(n, 1).astype(float, order='F')
    u = np.random.randn(mwork, 1).astype(float, order='F')

    x_next = a @ x + b @ u
    y = c @ x + d @ u

    x_next_manual = np.zeros_like(x)
    for i in range(n):
        for j in range(n):
            x_next_manual[i, 0] += a[i, j] * x[j, 0]
        for k in range(mwork):
            x_next_manual[i, 0] += b[i, k] * u[k, 0]

    assert_allclose(x_next, x_next_manual, rtol=1e-14, atol=1e-15)
