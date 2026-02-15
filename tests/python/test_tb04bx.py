"""
Tests for TB04BX: Gain of a SISO linear system.

TB04BX computes the gain of a single-input single-output linear system,
given its state-space representation (A,b,c,d), and its poles and zeros.
The matrix A is assumed to be in upper Hessenberg form.

The gain formula is:
    g = (c*(S0*I - A)^(-1)*b + d) * prod(S0 - Pi) / prod(S0 - Zi)

where Pi are poles, Zi are zeros, and S0 is chosen to be different from
all poles and zeros.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tb04bx_simple_first_order():
    """
    Test gain computation for simple first-order system.

    System: H(s) = 2 / (s + 1) with gain = 2

    State-space: A = [-1], B = [1], C = [2], D = 0
    Poles: -1, Zeros: none
    """
    from ctrlsys import tb04bx

    ip = 1
    iz = 0

    a = np.array([[-1.0]], order='F', dtype=float)
    b = np.array([1.0], dtype=float)
    c = np.array([2.0], dtype=float)
    d = 0.0

    pr = np.array([-1.0], dtype=float)
    pi = np.array([0.0], dtype=float)

    zr = np.array([], dtype=float)
    zi = np.array([], dtype=float)

    gain = tb04bx(ip, iz, a, b, c, d, pr, pi, zr, zi)

    assert_allclose(gain, 2.0, rtol=1e-14)


def test_tb04bx_second_order_real_poles():
    """
    Test gain computation for second-order system with real poles.

    System: H(s) = 6 / ((s+1)(s+2)) = 6 / (s^2 + 3s + 2)
    DC gain = 6/2 = 3

    State-space (controllable canonical form):
    A = [[0, 1], [-2, -3]] (upper Hessenberg form)
    B = [0, 1]^T
    C = [6, 0]
    D = 0

    Poles: -1, -2
    Zeros: none
    """
    from ctrlsys import tb04bx

    ip = 2
    iz = 0

    a = np.array([
        [0.0, 1.0],
        [-2.0, -3.0]
    ], order='F', dtype=float)

    b = np.array([0.0, 1.0], dtype=float)
    c = np.array([6.0, 0.0], dtype=float)
    d = 0.0

    pr = np.array([-1.0, -2.0], dtype=float)
    pi = np.array([0.0, 0.0], dtype=float)

    zr = np.array([], dtype=float)
    zi = np.array([], dtype=float)

    gain = tb04bx(ip, iz, a, b, c, d, pr, pi, zr, zi)

    assert_allclose(gain, 6.0, rtol=1e-14)


def test_tb04bx_with_zeros():
    """
    Test gain computation for system with both poles and zeros.

    System: H(s) = 3(s+2) / ((s+1)(s+3)) = (3s+6) / (s^2 + 4s + 3)
    DC gain = 6/3 = 2, Gain = 3

    State-space (controllable canonical form):
    A = [[0, 1], [-3, -4]] (upper Hessenberg)
    B = [0, 1]^T
    C = [6, 3] (gives numerator 6 + 3s = 3(s+2))
    D = 0

    Poles: -1, -3
    Zeros: -2
    """
    from ctrlsys import tb04bx

    ip = 2
    iz = 1

    a = np.array([
        [0.0, 1.0],
        [-3.0, -4.0]
    ], order='F', dtype=float)

    b = np.array([0.0, 1.0], dtype=float)
    c = np.array([6.0, 3.0], dtype=float)
    d = 0.0

    pr = np.array([-1.0, -3.0], dtype=float)
    pi = np.array([0.0, 0.0], dtype=float)

    zr = np.array([-2.0], dtype=float)
    zi = np.array([0.0], dtype=float)

    gain = tb04bx(ip, iz, a, b, c, d, pr, pi, zr, zi)

    assert_allclose(gain, 3.0, rtol=1e-14)


def test_tb04bx_complex_conjugate_poles():
    """
    Test gain computation with complex conjugate poles.

    System: H(s) = 5 / (s^2 + 2s + 5)
    Poles: -1 +/- 2j (complex conjugates)
    DC gain = 5/5 = 1

    State-space (controllable canonical form):
    A = [[0, 1], [-5, -2]]
    B = [0, 1]^T
    C = [5, 0]
    D = 0
    """
    from ctrlsys import tb04bx

    ip = 2
    iz = 0

    a = np.array([
        [0.0, 1.0],
        [-5.0, -2.0]
    ], order='F', dtype=float)

    b = np.array([0.0, 1.0], dtype=float)
    c = np.array([5.0, 0.0], dtype=float)
    d = 0.0

    pr = np.array([-1.0, -1.0], dtype=float)
    pi = np.array([2.0, -2.0], dtype=float)

    zr = np.array([], dtype=float)
    zi = np.array([], dtype=float)

    gain = tb04bx(ip, iz, a, b, c, d, pr, pi, zr, zi)

    assert_allclose(gain, 5.0, rtol=1e-14)


def test_tb04bx_with_feedthrough():
    """
    Test gain computation with nonzero feedthrough D.

    System: H(s) = (s + 4) / (s + 1) = 1 + 3/(s+1)

    State-space:
    A = [-1]
    B = [1]
    C = [3]
    D = 1

    Poles: -1
    Zeros: -4
    Gain = 1
    """
    from ctrlsys import tb04bx

    ip = 1
    iz = 1

    a = np.array([[-1.0]], order='F', dtype=float)
    b = np.array([1.0], dtype=float)
    c = np.array([3.0], dtype=float)
    d = 1.0

    pr = np.array([-1.0], dtype=float)
    pi = np.array([0.0], dtype=float)

    zr = np.array([-4.0], dtype=float)
    zi = np.array([0.0], dtype=float)

    gain = tb04bx(ip, iz, a, b, c, d, pr, pi, zr, zi)

    assert_allclose(gain, 1.0, rtol=1e-14)


def test_tb04bx_zero_poles():
    """
    Test quick return when IP = 0 (no poles).

    When IP = 0, the gain should be 0.0.
    """
    from ctrlsys import tb04bx

    ip = 0
    iz = 0

    a = np.array([[]], order='F', dtype=float).reshape(0, 0)
    b = np.array([], dtype=float)
    c = np.array([], dtype=float)
    d = 5.0

    pr = np.array([], dtype=float)
    pi = np.array([], dtype=float)
    zr = np.array([], dtype=float)
    zi = np.array([], dtype=float)

    gain = tb04bx(ip, iz, a, b, c, d, pr, pi, zr, zi)

    assert gain == 0.0


def test_tb04bx_third_order_mixed():
    """
    Test with third-order system with mixed real and complex poles.

    System: H(s) = 10(s+1) / ((s+2)(s^2 + 2s + 2))
                 = (10s+10) / (s^3 + 4s^2 + 6s + 4)

    Poles: -2, -1+j, -1-j
    Zeros: -1
    Gain = 10

    Controllable canonical form:
    A = [[0, 1, 0], [0, 0, 1], [-4, -6, -4]] (upper Hessenberg)
    B = [0, 0, 1]^T
    C = [10, 10, 0]
    D = 0

    Note: Higher-order systems have inherent numerical sensitivity in the
    gain computation formula due to the inverse and products involved.
    """
    from ctrlsys import tb04bx

    ip = 3
    iz = 1

    a = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-4.0, -6.0, -4.0]
    ], order='F', dtype=float)

    b = np.array([0.0, 0.0, 1.0], dtype=float)
    c = np.array([10.0, 10.0, 0.0], dtype=float)
    d = 0.0

    pr = np.array([-2.0, -1.0, -1.0], dtype=float)
    pi = np.array([0.0, 1.0, -1.0], dtype=float)

    zr = np.array([-1.0], dtype=float)
    zi = np.array([0.0], dtype=float)

    gain = tb04bx(ip, iz, a, b, c, d, pr, pi, zr, zi)

    assert_allclose(gain, 10.0, rtol=0.03)


def test_tb04bx_complex_conjugate_zeros():
    """
    Test with complex conjugate zeros.

    System: H(s) = (s^2 + 4) / ((s+1)(s+2)) = (s^2+4) / (s^2+3s+2)

    This is bi-proper (degree num = degree den), so D = 1.
    The strictly proper part: (s^2+4)/(s^2+3s+2) - 1 = (2-3s)/(s^2+3s+2)

    State-space:
    A = [[0, 1], [-2, -3]], B = [0, 1]^T, C = [2, -3], D = 1

    Poles: -1, -2
    Zeros: +2j, -2j (complex conjugates)
    Gain = 1 (leading coefficient ratio of num/den)

    DC gain = 4/2 = 2
    """
    from ctrlsys import tb04bx

    ip = 2
    iz = 2

    a = np.array([
        [0.0, 1.0],
        [-2.0, -3.0]
    ], order='F', dtype=float)

    b = np.array([0.0, 1.0], dtype=float)
    c = np.array([2.0, -3.0], dtype=float)
    d = 1.0

    pr = np.array([-1.0, -2.0], dtype=float)
    pi = np.array([0.0, 0.0], dtype=float)

    zr = np.array([0.0, 0.0], dtype=float)
    zi = np.array([2.0, -2.0], dtype=float)

    gain = tb04bx(ip, iz, a, b, c, d, pr, pi, zr, zi)

    assert_allclose(gain, 1.0, rtol=1e-14)
