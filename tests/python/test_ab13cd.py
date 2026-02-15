"""
Tests for AB13CD: H-infinity norm of continuous-time stable system.

Computes the H-infinity norm of G(s) = C*inv(sI - A)*B + D, which is
the peak gain of the frequency response (largest singular value in MIMO).

Uses numpy only.
"""

import numpy as np


def test_ab13cd_html_example():
    """
    Test AB13CD with example from SLICOT HTML documentation.

    N=6, M=1, NP=1
    System: Three coupled oscillators with different damping
    Expected: H-infinity norm ~ 5.0e5, peak frequency ~ 1.414
    """
    from ctrlsys import ab13cd

    n, m, np_ = 6, 1, 1

    a = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5, -0.0002, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -0.00002, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -2.0, -0.000002]
    ], order='F', dtype=float)

    b = np.array([
        [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    ], order='F', dtype=float)

    d = np.array([[0.0]], order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 0
    np.testing.assert_allclose(hnorm, 5.0e5, rtol=0.01)
    np.testing.assert_allclose(fpeak, 1.414213562, rtol=0.01)


def test_ab13cd_siso_first_order():
    """
    Test AB13CD with simple first-order SISO system.

    G(s) = 1/(s+1) has H-infinity norm = 1 (at omega = 0).
    """
    from ctrlsys import ab13cd

    n, m, np_ = 1, 1, 1

    a = np.array([[-1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 0
    np.testing.assert_allclose(hnorm, 1.0, rtol=1e-6)
    np.testing.assert_allclose(fpeak, 0.0, atol=1e-6)


def test_ab13cd_mimo_2x2():
    """
    Test AB13CD with 2x2 MIMO system.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab13cd

    n, m, np_ = 2, 2, 2

    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    d = np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ], order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 0
    np.testing.assert_allclose(hnorm, 1.0, rtol=1e-6)


def test_ab13cd_d_nonzero():
    """
    Test AB13CD with nonzero D matrix.

    G(s) = 1/(s+1) + 0.5 = (s + 1.5)/(s + 1)
    H-infinity norm is max over omega of |G(j*omega)|.
    At omega=0: G(0) = 1.5
    At omega->inf: G(inf) = 1 (from D)
    Peak is at omega=0, norm = 1.5
    """
    from ctrlsys import ab13cd

    n, m, np_ = 1, 1, 1

    a = np.array([[-1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[0.5]], order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 0
    np.testing.assert_allclose(hnorm, 1.5, rtol=1e-6)


def test_ab13cd_n0():
    """
    Test AB13CD with n=0 (static gain only).

    G(s) = D, norm is just sigma_max(D).
    """
    from ctrlsys import ab13cd

    n, m, np_ = 0, 2, 2

    a = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, 2), order='F', dtype=float)
    c = np.zeros((2, 1), order='F', dtype=float)
    d = np.array([
        [3.0, 4.0],
        [0.0, 0.0]
    ], order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 0
    np.testing.assert_allclose(hnorm, 5.0, rtol=1e-14)


def test_ab13cd_m0():
    """Test AB13CD with m=0 (quick return, norm=0)."""
    from ctrlsys import ab13cd

    n, m, np_ = 2, 0, 1

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
    b = np.zeros((2, 1), order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 0
    assert hnorm == 0.0


def test_ab13cd_np0():
    """Test AB13CD with np=0 (quick return, norm=0)."""
    from ctrlsys import ab13cd

    n, m, np_ = 2, 1, 0

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.zeros((1, 2), order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 0
    assert hnorm == 0.0


def test_ab13cd_unstable_system():
    """
    Test AB13CD with unstable system returns info=1.

    System with eigenvalue at +1 (unstable).
    """
    from ctrlsys import ab13cd

    n, m, np_ = 1, 1, 1

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 1


def test_ab13cd_resonance():
    """
    Test AB13CD with lightly damped oscillator.

    Second-order system with natural frequency omega_n and damping zeta.
    G(s) = 1/(s^2 + 2*zeta*omega_n*s + omega_n^2)

    For zeta=0.01, omega_n=1:
    Peak gain at resonance ~ 1/(2*zeta) = 50 at omega ~ 1

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab13cd

    np.random.seed(123)
    n, m, np_ = 2, 1, 1

    zeta = 0.01
    omega_n = 1.0

    a = np.array([
        [0.0, 1.0],
        [-omega_n**2, -2*zeta*omega_n]
    ], order='F', dtype=float)

    b = np.array([[0.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 0
    np.testing.assert_allclose(hnorm, 50.0, rtol=0.01)
    np.testing.assert_allclose(fpeak, 1.0, rtol=0.01)


def test_ab13cd_negative_n():
    """Test AB13CD returns error for negative n."""
    from ctrlsys import ab13cd

    n, m, np_ = -1, 1, 1

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == -1


def test_ab13cd_eigenvalue_preservation():
    """
    Validate mathematical property: system with poles at -1 and -2.

    The H-infinity norm should be determined by the slowest pole contribution.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import ab13cd

    np.random.seed(456)
    n, m, np_ = 2, 1, 1

    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], order='F', dtype=float)

    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    tol = 1e-9

    hnorm, fpeak, info = ab13cd(n, m, np_, a, b, c, d, tol)

    assert info == 0
    assert hnorm > 0
    assert fpeak >= 0
