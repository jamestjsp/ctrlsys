import numpy as np
import pytest

pytest.importorskip("ctrlsys")
import ctrlsys


"""Test MD03BD - Levenberg-Marquardt nonlinear least squares optimizer"""

def test_basic_kowalik_osborne():
    """Test MD03BD with Kowalik-Osborne problem from MINPACK

    This test matches the example in MD03BD.html documentation.
    Data from: Kowalik and Osborne function
    m=15 observations, n=3 parameters
    """
    # Problem dimensions
    m = 15
    n = 3

    # Observed data (from MD03BF subroutine in example)
    y = np.array([
        0.14, 0.18, 0.22, 0.25, 0.29,
        0.32, 0.35, 0.39, 0.37, 0.58,
        0.73, 0.96, 1.34, 2.10, 4.39
    ], dtype=np.float64, order='F')

    # Initial guess for parameters
    x0 = np.array([1.0, 1.0, 1.0], dtype=np.float64, order='F')

    # Error function: e(i) = y(i) - (x1 + i/(x2*(16-i) + x3*tmp3))
    # where tmp3 = (16-i) if i > 8 else i
    def fcn(x):
        e = np.zeros(m, dtype=np.float64, order='F')
        for i in range(1, m + 1):  # 1-based for compatibility
            tmp1 = float(i)
            tmp2 = float(16 - i)
            tmp3 = tmp2 if i > 8 else tmp1
            e[i-1] = y[i-1] - (x[0] + tmp1 / (x[1]*tmp2 + x[2]*tmp3))
        return e

    def jac(x):
        j = np.zeros((m, n), dtype=np.float64, order='F')
        for i in range(1, m + 1):  # 1-based
            tmp1 = float(i)
            tmp2 = float(16 - i)
            tmp3 = tmp2 if i > 8 else tmp1
            tmp4 = (x[1]*tmp2 + x[2]*tmp3)**2
            j[i-1, 0] = -1.0
            j[i-1, 1] = tmp1*tmp2/tmp4
            j[i-1, 2] = tmp1*tmp3/tmp4
        return j

    # Tolerances (negative values use defaults)
    ftol = -1.0  # Use sqrt(eps)
    xtol = -1.0  # Use sqrt(eps)
    gtol = -1.0  # Use eps

    # Call MD03BD
    x, nfev, njev, fnorm, iwarn, info = ctrlsys.md03bd(
        m, n, x0, fcn, jac,
        itmax=100,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol
    )

    # Expected results from MD03BD.html
    # Final approximate solution: 0.0824, 1.1330, 2.3437
    # Final residual norm: 0.9063596E-01
    x_expected = np.array([0.0824, 1.1330, 2.3437], dtype=np.float64)
    fnorm_expected = 0.9063596e-01

    assert info == 0, f"MD03BD failed with info={info}"
    assert iwarn in [1, 2, 3], f"Unexpected convergence status: iwarn={iwarn}"

    # Check solution (looser tolerance due to stopping criteria differences)
    np.testing.assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)

    # Check residual norm
    np.testing.assert_allclose(fnorm, fnorm_expected, rtol=1e-3)

    # Verify function/Jacobian evaluation counts are reasonable
    assert nfev > 0, "No function evaluations"
    assert njev > 0, "No Jacobian evaluations"
    assert nfev < 100, f"Too many function evaluations: {nfev}"
    assert njev < 50, f"Too many Jacobian evaluations: {njev}"

def test_zero_residual():
    """Test with problem that has exact solution (zero residual)"""
    m = 5
    n = 2

    # Linear problem: y = 2*x1 + 3*x2 (exact fit possible)
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, order='F')
    y = 2.0 * t + 3.0  # y = 2*t + 3

    x0 = np.array([0.0, 0.0], dtype=np.float64, order='F')

    def fcn(x):
        return y - (x[0] * t + x[1])

    def jac(x):
        j = np.zeros((m, n), dtype=np.float64, order='F')
        j[:, 0] = -t
        j[:, 1] = -1.0
        return j

    x, nfev, njev, fnorm, iwarn, info = ctrlsys.md03bd(
        m, n, x0, fcn, jac,
        itmax=50,
        ftol=-1.0,
        xtol=-1.0,
        gtol=-1.0
    )

    assert info == 0
    np.testing.assert_allclose(x, [2.0, 3.0], rtol=1e-10)
    np.testing.assert_allclose(fnorm, 0.0, atol=1e-10)

def test_error_invalid_dimensions():
    """Test error handling for invalid dimensions"""
    m = 2
    n = 3  # Invalid: n > m

    x0 = np.zeros(n, dtype=np.float64, order='F')

    def fcn(x):
        return np.zeros(m, dtype=np.float64, order='F')

    def jac(x):
        return np.zeros((m, n), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        ctrlsys.md03bd(m, n, x0, fcn, jac, itmax=10)
