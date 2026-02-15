"""
Tests for SB08NY - Compute B(z) = A(1/z) * A(z) for discrete-time spectral factorization.

SB08NY computes the coefficients of the polynomial B(z) = A(1/z) * A(z)
where A(z) is a polynomial given in increasing powers of z.

Also computes an accuracy norm for the computed coefficients.
"""
import numpy as np
from ctrlsys import sb08ny


def test_sb08ny_constant():
    """
    Test with constant polynomial A(z) = 3.

    A(1/z) = 3
    B(z) = A(1/z)*A(z) = 3*3 = 9
    B = [9]
    """
    a = np.array([3.0], dtype=float, order='F')

    b, epsb = sb08ny(a)

    assert b.shape == (1,)
    np.testing.assert_allclose(b[0], 9.0, rtol=1e-14)
    assert epsb > 0


def test_sb08ny_linear():
    """
    Test with linear polynomial A(z) = 1 + 2*z.

    A(1/z) = 1 + 2/z = (z + 2)/z
    B(z) = A(1/z)*A(z) = (1 + 2/z)(1 + 2z) = 1 + 2z + 2/z + 4
         = 5 + 2z + 2/z = (2 + 5z + 2z^2)/z

    Coefficients of B(z) in increasing powers of z (DA+1 = 2):
    B[i] = sum_{k=0}^{DA-i+1} A[k] * A[i+k-1] for Fortran 1-based
    For C 0-based with DA=1:
    B[0] = A[0]*A[0] + A[1]*A[1] = 1*1 + 2*2 = 5
    B[1] = A[0]*A[1] = 1*2 = 2
    """
    a = np.array([1.0, 2.0], dtype=float, order='F')

    b, epsb = sb08ny(a)

    assert b.shape == (2,)
    np.testing.assert_allclose(b[0], 5.0, rtol=1e-14)
    np.testing.assert_allclose(b[1], 2.0, rtol=1e-14)


def test_sb08ny_quadratic():
    """
    Test with quadratic polynomial A(z) = 1 + 2*z + 3*z^2.

    For DA=2, B has 3 coefficients.
    Using the dot product formula B(I) = DDOT(DA-I+2, A(1), 1, A(I), 1):
    B[0] = A[0]*A[0] + A[1]*A[1] + A[2]*A[2] = 1 + 4 + 9 = 14
    B[1] = A[0]*A[1] + A[1]*A[2] = 1*2 + 2*3 = 8
    B[2] = A[0]*A[2] = 1*3 = 3
    """
    a = np.array([1.0, 2.0, 3.0], dtype=float, order='F')

    b, epsb = sb08ny(a)

    assert b.shape == (3,)
    np.testing.assert_allclose(b[0], 14.0, rtol=1e-14)
    np.testing.assert_allclose(b[1], 8.0, rtol=1e-14)
    np.testing.assert_allclose(b[2], 3.0, rtol=1e-14)


def test_sb08ny_property_autocorrelation():
    """
    Verify B(z) = A(1/z) * A(z) represents autocorrelation coefficients.

    For polynomial A(z) = sum_{k=0}^{n} a_k * z^k,
    B(z) = sum_{k=-n}^{n} r_k * z^k where r_k = sum_{j} a_j * a_{j+|k|}

    The output B contains r_0, r_1, ..., r_n (non-negative lags).
    Due to symmetry r_k = r_{-k}.

    Property: B(0) = ||A||^2 = sum of squares of coefficients.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    for _ in range(10):
        da = np.random.randint(1, 8)
        a = np.random.randn(da + 1)
        a = np.array(a, dtype=float, order='F')

        b, epsb = sb08ny(a.copy(order='F'))

        # B[0] = sum of squared coefficients (autocorrelation at lag 0)
        expected_b0 = np.sum(a ** 2)
        np.testing.assert_allclose(b[0], expected_b0, rtol=1e-14)


def test_sb08ny_property_polynomial_evaluation():
    """
    Verify B(z) = A(1/z) * A(z) by evaluating at specific points.

    For z != 0, evaluate A(1/z)*A(z) and compare with polynomial B.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    da = 3
    a = np.random.randn(da + 1)
    a = np.array(a, dtype=float, order='F')

    b, epsb = sb08ny(a.copy(order='F'))

    # Test points (avoiding z=0)
    for z in [0.5, 1.0, 1.5, 2.0, -0.5, -1.0]:
        # A(z) = sum_{k=0}^{da} a[k] * z^k
        a_z = sum(a[k] * (z ** k) for k in range(da + 1))
        # A(1/z) = sum_{k=0}^{da} a[k] * (1/z)^k
        a_inv_z = sum(a[k] * (z ** (-k)) for k in range(da + 1))
        # Product
        product = a_z * a_inv_z

        # B(z) = sum_{k=-da}^{da} r_k * z^k
        # With r_k = r_{-k}, we have:
        # B(z) = r_0 + sum_{k=1}^{da} r_k * (z^k + z^{-k})
        b_z = b[0]
        for k in range(1, da + 1):
            b_z += b[k] * (z ** k + z ** (-k))

        np.testing.assert_allclose(b_z, product, rtol=1e-12)


def test_sb08ny_symmetry():
    """
    Verify autocorrelation symmetry property.

    B(z) has the property that coefficient of z^k equals coefficient of z^{-k}.
    The output array contains only non-negative lag coefficients.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    da = 4
    a = np.random.randn(da + 1)
    a = np.array(a, dtype=float, order='F')

    b, epsb = sb08ny(a.copy(order='F'))

    # Verify by explicit autocorrelation computation
    # r_k = sum_{j=0}^{n-k} a[j] * a[j+k]
    for k in range(da + 1):
        expected_rk = sum(a[j] * a[j + k] for j in range(da + 1 - k))
        np.testing.assert_allclose(b[k], expected_rk, rtol=1e-14)


def test_sb08ny_zero_polynomial():
    """
    Test edge case with zero polynomial A(z) = 0.

    B(z) = 0 * 0 = 0
    """
    a = np.array([0.0], dtype=float, order='F')

    b, epsb = sb08ny(a)

    assert b.shape == (1,)
    np.testing.assert_allclose(b[0], 0.0, atol=1e-14)


def test_sb08ny_accuracy_norm():
    """
    Test that accuracy norm is computed correctly.

    EPSB = 3 * machine_eps * B[0]
    where B[0] = sum of squares of coefficients.

    Note: DLAMCH('E') returns machine epsilon = eps/2 (unit roundoff),
    not numpy's eps. This is the relative machine precision b^(1-p).
    """
    a = np.array([1.0, 2.0, 3.0], dtype=float, order='F')

    b, epsb = sb08ny(a)

    # DLAMCH('E') returns machine epsilon = eps/2 (unit roundoff)
    machine_eps = np.finfo(float).eps / 2.0
    expected_epsb = 3.0 * machine_eps * b[0]
    np.testing.assert_allclose(epsb, expected_epsb, rtol=1e-14)


def test_sb08ny_numerical_example():
    """
    Test with specific numerical example.

    A(z) = 2 + 3*z
    A(1/z) = 2 + 3/z
    B(z) = (2 + 3/z)(2 + 3z) = 4 + 6z + 6/z + 9 = 13 + 6z + 6/z

    Autocorrelation: r_0 = 2^2 + 3^2 = 13, r_1 = 2*3 = 6
    """
    a = np.array([2.0, 3.0], dtype=float, order='F')

    b, epsb = sb08ny(a)

    np.testing.assert_allclose(b[0], 13.0, rtol=1e-14)
    np.testing.assert_allclose(b[1], 6.0, rtol=1e-14)
