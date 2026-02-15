"""
Test MD03BF - FCN callback for Kowalik-Osborne nonlinear least squares problem.

MD03BF is the example FCN routine for solving a standard nonlinear least squares
problem using MD03BD. It implements the Kowalik-Osborne test function from MINPACK.

Problem: minimize sum_{i=1}^{15} (y_i - (x_1 + i/(x_2*(16-i) + x_3*tmp3)))^2
where tmp3 = (16-i) if i > 8, else i

The test data Y is from the MINPACK LMDER example:
Y = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58,
     0.73, 0.96, 1.34, 2.10, 4.39]
"""
import numpy as np
import pytest

try:
    from ctrlsys import md03bf
    HAS_SLICOT = True
except ImportError:
    HAS_SLICOT = False


def compute_kowalik_osborne_error(x):
    """
    Compute Kowalik-Osborne error function values.

    e(i) = y(i) - (x[0] + i/(x[1]*(16-i) + x[2]*tmp3))
    where tmp3 = (16-i) if i > 8, else i

    This is the reference implementation in Python.
    """
    y = np.array([0.14, 0.18, 0.22, 0.25, 0.29,
                  0.32, 0.35, 0.39, 0.37, 0.58,
                  0.73, 0.96, 1.34, 2.10, 4.39], dtype=np.float64)
    m = 15
    e = np.zeros(m, dtype=np.float64)
    for i in range(1, m + 1):
        tmp1 = float(i)
        tmp2 = float(16 - i)
        tmp3 = tmp2 if i > 8 else tmp1
        e[i-1] = y[i-1] - (x[0] + tmp1 / (x[1]*tmp2 + x[2]*tmp3))
    return e


def compute_kowalik_osborne_jacobian(x):
    """
    Compute Kowalik-Osborne Jacobian matrix.

    J(i,1) = -1
    J(i,2) = i*(16-i) / (x[1]*(16-i) + x[2]*tmp3)^2
    J(i,3) = i*tmp3 / (x[1]*(16-i) + x[2]*tmp3)^2

    This is the reference implementation in Python.
    """
    m = 15
    n = 3
    j = np.zeros((m, n), dtype=np.float64, order='F')
    for i in range(1, m + 1):
        tmp1 = float(i)
        tmp2 = float(16 - i)
        tmp3 = tmp2 if i > 8 else tmp1
        tmp4 = (x[1]*tmp2 + x[2]*tmp3)**2
        j[i-1, 0] = -1.0
        j[i-1, 1] = tmp1*tmp2/tmp4
        j[i-1, 2] = tmp1*tmp3/tmp4
    return j


@pytest.mark.skipif(not HAS_SLICOT, reason="SLICOT not available")
class TestMd03bf:
    """Tests for MD03BF Kowalik-Osborne FCN callback."""

    def test_error_function_at_initial_point(self):
        """
        Test error function evaluation at initial point x = [1, 1, 1].

        Verifies IFLAG=1 computes correct error function values.
        """
        x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        e, info = md03bf(1, x)

        assert info == 0
        e_expected = compute_kowalik_osborne_error(x)
        np.testing.assert_allclose(e, e_expected, rtol=1e-14)

    def test_error_function_at_solution(self):
        """
        Test error function evaluation near the optimal solution.

        Solution from MD03BD.html: x = [0.0824, 1.1330, 2.3437]
        Final residual norm: 0.9063596E-01
        """
        x = np.array([0.0824, 1.1330, 2.3437], dtype=np.float64)
        e, info = md03bf(1, x)

        assert info == 0
        e_expected = compute_kowalik_osborne_error(x)
        np.testing.assert_allclose(e, e_expected, rtol=1e-14)

        residual_norm = np.linalg.norm(e)
        np.testing.assert_allclose(residual_norm, 0.09063596, rtol=1e-3)

    def test_jacobian_at_initial_point(self):
        """
        Test Jacobian evaluation at initial point x = [1, 1, 1].

        Verifies IFLAG=2 computes correct Jacobian matrix.
        """
        x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        j, e, nfevl, info = md03bf(2, x)

        assert info == 0
        assert nfevl == 0

        j_expected = compute_kowalik_osborne_jacobian(x)
        np.testing.assert_allclose(j, j_expected, rtol=1e-14)

    def test_jacobian_at_random_point(self):
        """
        Test Jacobian against finite differences at random point.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        x = np.random.uniform(0.5, 2.0, 3).astype(np.float64)

        j, e, nfevl, info = md03bf(2, x)

        assert info == 0

        j_expected = compute_kowalik_osborne_jacobian(x)
        np.testing.assert_allclose(j, j_expected, rtol=1e-14)

        eps = 1e-7
        m, n = 15, 3
        j_fd = np.zeros((m, n), dtype=np.float64, order='F')
        e0 = compute_kowalik_osborne_error(x)
        for k in range(n):
            x_pert = x.copy()
            x_pert[k] += eps
            e_pert = compute_kowalik_osborne_error(x_pert)
            j_fd[:, k] = (e_pert - e0) / eps

        np.testing.assert_allclose(j, j_fd, rtol=1e-5, atol=1e-6)

    def test_workspace_query(self):
        """
        Test IFLAG=3 returns correct workspace requirements.

        Expected: ipar = (M*N, 0, 0, 4*N+1, 4*N)
        """
        m, n = 15, 3
        x = np.zeros(n, dtype=np.float64)
        ipar, info = md03bf(3, x)

        assert info == 0
        assert ipar[0] == m * n
        assert ipar[1] == 0
        assert ipar[2] == 0
        assert ipar[3] == 4*n + 1
        assert ipar[4] == 4*n

    def test_jacobian_first_column_all_minus_one(self):
        """
        Mathematical property: first column of Jacobian is always -1.

        J(:,1) = de/dx1 = -1 for all i
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m = 15

        for _ in range(5):
            x = np.random.uniform(0.1, 5.0, 3).astype(np.float64)
            j, e, nfevl, info = md03bf(2, x)

            assert info == 0
            np.testing.assert_allclose(j[:, 0], np.full(m, -1.0), rtol=1e-15)
