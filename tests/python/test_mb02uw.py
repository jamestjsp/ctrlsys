"""
Tests for MB02UW: Solve linear system AX = sB or A'X = sB with scaling/perturbation.

MB02UW solves 1x1 or 2x2 linear systems with automatic scaling to prevent
overflow and perturbation of near-singular matrices.
"""
import numpy as np
import pytest
from ctrlsys import mb02uw


class TestMB02UWBasic:
    """Basic functionality tests for MB02UW."""

    def test_1x1_system(self):
        """
        Test 1x1 system solve: A * X = B.

        Simple scalar equation: 4 * X = 2, X = 0.5
        """
        a = np.array([[4.0]], order='F', dtype=float)
        b = np.array([[2.0]], order='F', dtype=float)
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(False, par, a, b)

        assert iwarn == 0
        assert scale == 1.0
        np.testing.assert_allclose(x, [[0.5]], rtol=1e-14)

    def test_1x1_multiple_rhs(self):
        """
        Test 1x1 system with multiple RHS: A * X = B, A scalar, B is 1xM.

        2 * X = [4, 6, 8] => X = [2, 3, 4]
        """
        a = np.array([[2.0]], order='F', dtype=float)
        b = np.array([[4.0, 6.0, 8.0]], order='F', dtype=float)
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(False, par, a, b)

        assert iwarn == 0
        assert scale == 1.0
        np.testing.assert_allclose(x, [[2.0, 3.0, 4.0]], rtol=1e-14)

    def test_2x2_system(self):
        """
        Test 2x2 system solve: A * X = B.

        A = [[3, 1], [1, 2]], B = [[5], [5]]
        X = [[1], [2]] since 3*1 + 1*2 = 5 and 1*1 + 2*2 = 5
        """
        a = np.array([[3.0, 1.0], [1.0, 2.0]], order='F', dtype=float)
        b = np.array([[5.0], [5.0]], order='F', dtype=float)
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(False, par, a, b)

        assert iwarn == 0
        assert scale == 1.0
        np.testing.assert_allclose(x, [[1.0], [2.0]], rtol=1e-14)

    def test_2x2_transpose(self):
        """
        Test 2x2 transpose system: A' * X = B.

        A = [[3, 1], [1, 2]], A' = [[3, 1], [1, 2]] (symmetric case)
        B = [[5], [5]], X = [[1], [2]]
        """
        a = np.array([[3.0, 1.0], [1.0, 2.0]], order='F', dtype=float)
        b = np.array([[5.0], [5.0]], order='F', dtype=float)
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x_trans, scale_trans, iwarn_trans = mb02uw(True, par, a, b.copy())
        x_no_trans, scale_no_trans, iwarn_no_trans = mb02uw(False, par, a, b.copy())

        assert iwarn_trans == 0
        np.testing.assert_allclose(x_trans, x_no_trans, rtol=1e-14)

    def test_2x2_non_symmetric_transpose(self):
        """
        Test 2x2 with non-symmetric A and transpose.

        A = [[2, 3], [1, 4]] in NumPy row-major notation
        Memory layout (F-order): [2, 1, 3, 4] = col1=[2,1], col2=[3,4]

        For ltrans=False: A*X = B where X = [[1],[3]]
        B = A @ X = [[2,3],[1,4]] @ [[1],[3]] = [[11],[13]]

        For ltrans=True: A'*X = B where A' = [[2,1],[3,4]]
        B = A' @ X = [[2,1],[3,4]] @ [[1],[3]] = [[5],[15]]
        """
        a = np.array([[2.0, 3.0], [1.0, 4.0]], order='F', dtype=float)
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        b1 = np.array([[11.0], [13.0]], order='F', dtype=float)
        x1, scale1, iwarn1 = mb02uw(False, par, a, b1)
        assert iwarn1 == 0
        np.testing.assert_allclose(x1, [[1.0], [3.0]], rtol=1e-13)

        b2 = np.array([[5.0], [15.0]], order='F', dtype=float)
        x2, scale2, iwarn2 = mb02uw(True, par, a, b2)
        assert iwarn2 == 0
        np.testing.assert_allclose(x2, [[1.0], [3.0]], rtol=1e-13)


class TestMB02UWMultipleRHS:
    """Tests with multiple right-hand sides."""

    def test_2x2_multiple_rhs(self):
        """
        Test 2x2 system with M=3 right-hand sides.

        A = [[2, 0], [0, 3]]
        B = [[2, 4, 6], [3, 6, 9]]
        X = [[1, 2, 3], [1, 2, 3]]
        """
        a = np.array([[2.0, 0.0], [0.0, 3.0]], order='F', dtype=float)
        b = np.array([[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], order='F', dtype=float)
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(False, par, a, b)

        assert iwarn == 0
        assert scale == 1.0
        np.testing.assert_allclose(x, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], rtol=1e-14)


class TestMB02UWPerturbation:
    """Tests for near-singular matrix perturbation."""

    def test_near_singular_1x1(self):
        """
        Test 1x1 near-singular case: very small A triggers perturbation.

        A = [[1e-20]], which is less than SMIN=1e-10
        Should perturb A to SMIN and set iwarn=1.
        """
        a = np.array([[1e-20]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(False, par, a, b)

        assert iwarn == 1
        assert x[0, 0] == pytest.approx(1.0 / 1e-10, rel=1e-10)

    def test_near_singular_2x2(self):
        """
        Test 2x2 near-singular case: all elements very small.

        A = [[1e-20, 0], [0, 1e-20]], triggers perturbation.
        """
        a = np.array([[1e-20, 0.0], [0.0, 1e-20]], order='F', dtype=float)
        b = np.array([[1.0], [1.0]], order='F', dtype=float)
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(False, par, a, b)

        assert iwarn == 1


class TestMB02UWScaling:
    """Tests for overflow prevention scaling."""

    def test_scaling_large_b(self):
        """
        Test scaling when B is very large.

        When B is large and A is small, scale < 1 to prevent overflow.
        """
        a = np.array([[0.1]], order='F', dtype=float)
        b = np.array([[1e307]], order='F', dtype=float)
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(False, par, a, b)

        assert scale <= 1.0
        np.testing.assert_allclose(a[0, 0] * x[0, 0], scale * 1e307, rtol=1e-10)


class TestMB02UWMathematicalProperties:
    """
    Mathematical property tests to validate numerical correctness.
    """

    def test_residual_1x1(self):
        """
        Verify residual: A * X = scale * B for 1x1 system.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        a = np.random.randn(1, 1).astype(float, order='F')
        a[0, 0] = max(abs(a[0, 0]), 0.1)
        b = np.random.randn(1, 3).astype(float, order='F')
        b_orig = b.copy()
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(False, par, a, b)

        residual = a @ x - scale * b_orig
        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_residual_2x2(self):
        """
        Verify residual: A * X = scale * B for 2x2 system.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        a = np.random.randn(2, 2).astype(float, order='F')
        a += np.eye(2)
        b = np.random.randn(2, 2).astype(float, order='F')
        b_orig = b.copy()
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(False, par, a, b)

        residual = a @ x - scale * b_orig
        np.testing.assert_allclose(residual, 0.0, atol=1e-13)

    def test_residual_2x2_transpose(self):
        """
        Verify residual: A' * X = scale * B for 2x2 system.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        a = np.random.randn(2, 2).astype(float, order='F')
        a += np.eye(2)
        b = np.random.randn(2, 3).astype(float, order='F')
        b_orig = b.copy()
        par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

        x, scale, iwarn = mb02uw(True, par, a, b)

        residual = a.T @ x - scale * b_orig
        np.testing.assert_allclose(residual, 0.0, atol=1e-13)

    def test_scale_bound(self):
        """
        Verify scale is always <= 1.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        for _ in range(10):
            n = np.random.choice([1, 2])
            m = np.random.randint(1, 5)
            a = np.random.randn(n, n).astype(float, order='F')
            if n == 2:
                a += np.eye(2)
            else:
                a[0, 0] = max(abs(a[0, 0]), 0.1)
            b = np.random.randn(n, m).astype(float, order='F')
            par = np.array([2.22e-16, 2.23e-308, 1e-10], dtype=float)

            x, scale, iwarn = mb02uw(False, par, a, b)

            assert scale <= 1.0
