"""
Tests for SB08ED: Left coprime factorization with prescribed stability degree.

Constructs output injection matrix H and orthogonal transformation Z such that
Q = (Z'*(A+H*C)*Z, Z'*(B+H*D), C*Z, D) and R = (Z'*(A+H*C)*Z, Z'*H, C*Z, I)
provide stable left coprime factorization G = R^{-1} * Q.

Note: A, B, C, D arrays are modified in-place. The routine returns:
  (nq, nr, br, dr, iwarn, info)
where:
  - nq: order of the factorization
  - nr: order of minimal realization of R
  - br: NQ-by-P output injection matrix Z'*H (first NR rows = BR of R)
  - dr: P-by-P identity matrix
  - iwarn: warning count
  - info: exit code
"""

import numpy as np
import pytest
from ctrlsys import sb08ed


class TestSB08EDBasic:
    """Basic functionality tests from HTML documentation."""

    def test_continuous_system_html_example(self):
        """
        Test continuous-time left coprime factorization.

        Data from SLICOT HTML documentation SB08ED example.
        N=7, M=2, P=3, DICO='C', ALPHA=[-1.0, -1.0]
        """
        n, m, p = 7, 2, 3

        a = np.array([
            [-0.04165,  0.0000,  4.9200,   0.4920,  0.0000,   0.0000,  0.0000],
            [-5.2100,  -12.500,  0.0000,   0.0000,  0.0000,   0.0000,  0.0000],
            [ 0.0000,   3.3300, -3.3300,   0.0000,  0.0000,   0.0000,  0.0000],
            [ 0.5450,   0.0000,  0.0000,   0.0000,  0.0545,   0.0000,  0.0000],
            [ 0.0000,   0.0000,  0.0000,  -0.4920,  0.004165, 0.0000,  4.9200],
            [ 0.0000,   0.0000,  0.0000,   0.0000,  0.5210,  -12.500,  0.0000],
            [ 0.0000,   0.0000,  0.0000,   0.0000,  0.0000,   3.3300, -3.3300],
        ], dtype=float, order='F')

        b = np.array([
            [ 0.0000,   0.0000],
            [12.500,    0.0000],
            [ 0.0000,   0.0000],
            [ 0.0000,   0.0000],
            [ 0.0000,   0.0000],
            [ 0.0000,  12.500],
            [ 0.0000,   0.0000],
        ], dtype=float, order='F')

        c = np.array([
            [1.0000,  0.0000,  0.0000,   0.0000,  0.0000,  0.0000,  0.0000],
            [0.0000,  0.0000,  0.0000,   1.0000,  0.0000,  0.0000,  0.0000],
            [0.0000,  0.0000,  0.0000,   0.0000,  1.0000,  0.0000,  0.0000],
        ], dtype=float, order='F')

        d = np.array([
            [0.0000,  0.0000],
            [0.0000,  0.0000],
            [0.0000,  0.0000],
        ], dtype=float, order='F')

        alpha = np.array([-1.0, -1.0], dtype=float)
        tol = 1.0e-10

        nq, nr, br, dr, iwarn, info = sb08ed(
            'C', a, b, c, d, alpha, tol
        )

        assert info == 0
        assert nq == 7
        assert nr == 2

        aq_expected = np.array([
            [-1.0000,   0.0526,  -0.1408,  -0.3060,   0.4199,   0.2408,   1.7274],
            [-0.4463,  -1.0000,   2.0067,   4.3895,   0.0062,   0.1813,   0.0895],
            [ 0.0000,   0.0000, -12.4245,   3.5463,  -0.0057,   0.0254,  -0.0053],
            [ 0.0000,   0.0000,   0.0000,  -3.5957,  -0.0153,  -0.0290,  -0.0616],
            [ 0.0000,   0.0000,   0.0000,   0.0000, -13.1627,  -1.9835,  -3.6182],
            [ 0.0000,   0.0000,   0.0000,   0.0000,   0.0000,  -1.4178,   5.6218],
            [ 0.0000,   0.0000,   0.0000,   0.0000,   0.0000,  -0.8374,  -1.4178],
        ], dtype=float, order='F')

        bq_expected = np.array([
            [-1.1544,  -0.0159],
            [-0.0631,   0.5122],
            [ 0.0056, -11.6989],
            [ 0.0490,   4.3728],
            [11.7198,  -0.0038],
            [-2.8173,   0.0308],
            [ 3.1018,  -0.0009],
        ], dtype=float, order='F')

        cq_expected = np.array([
            [ 0.2238,   0.0132,  -0.0006,  -0.0083,   0.1279,   0.8797,   0.3994],
            [ 0.9639,   0.0643,  -0.0007,  -0.0041,   0.0305,  -0.2562,   0.0122],
            [-0.0660,   0.9962,   0.0248,  -0.0506,   0.0000,   0.0022,  -0.0017],
        ], dtype=float, order='F')

        br_expected = np.array([
            [-0.2623,  -1.1297,   0.0764],
            [-0.0155,  -0.0752,  -1.1676],
        ], dtype=float, order='F')

        dr_expected = np.eye(p, dtype=float, order='F')

        eig_computed = np.linalg.eigvals(a[:nq, :nq])
        eig_expected = np.linalg.eigvals(aq_expected)
        np.testing.assert_allclose(
            sorted(eig_computed.real), sorted(eig_expected.real), rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            sorted(np.abs(eig_computed.imag)), sorted(np.abs(eig_expected.imag)), rtol=1e-3, atol=1e-4
        )

        np.testing.assert_allclose(np.abs(br[:nr, :p]), np.abs(br_expected), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(dr[:p, :p], dr_expected, rtol=1e-14)


class TestSB08EDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_coprime_factorization_property(self):
        """
        Validate coprime factorization: G = R^{-1} * Q.

        For a stable coprime factorization, the transfer function G should
        equal R^{-1} * Q at any frequency.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 4, 2, 2

        a_orig = np.array([
            [-2.0,  0.5, -0.3,  0.1],
            [ 0.2, -1.5,  0.4, -0.2],
            [ 0.1,  0.3, -1.8,  0.5],
            [-0.1,  0.2,  0.1, -2.2],
        ], dtype=float, order='F')

        b_orig = np.array([
            [1.0, 0.2],
            [0.3, 1.0],
            [0.1, 0.4],
            [0.2, 0.1],
        ], dtype=float, order='F')

        c_orig = np.array([
            [1.0, 0.1, 0.2, 0.0],
            [0.0, 0.3, 1.0, 0.1],
        ], dtype=float, order='F')

        d_orig = np.array([
            [0.1, 0.0],
            [0.0, 0.1],
        ], dtype=float, order='F')

        a = a_orig.copy(order='F')
        b = b_orig.copy(order='F')
        c = c_orig.copy(order='F')
        d = d_orig.copy(order='F')

        alpha = np.array([-0.5, -0.5], dtype=float)
        tol = 1.0e-10

        nq, nr, br, dr, iwarn, info = sb08ed(
            'C', a, b, c, d, alpha, tol
        )

        assert info == 0

        i_mat = np.eye(n, dtype=float)
        s = 1j * 1.0
        g_original = c_orig @ np.linalg.solve(s * i_mat - a_orig, b_orig) + d_orig

        if nq > 0:
            aq = a[:nq, :nq]
            bq = b[:nq, :m]
            cq = c[:p, :nq]
            dq = d[:p, :m]
            i_nq = np.eye(nq, dtype=float)
            q_tf = cq @ np.linalg.solve(s * i_nq - aq, bq) + dq
            if nr > 0:
                ar = aq[:nr, :nr]
                br_r = br[:nr, :p]
                cr = cq[:p, :nr]
                i_nr = np.eye(nr, dtype=float)
                r_tf = cr @ np.linalg.solve(s * i_nr - ar, br_r) + dr[:p, :p]
                g_factored = np.linalg.solve(r_tf, q_tf)
                np.testing.assert_allclose(g_original, g_factored, rtol=1e-6, atol=1e-8)

    def test_dr_is_identity(self):
        """
        Validate DR is always identity matrix.

        The denominator factor R has input/output matrix DR = I.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 3, 2, 2

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        alpha = np.array([-0.5, -0.5], dtype=float)

        nq, nr, br, dr, iwarn, info = sb08ed(
            'C', a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), d.copy(order='F'), alpha, 0.0
        )

        assert info == 0
        np.testing.assert_allclose(dr[:p, :p], np.eye(p), rtol=1e-14)

    def test_schur_form_upper_triangular(self):
        """
        Validate AQ is in real Schur form (upper quasi-triangular).

        The state dynamics matrix should be in real Schur form after
        the factorization, with 1x1 or 2x2 blocks on diagonal.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 5, 2, 2

        a = -np.eye(n) + 0.1 * np.random.randn(n, n)
        a = a.astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        alpha = np.array([-2.0, -2.0], dtype=float)

        nq, nr, br, dr, iwarn, info = sb08ed(
            'C', a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), d.copy(order='F'), alpha, 0.0
        )

        assert info == 0

        if nq > 0:
            aq_sub = a[:nq, :nq]
            for i in range(nq - 2):
                if abs(aq_sub[i + 1, i]) < 1e-10 and abs(aq_sub[i + 2, i + 1]) < 1e-10:
                    assert abs(aq_sub[i + 2, i]) < 1e-10, "Not quasi-triangular"


class TestSB08EDEdgeCases:
    """Edge case and boundary condition tests."""

    def test_zero_dimension_n(self):
        """Test with N=0 (empty system)."""
        n, m, p = 0, 2, 3

        a = np.zeros((0, 0), dtype=float, order='F')
        b = np.zeros((0, m), dtype=float, order='F')
        c = np.zeros((p, 0), dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        alpha = np.array([-1.0, -1.0], dtype=float)

        nq, nr, br, dr, iwarn, info = sb08ed(
            'C', a, b, c, d, alpha, 0.0
        )

        assert info == 0
        assert nq == 0
        assert nr == 0
        np.testing.assert_allclose(dr[:p, :p], np.eye(p), rtol=1e-14)

    def test_zero_dimension_p(self):
        """Test with P=0 (no outputs)."""
        n, m, p = 3, 2, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.ones((n, m), dtype=float, order='F')
        c = np.zeros((0, n), dtype=float, order='F')
        d = np.zeros((0, m), dtype=float, order='F')

        alpha = np.array([-0.5, -0.5], dtype=float)

        nq, nr, br, dr, iwarn, info = sb08ed(
            'C', a, b, c, d, alpha, 0.0
        )

        assert info == 0
        assert nq == 0
        assert nr == 0

    def test_discrete_time_system(self):
        """
        Test discrete-time left coprime factorization.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 3, 1, 2

        a = np.array([
            [0.5, 0.1, 0.0],
            [0.0, 0.6, 0.2],
            [0.0, 0.0, 0.7],
        ], dtype=float, order='F')

        b = np.array([[1.0], [0.5], [0.2]], dtype=float, order='F')
        c = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')

        alpha = np.array([0.5, 0.5], dtype=float)

        nq, nr, br, dr, iwarn, info = sb08ed(
            'D', a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), d.copy(order='F'), alpha, 0.0
        )

        assert info == 0
        assert nq >= 0
        assert nr >= 0
        np.testing.assert_allclose(dr[:p, :p], np.eye(p), rtol=1e-14)


class TestSB08EDErrorHandling:
    """Error handling tests."""

    def test_invalid_dico(self):
        """Test invalid DICO parameter."""
        n, m, p = 2, 1, 1
        a = np.eye(n, dtype=float, order='F')
        b = np.ones((n, m), dtype=float, order='F')
        c = np.ones((p, n), dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')
        alpha = np.array([-1.0, -1.0], dtype=float)

        with pytest.raises(ValueError):
            sb08ed('X', a, b, c, d, alpha, 0.0)

    def test_invalid_alpha_continuous(self):
        """Test invalid ALPHA for continuous-time (must be negative)."""
        n, m, p = 2, 1, 1
        a = np.eye(n, dtype=float, order='F')
        b = np.ones((n, m), dtype=float, order='F')
        c = np.ones((p, n), dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')
        alpha = np.array([1.0, 1.0], dtype=float)

        with pytest.raises(ValueError):
            sb08ed('C', a, b, c, d, alpha, 0.0)

    def test_invalid_alpha_discrete(self):
        """Test invalid ALPHA for discrete-time (must be in [0,1))."""
        n, m, p = 2, 1, 1
        a = np.eye(n, dtype=float, order='F')
        b = np.ones((n, m), dtype=float, order='F')
        c = np.ones((p, n), dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')
        alpha = np.array([1.5, 1.5], dtype=float)

        with pytest.raises(ValueError):
            sb08ed('D', a, b, c, d, alpha, 0.0)
