"""
Tests for sb01dd - Eigenstructure assignment for multi-input system.

Computes feedback matrix G such that A - B*G has desired eigenvalues.
Input (A, B) must be in orthogonal canonical form (from ab01nd).
"""
import numpy as np
import pytest
from ctrlsys import ab01nd, sb01dd


class TestSB01DDBasic:
    """Basic tests from HTML documentation example."""

    def test_html_doc_example(self):
        """
        Test case from SLICOT HTML documentation.

        Expected output:
        G = [[-5.2339,  3.1725, -15.7885, 21.7043],
             [-1.6022,  0.8504,  -5.1914,  6.2339]]
        """
        n = 4
        m = 2
        tol = 0.0

        a = np.array([
            [-1.0,  0.0,  2.0, -3.0],
            [ 1.0, -4.0,  3.0, -1.0],
            [ 0.0,  2.0,  4.0, -5.0],
            [ 0.0,  0.0, -1.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        wr = np.array([-1.0, -1.0, -1.0, -1.0], dtype=float)
        wi = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        y = np.array([1.0, 2.0, 2.0, 1.0, -1.0, -2.0, 3.0, 1.0], dtype=float)

        a_can, b_can, ncont, indcon, nblk, z, tau, info1 = ab01nd(
            'I', a, b, tol
        )
        assert info1 == 0, f"ab01nd failed with info={info1}"
        assert ncont == n, "System should be fully controllable"

        a_schur, b_out, z_out, g, count, info2 = sb01dd(
            a_can, b_can, indcon, nblk[:indcon].copy(), wr, wi, z, y, tol=tol
        )
        assert info2 == 0, f"sb01dd failed with info={info2}"

        g_expected = np.array([
            [-5.2339,  3.1725, -15.7885, 21.7043],
            [-1.6022,  0.8504,  -5.1914,  6.2339]
        ], order='F', dtype=float)

        np.testing.assert_allclose(g, g_expected, rtol=1e-3, atol=1e-4)


class TestSB01DDProperties:
    """Mathematical property tests."""

    def test_schur_eigenvalues_real(self):
        """
        Validate eigenvalues of returned Schur form match desired poles.

        The Schur form a_schur has eigenvalues close to desired poles,
        even when A-B*G eigenvalue problem is ill-conditioned.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3
        m = 2

        a = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -2.0, -3.0]
        ], order='F', dtype=float)
        b = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        wr = np.array([-2.0, -3.0, -4.0], dtype=float)
        wi = np.zeros(n, dtype=float)
        y = np.random.randn(m * n)

        a_can, b_can, ncont, indcon, nblk, z, tau, info1 = ab01nd(
            'I', a, b, 0.0
        )
        assert info1 == 0

        if ncont < n:
            pytest.skip("System not fully controllable")

        a_schur, b_out, z_out, g, count, info2 = sb01dd(
            a_can, b_can, indcon, nblk[:indcon].copy(), wr, wi, z, y, tol=0.0
        )
        assert info2 == 0
        assert g.shape == (m, n)

        eigs_schur = np.linalg.eigvals(a_schur)
        desired = wr + 1j * wi

        for d in desired:
            closest = np.min(np.abs(eigs_schur - d))
            assert closest < 1e-10, f"Desired pole {d} not in Schur form. Closest: {closest}"

    def test_schur_eigenvalues_complex(self):
        """
        Validate Schur form eigenvalues with complex conjugate poles.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4
        m = 2

        a = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, -2.0, -3.0, -4.0]
        ], order='F', dtype=float)
        b = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        wr = np.array([-1.0, -1.0, -2.0, -2.0], dtype=float)
        wi = np.array([1.0, -1.0, 0.5, -0.5], dtype=float)
        y = np.random.randn(m * n)

        a_can, b_can, ncont, indcon, nblk, z, tau, info1 = ab01nd(
            'I', a, b, 0.0
        )
        assert info1 == 0

        if ncont < n:
            pytest.skip("System not fully controllable")

        a_schur, b_out, z_out, g, count, info2 = sb01dd(
            a_can, b_can, indcon, nblk[:indcon].copy(), wr, wi, z, y, tol=0.0
        )
        assert info2 == 0
        assert g.shape == (m, n)

        desired_poles = wr + 1j * wi
        eigs_schur = np.linalg.eigvals(a_schur)

        for desired in desired_poles:
            closest = np.min(np.abs(eigs_schur - desired))
            assert closest < 1e-10, f"Desired pole {desired} not in Schur form. Closest: {closest}"


class TestSB01DDEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_input_system(self):
        """
        Test single-input system (m=1).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 2
        m = 1

        a = np.array([
            [0.0, 1.0],
            [-2.0, -3.0]
        ], order='F', dtype=float)
        b = np.array([
            [0.0],
            [1.0]
        ], order='F', dtype=float)

        wr = np.array([-1.0, -2.0], dtype=float)
        wi = np.zeros(n, dtype=float)
        y = np.random.randn(m * n)

        a_can, b_can, ncont, indcon, nblk, z, tau, info1 = ab01nd(
            'I', a, b, 0.0
        )
        assert info1 == 0

        if ncont < n:
            pytest.skip("System not fully controllable")

        a_schur, b_out, z_out, g, count, info2 = sb01dd(
            a_can, b_can, indcon, nblk[:indcon].copy(), wr, wi, z, y, tol=0.0
        )
        assert info2 == 0
        assert g.shape == (m, n)


class TestSB01DDErrors:
    """Error handling tests."""

    def test_invalid_indcon(self):
        """Test error when indcon > n."""
        n = 2
        m = 1
        a = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)
        wr = np.zeros(n, dtype=float)
        wi = np.zeros(n, dtype=float)
        y = np.zeros(m * n, dtype=float)
        nblk = np.array([n], dtype=np.int32)

        a_schur, b_out, z_out, g, count, info = sb01dd(
            a, b, indcon=n+1, nblk=nblk, wr=wr, wi=wi, z=z, y=y, tol=0.0
        )
        assert info == -3

    def test_invalid_nblk_sum(self):
        """Test error when sum(nblk) != n."""
        n = 3
        m = 1
        a = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)
        wr = np.zeros(n, dtype=float)
        wi = np.zeros(n, dtype=float)
        y = np.zeros(m * n, dtype=float)
        nblk = np.array([1, 1], dtype=np.int32)

        a_schur, b_out, z_out, g, count, info = sb01dd(
            a, b, indcon=2, nblk=nblk, wr=wr, wi=wi, z=z, y=y, tol=0.0
        )
        assert info == -8
