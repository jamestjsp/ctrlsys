"""
Tests for SB10VD - State feedback and output injection for H2 controller.
"""
import numpy as np
import pytest
import ctrlsys


class TestSB10VDBasic:
    """Basic functionality tests for SB10VD."""

    def test_basic(self):
        """Test basic SB10VD functionality."""
        n, m, np_, ncon, nmeas = 6, 5, 5, 2, 2

        a = np.array([
            [-1.0, -2.0, -6.0, -8.0,  2.0,  3.0],
            [ 0.0,  4.0,  9.0,  4.0,  5.0, -5.0],
            [ 4.0, -7.0, -5.0,  7.0,  8.0,  8.0],
            [ 5.0, -2.0,  0.0, -1.0, -9.0,  0.0],
            [-3.0,  0.0,  2.0, -3.0,  1.0,  2.0],
            [-2.0,  3.0, -1.0,  0.0, -4.0, -6.0]
        ], dtype=np.float64, order='F')

        b = np.array([
            [-3.0,  2.0, -5.0,  4.0, -3.0,  1.0],
            [-4.0,  0.0, -7.0, -6.0,  9.0, -2.0],
            [-2.0,  1.0,  0.0,  1.0, -8.0,  3.0],
            [ 1.0, -5.0,  7.0,  1.0,  0.0, -6.0],
            [ 0.0,  2.0, -2.0, -2.0,  5.0, -2.0]
        ], dtype=np.float64, order='F').T

        c = np.array([
            [ 1.0, -3.0, -7.0,  9.0,  0.0],
            [-1.0,  0.0,  5.0, -3.0,  1.0],
            [ 2.0,  5.0,  0.0,  4.0, -2.0],
            [-4.0, -1.0, -8.0,  0.0,  1.0],
            [ 0.0,  1.0,  2.0,  3.0, -6.0],
            [-3.0,  1.0, -2.0,  7.0, -2.0]
        ], dtype=np.float64, order='F').T

        f, h, x, y, xcond, ycond, info = slicot.sb10vd(ncon, nmeas, a, b, c)

        assert info == 0, f"SB10VD returned info = {info}"
        assert f.shape == (ncon, n)
        assert h.shape == (n, nmeas)
        assert x.shape == (n, n)
        assert y.shape == (n, n)
        assert xcond >= 0
        assert ycond >= 0

        m1 = m - ncon
        np1 = np_ - nmeas
        nd1 = np1 - ncon
        nd2 = m1 - nmeas
        b1 = b[:, :m1]
        b2 = b[:, m1:]
        c1 = c[:np1, :]
        c2 = c[np1:, :]

        # Ax = A - B2 * C1[nd1:np1, :] (D12'*C1 = last ncon rows of C1)
        ax = a - b2 @ c1[nd1:, :]
        # Cx = C1[:nd1,:]' * C1[:nd1,:] (if nd1 > 0)
        cx = c1[:nd1, :].T @ c1[:nd1, :] if nd1 > 0 else np.zeros((n, n))
        dx = b2 @ b2.T
        res_x = ax.T @ x + x @ ax + cx - x @ dx @ x
        assert np.linalg.norm(res_x) / max(np.linalg.norm(x), 1.0) < 1e-6

        # Ay = A - B1[:, nd2:m1] * C2 (B1*D21' = last nmeas columns of B1)
        ay = a - b1[:, nd2:] @ c2
        cy = b1[:, :nd2] @ b1[:, :nd2].T if nd2 > 0 else np.zeros((n, n))
        dy = c2.T @ c2
        res_y = ay @ y + y @ ay.T + cy - y @ dy @ y
        assert np.linalg.norm(res_y) / max(np.linalg.norm(y), 1.0) < 1e-6

        # F = -D12'*C1 - B2'*X = -C1[nd1:,:] - B2'*X
        np.testing.assert_allclose(f, -c1[nd1:, :] - b2.T @ x, atol=1e-8)
        # H = -B1*D21' - Y*C2' = -B1[:,nd2:] - Y*C2'
        np.testing.assert_allclose(h, -b1[:, nd2:] - y @ c2.T, atol=1e-8)


class TestSB10VDStability:
    """Verify stability of feedback/injection."""

    def test_a_plus_b2f_stable(self):
        """Verify A + B2*F has eigenvalues with negative real parts."""
        n, m, np_, ncon, nmeas = 6, 5, 5, 2, 2
        m1 = m - ncon

        a = np.array([
            [-1.0, -2.0, -6.0, -8.0,  2.0,  3.0],
            [ 0.0,  4.0,  9.0,  4.0,  5.0, -5.0],
            [ 4.0, -7.0, -5.0,  7.0,  8.0,  8.0],
            [ 5.0, -2.0,  0.0, -1.0, -9.0,  0.0],
            [-3.0,  0.0,  2.0, -3.0,  1.0,  2.0],
            [-2.0,  3.0, -1.0,  0.0, -4.0, -6.0]
        ], dtype=np.float64, order='F')

        b = np.array([
            [-3.0,  2.0, -5.0,  4.0, -3.0,  1.0],
            [-4.0,  0.0, -7.0, -6.0,  9.0, -2.0],
            [-2.0,  1.0,  0.0,  1.0, -8.0,  3.0],
            [ 1.0, -5.0,  7.0,  1.0,  0.0, -6.0],
            [ 0.0,  2.0, -2.0, -2.0,  5.0, -2.0]
        ], dtype=np.float64, order='F').T

        c = np.array([
            [ 1.0, -3.0, -7.0,  9.0,  0.0],
            [-1.0,  0.0,  5.0, -3.0,  1.0],
            [ 2.0,  5.0,  0.0,  4.0, -2.0],
            [-4.0, -1.0, -8.0,  0.0,  1.0],
            [ 0.0,  1.0,  2.0,  3.0, -6.0],
            [-3.0,  1.0, -2.0,  7.0, -2.0]
        ], dtype=np.float64, order='F').T

        f, h, x, y, xcond, ycond, info = slicot.sb10vd(ncon, nmeas, a, b, c)

        assert info == 0

        b2 = b[:, m1:]
        acl_f = a + b2 @ f
        eig_f = np.linalg.eigvals(acl_f)
        assert np.all(eig_f.real < 0), f"A+B2*F not stable: {eig_f}"

    def test_a_plus_hc2_stable(self):
        """Verify A + H*C2 has eigenvalues with negative real parts."""
        n, m, np_, ncon, nmeas = 6, 5, 5, 2, 2
        np1 = np_ - nmeas

        a = np.array([
            [-1.0, -2.0, -6.0, -8.0,  2.0,  3.0],
            [ 0.0,  4.0,  9.0,  4.0,  5.0, -5.0],
            [ 4.0, -7.0, -5.0,  7.0,  8.0,  8.0],
            [ 5.0, -2.0,  0.0, -1.0, -9.0,  0.0],
            [-3.0,  0.0,  2.0, -3.0,  1.0,  2.0],
            [-2.0,  3.0, -1.0,  0.0, -4.0, -6.0]
        ], dtype=np.float64, order='F')

        b = np.array([
            [-3.0,  2.0, -5.0,  4.0, -3.0,  1.0],
            [-4.0,  0.0, -7.0, -6.0,  9.0, -2.0],
            [-2.0,  1.0,  0.0,  1.0, -8.0,  3.0],
            [ 1.0, -5.0,  7.0,  1.0,  0.0, -6.0],
            [ 0.0,  2.0, -2.0, -2.0,  5.0, -2.0]
        ], dtype=np.float64, order='F').T

        c = np.array([
            [ 1.0, -3.0, -7.0,  9.0,  0.0],
            [-1.0,  0.0,  5.0, -3.0,  1.0],
            [ 2.0,  5.0,  0.0,  4.0, -2.0],
            [-4.0, -1.0, -8.0,  0.0,  1.0],
            [ 0.0,  1.0,  2.0,  3.0, -6.0],
            [-3.0,  1.0, -2.0,  7.0, -2.0]
        ], dtype=np.float64, order='F').T

        f, h, x, y, xcond, ycond, info = slicot.sb10vd(ncon, nmeas, a, b, c)

        assert info == 0

        c2 = c[np1:, :]
        acl_h = a + h @ c2
        eig_h = np.linalg.eigvals(acl_h)
        assert np.all(eig_h.real < 0), f"A+H*C2 not stable: {eig_h}"


class TestSB10VDEdgeCases:
    """Edge case tests for SB10VD."""

    def test_small_system(self):
        """Test with a 2x2 stable system."""
        n, m, np_, ncon, nmeas = 2, 2, 2, 1, 1

        a = np.array([[-1.0, 0.5], [0.0, -2.0]], dtype=np.float64, order='F')
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64, order='F')
        c = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64, order='F')

        f, h, x, y, xcond, ycond, info = slicot.sb10vd(ncon, nmeas, a, b, c)

        assert info == 0, f"SB10VD returned info = {info}"
        assert f.shape == (ncon, n)
        assert h.shape == (n, nmeas)

        m1 = m - ncon
        np1 = np_ - nmeas
        nd1 = np1 - ncon
        nd2 = m1 - nmeas
        b1 = b[:, :m1]
        b2 = b[:, m1:]
        c1 = c[:np1, :]
        c2 = c[np1:, :]

        ax = a - b2 @ c1[nd1:, :]
        cx = c1[:nd1, :].T @ c1[:nd1, :] if nd1 > 0 else np.zeros((n, n))
        dx = b2 @ b2.T
        res_x = ax.T @ x + x @ ax + cx - x @ dx @ x
        assert np.linalg.norm(res_x) / max(np.linalg.norm(x), 1.0) < 1e-6

        ay = a - b1[:, nd2:] @ c2
        cy = b1[:, :nd2] @ b1[:, :nd2].T if nd2 > 0 else np.zeros((n, n))
        dy = c2.T @ c2
        res_y = ay @ y + y @ ay.T + cy - y @ dy @ y
        assert np.linalg.norm(res_y) / max(np.linalg.norm(y), 1.0) < 1e-6

    def test_mimo_ncon2_nmeas2(self):
        """Test MIMO system with ncon=2, nmeas=2."""
        n = 4
        m = 4
        np_ = 4
        ncon = 2
        nmeas = 2
        m1 = m - ncon
        np1 = np_ - nmeas

        a = np.array([[-1, 0.1, 0, 0],
                       [0, -2, 0.1, 0],
                       [0, 0, -3, 0.1],
                       [0, 0, 0, -4]], dtype=np.float64, order='F')

        b = np.array([[1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.float64, order='F')

        c = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.float64, order='F')

        f, h, x, y, xcond, ycond, info = slicot.sb10vd(ncon, nmeas, a, b, c)

        assert info == 0
        assert f.shape == (ncon, n)
        assert h.shape == (n, nmeas)

        b2 = b[:, m1:]
        eig_f = np.linalg.eigvals(a + b2 @ f)
        assert np.all(eig_f.real < 0), f"A+B2*F not stable: {eig_f}"

        c2 = c[np1:, :]
        eig_h = np.linalg.eigvals(a + h @ c2)
        assert np.all(eig_h.real < 0), f"A+H*C2 not stable: {eig_h}"
