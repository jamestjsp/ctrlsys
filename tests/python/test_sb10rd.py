"""
Tests for SB10RD - H-infinity controller from F and H matrices.
"""
import numpy as np
import pytest
import slicot


class TestSB10RDBasic:
    """Basic functionality tests for SB10RD."""

    def test_basic(self):
        """Test with simple 2x2 system."""
        n, m, np_, ncon, nmeas = 2, 3, 3, 1, 1
        gamma = 10.0

        # a: (n, n) = (2, 2)
        a = np.array([[0.5, 0.1], [0.0, 0.4]], dtype=np.float64, order='F')
        # b: (n, m) = (2, 3)
        b = np.array([
            [0.3, 0.2, 1.0],
            [0.1, 0.5, 0.0]
        ], dtype=np.float64, order='F')
        # c: (np, n) = (3, 2)
        c = np.array([
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5]
        ], dtype=np.float64, order='F')
        # d: (np, m) = (3, 3)
        d = np.array([
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.1],
            [0.1, 0.1, 0.0]
        ], dtype=np.float64, order='F')

        # f: (m, n) = (3, 2)
        f = np.array([
            [-0.1, 0.0],
            [0.0, -0.1],
            [0.0, -0.2]
        ], dtype=np.float64, order='F')
        # h: (n, np) = (2, 3)
        h = np.array([
            [-0.1, 0.0, -0.05],
            [0.0, -0.1, -0.05]
        ], dtype=np.float64, order='F')

        tu = np.array([[1.0]], dtype=np.float64, order='F')
        ty = np.array([[1.0]], dtype=np.float64, order='F')

        x = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64, order='F')
        y = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64, order='F')

        ak, bk, ck, dk, info = slicot.sb10rd(
            n, m, np_, ncon, nmeas, gamma,
            a, b, c, d, f, h, tu, ty, x, y
        )

        assert info == 0, f"SB10RD returned info = {info}"
        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)


class TestSB10RDEdgeCases:
    """Edge case tests for SB10RD."""

    def test_larger_system(self):
        """Test with a 4x4 system."""
        n, m, np_, ncon, nmeas = 4, 6, 6, 2, 2
        gamma = 5.0

        a = np.eye(n, dtype=np.float64, order='F') * 0.5
        b = np.random.rand(n, m).astype(np.float64, order='F') * 0.1
        c = np.random.rand(np_, n).astype(np.float64, order='F') * 0.1
        d = np.zeros((np_, m), dtype=np.float64, order='F')
        d[np_ - nmeas:, m - ncon:] = 0.01

        f = np.random.rand(m, n).astype(np.float64, order='F') * 0.1
        h = np.random.rand(n, np_).astype(np.float64, order='F') * 0.1

        tu = np.eye(ncon, dtype=np.float64, order='F')
        ty = np.eye(nmeas, dtype=np.float64, order='F')

        x = np.eye(n, dtype=np.float64, order='F') * 0.1
        y = np.eye(n, dtype=np.float64, order='F') * 0.1

        ak, bk, ck, dk, info = slicot.sb10rd(
            n, m, np_, ncon, nmeas, gamma,
            a, b, c, d, f, h, tu, ty, x, y
        )

        assert info == 0, f"SB10RD returned info = {info}"
        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)
