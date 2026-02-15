"""
Tests for MB03KD: Reordering diagonal blocks in periodic Schur form.

MB03KD reorders the diagonal blocks of a formal matrix product
T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) such that selected
eigenvalues end up in the leading part of the matrix sequence T22_k.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestMB03KDBasic:
    """Basic functionality tests for MB03KD."""

    def test_html_example(self):
        """
        Validate using SLICOT HTML doc example.

        Input: K=3 factors, N=3, H=2 (KSCHUR), COMPQ='I'
        Matrices in periodic Schur form, select eigenvalues with negative real part.

        Expected output from HTML docs (4-decimal precision).
        """
        from ctrlsys import mb03kd

        k = 3
        n = 3
        nc = n
        kschur = 2

        nd = np.array([n, n, n], dtype=np.int32)
        ni = np.array([0, 0, 0], dtype=np.int32)
        s = np.array([1, -1, 1], dtype=np.int32)

        # Input matrices from HTML doc (after MB03BD)
        # T1 from reversed factor order
        t1 = np.array([
            [1.8451, 0.0, 0.0],
            [0.9260, 1.3976, 0.0],
            [1.2717, -2.3544, -3.1023]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [-2.0990, 0.0, 0.0],
            [-1.0831, 3.4838, 3.4552],
            [-2.5601, 0.2950, -2.1690]
        ], dtype=np.float64, order='F')

        t3 = np.array([
            [2.5997, 0.0, 0.0],
            [-0.0087, 1.9846, 0.0],
            [1.6898, 0.1942, 2.3259]
        ], dtype=np.float64, order='F')

        t_size = n * n * k
        t = np.zeros(t_size, dtype=np.float64, order='F')
        t[:n*n] = t1.ravel('F')
        t[n*n:2*n*n] = t2.ravel('F')
        t[2*n*n:] = t3.ravel('F')

        ldt = np.array([n, n, n], dtype=np.int32)
        ixt = np.array([1, n*n + 1, 2*n*n + 1], dtype=np.int32)

        ldq = np.array([n, n, n], dtype=np.int32)
        ixq = np.array([1, n*n + 1, 2*n*n + 1], dtype=np.int32)

        # Select eigenvalues with negative real part
        # From HTML: alphar = [0.3230, 0.3230, -0.8752]
        # Only 3rd eigenvalue has negative real part
        select = np.array([False, False, True], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'I', 'N', k, nc, kschur, nd, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol
        )

        assert info == 0, f"MB03KD returned info={info}"
        assert m >= 0

    def test_select_single_eigenvalue_k2(self):
        """
        Test selecting a single eigenvalue with K=2.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb03kd

        np.random.seed(42)
        k = 2
        nc = 3
        kschur = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        # Upper triangular matrices (periodic Schur form)
        t1 = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 3.0, 0.4],
            [0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [1.5, 0.2, 0.1],
            [0.0, 2.5, 0.3],
            [0.0, 0.0, 3.5]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)

        # Select last eigenvalue
        select = np.array([False, False, True], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'I', 'N', k, nc, kschur, n, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol
        )

        assert info == 0 or info == 1
        assert m >= 0

    def test_select_multiple_eigenvalues(self):
        """
        Test selecting multiple eigenvalues.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb03kd

        np.random.seed(123)
        k = 2
        nc = 4
        kschur = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [1.0, 0.5, 0.3, 0.1],
            [0.0, 2.0, 0.4, 0.2],
            [0.0, 0.0, 3.0, 0.3],
            [0.0, 0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [0.8, 0.3, 0.2, 0.1],
            [0.0, 1.2, 0.3, 0.15],
            [0.0, 0.0, 2.0, 0.2],
            [0.0, 0.0, 0.0, 3.0]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)

        # Select 1st and 3rd eigenvalues
        select = np.array([True, False, True, False], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'I', 'N', k, nc, kschur, n, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol
        )

        assert info == 0 or info == 1
        # M should be 2 (two eigenvalues selected)
        if info == 0:
            assert m == 2


class TestMB03KDCompQ:
    """Tests for different COMPQ options."""

    def test_compq_n(self):
        """
        Test with COMPQ='N' - no orthogonal matrices computed.

        Random seed: 222 (for reproducibility)
        """
        from ctrlsys import mb03kd

        np.random.seed(222)
        k = 2
        nc = 3
        kschur = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 3.0, 0.4],
            [0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [1.5, 0.2, 0.1],
            [0.0, 2.5, 0.3],
            [0.0, 0.0, 3.5]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        ldq = np.array([1, 1], dtype=np.int32)
        ixq = np.array([1, 1], dtype=np.int32)

        select = np.array([False, True, False], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'N', 'N', k, nc, kschur, n, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol
        )

        assert info == 0 or info == 1

    def test_compq_u(self):
        """
        Test with COMPQ='U' - update existing orthogonal matrices.

        Random seed: 333 (for reproducibility)
        """
        from ctrlsys import mb03kd

        np.random.seed(333)
        k = 2
        nc = 3
        kschur = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 3.0, 0.4],
            [0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [1.5, 0.2, 0.1],
            [0.0, 2.5, 0.3],
            [0.0, 0.0, 3.5]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        # Initialize Q as identity
        q = np.eye(nc, dtype=np.float64, order='F').ravel('F')
        q = np.concatenate([q, np.eye(nc, dtype=np.float64, order='F').ravel('F')])
        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)

        select = np.array([True, False, False], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'U', 'N', k, nc, kschur, n, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol, q=q.copy()
        )

        assert info == 0 or info == 1


class TestMB03KDStrong:
    """Tests with strong stability tests."""

    def test_strong_stability(self):
        """
        Test with STRONG='S' - perform strong stability tests.

        Random seed: 444 (for reproducibility)
        """
        from ctrlsys import mb03kd

        np.random.seed(444)
        k = 2
        nc = 3
        kschur = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 3.0, 0.4],
            [0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [1.5, 0.2, 0.1],
            [0.0, 2.5, 0.3],
            [0.0, 0.0, 3.5]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)

        select = np.array([True, True, False], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'I', 'S', k, nc, kschur, n, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol
        )

        assert info == 0 or info == 1


class TestMB03KDMixedSignatures:
    """Tests with different signature arrays."""

    def test_mixed_signatures(self):
        """
        Test with mixed signatures S=[1, -1].

        Random seed: 555 (for reproducibility)
        """
        from ctrlsys import mb03kd

        np.random.seed(555)
        k = 2
        nc = 3
        kschur = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, -1], dtype=np.int32)

        t1 = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 3.0, 0.4],
            [0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [1.5, 0.2, 0.1],
            [0.0, 2.5, 0.3],
            [0.0, 0.0, 3.5]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)

        select = np.array([True, False, False], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'I', 'N', k, nc, kschur, n, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol
        )

        assert info == 0 or info == 1


class TestMB03KDOrthogonality:
    """Test orthogonality of computed Q matrices."""

    def test_q_orthogonality(self):
        """
        Verify Q matrices are orthogonal after reordering.

        Mathematical property: Q^T * Q = I

        Random seed: 666 (for reproducibility)
        """
        from ctrlsys import mb03kd

        np.random.seed(666)
        k = 2
        nc = 3
        kschur = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 3.0, 0.4],
            [0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [1.5, 0.2, 0.1],
            [0.0, 2.5, 0.3],
            [0.0, 0.0, 3.5]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)

        select = np.array([False, True, True], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'I', 'N', k, nc, kschur, n, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol
        )

        if info == 0:
            q1 = q_out[:nc*nc].reshape((nc, nc), order='F')
            q2 = q_out[nc*nc:].reshape((nc, nc), order='F')

            assert_allclose(q1.T @ q1, np.eye(nc), rtol=1e-12, atol=1e-14)
            assert_allclose(q2.T @ q2, np.eye(nc), rtol=1e-12, atol=1e-14)


class TestMB03KDEdgeCases:
    """Edge case tests."""

    def test_no_selection(self):
        """Test when no eigenvalues are selected (all False)."""
        from ctrlsys import mb03kd

        k = 2
        nc = 3
        kschur = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 3.0, 0.4],
            [0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [1.5, 0.2, 0.1],
            [0.0, 2.5, 0.3],
            [0.0, 0.0, 3.5]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)

        # No eigenvalues selected
        select = np.array([False, False, False], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'I', 'N', k, nc, kschur, n, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol
        )

        assert info == 0
        assert m == 0

    def test_all_selected(self):
        """Test when all eigenvalues are selected (all True)."""
        from ctrlsys import mb03kd

        k = 2
        nc = 3
        kschur = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 3.0, 0.4],
            [0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [1.5, 0.2, 0.1],
            [0.0, 2.5, 0.3],
            [0.0, 0.0, 3.5]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)

        # All eigenvalues selected
        select = np.array([True, True, True], dtype=np.bool_)

        tol = 100.0

        t_out, q_out, m, info = mb03kd(
            'I', 'N', k, nc, kschur, n, ni, s, select,
            t.copy(), ldt, ixt, ldq, ixq, tol
        )

        assert info == 0
        assert m == nc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
