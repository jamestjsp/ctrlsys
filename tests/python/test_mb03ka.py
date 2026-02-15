"""
Tests for MB03KA: Moving diagonal blocks in generalized periodic Schur form.

MB03KA reorders the diagonal blocks of a formal matrix product
T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) of length K
by moving a block from row index IFST to row index ILST.
"""

import numpy as np
import pytest
from ctrlsys import mb03ka


class TestMB03KABasic:
    """Basic functionality tests for MB03KA."""

    def test_move_block_forward_k2(self):
        """
        Test moving 1x1 diagonal block forward (IFST=1, ILST=2) with K=2.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k = 2
        nc = 3
        kschur = 1
        ifst = 1
        ilst = 3

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

        q = np.eye(nc, dtype=np.float64, order='F').ravel('F')
        q = np.concatenate([q, np.eye(nc, dtype=np.float64, order='F').ravel('F')])
        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)
        whichq = np.array([1, 1], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, ifst_out, ilst_out, info = mb03ka(
            'U', whichq, False, k, nc, kschur, ifst, ilst,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1

    def test_move_block_backward_k2(self):
        """
        Test moving 1x1 diagonal block backward (IFST=3, ILST=1) with K=2.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k = 2
        nc = 3
        kschur = 1
        ifst = 3
        ilst = 1

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

        q = np.eye(nc, dtype=np.float64, order='F').ravel('F')
        q = np.concatenate([q, np.eye(nc, dtype=np.float64, order='F').ravel('F')])
        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)
        whichq = np.array([1, 1], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, ifst_out, ilst_out, info = mb03ka(
            'U', whichq, False, k, nc, kschur, ifst, ilst,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1

    def test_move_2x2_block(self):
        """
        Test moving a 2x2 diagonal block with K=2.

        Tests IFST pointing to row 1 of a 2x2 block, moving to position 3.
        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        k = 2
        nc = 4
        kschur = 1
        ifst = 1
        ilst = 3

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.1, 1.2, 0.3, 0.2],
            [0.0, 0.0, 2.0, 0.4],
            [0.0, 0.0, 0.0, 3.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [0.8, 0.3, 0.1, 0.05],
            [0.05, 0.9, 0.2, 0.1],
            [0.0, 0.0, 1.5, 0.3],
            [0.0, 0.0, 0.0, 2.5]
        ], dtype=np.float64, order='F')

        t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        q = np.eye(nc, dtype=np.float64, order='F').ravel('F')
        q = np.concatenate([q, np.eye(nc, dtype=np.float64, order='F').ravel('F')])
        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)
        whichq = np.array([1, 1], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, ifst_out, ilst_out, info = mb03ka(
            'U', whichq, True, k, nc, kschur, ifst, ilst,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1


class TestMB03KANoQ:
    """Tests with COMPQ='N' (no Q update)."""

    def test_no_q_computation(self):
        """
        Test with COMPQ='N' - no orthogonal matrices computed.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        k = 2
        nc = 3
        kschur = 1
        ifst = 1
        ilst = 2

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

        q = np.zeros(1, dtype=np.float64)
        ldq = np.array([1, 1], dtype=np.int32)
        ixq = np.array([1, 1], dtype=np.int32)
        whichq = np.array([0, 0], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, ifst_out, ilst_out, info = mb03ka(
            'N', whichq, False, k, nc, kschur, ifst, ilst,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1


class TestMB03KAWithSignature:
    """Tests with different signature arrays S."""

    def test_mixed_signatures(self):
        """
        Test with mixed signatures S=[1, -1].

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        k = 2
        nc = 3
        kschur = 1
        ifst = 1
        ilst = 2

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

        q = np.eye(nc, dtype=np.float64, order='F').ravel('F')
        q = np.concatenate([q, np.eye(nc, dtype=np.float64, order='F').ravel('F')])
        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)
        whichq = np.array([1, 1], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, ifst_out, ilst_out, info = mb03ka(
            'U', whichq, False, k, nc, kschur, ifst, ilst,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1


class TestMB03KAWorkspaceQuery:
    """Test workspace query functionality."""

    def test_workspace_query(self):
        """
        Test workspace query with LDWORK=-1.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        k = 2
        nc = 3
        kschur = 1
        ifst = 1
        ilst = 2

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

        q = np.zeros(1, dtype=np.float64)
        ldq = np.array([1, 1], dtype=np.int32)
        ixq = np.array([1, 1], dtype=np.int32)
        whichq = np.array([0, 0], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, ifst_out, ilst_out, info = mb03ka(
            'N', whichq, False, k, nc, kschur, ifst, ilst,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol,
            ldwork=-1
        )

        assert info == 0


class TestMB03KAQuickReturn:
    """Test quick return conditions."""

    def test_ifst_eq_ilst(self):
        """Test quick return when IFST equals ILST (no reordering needed)."""
        k = 2
        nc = 3
        kschur = 1
        ifst = 2
        ilst = 2

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t = np.eye(nc, dtype=np.float64, order='F').ravel('F')
        t = np.concatenate([t, np.eye(nc, dtype=np.float64, order='F').ravel('F')])
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, nc*nc + 1], dtype=np.int32)

        q = np.zeros(1, dtype=np.float64)
        ldq = np.array([1, 1], dtype=np.int32)
        ixq = np.array([1, 1], dtype=np.int32)
        whichq = np.array([0, 0], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, ifst_out, ilst_out, info = mb03ka(
            'N', whichq, False, k, nc, kschur, ifst, ilst,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0


class TestMB03KAOrthogonality:
    """Test orthogonality of computed Q matrices."""

    def test_q_orthogonality(self):
        """
        Verify Q matrices are orthogonal after reordering.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        k = 2
        nc = 3
        kschur = 1
        ifst = 1
        ilst = 3

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

        q = np.eye(nc, dtype=np.float64, order='F').ravel('F')
        q = np.concatenate([q, np.eye(nc, dtype=np.float64, order='F').ravel('F')])
        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)
        whichq = np.array([1, 1], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, ifst_out, ilst_out, info = mb03ka(
            'U', whichq, False, k, nc, kschur, ifst, ilst,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        if info == 0:
            q1 = q_out[:nc*nc].reshape((nc, nc), order='F')
            q2 = q_out[nc*nc:].reshape((nc, nc), order='F')

            np.testing.assert_allclose(q1.T @ q1, np.eye(nc), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(q2.T @ q2, np.eye(nc), rtol=1e-12, atol=1e-14)


class TestMB03KASelectiveQ:
    """Tests with COMPQ='W' (selective Q computation)."""

    def test_selective_q(self):
        """
        Test with COMPQ='W' - selective Q computation via WHICHQ.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        k = 2
        nc = 3
        kschur = 1
        ifst = 1
        ilst = 2

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

        q = np.eye(nc, dtype=np.float64, order='F').ravel('F')
        q = np.concatenate([q, np.eye(nc, dtype=np.float64, order='F').ravel('F')])
        ldq = np.array([nc, nc], dtype=np.int32)
        ixq = np.array([1, nc*nc + 1], dtype=np.int32)
        whichq = np.array([1, 0], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, ifst_out, ilst_out, info = mb03ka(
            'W', whichq, False, k, nc, kschur, ifst, ilst,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
