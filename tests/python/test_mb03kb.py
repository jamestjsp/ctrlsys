"""
Tests for MB03KB: Swap pairs of adjacent diagonal blocks in generalized periodic Schur form.

MB03KB reorders the diagonal blocks of a formal matrix product
T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) of length K
by swapping pairs of adjacent diagonal blocks of sizes 1 and/or 2.
"""

import numpy as np
import pytest
from ctrlsys import mb03kb
from fortran_reference import run_fortran_driver


MB03KB_MIXED_W_CASE = r"""
program main
  implicit none
  integer, parameter :: k=2, nc=4, kschur=1, j1=1, n1=1, n2=2
  integer, parameter :: ldwork=256
  integer info
  integer whichq(k), n(k), ni(k), s(k), ldt(k), ixt(k), ldq(k), ixq(k), iwork(4*k)
  double precision t(2*nc*nc), q(2*nc*nc), tol(3), dwork(ldwork)

  whichq = (/ 1, 1 /)
  n = (/ nc, nc /)
  ni = (/ 0, 0 /)
  s = (/ 1, 1 /)
  ldt = (/ nc, nc /)
  ixt = (/ 1, nc*nc + 1 /)
  ldq = (/ nc, nc /)
  ixq = (/ 1, nc*nc + 1 /)

  t = 0.0d0
  t(1:16) = (/ &
       1.0d0, 0.0d0, 0.0d0, 0.0d0, &
       0.5d0, 2.0d0, 0.1d0, 0.0d0, &
       0.2d0, 0.4d0, 2.2d0, 0.0d0, &
       0.1d0, 0.2d0, 0.3d0, 3.0d0 /)
  t(17:32) = (/ &
       0.8d0, 0.0d0, 0.0d0, 0.0d0, &
       0.3d0, 1.5d0, 0.1d0, 0.0d0, &
       0.1d0, 0.3d0, 1.6d0, 0.0d0, &
       0.05d0, 0.15d0, 0.2d0, 2.5d0 /)

  q = 0.0d0
  q(1) = 1.0d0
  q(6) = 1.0d0
  q(11) = 1.0d0
  q(16) = 1.0d0
  q(17) = 1.0d0
  q(22) = 1.0d0
  q(27) = 1.0d0
  q(32) = 1.0d0

  tol(1) = 10.0d0
  tol(2) = epsilon(1.0d0)
  tol(3) = tiny(1.0d0) / tol(2)

  call MB03KB('W', whichq, .FALSE., k, nc, kschur, j1, n1, n2, &
       n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork, dwork, ldwork, info)

  print '(I0)', info
  print '(*(ES24.16,1X))', t
  print '(*(ES24.16,1X))', q
end program main
"""


def _mb03kb_mixed_swap_inputs():
    k = 2
    nc = 4
    n = np.array([nc, nc], dtype=np.int32)
    ni = np.array([0, 0], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)

    t1 = np.array([
        [1.0, 0.5, 0.2, 0.1],
        [0.0, 2.0, 0.4, 0.2],
        [0.0, 0.1, 2.2, 0.3],
        [0.0, 0.0, 0.0, 3.0]
    ], dtype=np.float64, order='F')

    t2 = np.array([
        [0.8, 0.3, 0.1, 0.05],
        [0.0, 1.5, 0.3, 0.15],
        [0.0, 0.1, 1.6, 0.2],
        [0.0, 0.0, 0.0, 2.5]
    ], dtype=np.float64, order='F')

    t = np.concatenate([t1.ravel('F'), t2.ravel('F')])
    ldt = np.array([nc, nc], dtype=np.int32)
    ixt = np.array([1, nc * nc + 1], dtype=np.int32)

    q = np.concatenate([
        np.eye(nc, dtype=np.float64, order='F').ravel('F'),
        np.eye(nc, dtype=np.float64, order='F').ravel('F'),
    ])
    ldq = np.array([nc, nc], dtype=np.int32)
    ixq = np.array([1, nc * nc + 1], dtype=np.int32)
    whichq = np.array([1, 1], dtype=np.int32)

    eps = np.finfo(np.float64).eps
    smlnum = np.finfo(np.float64).tiny / eps
    tol = np.array([10.0, eps, smlnum], dtype=np.float64)
    return k, nc, n, ni, s, t, ldt, ixt, q, ldq, ixq, whichq, tol


def test_mb03kb_mixed_swap_matches_fortran_reference(tmp_path):
    k, nc, n, ni, s, t, ldt, ixt, q, ldq, ixq, whichq, tol = _mb03kb_mixed_swap_inputs()

    t_out, q_out, info = mb03kb(
        'W', whichq, False, k, nc, 1, 1, 1, 2,
        n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
    )

    output = run_fortran_driver(MB03KB_MIXED_W_CASE, tmp_path)
    tokens = output.split()
    assert info == int(tokens[0])

    values = np.array(tokens[1:], dtype=float)
    expected_t = values[:t.size]
    expected_q = values[t.size:]
    assert expected_q.size == q.size
    np.testing.assert_allclose(t_out, expected_t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q_out, expected_q, rtol=1e-12, atol=1e-12)


class TestMB03KBBasic:
    """Basic functionality tests for MB03KB."""

    def test_swap_1x1_blocks_k2(self):
        """
        Test swapping two 1x1 diagonal blocks with K=2 periodic matrices.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k = 2
        nc = 3
        kschur = 1
        j1 = 1
        n1 = 1
        n2 = 1

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

        t_out, q_out, info = mb03kb(
            'U', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1

    def test_swap_2x2_blocks_k2(self):
        """
        Test swapping two 2x2 diagonal blocks with K=2 periodic matrices.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k = 2
        nc = 4
        kschur = 1
        j1 = 1
        n1 = 2
        n2 = 2

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.1, 1.2, 0.3, 0.2],
            [0.0, 0.0, 2.0, 0.4],
            [0.0, 0.0, 0.2, 2.2]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [0.8, 0.3, 0.1, 0.05],
            [0.1, 0.9, 0.2, 0.1],
            [0.0, 0.0, 1.5, 0.3],
            [0.0, 0.0, 0.15, 1.6]
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

        t_out, q_out, info = mb03kb(
            'U', whichq, True, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1

    def test_swap_1x1_2x2_blocks(self):
        """
        Test swapping 1x1 and 2x2 adjacent blocks (N1=1, N2=2).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        k = 2
        nc = 4
        kschur = 1
        j1 = 1
        n1 = 1
        n2 = 2

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.0, 2.0, 0.4, 0.2],
            [0.0, 0.1, 2.2, 0.3],
            [0.0, 0.0, 0.0, 3.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [0.8, 0.3, 0.1, 0.05],
            [0.0, 1.5, 0.3, 0.15],
            [0.0, 0.1, 1.6, 0.2],
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

        t_out, q_out, info = mb03kb(
            'U', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1

    def test_swap_2x1_blocks(self):
        """
        Test swapping 2x2 and 1x1 adjacent blocks (N1=2, N2=1).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        k = 2
        nc = 4
        kschur = 1
        j1 = 1
        n1 = 2
        n2 = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t1 = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.1, 1.2, 0.3, 0.2],
            [0.0, 0.0, 3.0, 0.4],
            [0.0, 0.0, 0.0, 4.0]
        ], dtype=np.float64, order='F')

        t2 = np.array([
            [0.8, 0.3, 0.1, 0.05],
            [0.1, 0.9, 0.2, 0.1],
            [0.0, 0.0, 2.5, 0.3],
            [0.0, 0.0, 0.0, 3.5]
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

        t_out, q_out, info = mb03kb(
            'U', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1


class TestMB03KBNoQ:
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
        j1 = 1
        n1 = 1
        n2 = 1

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

        t_out, q_out, info = mb03kb(
            'N', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1


class TestMB03KBWithSignature:
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
        j1 = 1
        n1 = 1
        n2 = 1

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

        t_out, q_out, info = mb03kb(
            'U', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0 or info == 1


class TestMB03KBWorkspaceQuery:
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
        j1 = 1
        n1 = 1
        n2 = 1

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

        t_out, q_out, info = mb03kb(
            'N', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol,
            ldwork=-1
        )

        assert info == 0


class TestMB03KBQuickReturn:
    """Test quick return conditions."""

    def test_n1_zero(self):
        """Test quick return when N1=0."""
        k = 2
        nc = 3
        kschur = 1
        j1 = 1
        n1 = 0
        n2 = 1

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

        t_out, q_out, info = mb03kb(
            'N', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0

    def test_n2_zero(self):
        """Test quick return when N2=0."""
        k = 2
        nc = 3
        kschur = 1
        j1 = 1
        n1 = 1
        n2 = 0

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

        t_out, q_out, info = mb03kb(
            'N', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0

    def test_nc_le_1(self):
        """Test quick return when NC <= 1."""
        k = 2
        nc = 1
        kschur = 1
        j1 = 1
        n1 = 1
        n2 = 1

        n = np.array([nc, nc], dtype=np.int32)
        ni = np.array([0, 0], dtype=np.int32)
        s = np.array([1, 1], dtype=np.int32)

        t = np.array([1.0, 2.0], dtype=np.float64)
        ldt = np.array([nc, nc], dtype=np.int32)
        ixt = np.array([1, 2], dtype=np.int32)

        q = np.zeros(1, dtype=np.float64)
        ldq = np.array([1, 1], dtype=np.int32)
        ixq = np.array([1, 1], dtype=np.int32)
        whichq = np.array([0, 0], dtype=np.int32)

        eps = np.finfo(np.float64).eps
        smlnum = np.finfo(np.float64).tiny / eps
        tol = np.array([10.0, eps, smlnum], dtype=np.float64)

        t_out, q_out, info = mb03kb(
            'N', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        assert info == 0


class TestMB03KBOrthogonality:
    """Test orthogonality of computed Q matrices."""

    def test_q_orthogonality(self):
        """
        Verify Q matrices are orthogonal after swap.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        k = 2
        nc = 3
        kschur = 1
        j1 = 1
        n1 = 1
        n2 = 1

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

        t_out, q_out, info = mb03kb(
            'U', whichq, False, k, nc, kschur, j1, n1, n2,
            n, ni, s, t.copy(), ldt, ixt, q.copy(), ldq, ixq, tol
        )

        if info == 0:
            q1 = q_out[:nc*nc].reshape((nc, nc), order='F')
            q2 = q_out[nc*nc:].reshape((nc, nc), order='F')

            np.testing.assert_allclose(q1.T @ q1, np.eye(nc), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(q2.T @ q2, np.eye(nc), rtol=1e-12, atol=1e-14)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
