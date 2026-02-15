"""
Tests for SB10UD - SVD transformation for H2 controller normalization

Tests normalization of D12 and D21 matrices to unit diagonal form.
"""

import numpy as np
import pytest

import ctrlsys


class TestSB10UDBasic:
    """Basic functionality tests."""

    def test_basic_2x2_system(self):
        """
        Test basic 2x2 system with full rank D12 and D21.

        System: N=2, M=2, NP=2
        Partitioning: NCON=1 (M2=1), NMEAS=1 (NP2=1)
        """
        n, m, np_ = 2, 2, 2
        ncon, nmeas = 1, 1

        b = np.array([
            [1.0, 0.3],
            [0.5, 0.8]
        ], dtype=float, order='F')

        c = np.array([
            [0.6, 0.4],
            [0.2, 0.9]
        ], dtype=float, order='F')

        d = np.array([
            [0.0, 1.5],
            [2.0, 0.7]
        ], dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0, f"Expected info=0, got {info}"
        assert rcond[0] > 0.0, "RCOND(1) should be positive"
        assert rcond[1] > 0.0, "RCOND(2) should be positive"
        assert np.abs(tu[0, 0]) > 1e-10, "TU should be non-zero"
        assert np.abs(ty[0, 0]) > 1e-10, "TY should be non-zero"

        m1 = m - ncon
        np1 = np_ - nmeas

        # B1 transformed by orthogonal V21' => Frobenius norm preserved
        np.testing.assert_allclose(
            np.linalg.norm(b_out[:, :m1], 'fro'),
            np.linalg.norm(b[:, :m1], 'fro'), rtol=1e-10)

        # C1 transformed by orthogonal Q12' => Frobenius norm preserved
        np.testing.assert_allclose(
            np.linalg.norm(c_out[:np1, :], 'fro'),
            np.linalg.norm(c[:np1, :], 'fro'), rtol=1e-10)

    def test_larger_system_3x4x3(self):
        """
        Test larger 3x4x3 system.

        System: N=3, M=4, NP=3
        Partitioning: NCON=2 (M2=2), NMEAS=1 (NP2=1)
        """
        n, m, np_ = 3, 4, 3
        ncon, nmeas = 2, 1

        b = np.array([
            [1.0, 0.3, 0.4, 0.2],
            [0.2, 0.8, 0.1, 0.3],
            [0.1, 0.2, 0.9, 0.5]
        ], dtype=float, order='F')

        c = np.array([
            [0.5, 0.2, 0.4],
            [0.1, 0.7, 0.3],
            [0.3, 0.2, 0.6]
        ], dtype=float, order='F')

        d = np.array([
            [0.0, 0.0, 2.0, 0.1],
            [0.0, 0.0, 0.3, 1.8],
            [1.5, 0.8, 0.4, 0.6]
        ], dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0, f"Expected info=0, got {info}"
        assert rcond[0] > 0.0, "RCOND(1) should be positive"
        assert rcond[1] > 0.0, "RCOND(2) should be positive"

        m1 = m - ncon
        np1 = np_ - nmeas

        np.testing.assert_allclose(
            np.linalg.norm(b_out[:, :m1], 'fro'),
            np.linalg.norm(b[:, :m1], 'fro'), rtol=1e-10)
        np.testing.assert_allclose(
            np.linalg.norm(c_out[:np1, :], 'fro'),
            np.linalg.norm(c[:np1, :], 'fro'), rtol=1e-10)


class TestSB10UDQuickReturn:
    """Quick return tests for edge cases."""

    def test_zero_n(self):
        """Quick return when n=0."""
        n, m, np_ = 0, 2, 2
        ncon, nmeas = 1, 1

        b = np.zeros((1, m), dtype=float, order='F')
        c = np.zeros((np_, 1), dtype=float, order='F')
        d = np.zeros((np_, m), dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0

    def test_zero_ncon(self):
        """Quick return when ncon=0."""
        n, m, np_ = 2, 2, 2
        ncon, nmeas = 0, 1

        b = np.zeros((n, m), dtype=float, order='F')
        c = np.zeros((np_, n), dtype=float, order='F')
        d = np.zeros((np_, m), dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0

    def test_zero_nmeas(self):
        """Quick return when nmeas=0."""
        n, m, np_ = 2, 2, 2
        ncon, nmeas = 1, 0

        b = np.zeros((n, m), dtype=float, order='F')
        c = np.zeros((np_, n), dtype=float, order='F')
        d = np.zeros((np_, m), dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0


class TestSB10UDRankDeficient:
    """Tests for rank-deficient matrices."""

    def test_rank_deficient_d12(self):
        """D12 is rank-deficient => INFO = 1.

        With n=2, m=3, np=3, ncon=1, nmeas=1:
          M1=2, M2=1, NP1=2, NP2=1
          D12 is NP1-by-M2 = 2x1 submatrix at columns M1:M = columns 2:2
        """
        n, m, np_ = 2, 3, 3
        ncon, nmeas = 1, 1

        b = np.array([
            [1.0, 0.3, 0.2],
            [0.5, 0.8, 0.4]
        ], dtype=float, order='F')
        c = np.array([
            [0.6, 0.4],
            [0.2, 0.9],
            [0.3, 0.5]
        ], dtype=float, order='F')
        d = np.array([
            [0.0, 0.0, 0.0],  # D12 at column 2 is 0
            [0.0, 0.0, 0.0],
            [2.0, 0.8, 0.7]   # D21 at (2, 0:1), D22 at (2, 2)
        ], dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 1, f"Expected info=1 for rank-deficient D12, got {info}"
        assert rcond[1] == 0.0, "RCOND(2) should be 0 when INFO=1"

    def test_rank_deficient_d21(self):
        """D21 is rank-deficient => INFO = 2.

        With n=2, m=3, np=3, ncon=1, nmeas=1:
          M1=2, M2=1, NP1=2, NP2=1
          D21 is NP2-by-M1 = 1x2 submatrix at row NP1=2, columns 0:1
        """
        n, m, np_ = 2, 3, 3
        ncon, nmeas = 1, 1

        b = np.array([
            [1.0, 0.3, 0.2],
            [0.5, 0.8, 0.4]
        ], dtype=float, order='F')
        c = np.array([
            [0.6, 0.4],
            [0.2, 0.9],
            [0.3, 0.5]
        ], dtype=float, order='F')
        d = np.array([
            [0.0, 0.0, 2.0],  # D12 col at (0:1, 2)
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.7]   # D21 at (2, 0:1) is [0, 0] (rank deficient), D22 at (2,2)
        ], dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 2, f"Expected info=2 for rank-deficient D21, got {info}"


class TestSB10UDInvalidParameters:
    """Tests for invalid parameter detection."""

    def test_negative_n(self):
        """Negative n returns info < 0."""
        b = np.zeros((1, 2), dtype=float, order='F')
        c = np.zeros((2, 1), dtype=float, order='F')
        d = np.zeros((2, 2), dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=-1, m=2, np=2, ncon=1, nmeas=1, b=b, c=c, d=d)
        assert info == -1

    def test_ncon_too_large(self):
        """NCON > M returns info < 0."""
        b = np.zeros((2, 2), dtype=float, order='F')
        c = np.zeros((2, 2), dtype=float, order='F')
        d = np.zeros((2, 2), dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=2, m=2, np=2, ncon=3, nmeas=1, b=b, c=c, d=d)
        assert info == -4


class TestSB10UDTransformationProperties:
    """Tests for mathematical properties of the transformation."""

    def test_tu_invertible(self):
        """TU transformation matrix should be invertible."""
        n, m, np_ = 2, 3, 3
        ncon, nmeas = 2, 1

        b = np.array([
            [1.0, 0.3, 0.2],
            [0.5, 0.8, 0.4]
        ], dtype=float, order='F')

        c = np.array([
            [0.6, 0.4],
            [0.2, 0.9],
            [0.3, 0.5]
        ], dtype=float, order='F')

        d = np.array([
            [0.0, 2.0, 0.3],
            [0.0, 0.1, 1.8],
            [1.5, 0.3, 0.6]
        ], dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0
        det_tu = np.linalg.det(tu)
        assert np.abs(det_tu) > 1e-10, "TU should be invertible"

    def test_rcond_consistency(self):
        """RCOND values should be consistent with singular values."""
        n, m, np_ = 2, 2, 2
        ncon, nmeas = 1, 1

        b = np.array([[1.0, 0.3], [0.5, 0.8]], dtype=float, order='F')
        c = np.array([[0.6, 0.4], [0.2, 0.9]], dtype=float, order='F')
        d = np.array([[0.0, 1.5], [2.0, 0.7]], dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0
        assert 0.0 < rcond[0] <= 1.0, "RCOND(1) should be in (0, 1]"
        assert 0.0 < rcond[1] <= 1.0, "RCOND(2) should be in (0, 1]"

    def test_d12_full_column_rank_preserved(self):
        """Verify transformed D12 still has full column rank."""
        n, m, np_ = 2, 3, 3
        ncon, nmeas = 1, 1
        m1 = m - ncon
        np1 = np_ - nmeas

        b = np.array([
            [1.0, 0.3, 0.2],
            [0.5, 0.8, 0.4]
        ], dtype=float, order='F')
        c = np.array([
            [0.6, 0.4],
            [0.2, 0.9],
            [0.3, 0.5]
        ], dtype=float, order='F')
        d = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.3],
            [1.5, 0.8, 0.7]
        ], dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0
        d12 = d_out[:np1, m1:]
        sv = np.linalg.svd(d12, compute_uv=False)
        assert sv[-1] > 1e-10, "D12 should remain full column rank"

    def test_tu_ty_invertibility(self):
        """Verify both TU and TY are invertible."""
        n, m, np_ = 3, 4, 4
        ncon, nmeas = 2, 2

        b = np.array([
            [1.0, 0.2, 0.3, 0.1],
            [0.1, 0.8, 0.2, 0.4],
            [0.3, 0.1, 0.9, 0.2]
        ], dtype=float, order='F')
        c = np.array([
            [0.5, 0.3, 0.1],
            [0.2, 0.7, 0.4],
            [0.1, 0.2, 0.6],
            [0.4, 0.1, 0.3]
        ], dtype=float, order='F')
        d = np.array([
            [0.0, 0.0, 2.0, 0.1],
            [0.0, 0.0, 0.1, 1.8],
            [1.5, 0.3, 0.0, 0.0],
            [0.2, 1.2, 0.0, 0.0]
        ], dtype=float, order='F')

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0
        assert np.abs(np.linalg.det(tu)) > 1e-10
        assert np.abs(np.linalg.det(ty)) > 1e-10

    def test_mimo_ncon2_nmeas2(self):
        """Test MIMO with ncon=2, nmeas=2."""
        n, m, np_ = 4, 5, 5
        ncon, nmeas = 2, 2

        np.random.seed(42)
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(np_, n).astype(float, order='F')
        m1 = m - ncon
        np1 = np_ - nmeas
        d = np.zeros((np_, m), dtype=float, order='F')
        d[:np1, m1:] = np.array([[2.0, 0.1], [0.3, 1.5], [0.1, 0.2]])
        d[np1:, :m1] = np.array([[1.2, 0.4, 0.1], [0.2, 1.5, 0.3]])

        b_out, c_out, d_out, tu, ty, rcond, info = ctrlsys.sb10ud(n=n, m=m, np=np_, ncon=ncon, nmeas=nmeas, b=b, c=c, d=d, tol=0.0)

        assert info == 0
        assert rcond[0] > 0
        assert rcond[1] > 0
        assert tu.shape == (ncon, ncon)
        assert ty.shape == (nmeas, nmeas)

        np.testing.assert_allclose(
            np.linalg.norm(b_out[:, :m1], 'fro'),
            np.linalg.norm(b[:, :m1], 'fro'), rtol=1e-10)
        np.testing.assert_allclose(
            np.linalg.norm(c_out[:np1, :], 'fro'),
            np.linalg.norm(c[:np1, :], 'fro'), rtol=1e-10)
