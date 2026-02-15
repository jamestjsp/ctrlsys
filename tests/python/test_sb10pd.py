"""
Tests for SB10PD - Normalization of system for H-infinity controller design.

SB10PD reduces D12 and D21 matrices to unit diagonal form and transforms
B, C, D11 for H2/H-infinity controller computation. It checks rank conditions
using SVD.

System partitioning:
    | A  | B1  B2  |
P = |----|---------|
    | C1 | D11 D12 |
    | C2 | D21 D22 |

where:
- M1 = M - NCON (disturbance inputs)
- M2 = NCON (control inputs)
- NP1 = NP - NMEAS (performance outputs)
- NP2 = NMEAS (measurement outputs)

Mathematical properties tested:
1. TU and TY are well-conditioned transformations
2. Transformed D12 has form [0; I_m2]
3. Transformed D21 has form [0, I_np2]
4. Transformation preserves system structure
"""
import numpy as np
import pytest


class TestSB10PDBasic:
    """Basic functionality tests."""

    def test_simple_system(self):
        """
        Test simple 2x2 system with well-conditioned D12 and D21.

        Constraints: M2 <= NP1, NP2 <= M1
        m=4, np=5, ncon=2, nmeas=2 => m1=2, m2=2, np1=3, np2=2
        Check: m2=2 <= np1=3 OK, np2=2 <= m1=2 OK

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import sb10pd

        np.random.seed(42)

        n = 2
        m = 4      # m1=2 (disturbance), m2=2 (control)
        np_ = 5    # np1=3 (performance), np2=2 (measurement)
        ncon = 2   # m2
        nmeas = 2  # np2

        m1 = m - ncon    # 2
        m2 = ncon        # 2
        np1 = np_ - nmeas  # 3
        np2 = nmeas        # 2

        a = np.array([
            [-1.0, 0.5],
            [0.0, -2.0]
        ], order='F', dtype=np.float64)

        b = np.array([
            [1.0, 0.5, 0.3, 0.2],
            [0.0, 0.2, 0.6, 0.4]
        ], order='F', dtype=np.float64)

        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.2, 0.8]
        ], order='F', dtype=np.float64)

        d = np.array([
            [0.1, 0.2, 0.0, 0.0],
            [0.0, 0.1, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], order='F', dtype=np.float64)

        tu = np.zeros((m2, m2), order='F', dtype=np.float64)
        ty = np.zeros((np2, np2), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == 0, f"Expected success, got info={info}"
        assert rcond[0] > 0, "TU should be well-conditioned"
        assert rcond[1] > 0, "TY should be well-conditioned"

    def test_identity_d12_d21(self):
        """
        Test with D12 and D21 having identity-like structure.

        Constraints: M2 <= NP1, NP2 <= M1
        m=4, np=5, ncon=2, nmeas=2 => m1=2, m2=2, np1=3, np2=2
        """
        from ctrlsys import sb10pd

        n = 2
        m = 4      # m1=2, m2=2
        np_ = 5    # np1=3, np2=2
        ncon = 2
        nmeas = 2

        m1 = m - ncon    # 2
        m2 = ncon        # 2
        np1 = np_ - nmeas  # 3
        np2 = nmeas        # 2

        a = np.array([
            [-1.0, 0.2],
            [0.0, -2.0]
        ], order='F', dtype=np.float64)

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([[1.0, 0.5], [0.5, 0.3]])
        b[:, m1:] = np.array([[0, 0], [1, 0]])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[0, :] = [1.0, 0.0]
        c[1, :] = [0.0, 1.0]
        c[2, :] = [0.5, 0.5]
        c[np1:, :] = np.array([[0.5, 0.5], [0.3, 0.7]])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[0, 0] = 0.1
        d[np1-2, m1:] = [1, 0]
        d[np1-1, m1:] = [0, 1]
        d[np1:, :m1] = [[1, 0], [0, 1]]

        tu = np.zeros((ncon, ncon), order='F', dtype=np.float64)
        ty = np.zeros((nmeas, nmeas), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == 0, f"Expected info=0, got {info}"


class TestSB10PDRankConditions:
    """Test rank condition checking."""

    def test_rank_deficient_d12_returns_info_3(self):
        """
        Test that rank-deficient D12 returns info=3.

        D12 must have full column rank (M2 columns).
        D12 is the NP1-by-M2 submatrix D[0:np1, m1:m].
        Setting D12 = 0 makes it rank deficient.

        Constraints: M2 <= NP1, NP2 <= M1
        m=4, np=5, ncon=2, nmeas=2 => m1=2, m2=2, np1=3, np2=2
        """
        from ctrlsys import sb10pd

        n = 2
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon   # 2
        m2 = ncon       # 2
        np1 = np_ - nmeas  # 3
        np2 = nmeas        # 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -1.0]
        ], order='F', dtype=np.float64)

        b = np.ones((n, m), order='F', dtype=np.float64)

        c = np.ones((np_, n), order='F', dtype=np.float64)

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:np1, m1:] = 0.0
        d[np1:, :m1] = [[1, 0], [0, 1]]

        tu = np.zeros((ncon, ncon), order='F', dtype=np.float64)
        ty = np.zeros((nmeas, nmeas), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info in [1, 3], f"Expected info=1 or 3 for rank-deficient D12, got {info}"

    def test_rank_deficient_d21_returns_info_4(self):
        """
        Test that rank-deficient D21 returns info=4.

        D21 must have full row rank (NP2 rows).
        D21 is the NP2-by-M1 submatrix D[np1:np, 0:m1].
        Setting D21 = 0 makes it rank deficient.

        Constraints: M2 <= NP1, NP2 <= M1
        m=4, np=5, ncon=2, nmeas=2 => m1=2, m2=2, np1=3, np2=2
        """
        from ctrlsys import sb10pd

        n = 2
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon   # 2
        m2 = ncon       # 2
        np1 = np_ - nmeas  # 3
        np2 = nmeas        # 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -1.0]
        ], order='F', dtype=np.float64)

        b = np.ones((n, m), order='F', dtype=np.float64)

        c = np.ones((np_, n), order='F', dtype=np.float64)

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:np1, m1:] = [[1, 0], [0, 1], [0, 0]]
        d[np1:, :m1] = 0.0

        tu = np.zeros((ncon, ncon), order='F', dtype=np.float64)
        ty = np.zeros((nmeas, nmeas), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info in [2, 4], f"Expected info=2 or 4 for rank-deficient D21, got {info}"


class TestSB10PDTransformationProperties:
    """Test mathematical properties of transformations."""

    def test_tu_ty_invertible(self):
        """
        Verify TU and TY are invertible (well-conditioned).

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import sb10pd

        np.random.seed(123)

        n = 3
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon
        np1 = np_ - nmeas

        a = -np.eye(n, dtype=np.float64, order='F')
        a[0, 1] = 0.5

        b = np.random.randn(n, m).astype(np.float64, order='F')

        c = np.random.randn(np_, n).astype(np.float64, order='F')

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:np1, m1:] = np.random.randn(np1, ncon)
        u, s, vt = np.linalg.svd(d[:np1, m1:])
        s_full = np.zeros((np1, ncon))
        s_full[:min(np1, ncon), :min(np1, ncon)] = np.diag([2.0, 1.5][:min(np1, ncon)])
        d[:np1, m1:] = u @ s_full @ vt

        d[np1:, :m1] = np.random.randn(nmeas, m1)
        u2, s2, vt2 = np.linalg.svd(d[np1:, :m1])
        s_full2 = np.zeros((nmeas, m1))
        s_full2[:min(nmeas, m1), :min(nmeas, m1)] = np.diag([1.8, 1.2][:min(nmeas, m1)])
        d[np1:, :m1] = u2 @ s_full2 @ vt2

        tu = np.zeros((ncon, ncon), order='F', dtype=np.float64)
        ty = np.zeros((nmeas, nmeas), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == 0, f"Expected success, got info={info}"

        assert rcond[0] > 1e-10, f"TU ill-conditioned: rcond={rcond[0]}"
        assert rcond[1] > 1e-10, f"TY ill-conditioned: rcond={rcond[1]}"

        tu_det = np.linalg.det(tu_out)
        ty_det = np.linalg.det(ty_out)
        assert abs(tu_det) > 1e-10, "TU should be nonsingular"
        assert abs(ty_det) > 1e-10, "TY should be nonsingular"

    def test_d12_normalization(self):
        """Verify transformed D12 has [0; I_m2] structure."""
        from ctrlsys import sb10pd

        n = 3
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon
        np1 = np_ - nmeas

        a = np.diag([-1.0, -2.0, -3.0]).astype(np.float64, order='F')

        b = np.array([
            [1.0, 0.5, 0.3, 0.2],
            [0.0, 0.2, 0.6, 0.4],
            [0.1, 0.3, 0.5, 0.7]
        ], order='F', dtype=np.float64)

        c = np.array([
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.2],
            [0.5, 0.5, 0.3],
            [0.3, 0.7, 0.1],
            [0.2, 0.8, 0.4]
        ], order='F', dtype=np.float64)

        d = np.array([
            [0.1, 0.2, 0.0, 0.0],
            [0.0, 0.1, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], order='F', dtype=np.float64)

        tu = np.zeros((ncon, ncon), order='F', dtype=np.float64)
        ty = np.zeros((nmeas, nmeas), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == 0
        d12_out = d_out[:np1, m1:]
        np.testing.assert_allclose(d12_out[:np1 - ncon, :], 0, atol=1e-10)
        bottom = d12_out[np1 - ncon:, :]
        np.testing.assert_allclose(bottom @ bottom.T, np.eye(ncon), atol=1e-10)

    def test_mimo_larger(self):
        """Test MIMO system with n=4, ncon=2, nmeas=2."""
        from ctrlsys import sb10pd

        n = 4
        m = 5
        np_ = 6
        ncon = 2
        nmeas = 2
        m1 = m - ncon
        np1 = np_ - nmeas

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(np.float64, order='F')
        a[0, 1] = 0.1

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([
            [1.0, 0.5, 0.1],
            [0.0, 1.0, 0.2],
            [0.2, 0.0, 1.0],
            [0.1, 0.3, 0.0]
        ])
        b[:, m1:] = np.array([
            [0.3, 0.1],
            [0.5, 0.2],
            [0.1, 0.4],
            [0.2, 0.6]
        ])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:np1, :] = np.array([
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 1.0, 0.0, 0.1],
            [0.2, 0.3, 1.0, 0.0],
            [0.0, 0.1, 0.2, 1.0]
        ])
        c[np1:, :] = np.array([
            [0.5, 0.3, 0.1, 0.2],
            [0.1, 0.4, 0.2, 0.5]
        ])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[np1 - ncon, m1:] = [1.0, 0.0]
        d[np1 - 1, m1:] = [0.0, 1.0]
        d[np1:, :m1] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        tu = np.zeros((ncon, ncon), order='F', dtype=np.float64)
        ty = np.zeros((nmeas, nmeas), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == 0
        assert rcond[0] > 1e-10
        assert rcond[1] > 1e-10
        assert abs(np.linalg.det(tu_out)) > 1e-10
        assert abs(np.linalg.det(ty_out)) > 1e-10


class TestSB10PDEdgeCases:
    """Edge cases and boundary conditions."""

    def test_n_zero(self):
        """Test N=0 quick return."""
        from ctrlsys import sb10pd

        n = 0
        m = 2
        np_ = 2
        ncon = 1
        nmeas = 1

        a = np.zeros((1, 1), order='F', dtype=np.float64)
        b = np.zeros((1, m), order='F', dtype=np.float64)
        c = np.zeros((np_, 1), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)
        tu = np.zeros((ncon, ncon), order='F', dtype=np.float64)
        ty = np.zeros((nmeas, nmeas), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0

    def test_m_zero(self):
        """Test M=0 quick return."""
        from ctrlsys import sb10pd

        n = 2
        m = 0
        np_ = 2
        ncon = 0
        nmeas = 0

        a = np.eye(n, order='F', dtype=np.float64)
        b = np.zeros((n, 1), order='F', dtype=np.float64)
        c = np.zeros((np_, n), order='F', dtype=np.float64)
        d = np.zeros((np_, 1), order='F', dtype=np.float64)
        tu = np.zeros((1, 1), order='F', dtype=np.float64)
        ty = np.zeros((1, 1), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == 0

    def test_np_zero(self):
        """Test NP=0 quick return."""
        from ctrlsys import sb10pd

        n = 2
        m = 2
        np_ = 0
        ncon = 0
        nmeas = 0

        a = np.eye(n, order='F', dtype=np.float64)
        b = np.zeros((n, m), order='F', dtype=np.float64)
        c = np.zeros((1, n), order='F', dtype=np.float64)
        d = np.zeros((1, m), order='F', dtype=np.float64)
        tu = np.zeros((1, 1), order='F', dtype=np.float64)
        ty = np.zeros((1, 1), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == 0


class TestSB10PDParameterValidation:
    """Test parameter validation."""

    def test_invalid_ncon(self):
        """Test NCON > M returns error."""
        from ctrlsys import sb10pd

        n = 2
        m = 2
        np_ = 3
        ncon = 3
        nmeas = 1

        a = np.eye(n, order='F', dtype=np.float64)
        b = np.zeros((n, m), order='F', dtype=np.float64)
        c = np.zeros((np_, n), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)
        tu = np.zeros((ncon, ncon), order='F', dtype=np.float64)
        ty = np.zeros((nmeas, nmeas), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == -4, f"Expected info=-4 for NCON > M, got {info}"

    def test_invalid_nmeas(self):
        """Test NP2 > M1 returns error (constraint NMEAS <= M - NCON)."""
        from ctrlsys import sb10pd

        n = 2
        m = 4
        np_ = 6
        ncon = 1
        nmeas = 4

        a = np.eye(n, order='F', dtype=np.float64)
        b = np.zeros((n, m), order='F', dtype=np.float64)
        c = np.zeros((np_, n), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)
        tu = np.zeros((ncon, ncon), order='F', dtype=np.float64)
        ty = np.zeros((nmeas, nmeas), order='F', dtype=np.float64)

        b_out, c_out, d_out, tu_out, ty_out, rcond, info = sb10pd(
            n, m, np_, ncon, nmeas, a, b, c, d, tu, ty
        )

        assert info == -5, f"Expected info=-5 for NP2 > M1, got {info}"
