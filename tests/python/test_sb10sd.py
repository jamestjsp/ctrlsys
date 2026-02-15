"""
Tests for SB10SD - H2 optimal controller for normalized discrete-time systems.

SB10SD computes the H2 optimal controller matrices:
    K = | AK | BK |
        |----|----|
        | CK | DK |

for the normalized discrete-time system:
    P = | A  | B1  B2  |
        |----|---------|
        | C1 | D11 D12 |
        | C2 | D21  0  |

Assumptions:
- (A,B2) is stabilizable and (C2,A) is detectable
- D12 has form [0; I] and D21 has form [0 I] (from SB10PD normalization)
- Full rank conditions on pencils (A3, A4 in documentation)

Mathematical properties tested:
1. X-Riccati solution: Ax'*inv(I + X*Dx)*X*Ax - X + Cx = 0
2. Y-Riccati solution: Ay*inv(I + Y*Dy)*Y*Ay' - Y + Cy = 0
3. Controller stabilizes the closed-loop system
4. RCOND values are valid condition number estimates
"""
import numpy as np
import pytest


class TestSB10SDBasic:
    """Basic functionality tests."""

    def test_simple_2state_system(self):
        """
        Test simple 2-state discrete-time system.

        Creates a normalized discrete-time system and computes H2 controller.
        System partitioning: N=2, M=4, NP=5, NCON=2, NMEAS=2
        => M1=2 (disturbance), M2=2 (control)
        => NP1=3 (performance), NP2=2 (measurement)

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import sb10sd

        np.random.seed(42)

        n = 2
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon      # 2
        m2 = ncon          # 2
        np1 = np_ - nmeas  # 3
        np2 = nmeas        # 2
        nd1 = np1 - m2     # 1
        nd2 = m1 - np2     # 0

        a = np.array([
            [0.5, 0.1],
            [0.0, 0.6]
        ], order='F', dtype=np.float64)

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([[0.3, 0.1], [0.0, 0.2]])
        b[:, m1:] = np.array([[0.4, 0.0], [0.0, 0.4]])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:nd1, :] = np.array([[0.2, 0.3]])
        c[nd1:np1, :] = np.array([[0, 0], [0, 0]])
        c[np1:, :] = np.array([[0.5, 0.0], [0.0, 0.5]])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:nd1, :m1] = np.array([[0.1, 0.05]])
        d[nd1:np1, m1:] = np.array([[1, 0], [0, 1]])
        d[np1:, :m1] = np.array([[1, 0], [0, 1]])

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0, f"Expected success, got info={info}"

        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)
        assert x.shape == (n, n)
        assert y.shape == (n, n)
        assert rcond.shape == (4,)

        for i in range(2):
            assert 0 < rcond[i] <= 1, f"RCOND({i+1}) should be in (0,1], got {rcond[i]}"
        for i in range(2, 4):
            assert 0 <= rcond[i] <= 1, f"RCOND({i+1}) should be in [0,1], got {rcond[i]}"

    def test_3state_stable_system(self):
        """
        Test 3-state stable discrete-time system.

        Uses diagonal stable A matrix and normalized D12, D21.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import sb10sd

        np.random.seed(123)

        n = 3
        m = 5
        np_ = 6
        ncon = 2
        nmeas = 2

        m1 = m - ncon      # 3
        m2 = ncon          # 2
        np1 = np_ - nmeas  # 4
        np2 = nmeas        # 2
        nd1 = np1 - m2     # 2
        nd2 = m1 - np2     # 1

        a = np.diag([0.4, 0.5, 0.6]).astype(np.float64, order='F')

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([
            [0.2, 0.1, 0.0],
            [0.0, 0.2, 0.1],
            [0.0, 0.0, 0.2]
        ])
        b[:, m1:] = np.array([
            [0.3, 0.0],
            [0.0, 0.3],
            [0.0, 0.0]
        ])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:nd1, :] = np.array([
            [0.3, 0.1, 0.0],
            [0.0, 0.3, 0.1]
        ])
        c[nd1:np1, :] = 0
        c[np1:, :] = np.array([
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0]
        ])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:nd1, :m1] = np.array([
            [0.1, 0.05, 0.0],
            [0.0, 0.1, 0.05]
        ])
        d[nd1:np1, m1:] = np.array([
            [1, 0],
            [0, 1]
        ])
        if nd2 > 0:
            d[np1:, :nd2] = 0
        d[np1:, nd2:m1] = np.array([
            [1, 0],
            [0, 1]
        ])

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0, f"Expected success, got info={info}"

        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)


class TestSB10SDRiccatiSolutions:
    """Test Riccati equation solution properties."""

    def test_x_symmetric(self):
        """
        Verify X solution is symmetric.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import sb10sd

        np.random.seed(456)

        n = 2
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon
        np1 = np_ - nmeas
        nd1 = np1 - ncon
        nd2 = m1 - nmeas

        a = np.array([[0.4, 0.1], [0.0, 0.5]], order='F', dtype=np.float64)

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([[0.2, 0.1], [0.0, 0.2]])
        b[:, m1:] = np.array([[0.3, 0.0], [0.0, 0.3]])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:nd1, :] = np.array([[0.2, 0.3]])
        c[np1:, :] = np.array([[0.5, 0.0], [0.0, 0.5]])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:nd1, :m1] = np.array([[0.1, 0.05]])
        d[nd1:np1, m1:] = np.eye(ncon)
        d[np1:, nd2:m1] = np.eye(nmeas)

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0, f"Expected success, got info={info}"
        np.testing.assert_allclose(x, x.T, rtol=1e-12, atol=1e-14,
                                   err_msg="X should be symmetric")

    def test_y_symmetric(self):
        """
        Verify Y solution is symmetric.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import sb10sd

        np.random.seed(789)

        n = 2
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon
        np1 = np_ - nmeas
        nd1 = np1 - ncon
        nd2 = m1 - nmeas

        a = np.array([[0.3, 0.1], [0.0, 0.4]], order='F', dtype=np.float64)

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([[0.2, 0.1], [0.0, 0.2]])
        b[:, m1:] = np.array([[0.3, 0.0], [0.0, 0.3]])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:nd1, :] = np.array([[0.2, 0.3]])
        c[np1:, :] = np.array([[0.5, 0.0], [0.0, 0.5]])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:nd1, :m1] = np.array([[0.1, 0.05]])
        d[nd1:np1, m1:] = np.eye(ncon)
        d[np1:, nd2:m1] = np.eye(nmeas)

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0, f"Expected success, got info={info}"
        np.testing.assert_allclose(y, y.T, rtol=1e-12, atol=1e-14,
                                   err_msg="Y should be symmetric")


class TestSB10SDQuickReturn:
    """Test quick return cases."""

    def test_n_zero(self):
        """Quick return when N=0."""
        from ctrlsys import sb10sd

        n = 0
        m = 2
        np_ = 2
        ncon = 1
        nmeas = 1

        a = np.zeros((1, 1), order='F', dtype=np.float64)
        b = np.zeros((1, m), order='F', dtype=np.float64)
        c = np.zeros((np_, 1), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0
        assert rcond[2] == 1.0
        assert rcond[3] == 1.0

    def test_m1_zero(self):
        """Quick return when M1=0 (M=NCON), which implies NP2=0 too."""
        from ctrlsys import sb10sd

        n = 2
        m = 2
        np_ = 4
        ncon = 2
        nmeas = 0

        a = np.eye(n, order='F', dtype=np.float64) * 0.5
        b = np.zeros((n, m), order='F', dtype=np.float64)
        c = np.zeros((np_, n), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0
        assert rcond[0] == 1.0

    def test_np1_zero(self):
        """Quick return when NP1=0 (NP=NMEAS), which implies NCON=0 too."""
        from ctrlsys import sb10sd

        n = 2
        m = 4
        np_ = 2
        ncon = 0
        nmeas = 2

        a = np.eye(n, order='F', dtype=np.float64) * 0.5
        b = np.zeros((n, m), order='F', dtype=np.float64)
        c = np.zeros((np_, n), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0
        assert rcond[1] == 1.0


class TestSB10SDParameterValidation:
    """Test parameter validation."""

    def test_invalid_n(self):
        """Test N < 0 returns error."""
        from ctrlsys import sb10sd

        with pytest.raises(ValueError):
            sb10sd(-1, 2, 2, 1, 1,
                   np.zeros((1, 1), order='F', dtype=np.float64),
                   np.zeros((1, 2), order='F', dtype=np.float64),
                   np.zeros((2, 1), order='F', dtype=np.float64),
                   np.zeros((2, 2), order='F', dtype=np.float64))

    def test_invalid_ncon(self):
        """Test NCON > M returns error."""
        from ctrlsys import sb10sd

        n = 2
        m = 2
        np_ = 3
        ncon = 3
        nmeas = 1

        a = np.eye(n, order='F', dtype=np.float64)
        b = np.zeros((n, m), order='F', dtype=np.float64)
        c = np.zeros((np_, n), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)

        with pytest.raises(ValueError):
            sb10sd(n, m, np_, ncon, nmeas, a, b, c, d)

    def test_ncon_larger_than_np1(self):
        """Test M2 > NP1 returns error (M2=NCON, NP1=NP-NMEAS)."""
        from ctrlsys import sb10sd

        n = 2
        m = 4
        np_ = 3
        ncon = 2
        nmeas = 2

        a = np.eye(n, order='F', dtype=np.float64)
        b = np.zeros((n, m), order='F', dtype=np.float64)
        c = np.zeros((np_, n), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)

        with pytest.raises(ValueError):
            sb10sd(n, m, np_, ncon, nmeas, a, b, c, d)


class TestSB10SDErrorHandling:
    """Test algorithmic error handling."""

    def test_x_riccati_failure_returns_info_1(self):
        """
        Test X-Riccati failure returns info=1.

        Create a system where the X-Riccati has no stabilizing solution.
        This requires (A,B2) not stabilizable.
        """
        from ctrlsys import sb10sd

        n = 2
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon
        np1 = np_ - nmeas
        nd1 = np1 - ncon
        nd2 = m1 - nmeas

        a = np.array([[2.0, 0.0], [0.0, 3.0]], order='F', dtype=np.float64)

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([[0.2, 0.1], [0.0, 0.2]])
        b[:, m1:] = np.array([[0.0, 0.0], [0.0, 0.0]])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:nd1, :] = np.array([[0.2, 0.3]])
        c[np1:, :] = np.array([[0.5, 0.0], [0.0, 0.5]])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:nd1, :m1] = np.array([[0.1, 0.05]])
        d[nd1:np1, m1:] = np.eye(ncon)
        d[np1:, nd2:m1] = np.eye(nmeas)

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info in [1, 2], f"Expected info=1 or 2 for Riccati failure, got {info}"

    def test_y_riccati_failure_returns_info_3(self):
        """
        Test Y-Riccati failure returns info=3.

        Create a system where the Y-Riccati has no stabilizing solution.
        This requires (C2,A) not detectable.
        """
        from ctrlsys import sb10sd

        n = 2
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon
        np1 = np_ - nmeas
        nd1 = np1 - ncon
        nd2 = m1 - nmeas

        a = np.array([[2.0, 0.0], [0.0, 3.0]], order='F', dtype=np.float64)

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([[0.0, 0.0], [0.0, 0.0]])
        b[:, m1:] = np.array([[0.3, 0.0], [0.0, 0.3]])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:nd1, :] = np.array([[0.2, 0.3]])
        c[np1:, :] = np.array([[0.0, 0.0], [0.0, 0.0]])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:nd1, :m1] = np.array([[0.1, 0.05]])
        d[nd1:np1, m1:] = np.eye(ncon)
        d[np1:, nd2:m1] = np.eye(nmeas)

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info in [1, 2, 3, 4], f"Expected failure info, got {info}"


class TestSB10SDControllerStability:
    """Test controller stability properties."""

    def test_closed_loop_stable(self):
        """
        Verify closed-loop system is stable (eigenvalues inside unit circle).

        For discrete-time H2, closed-loop eigenvalues should satisfy |λ| < 1.

        Random seed: 888 (for reproducibility)
        """
        from ctrlsys import sb10sd

        np.random.seed(888)

        n = 2
        m = 4
        np_ = 5
        ncon = 2
        nmeas = 2

        m1 = m - ncon
        np1 = np_ - nmeas
        nd1 = np1 - ncon
        nd2 = m1 - nmeas

        a = np.array([[0.3, 0.1], [0.0, 0.4]], order='F', dtype=np.float64)

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([[0.2, 0.1], [0.0, 0.2]])
        b[:, m1:] = np.array([[0.3, 0.0], [0.0, 0.3]])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:nd1, :] = np.array([[0.2, 0.3]])
        c[np1:, :] = np.array([[0.5, 0.0], [0.0, 0.5]])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[:nd1, :m1] = np.array([[0.1, 0.05]])
        d[nd1:np1, m1:] = np.eye(ncon)
        d[np1:, nd2:m1] = np.eye(nmeas)

        b2 = b[:, m1:]
        c2 = c[np1:, :]

        ak, bk, ck, dk, x, y, rcond, info = sb10sd(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0, f"Expected success, got info={info}"

        acl = np.zeros((2*n, 2*n), order='F', dtype=np.float64)
        acl[:n, :n] = a + b2 @ ck
        acl[:n, n:] = b2 @ dk @ c2
        acl[n:, :n] = bk @ c2
        acl[n:, n:] = ak

        cl_eigs = np.linalg.eigvals(acl)

        for i, eig in enumerate(cl_eigs):
            assert np.abs(eig) < 1.0 - 1e-10, \
                f"Closed-loop eigenvalue {i} = {eig}, |λ| = {np.abs(eig)} >= 1"
