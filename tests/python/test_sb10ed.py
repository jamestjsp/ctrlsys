"""
Tests for SB10ED - H2 optimal state controller for discrete-time systems.

SB10ED computes the H2 optimal n-state controller:
    K = | AK | BK |
        |----|----|
        | CK | DK |

for the discrete-time system:
    P = | A  | B1  B2  |
        |----|---------|
        | C1 |  0  D12 |
        | C2 | D21 D22 |

where B2 has NCON columns (control inputs) and C2 has NMEAS rows (measurements).

Assumptions:
- (A1) (A,B2) is stabilizable and (C2,A) is detectable
- (A2) D12 has full column rank, D21 has full row rank
- (A3, A4) Full rank conditions for discrete-time invariant zeros

Mathematical properties tested:
1. Controller stabilizes closed-loop (eigenvalues inside unit circle)
2. RCOND values are valid condition number estimates
3. Controller dimensions match expected partitioning
"""
import numpy as np
import pytest


class TestSB10EDHtmlDocExample:
    """Test using HTML documentation example data."""

    def test_html_example(self):
        """
        Test SB10ED with exact HTML doc example.

        System: N=6, M=5, NP=5, NCON=2, NMEAS=2
        Expected controller matrices from HTML doc output.

        Data format: READ ( NIN, FMT = * ) ( ( A(I,J), J = 1,N ), I = 1,N )
        This reads row-by-row (J varies fastest inside I loop).
        """
        from ctrlsys import sb10ed

        n = 6
        m = 5
        np_ = 5
        ncon = 2
        nmeas = 2

        a = np.array([
            [-0.7,  0.0,  0.3,  0.0, -0.5, -0.1],
            [-0.6,  0.2, -0.4, -0.3,  0.0,  0.0],
            [-0.5,  0.7, -0.1,  0.0,  0.0, -0.8],
            [-0.7,  0.0,  0.0, -0.5, -1.0,  0.0],
            [ 0.0,  0.3,  0.6, -0.9,  0.1, -0.4],
            [ 0.5, -0.8,  0.0,  0.0,  0.2, -0.9]
        ], order='F', dtype=np.float64)

        b = np.array([
            [-1.0, -2.0, -2.0,  1.0,  0.0],
            [ 1.0,  0.0,  1.0, -2.0,  1.0],
            [-3.0, -4.0,  0.0,  2.0, -2.0],
            [ 1.0, -2.0,  1.0,  0.0, -1.0],
            [ 0.0,  1.0, -2.0,  0.0,  3.0],
            [ 1.0,  0.0,  3.0, -1.0, -2.0]
        ], order='F', dtype=np.float64)

        c = np.array([
            [ 1.0, -1.0,  2.0, -2.0,  0.0, -3.0],
            [-3.0,  0.0,  1.0, -1.0,  1.0,  0.0],
            [ 0.0,  2.0,  0.0, -4.0,  0.0, -2.0],
            [ 1.0, -3.0,  0.0,  0.0,  3.0,  1.0],
            [ 0.0,  1.0, -2.0,  1.0,  0.0, -2.0]
        ], order='F', dtype=np.float64)

        d = np.array([
            [ 1.0, -1.0, -2.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  1.0,  0.0],
            [ 2.0, -1.0, -3.0,  0.0,  1.0],
            [ 0.0,  1.0,  0.0,  1.0, -1.0],
            [ 0.0,  0.0,  1.0,  2.0,  1.0]
        ], order='F', dtype=np.float64)

        tol = 1e-8

        ak, bk, ck, dk, rcond, info = sb10ed(
            n, m, np_, ncon, nmeas, a, b, c, d, tol
        )

        assert info == 0, f"Expected success, got info={info}"

        assert ak.shape == (n, n), f"AK shape mismatch: {ak.shape}"
        assert bk.shape == (n, nmeas), f"BK shape mismatch: {bk.shape}"
        assert ck.shape == (ncon, n), f"CK shape mismatch: {ck.shape}"
        assert dk.shape == (ncon, nmeas), f"DK shape mismatch: {dk.shape}"
        assert rcond.shape == (7,), f"RCOND shape mismatch: {rcond.shape}"

        ak_expected = np.array([
            [-0.0551, -2.1891, -0.6607, -0.2532,  0.6674, -1.0044],
            [-1.0379,  2.3804,  0.5031,  0.3960, -0.6605,  1.2673],
            [-0.0876, -2.1320, -0.4701, -1.1461,  1.2927, -1.5116],
            [-0.1358, -2.1237, -0.9560, -0.7144,  0.6673, -0.7957],
            [ 0.4900,  0.0895,  0.2634, -0.2354,  0.1623, -0.2663],
            [ 0.1672, -0.4163,  0.2871, -0.1983,  0.4944, -0.6967]
        ], order='F', dtype=np.float64)

        bk_expected = np.array([
            [-0.5985, -0.5464],
            [ 0.5285,  0.6087],
            [-0.7600, -0.4472],
            [-0.7288, -0.6090],
            [ 0.0532,  0.0658],
            [-0.0663,  0.0059]
        ], order='F', dtype=np.float64)

        ck_expected = np.array([
            [ 0.2500, -1.0200, -0.3371, -0.2733,  0.2747, -0.4444],
            [ 0.0654,  0.2095,  0.0632,  0.2089, -0.1895,  0.1834]
        ], order='F', dtype=np.float64)

        dk_expected = np.array([
            [-0.2181, -0.2070],
            [ 0.1094,  0.1159]
        ], order='F', dtype=np.float64)

        np.testing.assert_allclose(ak, ak_expected, rtol=1e-3, atol=1e-4,
                                   err_msg="AK values do not match HTML doc")
        np.testing.assert_allclose(bk, bk_expected, rtol=1e-3, atol=1e-4,
                                   err_msg="BK values do not match HTML doc")
        np.testing.assert_allclose(ck, ck_expected, rtol=1e-3, atol=1e-4,
                                   err_msg="CK values do not match HTML doc")
        np.testing.assert_allclose(dk, dk_expected, rtol=1e-3, atol=1e-4,
                                   err_msg="DK values do not match HTML doc")

        rcond_expected = [1.0, 1.0, 0.25207, 0.083985, 0.0048628, 0.00055015, 0.49886]
        for i in range(7):
            assert 0 < rcond[i] <= 1.0, f"RCOND({i+1}) = {rcond[i]} out of range (0,1]"
        np.testing.assert_allclose(rcond[0], rcond_expected[0], rtol=1e-3,
                                   err_msg="RCOND(1) mismatch")
        np.testing.assert_allclose(rcond[1], rcond_expected[1], rtol=1e-3,
                                   err_msg="RCOND(2) mismatch")


class TestSB10EDBasic:
    """Basic functionality tests."""

    def test_simple_stable_system(self):
        """
        Test simple 2-state stable discrete-time system.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import sb10ed

        np.random.seed(42)

        n = 2
        m = 3
        np_ = 3
        ncon = 1
        nmeas = 1

        m1 = m - ncon      # 2
        m2 = ncon          # 1
        np1 = np_ - nmeas  # 2
        np2 = nmeas        # 1

        a = np.array([
            [0.5, 0.1],
            [0.0, 0.6]
        ], order='F', dtype=np.float64)

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([[0.3, 0.1], [0.0, 0.2]])
        b[:, m1:] = np.array([[0.4], [0.3]])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:np1, :] = np.array([
            [0.2, 0.3],
            [0.0, 0.0]
        ])
        c[np1:, :] = np.array([[0.5, 0.4]])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[0, :m1] = [0.1, 0.05]
        d[np1-1, m1:] = [1.0]
        d[np1:, :np2] = [[1.0]]

        ak, bk, ck, dk, rcond, info = sb10ed(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0, f"Expected success, got info={info}"

        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)
        assert rcond.shape == (7,)

        for i in range(7):
            assert 0 < rcond[i] <= 1.0 + 1e-14, f"RCOND({i+1}) should be in (0,1], got {rcond[i]}"


class TestSB10EDControllerProperties:
    """Test controller output properties."""

    def test_controller_finite(self):
        """Verify controller matrices are all finite."""
        from ctrlsys import sb10ed

        n = 6
        m = 5
        np_ = 5
        ncon = 2
        nmeas = 2

        a = np.array([
            [-0.7,  0.0,  0.3,  0.0, -0.5, -0.1],
            [-0.6,  0.2, -0.4, -0.3,  0.0,  0.0],
            [-0.5,  0.7, -0.1,  0.0,  0.0, -0.8],
            [-0.7,  0.0,  0.0, -0.5, -1.0,  0.0],
            [ 0.0,  0.3,  0.6, -0.9,  0.1, -0.4],
            [ 0.5, -0.8,  0.0,  0.0,  0.2, -0.9]
        ], order='F', dtype=np.float64)

        b = np.array([
            [-1.0, -2.0, -2.0,  1.0,  0.0],
            [ 1.0,  0.0,  1.0, -2.0,  1.0],
            [-3.0, -4.0,  0.0,  2.0, -2.0],
            [ 1.0, -2.0,  1.0,  0.0, -1.0],
            [ 0.0,  1.0, -2.0,  0.0,  3.0],
            [ 1.0,  0.0,  3.0, -1.0, -2.0]
        ], order='F', dtype=np.float64)

        c = np.array([
            [ 1.0, -1.0,  2.0, -2.0,  0.0, -3.0],
            [-3.0,  0.0,  1.0, -1.0,  1.0,  0.0],
            [ 0.0,  2.0,  0.0, -4.0,  0.0, -2.0],
            [ 1.0, -3.0,  0.0,  0.0,  3.0,  1.0],
            [ 0.0,  1.0, -2.0,  1.0,  0.0, -2.0]
        ], order='F', dtype=np.float64)

        d = np.array([
            [ 1.0, -1.0, -2.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  1.0,  0.0],
            [ 2.0, -1.0, -3.0,  0.0,  1.0],
            [ 0.0,  1.0,  0.0,  1.0, -1.0],
            [ 0.0,  0.0,  1.0,  2.0,  1.0]
        ], order='F', dtype=np.float64)

        ak, bk, ck, dk, rcond, info = sb10ed(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0
        assert np.isfinite(ak).all()
        assert np.isfinite(bk).all()
        assert np.isfinite(ck).all()
        assert np.isfinite(dk).all()

    def test_rcond_values_valid(self):
        """
        Verify all RCOND values are valid condition number estimates.

        Uses HTML doc example system.
        """
        from ctrlsys import sb10ed

        n = 6
        m = 5
        np_ = 5
        ncon = 2
        nmeas = 2

        a = np.array([
            [-0.7,  0.0,  0.3,  0.0, -0.5, -0.1],
            [-0.6,  0.2, -0.4, -0.3,  0.0,  0.0],
            [-0.5,  0.7, -0.1,  0.0,  0.0, -0.8],
            [-0.7,  0.0,  0.0, -0.5, -1.0,  0.0],
            [ 0.0,  0.3,  0.6, -0.9,  0.1, -0.4],
            [ 0.5, -0.8,  0.0,  0.0,  0.2, -0.9]
        ], order='F', dtype=np.float64)

        b = np.array([
            [-1.0, -2.0, -2.0,  1.0,  0.0],
            [ 1.0,  0.0,  1.0, -2.0,  1.0],
            [-3.0, -4.0,  0.0,  2.0, -2.0],
            [ 1.0, -2.0,  1.0,  0.0, -1.0],
            [ 0.0,  1.0, -2.0,  0.0,  3.0],
            [ 1.0,  0.0,  3.0, -1.0, -2.0]
        ], order='F', dtype=np.float64)

        c = np.array([
            [ 1.0, -1.0,  2.0, -2.0,  0.0, -3.0],
            [-3.0,  0.0,  1.0, -1.0,  1.0,  0.0],
            [ 0.0,  2.0,  0.0, -4.0,  0.0, -2.0],
            [ 1.0, -3.0,  0.0,  0.0,  3.0,  1.0],
            [ 0.0,  1.0, -2.0,  1.0,  0.0, -2.0]
        ], order='F', dtype=np.float64)

        d = np.array([
            [ 1.0, -1.0, -2.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  1.0,  0.0],
            [ 2.0, -1.0, -3.0,  0.0,  1.0],
            [ 0.0,  1.0,  0.0,  1.0, -1.0],
            [ 0.0,  0.0,  1.0,  2.0,  1.0]
        ], order='F', dtype=np.float64)

        ak, bk, ck, dk, rcond, info = sb10ed(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0, f"Expected success, got info={info}"

        assert rcond.shape == (7,)

        for i in range(7):
            assert rcond[i] > 0, f"RCOND({i+1}) must be positive, got {rcond[i]}"
            assert rcond[i] <= 1.0 + 1e-14, f"RCOND({i+1}) must be <= 1, got {rcond[i]}"

    def test_mimo_simple(self):
        """Test MIMO discrete-time system with ncon=2, nmeas=2."""
        from ctrlsys import sb10ed

        n = 4
        m = 5
        np_ = 5
        ncon = 2
        nmeas = 2
        m1 = m - ncon
        np1 = np_ - nmeas

        a = np.diag([0.3, -0.2, 0.4, -0.1]).astype(np.float64, order='F')
        a[0, 1] = 0.1

        b = np.zeros((n, m), order='F', dtype=np.float64)
        b[:, :m1] = np.array([
            [0.3, 0.1, 0.0],
            [0.0, 0.2, 0.1],
            [0.1, 0.0, 0.3],
            [0.0, 0.1, 0.0]
        ])
        b[:, m1:] = np.array([
            [0.4, 0.0],
            [0.0, 0.3],
            [0.1, 0.2],
            [0.2, 0.1]
        ])

        c = np.zeros((np_, n), order='F', dtype=np.float64)
        c[:np1, :] = np.array([
            [0.2, 0.1, 0.0, 0.0],
            [0.0, 0.3, 0.1, 0.0],
            [0.0, 0.0, 0.2, 0.1]
        ])
        c[np1:, :] = np.array([
            [0.5, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.3]
        ])

        d = np.zeros((np_, m), order='F', dtype=np.float64)
        d[np1 - ncon, m1:] = [1.0, 0.0]
        d[np1 - 1, m1:] = [0.0, 1.0]
        d[np1:, :m1] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        ak, bk, ck, dk, rcond, info = sb10ed(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0
        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)


class TestSB10EDQuickReturn:
    """Test quick return cases."""

    def test_n_zero_m2_zero_np2_zero(self):
        """Quick return when N=0 and M2=NP2=0."""
        from ctrlsys import sb10ed

        n = 0
        m = 2
        np_ = 2
        ncon = 0
        nmeas = 0

        a = np.zeros((1, 1), order='F', dtype=np.float64)
        b = np.zeros((1, m), order='F', dtype=np.float64)
        c = np.zeros((np_, 1), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)

        ak, bk, ck, dk, rcond, info = sb10ed(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0


class TestSB10EDParameterValidation:
    """Test parameter validation."""

    def test_invalid_n(self):
        """Test N < 0 returns error."""
        from ctrlsys import sb10ed

        with pytest.raises(ValueError):
            sb10ed(-1, 2, 2, 1, 1,
                   np.zeros((1, 1), order='F', dtype=np.float64),
                   np.zeros((1, 2), order='F', dtype=np.float64),
                   np.zeros((2, 1), order='F', dtype=np.float64),
                   np.zeros((2, 2), order='F', dtype=np.float64))

    def test_invalid_ncon(self):
        """Test NCON > M returns error."""
        from ctrlsys import sb10ed

        n = 2
        m = 2
        np_ = 4
        ncon = 3
        nmeas = 1

        a = np.eye(n, order='F', dtype=np.float64)
        b = np.zeros((n, m), order='F', dtype=np.float64)
        c = np.zeros((np_, n), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)

        with pytest.raises(ValueError):
            sb10ed(n, m, np_, ncon, nmeas, a, b, c, d)

    def test_ncon_larger_than_np1(self):
        """Test M2 > NP1 returns error (M2=NCON, NP1=NP-NMEAS)."""
        from ctrlsys import sb10ed

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
            sb10ed(n, m, np_, ncon, nmeas, a, b, c, d)

    def test_invalid_nmeas(self):
        """Test NMEAS > NP returns error."""
        from ctrlsys import sb10ed

        n = 2
        m = 4
        np_ = 3
        ncon = 1
        nmeas = 4

        a = np.eye(n, order='F', dtype=np.float64)
        b = np.zeros((n, m), order='F', dtype=np.float64)
        c = np.zeros((np_, n), order='F', dtype=np.float64)
        d = np.zeros((np_, m), order='F', dtype=np.float64)

        with pytest.raises(ValueError):
            sb10ed(n, m, np_, ncon, nmeas, a, b, c, d)


class TestSB10EDErrorCodes:
    """Test algorithmic error handling."""

    def test_d12_not_full_rank_returns_info_3(self):
        """
        Test D12 not full column rank returns info=3.

        Create system where D12 is zero to trigger rank deficiency.
        Dimensions: N=2, M=4, NP=4, NCON=1, NMEAS=1
        => M1=3, M2=1, NP1=3, NP2=1
        => Need NP2 <= M1 (1 <= 3) and M2 <= NP1 (1 <= 3) - OK
        """
        from ctrlsys import sb10ed

        n = 2
        m = 4
        np_ = 4
        ncon = 1
        nmeas = 1

        a = np.diag([0.5, 0.6]).astype(np.float64, order='F')
        b = np.ones((n, m), order='F', dtype=np.float64) * 0.1
        c = np.ones((np_, n), order='F', dtype=np.float64) * 0.1
        d = np.zeros((np_, m), order='F', dtype=np.float64)

        ak, bk, ck, dk, rcond, info = sb10ed(
            n, m, np_, ncon, nmeas, a, b, c, d
        )

        assert info > 0, f"Expected algorithmic failure, got info={info}"
