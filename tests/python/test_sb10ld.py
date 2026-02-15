"""
Tests for SB10LD: Closed-loop system matrices computation.

SB10LD computes the closed-loop system matrices G = (AC, BC, CC, DC) from:
- Open-loop plant P = (A, B, C, D)
- Controller K = (AK, BK, CK, DK)

The closed-loop interconnection involves feedback:
  u = u1 + u2 where u2 = K * y2 (controller output)
  y = y1 + y2 where y2 is measured output
"""

import numpy as np
import pytest

try:
    import ctrlsys
    HAS_SLICOT = True
except ImportError:
    HAS_SLICOT = False


@pytest.mark.skipif(not HAS_SLICOT, reason="slicot module not available")
class TestSB10LD:

    def test_basic(self):
        """
        Test basic closed-loop system computation.

        Uses small 2x2 system with 1 control input and 1 measurement.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        n = 2      # State dimension
        m = 2      # Total inputs (m1=1 external, m2=1 control)
        np_ = 2    # Total outputs (np1=1 performance, np2=1 measurement)
        ncon = 1   # Control inputs (m2)
        nmeas = 1  # Measurements (np2)

        m1 = m - ncon      # 1 external input
        np1 = np_ - nmeas  # 1 performance output

        A = np.array([[-1.0, 0.5],
                      [0.0, -2.0]], dtype=float, order='F')

        B = np.array([[1.0, 0.5],
                      [0.0, 1.0]], dtype=float, order='F')

        C = np.array([[1.0, 0.0],
                      [0.0, 1.0]], dtype=float, order='F')

        D = np.array([[0.0, 0.0],
                      [0.0, 0.1]], dtype=float, order='F')

        AK = np.array([[-3.0, 1.0],
                       [0.5, -4.0]], dtype=float, order='F')

        BK = np.array([[2.0],
                       [1.0]], dtype=float, order='F')

        CK = np.array([[1.0, 0.5]], dtype=float, order='F')

        DK = np.array([[0.2]], dtype=float, order='F')

        result = slicot.sb10ld(n, m, np_, ncon, nmeas, A, B, C, D, AK, BK, CK, DK)
        ac, bc, cc, dc, info = result

        assert info == 0, f"SB10LD failed with info={info}"

        n2 = 2 * n
        assert ac.shape == (n2, n2)
        assert bc.shape == (n2, m1)
        assert cc.shape == (np1, n2)
        assert dc.shape == (np1, m1)

        assert np.isfinite(ac).all()
        assert np.isfinite(bc).all()
        assert np.isfinite(cc).all()
        assert np.isfinite(dc).all()

    def test_closed_loop_structure(self):
        """
        Validate closed-loop system structure.

        For the special case DK=0 and D22=0, the closed-loop formulas simplify to:
        - AC(1:n, 1:n)      = A (since DK*inv(I-D22*DK)*C2 = 0)
        - AC(1:n, n+1:2n)   = B2*CK (since inv(I-DK*D22) = I when DK=0)
        - AC(n+1:2n, 1:n)   = BK*C2 (since inv(I-D22*DK) = I when D22=0)
        - AC(n+1:2n, n+1:2n)= AK (since BK*D22*inv(I-DK*D22)*CK = 0 when D22=0)

        where B2 = B[:,m1:], C2 = C[np1:,:], D22 = D[np1:,m1:]

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        n = 3
        m = 3      # m1=2, m2=1
        np_ = 3    # np1=2, np2=1
        ncon = 1
        nmeas = 1
        m1 = m - ncon
        np1 = np_ - nmeas

        A = np.array([[-1.0, 0.2, 0.1],
                      [0.3, -1.5, 0.2],
                      [0.1, 0.2, -2.0]], dtype=float, order='F')

        B = np.array([[1.0, 0.0, 0.5],
                      [0.0, 1.0, 0.3],
                      [0.2, 0.1, 0.8]], dtype=float, order='F')

        C = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.5, 1.0]], dtype=float, order='F')

        D = np.zeros((np_, m), dtype=float, order='F')

        AK = np.array([[-2.0, 0.5, 0.0],
                       [0.3, -2.5, 0.2],
                       [0.0, 0.1, -3.0]], dtype=float, order='F')

        BK = np.array([[1.5],
                       [0.8],
                       [0.5]], dtype=float, order='F')

        CK = np.array([[0.5, 0.3, 0.2]], dtype=float, order='F')

        DK = np.zeros((ncon, nmeas), dtype=float, order='F')

        result = slicot.sb10ld(n, m, np_, ncon, nmeas, A, B, C, D, AK, BK, CK, DK)
        ac, bc, cc, dc, info = result

        assert info == 0

        B2 = B[:, m1:]
        C2 = C[np1:, :]

        AC11_expected = A
        AC12_expected = B2 @ CK
        AC21_expected = BK @ C2
        AC22_expected = AK

        np.testing.assert_allclose(ac[:n, :n], AC11_expected, rtol=1e-13)
        np.testing.assert_allclose(ac[:n, n:], AC12_expected, rtol=1e-13)
        np.testing.assert_allclose(ac[n:, :n], AC21_expected, rtol=1e-13)
        np.testing.assert_allclose(ac[n:, n:], AC22_expected, rtol=1e-13)

    def test_singular_matrix_error(self):
        """
        Test error handling when I - D22*DK is singular.

        If D22*DK = I, then I - D22*DK = 0 (singular), should return info=1.
        """
        n = 2
        m = 2
        np_ = 2
        ncon = 1
        nmeas = 1

        A = np.array([[-1.0, 0.0],
                      [0.0, -1.0]], dtype=float, order='F')
        B = np.array([[1.0, 1.0],
                      [0.0, 1.0]], dtype=float, order='F')
        C = np.array([[1.0, 0.0],
                      [0.0, 1.0]], dtype=float, order='F')
        D = np.array([[0.0, 0.0],
                      [0.0, 1.0]], dtype=float, order='F')

        AK = np.array([[-1.0, 0.0],
                       [0.0, -1.0]], dtype=float, order='F')
        BK = np.array([[1.0],
                       [1.0]], dtype=float, order='F')
        CK = np.array([[1.0, 0.0]], dtype=float, order='F')
        DK = np.array([[1.0]], dtype=float, order='F')

        result = slicot.sb10ld(n, m, np_, ncon, nmeas, A, B, C, D, AK, BK, CK, DK)
        info = result[-1]

        assert info == 1, f"Expected info=1 for singular I-D22*DK, got {info}"

    def test_quick_return(self):
        """
        Test quick return when any dimension is zero.
        """
        n = 0
        m = 1
        np_ = 1
        ncon = 1
        nmeas = 0

        A = np.zeros((1, 1), dtype=float, order='F')
        B = np.zeros((1, 1), dtype=float, order='F')
        C = np.zeros((1, 1), dtype=float, order='F')
        D = np.zeros((1, 1), dtype=float, order='F')
        AK = np.zeros((1, 1), dtype=float, order='F')
        BK = np.zeros((1, 1), dtype=float, order='F')
        CK = np.zeros((1, 1), dtype=float, order='F')
        DK = np.zeros((1, 1), dtype=float, order='F')

        result = slicot.sb10ld(n, m, np_, ncon, nmeas, A, B, C, D, AK, BK, CK, DK)
        info = result[-1]

        assert info == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
