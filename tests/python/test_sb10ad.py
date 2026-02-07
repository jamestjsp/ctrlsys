"""
Tests for SB10AD: H-infinity optimal controller synthesis.

SB10AD computes an H-infinity optimal n-state controller K = (AK, BK, CK, DK)
using modified Glover's and Doyle's 1988 formulas, and the closed-loop
system G = (AC, BC, CC, DC).

The plant P has the form:
    | A  | B1  B2  |
P = |----|---------|
    | C1 | D11 D12 |
    | C2 | D21 D22 |

where B2 has NCON columns (control inputs) and C2 has NMEAS rows (measurements).

JOB modes:
  1: Bisection for gamma reduction
  2: Scan from gamma to 0
  3: Bisection then scanning
  4: Suboptimal controller only (current gamma)

Assumptions:
  (A1) (A,B2) stabilizable, (C2,A) detectable
  (A2) D12 full column rank, D21 full row rank
  (A3) [A-jwI, B2; C1, D12] full column rank for all w
  (A4) [A-jwI, B1; C2, D21] full row rank for all w
"""
import numpy as np
import pytest

try:
    import slicot
    HAS_SLICOT = True
except ImportError:
    HAS_SLICOT = False


@pytest.mark.skipif(not HAS_SLICOT, reason="slicot module not available")
class TestSB10AD:

    def test_job4_suboptimal(self):
        """
        Test JOB=4 (suboptimal controller only) with well-conditioned system.

        Uses the standard H-infinity benchmark problem from Doyle et al.
        D12 must have full column rank (here D12 = [0; 1])
        D21 must have full row rank (here D21 = [1, 0])

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        n = 2       # State dimension
        m = 2       # Total inputs: m1=1 external + m2=1 control
        np_ = 2     # Total outputs: np1=1 performance + np2=1 measurement
        ncon = 1    # Control inputs (m2)
        nmeas = 1   # Measurements (np2)

        A = np.array([[-1.0, 1.0],
                      [0.0, -2.0]], dtype=float, order='F')

        B = np.array([[1.0, 1.0],
                      [0.0, 1.0]], dtype=float, order='F')

        C = np.array([[1.0, 0.0],
                      [0.0, 1.0]], dtype=float, order='F')

        D = np.array([[0.0, 1.0],
                      [1.0, 0.0]], dtype=float, order='F')

        gamma = 5.0
        gtol = 1e-6
        actol = 0.0
        job = 4

        result = slicot.sb10ad(job, n, m, np_, ncon, nmeas, A, B, C, D, gamma, gtol, actol)
        ak, bk, ck, dk, ac, bc, cc, dc, gamma_out, rcond, info = result

        assert info == 0, f"SB10AD failed with info={info}"

        assert np.isfinite(ak).all(), "AK contains non-finite values"
        assert np.isfinite(bk).all(), "BK contains non-finite values"
        assert np.isfinite(ck).all(), "CK contains non-finite values"
        assert np.isfinite(dk).all(), "DK contains non-finite values"

        assert np.isfinite(ac).all(), "AC contains non-finite values"
        assert np.isfinite(bc).all(), "BC contains non-finite values"
        assert np.isfinite(cc).all(), "CC contains non-finite values"
        assert np.isfinite(dc).all(), "DC contains non-finite values"

        assert np.all(rcond >= 0) and np.all(rcond <= 1), "RCOND out of range [0,1]"

    def test_closed_loop_stability(self):
        """
        Verify closed-loop system is stable for successful synthesis.

        After H-infinity synthesis, all eigenvalues of AC should have negative
        real parts (continuous-time stability).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        n = 2
        m = 2       # m1=1, m2=1
        np_ = 2     # np1=1, np2=1
        ncon = 1
        nmeas = 1

        A = np.array([[-1.0, 1.0],
                      [0.0, -2.0]], dtype=float, order='F')

        B = np.array([[1.0, 1.0],
                      [0.0, 1.0]], dtype=float, order='F')

        C = np.array([[1.0, 0.0],
                      [0.0, 1.0]], dtype=float, order='F')

        D = np.array([[0.0, 1.0],
                      [1.0, 0.0]], dtype=float, order='F')

        gamma = 5.0
        gtol = 1e-4
        actol = 0.0
        job = 4

        result = slicot.sb10ad(job, n, m, np_, ncon, nmeas, A, B, C, D, gamma, gtol, actol)
        ak, bk, ck, dk, ac, bc, cc, dc, gamma_out, rcond, info = result

        assert info == 0, f"SB10AD failed with info={info}"

        eigenvalues = np.linalg.eigvals(ac)
        max_real = np.max(eigenvalues.real)

        assert max_real < 0, f"Closed-loop not stable: max(Re(eig(AC))) = {max_real}"

    def test_quick_return_zero_n(self):
        """
        Test quick return when N=0.
        """
        n = 0
        m = 2
        np_ = 2
        ncon = 1
        nmeas = 1

        A = np.zeros((1, 1), dtype=float, order='F')
        B = np.zeros((1, m), dtype=float, order='F')
        C = np.zeros((np_, 1), dtype=float, order='F')
        D = np.zeros((np_, m), dtype=float, order='F')

        gamma = 1.0
        gtol = 1e-6
        actol = 0.0
        job = 4

        result = slicot.sb10ad(job, n, m, np_, ncon, nmeas, A, B, C, D, gamma, gtol, actol)
        ak, bk, ck, dk, ac, bc, cc, dc, gamma_out, rcond, info = result

        assert info == 0, f"Quick return failed with info={info}"
        np.testing.assert_allclose(rcond, np.ones(4), rtol=1e-14)

    def test_gamma_too_small(self):
        """
        Test error INFO=6 when gamma is too small.

        When gamma <= max(sigma[D1111,D1112], sigma[D1111',D1121']),
        the controller is not admissible.
        """
        n = 2
        m = 3
        np_ = 3
        ncon = 1
        nmeas = 1

        A = np.array([[-1.0, 0.0],
                      [0.0, -1.0]], dtype=float, order='F')

        B = np.array([[1.0, 0.0, 1.0],
                      [0.0, 1.0, 0.5]], dtype=float, order='F')

        C = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]], dtype=float, order='F')

        D = np.array([[2.0, 0.0, 0.0],
                      [0.0, 2.0, 1.0],
                      [0.0, 0.0, 0.0]], dtype=float, order='F')

        gamma = 0.5
        gtol = 1e-6
        actol = 0.0
        job = 4

        result = slicot.sb10ad(job, n, m, np_, ncon, nmeas, A, B, C, D, gamma, gtol, actol)
        info = result[-1]

        assert info == 6, f"Expected info=6 for gamma too small, got {info}"

    def test_invalid_job(self):
        """
        Test error for invalid JOB parameter.
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

        gamma = 10.0
        gtol = 1e-6
        actol = 0.0
        job = 0

        result = slicot.sb10ad(job, n, m, np_, ncon, nmeas, A, B, C, D, gamma, gtol, actol)
        info = result[-1]

        assert info == -1, f"Expected info=-1 for invalid JOB, got {info}"

    def test_job1_bisection(self):
        """
        Test JOB=1 (bisection method for gamma reduction).

        Uses a well-conditioned system where bisection should converge.
        The returned gamma should be <= initial gamma.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        n = 2
        m = 2
        np_ = 2
        ncon = 1
        nmeas = 1

        A = np.array([[-1.0, 1.0],
                      [0.0, -2.0]], dtype=float, order='F')

        B = np.array([[1.0, 1.0],
                      [0.0, 1.0]], dtype=float, order='F')

        C = np.array([[1.0, 0.0],
                      [0.0, 1.0]], dtype=float, order='F')

        D = np.array([[0.0, 1.0],
                      [1.0, 0.0]], dtype=float, order='F')

        gamma_init = 10.0
        gtol = 0.1
        actol = 0.0
        job = 1

        result = slicot.sb10ad(job, n, m, np_, ncon, nmeas, A, B, C, D, gamma_init, gtol, actol)
        ak, bk, ck, dk, ac, bc, cc, dc, gamma_out, rcond, info = result

        assert info == 0, f"SB10AD JOB=1 failed with info={info}"

        assert gamma_out <= gamma_init, \
            f"Gamma should decrease: got {gamma_out}, started at {gamma_init}"

        eigenvalues = np.linalg.eigvals(ac)
        max_real = np.max(eigenvalues.real)
        assert max_real < 0, f"Closed-loop not stable: max(Re(eig(AC))) = {max_real}"

    def test_mimo_ncon2_nmeas2(self):
        """Test MIMO system with ncon=2, nmeas=2."""
        n = 4
        m = 4       # m1=2, m2=2
        np_ = 4     # np1=2, np2=2
        ncon = 2
        nmeas = 2

        a = np.array([[-1, 0.1, 0, 0],
                       [0, -2, 0.1, 0],
                       [0, 0, -3, 0.1],
                       [0, 0, 0, -4]], dtype=float, order='F')

        b = np.array([[1, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]], dtype=float, order='F')

        c = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]], dtype=float, order='F')

        d = np.array([[0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0]], dtype=float, order='F')

        gamma = 10.0
        gtol = 1e-4
        actol = 0.0
        job = 4

        result = slicot.sb10ad(job, n, m, np_, ncon, nmeas, a, b, c, d, gamma, gtol, actol)
        ak, bk, ck, dk, ac, bc, cc, dc, gamma_out, rcond, info = result

        assert info == 0, f"SB10AD MIMO failed with info={info}"

        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)

        eig_cl = np.linalg.eigvals(ac)
        assert np.all(eig_cl.real < 0), f"Closed-loop not stable: {eig_cl}"

        assert np.all(rcond >= 0) and np.all(rcond <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
