"""
Tests for SB08HD: State-space from right coprime factorization.

Constructs G = (A,B,C,D) from factors Q = (AQR,BQR,CQ,DQ) and R = (AQR,BQR,CR,DR)
of the right coprime factorization G = Q * R^{-1}.

Formulas:
    A = AQR - BQR * DR^{-1} * CR
    B = BQR * DR^{-1}
    C = CQ - DQ * DR^{-1} * CR
    D = DQ * DR^{-1}

Mathematical properties tested:
- Reconstruction from coprime factors
- State equation consistency
- INFO codes for singular DR

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb08hd_basic():
    """
    Test basic reconstruction from coprime factors.

    Random seed: 42 (for reproducibility)

    Creates a known system G and its coprime factors Q, R, then verifies
    that sb08hd reconstructs G correctly.
    """
    from ctrlsys import sb08hd

    np.random.seed(42)

    n, m, p = 3, 2, 2

    # Create random well-conditioned system components
    # DR must be invertible (we use identity + small perturbation)
    dr = np.eye(m, dtype=float, order='F') + 0.1 * np.random.randn(m, m)
    dr = np.asfortranarray(dr)

    # Random matrices for Q and R factors
    aqr = np.random.randn(n, n).astype(float, order='F')
    bqr = np.random.randn(n, m).astype(float, order='F')
    cq = np.random.randn(p, n).astype(float, order='F')
    dq = np.random.randn(p, m).astype(float, order='F')
    cr = np.random.randn(m, n).astype(float, order='F')

    # Compute expected outputs using formulas from documentation
    dr_inv = np.linalg.inv(dr)
    a_expected = aqr - bqr @ dr_inv @ cr
    b_expected = bqr @ dr_inv
    c_expected = cq - dq @ dr_inv @ cr
    d_expected = dq @ dr_inv

    # Call sb08hd
    # Inputs: n, m, p, a (AQR), b (BQR), c (CQ), d (DQ), cr, dr
    # a, b, c, d are modified in-place
    # dr is modified to contain LU factorization
    a = aqr.copy(order='F')
    b = bqr.copy(order='F')
    c = cq.copy(order='F')
    d = dq.copy(order='F')
    dr_input = dr.copy(order='F')

    rcond, info = sb08hd(a, b, c, d, cr, dr_input)

    assert info == 0, f"sb08hd failed with info={info}"
    assert rcond > 1e-15, f"DR should be well-conditioned, rcond={rcond}"

    # Validate numerical correctness
    assert_allclose(a, a_expected, rtol=1e-13, atol=1e-14)
    assert_allclose(b, b_expected, rtol=1e-13, atol=1e-14)
    assert_allclose(c, c_expected, rtol=1e-13, atol=1e-14)
    assert_allclose(d, d_expected, rtol=1e-13, atol=1e-14)


def test_sb08hd_identity_dr():
    """
    Test with DR = identity matrix (simplest case).

    When DR = I:
        A = AQR - BQR * CR
        B = BQR
        C = CQ - DQ * CR
        D = DQ

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb08hd

    np.random.seed(123)

    n, m, p = 4, 2, 3

    # Create random matrices
    aqr = np.random.randn(n, n).astype(float, order='F')
    bqr = np.random.randn(n, m).astype(float, order='F')
    cq = np.random.randn(p, n).astype(float, order='F')
    dq = np.random.randn(p, m).astype(float, order='F')
    cr = np.random.randn(m, n).astype(float, order='F')
    dr = np.eye(m, dtype=float, order='F')

    # Expected outputs (DR = I simplifies formulas)
    a_expected = aqr - bqr @ cr
    b_expected = bqr.copy()
    c_expected = cq - dq @ cr
    d_expected = dq.copy()

    a = aqr.copy(order='F')
    b = bqr.copy(order='F')
    c = cq.copy(order='F')
    d = dq.copy(order='F')
    dr_input = dr.copy(order='F')

    rcond, info = sb08hd(a, b, c, d, cr, dr_input)

    assert info == 0
    assert rcond > 0.99, f"Identity DR should have rcond ~1, got {rcond}"

    assert_allclose(a, a_expected, rtol=1e-14, atol=1e-15)
    assert_allclose(b, b_expected, rtol=1e-14, atol=1e-15)
    assert_allclose(c, c_expected, rtol=1e-14, atol=1e-15)
    assert_allclose(d, d_expected, rtol=1e-14, atol=1e-15)


def test_sb08hd_singular_dr():
    """
    Test error handling for singular DR matrix.

    When DR is exactly singular, info should be 1.
    """
    from ctrlsys import sb08hd

    n, m, p = 2, 2, 2

    # Create singular DR (rank deficient)
    dr = np.array([[1.0, 2.0],
                   [2.0, 4.0]], order='F', dtype=float)  # Second row = 2 * first row

    aqr = np.eye(n, dtype=float, order='F')
    bqr = np.eye(n, m, dtype=float, order='F')
    cq = np.eye(p, n, dtype=float, order='F')
    dq = np.eye(p, m, dtype=float, order='F')
    cr = np.eye(m, n, dtype=float, order='F')

    a = aqr.copy(order='F')
    b = bqr.copy(order='F')
    c = cq.copy(order='F')
    d = dq.copy(order='F')
    dr_input = dr.copy(order='F')

    rcond, info = sb08hd(a, b, c, d, cr, dr_input)

    assert info == 1, f"Expected info=1 for singular DR, got info={info}"
    assert rcond == 0.0, f"Expected rcond=0 for singular DR, got {rcond}"


def test_sb08hd_nearly_singular_dr():
    """
    Test warning for numerically singular DR (ill-conditioned).

    When DR is numerically singular but not exactly singular, info should be 2.
    """
    from ctrlsys import sb08hd

    n, m, p = 2, 2, 2

    # Create nearly singular DR
    eps = np.finfo(float).eps
    dr = np.array([[1.0, 0.0],
                   [0.0, eps * 0.1]], order='F', dtype=float)

    aqr = np.eye(n, dtype=float, order='F')
    bqr = np.eye(n, m, dtype=float, order='F')
    cq = np.eye(p, n, dtype=float, order='F')
    dq = np.eye(p, m, dtype=float, order='F')
    cr = np.eye(m, n, dtype=float, order='F')

    a = aqr.copy(order='F')
    b = bqr.copy(order='F')
    c = cq.copy(order='F')
    d = dq.copy(order='F')
    dr_input = dr.copy(order='F')

    rcond, info = sb08hd(a, b, c, d, cr, dr_input)

    assert info == 2, f"Expected info=2 for nearly singular DR, got info={info}"
    assert rcond < eps, f"Expected tiny rcond, got {rcond}"


def test_sb08hd_quick_return_m_zero():
    """
    Test quick return when M=0 (no inputs).

    With M=0, the routine should return immediately with rcond=1.
    """
    from ctrlsys import sb08hd

    n, m, p = 3, 0, 2

    # Empty arrays for m=0
    aqr = np.random.randn(n, n).astype(float, order='F')
    bqr = np.zeros((n, 0), dtype=float, order='F')
    cq = np.random.randn(p, n).astype(float, order='F')
    dq = np.zeros((p, 0), dtype=float, order='F')
    cr = np.zeros((0, n), dtype=float, order='F')
    dr = np.zeros((0, 0), dtype=float, order='F')

    a = aqr.copy(order='F')
    b = bqr.copy(order='F')
    c = cq.copy(order='F')
    d = dq.copy(order='F')
    dr_input = dr.copy(order='F')

    rcond, info = sb08hd(a, b, c, d, cr, dr_input)

    assert info == 0
    assert rcond == 1.0


def test_sb08hd_state_space_consistency():
    """
    Validate state-space equations: verify G reconstructs from Q and R.

    Given G = Q * R^{-1}, verify the transfer function relationship holds
    by checking that G(s) * R(s) = Q(s) at specific frequency points.

    For a state-space system (A,B,C,D), the transfer function is:
        G(s) = C * (sI - A)^{-1} * B + D

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb08hd

    np.random.seed(456)

    n, m, p = 3, 2, 2

    # Create stable AQR (negative eigenvalues)
    aqr = -np.eye(n, dtype=float) + 0.3 * np.random.randn(n, n)
    aqr = np.asfortranarray(aqr, dtype=float)

    bqr = np.random.randn(n, m).astype(float, order='F')
    cq = np.random.randn(p, n).astype(float, order='F')
    dq = np.random.randn(p, m).astype(float, order='F')
    cr = np.random.randn(m, n).astype(float, order='F')
    dr = np.eye(m, dtype=float, order='F') + 0.2 * np.random.randn(m, m)
    dr = np.asfortranarray(dr)

    # Compute G using sb08hd
    a = aqr.copy(order='F')
    b = bqr.copy(order='F')
    c = cq.copy(order='F')
    d = dq.copy(order='F')
    dr_input = dr.copy(order='F')

    rcond, info = sb08hd(a, b, c, d, cr, dr_input)
    assert info == 0

    # G's state-space: (a, b, c, d) after call
    # Q's state-space: (aqr, bqr, cq, dq)
    # R's state-space: (aqr, bqr, cr, dr)

    # Verify G = Q * R^{-1} by checking G(s)*R(s) = Q(s) at several frequencies
    test_freqs = [0.1, 1.0, 10.0]

    for freq in test_freqs:
        s = 1j * freq

        # G(s) = c * (sI - a)^{-1} * b + d
        G_s = c @ np.linalg.solve(s * np.eye(n) - a, b) + d

        # Q(s) = cq * (sI - aqr)^{-1} * bqr + dq
        Q_s = cq @ np.linalg.solve(s * np.eye(n) - aqr, bqr) + dq

        # R(s) = cr * (sI - aqr)^{-1} * bqr + dr
        R_s = cr @ np.linalg.solve(s * np.eye(n) - aqr, bqr) + dr

        # Verify G(s) * R(s) = Q(s)
        GR_s = G_s @ R_s
        assert_allclose(GR_s, Q_s, rtol=1e-12, atol=1e-14,
                        err_msg=f"Transfer function mismatch at s={s}")
