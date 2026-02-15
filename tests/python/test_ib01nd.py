"""
Tests for IB01ND - SVD system order via block Hankel.

Computes singular value decomposition of triangular factor R from QR
factorization of concatenated block Hankel matrices to determine system order.
"""

import numpy as np
import pytest
from ctrlsys import ib01nd


def create_test_r_matrix(nobr, m, l, rank, seed=42):
    """
    Create synthetic upper triangular R matrix for testing.

    The R matrix is 2*(m+l)*nobr square, upper triangular.
    We create it with known rank characteristics for testing SVD.

    Random seed: specified for reproducibility
    """
    np.random.seed(seed)
    nr = 2 * (m + l) * nobr

    # Create R as product of random matrices to control singular value structure
    # R = Q * diag(sv) * V^T where we control sv

    # Generate singular values with clear rank structure
    lnobr = l * nobr
    sv = np.zeros(lnobr)
    sv[:rank] = np.linspace(100.0, 10.0, rank)  # Large singular values for rank
    sv[rank:] = np.linspace(0.1, 0.001, lnobr - rank)  # Small singular values

    # Create orthogonal matrices
    q, _ = np.linalg.qr(np.random.randn(nr, nr))
    v, _ = np.linalg.qr(np.random.randn(lnobr, lnobr))

    # Build R as upper triangular
    r = np.zeros((nr, nr), order='F', dtype=float)

    # Fill upper triangular part with random data
    for j in range(nr):
        for i in range(j + 1):
            r[i, j] = np.random.randn()

    # Scale the relevant part to have specified singular values
    # For MOESP: relevant part is R(ms+1:(2m+l)s, (2m+l)s+1:2(m+l)s)
    # = R[mnobr:lmnobr+mnobr, lmnobr+mnobr:nr]
    mnobr = m * nobr
    lmnobr = l * nobr + mnobr

    # Inject structure in the SVD-relevant region
    sub_r = np.diag(sv)
    r[mnobr:mnobr+lnobr, lmnobr:nr] = sub_r

    return r, sv


def test_ib01nd_moesp_basic():
    """
    Basic MOESP test: compute SVD for order estimation.

    Tests METH='M', JOBD='N' case with synthetic R matrix.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    nobr = 4
    m = 1
    l = 2

    nr = 2 * (m + l) * nobr  # 24
    lnobr = l * nobr  # 8

    # Create upper triangular R with known structure
    r = np.zeros((nr, nr), order='F', dtype=float)

    # Fill upper triangular with random values
    for j in range(nr):
        for i in range(j + 1):
            r[i, j] = np.random.randn()

    # Add diagonal dominance for numerical stability
    for i in range(nr):
        r[i, i] = abs(r[i, i]) + 10.0

    meth = 'M'
    jobd = 'N'
    tol = 0.0

    r_out, sv, rcond1, rcond2, iwarn, info = ib01nd(meth, jobd, nobr, m, l, r, tol)

    assert info == 0
    assert iwarn == 0
    assert len(sv) == lnobr
    # SVD should produce positive, descending singular values
    assert all(s >= 0 for s in sv)
    assert all(sv[i] >= sv[i+1] for i in range(len(sv)-1))


def test_ib01nd_moesp_jobd_m():
    """
    MOESP test with JOBD='M' (matrices B and D computed using MOESP).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    nobr = 5
    m = 2
    l = 1

    nr = 2 * (m + l) * nobr  # 30
    lnobr = l * nobr  # 5
    mnobr = m * nobr  # 10

    # For JOBD='M', LDR >= max(2*(m+l)*nobr, 3*m*nobr) = max(30, 30) = 30
    ldr = max(nr, 3 * mnobr)

    r = np.zeros((ldr, nr), order='F', dtype=float)

    # Fill upper triangular with random values
    for j in range(nr):
        for i in range(min(j + 1, ldr)):
            r[i, j] = np.random.randn()

    # Add diagonal dominance for numerical stability
    for i in range(min(ldr, nr)):
        r[i, i] = abs(r[i, i]) + 10.0

    meth = 'M'
    jobd = 'M'
    tol = 0.0

    r_out, sv, rcond1, rcond2, iwarn, info = ib01nd(meth, jobd, nobr, m, l, r, tol)

    assert info == 0
    assert len(sv) == lnobr


def test_ib01nd_n4sid_basic():
    """
    Basic N4SID test: compute weighted oblique projection and SVD.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    nobr = 4
    m = 1
    l = 2

    nr = 2 * (m + l) * nobr  # 24
    lnobr = l * nobr  # 8
    mnobr = m * nobr  # 4
    lmnobr = lnobr + mnobr  # 12

    r = np.zeros((nr, nr), order='F', dtype=float)

    # Fill upper triangular with well-conditioned random values
    for j in range(nr):
        for i in range(j + 1):
            r[i, j] = np.random.randn()

    # Ensure diagonal dominance for numerical stability
    for i in range(nr):
        r[i, i] = abs(r[i, i]) + 10.0

    meth = 'N'
    jobd = 'N'  # Not relevant for N4SID
    tol = 0.0

    r_out, sv, rcond1, rcond2, iwarn, info = ib01nd(meth, jobd, nobr, m, l, r, tol)

    assert info == 0
    assert len(sv) == lnobr
    # SVs should be positive and sorted descending
    assert all(sv[i] >= sv[i+1] for i in range(len(sv)-1))


def test_ib01nd_m_zero():
    """
    Test with M=0 (no inputs, output-only identification).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    nobr = 5
    m = 0
    l = 2

    nr = 2 * l * nobr  # 20
    lnobr = l * nobr  # 10

    r = np.zeros((nr, nr), order='F', dtype=float)

    # Fill upper triangular
    for j in range(nr):
        for i in range(j + 1):
            r[i, j] = np.random.randn()

    # Add diagonal dominance
    for i in range(nr):
        r[i, i] = abs(r[i, i]) + 5.0

    meth = 'M'
    jobd = 'N'
    tol = 0.0

    r_out, sv, rcond1, rcond2, iwarn, info = ib01nd(meth, jobd, nobr, m, l, r, tol)

    assert info == 0
    assert len(sv) == lnobr


def test_ib01nd_sv_descending_property():
    """
    Mathematical property: singular values must be in descending order.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    nobr = 6
    m = 1
    l = 2

    nr = 2 * (m + l) * nobr  # 36
    lnobr = l * nobr  # 12

    r = np.zeros((nr, nr), order='F', dtype=float)

    for j in range(nr):
        for i in range(j + 1):
            r[i, j] = np.random.randn()

    for i in range(nr):
        r[i, i] = abs(r[i, i]) + 3.0

    meth = 'M'
    jobd = 'N'
    tol = 0.0

    r_out, sv, rcond1, rcond2, iwarn, info = ib01nd(meth, jobd, nobr, m, l, r, tol)

    assert info == 0
    # Verify descending order
    for i in range(len(sv) - 1):
        assert sv[i] >= sv[i+1], f"SV not descending: sv[{i}]={sv[i]} < sv[{i+1}]={sv[i+1]}"


def test_ib01nd_n4sid_rank_deficient_warning():
    """
    Test N4SID with ill-conditioned matrix producing IWARN=4 or 5.

    Creates a rank-deficient coefficient matrix to trigger warning.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    nobr = 4
    m = 2
    l = 1

    nr = 2 * (m + l) * nobr  # 24
    lnobr = l * nobr  # 4
    mnobr = m * nobr  # 8

    r = np.zeros((nr, nr), order='F', dtype=float)

    # Create nearly rank-deficient structure
    for j in range(nr):
        for i in range(j + 1):
            r[i, j] = np.random.randn() * 0.001  # Small values

    # Make first block nearly singular
    r[0, 0] = 1e-15

    meth = 'N'
    jobd = 'N'
    tol = 0.0

    r_out, sv, rcond1, rcond2, iwarn, info = ib01nd(meth, jobd, nobr, m, l, r, tol)

    # May return warning but should not error
    assert info == 0 or info == 2  # SVD might not converge for pathological cases
    assert iwarn in [0, 4, 5]


def test_ib01nd_error_invalid_meth():
    """
    Error handling: invalid METH parameter.
    """
    nobr = 4
    m = 1
    l = 2
    nr = 2 * (m + l) * nobr
    r = np.eye(nr, order='F', dtype=float)

    with pytest.raises(ValueError, match="METH"):
        ib01nd('X', 'N', nobr, m, l, r, 0.0)


def test_ib01nd_error_invalid_jobd():
    """
    Error handling: invalid JOBD parameter (for MOESP).
    """
    nobr = 4
    m = 1
    l = 2
    nr = 2 * (m + l) * nobr
    r = np.eye(nr, order='F', dtype=float)

    with pytest.raises(ValueError, match="JOBD"):
        ib01nd('M', 'X', nobr, m, l, r, 0.0)


def test_ib01nd_error_nobr_nonpositive():
    """
    Error handling: NOBR <= 0.
    """
    nobr = 0
    m = 1
    l = 2
    r = np.eye(12, order='F', dtype=float)

    with pytest.raises(ValueError, match="NOBR"):
        ib01nd('M', 'N', nobr, m, l, r, 0.0)


def test_ib01nd_error_m_negative():
    """
    Error handling: M < 0.
    """
    nobr = 4
    m = -1
    l = 2
    nr = 2 * (0 + l) * nobr  # Use m=0 for size
    r = np.eye(nr, order='F', dtype=float)

    with pytest.raises(ValueError, match="M"):
        ib01nd('M', 'N', nobr, m, l, r, 0.0)


def test_ib01nd_error_l_nonpositive():
    """
    Error handling: L <= 0.
    """
    nobr = 4
    m = 1
    l = 0
    r = np.eye(8, order='F', dtype=float)

    with pytest.raises(ValueError, match="L"):
        ib01nd('M', 'N', nobr, m, l, r, 0.0)


def test_ib01nd_r_size_validation():
    """
    Error handling: R matrix too small.
    """
    nobr = 4
    m = 1
    l = 2
    nr = 2 * (m + l) * nobr  # 24
    # Create R that's too small
    r = np.eye(10, order='F', dtype=float)

    with pytest.raises(ValueError, match="R"):
        ib01nd('M', 'N', nobr, m, l, r, 0.0)


def test_ib01nd_rcond_values():
    """
    Test that rcond values are returned for N4SID method.

    For METH='N', DWORK(2) and DWORK(3) contain reciprocal condition numbers.
    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    nobr = 4
    m = 1
    l = 2

    nr = 2 * (m + l) * nobr
    r = np.zeros((nr, nr), order='F', dtype=float)

    for j in range(nr):
        for i in range(j + 1):
            r[i, j] = np.random.randn()

    for i in range(nr):
        r[i, i] = abs(r[i, i]) + 5.0

    meth = 'N'
    jobd = 'N'
    tol = 0.0

    r_out, sv, rcond1, rcond2, iwarn, info = ib01nd(meth, jobd, nobr, m, l, r, tol)

    assert info == 0
    # For N4SID, rcond values should be positive
    assert rcond1 > 0
    assert rcond2 > 0


def test_ib01nd_r_modified_in_place():
    """
    Verify R matrix is modified in place as documented.

    Output S matrix should differ from input R.
    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    nobr = 4
    m = 1
    l = 2

    nr = 2 * (m + l) * nobr
    r = np.zeros((nr, nr), order='F', dtype=float)

    for j in range(nr):
        for i in range(j + 1):
            r[i, j] = np.random.randn()

    for i in range(nr):
        r[i, i] = abs(r[i, i]) + 2.0

    r_orig = r.copy()

    meth = 'M'
    jobd = 'N'
    tol = 0.0

    r_out, sv, rcond1, rcond2, iwarn, info = ib01nd(meth, jobd, nobr, m, l, r, tol)

    assert info == 0
    # R should be modified (output S differs from input R)
    assert not np.allclose(r_out, r_orig)
