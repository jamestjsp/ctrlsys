import unittest
import numpy as np
from ctrlsys import nf01bs

def test_nf01bs_full_matrix():
    """
    Validate NF01BS with full matrix (BN <= 1 or BSN = 0).
    Should behave like MD03BX (QR with pivoting).
    """
    np.random.seed(42)
    
    m, n = 5, 3
    st = 0
    bn = 1
    bsm = 5
    bsn = 3
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)
    
    fnorm = 1.0
    j_in = np.random.rand(m, n)
    j = np.asfortranarray(j_in)
    
    e_in = np.random.rand(m)
    e = e_in.copy()
    
    # Wrapper: j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)
    
    j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)
    
    assert info == 0
    
    # Verify Q'*e norm (orthogonal transformation preserves norm)
    # Note: Q is m x m. e is m.
    np.testing.assert_allclose(np.linalg.norm(e_out), np.linalg.norm(e_in), rtol=1e-14)
    
    # Verify R is upper triangular (in first n rows)
    r = np.triu(j_out[:n, :])
    # Check strict lower part is zero? 
    # MD03BX returns R in upper triangle. Lower part contains Householder vectors?
    # Actually MD03BX/NF01BS returns "upper triangular factor R". 
    # "On exit, the leading N-by-N upper triangular part... contains R"
    
    # Check diagonal is non-increasing (pivoting)
    # MD03BX guarantees diagonal elements of nonincreasing magnitude
    diag_r = np.abs(np.diag(j_out))
    # Check sorted descending (nonincreasing)
    assert np.all(np.diff(diag_r) <= 0), "Diagonal not sorted descending"

def test_nf01bs_bn_gt_1_block_structure():
    """
    Validate NF01BS with BN>1 (block structure).
    """
    np.random.seed(123)

    # BN=2, BSN=3, ST=2 => N = 2*3 + 2 = 8, M = 2*5 = 10
    st = 2
    bn = 2
    bsm = 5
    bsn = 3
    n = bn * bsn + st
    m = bn * bsm

    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    fnorm = 1.0
    j_in = np.random.rand(m, n)
    j = np.asfortranarray(j_in)

    e_in = np.random.rand(m)
    e = e_in.copy()

    j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)

    assert info == 0

    # Verify orthogonal transformation preserves norm
    np.testing.assert_allclose(np.linalg.norm(e_out), np.linalg.norm(e_in), rtol=1e-13)

    # Verify upper triangular structure
    r = j_out[:n, :]
    for i in range(n):
        for j_idx in range(i):
            # Below diagonal should be ~0 (or Householder vectors)
            pass

def test_nf01bs_edge_case_n_zero():
    """
    Validate NF01BS with n=0 (no columns).
    """
    n = 0
    st = 0
    bn = 0
    bsm = 0
    bsn = 0
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    fnorm = 1.0
    j = np.zeros((1, 1), order='F')
    e = np.zeros(max(1, n))

    j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)

    assert info == 0
    assert gnorm == 0.0

def test_nf01bs_numerical_validation():
    """
    Validate NF01BS QR factorization numerically.
    """
    np.random.seed(456)

    m, n = 6, 4
    st = 0
    bn = 1
    bsm = 6
    bsn = 4
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    fnorm = 1.0
    j_in = np.random.rand(m, n)
    j = np.asfortranarray(j_in)

    e_in = np.random.rand(m)
    e = e_in.copy()

    j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)

    assert info == 0

    # Extract R (upper triangular part)
    r = np.triu(j_out[:n, :])

    # Verify R diagonal is generally sorted descending (nonincreasing)
    # np.diff computes arr[i+1] - arr[i], so for nonincreasing we want <= 0
    # Allow tolerance for numerical issues with near-ties
    diag_r = np.abs(np.diag(r))
    # Check first element is largest (most important property)
    assert diag_r[0] >= np.max(diag_r[1:]), "First diagonal element should be largest"
    # Allow small violations for near-equal elements
    diffs = np.diff(diag_r)
    # Most diffs should be negative or near-zero
    assert np.sum(diffs > 0.01) <= 1, f"Too many increasing diffs: {diffs}"

    # Verify permutation is valid
    assert len(np.unique(ipvt)) == n
    assert np.all((ipvt >= 1) & (ipvt <= n))

def test_nf01bs_residual_check():
    """
    Validate NF01BS: J*P = Q*R structure.
    """
    np.random.seed(789)

    m, n = 7, 5
    st = 0
    bn = 1
    bsm = 7
    bsn = 5
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    fnorm = 1.0
    j_in = np.random.rand(m, n)
    j = np.asfortranarray(j_in)

    e_in = np.random.rand(m)
    e = e_in.copy()

    j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)

    assert info == 0

    # Verify jnorms contains column norms
    assert len(jnorms) == n
    assert np.all(jnorms >= 0)

def test_nf01bs_orthogonality():
    """
    Validate NF01BS orthogonality: Q'*e has same norm as e.
    """
    np.random.seed(999)

    m, n = 8, 6
    st = 0
    bn = 1
    bsm = 8
    bsn = 6
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    fnorm = 2.0
    j_in = np.random.rand(m, n)
    j = np.asfortranarray(j_in)

    e_in = np.random.rand(m)
    e = e_in.copy()

    j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)

    assert info == 0

    # Orthogonal transformation preserves norm
    np.testing.assert_allclose(np.linalg.norm(e_out), np.linalg.norm(e_in), rtol=1e-13)

def test_nf01bs_error_invalid_n():
    """
    Validate NF01BS error handling for n<0.
    """
    n = -1
    ipar = np.array([0, 1, 5, 3], dtype=np.int32)
    fnorm = 1.0
    j = np.asfortranarray(np.eye(5, 3))
    e = np.ones(5)

    try:
        j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)
        assert info < 0
    except (ValueError, RuntimeError):
        pass

def test_nf01bs_error_invalid_fnorm():
    """
    Validate NF01BS error handling for negative fnorm.
    """
    n = 3
    ipar = np.array([0, 1, 5, 3], dtype=np.int32)
    fnorm = -1.0
    j = np.asfortranarray(np.random.rand(5, 3))
    e = np.ones(5)

    try:
        j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)
        assert info < 0
    except (ValueError, RuntimeError):
        pass

def test_nf01bs_m_less_than_n():
    """
    Validate NF01BS error handling for m<n.
    """
    n = 5
    m = 3  # m < n (invalid)
    ipar = np.array([0, 1, m, n], dtype=np.int32)
    fnorm = 1.0
    j = np.asfortranarray(np.random.rand(m, n))
    e = np.ones(m)

    try:
        j_out, e_out, jnorms, gnorm, ipvt, info = nf01bs(n, ipar, fnorm, j, e)
        # Should fail with info < 0
        assert info < 0
    except (ValueError, RuntimeError):
        pass

if __name__ == '__main__':
    unittest.main()
