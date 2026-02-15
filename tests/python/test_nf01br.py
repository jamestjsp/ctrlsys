import unittest
import numpy as np
from ctrlsys import nf01br

def test_nf01br_full_rank():
    """
    Validate NF01BR with full rank matrix (BN <= 1 or BSN = 0).
    Solves R*x = b.
    """
    np.random.seed(42)
    
    # Case 1: Full upper triangular matrix (BN <= 1)
    # n = 4, ipar structure: st, bn, bsm, bsn
    # Let's use BN=1, BSM=4, BSN=4, ST=0 => N=4.
    
    n = 4
    st = 0
    bn = 1
    bsm = 4
    bsn = 4
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)
    
    ldr = n
    r = np.triu(np.random.rand(n, n))
    # Ensure non-singular
    for i in range(n):
        r[i, i] += 2.0
        
    r_arr = np.asfortranarray(r)
    
    b = np.random.rand(n)
    b_in = b.copy()
    
    # sdiag and s are not used for UPLO='U'
    sdiag = np.zeros(n)
    s = np.zeros((1, 1))
    ranks = np.zeros(bn+1, dtype=np.int32)
    
    # Call nf01br(cond, uplo, trans, n, ipar, r, sdiag, s, b, ranks, tol)
    # cond='N', uplo='U', trans='N'
    
    b_out, ranks_out, info = nf01br('N', 'U', 'N', n, ipar, r_arr, sdiag, s, b_in, ranks, 0.0)
    
    assert info == 0
    
    # Verify R*x = b
    x = b_out
    b_rec = r @ x
    np.testing.assert_allclose(b_rec, b, rtol=1e-14)

def test_nf01br_transpose():
    """
    Validate NF01BR with full rank matrix, solving R'*x = b.
    """
    np.random.seed(42)
    n = 4
    st = 0
    bn = 1
    bsm = 4
    bsn = 4
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    r = np.triu(np.random.rand(n, n))
    for i in range(n): r[i, i] += 2.0
    r_arr = np.asfortranarray(r)

    b = np.random.rand(n)
    b_in = b.copy()

    sdiag = np.zeros(n)
    s = np.zeros((1, 1))
    ranks = np.zeros(bn+1, dtype=np.int32)

    # Solve R'*x = b
    b_out, ranks_out, info = nf01br('N', 'U', 'T', n, ipar, r_arr, sdiag, s, b_in, ranks, 0.0)

    assert info == 0

    x = b_out
    b_rec = r.T @ x
    np.testing.assert_allclose(b_rec, b, rtol=1e-14)

def test_nf01br_bn_gt_1_block_structure():
    """
    Validate NF01BR with BN>1 (block structure).
    For BN>1, R is stored in compressed format: N x (BSN+ST).
    """
    np.random.seed(123)

    # BN=2, BSN=3, ST=2 => N = 2*3 + 2 = 8
    st = 2
    bn = 2
    bsm = 5
    bsn = 3
    n = bn * bsn + st
    nc = bsn + st  # Compressed column count

    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    # Create compressed block upper triangular matrix (N x NC)
    # Structure:
    #   R_1 (BSN x BSN) | L_1 (BSN x ST)
    #   R_2 (BSN x BSN) | L_2 (BSN x ST)
    #     X  (ST x BSN) | R_3 (ST  x ST)
    r_compressed = np.zeros((n, nc), order='F')

    # Fill R_1 (rows 0:3, cols 0:3)
    r1 = np.triu(np.random.rand(bsn, bsn))
    for i in range(bsn): r1[i, i] += 2.0
    r_compressed[:bsn, :bsn] = r1

    # Fill L_1 (rows 0:3, cols 3:5)
    l1 = np.random.rand(bsn, st)
    r_compressed[:bsn, bsn:] = l1

    # Fill R_2 (rows 3:6, cols 0:3)
    r2 = np.triu(np.random.rand(bsn, bsn))
    for i in range(bsn): r2[i, i] += 2.0
    r_compressed[bsn:2*bsn, :bsn] = r2

    # Fill L_2 (rows 3:6, cols 3:5)
    l2 = np.random.rand(bsn, st)
    r_compressed[bsn:2*bsn, bsn:] = l2

    # Fill R_3 (rows 6:8, cols 3:5) - last block
    r3 = np.triu(np.random.rand(st, st))
    for i in range(st): r3[i, i] += 2.0
    r_compressed[2*bsn:, bsn:] = r3

    r_arr = np.asfortranarray(r_compressed)

    b = np.random.rand(n)
    b_in = b.copy()

    sdiag = np.zeros(n)
    s = np.zeros((1, 1), order='F')
    ranks = np.zeros(bn+1, dtype=np.int32)

    b_out, ranks_out, info = nf01br('N', 'U', 'N', n, ipar, r_arr, sdiag, s, b_in, ranks, 0.0)

    assert info == 0

    # Reconstruct full matrix from compressed format to verify
    r_full = np.zeros((n, n))
    # R_1 block
    r_full[:bsn, :bsn] = r_compressed[:bsn, :bsn]
    r_full[:bsn, 2*bsn:] = r_compressed[:bsn, bsn:]  # L_1
    # R_2 block
    r_full[bsn:2*bsn, bsn:2*bsn] = r_compressed[bsn:2*bsn, :bsn]
    r_full[bsn:2*bsn, 2*bsn:] = r_compressed[bsn:2*bsn, bsn:]  # L_2
    # R_3 block
    r_full[2*bsn:, 2*bsn:] = r_compressed[2*bsn:, bsn:]

    x = b_out
    b_rec = r_full @ x
    np.testing.assert_allclose(b_rec, b, rtol=1e-13)

def test_nf01br_edge_case_n_zero():
    """
    Validate NF01BR with n=0 (empty system).
    """
    n = 0
    st = 0
    bn = 0
    bsm = 0
    bsn = 0
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    r_arr = np.zeros((1, 1), order='F')
    b_in = np.zeros(max(1, n))
    sdiag = np.zeros(max(1, n))
    s = np.zeros((1, 1), order='F')
    ranks = np.zeros(max(1, bn+1), dtype=np.int32)

    b_out, ranks_out, info = nf01br('N', 'U', 'N', n, ipar, r_arr, sdiag, s, b_in, ranks, 0.0)

    assert info == 0

def test_nf01br_condition_estimation():
    """
    Validate NF01BR with condition estimation (COND='E').
    """
    np.random.seed(456)

    n = 4
    st = 0
    bn = 1
    bsm = 4
    bsn = 4
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    r = np.triu(np.random.rand(n, n))
    for i in range(n):
        r[i, i] += 2.0
    r_arr = np.asfortranarray(r)

    b = np.random.rand(n)
    b_in = b.copy()

    sdiag = np.zeros(n)
    s = np.zeros((1, 1), order='F')
    ranks = np.zeros(bn+1, dtype=np.int32)

    # COND='E' requires workspace
    b_out, ranks_out, info = nf01br('E', 'U', 'N', n, ipar, r_arr, sdiag, s, b_in, ranks, 1e-10)

    assert info == 0

    x = b_out
    b_rec = r @ x
    np.testing.assert_allclose(b_rec, b, rtol=1e-13)

    # Check ranks are determined
    assert ranks_out[0] <= n

def test_nf01br_rank_deficient():
    """
    Validate NF01BR with rank-deficient matrix.
    """
    np.random.seed(789)

    n = 5
    st = 0
    bn = 1
    bsm = 5
    bsn = 5
    ipar = np.array([st, bn, bsm, bsn], dtype=np.int32)

    # Create rank-deficient upper triangular (set last diagonal to ~0)
    r = np.triu(np.random.rand(n, n))
    for i in range(n):
        r[i, i] += 2.0
    r[n-1, n-1] = 1e-14  # Make nearly singular
    r_arr = np.asfortranarray(r)

    b = np.random.rand(n)
    b_in = b.copy()

    sdiag = np.zeros(n)
    s = np.zeros((1, 1), order='F')
    ranks = np.zeros(bn+1, dtype=np.int32)

    # Use condition estimation with tolerance
    b_out, ranks_out, info = nf01br('E', 'U', 'N', n, ipar, r_arr, sdiag, s, b_in, ranks, 1e-10)

    assert info == 0
    # Rank should be less than n
    assert ranks_out[0] < n

def test_nf01br_error_invalid_cond():
    """
    Validate NF01BR error handling for invalid COND parameter.
    """
    n = 4
    ipar = np.array([0, 1, 4, 4], dtype=np.int32)
    r_arr = np.asfortranarray(np.eye(n))
    b_in = np.ones(n)
    sdiag = np.zeros(n)
    s = np.zeros((1, 1), order='F')
    ranks = np.zeros(2, dtype=np.int32)

    try:
        b_out, ranks_out, info = nf01br('X', 'U', 'N', n, ipar, r_arr, sdiag, s, b_in, ranks, 0.0)
        # Wrapper may raise ValueError or return info<0
        assert info < 0 or info == -1
    except (ValueError, RuntimeError):
        pass  # Expected

def test_nf01br_error_invalid_n():
    """
    Validate NF01BR error handling for n<0.
    """
    n = -1
    ipar = np.array([0, 1, 4, 4], dtype=np.int32)
    r_arr = np.asfortranarray(np.eye(1))
    b_in = np.ones(1)
    sdiag = np.zeros(1)
    s = np.zeros((1, 1), order='F')
    ranks = np.zeros(2, dtype=np.int32)

    try:
        b_out, ranks_out, info = nf01br('N', 'U', 'N', n, ipar, r_arr, sdiag, s, b_in, ranks, 0.0)
        assert info < 0
    except (ValueError, RuntimeError):
        pass

if __name__ == '__main__':
    unittest.main()
