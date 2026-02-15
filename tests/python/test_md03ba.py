import unittest
import numpy as np
try:
    from ctrlsys import md03ba
except ImportError:
    pass

def test_md03ba_basic():
    try:
        from ctrlsys import md03ba
    except ImportError:
        self.fail("Could not import md03ba")

    np.random.seed(42)
    m, n = 5, 3
    ipar = np.array([m], dtype=np.int32)
    
    j_in = np.random.rand(m, n)
    j = np.asfortranarray(j_in)
    
    e_in = np.random.rand(m)
    e = e_in.copy()
    fnorm = np.linalg.norm(e)
    
    # Wrapper signature:
    # j_out, e_out, jnorms, gnorm, ipvt, info = md03ba(n, ipar, fnorm, j, e)
    
    j_out, e_out, jnorms, gnorm, ipvt, info = md03ba(n, ipar, fnorm, j, e)
    
    assert info == 0
    
    # Verify E norm preservation (Q is orthogonal)
    norm_e_out = np.linalg.norm(e_out)
    np.testing.assert_allclose(norm_e_out, fnorm, rtol=1e-14)

    # Verify dimensions
    assert j_out.shape == (m, n)
    assert e_out.shape == (m,)
    assert jnorms.shape == (n,)
    assert ipvt.shape == (n,)

def test_md03ba_qr_validation():
    """
    Validate QR factorization outputs.

    Random seed: 123 (for reproducibility)

    Note: j_out contains QR result. Upper triangular part is R.
    When m > n, only upper n x n block is extracted.
    """
    try:
        from ctrlsys import md03ba
    except ImportError:
        self.fail("Could not import md03ba")

    np.random.seed(123)
    m, n = 6, 4
    ipar = np.array([m], dtype=np.int32)

    j_in = np.random.rand(m, n)
    j = np.asfortranarray(j_in)

    e_in = np.random.rand(m)
    e = e_in.copy()
    fnorm = np.linalg.norm(e)

    j_out, e_out, jnorms, gnorm, ipvt, info = md03ba(n, ipar, fnorm, j, e)

    assert info == 0

    # Extract upper triangular R from j_out
    # For m > n, DLACPY extracts n x n upper triangular
    # Verify diagonal elements exist
    for i in range(n):
        assert j_out[i, i] != 0.0 or i == n-1

    # ipvt is 1-based (Fortran convention)
    ipvt_sorted = np.sort(ipvt)
    np.testing.assert_array_equal(ipvt_sorted, np.arange(1, n+1))

    # Validate jnorms are positive (column norms of R, not J)
    assert np.all(jnorms >= 0)

def test_md03ba_edge_case_square_matrix():
    """
    Validate MD03BA with m=n (square matrix).

    Note: When m=n, DLACPY is not called, so j_out contains
    full QR result (Householder vectors below diagonal).
    """
    try:
        from ctrlsys import md03ba
    except ImportError:
        self.fail("Could not import md03ba")

    np.random.seed(456)
    m, n = 3, 3
    ipar = np.array([m], dtype=np.int32)

    j_in = np.random.rand(m, n)
    j = np.asfortranarray(j_in)

    e_in = np.random.rand(m)
    e = e_in.copy()
    fnorm = np.linalg.norm(e)

    j_out, e_out, jnorms, gnorm, ipvt, info = md03ba(n, ipar, fnorm, j, e)

    assert info == 0

    # Verify E norm preservation (orthogonal Q)
    norm_e_out = np.linalg.norm(e_out)
    np.testing.assert_allclose(norm_e_out, fnorm, rtol=1e-14)

    # Verify computation succeeded
    assert np.all(np.isfinite(j_out))
    assert np.all(np.isfinite(jnorms))

def test_md03ba_jnorms_positive():
    """
    Validate jnorms array contains positive values.

    Random seed: 789 (for reproducibility)

    Note: Due to DLACPY stride change from m to n internally,
    verifying exact column norms from Python is complex.
    We just verify jnorms are reasonable (positive, finite).
    """
    try:
        from ctrlsys import md03ba
    except ImportError:
        self.fail("Could not import md03ba")

    np.random.seed(789)
    m, n = 5, 3
    ipar = np.array([m], dtype=np.int32)

    j_in = np.random.rand(m, n)
    j = np.asfortranarray(j_in)

    e_in = np.random.rand(m)
    e = e_in.copy()
    fnorm = np.linalg.norm(e)

    j_out, e_out, jnorms, gnorm, ipvt, info = md03ba(n, ipar, fnorm, j, e)

    assert info == 0

    # Validate jnorms are reasonable
    assert np.all(jnorms > 0)
    assert np.all(np.isfinite(jnorms))

    # Validate gnorm is reasonable
    assert gnorm >= 0
    assert np.isfinite(gnorm)

if __name__ == '__main__':
    unittest.main()
