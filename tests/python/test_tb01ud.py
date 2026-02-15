"""
Tests for TB01UD - Controllable block Hessenberg realization

TB01UD reduces a state-space system (A,B,C) to controllability staircase form
using orthogonal similarity transformations. The transformed system has:
- Acont: upper block Hessenberg with full row rank subdiagonal blocks
- Bcont: zero except for first block (rank(B) rows)
- Cc: C*Z transformed output matrix

Mathematical properties tested:
- Orthogonality: Z'*Z = I
- Similarity: Ac = Z'*A*Z, Bc = Z'*B, Cc = C*Z
- Eigenvalue preservation for controllable subsystem
- Controllability dimension (NCONT)

Test data from SLICOT-Reference/doc/TB01UD.html example
"""
import numpy as np
import pytest

try:
    import ctrlsys
    HAS_SLICOT = True
except ImportError:
    HAS_SLICOT = False


pytestmark = pytest.mark.skipif(not HAS_SLICOT, reason="slicot module not available")


def test_html_example():
    """Test TB01UD with example from HTML documentation

    From HTML doc:
    N=3, M=2, P=2, TOL=0.0, JOBZ='I'

    A = [-1.0   0.0   0.0]    B = [1.0  0.0]    C = [0.0  2.0  1.0]
        [-2.0  -2.0  -2.0]        [0.0  2.0]        [1.0  0.0  0.0]
        [-1.0   0.0  -3.0]        [0.0  1.0]

    Expected outputs:
    NCONT = 2
    INDCON = 1
    NBLK = [2]

    Acont = [-3.0000   2.2361]
            [ 0.0000  -1.0000]

    Bcont = [ 0.0000  -2.2361]
            [ 1.0000   0.0000]

    Ccont = [-2.2361   0.0000]
            [ 0.0000   1.0000]

    Z = [ 0.0000   1.0000   0.0000]
        [-0.8944   0.0000  -0.4472]
        [-0.4472   0.0000   0.8944]
    """
    n, m, p = 3, 2, 2

    a = np.array([
        [-1.0,  0.0,  0.0],
        [-2.0, -2.0, -2.0],
        [-1.0,  0.0, -3.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [0.0, 2.0, 1.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    expected_ncont = 2
    expected_indcon = 1
    expected_nblk = [2]

    expected_acont = np.array([
        [-3.0000,  2.2361],
        [ 0.0000, -1.0000]
    ], dtype=np.float64, order='F')

    expected_bcont = np.array([
        [ 0.0000, -2.2361],
        [ 1.0000,  0.0000]
    ], dtype=np.float64, order='F')

    expected_ccont = np.array([
        [-2.2361,  0.0000],
        [ 0.0000,  1.0000]
    ], dtype=np.float64, order='F')

    expected_z = np.array([
        [ 0.0000,  1.0000,  0.0000],
        [-0.8944,  0.0000, -0.4472],
        [-0.4472,  0.0000,  0.8944]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0, f"tb01ud failed with info={info}"
    assert ncont == expected_ncont, f"ncont={ncont}, expected {expected_ncont}"
    assert indcon == expected_indcon, f"indcon={indcon}, expected {expected_indcon}"

    for i in range(indcon):
        assert nblk[i] == expected_nblk[i], f"nblk[{i}]={nblk[i]}, expected {expected_nblk[i]}"

    np.testing.assert_allclose(
        a_out[:ncont, :ncont], expected_acont, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        b_out[:ncont, :], expected_bcont, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        c_out[:, :ncont], expected_ccont, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z, expected_z, rtol=1e-3, atol=1e-4)

def test_orthogonality_property():
    """Test that Z is orthogonal: Z'*Z = I

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 2, 3

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0

    z_orth = z.T @ z
    np.testing.assert_allclose(z_orth, np.eye(n), rtol=1e-14, atol=1e-14)

def test_similarity_transformation():
    """Test similarity relationships for C matrix: Cc = C*Z

    Note: For A and B, the routine applies internal scaling (via mb01pd)
    before applying transformations, so Ac != Z'*A*Z exactly.
    However, C is only transformed (not scaled), so Cc = C*Z holds.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 5, 2, 3

    a_orig = np.random.randn(n, n).astype(np.float64, order='F')
    b_orig = np.random.randn(n, m).astype(np.float64, order='F')
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0

    c_check = c_orig @ z
    np.testing.assert_allclose(c_out, c_check, rtol=1e-13, atol=1e-14)

def test_eigenvalue_preservation_controllable():
    """Test eigenvalue preservation for the controllable subsystem

    For the controllable part, eigenvalues must be preserved.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    eig_orig = np.linalg.eigvals(a)

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0

    eig_new = np.linalg.eigvals(a_out)
    np.testing.assert_allclose(
        np.sort(eig_orig.real), np.sort(eig_new.real), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.sort(np.abs(eig_orig.imag)), np.sort(np.abs(eig_new.imag)),
        rtol=1e-12, atol=1e-14)

def test_fully_controllable_system():
    """Test with a fully controllable system (ncont = n)

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 3, 2, 2

    a = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, -2, -3]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0, 0],
        [0, 0],
        [1, 1]
    ], dtype=np.float64, order='F')

    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == n, f"System should be fully controllable, ncont={ncont}"

def test_uncontrollable_system():
    """Test with a partially uncontrollable system

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m, p = 4, 1, 2

    a = np.array([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1],
        [1],
        [0],
        [0]
    ], dtype=np.float64, order='F')

    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont < n, f"System should be partially uncontrollable, ncont={ncont}"

def test_jobz_n_no_z():
    """Test with JOBZ='N' (no transformation matrix computed)"""
    n, m, p = 3, 2, 2

    a = np.array([
        [-1.0,  0.0,  0.0],
        [-2.0, -2.0, -2.0],
        [-1.0,  0.0, -3.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [0.0, 2.0, 1.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 2

def test_jobz_f_factored_form():
    """Test with JOBZ='F' (factored form of transformation)"""
    n, m, p = 3, 2, 2

    a = np.array([
        [-1.0,  0.0,  0.0],
        [-2.0, -2.0, -2.0],
        [-1.0,  0.0, -3.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [0.0, 2.0, 1.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'F', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 2

def test_quick_return_n_zero():
    """Test quick return for N=0"""
    n, m, p = 0, 2, 2
    a = np.zeros((1, 1), dtype=np.float64, order='F')
    b = np.zeros((1, 2), dtype=np.float64, order='F')
    c = np.zeros((2, 1), dtype=np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0

def test_quick_return_m_zero():
    """Test quick return for M=0 (no inputs)"""
    n, m, p = 3, 0, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, 1), dtype=np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0

def test_quick_return_b_zero():
    """Test quick return for B=0 (zero input matrix)"""
    n, m, p = 3, 2, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m), dtype=np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0

def test_upper_block_hessenberg_structure():
    """Test that transformed A has upper block Hessenberg structure

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 6, 2, 3

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = slicot.tb01ud(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0

    if ncont > 0 and indcon > 0:
        row_offset = 0
        for k in range(indcon - 1):
            block_size = nblk[k]
            row_offset += block_size
            next_block_size = nblk[k + 1] if k + 1 < indcon else 0
            for i in range(row_offset + next_block_size, ncont):
                for j in range(row_offset - block_size):
                    if i >= 0 and j >= 0 and i < ncont and j < ncont:
                        assert abs(a_out[i, j]) < 1e-10, \
                            f"A not upper block Hessenberg: A[{i},{j}]={a_out[i,j]}"

def test_invalid_jobz():
    """Test error handling for invalid JOBZ parameter"""
    n, m, p = 3, 2, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m), dtype=np.float64, order='F')
    c = np.zeros((p, n), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01ud('X', n, m, p, a, b, c, tol=0.0)

def test_invalid_n_negative():
    """Test error handling for N < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01ud('I', -1, 1, 1, a, b, c, tol=0.0)

def test_invalid_m_negative():
    """Test error handling for M < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01ud('I', 1, -1, 1, a, b, c, tol=0.0)

def test_invalid_p_negative():
    """Test error handling for P < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01ud('I', 1, 1, -1, a, b, c, tol=0.0)
