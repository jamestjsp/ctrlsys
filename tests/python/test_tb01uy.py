"""
Tests for TB01UY - Controllable realization for M1+M2 input system

TB01UY reduces a state-space system (A,[B1,B2],C) to controllability staircase form
using orthogonal similarity transformations. The compound input matrix is split
into M1 and M2 columns, with alternating rank detection between B1 and B2.

Mathematical properties tested:
- Orthogonality: Z'*Z = I (when JOBZ='I')
- Similarity: Ac = Z'*A*Z, [Bc1,Bc2] = Z'*[B1,B2], Cc = C*Z
- Eigenvalue preservation
- Controllability dimension (NCONT)
- Block structure preservation (INDCON, NBLK)

Test data from SLICOT-Reference/doc/TB01UY.html example
"""
import numpy as np
import pytest

try:
    import ctrlsys
    HAS_CTRLSYS = True
except ImportError:
    HAS_CTRLSYS = False


pytestmark = pytest.mark.skipif(not HAS_CTRLSYS, reason="ctrlsys module not available")


def test_html_example():
    """Test TB01UY with example from HTML documentation

    From HTML doc:
    N=3, M1=1, M2=1, P=2, TOL=0.0, JOBZ='I'

    A read row-by-row: ((A(I,J), J=1,N), I=1,N)
    A = [-1.0   0.0   0.0]
        [-2.0  -2.0  -2.0]
        [-1.0   0.0  -3.0]

    B read column-by-column: ((B(I,J), I=1,N), J=1,M) where M=M1+M2=2
    Data: 1.0 0.0 0.0 / 0.0 2.0 1.0
    B = [1.0  0.0]
        [0.0  2.0]
        [0.0  1.0]

    C read row-by-row: ((C(I,J), J=1,N), I=1,P)
    Data: 0.0 2.0 1.0 / 1.0 0.0 0.0
    C = [0.0  2.0  1.0]
        [1.0  0.0  0.0]

    Expected outputs:
    NCONT = 2
    INDCON = 2
    NBLK = [1, 1]

    Acont = [-1.0000   0.0000]
            [ 2.2361  -3.0000]

    Bcont = [1.0000   0.0000]
            [0.0000  -2.2361]

    Ccont = [0.0000  -2.2361]
            [1.0000   0.0000]

    Z = [ 1.0000   0.0000   0.0000]
        [ 0.0000  -0.8944  -0.4472]
        [ 0.0000  -0.4472   0.8944]
    """
    n, m1, m2, p = 3, 1, 1, 2

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
    expected_indcon = 2
    expected_nblk = [1, 1]

    expected_acont = np.array([
        [-1.0000,  0.0000],
        [ 2.2361, -3.0000]
    ], dtype=np.float64, order='F')

    expected_bcont = np.array([
        [1.0000,  0.0000],
        [0.0000, -2.2361]
    ], dtype=np.float64, order='F')

    expected_ccont = np.array([
        [0.0000, -2.2361],
        [1.0000,  0.0000]
    ], dtype=np.float64, order='F')

    expected_z = np.array([
        [ 1.0000,  0.0000,  0.0000],
        [ 0.0000, -0.8944, -0.4472],
        [ 0.0000, -0.4472,  0.8944]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0, f"tb01uy failed with info={info}"
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
    n, m1, m2, p = 4, 2, 1, 3

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m1 + m2).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0

    z_orth = z.T @ z
    np.testing.assert_allclose(z_orth, np.eye(n), rtol=1e-14, atol=1e-14)


def test_similarity_c_transformation():
    """Test similarity relationship: Cc = C*Z

    The C matrix is only transformed (not scaled), so Cc = C*Z holds exactly.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m1, m2, p = 5, 2, 2, 3

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m1 + m2).astype(np.float64, order='F')
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')
    c = c_orig.copy(order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0

    c_check = c_orig @ z
    np.testing.assert_allclose(c_out, c_check, rtol=1e-13, atol=1e-14)


def test_eigenvalue_preservation():
    """Test eigenvalue preservation under similarity transformation

    Eigenvalues of A must equal eigenvalues of Ac = Z'*A*Z.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m1, m2, p = 4, 1, 1, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m1 + m2).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    eig_orig = np.linalg.eigvals(a)

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0

    eig_new = np.linalg.eigvals(a_out)
    np.testing.assert_allclose(
        np.sort(eig_orig.real), np.sort(eig_new.real), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.sort(np.abs(eig_orig.imag)), np.sort(np.abs(eig_new.imag)),
        rtol=1e-12, atol=1e-14)


def test_alternating_m1_m2_rank():
    """Test alternating M1/M2 rank detection

    Create a system where B1 and B2 contribute alternately to controllability.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m1, m2, p = 4, 2, 2, 2

    a = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    c = np.eye(p, n, dtype=np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont <= n


def test_m1_only():
    """Test with M2=0 (only B1 inputs)

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m1, m2, p = 3, 2, 0, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m1).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0


def test_m2_only():
    """Test with M1=0 (only B2 inputs)

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m1, m2, p = 3, 0, 2, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m2).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0


def test_jobz_n_no_z():
    """Test with JOBZ='N' (no transformation matrix computed)"""
    n, m1, m2, p = 3, 1, 1, 2

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

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'N', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 2


def test_jobz_f_factored_form():
    """Test with JOBZ='F' (factored form of transformation)"""
    n, m1, m2, p = 3, 1, 1, 2

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

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'F', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 2


def test_quick_return_n_zero():
    """Test quick return for N=0"""
    n, m1, m2, p = 0, 1, 1, 2
    a = np.zeros((1, 1), dtype=np.float64, order='F')
    b = np.zeros((1, 2), dtype=np.float64, order='F')
    c = np.zeros((2, 1), dtype=np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0


def test_quick_return_m_zero():
    """Test quick return for M1=M2=0 (no inputs)"""
    n, m1, m2, p = 3, 0, 0, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, 1), dtype=np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0


def test_quick_return_b_zero():
    """Test quick return for B=0 (zero input matrix)"""
    n, m1, m2, p = 3, 1, 1, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m1 + m2), dtype=np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 0
    assert indcon == 0


def test_nblk_alternating_structure():
    """Test that NBLK has alternating M1/M2 structure

    INDCON is always even, with INDCON/2 odd components (M1)
    and INDCON/2 even components (M2).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m1, m2, p = 5, 2, 2, 3

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m1 + m2).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, indcon, nblk, z, tau, info = ctrlsys.tb01uy(
        'I', n, m1, m2, p, a, b, c, tol=0.0)

    assert info == 0
    assert indcon % 2 == 0, f"INDCON should be even, got {indcon}"


def test_invalid_jobz():
    """Test error handling for invalid JOBZ parameter"""
    n, m1, m2, p = 3, 1, 1, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m1 + m2), dtype=np.float64, order='F')
    c = np.zeros((p, n), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        ctrlsys.tb01uy('X', n, m1, m2, p, a, b, c, tol=0.0)


def test_invalid_n_negative():
    """Test error handling for N < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 2), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        ctrlsys.tb01uy('I', -1, 1, 1, 1, a, b, c, tol=0.0)


def test_invalid_m1_negative():
    """Test error handling for M1 < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        ctrlsys.tb01uy('I', 1, -1, 1, 1, a, b, c, tol=0.0)


def test_invalid_m2_negative():
    """Test error handling for M2 < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        ctrlsys.tb01uy('I', 1, 1, -1, 1, a, b, c, tol=0.0)


def test_invalid_p_negative():
    """Test error handling for P < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 2), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        ctrlsys.tb01uy('I', 1, 1, 1, -1, a, b, c, tol=0.0)
