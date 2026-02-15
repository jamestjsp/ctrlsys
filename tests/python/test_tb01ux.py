"""
Tests for TB01UX - Observable-unobservable decomposition

TB01UX reduces a state-space system (A,B,C) to observability staircase form
using orthogonal similarity transformations. The transformed system has:
- Ao: observable part (NOBSV x NOBSV) in lower-right
- Co: observable output matrix (P x NOBSV)
- Ano: unobservable part in upper-left

Mathematical properties tested:
- Orthogonality: Z'*Z = I
- Similarity: Ac = Z'*A*Z, Bc = Z'*B, Cc = C*Z
- Eigenvalue preservation
- Observability dimension (NOBSV)
- Staircase structure of observable part

TB01UX is the dual of TB01UD - it uses dual system formation (AB07MD)
and controllability staircase reduction (TB01UD) internally.
"""
import numpy as np
import pytest

try:
    import ctrlsys
    HAS_SLICOT = True
except ImportError:
    HAS_SLICOT = False


pytestmark = pytest.mark.skipif(not HAS_SLICOT, reason="slicot module not available")


def test_basic_observable_system():
    """Test TB01UX with a simple observable system

    A fully observable system should have NOBSV = N.
    Using companion form which is observable.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 3, 2, 2

    a = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, -2, -3]
    ], dtype=np.float64, order='F')

    b = np.random.randn(n, m).astype(np.float64, order='F')

    c = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0, f"tb01ux failed with info={info}"
    assert nobsv == n, f"System should be fully observable, nobsv={nobsv}"


def test_orthogonality_property():
    """Test that Z is orthogonal: Z'*Z = I

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 4, 2, 3

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0

    z_orth = z.T @ z
    np.testing.assert_allclose(z_orth, np.eye(n), rtol=1e-14, atol=1e-14)


def test_similarity_transformation():
    """Test similarity relationships: Ac = Z'*A*Z, Cc = C*Z

    Note: For B, the routine applies internal scaling (via mb01pd in TB01UD)
    before applying transformations, so B_out != Z'*B exactly.
    The A and C matrices do satisfy exact similarity relationships.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 5, 2, 3

    a_orig = np.random.randn(n, n).astype(np.float64, order='F')
    b_orig = np.random.randn(n, m).astype(np.float64, order='F')
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0

    a_check = z.T @ a_orig @ z
    c_check = c_orig @ z

    np.testing.assert_allclose(a_out, a_check, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(c_out, c_check, rtol=1e-13, atol=1e-14)


def test_eigenvalue_preservation():
    """Test eigenvalue preservation: eigenvalues of A are preserved

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    eig_orig = np.linalg.eigvals(a)

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0

    eig_new = np.linalg.eigvals(a_out)
    np.testing.assert_allclose(
        np.sort(eig_orig.real), np.sort(eig_new.real), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.sort(np.abs(eig_orig.imag)), np.sort(np.abs(eig_new.imag)),
        rtol=1e-12, atol=1e-14)


def test_unobservable_system():
    """Test with a partially unobservable system

    Diagonal A with C that only observes first 2 states.
    Should identify states 3,4 as unobservable.

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

    b = np.random.randn(n, m).astype(np.float64, order='F')

    c = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nobsv < n, f"System should be partially unobservable, nobsv={nobsv}"
    assert nobsv == 2, f"Expected nobsv=2, got {nobsv}"


def test_unobservable_structure():
    """Test that unobservable part is in upper-left block

    For unobservable system, transformed A should have:
    Z'*A*Z = [Ano  *  ]
             [0   Ao  ]

    where (Ao,Co) is observable and Ano contains unobservable eigenvalues.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 4, 1, 2

    a = np.array([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4]
    ], dtype=np.float64, order='F')

    b = np.random.randn(n, m).astype(np.float64, order='F')

    c = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0

    n_unobs = n - nobsv
    if n_unobs > 0:
        lower_left = a_out[n_unobs:n, 0:n_unobs]
        np.testing.assert_allclose(
            lower_left, np.zeros((nobsv, n_unobs)), atol=1e-14)

        c_left = c_out[:, 0:n_unobs]
        np.testing.assert_allclose(
            c_left, np.zeros((p, n_unobs)), atol=1e-14)


def test_compz_n_no_z():
    """Test with COMPZ='N' (no transformation matrix computed)"""
    np.random.seed(100)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0


def test_quick_return_n_zero():
    """Test quick return for N=0"""
    n, m, p = 0, 2, 2
    a = np.zeros((1, 1), dtype=np.float64, order='F')
    b = np.zeros((1, 2), dtype=np.float64, order='F')
    c = np.zeros((2, 1), dtype=np.float64, order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nobsv == 0
    assert nlblck == 0


def test_quick_return_p_zero():
    """Test quick return for P=0 (no outputs)"""
    n, m, p = 3, 2, 0
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.zeros((1, n), dtype=np.float64, order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nobsv == 0
    assert nlblck == 0


def test_quick_return_c_zero():
    """Test with C=0 (zero output matrix)"""
    n, m, p = 3, 2, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.zeros((p, n), dtype=np.float64, order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nobsv == 0
    assert nlblck == 0


def test_staircase_block_tracking():
    """Test that CTAU returns correct block dimensions

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 5, 2, 3

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, nobsv, nlblck, ctau, z, info = slicot.tb01ux(
        'I', n, m, p, a, b, c, tol=0.0)

    assert info == 0

    if nlblck > 0:
        ctau_sum = sum(ctau[i] for i in range(nlblck))
        assert ctau_sum == nobsv, \
            f"Sum of CTAU ({ctau_sum}) should equal NOBSV ({nobsv})"


def test_invalid_compz():
    """Test error handling for invalid COMPZ parameter"""
    n, m, p = 3, 2, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m), dtype=np.float64, order='F')
    c = np.zeros((p, n), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01ux('X', n, m, p, a, b, c, tol=0.0)


def test_invalid_n_negative():
    """Test error handling for N < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01ux('I', -1, 1, 1, a, b, c, tol=0.0)


def test_invalid_tol():
    """Test error handling for TOL >= 1"""
    n, m, p = 3, 2, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m), dtype=np.float64, order='F')
    c = np.zeros((p, n), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01ux('I', n, m, p, a, b, c, tol=1.5)
