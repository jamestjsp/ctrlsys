"""
Tests for TB01ZD - Single-input controllable realization

TB01ZD finds a controllable realization for a linear time-invariant
single-input system dX/dt = A*X + B*U, where A is N-by-N and B is an N
element vector. Reduces system to orthogonal canonical form.

Mathematical properties tested:
- Orthogonality: Z'*Z = I (when JOBZ='I')
- Upper Hessenberg form: transformed A is upper Hessenberg
- Controllability detection via sub-diagonal elements
- B(1) contains norm, rest are zero

No HTML doc example available - using synthetic test data.
"""
import numpy as np
import pytest

try:
    import ctrlsys
    HAS_CTRLSYS = True
except ImportError:
    HAS_CTRLSYS = False


pytestmark = pytest.mark.skipif(not HAS_CTRLSYS, reason="ctrlsys module not available")


def test_basic_controllable_system():
    """Test TB01ZD with a fully controllable single-input system

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, p = 3, 2

    a = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -2.0, -3.0]
    ], dtype=np.float64, order='F')

    b = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    c = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'I', n, p, a, b, c, tol=0.0)

    assert info == 0, f"tb01zd failed with info={info}"
    assert ncont == n, f"System should be fully controllable, got ncont={ncont}"


def test_orthogonality_property():
    """Test that Z is orthogonal: Z'*Z = I

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, p = 4, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64)
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'I', n, p, a, b, c, tol=0.0)

    assert info == 0

    z_orth = z.T @ z
    np.testing.assert_allclose(z_orth, np.eye(n), rtol=1e-14, atol=1e-14)


def test_upper_hessenberg_structure():
    """Test that transformed A is upper Hessenberg in controllable part

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, p = 5, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64)
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'I', n, p, a, b, c, tol=0.0)

    assert info == 0

    if ncont > 2:
        for i in range(2, ncont):
            for j in range(i - 1):
                assert abs(a_out[i, j]) < 1e-10, \
                    f"A not upper Hessenberg: A[{i},{j}]={a_out[i,j]}"


def test_b_canonical_form():
    """Test that B is in canonical form: only B(0) nonzero

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, p = 4, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64)
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'I', n, p, a, b, c, tol=0.0)

    assert info == 0

    if ncont > 1:
        np.testing.assert_allclose(b_out[1:ncont], 0.0, atol=1e-14)


def test_similarity_c_transformation():
    """Test C transformation: Cc = C*Z

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, p = 4, 3

    a_orig = np.random.randn(n, n).astype(np.float64, order='F')
    b_orig = np.random.randn(n).astype(np.float64)
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy()
    c = c_orig.copy(order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'I', n, p, a, b, c, tol=0.0)

    assert info == 0

    c_check = c_orig @ z
    np.testing.assert_allclose(c_out, c_check, rtol=1e-13, atol=1e-14)


def test_zero_b_system():
    """Test with B=0 (completely uncontrollable)

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, p = 3, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.zeros(n, dtype=np.float64)
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'I', n, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 0, f"System with B=0 should have ncont=0, got {ncont}"


def test_jobz_n_no_z():
    """Test with JOBZ='N' (no transformation matrix)

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, p = 3, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64)
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'N', n, p, a, b, c, tol=0.0)

    assert info == 0


def test_jobz_f_factored():
    """Test with JOBZ='F' (factored form)

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, p = 3, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64)
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'F', n, p, a, b, c, tol=0.0)

    assert info == 0


def test_quick_return_n_zero():
    """Test quick return for N=0"""
    n, p = 0, 2
    a = np.zeros((1, 1), dtype=np.float64, order='F')
    b = np.zeros(1, dtype=np.float64)
    c = np.zeros((p, 1), dtype=np.float64, order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'I', n, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 0


def test_eigenvalue_preservation():
    """Test that eigenvalues are preserved under similarity transformation

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, p = 4, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64)
    c = np.random.randn(p, n).astype(np.float64, order='F')

    eig_orig = np.linalg.eigvals(a.copy())

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'I', n, p, a, b, c, tol=0.0)

    assert info == 0

    eig_new = np.linalg.eigvals(a_out)
    np.testing.assert_allclose(
        np.sort(eig_orig.real), np.sort(eig_new.real), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.sort(np.abs(eig_orig.imag)), np.sort(np.abs(eig_new.imag)),
        rtol=1e-12, atol=1e-14)


def test_partially_controllable():
    """Test partially controllable system

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, p = 4, 2

    a = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float64, order='F')
    b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, b_out, c_out, ncont, z, tau, info = ctrlsys.tb01zd(
        'I', n, p, a, b, c, tol=0.0)

    assert info == 0
    assert ncont == 1, f"Expected ncont=1, got {ncont}"


def test_invalid_jobz():
    """Test error handling for invalid JOBZ"""
    n, p = 3, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.ones(n, dtype=np.float64)
    c = np.zeros((p, n), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        ctrlsys.tb01zd('X', n, p, a, b, c, tol=0.0)


def test_invalid_n_negative():
    """Test error handling for N < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.ones(1, dtype=np.float64)
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        ctrlsys.tb01zd('I', -1, 1, a, b, c, tol=0.0)


def test_invalid_p_negative():
    """Test error handling for P < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.ones(1, dtype=np.float64)
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        ctrlsys.tb01zd('I', 1, -1, a, b, c, tol=0.0)
