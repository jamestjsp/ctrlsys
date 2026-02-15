"""Tests for AB07MD: Dual of state-space representation."""

import numpy as np
import pytest
from ctrlsys import ab07md


def prepare_arrays(n, m, p, a_in, b_in, c_in, d_in):
    """
    Prepare arrays with proper dimensions for AB07MD.

    AB07MD requires:
    - B: (n, max(m,p)) - input data in first m columns
    - C: (max(m,p), n) - input data in first p rows
    - D: (max(m,p), max(m,p)) - input data in (p, m) submatrix
    """
    mplim = max(m, p)

    a = a_in.copy(order='F')

    b = np.zeros((n, mplim), order='F', dtype=float)
    if n > 0 and m > 0:
        b[:n, :m] = b_in[:n, :m]

    c = np.zeros((mplim, n), order='F', dtype=float)
    if n > 0 and p > 0:
        c[:p, :n] = c_in[:p, :n]

    d = np.zeros((mplim, mplim), order='F', dtype=float)
    if p > 0 and m > 0:
        d[:p, :m] = d_in[:p, :m]

    return a, b, c, d


"""Basic functionality tests from HTML doc example."""

def test_ab07md_html_example():
    """
    Validate AB07MD using HTML doc example.

    Input: 3-state, 1-input, 2-output system (A,B,C,D)
    Output: Dual 2-input, 1-output system (A',C',B',D')
    """
    n, m, p = 3, 1, 2

    a_in = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    b_in = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    c_in = np.array([
        [0.0, 1.0, -1.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    d_in = np.array([
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    a, b, c, d = prepare_arrays(n, m, p, a_in, b_in, c_in, d_in)

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a, b, c, d)

    assert info == 0

    a_dual_expected = np.array([
        [1.0, 4.0, 0.0],
        [2.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    b_dual_expected = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0]
    ], order='F', dtype=float)

    c_dual_expected = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    d_dual_expected = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ], order='F', dtype=float)

    np.testing.assert_allclose(a_dual, a_dual_expected, rtol=1e-14)
    np.testing.assert_allclose(b_dual[:, :p], b_dual_expected[:, :p], rtol=1e-14)
    np.testing.assert_allclose(c_dual[:m, :], c_dual_expected[:m, :], rtol=1e-14)
    np.testing.assert_allclose(d_dual[:m, :p], d_dual_expected[:m, :p], rtol=1e-14)


"""Mathematical property tests for dual transformation."""

def test_dual_involution():
    """
    Validate involution property: dual(dual(sys)) = sys.

    Applying dual transformation twice returns original system.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 2, 3
    mplim = max(m, p)

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b_orig = np.zeros((n, mplim), order='F', dtype=float)
    b_orig[:, :m] = np.random.randn(n, m)
    c_orig = np.zeros((mplim, n), order='F', dtype=float)
    c_orig[:p, :] = np.random.randn(p, n)
    d_orig = np.zeros((mplim, mplim), order='F', dtype=float)
    d_orig[:p, :m] = np.random.randn(p, m)

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    a1, b1, c1, d1, info1 = ab07md('D', n, m, p, a, b, c, d)
    assert info1 == 0

    a2, b2, c2, d2, info2 = ab07md('D', n, p, m, a1, b1, c1, d1)
    assert info2 == 0

    np.testing.assert_allclose(a2, a_orig, rtol=1e-14)
    np.testing.assert_allclose(b2[:, :m], b_orig[:, :m], rtol=1e-14)
    np.testing.assert_allclose(c2[:p, :], c_orig[:p, :], rtol=1e-14)
    np.testing.assert_allclose(d2[:p, :m], d_orig[:p, :m], rtol=1e-14)

def test_dual_transpose_relationships():
    """
    Validate dual relationships: A_dual = A^T, B_dual = C^T, C_dual = B^T, D_dual = D^T.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 3, 2, 4
    mplim = max(m, p)

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b_orig = np.zeros((n, mplim), order='F', dtype=float)
    b_orig[:, :m] = np.random.randn(n, m)
    c_orig = np.zeros((mplim, n), order='F', dtype=float)
    c_orig[:p, :] = np.random.randn(p, n)
    d_orig = np.zeros((mplim, mplim), order='F', dtype=float)
    d_orig[:p, :m] = np.random.randn(p, m)

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a, b, c, d)

    assert info == 0
    np.testing.assert_allclose(a_dual, a_orig.T, rtol=1e-14)
    np.testing.assert_allclose(b_dual[:, :p], c_orig[:p, :].T, rtol=1e-14)
    np.testing.assert_allclose(c_dual[:m, :], b_orig[:, :m].T, rtol=1e-14)
    np.testing.assert_allclose(d_dual[:m, :p], d_orig[:p, :m].T, rtol=1e-14)

def test_eigenvalue_preservation():
    """
    Validate eigenvalue preservation: eigenvalues of A and A' are identical.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 5, 2, 3
    mplim = max(m, p)

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b = np.zeros((n, mplim), order='F', dtype=float)
    b[:, :m] = np.random.randn(n, m)
    c = np.zeros((mplim, n), order='F', dtype=float)
    c[:p, :] = np.random.randn(p, n)
    d = np.zeros((mplim, mplim), order='F', dtype=float)
    d[:p, :m] = np.random.randn(p, m)

    eig_before = np.linalg.eigvals(a_orig)

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a_orig, b, c, d)

    assert info == 0
    eig_after = np.linalg.eigvals(a_dual)

    np.testing.assert_allclose(
        sorted(eig_before.real),
        sorted(eig_after.real),
        rtol=1e-14
    )
    np.testing.assert_allclose(
        sorted(eig_before.imag),
        sorted(eig_after.imag),
        rtol=1e-14
    )


"""Edge case tests."""

def test_jobd_z_no_d_processing():
    """
    Test JOBD='Z' mode where D is zero matrix (not processed).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 3, 2, 2
    mplim = max(m, p)

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b_orig = np.zeros((n, mplim), order='F', dtype=float)
    b_orig[:, :m] = np.random.randn(n, m)
    c_orig = np.zeros((mplim, n), order='F', dtype=float)
    c_orig[:p, :] = np.random.randn(p, n)
    d = np.zeros((mplim, mplim), order='F', dtype=float)

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    a_dual, b_dual, c_dual, d_dual, info = ab07md('Z', n, m, p, a, b, c, d)

    assert info == 0
    np.testing.assert_allclose(a_dual, a_orig.T, rtol=1e-14)
    np.testing.assert_allclose(b_dual[:, :p], c_orig[:p, :].T, rtol=1e-14)
    np.testing.assert_allclose(c_dual[:m, :], b_orig[:, :m].T, rtol=1e-14)

def test_square_system_m_equals_p():
    """
    Test square system where M = P.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 3, 2, 2
    mplim = max(m, p)

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(p, n).astype(float, order='F')
    d_orig = np.random.randn(p, m).astype(float, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a, b, c, d)

    assert info == 0
    np.testing.assert_allclose(a_dual, a_orig.T, rtol=1e-14)
    np.testing.assert_allclose(b_dual, c_orig.T, rtol=1e-14)
    np.testing.assert_allclose(c_dual, b_orig.T, rtol=1e-14)
    np.testing.assert_allclose(d_dual, d_orig.T, rtol=1e-14)

def test_siso_system():
    """
    Test SISO system (M=1, P=1).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m, p = 4, 1, 1

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(p, n).astype(float, order='F')
    d_orig = np.random.randn(p, m).astype(float, order='F')

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a, b, c, d)

    assert info == 0
    np.testing.assert_allclose(a_dual, a_orig.T, rtol=1e-14)
    np.testing.assert_allclose(b_dual, c_orig.T, rtol=1e-14)
    np.testing.assert_allclose(c_dual, b_orig.T, rtol=1e-14)
    np.testing.assert_allclose(d_dual, d_orig.T, rtol=1e-14)

def test_n_zero():
    """
    Test N=0 case (quick return).
    """
    n, m, p = 0, 2, 3
    mplim = max(m, p)

    a = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, mplim), order='F', dtype=float)
    c = np.zeros((mplim, 1), order='F', dtype=float)
    d = np.zeros((mplim, mplim), order='F', dtype=float)

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a, b, c, d)

    assert info == 0

def test_m_greater_than_p():
    """
    Test M > P case (more inputs than outputs).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, p = 3, 4, 2
    mplim = max(m, p)

    a_orig = np.random.randn(n, n).astype(float, order='F')
    b_orig = np.zeros((n, mplim), order='F', dtype=float)
    b_orig[:, :m] = np.random.randn(n, m)
    c_orig = np.zeros((mplim, n), order='F', dtype=float)
    c_orig[:p, :] = np.random.randn(p, n)
    d_orig = np.zeros((mplim, mplim), order='F', dtype=float)
    d_orig[:p, :m] = np.random.randn(p, m)

    a = a_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    d = d_orig.copy(order='F')

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a, b, c, d)

    assert info == 0
    np.testing.assert_allclose(a_dual, a_orig.T, rtol=1e-14)
    np.testing.assert_allclose(b_dual[:, :p], c_orig[:p, :].T, rtol=1e-14)
    np.testing.assert_allclose(c_dual[:m, :], b_orig[:, :m].T, rtol=1e-14)
    np.testing.assert_allclose(d_dual[:m, :p], d_orig[:p, :m].T, rtol=1e-14)


"""Error handling tests."""

def test_invalid_jobd():
    """Test invalid JOBD parameter."""
    n, m, p = 2, 1, 1
    a = np.zeros((n, n), order='F', dtype=float)
    b = np.zeros((n, 1), order='F', dtype=float)
    c = np.zeros((1, n), order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    with pytest.raises(ValueError):
        ab07md('X', n, m, p, a, b, c, d)

def test_negative_n():
    """Test negative N parameter."""
    n, m, p = -1, 1, 1
    a = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, 1), order='F', dtype=float)
    c = np.zeros((1, 1), order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a, b, c, d)

    assert info == -2

def test_negative_m():
    """Test negative M parameter."""
    n, m, p = 2, -1, 1
    a = np.zeros((n, n), order='F', dtype=float)
    b = np.zeros((n, 1), order='F', dtype=float)
    c = np.zeros((1, n), order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a, b, c, d)

    assert info == -3

def test_negative_p():
    """Test negative P parameter."""
    n, m, p = 2, 1, -1
    a = np.zeros((n, n), order='F', dtype=float)
    b = np.zeros((n, 1), order='F', dtype=float)
    c = np.zeros((1, n), order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    a_dual, b_dual, c_dual, d_dual, info = ab07md('D', n, m, p, a, b, c, d)

    assert info == -4
