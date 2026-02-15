"""
Tests for TB03AD - State-space to polynomial matrix representation

TB03AD converts a state-space representation (A,B,C,D) to a left or right
polynomial matrix fraction representation:
    T(s) = C*inv(s*I-A)*B + D = inv(P(s))*Q(s) (left)
                               = Q(s)*inv(P(s)) (right)

The routine first computes a minimal realization, then extracts polynomial
coefficients for the denominator P(s) and numerator Q(s) matrices.

Test data from SLICOT-Reference/doc/TB03AD.html example
"""
import numpy as np
import pytest

try:
    import ctrlsys
    HAS_SLICOT = True
except ImportError:
    HAS_SLICOT = False

pytestmark = pytest.mark.skipif(not HAS_SLICOT, reason="slicot module not available")


def test_html_example_right_fraction():
    """Test TB03AD with example from HTML documentation (right fraction)

    From HTML doc:
    N=3, M=1, P=2, TOL=0.0, LERI='R', EQUIL='N'

    A = [1.0  2.0  0.0]    B = [1.0]    C = [0.0  1.0 -1.0]    D = [0.0]
        [4.0 -1.0  0.0]        [0.0]        [0.0  0.0  1.0]        [1.0]
        [0.0  0.0  1.0]        [1.0]

    (Note: HTML shows row-by-row reading)

    Expected outputs:
    NR = 3 (minimal order)
    INDEX = [3]

    Amin = [ 1.0000  -1.4142   0.0000]
           [-2.8284  -1.0000   2.8284]
           [ 0.0000   1.4142   1.0000]

    PCOEFF(:,:,k) for k=1..4:
      [ 0.1768  -0.1768  -1.5910   1.5910]

    QCOEFF (P x M x kpcoef):
      Row 1: [ 0.0000  -0.1768   0.7071   0.8839]
      Row 2: [ 0.1768   0.0000  -1.5910   0.0000]
    """
    n, m, p = 3, 1, 2

    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [0.0, 1.0, -1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    d = np.array([
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    expected_nr = 3
    expected_index = np.array([3], dtype=np.int32)

    expected_amin = np.array([
        [1.0000, -1.4142, 0.0000],
        [-2.8284, -1.0000, 2.8284],
        [0.0000, 1.4142, 1.0000]
    ], dtype=np.float64, order='F')

    expected_pcoeff = np.array([0.1768, -0.1768, -1.5910, 1.5910],
                               dtype=np.float64)

    expected_qcoeff_row1 = np.array([0.0000, -0.1768, 0.7071, 0.8839],
                                    dtype=np.float64)
    expected_qcoeff_row2 = np.array([0.1768, 0.0000, -1.5910, 0.0000],
                                    dtype=np.float64)

    result = slicot.tb03ad('R', 'N', n, m, p, a, b, c, d, tol=0.0)

    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0, f"tb03ad failed with info={info}"
    assert nr == expected_nr, f"nr={nr}, expected {expected_nr}"

    np.testing.assert_array_equal(index[:m], expected_index)

    # Note: Amin matrix can vary due to similarity transforms, so we check
    # eigenvalues instead of exact values
    actual_eigs = np.sort(np.linalg.eigvals(a_out[:nr, :nr]).real)
    expected_eigs = np.sort(np.linalg.eigvals(expected_amin).real)
    np.testing.assert_allclose(actual_eigs, expected_eigs, rtol=1e-3, atol=1e-4)

    kpcoef = max(index[:m]) + 1
    np.testing.assert_allclose(
        pcoeff[0, 0, :kpcoef], expected_pcoeff, rtol=1e-3, atol=1e-4)

    np.testing.assert_allclose(
        qcoeff[0, 0, :kpcoef], expected_qcoeff_row1, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        qcoeff[1, 0, :kpcoef], expected_qcoeff_row2, rtol=1e-3, atol=1e-4)


def test_left_fraction_basic():
    """Test TB03AD with left fraction (LERI='L')

    For left fraction: T(s) = inv(P(s))*Q(s)
    - INDEX has dimension P
    - PCOEFF is P x P x kpcoef
    - QCOEFF is P x M x kpcoef

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 2, 1, 2

    a = np.array([
        [0.0, 1.0],
        [-2.0, -3.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    d = np.zeros((p, m), dtype=np.float64, order='F')

    result = slicot.tb03ad('L', 'N', n, m, p, a, b, c, d, tol=0.0)

    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0, f"tb03ad failed with info={info}"
    assert nr >= 0
    assert nr <= n


def test_quick_return_n_zero():
    """Test quick return when N=0"""
    n, m, p = 0, 1, 1

    a = np.zeros((1, 1), dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')
    d = np.zeros((1, 1), dtype=np.float64, order='F')

    result = slicot.tb03ad('R', 'N', n, m, p, a, b, c, d, tol=0.0)
    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0
    assert nr == 0


def test_quick_return_m_zero():
    """Test quick return when M=0"""
    n, m, p = 2, 0, 1

    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, 1), dtype=np.float64, order='F')
    c = np.zeros((1, n), dtype=np.float64, order='F')
    d = np.zeros((1, 1), dtype=np.float64, order='F')

    result = slicot.tb03ad('R', 'N', n, m, p, a, b, c, d, tol=0.0)
    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0
    assert nr == 0


def test_quick_return_p_zero():
    """Test quick return when P=0"""
    n, m, p = 2, 1, 0

    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, 1), dtype=np.float64, order='F')
    c = np.zeros((1, n), dtype=np.float64, order='F')
    d = np.zeros((1, 1), dtype=np.float64, order='F')

    result = slicot.tb03ad('L', 'N', n, m, p, a, b, c, d, tol=0.0)
    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0
    assert nr == 0


def test_with_balancing():
    """Test TB03AD with balancing enabled (EQUIL='S')

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 3, 1, 2

    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [0.0, 1.0, -1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    d = np.array([
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    result = slicot.tb03ad('R', 'S', n, m, p, a, b, c, d, tol=0.0)
    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0
    assert nr >= 0


def test_siso_system():
    """Test TB03AD with SISO system (M=1, P=1)

    Transfer function: G(s) = 1/(s+1)

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 1, 1, 1

    a = np.array([[-1.0]], dtype=np.float64, order='F')
    b = np.array([[1.0]], dtype=np.float64, order='F')
    c = np.array([[1.0]], dtype=np.float64, order='F')
    d = np.array([[0.0]], dtype=np.float64, order='F')

    result = slicot.tb03ad('R', 'N', n, m, p, a, b, c, d, tol=0.0)
    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0
    assert nr == 1


def test_index_ordering():
    """Test that INDEX values are in decreasing order

    For right fraction: INDEX(1) >= INDEX(2) >= ... >= INDEX(M)
    For left fraction: INDEX(1) >= INDEX(2) >= ... >= INDEX(P)

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 4, 2, 2

    a = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, -2.0, -3.0, -4.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    d = np.zeros((p, m), dtype=np.float64, order='F')

    result = slicot.tb03ad('R', 'N', n, m, p, a, b, c, d, tol=0.0)
    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0

    for i in range(m - 1):
        assert index[i] >= index[i + 1], \
            f"INDEX not decreasing: INDEX[{i}]={index[i]}, INDEX[{i+1}]={index[i+1]}"


def test_mimo_system():
    """Test TB03AD with MIMO system (M>1, P>1)

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(np.float64, order='F')
    a = a - 2.0 * np.eye(n, dtype=np.float64, order='F')

    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')
    d = np.random.randn(p, m).astype(np.float64, order='F')

    result = slicot.tb03ad('R', 'N', n, m, p, a, b, c, d, tol=0.0)
    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0
    assert nr >= 0
    assert nr <= n


def test_d_nonzero():
    """Test TB03AD with nonzero D matrix

    The feedthrough matrix D contributes to Q(s) = V(s)*B + P(s)*D

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 2, 1, 1

    a = np.array([
        [0.0, 1.0],
        [-2.0, -3.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    c = np.array([[1.0, 0.0]], dtype=np.float64, order='F')

    d = np.array([[2.0]], dtype=np.float64, order='F')

    result = slicot.tb03ad('R', 'N', n, m, p, a, b, c, d, tol=0.0)
    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0


def test_invalid_leri():
    """Test error handling for invalid LERI parameter"""
    n, m, p = 2, 1, 1
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m), dtype=np.float64, order='F')
    c = np.zeros((p, n), dtype=np.float64, order='F')
    d = np.zeros((p, m), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb03ad('X', 'N', n, m, p, a, b, c, d, tol=0.0)


def test_invalid_equil():
    """Test error handling for invalid EQUIL parameter"""
    n, m, p = 2, 1, 1
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m), dtype=np.float64, order='F')
    c = np.zeros((p, n), dtype=np.float64, order='F')
    d = np.zeros((p, m), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb03ad('R', 'X', n, m, p, a, b, c, d, tol=0.0)


def test_negative_n():
    """Test error handling for N < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')
    d = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb03ad('R', 'N', -1, 1, 1, a, b, c, d, tol=0.0)


def test_negative_m():
    """Test error handling for M < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')
    d = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb03ad('R', 'N', 1, -1, 1, a, b, c, d, tol=0.0)


def test_negative_p():
    """Test error handling for P < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')
    d = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb03ad('R', 'N', 1, 1, -1, a, b, c, d, tol=0.0)


def test_polynomial_degree_property():
    """Test that polynomial degrees match minimal realization order

    For a minimal realization, sum of INDEX values equals NR.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 3, 1, 2

    a = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [0.0, 1.0, -1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    d = np.array([
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    result = slicot.tb03ad('R', 'N', n, m, p, a, b, c, d, tol=0.0)
    a_out, b_out, c_out, nr, index, pcoeff, qcoeff, vcoeff, iwork, info = result

    assert info == 0

    index_sum = sum(index[:m])
    assert index_sum == nr, f"sum(INDEX)={index_sum} != NR={nr}"
