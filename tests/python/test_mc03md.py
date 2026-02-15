"""
Tests for MC03MD - Real polynomial matrix operation P(x) = P1(x) * P2(x) + alpha * P3(x).

Computes the coefficients of a polynomial matrix product plus scaled term.
Each polynomial matrix coefficient is stored as a 3D array with shape
(rows, cols, degree+1), representing rows x cols matrices of polynomials.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mc03md_html_example():
    """
    Test using SLICOT HTML doc example.

    Input:
      RP1=3, CP1=2, CP2=2
      P1(x): 3x2 matrix, degree 2
      P2(x): 2x2 matrix, degree 1
      P3(x): 3x2 matrix, degree 1
      alpha = 1.0

    Expected output:
      P(x) = P1(x)*P2(x) + alpha*P3(x): 3x2 matrix, degree 3

    The HTML data is read column-by-column for each coefficient matrix:
      P1 data (degree 2, 3 coefficient matrices):
        k=1: [[1,2], [0,-1], [3,2]]  (cols read first)
        k=2: [[-2,3], [4,7], [9,-2]]
        k=3: [[6,1], [2,2], [-3,4]]

      P2 data (degree 1, 2 coefficient matrices):
        k=1: [[6,1], [-9,7]]
        k=2: [[1,8], [-6,7]]

      P3 data (degree 1, 2 coefficient matrices):
        k=1: [[1,0], [1,1], [0,1]]
        k=2: [[-1,-1], [1,-1], [1,1]]
    """
    from ctrlsys import mc03md

    rp1, cp1, cp2 = 3, 2, 2
    dp1, dp2, dp3 = 2, 1, 1

    # P1: 3x2 matrix coefficients, degree 2 (3 coefficient matrices)
    # HTML data: columns read first (I varies fastest)
    # k=1: 1 0 3, 2 -1 2 => col1=[1,0,3], col2=[2,-1,2]
    # k=2: -2 4 9, 3 7 -2 => col1=[-2,4,9], col2=[3,7,-2]
    # k=3: 6 2 -3, 1 2 4 => col1=[6,2,-3], col2=[1,2,4]
    p1 = np.zeros((rp1, cp1, dp1 + 1), order='F', dtype=float)
    p1[:, :, 0] = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 2.0]], order='F')
    p1[:, :, 1] = np.array([[-2.0, 3.0], [4.0, 7.0], [9.0, -2.0]], order='F')
    p1[:, :, 2] = np.array([[6.0, 1.0], [2.0, 2.0], [-3.0, 4.0]], order='F')

    # P2: 2x2 matrix coefficients, degree 1 (2 coefficient matrices)
    # k=1: 6 1, 1 7 => col1=[6,1], col2=[1,7] => [[6,1],[1,7]] NO!
    # Actually READ ((P2(I,J,K), I=1,CP1), J=1,CP2): I=rows of P2 (CP1=2), J=cols (CP2=2)
    # Data: 6 1 / 1 7
    # I varies fastest: 6, 1 for J=1 (col 1), then 1, 7 for J=2 (col 2)
    # So P2[:,:,0] = [[6,1],[1,7]]
    # k=2: -9 -6 / 7 8 => col1=[-9,-6], col2=[7,8] => [[-9,7],[-6,8]]
    p2 = np.zeros((cp1, cp2, dp2 + 1), order='F', dtype=float)
    p2[:, :, 0] = np.array([[6.0, 1.0], [1.0, 7.0]], order='F')
    p2[:, :, 1] = np.array([[-9.0, 7.0], [-6.0, 8.0]], order='F')

    # P3: 3x2 matrix coefficients, degree 1 (2 coefficient matrices)
    # k=1: 1 1 0, 0 1 1 => col1=[1,1,0], col2=[0,1,1]
    # k=2: -1 1 1, -1 -1 1 => col1=[-1,1,1], col2=[-1,-1,1]
    p3 = np.zeros((rp1, cp2, max(dp1 + dp2, dp3) + 1), order='F', dtype=float)
    p3[:, :, 0] = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], order='F')
    p3[:, :, 1] = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]], order='F')

    alpha = 1.0

    p_out, dp_out, info = mc03md(p1, p2, p3.copy(), alpha)

    assert info == 0
    assert dp_out == 3

    # Expected results from HTML doc
    # element (1,1): 9.00 -31.00 37.00 -60.00
    # element (1,2): 15.00 41.00 23.00 50.00
    # element (2,1): 0.00 38.00 -64.00 -30.00
    # element (2,2): -6.00 44.00 100.00 30.00
    # element (3,1): 20.00 14.00 -83.00 3.00
    # element (3,2): 18.00 33.00 72.00 11.00

    # Coefficients are: P(i,j,k) = coefficient of x^(k-1) in element (i,j)
    assert_allclose(p_out[0, 0, :4], [9.0, -31.0, 37.0, -60.0], rtol=1e-10)
    assert_allclose(p_out[0, 1, :4], [15.0, 41.0, 23.0, 50.0], rtol=1e-10)
    assert_allclose(p_out[1, 0, :4], [0.0, 38.0, -64.0, -30.0], rtol=1e-10)
    assert_allclose(p_out[1, 1, :4], [-6.0, 44.0, 100.0, 30.0], rtol=1e-10)
    assert_allclose(p_out[2, 0, :4], [20.0, 14.0, -83.0, 3.0], rtol=1e-10)
    assert_allclose(p_out[2, 1, :4], [18.0, 33.0, 72.0, 11.0], rtol=1e-10)


def test_mc03md_zero_alpha():
    """
    Test with alpha = 0 (P3 contribution nullified).

    P(x) = P1(x) * P2(x) + 0 * P3(x) = P1(x) * P2(x)

    Use simple 2x2 identity-like matrices.
    """
    from ctrlsys import mc03md

    rp1, cp1, cp2 = 2, 2, 2
    dp1, dp2, dp3 = 0, 0, 0

    # P1 = [[1,0],[0,1]] (identity)
    p1 = np.zeros((rp1, cp1, dp1 + 1), order='F', dtype=float)
    p1[:, :, 0] = np.eye(2)

    # P2 = [[2,1],[3,4]]
    p2 = np.zeros((cp1, cp2, dp2 + 1), order='F', dtype=float)
    p2[:, :, 0] = np.array([[2.0, 1.0], [3.0, 4.0]], order='F')

    # P3 arbitrary (will be scaled by 0)
    p3 = np.zeros((rp1, cp2, max(dp1 + dp2, dp3) + 1), order='F', dtype=float)
    p3[:, :, 0] = np.array([[99.0, 88.0], [77.0, 66.0]], order='F')

    alpha = 0.0

    p_out, dp_out, info = mc03md(p1, p2, p3.copy(), alpha)

    assert info == 0
    # P1*P2 = I*P2 = P2
    assert_allclose(p_out[:, :, 0], p2[:, :, 0], rtol=1e-14)


def test_mc03md_zero_p1():
    """
    Test with dp1 = -1 (P1 is zero polynomial).

    P(x) = 0 * P2(x) + alpha * P3(x) = alpha * P3(x)
    """
    from ctrlsys import mc03md

    rp1, cp1, cp2 = 2, 2, 2
    dp1, dp2, dp3 = -1, 1, 1

    # P1 dummy (not used when dp1=-1), but dimensions must be correct
    p1 = np.zeros((rp1, cp1, 1), order='F', dtype=float)

    # P2: 2x2, degree 1
    p2 = np.zeros((cp1, cp2, dp2 + 1), order='F', dtype=float)
    p2[:, :, 0] = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')
    p2[:, :, 1] = np.array([[5.0, 6.0], [7.0, 8.0]], order='F')

    # P3: 2x2, degree 1
    p3 = np.zeros((rp1, cp2, dp3 + 1), order='F', dtype=float)
    p3[:, :, 0] = np.array([[10.0, 20.0], [30.0, 40.0]], order='F')
    p3[:, :, 1] = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')

    alpha = 2.0

    p_out, dp_out, info = mc03md(p1, p2, p3.copy(), alpha, dp1=dp1)

    assert info == 0
    assert dp_out == 1
    # Result should be alpha * P3
    assert_allclose(p_out[:, :, 0], alpha * p3[:, :, 0], rtol=1e-14)
    assert_allclose(p_out[:, :, 1], alpha * p3[:, :, 1], rtol=1e-14)


def test_mc03md_zero_p2():
    """
    Test with dp2 = -1 (P2 is zero polynomial).

    P(x) = P1(x) * 0 + alpha * P3(x) = alpha * P3(x)
    """
    from ctrlsys import mc03md

    rp1, cp1, cp2 = 2, 2, 2
    dp1, dp2, dp3 = 1, -1, 1

    # P1: 2x2, degree 1
    p1 = np.zeros((rp1, cp1, dp1 + 1), order='F', dtype=float)
    p1[:, :, 0] = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')
    p1[:, :, 1] = np.array([[5.0, 6.0], [7.0, 8.0]], order='F')

    # P2 dummy (not used when dp2=-1) - must have correct row/col dimensions
    p2 = np.zeros((cp1, cp2, 1), order='F', dtype=float)

    # P3: 2x2, degree 1
    p3 = np.zeros((rp1, cp2, dp3 + 1), order='F', dtype=float)
    p3[:, :, 0] = np.array([[10.0, 20.0], [30.0, 40.0]], order='F')
    p3[:, :, 1] = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')

    alpha = 3.0

    p_out, dp_out, info = mc03md(p1, p2, p3.copy(), alpha, dp2=dp2)

    assert info == 0
    assert dp_out == 1
    # Result should be alpha * P3
    assert_allclose(p_out[:, :, 0], alpha * p3[:, :, 0], rtol=1e-14)
    assert_allclose(p_out[:, :, 1], alpha * p3[:, :, 1], rtol=1e-14)


def test_mc03md_zero_p3():
    """
    Test with dp3 = -1 (P3 is zero polynomial).

    P(x) = P1(x) * P2(x) + alpha * 0 = P1(x) * P2(x)
    """
    from ctrlsys import mc03md

    rp1, cp1, cp2 = 2, 2, 2
    dp1, dp2, dp3 = 0, 0, -1

    # P1 = [[2,0],[0,2]]
    p1 = np.zeros((rp1, cp1, dp1 + 1), order='F', dtype=float)
    p1[:, :, 0] = 2.0 * np.eye(2)

    # P2 = [[1,2],[3,4]]
    p2 = np.zeros((cp1, cp2, dp2 + 1), order='F', dtype=float)
    p2[:, :, 0] = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')

    # P3 empty (dp3=-1)
    p3 = np.zeros((rp1, cp2, 1), order='F', dtype=float)

    alpha = 5.0

    p_out, dp_out, info = mc03md(p1, p2, p3.copy(), alpha, dp3=dp3)

    assert info == 0
    assert dp_out == 0
    # P1*P2 = 2*I * P2 = 2*P2
    expected = 2.0 * p2[:, :, 0]
    assert_allclose(p_out[:, :, 0], expected, rtol=1e-14)


def test_mc03md_polynomial_multiplication():
    """
    Validate polynomial matrix multiplication property.

    For scalar polynomials p1(x) and p2(x):
    (p1*p2)(x) coefficient of x^k = sum_{i+j=k} p1_i * p2_j

    Test with 1x1 polynomial matrices (scalar polynomials).

    p1(x) = 1 + 2x (degree 1)
    p2(x) = 3 + x  (degree 1)
    p1*p2 = 3 + 7x + 2x^2

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc03md

    rp1, cp1, cp2 = 1, 1, 1
    dp1, dp2, dp3 = 1, 1, -1

    # p1(x) = 1 + 2x
    p1 = np.zeros((1, 1, 2), order='F', dtype=float)
    p1[0, 0, 0] = 1.0
    p1[0, 0, 1] = 2.0

    # p2(x) = 3 + x
    p2 = np.zeros((1, 1, 2), order='F', dtype=float)
    p2[0, 0, 0] = 3.0
    p2[0, 0, 1] = 1.0

    # P3 = 0 (dp3=-1)
    p3 = np.zeros((1, 1, 3), order='F', dtype=float)

    alpha = 1.0

    p_out, dp_out, info = mc03md(p1, p2, p3.copy(), alpha, dp3=dp3)

    assert info == 0
    assert dp_out == 2
    # p1*p2 = 3 + 7x + 2x^2
    assert_allclose(p_out[0, 0, 0], 3.0, rtol=1e-14)
    assert_allclose(p_out[0, 0, 1], 7.0, rtol=1e-14)
    assert_allclose(p_out[0, 0, 2], 2.0, rtol=1e-14)


def test_mc03md_degree_reduction():
    """
    Test that routine correctly computes actual degree when
    leading coefficients become zero.

    P1(x) = [[1,0],[0,1]] (constant identity, degree 0)
    P2(x) = [[1]] + [[1]]*x (degree 1)
    P3(x) = [[0]] + [[-1]]*x (degree 1)
    alpha = 1.0

    P = P1*P2 + P3 should have degree 0 if leading term cancels.
    """
    from ctrlsys import mc03md

    rp1, cp1, cp2 = 1, 1, 1
    dp1, dp2, dp3 = 0, 1, 1

    # P1 = 1 (constant)
    p1 = np.zeros((1, 1, 1), order='F', dtype=float)
    p1[0, 0, 0] = 1.0

    # P2(x) = 1 + x
    p2 = np.zeros((1, 1, 2), order='F', dtype=float)
    p2[0, 0, 0] = 1.0
    p2[0, 0, 1] = 1.0

    # P3(x) = 0 - x
    p3 = np.zeros((1, 1, 2), order='F', dtype=float)
    p3[0, 0, 0] = 0.0
    p3[0, 0, 1] = -1.0

    alpha = 1.0

    p_out, dp_out, info = mc03md(p1, p2, p3.copy(), alpha)

    assert info == 0
    # P = 1*P2 + P3 = (1+x) + (0-x) = 1 (degree 0)
    assert dp_out == 0
    assert_allclose(p_out[0, 0, 0], 1.0, rtol=1e-14)


def test_mc03md_random_verification():
    """
    Verify polynomial matrix multiplication with random data.

    P(x) = P1(x)*P2(x) + alpha*P3(x)

    Evaluate at multiple x values to verify correctness.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc03md

    np.random.seed(42)

    rp1, cp1, cp2 = 2, 3, 2
    dp1, dp2, dp3 = 2, 1, 2

    p1 = np.random.randn(rp1, cp1, dp1 + 1).astype(float)
    p1 = np.asfortranarray(p1)

    p2 = np.random.randn(cp1, cp2, dp2 + 1).astype(float)
    p2 = np.asfortranarray(p2)

    max_deg = max(dp1 + dp2, dp3)
    p3 = np.random.randn(rp1, cp2, max_deg + 1).astype(float)
    p3 = np.asfortranarray(p3)

    alpha = 2.5

    p_out, dp_out, info = mc03md(p1, p2, p3.copy(), alpha, dp3=dp3)

    assert info == 0

    # Verify at test points
    test_x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])

    for x in test_x:
        # Evaluate P1(x): sum over k of P1[:,:,k] * x^k
        p1_at_x = sum(p1[:, :, k] * (x ** k) for k in range(dp1 + 1))
        # Evaluate P2(x)
        p2_at_x = sum(p2[:, :, k] * (x ** k) for k in range(dp2 + 1))
        # Evaluate P3(x) - original input scaled by alpha
        p3_at_x = sum(p3[:, :, k] * (x ** k) for k in range(dp3 + 1))

        # Expected: P1(x) @ P2(x) + alpha * P3(x)
        expected = p1_at_x @ p2_at_x + alpha * p3_at_x

        # Evaluate output polynomial P(x)
        p_at_x = sum(p_out[:, :, k] * (x ** k) for k in range(dp_out + 1))

        assert_allclose(p_at_x, expected, rtol=1e-10, atol=1e-12)
