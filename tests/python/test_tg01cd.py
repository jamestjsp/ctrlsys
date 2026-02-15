"""
Tests for TG01CD - QR-coordinate form reduction of descriptor system.

TG01CD reduces the descriptor system pair (A-lambda E, B) to QR-coordinate form
by computing an orthogonal transformation matrix Q such that Q'*E is upper
trapezoidal.

Test data sources:
- SLICOT HTML documentation TG01CD.html example
- Mathematical properties of QR transformation
"""

import numpy as np
import pytest

from ctrlsys import tg01cd


def test_tg01cd_html_example():
    """
    Test using HTML documentation example data.

    Validates numerical correctness against published example.
    """
    l, n, m = 4, 4, 2

    # Input from HTML: row-wise (( A(I,J), J = 1,N ), I = 1,L)
    a = np.array([
        [-1.0,  0.0,  0.0,  3.0],
        [ 0.0,  0.0,  1.0,  2.0],
        [ 1.0,  1.0,  0.0,  4.0],
        [ 0.0,  0.0,  0.0,  0.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [3.0, 9.0, 6.0, 3.0],
        [0.0, 0.0, 2.0, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ], order='F', dtype=float)

    a_out, e_out, b_out, q_out, info = tg01cd('I', a, e, b)

    assert info == 0

    # Expected from HTML docs
    a_expected = np.array([
        [-0.6325, -0.9487,  0.0000, -4.7434],
        [-0.8706, -0.2176, -0.7255, -0.3627],
        [-0.5203, -0.1301,  0.3902,  1.4307],
        [-0.7559, -0.1890,  0.5669,  2.0788]
    ], order='F', dtype=float)

    e_expected = np.array([
        [-3.1623, -9.1706, -5.6921, -2.8460],
        [ 0.0000, -1.3784, -1.3059, -1.3784],
        [ 0.0000,  0.0000, -2.4279,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000]
    ], order='F', dtype=float)

    b_expected = np.array([
        [-0.3162, -0.9487],
        [ 0.6529, -0.2176],
        [-0.4336, -0.9538],
        [ 1.1339,  0.3780]
    ], order='F', dtype=float)

    q_expected = np.array([
        [-0.3162,  0.6529,  0.3902,  0.5669],
        [ 0.0000, -0.7255,  0.3902,  0.5669],
        [-0.9487, -0.2176, -0.1301, -0.1890],
        [ 0.0000,  0.0000, -0.8238,  0.5669]
    ], order='F', dtype=float)

    # HTML shows 4 decimal places, use rtol=1e-3
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(e_out, e_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(q_out, q_expected, rtol=1e-3, atol=1e-4)


def test_tg01cd_upper_trapezoidal():
    """
    Test that Q'*E is upper trapezoidal after transformation.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    l, n, m = 5, 4, 2

    a = np.random.randn(l, n).astype(float, order='F')
    e = np.random.randn(l, n).astype(float, order='F')
    b = np.random.randn(l, m).astype(float, order='F')

    a_out, e_out, b_out, q_out, info = tg01cd('I', a, e, b)

    assert info == 0

    # E should be upper trapezoidal (zeros below diagonal in first min(l,n) columns)
    ln = min(l, n)
    for i in range(ln + 1, l):
        for j in range(ln):
            assert abs(e_out[i, j]) < 1e-14, f"E[{i},{j}] = {e_out[i, j]} should be zero"


def test_tg01cd_orthogonality():
    """
    Test that Q is orthogonal: Q'*Q = I.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    l, n, m = 4, 4, 2

    a = np.random.randn(l, n).astype(float, order='F')
    e = np.random.randn(l, n).astype(float, order='F')
    b = np.random.randn(l, m).astype(float, order='F')

    a_out, e_out, b_out, q_out, info = tg01cd('I', a, e, b)

    assert info == 0

    # Q should be orthogonal
    qtq = q_out.T @ q_out
    np.testing.assert_allclose(qtq, np.eye(l), rtol=1e-14, atol=1e-14)


def test_tg01cd_transformation_consistency():
    """
    Test that transformations are consistent: A_out = Q'*A, E_out = Q'*E, B_out = Q'*B.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    l, n, m = 4, 4, 2

    a = np.random.randn(l, n).astype(float, order='F')
    e = np.random.randn(l, n).astype(float, order='F')
    b = np.random.randn(l, m).astype(float, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    a_out, e_out, b_out, q_out, info = tg01cd('I', a, e, b)

    assert info == 0

    # A_out = Q' * A
    a_check = q_out.T @ a_orig
    np.testing.assert_allclose(a_out, a_check, rtol=1e-13, atol=1e-14)

    # E_out = Q' * E
    e_check = q_out.T @ e_orig
    np.testing.assert_allclose(e_out, e_check, rtol=1e-13, atol=1e-14)

    # B_out = Q' * B
    b_check = q_out.T @ b_orig
    np.testing.assert_allclose(b_out, b_check, rtol=1e-13, atol=1e-14)


def test_tg01cd_compq_n():
    """
    Test with COMPQ='N' (do not compute Q).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    l, n, m = 3, 3, 1

    a = np.random.randn(l, n).astype(float, order='F')
    e = np.random.randn(l, n).astype(float, order='F')
    b = np.random.randn(l, m).astype(float, order='F')

    a_out, e_out, b_out, q_out, info = tg01cd('N', a, e, b)

    assert info == 0

    # E should still be upper trapezoidal
    for i in range(1, l):
        for j in range(min(i, n)):
            assert abs(e_out[i, j]) < 1e-14


def test_tg01cd_compq_u():
    """
    Test with COMPQ='U' (update existing Q).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    l, n, m = 4, 4, 2

    a = np.random.randn(l, n).astype(float, order='F')
    e = np.random.randn(l, n).astype(float, order='F')
    b = np.random.randn(l, m).astype(float, order='F')

    # Create initial orthogonal Q1 from QR of random matrix
    q1_temp = np.random.randn(l, l).astype(float, order='F')
    q1, _ = np.linalg.qr(q1_temp)
    q1 = np.asfortranarray(q1)

    a_out, e_out, b_out, q_out, info = tg01cd('U', a, e, b, q1.copy())

    assert info == 0

    # Q_out should still be orthogonal
    qtq = q_out.T @ q_out
    np.testing.assert_allclose(qtq, np.eye(l), rtol=1e-14, atol=1e-14)


def test_tg01cd_wide_matrix():
    """
    Test with L < N (wide matrix, more columns than rows).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    l, n, m = 3, 5, 2

    a = np.random.randn(l, n).astype(float, order='F')
    e = np.random.randn(l, n).astype(float, order='F')
    b = np.random.randn(l, m).astype(float, order='F')

    a_out, e_out, b_out, q_out, info = tg01cd('I', a, e, b)

    assert info == 0

    # E should be upper trapezoidal: E11 (L x L upper triangular), E12 (L x (N-L))
    for i in range(1, l):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-14, f"E[{i},{j}] = {e_out[i, j]} should be zero"


def test_tg01cd_tall_matrix():
    """
    Test with L > N (tall matrix, more rows than columns).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    l, n, m = 5, 3, 2

    a = np.random.randn(l, n).astype(float, order='F')
    e = np.random.randn(l, n).astype(float, order='F')
    b = np.random.randn(l, m).astype(float, order='F')

    a_out, e_out, b_out, q_out, info = tg01cd('I', a, e, b)

    assert info == 0

    # E should be upper trapezoidal with zeros in rows N+1:L
    for i in range(n, l):
        for j in range(n):
            assert abs(e_out[i, j]) < 1e-14, f"E[{i},{j}] = {e_out[i, j]} should be zero"


def test_tg01cd_zero_m():
    """
    Test with M=0 (no B matrix columns).

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    l, n, m = 4, 4, 0

    a = np.random.randn(l, n).astype(float, order='F')
    e = np.random.randn(l, n).astype(float, order='F')
    b = np.empty((l, 0), order='F', dtype=float)

    a_out, e_out, b_out, q_out, info = tg01cd('I', a, e, b)

    assert info == 0
    assert b_out.shape == (l, 0)


def test_tg01cd_quick_return():
    """
    Test quick return for L=0 or N=0.
    """
    # L=0
    a = np.empty((0, 4), order='F', dtype=float)
    e = np.empty((0, 4), order='F', dtype=float)
    b = np.empty((0, 2), order='F', dtype=float)

    a_out, e_out, b_out, q_out, info = tg01cd('I', a, e, b)
    assert info == 0

    # N=0
    a = np.empty((4, 0), order='F', dtype=float)
    e = np.empty((4, 0), order='F', dtype=float)
    b = np.random.randn(4, 2).astype(float, order='F')

    a_out, e_out, b_out, q_out, info = tg01cd('I', a, e, b)
    assert info == 0


def test_tg01cd_invalid_compq():
    """
    Test error handling for invalid COMPQ parameter.
    """
    l, n, m = 3, 3, 1

    a = np.zeros((l, n), order='F', dtype=float)
    e = np.zeros((l, n), order='F', dtype=float)
    b = np.zeros((l, m), order='F', dtype=float)

    a_out, e_out, b_out, q_out, info = tg01cd('X', a, e, b)
    assert info == -1
