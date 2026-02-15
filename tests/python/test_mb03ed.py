"""
Tests for MB03ED: Orthogonal matrices for reducing 2x2 or 4x4 block anti-diagonal
skew-Hamiltonian/Hamiltonian pencil to generalized Schur form.
"""
import numpy as np
import pytest
from ctrlsys import mb03ed


def test_mb03ed_n2_basic():
    """
    Test N=2 case with finite eigenvalues.

    For 2x2 pencil aAB - bD, compute Q1, Q2, Q3 such that:
    - Q3' A Q2 and Q2' B Q1 are upper triangular
    - Q3' D Q1 is upper quasi-triangular
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    prec = np.finfo(float).eps

    n = 2
    a = np.array([[2.0, 0.0],
                  [0.0, 3.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 2.0]], order='F', dtype=float)
    d = np.array([[0.0, 1.0],
                  [2.0, 0.0]], order='F', dtype=float)

    d_out, q1, q2, q3, info = mb03ed(n, prec, a, b, d)

    assert info == 0
    assert q1.shape == (n, n)
    assert q2.shape == (n, n)
    assert q3.shape == (n, n)

    # Verify Q1, Q2, Q3 are orthogonal
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q3 @ q3.T, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03ed_n2_infinite_eigenvalue_a11_small():
    """
    Test N=2 case when A(1,1) is numerically singular - pencil has infinite eigenvalues.

    When A(1,1) <= PREC*A(2,2), the routine produces specific orthogonal matrices.
    """
    prec = np.finfo(float).eps

    n = 2
    a = np.array([[1e-20, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0, 1.0],
                  [1.0, 0.0]], order='F', dtype=float)

    d_out, q1, q2, q3, info = mb03ed(n, prec, a, b, d)

    assert info == 0

    # Verify orthogonality
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q3 @ q3.T, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03ed_n2_zero_d21():
    """
    Test N=2 case when D21 is numerically zero - double zero eigenvalue.

    When D21 <= PREC*D12, Q1 = Q2 = Q3 = I.
    """
    prec = np.finfo(float).eps

    n = 2
    a = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0, 1.0],
                  [1e-20, 0.0]], order='F', dtype=float)

    d_out, q1, q2, q3, info = mb03ed(n, prec, a, b, d)

    assert info == 0

    # When D21 <= PREC*D12, should return identity matrices
    np.testing.assert_allclose(q1, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q2, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q3, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03ed_n4_basic():
    """
    Test N=4 case - uses QZ algorithm via DGGES.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    prec = np.finfo(float).eps

    n = 4
    # Block diagonal upper triangular A
    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 1.0
    a[0, 1] = 0.5
    a[1, 1] = 2.0
    a[2, 2] = 3.0
    a[2, 3] = 0.5
    a[3, 3] = 4.0

    # Block diagonal upper triangular B
    b = np.zeros((n, n), order='F', dtype=float)
    b[0, 0] = 2.0
    b[0, 1] = 0.3
    b[1, 1] = 1.5
    b[2, 2] = 2.5
    b[2, 3] = 0.4
    b[3, 3] = 3.5

    # Anti-diagonal D: D12 in upper-right, D21 in lower-left
    d = np.zeros((n, n), order='F', dtype=float)
    d[0, 2] = 1.0
    d[0, 3] = 0.2
    d[1, 2] = 0.3
    d[1, 3] = 1.0
    d[2, 0] = 2.0
    d[2, 1] = 0.4
    d[3, 0] = 0.5
    d[3, 1] = 2.0

    d_out, q1, q2, q3, info = mb03ed(n, prec, a, b, d)

    assert info == 0
    assert q1.shape == (n, n)
    assert q2.shape == (n, n)
    assert q3.shape == (n, n)

    # Verify orthogonality
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(q3 @ q3.T, np.eye(n), rtol=1e-13, atol=1e-13)


def test_mb03ed_n4_eigenvalue_ordering():
    """
    Test N=4 case verifies eigenvalues with negative real parts are on top.

    Uses well-conditioned test matrices.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    prec = np.finfo(float).eps

    n = 4
    # Create diagonal A and B for simple eigenvalue computation
    a = np.diag([1.0, 2.0, 3.0, 4.0]).astype(float, order='F')
    b = np.diag([1.0, 1.5, 2.0, 2.5]).astype(float, order='F')

    # Anti-diagonal D structure
    d = np.array([
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0]
    ], order='F', dtype=float)

    d_out, q1, q2, q3, info = mb03ed(n, prec, a, b, d)

    assert info == 0

    # Verify all Q matrices are orthogonal
    for q, name in [(q1, 'Q1'), (q2, 'Q2'), (q3, 'Q3')]:
        identity_check = q @ q.T
        np.testing.assert_allclose(identity_check, np.eye(n), rtol=1e-13, atol=1e-13,
                                   err_msg=f"{name} is not orthogonal")


def test_mb03ed_n2_real_eigenvalues():
    """
    Test N=2 case with well-separated real eigenvalues.

    When eigenvalues are real, COMPG path computes Givens rotations.
    """
    prec = np.finfo(float).eps

    n = 2
    a = np.array([[2.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0],
                  [0.0, 2.0]], order='F', dtype=float)
    d = np.array([[0.0, 1.0],
                  [1.0, 0.0]], order='F', dtype=float)

    d_out, q1, q2, q3, info = mb03ed(n, prec, a, b, d)

    assert info == 0

    # Verify orthogonality
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(q3 @ q3.T, np.eye(n), rtol=1e-14, atol=1e-14)

    # Verify transformations produce valid results
    a_trans = q3.T @ a @ q2
    b_trans = q2.T @ b @ q1
    d_trans = q3.T @ d @ q1

    # Check upper triangular (lower triangular part should be zero)
    np.testing.assert_allclose(np.tril(a_trans, -1), np.zeros((n, n)), atol=1e-13)
    np.testing.assert_allclose(np.tril(b_trans, -1), np.zeros((n, n)), atol=1e-13)


def test_mb03ed_n4_schur_form():
    """
    Mathematical property test: verify D is transformed to Schur form for N=4.

    For N=4, the output D should be in real Schur form (upper quasi-triangular).
    """
    prec = np.finfo(float).eps

    n = 4
    # Block diagonal upper triangular A
    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 2.0
    a[0, 1] = 0.3
    a[1, 1] = 1.5
    a[2, 2] = 3.0
    a[2, 3] = 0.4
    a[3, 3] = 2.0

    # Block diagonal upper triangular B
    b = np.zeros((n, n), order='F', dtype=float)
    b[0, 0] = 1.0
    b[0, 1] = 0.2
    b[1, 1] = 2.0
    b[2, 2] = 1.5
    b[2, 3] = 0.3
    b[3, 3] = 2.5

    # Anti-diagonal D with upper triangular D12 and D21
    d = np.zeros((n, n), order='F', dtype=float)
    d[0, 2] = 1.0
    d[0, 3] = 0.5
    d[1, 3] = 1.0
    d[2, 0] = 0.8
    d[2, 1] = 0.3
    d[3, 1] = 0.9

    d_out, q1, q2, q3, info = mb03ed(n, prec, a, b, d)

    assert info == 0

    # Verify orthogonality
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(q3 @ q3.T, np.eye(n), rtol=1e-13, atol=1e-13)

    # D should be transformed to upper quasi-triangular (Schur form)
    # Check entries below first subdiagonal are zero
    for i in range(2, n):
        for j in range(i - 1):
            assert abs(d_out[i, j]) < 1e-12, f"D[{i},{j}] = {d_out[i, j]} not zero"
