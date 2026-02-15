"""
Tests for SG03BD - Generalized Lyapunov equation solver (Cholesky factor)

Test data extracted from SLICOT-Reference/doc/SG03BD.html
"""

import pytest
import numpy as np
from ctrlsys import sg03bd


def test_sg03bd_continuous_basic():
    """
    Test SG03BD with reference example from HTML documentation.
    N=3, M=1, DICO='C', FACT='N', TRANS='N'

    Reference data from SG03BD.html Program Data section.
    READ statements:
    - READ ( NIN, FMT = * ) ( ( A(I,J), J = 1,N ), I = 1,N )  # Row-wise
    - READ ( NIN, FMT = * ) ( ( E(I,J), J = 1,N ), I = 1,N )  # Row-wise
    - READ ( NIN, FMT = * ) ( ( B(I,J), J = 1,N ), I = 1,M )  # Row-wise (M x N)
    """
    n = 3
    m = 1

    # A matrix (N x N) - Fortran READ row-wise, stored in Fortran column-major
    a = np.array([
        [-1.0,  3.0, -4.0],
        [ 0.0,  5.0, -2.0],
        [-4.0,  4.0,  1.0]
    ], dtype=np.float64, order='F')

    # E matrix (N x N) - Fortran READ row-wise, stored in Fortran column-major
    e = np.array([
        [2.0, 1.0, 3.0],
        [2.0, 0.0, 1.0],
        [4.0, 5.0, 1.0]
    ], dtype=np.float64, order='F')

    # B matrix (M x N) - Fortran READ row-wise, stored in Fortran column-major
    # For TRANS='N', need LDB >= max(1,M,N) since B is overwritten with N×N matrix U
    b = np.array([
        [2.0, -1.0, 7.0],
        [0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0]
    ], dtype=np.float64, order='F')  # Shape (3, 3)

    # Expected result from Program Results
    # Fortran prints row-by-row, giving upper triangular matrix directly
    expected_u = np.array([
        [1.6003, -0.4418, -0.1523],
        [0.0000,  0.6795, -0.2499],
        [0.0000,  0.0000,  0.2041]
    ], dtype=np.float64, order='F')

    expected_scale = 1.0

    # Save original inputs for verification
    a_orig = np.array([
        [-1.0,  3.0, -4.0],
        [ 0.0,  5.0, -2.0],
        [-4.0,  4.0,  1.0]
    ], dtype=np.float64, order='F')

    e_orig = np.array([
        [2.0, 1.0, 3.0],
        [2.0, 0.0, 1.0],
        [4.0, 5.0, 1.0]
    ], dtype=np.float64, order='F')

    b_orig = np.array([[2.0], [-1.0], [7.0]], dtype=np.float64, order='F')

    # Call SG03BD (positional args: dico, fact, trans, n, m, a, e, b)
    u, scale, alphar, alphai, beta, info = sg03bd('C', 'N', 'N', n, m, a, e, b)

    # Verify success
    assert info == 0, f"SG03BD failed with INFO={info}"
    np.testing.assert_allclose(scale, expected_scale, rtol=1e-14)

    # Verify U is upper triangular
    np.testing.assert_allclose(np.tril(u, -1), 0, atol=1e-14)

    # Verify Lyapunov equation: A^T*X*E + E^T*X*A = -scale^2*B^T*B
    # where X = U^T*U
    x = u.T @ u
    lhs = a_orig.T @ x @ e_orig + e_orig.T @ x @ a_orig
    rhs = -scale**2 * (b_orig @ b_orig.T)
    residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), np.linalg.norm(rhs))
    assert residual < 1e-10, f"Lyapunov equation residual too large: {residual}"

    # Verify eigenvalues are returned (size N)
    assert alphar.shape == (n,)
    assert alphai.shape == (n,)
    assert beta.shape == (n,)


def test_sg03bd_zero_m():
    """Test SG03BD with M=0 (no B matrix)"""
    n = 3
    m = 0

    a = np.array([
        [-1.0,  0.5,  0.0],
        [ 0.0, -2.0,  0.5],
        [ 0.0,  0.0, -3.0]
    ], dtype=np.float64, order='F')

    e = np.eye(3, dtype=np.float64, order='F')

    # Empty B matrix (M=0, N=3)
    # For TRANS='N', need LDB >= max(1,M,N) = 3
    b = np.zeros((max(m, n), n), dtype=np.float64, order='F')

    u, scale, alphar, alphai, beta, info = sg03bd('C', 'N', 'N', n, m, a, e, b)

    # With M=0, U should be zero
    assert info == 0
    np.testing.assert_allclose(u, np.zeros((n, n)), rtol=1e-14)


def test_sg03bd_zero_n():
    """Test SG03BD with N=0 (quick return)"""
    n = 0
    m = 0

    a = np.zeros((1, 1), dtype=np.float64, order='F')  # Need valid arrays
    e = np.zeros((1, 1), dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')

    u, scale, alphar, alphai, beta, info = sg03bd('C', 'N', 'N', n, m, a, e, b)

    # With N=0, should be quick return with INFO=0
    assert info == 0
    assert u.shape == (0, 0)
    assert alphar.shape == (0,)
    assert alphai.shape == (0,)
    assert beta.shape == (0,)


def test_sg03bd_discrete():
    """Test SG03BD with discrete-time equation (DICO='D')"""
    n = 2
    m = 1

    # Stable discrete-time system (eigenvalues inside unit circle)
    a = np.array([
        [0.5, 0.1],
        [0.0, 0.3]
    ], dtype=np.float64, order='F')

    e = np.eye(2, dtype=np.float64, order='F')

    # For TRANS='N', need LDB >= max(1,M,N) = max(1,1,2) = 2
    b = np.array([
        [1.0, 0.5],
        [0.0, 0.0]
    ], dtype=np.float64, order='F')

    u, scale, alphar, alphai, beta, info = sg03bd('D', 'N', 'N', n, m, a, e, b)

    # Should succeed for stable discrete system
    assert info == 0
    assert u.shape == (n, n)
    # U should be upper triangular
    assert np.allclose(np.tril(u, -1), 0.0)


def test_sg03bd_transposed():
    """Test SG03BD with TRANS='T'"""
    n = 2
    m = 1

    a = np.array([
        [-1.0,  0.5],
        [ 0.0, -2.0]
    ], dtype=np.float64, order='F')

    e = np.eye(2, dtype=np.float64, order='F')

    # For TRANS='T', B is N x M
    b = np.array([
        [1.0],
        [0.5]
    ], dtype=np.float64, order='F')

    u, scale, alphar, alphai, beta, info = sg03bd('C', 'N', 'T', n, m, a, e, b)

    assert info == 0
    assert u.shape == (n, n)
    assert np.allclose(np.tril(u, -1), 0.0)


def test_sg03bd_invalid_dico():
    """Test SG03BD with invalid DICO parameter"""
    n = 2
    m = 1

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    # For TRANS='N', need LDB >= max(1,M,N) = 2
    b = np.ones((max(m, n), n), dtype=np.float64, order='F')

    # SLICOT returns INFO=-1 for invalid DICO, not exception
    u, scale, alphar, alphai, beta, info = sg03bd('X', 'N', 'N', n, m, a, e, b)
    assert info == -1


def test_sg03bd_invalid_n():
    """Test SG03BD with negative N"""
    n = -1
    m = 1

    a = np.zeros((1, 1), dtype=np.float64, order='F')
    e = np.zeros((1, 1), dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        sg03bd('C', 'N', 'N', n, m, a, e, b)
