"""
Tests for TB01KX: Block-diagonalization of state-space system with Schur input.

Computes additive spectral decomposition by block-diagonalizing A (assumed in
real Schur form) using Sylvester equation. The leading NDIM-by-NDIM block has
eigenvalues distinct from trailing block.

Transformation:
  A <- V*A*U, B <- V*B, C <- C*U where V = inv(U)

Input Requirements:
  - A must be in real Schur form
  - Leading NDIM eigenvalues distinct from trailing N-NDIM eigenvalues
  - U is an initial transformation matrix (typically identity)

Tests:
1. Basic 2x2 block diagonal case
2. Eigenvalue preservation property
3. Block diagonal structure verification
4. Transformation validity: V = inv(U)
5. Edge case: ndim=0 (no transformation needed)
6. Edge case: ndim=n (no transformation needed)
7. Error handling

Random seeds: 42, 123, 456, 789 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tb01kx_basic_block_diagonal():
    """
    Basic test with 4x4 Schur matrix, splitting into 2+2 blocks.

    Input: A in Schur form with distinct eigenvalues in each block.
    Expected: A12 block becomes zero, eigenvalues preserved.
    """
    from ctrlsys import tb01kx

    n, m, p = 4, 2, 2
    ndim = 2

    a = np.array([
        [-1.0, 0.5, 0.3, 0.2],
        [ 0.0, -2.0, 0.4, 0.1],
        [ 0.0, 0.0, 1.0, 0.6],
        [ 0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.5],
        [0.5, 1.0],
        [0.3, 0.2],
        [0.2, 0.4]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, 0.5, 0.3],
        [0.0, 1.0, 0.2, 0.4]
    ], order='F', dtype=float)

    u = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()

    a_out, b_out, c_out, u_out, v_out, info = tb01kx(
        ndim, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F')
    )

    assert info == 0

    assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-14)

    eig_before = np.sort(np.linalg.eigvals(a_orig).real)
    eig_after = np.sort(np.linalg.eigvals(a_out).real)
    assert_allclose(eig_before, eig_after, rtol=1e-14)


def test_tb01kx_eigenvalue_preservation():
    """
    Validate eigenvalues are preserved by similarity transformation.

    A similarity transformation preserves eigenvalues: lambda(A) = lambda(V*A*U)
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tb01kx

    np.random.seed(42)
    n, m, p = 5, 2, 3
    ndim = 2

    eigs = np.array([-3.0, -2.0, 1.0, 2.0, 3.0])
    a = np.diag(eigs).astype(float, order='F')

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    u = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()

    a_out, b_out, c_out, u_out, v_out, info = tb01kx(
        ndim, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F')
    )

    assert info == 0

    eig_before = np.sort(np.linalg.eigvals(a_orig).real)
    eig_after = np.sort(np.linalg.eigvals(a_out).real)
    assert_allclose(eig_before, eig_after, rtol=1e-14)


def test_tb01kx_block_diagonal_structure():
    """
    Validate A becomes block-diagonal after transformation.

    For 0 < NDIM < N:
      - A12 block (rows 0:NDIM, cols NDIM:N) must be zero
      - Elements below first subdiagonal must be zero (quasi-triangular)
    """
    from ctrlsys import tb01kx

    n, m, p = 6, 2, 2
    ndim = 3

    a = np.array([
        [-1.0, 0.3, 0.2, 0.1, 0.1, 0.1],
        [ 0.0, -2.0, 0.4, 0.2, 0.1, 0.1],
        [ 0.0, 0.0, -3.0, 0.3, 0.2, 0.1],
        [ 0.0, 0.0, 0.0, 1.0, 0.5, 0.2],
        [ 0.0, 0.0, 0.0, 0.0, 2.0, 0.4],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    a_out, b_out, c_out, u_out, v_out, info = tb01kx(
        ndim, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F')
    )

    assert info == 0

    assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-14)

    for i in range(2, n):
        for j in range(i - 1):
            assert abs(a_out[i, j]) < 1e-14, f"a_out[{i},{j}] = {a_out[i,j]} not zero"


def test_tb01kx_transformation_validity():
    """
    Validate V = inv(U) and transformation equations.

    The transformation satisfies:
      - V @ U = I (V is inverse of U)
      - A_out = V @ A_orig @ U (similarity transformation)
      - B_out = V @ B_orig
      - C_out = C_orig @ U

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import tb01kx

    np.random.seed(123)
    n, m, p = 4, 2, 3
    ndim = 2

    a = np.array([
        [-1.0, 0.5, 0.3, 0.2],
        [ 0.0, -2.0, 0.4, 0.1],
        [ 0.0, 0.0, 1.0, 0.6],
        [ 0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    u = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, b_out, c_out, u_out, v_out, info = tb01kx(
        ndim, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F')
    )

    assert info == 0

    vu = v_out @ u_out
    assert_allclose(vu, np.eye(n), atol=1e-14)

    a_reconstructed = v_out @ a_orig @ u_out
    assert_allclose(a_out, a_reconstructed, atol=1e-14)

    b_reconstructed = v_out @ b_orig
    assert_allclose(b_out, b_reconstructed, atol=1e-14)

    c_reconstructed = c_orig @ u_out
    assert_allclose(c_out, c_reconstructed, atol=1e-14)


def test_tb01kx_ndim_zero():
    """
    Edge case: ndim=0 means no leading block, only V = U^T computed.

    When NDIM=0, no Sylvester equation needed, just transpose U to get V.
    """
    from ctrlsys import tb01kx

    n, m, p = 3, 1, 1
    ndim = 0

    a = np.array([
        [1.0, 0.5, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    a_out, b_out, c_out, u_out, v_out, info = tb01kx(
        ndim, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F')
    )

    assert info == 0

    assert_allclose(v_out, u.T, atol=1e-14)


def test_tb01kx_ndim_equals_n():
    """
    Edge case: ndim=n means no trailing block, only V = U^T computed.

    When NDIM=N, no Sylvester equation needed, just transpose U to get V.
    """
    from ctrlsys import tb01kx

    n, m, p = 3, 1, 1
    ndim = n

    a = np.array([
        [-1.0, 0.5, 0.3],
        [ 0.0, -2.0, 0.4],
        [ 0.0, 0.0, -3.0]
    ], order='F', dtype=float)

    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    a_out, b_out, c_out, u_out, v_out, info = tb01kx(
        ndim, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F')
    )

    assert info == 0

    assert_allclose(v_out, u.T, atol=1e-14)


def test_tb01kx_n_zero():
    """
    Edge case: n=0, quick return.
    """
    from ctrlsys import tb01kx

    n, m, p = 0, 2, 2
    ndim = 0

    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, m), order='F', dtype=float)
    c = np.zeros((p, 0), order='F', dtype=float)
    u = np.zeros((0, 0), order='F', dtype=float)

    a_out, b_out, c_out, u_out, v_out, info = tb01kx(
        ndim, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F')
    )

    assert info == 0


def test_tb01kx_complex_eigenvalues():
    """
    Test with 2x2 block containing complex conjugate eigenvalues.

    A Schur form with 2x2 block on diagonal represents complex eigenvalues.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tb01kx

    np.random.seed(456)
    n, m, p = 4, 2, 2
    ndim = 2

    a = np.array([
        [-1.0, 2.0, 0.3, 0.2],
        [-2.0, -1.0, 0.4, 0.1],
        [ 0.0, 0.0, 1.0, 3.0],
        [ 0.0, 0.0, -3.0, 1.0]
    ], order='F', dtype=float)

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    u = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()

    a_out, b_out, c_out, u_out, v_out, info = tb01kx(
        ndim, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F')
    )

    assert info == 0

    assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-13)

    eig_before = np.sort_complex(np.linalg.eigvals(a_orig))
    eig_after = np.sort_complex(np.linalg.eigvals(a_out))
    assert_allclose(eig_before, eig_after, rtol=1e-13)


def test_tb01kx_nonidentity_u():
    """
    Test with non-identity initial U matrix.

    U is accumulated transformation, not just initialized.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tb01kx

    np.random.seed(789)
    n, m, p = 4, 2, 2
    ndim = 2

    a = np.array([
        [-1.0, 0.5, 0.3, 0.2],
        [ 0.0, -2.0, 0.4, 0.1],
        [ 0.0, 0.0, 1.0, 0.6],
        [ 0.0, 0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    q, _ = np.linalg.qr(np.random.randn(n, n))
    u_init = np.asfortranarray(q)

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    b_orig = b.copy()
    c_orig = c.copy()

    a_out, b_out, c_out, u_out, v_out, info = tb01kx(
        ndim, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u_init.copy(order='F')
    )

    assert info == 0

    vu = v_out @ u_out
    assert_allclose(vu, np.eye(n), atol=1e-14)

    assert_allclose(a_out[:ndim, ndim:], 0.0, atol=1e-13)


def test_tb01kx_invalid_ndim():
    """
    Test error handling for invalid NDIM parameter.

    NDIM must satisfy: 0 <= NDIM <= N
    """
    from ctrlsys import tb01kx

    n, m, p = 3, 1, 1

    a = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    with pytest.raises(ValueError):
        tb01kx(-1, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F'))

    with pytest.raises(ValueError):
        tb01kx(n + 1, a.copy(order='F'), b.copy(order='F'), c.copy(order='F'), u.copy(order='F'))
