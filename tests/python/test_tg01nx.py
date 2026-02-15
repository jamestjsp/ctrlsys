"""
Tests for TG01NX - Block-diagonal decomposition of descriptor system in generalized Schur form.

TG01NX computes equivalence transformation matrices Q and Z which reduce
the regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C),
with (A,E) in generalized real Schur form, to block-diagonal form.

The decomposition produces:
    Q*A*Z = diag(A1, A2)
    Q*E*Z = diag(E1, E2)

where (A1,E1) and (A2,E2) have no common generalized eigenvalues.
"""
import pytest
import numpy as np
from ctrlsys import tg01nx


def test_tg01nx_basic_direct():
    """Test TG01NX with direct transformation (JOBT='D').

    Creates a simple 4x4 system in generalized Schur form and verifies
    that the routine produces block-diagonal A and E matrices.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 2, 2
    ndim = 2

    a = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.0, 2.0, 0.4, 0.1],
        [0.0, 0.0, 3.0, 0.6],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.2, 0.05],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.0, 0.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == 0, f"TG01NX failed with info={info}"

    tol = 1e-14
    np.testing.assert_allclose(a_out[:ndim, ndim:], np.zeros((ndim, n-ndim)),
                               atol=tol, err_msg="A12 block should be zero")
    np.testing.assert_allclose(e_out[:ndim, ndim:], np.zeros((ndim, n-ndim)),
                               atol=tol, err_msg="E12 block should be zero")


def test_tg01nx_inverse_transformation():
    """Test TG01NX with inverse transformation (JOBT='I').

    When JOBT='I', the routine computes inv(Q) and inv(Z) instead of Q and Z.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 4, 2, 2
    ndim = 2

    a = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.0, 2.0, 0.4, 0.1],
        [0.0, 0.0, 3.0, 0.6],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.2, 0.05],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.0, 0.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'I', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == 0, f"TG01NX failed with info={info}"

    tol = 1e-14
    np.testing.assert_allclose(a_out[:ndim, ndim:], np.zeros((ndim, n-ndim)),
                               atol=tol, err_msg="A12 block should be zero")
    np.testing.assert_allclose(e_out[:ndim, ndim:], np.zeros((ndim, n-ndim)),
                               atol=tol, err_msg="E12 block should be zero")


def test_tg01nx_edge_ndim_zero():
    """Test TG01NX with NDIM=0 (no leading block).

    When NDIM=0, there's nothing to separate - should return immediately.
    """
    n, m, p = 4, 2, 2
    ndim = 0

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')
    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_orig = a.copy(order='F')
    e_orig = e.copy(order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == 0, f"TG01NX failed with info={info}"


def test_tg01nx_edge_ndim_equals_n():
    """Test TG01NX with NDIM=N (no trailing block).

    When NDIM=N, there's no trailing block to separate.
    """
    n, m, p = 4, 2, 2
    ndim = n

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')
    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == 0, f"TG01NX failed with info={info}"


def test_tg01nx_edge_n_zero():
    """Test TG01NX with N=0 (empty system).

    Should return immediately with info=0.
    """
    n, m, p = 0, 2, 2
    ndim = 0

    a = np.zeros((0, 0), dtype=np.float64, order='F')
    e = np.zeros((0, 0), dtype=np.float64, order='F')
    b = np.zeros((0, m), dtype=np.float64, order='F')
    c = np.zeros((p, 0), dtype=np.float64, order='F')
    q = np.zeros((0, 0), dtype=np.float64, order='F')
    z = np.zeros((0, 0), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == 0, f"TG01NX should succeed with N=0, got info={info}"


def test_tg01nx_invalid_jobt():
    """Test TG01NX with invalid JOBT parameter."""
    n, m, p = 4, 2, 2
    ndim = 2

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')
    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'X', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == -1, f"Expected info=-1 for invalid JOBT, got {info}"


def test_tg01nx_invalid_n():
    """Test TG01NX with invalid N parameter (N < 0)."""
    a = np.eye(1, dtype=np.float64, order='F')
    e = np.eye(1, dtype=np.float64, order='F')
    b = np.ones((1, 1), dtype=np.float64, order='F')
    c = np.ones((1, 1), dtype=np.float64, order='F')
    q = np.eye(1, dtype=np.float64, order='F')
    z = np.eye(1, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', -1, 1, 1, 0, a, e, b, c, q, z
    )

    assert info == -2, f"Expected info=-2 for N<0, got {info}"


def test_tg01nx_invalid_ndim():
    """Test TG01NX with invalid NDIM parameter (NDIM > N)."""
    n, m, p = 4, 2, 2
    ndim = 5

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')
    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == -5, f"Expected info=-5 for NDIM>N, got {info}"


def test_tg01nx_eigenvalue_preservation():
    """Test TG01NX preserves eigenvalues of the pencil (A,E).

    The generalized eigenvalues of (A,E) should be preserved by the
    similarity transformation.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 4, 2, 2
    ndim = 2

    a = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.0, 2.0, 0.4, 0.1],
        [0.0, 0.0, 3.0, 0.6],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.2, 0.05],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    eig_before = np.linalg.eigvals(np.linalg.solve(e.T, a.T).T)

    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')
    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == 0, f"TG01NX failed with info={info}"

    eig_after = np.linalg.eigvals(np.linalg.solve(e_out.T, a_out.T).T)

    np.testing.assert_allclose(
        sorted(eig_before.real), sorted(eig_after.real),
        rtol=1e-12, atol=1e-14,
        err_msg="Eigenvalues not preserved"
    )


def test_tg01nx_transformation_property_direct():
    """Test TG01NX transformation: Q*A_orig*Z = A_out for JOBT='D'.

    Validates that the transformation matrices correctly relate the
    original and transformed system matrices.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 4, 2, 2
    ndim = 2

    a_orig = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.0, 2.0, 0.4, 0.1],
        [0.0, 0.0, 3.0, 0.6],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    e_orig = np.array([
        [1.0, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.2, 0.05],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b_orig = np.random.randn(n, m).astype(np.float64, order='F')
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')

    a = a_orig.copy(order='F')
    e = e_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')
    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == 0, f"TG01NX failed with info={info}"

    a_check = q_out @ a_orig @ z_out
    e_check = q_out @ e_orig @ z_out
    b_check = q_out @ b_orig
    c_check = c_orig @ z_out

    np.testing.assert_allclose(a_out, a_check, rtol=1e-13, atol=1e-14,
                               err_msg="Q*A*Z != A_out")
    np.testing.assert_allclose(e_out, e_check, rtol=1e-13, atol=1e-14,
                               err_msg="Q*E*Z != E_out")
    np.testing.assert_allclose(b_out, b_check, rtol=1e-13, atol=1e-14,
                               err_msg="Q*B != B_out")
    np.testing.assert_allclose(c_out, c_check, rtol=1e-13, atol=1e-14,
                               err_msg="C*Z != C_out")


def test_tg01nx_transformation_property_inverse():
    """Test TG01NX transformation for JOBT='I'.

    With JOBT='I', the routine returns inv(Q) and inv(Z) where Q and Z
    are the transformation matrices from JOBT='D'. This test verifies
    that Q_i @ Q_d = I and Z_i @ Z_d = I.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 4, 2, 2
    ndim = 2

    a_orig = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.0, 2.0, 0.4, 0.1],
        [0.0, 0.0, 3.0, 0.6],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    e_orig = np.array([
        [1.0, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.2, 0.05],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b_orig = np.random.randn(n, m).astype(np.float64, order='F')
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')

    # Run with JOBT='D' first
    a_d = a_orig.copy(order='F')
    e_d = e_orig.copy(order='F')
    b_d = b_orig.copy(order='F')
    c_d = c_orig.copy(order='F')
    q_d = np.eye(n, dtype=np.float64, order='F')
    z_d = np.eye(n, dtype=np.float64, order='F')

    _, _, _, _, q_direct, z_direct, info_d = tg01nx(
        'D', n, m, p, ndim, a_d, e_d, b_d, c_d, q_d, z_d
    )
    assert info_d == 0

    # Run with JOBT='I'
    a_i = a_orig.copy(order='F')
    e_i = e_orig.copy(order='F')
    b_i = b_orig.copy(order='F')
    c_i = c_orig.copy(order='F')
    q_i = np.eye(n, dtype=np.float64, order='F')
    z_i = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, _, _, q_inv, z_inv, info_i = tg01nx(
        'I', n, m, p, ndim, a_i, e_i, b_i, c_i, q_i, z_i
    )
    assert info_i == 0

    # Verify Q_inv @ Q_direct = I
    np.testing.assert_allclose(q_inv @ q_direct, np.eye(n), rtol=1e-13, atol=1e-14,
                               err_msg="Q_inv @ Q_direct != I")

    # Verify Z_inv @ Z_direct = I
    np.testing.assert_allclose(z_inv @ z_direct, np.eye(n), rtol=1e-13, atol=1e-14,
                               err_msg="Z_inv @ Z_direct != I")

    # Verify both modes produce same block-diagonal A and E
    np.testing.assert_allclose(a_out, a_d, rtol=1e-13, atol=1e-14,
                               err_msg="A_out differs between JOBT='D' and 'I'")
    np.testing.assert_allclose(e_out, e_d, rtol=1e-13, atol=1e-14,
                               err_msg="E_out differs between JOBT='D' and 'I'")


def test_tg01nx_m_zero():
    """Test TG01NX with M=0 (no inputs)."""
    n, m, p = 4, 0, 2
    ndim = 2

    a = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.0, 2.0, 0.4, 0.1],
        [0.0, 0.0, 3.0, 0.6],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.2, 0.05],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.zeros((n, 0), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')
    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == 0, f"TG01NX failed with info={info}"


def test_tg01nx_p_zero():
    """Test TG01NX with P=0 (no outputs)."""
    n, m, p = 4, 2, 0
    ndim = 2

    a = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.0, 2.0, 0.4, 0.1],
        [0.0, 0.0, 3.0, 0.6],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.2, 0.05],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.zeros((0, n), dtype=np.float64, order='F')
    q = np.eye(n, dtype=np.float64, order='F')
    z = np.eye(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01nx(
        'D', n, m, p, ndim, a, e, b, c, q, z
    )

    assert info == 0, f"TG01NX failed with info={info}"
