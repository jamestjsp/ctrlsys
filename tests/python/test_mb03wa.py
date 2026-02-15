"""
Tests for MB03WA - Swapping two adjacent diagonal blocks in periodic real Schur form.

MB03WA swaps adjacent diagonal blocks A11*B11 and A22*B22 of size 1x1 or 2x2
in an upper (quasi) triangular matrix product A*B by an orthogonal equivalence
transformation.

Test data sources:
- Mathematical properties of periodic Schur form
- Eigenvalue preservation after swapping
"""

import numpy as np
import pytest

from ctrlsys import mb03wa


def test_mb03wa_swap_1x1_blocks():
    """
    Test swapping two 1x1 blocks.

    For N1=N2=1, the matrices are 2x2 and the swap involves simple Givens rotations.
    Eigenvalues of A*B product should be preserved.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    a = np.array([
        [2.0, 0.5],
        [0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.3],
        [0.0, 2.0]
    ], order='F', dtype=float)

    q = np.eye(2, order='F', dtype=float)
    z = np.eye(2, order='F', dtype=float)

    eig_before = np.linalg.eigvals(a @ b)

    a_out, b_out, q_out, z_out, info = mb03wa(True, True, 1, 1, a, b, q, z)

    assert info == 0, f"Expected info=0, got {info}"

    eig_after = np.linalg.eigvals(a_out @ b_out)
    np.testing.assert_allclose(
        sorted(eig_before.real),
        sorted(eig_after.real),
        rtol=1e-10
    )

    np.testing.assert_allclose(q_out @ q_out.T, np.eye(2), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z_out @ z_out.T, np.eye(2), rtol=1e-14, atol=1e-14)


def test_mb03wa_swap_1x1_and_2x2_blocks():
    """
    Test swapping 1x1 block with 2x2 block.

    N1=1, N2=2 means 3x3 matrices.
    Eigenvalues should be preserved.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    a = np.array([
        [3.0,  0.5, 0.2],
        [0.0,  1.0, 2.0],
        [0.0, -0.5, 1.0]
    ], order='F', dtype=float)

    b = np.array([
        [2.0, 0.3, 0.1],
        [0.0, 1.5, 0.4],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    q = np.eye(3, order='F', dtype=float)
    z = np.eye(3, order='F', dtype=float)

    eig_before = np.linalg.eigvals(a @ b)

    a_out, b_out, q_out, z_out, info = mb03wa(True, True, 1, 2, a, b, q, z)

    if info == 0:
        eig_after = np.linalg.eigvals(a_out @ b_out)
        np.testing.assert_allclose(
            sorted(np.abs(eig_before)),
            sorted(np.abs(eig_after)),
            rtol=1e-8
        )

        np.testing.assert_allclose(q_out @ q_out.T, np.eye(3), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(3), rtol=1e-14, atol=1e-14)
    else:
        assert info == 1


def test_mb03wa_swap_2x2_and_1x1_blocks():
    """
    Test swapping 2x2 block with 1x1 block.

    N1=2, N2=1 means 3x3 matrices.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    a = np.array([
        [1.0,  2.0, 0.3],
        [-0.5, 1.0, 0.2],
        [0.0,  0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.5, 0.3, 0.1],
        [0.0, 1.0, 0.2],
        [0.0, 0.0, 2.0]
    ], order='F', dtype=float)

    q = np.eye(3, order='F', dtype=float)
    z = np.eye(3, order='F', dtype=float)

    eig_before = np.linalg.eigvals(a @ b)

    a_out, b_out, q_out, z_out, info = mb03wa(True, True, 2, 1, a, b, q, z)

    if info == 0:
        eig_after = np.linalg.eigvals(a_out @ b_out)
        np.testing.assert_allclose(
            sorted(np.abs(eig_before)),
            sorted(np.abs(eig_after)),
            rtol=1e-8
        )

        np.testing.assert_allclose(q_out @ q_out.T, np.eye(3), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(3), rtol=1e-14, atol=1e-14)
    else:
        assert info == 1


def test_mb03wa_swap_2x2_blocks():
    """
    Test swapping two 2x2 blocks.

    N1=N2=2 means 4x4 matrices.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    a = np.array([
        [1.0,  2.0,  0.3, 0.1],
        [-0.5, 1.0,  0.2, 0.0],
        [0.0,  0.0,  2.0, 1.5],
        [0.0,  0.0, -0.3, 2.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.5, 0.3, 0.1, 0.05],
        [0.0, 1.0, 0.2, 0.1],
        [0.0, 0.0, 2.0, 0.4],
        [0.0, 0.0, 0.0, 1.5]
    ], order='F', dtype=float)

    q = np.eye(4, order='F', dtype=float)
    z = np.eye(4, order='F', dtype=float)

    eig_before = np.linalg.eigvals(a @ b)

    a_out, b_out, q_out, z_out, info = mb03wa(True, True, 2, 2, a, b, q, z)

    if info == 0:
        eig_after = np.linalg.eigvals(a_out @ b_out)
        np.testing.assert_allclose(
            sorted(np.abs(eig_before)),
            sorted(np.abs(eig_after)),
            rtol=1e-8
        )

        np.testing.assert_allclose(q_out @ q_out.T, np.eye(4), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(z_out @ z_out.T, np.eye(4), rtol=1e-14, atol=1e-14)
    else:
        assert info == 1


def test_mb03wa_no_q_accumulation():
    """
    Test with WANTQ=False - Q should not be referenced.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    a = np.array([
        [2.0, 0.5],
        [0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.3],
        [0.0, 2.0]
    ], order='F', dtype=float)

    q = np.zeros((2, 2), order='F', dtype=float)
    z = np.eye(2, order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03wa(False, True, 1, 1, a, b, q, z)

    assert info == 0
    np.testing.assert_allclose(z_out @ z_out.T, np.eye(2), rtol=1e-14, atol=1e-14)


def test_mb03wa_no_z_accumulation():
    """
    Test with WANTZ=False - Z should not be referenced.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)

    a = np.array([
        [2.0, 0.5],
        [0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.3],
        [0.0, 2.0]
    ], order='F', dtype=float)

    q = np.eye(2, order='F', dtype=float)
    z = np.zeros((2, 2), order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03wa(True, False, 1, 1, a, b, q, z)

    assert info == 0
    np.testing.assert_allclose(q_out @ q_out.T, np.eye(2), rtol=1e-14, atol=1e-14)


def test_mb03wa_zero_n1():
    """
    Test with N1=0 - quick return expected.
    """
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[2.0]], order='F', dtype=float)
    q = np.eye(1, order='F', dtype=float)
    z = np.eye(1, order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03wa(True, True, 0, 1, a, b, q, z)

    assert info == 0
    np.testing.assert_allclose(a_out, a, rtol=1e-14)
    np.testing.assert_allclose(b_out, b, rtol=1e-14)


def test_mb03wa_zero_n2():
    """
    Test with N2=0 - quick return expected.
    """
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[2.0]], order='F', dtype=float)
    q = np.eye(1, order='F', dtype=float)
    z = np.eye(1, order='F', dtype=float)

    a_out, b_out, q_out, z_out, info = mb03wa(True, True, 1, 0, a, b, q, z)

    assert info == 0
    np.testing.assert_allclose(a_out, a, rtol=1e-14)
    np.testing.assert_allclose(b_out, b, rtol=1e-14)


def test_mb03wa_transformation_consistency():
    """
    Test that Q and Z satisfy the periodic Schur transformation equations:
        Q(in) * A(in) * Z(in)' = Q(out) * A(out) * Z(out)'
        Z(in) * B(in) * Q(in)' = Z(out) * B(out) * Q(out)'

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)

    a = np.array([
        [2.0, 0.5],
        [0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.3],
        [0.0, 2.0]
    ], order='F', dtype=float)

    q_in = np.eye(2, order='F', dtype=float)
    z_in = np.eye(2, order='F', dtype=float)

    a_in = a.copy()
    b_in = b.copy()

    a_out, b_out, q_out, z_out, info = mb03wa(True, True, 1, 1, a, b, q_in.copy(), z_in.copy())

    if info == 0:
        lhs_a = q_in @ a_in @ z_in.T
        rhs_a = q_out @ a_out @ z_out.T
        np.testing.assert_allclose(lhs_a, rhs_a, rtol=1e-10, atol=1e-14)

        lhs_b = z_in @ b_in @ q_in.T
        rhs_b = z_out @ b_out @ q_out.T
        np.testing.assert_allclose(lhs_b, rhs_b, rtol=1e-10, atol=1e-14)
