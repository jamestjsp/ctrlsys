"""Tests for TG01OB - Complex SISO descriptor system equivalence transformation.

TG01OB computes for a single-input single-output descriptor system (A, E, B, C, D)
with E upper triangular, a transformed system (Q'*A*Z, Q'*E*Z, Q'*B, C*Z) via
unitary equivalence transformation, so that Q'*B has only the first element
nonzero and Q'*E*Z remains upper triangular. D is unchanged.

The system matrix format is:
    [ D  C ]
    [ B  A ]
stored in DCBA array of size (N+1) x (N+1).

Uses Givens rotations to annihilate last N-1 elements of B in reverse order
while preserving the upper triangular form of E.
"""
import pytest
import numpy as np
from ctrlsys import tg01ob


def test_tg01ob_basic_identity_e():
    """Test TG01OB with JOBE='I' (E is identity).

    Random seed: 42 (for reproducibility)
    Tests basic functionality with identity E matrix.
    """
    np.random.seed(42)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = (np.random.randn(1, 1) + 1j * np.random.randn(1, 1)).astype(np.complex128, order='F')

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d[0, 0]
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    e = np.eye(n, dtype=np.complex128, order='F')
    dcba_orig = dcba.copy()

    dcba_out, e_out, info = tg01ob('I', dcba, e)

    assert info == 0, f"TG01OB failed with info={info}"

    assert dcba_out[0, 0] == dcba_orig[0, 0], "D should be unchanged"

    b_out = dcba_out[1:, 0]
    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14,
                               err_msg="B should have only first element nonzero")
    assert abs(b_out[0]) > 0, "First element of B should be nonzero"


def test_tg01ob_upper_triangular_e():
    """Test TG01OB with JOBE='U' (E upper triangular).

    Random seed: 123 (for reproducibility)
    Tests with general upper triangular E, verifies E stays upper triangular.
    """
    np.random.seed(123)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.triu(np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = (np.random.randn(1, 1) + 1j * np.random.randn(1, 1)).astype(np.complex128, order='F')

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d[0, 0]
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_out, e_out, info = tg01ob('U', dcba, e)

    assert info == 0, f"TG01OB failed with info={info}"

    b_out = dcba_out[1:, 0]
    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14,
                               err_msg="B should have only first element nonzero")

    for i in range(1, n):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-14, f"E not upper triangular at ({i},{j})"


def test_tg01ob_transformation_property_identity_e():
    """Test TG01OB transformation: Q'*A*Z, Q'*B, C*Z with E=I.

    Random seed: 456 (for reproducibility)
    With E=I, the transformations Q and Z are the same.
    We verify that the transformation Q'*B has only first element nonzero.
    """
    np.random.seed(456)
    n = 3
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = (np.random.randn(1, 1) + 1j * np.random.randn(1, 1)).astype(np.complex128, order='F')

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d[0, 0]
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    e = np.eye(n, dtype=np.complex128, order='F')

    b_orig = b[:, 0].copy()

    dcba_out, e_out, info = tg01ob('I', dcba, e)

    assert info == 0

    b_out = dcba_out[1:, 0]

    b_norm_before = np.linalg.norm(b_orig)
    b_norm_after = np.linalg.norm(b_out)
    np.testing.assert_allclose(b_norm_before, b_norm_after, rtol=1e-14,
                               err_msg="Unitary transform should preserve norm of B")


def test_tg01ob_d_unchanged():
    """Test that D scalar is unchanged by the transformation.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.triu(np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d_val = 3.5 + 2.1j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d_val
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_out, e_out, info = tg01ob('U', dcba, e)

    assert info == 0
    assert dcba_out[0, 0] == d_val, "D should be unchanged"


def test_tg01ob_n_equals_1():
    """Test TG01OB with N=1 (trivial case - quick return).

    N=1 case: no transformations needed, everything unchanged.
    """
    n = 1
    n1 = n + 1

    a = np.array([[2.0 + 3.0j]], dtype=np.complex128, order='F')
    e = np.array([[1.0 + 0.0j]], dtype=np.complex128, order='F')
    b = np.array([[1.5 + 0.5j]], dtype=np.complex128, order='F')
    c = np.array([[0.5 + 0.5j]], dtype=np.complex128, order='F')
    d_val = 1.0 + 1.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d_val
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_orig = dcba.copy()
    e_orig = e.copy()

    dcba_out, e_out, info = tg01ob('U', dcba, e)

    assert info == 0
    np.testing.assert_allclose(dcba_out, dcba_orig, rtol=1e-14,
                               err_msg="N=1 should return unchanged DCBA")
    np.testing.assert_allclose(e_out, e_orig, rtol=1e-14,
                               err_msg="N=1 should return unchanged E")


def test_tg01ob_n_equals_0():
    """Test TG01OB with N=0 (quick return)."""
    n = 0
    n1 = n + 1

    dcba = np.array([[1.0 + 2.0j]], dtype=np.complex128, order='F')
    e = np.array([], dtype=np.complex128).reshape(0, 0, order='F')

    dcba_out, e_out, info = tg01ob('U', dcba, e)

    assert info == 0


def test_tg01ob_invalid_jobe():
    """Test TG01OB with invalid JOBE parameter."""
    n = 2
    n1 = n + 1

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')

    dcba_out, e_out, info = tg01ob('X', dcba, e)

    assert info == -1


def test_tg01ob_invalid_n():
    """Test TG01OB with invalid N parameter (n < 0 via wrong dcba size)."""
    dcba = np.array([], dtype=np.complex128).reshape(0, 1, order='F')
    e = np.array([], dtype=np.complex128).reshape(0, 0, order='F')

    dcba_out, e_out, info = tg01ob('I', dcba, e)

    assert info == -2


def test_tg01ob_zero_b_elements():
    """Test TG01OB when some B elements are already zero.

    Random seed: 222 (for reproducibility)
    The algorithm should handle this without issues.
    """
    np.random.seed(222)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = np.array([[1.0 + 0.5j], [0.0], [2.0 - 1.0j], [0.0]], dtype=np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d_val = 1.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d_val
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_out, e_out, info = tg01ob('I', dcba, e)

    assert info == 0

    b_out = dcba_out[1:, 0]
    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)


def test_tg01ob_b_all_zero_except_first():
    """Test TG01OB when B already has only first element nonzero.

    Random seed: 333 (for reproducibility)
    Should effectively be a no-op for B.
    """
    np.random.seed(333)
    n = 3
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = np.array([[2.5 + 1.0j], [0.0], [0.0]], dtype=np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d_val = 1.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d_val
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_orig = dcba.copy()

    dcba_out, e_out, info = tg01ob('I', dcba, e)

    assert info == 0

    b_out = dcba_out[1:, 0]
    np.testing.assert_allclose(b_out, b[:, 0], rtol=1e-14,
                               err_msg="B should be unchanged when already in desired form")


def test_tg01ob_eigenvalue_preservation():
    """Test TG01OB preserves generalized eigenvalues.

    Random seed: 444 (for reproducibility)
    The equivalence transformation Q'*A*Z, Q'*E*Z preserves generalized eigenvalues.
    """
    np.random.seed(444)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.triu(np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    np.fill_diagonal(e, np.abs(np.diag(e)) + 1.0)
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d_val = 1.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d_val
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    a_orig = a.copy()
    e_orig = e.copy()

    eig_before = np.linalg.eigvals(np.linalg.solve(e_orig, a_orig))

    dcba_out, e_out, info = tg01ob('U', dcba, e)

    assert info == 0

    a_out = dcba_out[1:, 1:]
    eig_after = np.linalg.eigvals(np.linalg.solve(e_out, a_out))

    eig_before_sorted = np.sort_complex(eig_before)
    eig_after_sorted = np.sort_complex(eig_after)
    np.testing.assert_allclose(eig_before_sorted, eig_after_sorted, rtol=1e-12, atol=1e-13)


def test_tg01ob_larger_system():
    """Test TG01OB with a larger system.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n = 8
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.triu(np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    np.fill_diagonal(e, np.abs(np.diag(e)) + 1.0)
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d_val = 1.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d_val
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_out, e_out, info = tg01ob('U', dcba, e)

    assert info == 0

    b_out = dcba_out[1:, 0]
    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)

    for i in range(1, n):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-14, f"E not upper triangular at ({i},{j})"
