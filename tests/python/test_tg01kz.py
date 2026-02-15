"""Tests for TG01KZ - Complex SISO descriptor system equivalence transformation.

TG01KZ computes for a single-input single-output descriptor system (A, E, B, C)
with E upper triangular, a transformed system (Q'*A*Z, Q'*E*Z, Q'*B, C*Z) via
unitary equivalence transformation, so that Q'*B has only the first element
nonzero and Q'*E*Z remains upper triangular.

Uses Givens rotations to annihilate last N-1 elements of B in reverse order
while preserving the upper triangular form of E.
"""
import pytest
import numpy as np
from ctrlsys import tg01kz


def test_tg01kz_basic_identity_e():
    """Test TG01KZ with JOBE='I' (E is identity).

    Random seed: 42 (for reproducibility)
    Tests basic functionality with identity E matrix.
    """
    np.random.seed(42)
    n = 4

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')
    c = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')

    a_orig = a.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'C', 'I', 'I', a, e, b, c, incc=1
    )

    assert info == 0, f"TG01KZ failed with info={info}"

    assert b_out.shape == (n,)
    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)
    assert abs(b_out[0]) > 0

    assert q_out.shape == (n, n)
    assert z_out.shape == (n, n)

    q_unitary = q_out.conj().T @ q_out
    np.testing.assert_allclose(q_unitary, np.eye(n), rtol=1e-14, atol=1e-14)

    z_unitary = z_out.conj().T @ z_out
    np.testing.assert_allclose(z_unitary, np.eye(n), rtol=1e-14, atol=1e-14)

    a_expected = q_out.conj().T @ a_orig @ z_out
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-13, atol=1e-14)

    b_expected = q_out.conj().T @ b_orig
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-13, atol=1e-14)

    c_expected = c_orig @ z_out
    np.testing.assert_allclose(c_out, c_expected, rtol=1e-13, atol=1e-14)


def test_tg01kz_upper_triangular_e():
    """Test TG01KZ with JOBE='U' (E upper triangular).

    Random seed: 123 (for reproducibility)
    Tests with general upper triangular E, verifies E stays upper triangular.
    """
    np.random.seed(123)
    n = 4

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.triu(np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')
    c = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'U', 'C', 'I', 'I', a, e, b, c, incc=1
    )

    assert info == 0, f"TG01KZ failed with info={info}"

    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)

    for i in range(1, n):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-14, f"E not upper triangular at ({i},{j})"

    a_expected = q_out.conj().T @ a_orig @ z_out
    e_expected = q_out.conj().T @ e_orig @ z_out
    b_expected = q_out.conj().T @ b_orig
    c_expected = c_orig @ z_out

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(e_out, e_expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(c_out, c_expected, rtol=1e-13, atol=1e-14)


def test_tg01kz_no_c_transform():
    """Test TG01KZ with COMPC='N' (do not transform C).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')
    c = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    b_orig = b.copy()

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'N', 'I', 'I', a, e, b, c, incc=1
    )

    assert info == 0

    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)

    a_expected = q_out.conj().T @ a_orig @ z_out
    b_expected = q_out.conj().T @ b_orig

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-13, atol=1e-14)


def test_tg01kz_no_q_accumulation():
    """Test TG01KZ with COMPQ='N' (do not accumulate Q).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')
    c = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'C', 'N', 'I', a, e, b, c, incc=1
    )

    assert info == 0

    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)

    z_unitary = z_out.conj().T @ z_out
    np.testing.assert_allclose(z_unitary, np.eye(n), rtol=1e-14, atol=1e-14)


def test_tg01kz_update_q():
    """Test TG01KZ with COMPQ='U' (update given Q).

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n = 3

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')
    c = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')

    q_init, _ = np.linalg.qr(np.random.randn(n, n) + 1j * np.random.randn(n, n))
    q_init = q_init.astype(np.complex128, order='F')
    q_init_copy = q_init.copy()

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'C', 'U', 'I', a, e, b, c, incc=1, q=q_init
    )

    assert info == 0

    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)

    q_updated_unitary = q_out.conj().T @ q_out
    np.testing.assert_allclose(q_updated_unitary, np.eye(n), rtol=1e-14, atol=1e-14)


def test_tg01kz_update_z():
    """Test TG01KZ with COMPZ='U' (update given Z).

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 3

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')
    c = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')

    z_init, _ = np.linalg.qr(np.random.randn(n, n) + 1j * np.random.randn(n, n))
    z_init = z_init.astype(np.complex128, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'C', 'I', 'U', a, e, b, c, incc=1, z=z_init
    )

    assert info == 0

    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)

    z_updated_unitary = z_out.conj().T @ z_out
    np.testing.assert_allclose(z_updated_unitary, np.eye(n), rtol=1e-14, atol=1e-14)


def test_tg01kz_n_equals_1():
    """Test TG01KZ with N=1 (trivial case).

    N=1 case: no transformations needed, Q and Z should be identity.
    """
    n = 1

    a = np.array([[2.0 + 3.0j]], dtype=np.complex128, order='F')
    e = np.array([[1.0 + 0.0j]], dtype=np.complex128, order='F')
    b = np.array([1.5 + 0.5j], dtype=np.complex128, order='F')
    c = np.array([0.5 + 0.5j], dtype=np.complex128, order='F')

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'U', 'C', 'I', 'I', a, e, b, c, incc=1
    )

    assert info == 0

    np.testing.assert_allclose(q_out, np.eye(1, dtype=np.complex128), rtol=1e-14)
    np.testing.assert_allclose(z_out, np.eye(1, dtype=np.complex128), rtol=1e-14)


def test_tg01kz_n_equals_0():
    """Test TG01KZ with N=0 (quick return)."""
    n = 0

    a = np.array([], dtype=np.complex128).reshape(0, 0, order='F')
    e = np.array([], dtype=np.complex128).reshape(0, 0, order='F')
    b = np.array([], dtype=np.complex128)
    c = np.array([], dtype=np.complex128)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'U', 'C', 'I', 'I', a, e, b, c, incc=1
    )

    assert info == 0


def test_tg01kz_c_with_stride():
    """Test TG01KZ with C having a non-unit increment (INCC > 1).

    Random seed: 111 (for reproducibility)
    Tests that strided C array is handled correctly.
    """
    np.random.seed(111)
    n = 3
    incc = 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')

    c_full = (np.random.randn((n - 1) * incc + 1) + 1j * np.random.randn((n - 1) * incc + 1)).astype(np.complex128)
    c_values_orig = c_full[::incc].copy()

    a_orig = a.copy()
    b_orig = b.copy()

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'C', 'I', 'I', a, e, b, c_full, incc=incc
    )

    assert info == 0

    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)

    c_expected = c_values_orig @ z_out
    c_transformed = c_out[::incc]
    np.testing.assert_allclose(c_transformed, c_expected, rtol=1e-13, atol=1e-14)


def test_tg01kz_invalid_jobe():
    """Test TG01KZ with invalid JOBE parameter."""
    n = 2

    a = np.eye(n, dtype=np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = np.ones(n, dtype=np.complex128)
    c = np.ones(n, dtype=np.complex128)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'X', 'C', 'I', 'I', a, e, b, c, incc=1
    )

    assert info == -1


def test_tg01kz_invalid_compc():
    """Test TG01KZ with invalid COMPC parameter."""
    n = 2

    a = np.eye(n, dtype=np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = np.ones(n, dtype=np.complex128)
    c = np.ones(n, dtype=np.complex128)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'X', 'I', 'I', a, e, b, c, incc=1
    )

    assert info == -2


def test_tg01kz_invalid_compq():
    """Test TG01KZ with invalid COMPQ parameter."""
    n = 2

    a = np.eye(n, dtype=np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = np.ones(n, dtype=np.complex128)
    c = np.ones(n, dtype=np.complex128)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'C', 'X', 'I', a, e, b, c, incc=1
    )

    assert info == -3


def test_tg01kz_invalid_compz():
    """Test TG01KZ with invalid COMPZ parameter."""
    n = 2

    a = np.eye(n, dtype=np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = np.ones(n, dtype=np.complex128)
    c = np.ones(n, dtype=np.complex128)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'C', 'I', 'X', a, e, b, c, incc=1
    )

    assert info == -4


def test_tg01kz_invalid_incc():
    """Test TG01KZ with invalid INCC (INCC <= 0 when COMPC='C')."""
    n = 2

    a = np.eye(n, dtype=np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = np.ones(n, dtype=np.complex128)
    c = np.ones(n, dtype=np.complex128)

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'C', 'I', 'I', a, e, b, c, incc=0
    )

    assert info == -12


def test_tg01kz_zero_b_elements():
    """Test TG01KZ when some B elements are already zero.

    The algorithm should handle this without issues.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 4

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')
    b = np.array([1.0 + 0.5j, 0.0, 2.0 - 1.0j, 0.0], dtype=np.complex128)
    c = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    b_orig = b.copy()

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'I', 'C', 'I', 'I', a, e, b, c, incc=1
    )

    assert info == 0

    np.testing.assert_allclose(b_out[1:], 0, atol=1e-14)

    b_expected = q_out.conj().T @ b_orig
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-13, atol=1e-14)


def test_tg01kz_eigenvalue_preservation():
    """Test TG01KZ preserves generalized eigenvalues.

    Random seed: 333 (for reproducibility)
    The equivalence transformation Q'*A*Z, Q'*E*Z preserves generalized eigenvalues.
    """
    np.random.seed(333)
    n = 4

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = np.triu(np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    np.fill_diagonal(e, np.abs(np.diag(e)) + 1.0)
    b = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')
    c = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    e_orig = e.copy()

    eig_before = np.linalg.eigvals(np.linalg.solve(e_orig, a_orig))

    a_out, e_out, b_out, c_out, q_out, z_out, info = tg01kz(
        'U', 'C', 'I', 'I', a, e, b, c, incc=1
    )

    assert info == 0

    eig_after = np.linalg.eigvals(np.linalg.solve(e_out, a_out))

    eig_before_sorted = np.sort(eig_before)
    eig_after_sorted = np.sort(eig_after)
    np.testing.assert_allclose(eig_before_sorted, eig_after_sorted, rtol=1e-12, atol=1e-13)
