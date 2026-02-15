"""
Tests for AB08MZ: Normal rank of transfer-function matrix (complex case).

AB08MZ computes the normal rank of the transfer-function matrix of a
complex state-space model (A,B,C,D) by reducing the compound matrix [B A; D C]
to one with the same invariant zeros and with D of full row rank.
"""

import numpy as np
import pytest
from ctrlsys import ab08mz


"""Basic functionality tests for AB08MZ."""

def test_full_rank_siso_system():
    """
    Test SISO system with full rank transfer function.

    System: x' = -x + u, y = x + u (complex version)
    Transfer function: G(s) = (s+2)/(s+1), which has rank 1.

    Random seed: N/A (deterministic)
    """
    n, m, p = 1, 1, 1

    a = np.array([[-1.0+0j]], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j]], order='F', dtype=np.complex128)
    d = np.array([[1.0+0j]], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == 1

def test_rank_two_mimo_system():
    """
    Test 2-input, 2-output system with full rank (rank 2).

    The D matrix is identity, which has full rank.

    Random seed: N/A (deterministic)
    """
    n, m, p = 2, 2, 2

    a = np.array([[-1.0+0j, 0.0+0j],
                  [0.0+0j, -2.0+0j]], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j, 0.0+0j],
                  [0.0+0j, 1.0+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j, 0.0+0j],
                  [0.0+0j, 1.0+0j]], order='F', dtype=np.complex128)
    d = np.array([[1.0+0j, 0.0+0j],
                  [0.0+0j, 1.0+0j]], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == 2

def test_rank_deficient_system():
    """
    Test system where outputs are linearly dependent.

    Two outputs are identical, so normal rank is 1 even with 2 outputs.

    Random seed: N/A (deterministic)
    """
    n, m, p = 1, 1, 2

    a = np.array([[-1.0+0j]], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j], [1.0+0j]], order='F', dtype=np.complex128)
    d = np.array([[1.0+0j], [1.0+0j]], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == 1

def test_zero_d_matrix():
    """
    Test system with D=0 (strictly proper).

    System: x' = -x + u, y = x (D=0)
    Transfer function: G(s) = 1/(s+1), rank 1.

    Random seed: N/A (deterministic)
    """
    n, m, p = 1, 1, 1

    a = np.array([[-1.0+0j]], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j]], order='F', dtype=np.complex128)
    d = np.array([[0.0+0j]], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == 1


"""Tests for AB08MZ with scaling option."""

def test_with_scaling():
    """
    Test with scaling enabled (EQUIL='S').

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F') * 100
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
        np.complex128, order='F') * 0.01
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(
        np.complex128, order='F') * 100
    d = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(
        np.complex128, order='F')

    rank_no_scale, info1 = ab08mz('N', n, m, p, a.copy(), b.copy(),
                                   c.copy(), d.copy())
    rank_scale, info2 = ab08mz('S', n, m, p, a.copy(), b.copy(),
                                c.copy(), d.copy())

    assert info1 == 0
    assert info2 == 0
    assert rank_no_scale == rank_scale


"""Edge case tests for AB08MZ."""

def test_zero_n():
    """
    Test with n=0 (no state variables, static system).

    For static system, rank = rank(D).

    Random seed: N/A (deterministic)
    """
    n, m, p = 0, 2, 2

    a = np.zeros((0, 0), dtype=np.complex128, order='F')
    b = np.zeros((0, 2), dtype=np.complex128, order='F')
    c = np.zeros((2, 0), dtype=np.complex128, order='F')
    d = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == 2

def test_zero_m():
    """
    Test with m=0 (no inputs).

    When there are no inputs, normal rank should be 0.

    Random seed: N/A (deterministic)
    """
    n, m, p = 2, 0, 2

    a = np.array([[-1.0+0j, 0.0+0j], [0.0+0j, -2.0+0j]], order='F', dtype=np.complex128)
    b = np.zeros((2, 0), dtype=np.complex128, order='F')
    c = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], order='F', dtype=np.complex128)
    d = np.zeros((2, 0), dtype=np.complex128, order='F')

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == 0

def test_zero_p():
    """
    Test with p=0 (no outputs).

    When there are no outputs, normal rank should be 0.

    Random seed: N/A (deterministic)
    """
    n, m, p = 2, 2, 0

    a = np.array([[-1.0+0j, 0.0+0j], [0.0+0j, -2.0+0j]], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], order='F', dtype=np.complex128)
    c = np.zeros((0, 2), dtype=np.complex128, order='F')
    d = np.zeros((0, 2), dtype=np.complex128, order='F')

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == 0


"""Tests with complex-valued matrices."""

def test_complex_eigenvalues():
    """
    Test system with complex eigenvalues.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 2, 1, 1

    a = np.array([[0.0+1j, 1.0+0j],
                  [-1.0+0j, 0.0+1j]], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j], [0.0+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j, 0.0+0j]], order='F', dtype=np.complex128)
    d = np.array([[0.0+0j]], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == 1

def test_purely_imaginary_system():
    """
    Test system with purely imaginary elements.

    Random seed: 456 (for reproducibility)
    """
    n, m, p = 2, 2, 2

    a = np.array([[0.0+1j, 0.0+0j],
                  [0.0+0j, 0.0-1j]], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j, 0.0+0j],
                  [0.0+0j, 1.0+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j, 0.0+0j],
                  [0.0+0j, 1.0+0j]], order='F', dtype=np.complex128)
    d = np.array([[1.0+0j, 0.0+0j],
                  [0.0+0j, 1.0+0j]], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == 2

def test_general_complex_system():
    """
    Test with general complex-valued system matrices.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 3, 2, 2

    a = np.array([
        [1.0+1j, 2.0-1j, 0.0+0j],
        [0.0+0j, -1.0+2j, 1.0+0j],
        [1.0+0j, 0.0+0j, -2.0-1j],
    ], order='F', dtype=np.complex128)
    b = np.array([
        [1.0+0.5j, 0.0+0j],
        [0.0+0j, 1.0-0.5j],
        [0.5+0j, 0.5+0j],
    ], order='F', dtype=np.complex128)
    c = np.array([
        [1.0+0j, 0.0+1j, 0.0+0j],
        [0.0+0j, 1.0+0j, 0.0-1j],
    ], order='F', dtype=np.complex128)
    d = np.array([
        [0.1+0.1j, 0.0+0j],
        [0.0+0j, 0.1-0.1j],
    ], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == min(m, p)


"""Tests with larger state-space systems."""

def test_random_full_rank_system():
    """
    Test random full-rank system.

    Create a system where D has full row rank, so normal rank = min(m, p).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 4, 3, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F')
    a = a - 2.0 * np.eye(n)
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
        np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(
        np.complex128, order='F')
    d = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(
        np.complex128, order='F')

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == min(m, p)

def test_higher_order_system():
    """
    Test with higher order system (n=6).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n, m, p = 6, 2, 3

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F')
    a = a - 3.0 * np.eye(n)
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
        np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(
        np.complex128, order='F')
    d = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(
        np.complex128, order='F')

    rank, info = ab08mz('N', n, m, p, a.copy(order='F'), b.copy(order='F'),
                        c.copy(order='F'), d.copy(order='F'))

    assert info == 0
    assert 0 <= rank <= min(m, p)

    max_numerical_rank = 0
    for omega in [0.01, 0.1, 1.0, 10.0, 100.0]:
        G = c @ np.linalg.solve(1j * omega * np.eye(n) - a, b) + d
        sv = np.linalg.svd(G, compute_uv=False)
        max_numerical_rank = max(max_numerical_rank, int(np.sum(sv > 1e-10)))
    assert rank == max_numerical_rank


"""Tests for tolerance parameter."""

def test_custom_tolerance():
    """
    Test with custom tolerance value.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F')
    a = a - 2.0 * np.eye(n)
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
        np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(
        np.complex128, order='F')
    d = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(
        np.complex128, order='F')

    rank1, info1 = ab08mz('N', n, m, p, a.copy(), b.copy(),
                           c.copy(), d.copy(), tol=1e-10)
    rank2, info2 = ab08mz('N', n, m, p, a.copy(), b.copy(),
                           c.copy(), d.copy(), tol=1e-3)

    assert info1 == 0
    assert info2 == 0
    assert rank1 == min(m, p)
    assert rank2 == min(m, p)


"""Tests validating mathematical properties of the normal rank."""

def test_rank_bounded_by_min_m_p():
    """
    Validate: normal rank <= min(m, p).

    The normal rank of G(s) = C(sI-A)^{-1}B + D cannot exceed
    min(m, p), the minimum of inputs and outputs.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)

    for n, m, p in [(3, 2, 4), (5, 4, 2), (4, 3, 3)]:
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
            np.complex128, order='F')
        a = a - 2.0 * np.eye(n)
        b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
            np.complex128, order='F')
        c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(
            np.complex128, order='F')
        d = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(
            np.complex128, order='F')

        rank, info = ab08mz('N', n, m, p, a, b, c, d)

        assert info == 0
        assert rank <= min(m, p), f"Rank {rank} > min(m,p)={min(m,p)}"

def test_rank_at_least_rank_d():
    """
    Validate: normal rank >= rank(D).

    The normal rank is at least the rank of D since it's the
    leading term in the Laurent expansion at infinity.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    n, m, p = 3, 2, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F')
    a = a - 2.0 * np.eye(n)
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
        np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(
        np.complex128, order='F')
    d = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(
        np.complex128, order='F')

    rank, info = ab08mz('N', n, m, p, a, b, c, d)
    d_rank = np.linalg.matrix_rank(d)

    assert info == 0
    assert rank >= d_rank

def test_full_rank_d_gives_full_normal_rank():
    """
    Validate: if D has full row rank, normal rank = min(m, p).

    When D is full row rank, the transfer function is guaranteed
    to have normal rank equal to min(m, p).

    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    n, m, p = 4, 3, 2

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F')
    a = a - 2.0 * np.eye(n)
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
        np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(
        np.complex128, order='F')

    u, s, vh = np.linalg.svd(np.random.randn(p, m) + 1j * np.random.randn(p, m))
    s_full = np.zeros((p, m), dtype=np.complex128)
    for i in range(min(p, m)):
        s_full[i, i] = s[i] + 1.0
    d = np.asfortranarray((u @ s_full @ vh).astype(np.complex128))

    rank, info = ab08mz('N', n, m, p, a, b, c, d)

    assert info == 0
    assert rank == min(m, p)


def test_rank_via_complex_transfer_function_svd():
    """
    Verify normal rank by evaluating G(jw) = C*(jwI-A)^{-1}*B + D
    at random frequencies for complex system.
    """
    n, m, p = 3, 2, 2

    a = np.array([[-1.0+0.5j, 0.5+0j, 0.0],
                  [0.0, -2.0+0.3j, 0.3+0j],
                  [0.0, 0.0, -3.0-0.2j]], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j, 0.0],
                  [0.0, 1.0+0.1j],
                  [0.5+0j, 0.5+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j, 0.0, 1.0-0.1j],
                  [0.0, 1.0+0j, 0.0]], order='F', dtype=np.complex128)
    d = np.array([[0.1+0.1j, 0.0],
                  [0.0, 0.2-0.1j]], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a.copy(order='F'), b.copy(order='F'),
                        c.copy(order='F'), d.copy(order='F'))
    assert info == 0

    max_numerical_rank = 0
    for omega in [0.1, 1.0, 10.0, 100.0]:
        sI_A = 1j * omega * np.eye(n) - a
        G = c @ np.linalg.solve(sI_A, b) + d
        sv = np.linalg.svd(G, compute_uv=False)
        numerical_rank = np.sum(sv > 1e-10)
        max_numerical_rank = max(max_numerical_rank, numerical_rank)

    assert rank == max_numerical_rank


def test_complex_rank_deficient_via_svd():
    """
    Verify rank-deficient complex system via SVD of G(jw).
    """
    n, m, p = 1, 1, 2
    a = np.array([[-1.0+0.5j]], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j], [1.0+0j]], order='F', dtype=np.complex128)
    d = np.array([[0.0+0j], [0.0+0j]], order='F', dtype=np.complex128)

    rank, info = ab08mz('N', n, m, p, a.copy(order='F'), b.copy(order='F'),
                        c.copy(order='F'), d.copy(order='F'))
    assert info == 0
    assert rank == 1

    for omega in [0.5, 5.0, 50.0]:
        G = c @ np.linalg.solve(1j * omega * np.eye(n) - a, b) + d
        sv = np.linalg.svd(G, compute_uv=False)
        assert np.sum(sv > 1e-10) == 1


"""Error handling tests."""

def test_invalid_equil():
    """Test invalid EQUIL parameter."""
    n, m, p = 2, 1, 1
    a = np.eye(n, dtype=np.complex128, order='F')
    b = np.ones((n, m), dtype=np.complex128, order='F')
    c = np.ones((p, n), dtype=np.complex128, order='F')
    d = np.zeros((p, m), dtype=np.complex128, order='F')

    with pytest.raises(ValueError, match="equil must be 'S' or 'N'"):
        ab08mz('X', n, m, p, a, b, c, d)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
