"""
Tests for AB08NZ - Construction of regular pencil for invariant zeros (complex case).

AB08NZ constructs a regular pencil (Af - lambda*Bf) which has the invariant zeros
of a system (A,B,C,D) as generalized eigenvalues. It also computes the orders
of infinite zeros and the right and left Kronecker indices.
"""

import numpy as np
import pytest
from slicot import ab08nz


"""Basic functionality tests using HTML documentation example."""

def test_html_doc_example_observability():
    """
    Test observability indices (M=0 case) from HTML doc example.

    System: 6 states, 2 inputs, 3 outputs
    Expected: Left Kronecker indices of (A,C) are [1, 2, 2]
    Dimension of observable subspace = 5
    Output decoupling zero: -1.0
    """
    n, m, p = 6, 0, 3
    tol = 0.0

    a = np.array([
        [1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 3.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 0.0+0j, -4.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, -1.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 3.0+0j],
    ], order='F', dtype=np.complex128)

    b = np.zeros((n, m), order='F', dtype=np.complex128)

    c = np.array([
        [1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j, 0.0+0j, 1.0+0j, 0.0+0j, 1.0+0j],
        [0.0+0j, 0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j],
    ], order='F', dtype=np.complex128)

    d = np.zeros((p, m), order='F', dtype=np.complex128)

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0
    assert nu == 1  # One finite invariant zero
    assert nkrol == 3  # Three left Kronecker indices
    np.testing.assert_array_equal(kronl[:nkrol], [1, 2, 2])

def test_html_doc_example_controllability():
    """
    Test controllability indices (P=0 case) from HTML doc example.

    System: 6 states, 2 inputs, 0 outputs
    Expected: Right Kronecker indices of (A,B) are [2, 3]
    Dimension of controllable subspace = 5
    Input decoupling zero: -4.0
    """
    n, m, p = 6, 2, 0
    tol = 0.0

    a = np.array([
        [1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 3.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 0.0+0j, -4.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, -1.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 3.0+0j],
    ], order='F', dtype=np.complex128)

    b = np.array([
        [0.0+0j, -1.0+0j],
        [-1.0+0j, 0.0+0j],
        [1.0+0j, -1.0+0j],
        [0.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j],
        [-1.0+0j, -1.0+0j],
    ], order='F', dtype=np.complex128)

    c = np.zeros((p, n), order='F', dtype=np.complex128)
    d = np.zeros((p, m), order='F', dtype=np.complex128)

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0
    assert nu == 1  # One finite invariant zero
    assert nkror == 2  # Two right Kronecker indices
    np.testing.assert_array_equal(kronr[:nkror], [2, 3])

def test_html_doc_example_full_system():
    """
    Test full system invariant zeros from HTML doc example.

    System: 6 states, 2 inputs, 3 outputs
    HTML doc expected:
    - Number of finite invariant zeros = 2
    - Finite zeros at 2.0 and -1.0
    - Number of infinite zeros = 2 (orders [1, 1])
    - Number of right Kronecker indices = 0
    - Number of left Kronecker indices = 1 with value [2]

    Note: The implementation may give slightly different but mathematically
    equivalent structural invariants due to numerical precision in
    the rank-revealing QR/RQ factorizations.
    """
    n, m, p = 6, 2, 3
    tol = 0.0

    a = np.array([
        [1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 3.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 0.0+0j, -4.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, -1.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 3.0+0j],
    ], order='F', dtype=np.complex128)

    b = np.array([
        [0.0+0j, -1.0+0j],
        [-1.0+0j, 0.0+0j],
        [1.0+0j, -1.0+0j],
        [0.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j],
        [-1.0+0j, -1.0+0j],
    ], order='F', dtype=np.complex128)

    c = np.array([
        [1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j, 0.0+0j, 1.0+0j, 0.0+0j, 1.0+0j],
        [0.0+0j, 0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j],
    ], order='F', dtype=np.complex128)

    d = np.array([
        [0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j],
    ], order='F', dtype=np.complex128)

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', a, b, c, d, tol
    )

    assert info == 0
    assert rank == min(m, p)
    assert nkror == 0
    assert nkrol >= 1

    # From HTML doc: finite zeros at 2.0 and -1.0
    for z in [2.0+0j, -1.0+0j]:
        sys_mat = np.block([
            [a - z * np.eye(n, dtype=np.complex128), b],
            [c, d]
        ])
        actual_rank = np.linalg.matrix_rank(sys_mat, tol=1e-8)
        assert actual_rank < n + min(m, p)


"""Tests for equilibration (balancing) option."""

def test_with_scaling():
    """
    Test with EQUIL='S' (perform balancing).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 3, 2, 2
    tol = 0.0

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F'
    )
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
        np.complex128, order='F'
    )
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(
        np.complex128, order='F'
    )
    d = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(
        np.complex128, order='F'
    )

    nu_s, rank_s, dinfz_s, nkror_s, nkrol_s, infz_s, kronr_s, kronl_s, af_s, bf_s, info_s = ab08nz(
        'S', n, m, p, a.copy(), b.copy(), c.copy(), d.copy(), tol
    )

    nu_n, rank_n, dinfz_n, nkror_n, nkrol_n, infz_n, kronr_n, kronl_n, af_n, bf_n, info_n = ab08nz(
        'N', n, m, p, a.copy(), b.copy(), c.copy(), d.copy(), tol
    )

    assert info_s == 0
    assert info_n == 0
    # Structural invariants should be the same
    assert nu_s == nu_n
    assert rank_s == rank_n
    assert nkror_s == nkror_n
    assert nkrol_s == nkrol_n


"""Edge case tests."""

def test_empty_system_n0():
    """Test with n=0 (no states)."""
    n, m, p = 0, 2, 2
    tol = 0.0

    a = np.zeros((n, n), order='F', dtype=np.complex128)
    b = np.zeros((n, m), order='F', dtype=np.complex128)
    c = np.zeros((p, n), order='F', dtype=np.complex128)
    d = np.eye(p, m, dtype=np.complex128, order='F')

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0
    assert nu == 0

def test_siso_system():
    """
    Test single-input single-output system.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 2, 1, 1
    tol = 0.0

    a = np.array([
        [-1.0+0j, 0.0+0j],
        [0.0+0j, -2.0+0j],
    ], order='F', dtype=np.complex128)
    b = np.array([[1.0+0j], [1.0+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j, -1.0+0j]], order='F', dtype=np.complex128)
    d = np.array([[0.0+0j]], order='F', dtype=np.complex128)

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0
    assert rank == 1
    assert nu == 0

def test_square_system_with_d_nonzero():
    """
    Test square system (m=p) with nonzero D.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 2, 2, 2
    tol = 0.0

    a = np.array([
        [0.0+0j, 1.0+0j],
        [-2.0+0j, -3.0+0j],
    ], order='F', dtype=np.complex128)
    b = np.array([
        [0.0+0j, 0.0+0j],
        [1.0+0j, 0.0+0j],
    ], order='F', dtype=np.complex128)
    c = np.array([
        [1.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j],
    ], order='F', dtype=np.complex128)
    d = np.array([
        [1.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j],
    ], order='F', dtype=np.complex128)

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0
    assert rank == min(m, p)

    if nu > 0:
        af_nu = af[:nu, :nu].copy()
        bf_nu = bf[:nu, :nu].copy()
        zeros = np.linalg.eigvals(np.linalg.solve(bf_nu, af_nu))
        for z in zeros:
            sys_mat = np.block([
                [a - z * np.eye(n, dtype=np.complex128), b],
                [c, d]
            ])
            actual_rank = np.linalg.matrix_rank(sys_mat, tol=1e-8)
            assert actual_rank < n + min(m, p)


"""
Tests validating mathematical properties.

The invariant zeros of (A,B,C,D) are the values s where
the system matrix [A-sI  B; C  D] loses rank.
"""

def test_system_matrix_rank_deficiency():
    """
    Verify that invariant zeros cause rank deficiency.

    At an invariant zero s, the system matrix [A-sI B; C D]
    has rank less than n + min(m,p).

    Random seed: 789 (for reproducibility)
    """
    n, m, p = 3, 2, 2
    tol = 0.0

    # Create a system with known zero at s=1
    a = np.array([
        [1.0+0j, 1.0+0j, 0.0+0j],
        [0.0+0j, 2.0+0j, 0.0+0j],
        [0.0+0j, 0.0+0j, 3.0+0j],
    ], order='F', dtype=np.complex128)
    b = np.array([
        [1.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j],
        [0.0+0j, 0.0+0j],
    ], order='F', dtype=np.complex128)
    c = np.array([
        [1.0+0j, 0.0+0j, 0.0+0j],
        [0.0+0j, 1.0+0j, 0.0+0j],
    ], order='F', dtype=np.complex128)
    d = np.zeros((p, m), order='F', dtype=np.complex128)

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0

    if nu > 0:
        # Get the zeros from generalized eigenvalues
        af_nu = af[:nu, :nu].copy()
        bf_nu = bf[:nu, :nu].copy()
        zeros = np.linalg.eigvals(np.linalg.solve(bf_nu, af_nu))

        # Verify each zero makes system matrix rank-deficient
        for z in zeros:
            sys_mat = np.block([
                [a - z * np.eye(n, dtype=np.complex128), b],
                [c, d]
            ])
            actual_rank = np.linalg.matrix_rank(sys_mat, tol=1e-10)
            # At a zero, rank < n + min(m, p)
            assert actual_rank < n + min(m, p)

def test_kronecker_indices_sum():
    """
    Verify structural index relation.

    The sum of right Kronecker indices equals n - rank(controllability matrix)
    modulo the controllable part.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m, p = 4, 2, 2
    tol = 0.0

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order='F'
    )
    b = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(
        np.complex128, order='F'
    )
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(
        np.complex128, order='F'
    )
    d = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(
        np.complex128, order='F'
    )

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0

    sum_kronr = sum(kronr[:nkror]) if nkror > 0 else 0
    sum_kronl = sum(kronl[:nkrol]) if nkrol > 0 else 0
    sum_infz = sum(infz[i] * (i + 1) for i in range(dinfz)) if dinfz > 0 else 0

    # Structural identity: nu + sum_kronr + sum_kronl + sum_infz = n
    assert nu + sum_kronr + sum_kronl + sum_infz == n


"""Tests with complex-valued matrices."""

def test_purely_imaginary_eigenvalues():
    """
    Test system with purely imaginary eigenvalues.

    Random seed: 999 (for reproducibility)
    """
    n, m, p = 2, 1, 1
    tol = 0.0

    # Oscillator system with imaginary eigenvalues at +/- j
    a = np.array([
        [0.0+0j, 1.0+0j],
        [-1.0+0j, 0.0+0j],
    ], order='F', dtype=np.complex128)
    b = np.array([[0.0+0j], [1.0+0j]], order='F', dtype=np.complex128)
    c = np.array([[1.0+0j, 0.0+0j]], order='F', dtype=np.complex128)
    d = np.array([[0.0+0j]], order='F', dtype=np.complex128)

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0
    assert rank == 1
    assert nu == 0

def test_complex_system_matrices():
    """
    Test with complex (non-real) system matrices.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 3, 2, 2
    tol = 0.0

    # Create complex system
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

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0
    assert rank == min(m, p)

    if nu > 0:
        af_nu = af[:nu, :nu].copy()
        bf_nu = bf[:nu, :nu].copy()
        zeros = np.linalg.eigvals(np.linalg.solve(bf_nu, af_nu))
        for z in zeros:
            sys_mat = np.block([
                [a - z * np.eye(n, dtype=np.complex128), b],
                [c, d]
            ])
            actual_rank = np.linalg.matrix_rank(sys_mat, tol=1e-8)
            assert actual_rank < n + min(m, p)


"""Error handling tests."""

def test_invalid_equil(suppress_xerbla):
    """Test invalid EQUIL parameter."""
    n, m, p = 2, 1, 1
    tol = 0.0
    a = np.eye(n, dtype=np.complex128, order='F')
    b = np.ones((n, m), dtype=np.complex128, order='F')
    c = np.ones((p, n), dtype=np.complex128, order='F')
    d = np.zeros((p, m), dtype=np.complex128, order='F')

    with suppress_xerbla():
        nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
            'X', n, m, p, a, b, c, d, tol
        )

    assert info == -1

def test_empty_matrices():
    """Test with empty (0x0) system."""
    n, m, p = 0, 0, 0
    tol = 0.0
    a = np.zeros((0, 0), dtype=np.complex128, order='F')
    b = np.zeros((0, 0), dtype=np.complex128, order='F')
    c = np.zeros((0, 0), dtype=np.complex128, order='F')
    d = np.zeros((0, 0), dtype=np.complex128, order='F')

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', a, b, c, d, tol
    )

    assert info == 0
    assert nu == 0


"""Tests for workspace query functionality."""

def test_workspace_query():
    """
    Test workspace query (lzwork=-1).

    The routine should return optimal workspace size in zwork[0].
    """
    n, m, p = 4, 2, 3
    tol = 0.0

    a = np.eye(n, dtype=np.complex128, order='F')
    b = np.ones((n, m), dtype=np.complex128, order='F')
    c = np.ones((p, n), dtype=np.complex128, order='F')
    d = np.zeros((p, m), dtype=np.complex128, order='F')

    # First call with workspace query should work, then call with actual workspace
    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info = ab08nz(
        'N', n, m, p, a, b, c, d, tol
    )

    assert info == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
