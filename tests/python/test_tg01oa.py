"""
Tests for TG01OA: Orthogonal equivalence transformation of SISO descriptor system.

TG01OA computes for a single-input single-output descriptor system,
given by the system matrix:

    [ D     C    ]
    [ B  A - s*E ]

with E upper triangular, a transformed system (Q'*A*Z, Q'*E*Z, Q'*B, C*Z),
via orthogonal equivalence transformation, so that Q'*B has only the first
element nonzero and Q'*E*Z remains upper triangular.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tg01oa_identity_e():
    """
    Test TG01OA with JOBE='I' (E is identity, not given).

    Random seed: 42 (for reproducibility)

    Mathematical property tested:
    - Q'*B should have only first element nonzero
    - The transformation preserves eigenvalues of A
    """
    from ctrlsys import tg01oa

    np.random.seed(42)
    n = 3

    d = 0.5
    c = np.array([1.0, 2.0, 3.0], dtype=float)
    b = np.array([1.0, 2.0, 3.0], dtype=float)
    a = np.random.randn(n, n).astype(float, order='F')

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    dcba_orig = dcba.copy(order='F')
    a_orig = a.copy(order='F')

    e = np.zeros((1, 1), dtype=float, order='F')

    dcba_out, e_out, info = tg01oa('I', dcba, e)

    assert info == 0

    assert dcba_out[0, 0] == pytest.approx(d)

    b_transformed = dcba_out[1:, 0]
    assert_allclose(b_transformed[1:], 0.0, atol=1e-14)
    assert abs(b_transformed[0]) > 1e-10

    b_norm_orig = np.linalg.norm(dcba_orig[1:, 0])
    b_norm_trans = np.linalg.norm(b_transformed)
    assert_allclose(b_norm_trans, b_norm_orig, rtol=1e-14)

    eig_orig = np.sort(np.linalg.eigvals(a_orig))
    a_trans = dcba_out[1:, 1:]
    eig_trans = np.sort(np.linalg.eigvals(a_trans))
    assert_allclose(eig_trans, eig_orig, rtol=1e-12)


def test_tg01oa_upper_triangular_e():
    """
    Test TG01OA with JOBE='U' (E is upper triangular).

    Random seed: 123 (for reproducibility)

    Mathematical properties tested:
    - Q'*B has only first element nonzero
    - Q'*E*Z remains upper triangular
    - Generalized eigenvalues preserved
    """
    from ctrlsys import tg01oa

    np.random.seed(123)
    n = 4

    d = 1.5
    c = np.array([0.5, 1.0, -0.5, 0.3], dtype=float)
    b = np.array([1.0, 0.5, -1.0, 0.2], dtype=float)
    a = np.random.randn(n, n).astype(float, order='F')
    e = np.triu(np.random.randn(n, n)).astype(float, order='F')
    np.fill_diagonal(e, np.abs(np.diag(e)) + 1.0)

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    dcba_orig = dcba.copy(order='F')
    a_orig = a.copy(order='F')
    e_orig = e.copy(order='F')

    dcba_out, e_out, info = tg01oa('U', dcba, e)

    assert info == 0

    b_transformed = dcba_out[1:, 0]
    assert_allclose(b_transformed[1:], 0.0, atol=1e-14)

    for i in range(1, n):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-14, f"E[{i},{j}] = {e_out[i,j]} should be zero"

    gen_eig_orig = np.linalg.eigvals(np.linalg.solve(e_orig, a_orig))
    gen_eig_trans = np.linalg.eigvals(np.linalg.solve(e_out, dcba_out[1:, 1:]))
    assert_allclose(np.sort(gen_eig_orig.real), np.sort(gen_eig_trans.real), rtol=1e-10)


def test_tg01oa_n_zero():
    """Test TG01OA with N=0 (quick return case)."""
    from ctrlsys import tg01oa

    n = 0
    dcba = np.array([[1.0]], dtype=float, order='F')
    e = np.zeros((1, 1), dtype=float, order='F')

    dcba_out, e_out, info = tg01oa('I', dcba, e)

    assert info == 0
    assert dcba_out[0, 0] == 1.0


def test_tg01oa_n_one():
    """
    Test TG01OA with N=1 (quick return case, B already has only one element).
    """
    from ctrlsys import tg01oa

    n = 1
    dcba = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=float, order='F')
    e = np.array([[2.0]], dtype=float, order='F')

    dcba_orig = dcba.copy(order='F')
    e_orig = e.copy(order='F')

    dcba_out, e_out, info = tg01oa('U', dcba, e)

    assert info == 0
    assert_allclose(dcba_out, dcba_orig, rtol=1e-14)
    assert_allclose(e_out, e_orig, rtol=1e-14)


def test_tg01oa_invalid_jobe():
    """Test TG01OA with invalid JOBE parameter."""
    from ctrlsys import tg01oa

    n = 2
    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    e = np.eye(n, dtype=float, order='F')

    with pytest.raises(ValueError):
        tg01oa('X', dcba, e)


def test_tg01oa_b_norm_preservation():
    """
    Test that Q'*B preserves the norm of B.

    Random seed: 456 (for reproducibility)

    Since Q is orthogonal: ||Q'*B|| = ||B||
    """
    from ctrlsys import tg01oa

    np.random.seed(456)
    n = 5

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = 1.0
    dcba[0, 1:] = np.random.randn(n)
    dcba[1:, 0] = np.random.randn(n)
    dcba[1:, 1:] = np.random.randn(n, n)

    e = np.triu(np.random.randn(n, n)).astype(float, order='F')
    np.fill_diagonal(e, np.abs(np.diag(e)) + 0.5)

    b_orig_norm = np.linalg.norm(dcba[1:, 0])

    dcba_out, e_out, info = tg01oa('U', dcba, e)

    assert info == 0

    b_trans_norm = np.linalg.norm(dcba_out[1:, 0])
    assert_allclose(b_trans_norm, b_orig_norm, rtol=1e-14)

    assert abs(dcba_out[1, 0]) > 1e-10
    assert_allclose(dcba_out[2:, 0], 0.0, atol=1e-14)


def test_tg01oa_zero_b():
    """
    Test TG01OA when B is already zero.

    The routine should not modify anything since no rotations needed.
    """
    from ctrlsys import tg01oa

    np.random.seed(789)
    n = 3

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = 1.0
    dcba[0, 1:] = [1.0, 2.0, 3.0]
    dcba[1:, 0] = 0.0
    dcba[1:, 1:] = np.random.randn(n, n)

    e = np.triu(np.eye(n, dtype=float, order='F'))

    dcba_orig = dcba.copy(order='F')

    dcba_out, e_out, info = tg01oa('U', dcba, e)

    assert info == 0
    assert_allclose(dcba_out[1:, 0], 0.0, atol=1e-14)
