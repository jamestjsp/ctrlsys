"""Tests for TG01FZ - Unitary reduction of complex descriptor system to SVD-like form."""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from ctrlsys import tg01fz


def test_tg01fz_basic_example():
    """Test TG01FZ with example from SLICOT HTML documentation.

    RANKE=3, RNKA22=1 expected.
    """
    l, n, m, p = 4, 4, 2, 2
    tol = 0.0

    # Input data from HTML doc (row-wise)
    a = np.array([
        [-1+0j,  0+0j,  0+0j,  3+0j],
        [ 0+0j,  0+0j,  1+0j,  2+0j],
        [ 1+0j,  1+0j,  0+0j,  4+0j],
        [ 0+0j,  0+0j,  0+0j,  0+0j]
    ], dtype=np.complex128, order='F')

    e = np.array([
        [1+0j, 2+0j, 0+0j, 0+0j],
        [0+0j, 1+0j, 0+0j, 1+0j],
        [3+0j, 9+0j, 6+0j, 3+0j],
        [0+0j, 0+0j, 2+0j, 0+0j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [1+0j, 0+0j],
        [0+0j, 0+0j],
        [0+0j, 1+0j],
        [1+0j, 1+0j]
    ], dtype=np.complex128, order='F')

    c = np.array([
        [-1+0j, 0+0j, 1+0j, 0+0j],
        [ 0+0j, 1+0j,-1+0j, 1+0j]
    ], dtype=np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'I', 'R', l, n, m, p, a, e, b, c, tol
    )

    assert info == 0
    assert ranke == 3
    assert rnka22 == 1


def test_tg01fz_unitary_property():
    """Test that Q and Z are unitary matrices.

    Q' * Q = I and Z' * Z = I
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    l, n, m, p = 5, 5, 2, 2

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'I', 'R', l, n, m, p, a, e, b, c, 0.0
    )

    assert info == 0

    # Q should be unitary
    q_h_q = np.conj(q.T) @ q
    assert_allclose(q_h_q, np.eye(l), rtol=1e-13, atol=1e-14)

    # Z should be unitary
    z_h_z = np.conj(z.T) @ z
    assert_allclose(z_h_z, np.eye(n), rtol=1e-13, atol=1e-14)


def test_tg01fz_transformation_property():
    """Test that Q'*A*Z, Q'*E*Z, Q'*B, C*Z transformations are correct.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    l, n, m, p = 4, 4, 2, 3

    a_orig = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e_orig = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b_orig = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c_orig = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a = a_orig.copy()
    e = e_orig.copy()
    b = b_orig.copy()
    c = c_orig.copy()

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'I', 'R', l, n, m, p, a, e, b, c, 0.0
    )

    assert info == 0

    # Verify Q'*A*Z = a_out
    qhaz = np.conj(q.T) @ a_orig @ z
    assert_allclose(qhaz, a_out, rtol=1e-13, atol=1e-14)

    # Verify Q'*E*Z = e_out
    qhez = np.conj(q.T) @ e_orig @ z
    assert_allclose(qhez, e_out, rtol=1e-13, atol=1e-14)

    # Verify Q'*B = b_out
    qhb = np.conj(q.T) @ b_orig
    assert_allclose(qhb, b_out, rtol=1e-13, atol=1e-14)

    # Verify C*Z = c_out
    cz = c_orig @ z
    assert_allclose(cz, c_out, rtol=1e-13, atol=1e-14)


def test_tg01fz_e_structure():
    """Test that Q'*E*Z has proper upper triangular structure.

    The transformed E should have the form:
        ( Er  0 )
        (  0  0 )
    where Er is RANKE-by-RANKE upper triangular.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    l, n, m, p = 5, 5, 2, 2

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'I', 'R', l, n, m, p, a, e, b, c, 0.0
    )

    assert info == 0

    # Check that Er (top-left RANKE x RANKE) is upper triangular
    Er = e_out[:ranke, :ranke]
    for i in range(1, ranke):
        for j in range(i):
            assert abs(Er[i, j]) < 1e-13

    # Check that the right part (columns ranke:n) is zero
    if ranke < n:
        assert_allclose(e_out[:, ranke:], 0.0, atol=1e-13)

    # Check that the bottom part (rows ranke:l) is zero
    if ranke < l:
        assert_allclose(e_out[ranke:, :], 0.0, atol=1e-13)


def test_tg01fz_zero_dimensions():
    """Test with zero dimensions (quick return case)."""
    # L=0 case
    a = np.array([], dtype=np.complex128).reshape(0, 4)
    e = np.array([], dtype=np.complex128).reshape(0, 4)
    b = np.array([], dtype=np.complex128).reshape(0, 2)
    c = np.zeros((2, 4), dtype=np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'I', 'R', 0, 4, 2, 2, a, e, b, c, 0.0
    )

    assert info == 0
    assert ranke == 0
    assert rnka22 == 0


def test_tg01fz_invalid_compq():
    """Test with invalid COMPQ parameter."""
    l, n, m, p = 4, 4, 2, 2
    a = np.zeros((l, n), dtype=np.complex128, order='F')
    e = np.zeros((l, n), dtype=np.complex128, order='F')
    b = np.zeros((l, m), dtype=np.complex128, order='F')
    c = np.zeros((p, n), dtype=np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'X', 'I', 'R', l, n, m, p, a, e, b, c, 0.0
    )

    assert info == -1


def test_tg01fz_invalid_compz():
    """Test with invalid COMPZ parameter."""
    l, n, m, p = 4, 4, 2, 2
    a = np.zeros((l, n), dtype=np.complex128, order='F')
    e = np.zeros((l, n), dtype=np.complex128, order='F')
    b = np.zeros((l, m), dtype=np.complex128, order='F')
    c = np.zeros((p, n), dtype=np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'X', 'R', l, n, m, p, a, e, b, c, 0.0
    )

    assert info == -2


def test_tg01fz_invalid_joba():
    """Test with invalid JOBA parameter."""
    l, n, m, p = 4, 4, 2, 2
    a = np.zeros((l, n), dtype=np.complex128, order='F')
    e = np.zeros((l, n), dtype=np.complex128, order='F')
    b = np.zeros((l, m), dtype=np.complex128, order='F')
    c = np.zeros((p, n), dtype=np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'I', 'X', l, n, m, p, a, e, b, c, 0.0
    )

    assert info == -3


def test_tg01fz_joba_n():
    """Test with JOBA='N' (no A22 reduction).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    l, n, m, p = 4, 4, 2, 2

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'I', 'N', l, n, m, p, a, e, b, c, 0.0
    )

    assert info == 0


def test_tg01fz_rectangular_tall():
    """Test with tall rectangular matrices (L > N).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    l, n, m, p = 6, 4, 2, 2

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'I', 'R', l, n, m, p, a, e, b, c, 0.0
    )

    assert info == 0
    assert ranke <= min(l, n)

    # Q should be L x L unitary
    q_h_q = np.conj(q.T) @ q
    assert_allclose(q_h_q, np.eye(l), rtol=1e-13, atol=1e-14)

    # Z should be N x N unitary
    z_h_z = np.conj(z.T) @ z
    assert_allclose(z_h_z, np.eye(n), rtol=1e-13, atol=1e-14)


def test_tg01fz_rectangular_wide():
    """Test with wide rectangular matrices (L < N).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    l, n, m, p = 4, 6, 2, 2

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fz(
        'I', 'I', 'R', l, n, m, p, a, e, b, c, 0.0
    )

    assert info == 0
    assert ranke <= min(l, n)

    # Q should be L x L unitary
    q_h_q = np.conj(q.T) @ q
    assert_allclose(q_h_q, np.eye(l), rtol=1e-13, atol=1e-14)

    # Z should be N x N unitary
    z_h_z = np.conj(z.T) @ z
    assert_allclose(z_h_z, np.eye(n), rtol=1e-13, atol=1e-14)
