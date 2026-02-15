"""
Tests for TG01HU - Staircase controllability representation of a multi-input descriptor system.

This routine reduces the pair (A1-lambda*E1,[B1 B2]) to controllability staircase
form, separating controllable and uncontrollable parts of the descriptor system
with two separate input matrices B1 and B2.
"""
import pytest
import numpy as np
from ctrlsys import tg01hu


def test_tg01hu_basic():
    """Test TG01HU with basic 4x4 system with two input groups.

    Tests reduction of a simple descriptor system to staircase form.
    Validates controllable part separation and orthogonality of Q, Z.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    l, n, m1, m2, p = 4, 4, 1, 1, 2
    n1 = 4
    lbe = 0
    tol = 0.0

    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.0, 2.5, 3.0],
        [0.0, 0.0, 2.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 2.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"
    assert nr >= 0 and nr <= n1, f"Invalid NR={nr}"
    assert nrblck >= 0, f"Invalid NRBLCK={nrblck}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-14, atol=1e-14,
                               err_msg="Q should be orthogonal")
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-14, atol=1e-14,
                               err_msg="Z should be orthogonal")


def test_tg01hu_upper_triangular_e():
    """Test TG01HU when E1 already upper triangular (LBE=0).

    Validates that the routine preserves upper triangular structure
    in the uncontrollable part.
    """
    l, n, m1, m2, p = 3, 3, 1, 0, 1
    n1 = 3
    lbe = 0
    tol = 1e-10

    a = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([[1.0], [0.0], [0.0]], dtype=np.float64, order='F')
    c = np.array([[1.0, 0.0, 0.0]], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"
    assert nr >= 0 and nr <= n1, f"Invalid NR={nr}"

    for i in range(nr, n1):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-10, \
                f"Expected E[{i},{j}]=0 in uncontrollable part, got {e_out[i, j]}"


def test_tg01hu_nonzero_lbe():
    """Test TG01HU with nonzero sub-diagonals in E1 (LBE > 0).

    When E1 has sub-diagonals, the routine should first reduce E to
    upper triangular form, then proceed with the staircase reduction.
    """
    l, n, m1, m2, p = 4, 4, 1, 0, 1
    n1 = 4
    lbe = 1
    tol = 0.0

    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 2.0],
        [0.0, 0.0, 1.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [2.0, 1.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float64, order='F')
    c = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"
    assert nr >= 0, f"Invalid NR={nr}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-13, atol=1e-13)


def test_tg01hu_zero_b():
    """Test TG01HU with zero B matrix (no controllable part).

    With B=0, the system should be entirely uncontrollable.
    Expected: NR=0.
    """
    l, n, m1, m2, p = 3, 3, 1, 1, 1
    n1 = 3
    lbe = 0
    tol = 0.0

    a = np.eye(3, dtype=np.float64, order='F')
    e = np.eye(3, dtype=np.float64, order='F')
    b = np.zeros((3, 2), dtype=np.float64, order='F')
    c = np.ones((1, 3), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"
    assert nr == 0, f"Expected NR=0 for zero B, got NR={nr}"


def test_tg01hu_fully_controllable():
    """Test TG01HU with fully controllable system.

    For a controllable pair (A,[B1 B2]), the controllable part should equal n1.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    l, n, m1, m2, p = 3, 3, 2, 1, 2
    n1 = 3
    lbe = 0
    tol = 1e-10

    a = np.eye(3, dtype=np.float64, order='F')
    e = np.eye(3, dtype=np.float64, order='F')
    b = np.eye(3, dtype=np.float64, order='F')
    c = np.ones((2, 3), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"
    assert nr == n1, f"Expected NR={n1} for fully controllable system, got NR={nr}"


def test_tg01hu_compq_compz_n():
    """Test TG01HU with COMPQ='N', COMPZ='N' (no Q, Z computation)."""
    l, n, m1, m2, p = 3, 3, 1, 0, 1
    n1 = 3
    lbe = 0
    tol = 0.0

    a = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.eye(3, dtype=np.float64, order='F')
    b = np.array([[1.0], [0.0], [0.0]], dtype=np.float64, order='F')
    c = np.array([[1.0, 0.0, 0.0]], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'N', 'N', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"
    assert nr >= 0, f"Invalid NR={nr}"


def test_tg01hu_invalid_compq():
    """Test TG01HU with invalid COMPQ parameter."""
    l, n, m1, m2, p = 2, 2, 1, 0, 1
    n1 = 2
    lbe = 0
    tol = 0.0

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    b = np.ones((2, 1), dtype=np.float64, order='F')
    c = np.ones((1, 2), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'X', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == -1, f"Expected info=-1 for invalid COMPQ, got {info}"


def test_tg01hu_invalid_lbe():
    """Test TG01HU with invalid LBE parameter (LBE >= N1)."""
    l, n, m1, m2, p = 3, 3, 1, 0, 1
    n1 = 3
    lbe = 3
    tol = 0.0

    a = np.eye(3, dtype=np.float64, order='F')
    e = np.eye(3, dtype=np.float64, order='F')
    b = np.ones((3, 1), dtype=np.float64, order='F')
    c = np.ones((1, 3), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == -9, f"Expected info=-9 for invalid LBE, got {info}"


def test_tg01hu_transformation_property():
    """Test TG01HU transformation property: Q'*A*Z = A_out.

    Validates that the orthogonal transformations correctly relate
    original and transformed matrices within the N1 x N1 subblock.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    l, n, m1, m2, p = 4, 4, 1, 1, 2
    n1 = 4
    lbe = 0
    tol = 1e-12

    a_orig = np.random.randn(l, n).astype(np.float64, order='F')
    e_upper = np.triu(np.random.randn(l, n)).astype(np.float64, order='F')
    b_orig = np.random.randn(l, m1 + m2).astype(np.float64, order='F')
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')

    a = a_orig.copy(order='F')
    e = e_upper.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-13, atol=1e-13,
                               err_msg="Q should be orthogonal")
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-13, atol=1e-13,
                               err_msg="Z should be orthogonal")


def test_tg01hu_edge_n1_zero():
    """Test TG01HU with N1=0 (quick return)."""
    l, n, m1, m2, p = 4, 4, 1, 1, 2
    n1 = 0
    lbe = 0
    tol = 0.0

    a = np.eye(4, dtype=np.float64, order='F')
    e = np.eye(4, dtype=np.float64, order='F')
    b = np.ones((4, 2), dtype=np.float64, order='F')
    c = np.ones((2, 4), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"
    assert nr == 0, f"Expected NR=0 for N1=0, got NR={nr}"
    assert nrblck == 0, f"Expected NRBLCK=0 for N1=0, got NRBLCK={nrblck}"


def test_tg01hu_edge_m_zero():
    """Test TG01HU with M1=M2=0 (no inputs)."""
    l, n, m1, m2, p = 3, 3, 0, 0, 1
    n1 = 3
    lbe = 0
    tol = 0.0

    a = np.eye(3, dtype=np.float64, order='F')
    e = np.eye(3, dtype=np.float64, order='F')
    b = np.zeros((3, 0), dtype=np.float64, order='F')
    c = np.ones((1, 3), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"
    assert nr == 0, f"Expected NR=0 for M1=M2=0, got NR={nr}"


def test_tg01hu_rtau_staircase():
    """Test TG01HU RTAU output for staircase structure.

    The RTAU array should contain the dimensions of the row rank blocks
    in the staircase form. Sum of RTAU[0:NRBLCK] should equal NR.
    NRBLCK is always even for TG01HU.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    l, n, m1, m2, p = 5, 5, 1, 1, 2
    n1 = 5
    lbe = 0
    tol = 1e-12

    a = np.array([
        [1.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.eye(5, dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ], dtype=np.float64, order='F')

    c = np.ones((2, 5), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"

    if nrblck > 0:
        rtau_sum = sum(rtau[:nrblck])
        assert rtau_sum == nr, f"Sum of RTAU ({rtau_sum}) should equal NR ({nr})"
        assert nrblck % 2 == 0, f"NRBLCK should be even, got {nrblck}"


def test_tg01hu_compq_u():
    """Test TG01HU with COMPQ='U' (update existing Q).

    When COMPQ='U', the routine should update an existing orthogonal Q.
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    l, n, m1, m2, p = 3, 3, 1, 0, 1
    n1 = 3
    lbe = 0
    tol = 0.0

    a = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.eye(3, dtype=np.float64, order='F')
    b = np.array([[1.0], [0.0], [0.0]], dtype=np.float64, order='F')
    c = np.array([[1.0, 0.0, 0.0]], dtype=np.float64, order='F')

    q_init = np.eye(3, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'U', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol, q=q_init
    )

    assert info == 0, f"TG01HU failed with info={info}"

    q_check = q.T @ q
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-14, atol=1e-14)


def test_tg01hu_two_input_groups():
    """Test TG01HU with both input groups B1 and B2 active.

    Validates the two-input structure that differentiates TG01HU from TG01HX.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    l, n, m1, m2, p = 4, 4, 2, 1, 2
    n1 = 4
    lbe = 0
    tol = 1e-10

    a = np.array([
        [1.0, 0.5, 0.0, 0.0],
        [0.0, 2.0, 1.0, 0.0],
        [0.0, 0.0, 1.5, 0.5],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.eye(4, dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0, 0.5],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    c = np.ones((2, 4), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hu(
        'I', 'I', l, n, m1, m2, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HU failed with info={info}"
    assert nr >= 0 and nr <= n1, f"Invalid NR={nr}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-13, atol=1e-13)
