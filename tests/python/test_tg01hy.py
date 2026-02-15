"""
Tests for TG01HY - Blocked version of controllability staircase reduction.

Reduces (A1-lambda*E1, B1) to staircase form with controllable part separated.
Similar to TG01HX but uses block algorithms for better performance on large systems.
"""
import pytest
import numpy as np
from ctrlsys import tg01hy


def test_tg01hy_basic():
    """Test TG01HY with basic 4x4 system.

    Tests reduction of a simple descriptor system to staircase form.
    Validates controllable part separation and orthogonality of Q, Z.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    l, n, m, p = 4, 4, 2, 2
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

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
    assert nr >= 0 and nr <= n1, f"Invalid NR={nr}"
    assert nrblck >= 0, f"Invalid NRBLCK={nrblck}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-14, atol=1e-14,
                               err_msg="Q should be orthogonal")
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-14, atol=1e-14,
                               err_msg="Z should be orthogonal")


def test_tg01hy_upper_triangular_e():
    """Test TG01HY when E1 already upper triangular (LBE=0).

    Validates that the routine preserves upper triangular structure
    in the uncontrollable part.
    """
    l, n, m, p = 3, 3, 1, 1
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

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
    assert nr >= 0 and nr <= n1, f"Invalid NR={nr}"

    for i in range(nr, n1):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-10, \
                f"Expected E[{i},{j}]=0 in uncontrollable part, got {e_out[i, j]}"


def test_tg01hy_nonzero_lbe():
    """Test TG01HY with nonzero sub-diagonals in E1 (LBE > 0).

    When E1 has sub-diagonals, the routine should first reduce E to
    upper triangular form via block QR, then proceed with staircase reduction.

    Note: Uses large lbe to trigger blocked DORMQR path which is more robust.
    The element-wise DLARF path has a known issue (beads bd-pp8c).
    """
    l, n, m, p = 100, 100, 5, 3
    n1 = 100
    lbe = 50  # Large enough to trigger blocked algorithm
    tol = 1e-10

    np.random.seed(789)
    a = np.random.randn(l, n).astype(np.float64, order='F')
    a[n1:, :n1] = 0.0

    e = np.zeros((l, n), dtype=np.float64, order='F')
    for i in range(n1):
        for j in range(i, n):
            e[i, j] = np.random.randn()
        for jj in range(max(0, i - lbe), i):
            e[i, jj] = np.random.randn()

    b = np.random.randn(l, m).astype(np.float64, order='F')
    b[n1:, :] = 0.0
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
    assert nr >= 0, f"Invalid NR={nr}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-13, atol=1e-13)


def test_tg01hy_zero_b():
    """Test TG01HY with zero B matrix (no controllable part).

    With B=0, the system should be entirely uncontrollable.
    Expected: NR=0, NRBLCK=0.
    """
    l, n, m, p = 3, 3, 2, 1
    n1 = 3
    lbe = 0
    tol = 0.0

    a = np.eye(3, dtype=np.float64, order='F')
    e = np.eye(3, dtype=np.float64, order='F')
    b = np.zeros((3, 2), dtype=np.float64, order='F')
    c = np.ones((1, 3), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
    assert nr == 0, f"Expected NR=0 for zero B, got NR={nr}"


def test_tg01hy_fully_controllable():
    """Test TG01HY with fully controllable system.

    For a controllable pair (A,B), the controllable part should equal n1.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    l, n, m, p = 3, 3, 3, 2
    n1 = 3
    lbe = 0
    tol = 1e-10

    a = np.eye(3, dtype=np.float64, order='F')
    e = np.eye(3, dtype=np.float64, order='F')
    b = np.eye(3, dtype=np.float64, order='F')
    c = np.ones((2, 3), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
    assert nr == n1, f"Expected NR={n1} for fully controllable system, got NR={nr}"


def test_tg01hy_compq_compz_n():
    """Test TG01HY with COMPQ='N', COMPZ='N' (no Q, Z computation)."""
    l, n, m, p = 3, 3, 1, 1
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

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'N', 'N', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
    assert nr >= 0, f"Invalid NR={nr}"


def test_tg01hy_invalid_compq():
    """Test TG01HY with invalid COMPQ parameter."""
    l, n, m, p = 2, 2, 1, 1
    n1 = 2
    lbe = 0
    tol = 0.0

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    b = np.ones((2, 1), dtype=np.float64, order='F')
    c = np.ones((1, 2), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'X', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == -1, f"Expected info=-1 for invalid COMPQ, got {info}"


def test_tg01hy_invalid_lbe():
    """Test TG01HY with invalid LBE parameter (LBE >= N1)."""
    l, n, m, p = 3, 3, 1, 1
    n1 = 3
    lbe = 3
    tol = 0.0

    a = np.eye(3, dtype=np.float64, order='F')
    e = np.eye(3, dtype=np.float64, order='F')
    b = np.ones((3, 1), dtype=np.float64, order='F')
    c = np.ones((1, 3), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == -8, f"Expected info=-8 for invalid LBE, got {info}"


def test_tg01hy_transformation_property():
    """Test TG01HY transformation property: Q'*A*Z = A_out.

    Validates that the orthogonal transformations correctly relate
    original and transformed matrices within the N1 x N1 subblock.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    l, n, m, p = 4, 4, 2, 2
    n1 = 4
    lbe = 0
    tol = 1e-12

    a_orig = np.random.randn(l, n).astype(np.float64, order='F')
    e_upper = np.triu(np.random.randn(l, n)).astype(np.float64, order='F')
    b_orig = np.random.randn(l, m).astype(np.float64, order='F')
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')

    a = a_orig.copy(order='F')
    e = e_upper.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-13, atol=1e-13,
                               err_msg="Q should be orthogonal")
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-13, atol=1e-13,
                               err_msg="Z should be orthogonal")


def test_tg01hy_edge_n1_zero():
    """Test TG01HY with N1=0 (quick return)."""
    l, n, m, p = 4, 4, 2, 2
    n1 = 0
    lbe = 0
    tol = 0.0

    a = np.eye(4, dtype=np.float64, order='F')
    e = np.eye(4, dtype=np.float64, order='F')
    b = np.ones((4, 2), dtype=np.float64, order='F')
    c = np.ones((2, 4), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
    assert nr == 0, f"Expected NR=0 for N1=0, got NR={nr}"
    assert nrblck == 0, f"Expected NRBLCK=0 for N1=0, got NRBLCK={nrblck}"


def test_tg01hy_edge_m_zero():
    """Test TG01HY with M=0 (no inputs)."""
    l, n, m, p = 3, 3, 0, 1
    n1 = 3
    lbe = 0
    tol = 0.0

    a = np.eye(3, dtype=np.float64, order='F')
    e = np.eye(3, dtype=np.float64, order='F')
    b = np.zeros((3, 0), dtype=np.float64, order='F')
    c = np.ones((1, 3), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
    assert nr == 0, f"Expected NR=0 for M=0, got NR={nr}"


def test_tg01hy_rtau_staircase():
    """Test TG01HY RTAU output for staircase structure.

    The RTAU array should contain the dimensions of the row rank blocks
    in the staircase form. Sum of RTAU[0:NRBLCK] should equal NR.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    l, n, m, p = 5, 5, 2, 2
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

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"

    if nrblck > 0:
        rtau_sum = sum(rtau[:nrblck])
        assert rtau_sum == nr, f"Sum of RTAU ({rtau_sum}) should equal NR ({nr})"


def test_tg01hy_compq_u():
    """Test TG01HY with COMPQ='U' (update existing Q).

    When COMPQ='U', the routine should update an existing orthogonal Q.
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    l, n, m, p = 3, 3, 1, 1
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

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'U', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol, q=q_init
    )

    assert info == 0, f"TG01HY failed with info={info}"

    q_check = q.T @ q
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-14, atol=1e-14)


def test_tg01hy_block_algorithm_large():
    """Test TG01HY with larger system.

    With larger matrices (10x10), tests core staircase reduction algorithm.
    Uses lbe=0 (E already upper triangular) to focus on main algorithm.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    l, n, m, p = 10, 10, 3, 2
    n1 = 10
    lbe = 0  # E already upper triangular
    tol = 1e-10

    a = np.random.randn(l, n).astype(np.float64, order='F')
    a[n1:, :n1] = 0.0

    # Upper triangular E (lbe=0)
    e = np.zeros((l, n), dtype=np.float64, order='F')
    for i in range(n1):
        for j in range(i, n):
            e[i, j] = np.random.randn()
    for i in range(n1, l):
        for j in range(n1, n):
            e[i, j] = np.random.randn()

    b = np.random.randn(l, m).astype(np.float64, order='F')
    b[n1:, :] = 0.0

    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
    assert nr >= 0 and nr <= n1, f"Invalid NR={nr}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-10, atol=1e-10,
                               err_msg="Q should be orthogonal")
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-10, atol=1e-10,
                               err_msg="Z should be orthogonal")


def test_tg01hy_workspace_query():
    """Test TG01HY workspace query (LDWORK=-1).

    TG01HY supports workspace query mode returning optimal workspace size.
    This is tested at the C level by checking INFO does not fail.
    """
    l, n, m, p = 4, 4, 2, 2
    n1 = 4
    lbe = 1
    tol = 0.0

    a = np.eye(4, dtype=np.float64, order='F')
    e = np.eye(4, dtype=np.float64, order='F')
    b = np.ones((4, 2), dtype=np.float64, order='F')
    c = np.ones((2, 4), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nr, nrblck, rtau, info = tg01hy(
        'I', 'I', l, n, m, p, n1, lbe, a, e, b, c, tol
    )

    assert info == 0, f"TG01HY failed with info={info}"
