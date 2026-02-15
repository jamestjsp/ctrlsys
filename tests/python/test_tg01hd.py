"""
Tests for TG01HD - Orthogonal reduction of descriptor system to controllability staircase form.

This routine reduces the N-th order descriptor system (A-lambda*E,B,C)
to controllability form, separating finite and/or infinite uncontrollable eigenvalues.
"""
import pytest
import numpy as np
from ctrlsys import tg01hd


def test_tg01hd_html_example():
    """Test TG01HD with HTML documentation example.

    N=7, M=3, P=2, TOL=0.0, JOBCON='C'
    Expected: NCONT=3, NIUCON=1, RTAU=[2,1]
    """
    n, m, p = 7, 3, 2
    tol = 0.0

    a = np.array([
        [2.0, 0.0, 2.0, 0.0, -1.0, 3.0, 1.0],
        [0.0, 1.0, 0.0, 0.0,  1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0, 0.0, -1.0, 3.0, 1.0],
        [0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0,  1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [1.0, 3.0, 0.0, 2.0,  0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    ], dtype=np.float64, order='F')

    result = tg01hd('C', 'I', 'I', n, m, p, a, e, b, c, tol)
    a_out, e_out, b_out, c_out, q, z, ncont, niucon, nrblck, rtau, info = result

    assert info == 0, f"TG01HD failed with info={info}"
    assert ncont == 3, f"Expected NCONT=3, got {ncont}"
    assert niucon == 1, f"Expected NIUCON=1, got {niucon}"
    assert nrblck == 2, f"Expected NRBLCK=2, got {nrblck}"
    np.testing.assert_array_equal(rtau[:nrblck], [2, 1],
                                  err_msg="RTAU should be [2,1]")

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(n), rtol=1e-13, atol=1e-13,
                               err_msg="Q should be orthogonal")
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-13, atol=1e-13,
                               err_msg="Z should be orthogonal")

    b_uncontrollable = b_out[ncont:, :]
    np.testing.assert_allclose(b_uncontrollable, np.zeros((n - ncont, m)),
                               rtol=1e-10, atol=1e-10,
                               err_msg="Uncontrollable part of B should be zero")


def test_tg01hd_jobcon_f():
    """Test TG01HD with JOBCON='F' (separate finite uncontrollable eigenvalues only).

    For JOBCON='F', NIUCON should be 0.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 2, 2
    tol = 1e-10

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

    result = tg01hd('F', 'I', 'I', n, m, p, a, e, b, c, tol)
    a_out, e_out, b_out, c_out, q, z, ncont, niucon, nrblck, rtau, info = result

    assert info == 0, f"TG01HD failed with info={info}"
    assert niucon == 0, f"For JOBCON='F', expected NIUCON=0, got {niucon}"
    assert ncont >= 0 and ncont <= n, f"Invalid NCONT={ncont}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(n), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-13, atol=1e-13)


def test_tg01hd_jobcon_i():
    """Test TG01HD with JOBCON='I' (separate infinite uncontrollable eigenvalues only).

    For JOBCON='I', NIUCON should also be 0 (set to 0 by routine).
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 4, 2, 2
    tol = 1e-10

    a = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    result = tg01hd('I', 'I', 'I', n, m, p, a, e, b, c, tol)
    a_out, e_out, b_out, c_out, q, z, ncont, niucon, nrblck, rtau, info = result

    assert info == 0, f"TG01HD failed with info={info}"
    assert niucon == 0, f"For JOBCON='I', expected NIUCON=0, got {niucon}"
    assert ncont >= 0 and ncont <= n, f"Invalid NCONT={ncont}"


def test_tg01hd_transformation_property():
    """Test that orthogonal transformations preserve matrix relationships.

    Validates: Q'*A*Z produces block structure, Q'*B has zeros in uncontrollable part.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 5, 2, 2
    tol = 1e-10

    a_orig = np.random.randn(n, n).astype(np.float64, order='F')
    e_orig = np.random.randn(n, n).astype(np.float64, order='F')
    b_orig = np.random.randn(n, m).astype(np.float64, order='F')
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')

    a = a_orig.copy(order='F')
    e = e_orig.copy(order='F')
    b = b_orig.copy(order='F')
    c = c_orig.copy(order='F')

    result = tg01hd('C', 'I', 'I', n, m, p, a, e, b, c, tol)
    a_out, e_out, b_out, c_out, q, z, ncont, niucon, nrblck, rtau, info = result

    assert info == 0, f"TG01HD failed with info={info}"

    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(n), rtol=1e-13, atol=1e-13,
                               err_msg="Q should be orthogonal")
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-13, atol=1e-13,
                               err_msg="Z should be orthogonal")

    a_transformed = q.T @ a_orig @ z
    e_transformed = q.T @ e_orig @ z
    b_transformed = q.T @ b_orig
    c_transformed = c_orig @ z

    np.testing.assert_allclose(a_out, a_transformed, rtol=1e-12, atol=1e-12,
                               err_msg="A_out should equal Q'*A*Z")
    np.testing.assert_allclose(e_out, e_transformed, rtol=1e-12, atol=1e-12,
                               err_msg="E_out should equal Q'*E*Z")
    np.testing.assert_allclose(b_out, b_transformed, rtol=1e-12, atol=1e-12,
                               err_msg="B_out should equal Q'*B")
    np.testing.assert_allclose(c_out, c_transformed, rtol=1e-12, atol=1e-12,
                               err_msg="C_out should equal C*Z")

    if ncont < n:
        b_uncontrollable = b_out[ncont:, :]
        np.testing.assert_allclose(b_uncontrollable, np.zeros((n - ncont, m)),
                                   rtol=1e-10, atol=1e-10,
                                   err_msg="Uncontrollable part of B should be zero")


def test_tg01hd_zero_b():
    """Test TG01HD with zero B matrix (fully uncontrollable).

    With B=0, NCONT should be 0.
    """
    n, m, p = 4, 2, 2
    tol = 0.0

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')

    result = tg01hd('C', 'I', 'I', n, m, p, a, e, b, c, tol)
    a_out, e_out, b_out, c_out, q, z, ncont, niucon, nrblck, rtau, info = result

    assert info == 0, f"TG01HD failed with info={info}"
    assert ncont == 0, f"Expected NCONT=0 for zero B, got {ncont}"


def test_tg01hd_fully_controllable():
    """Test TG01HD with fully controllable system.

    For controllable (A-lambda*E, B), NCONT should equal N.
    """
    n, m, p = 3, 3, 2
    tol = 1e-10

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.eye(n, dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')

    result = tg01hd('C', 'I', 'I', n, m, p, a, e, b, c, tol)
    a_out, e_out, b_out, c_out, q, z, ncont, niucon, nrblck, rtau, info = result

    assert info == 0, f"TG01HD failed with info={info}"
    assert ncont == n, f"Expected NCONT={n} for fully controllable, got {ncont}"


def test_tg01hd_compq_compz_n():
    """Test TG01HD with COMPQ='N', COMPZ='N' (no Q, Z computation)."""
    n, m, p = 3, 2, 2
    tol = 0.0

    a = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.eye(n, dtype=np.float64, order='F')
    b = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')

    result = tg01hd('C', 'N', 'N', n, m, p, a, e, b, c, tol)
    a_out, e_out, b_out, c_out, q, z, ncont, niucon, nrblck, rtau, info = result

    assert info == 0, f"TG01HD failed with info={info}"
    assert ncont >= 0, f"Invalid NCONT={ncont}"


def test_tg01hd_invalid_jobcon():
    """Test TG01HD with invalid JOBCON parameter."""
    n, m, p = 2, 1, 1
    tol = 0.0

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')

    result = tg01hd('X', 'I', 'I', n, m, p, a, e, b, c, tol)
    *_, info = result

    assert info == -1, f"Expected info=-1 for invalid JOBCON, got {info}"


def test_tg01hd_invalid_compq():
    """Test TG01HD with invalid COMPQ parameter."""
    n, m, p = 2, 1, 1
    tol = 0.0

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')

    result = tg01hd('C', 'X', 'I', n, m, p, a, e, b, c, tol)
    *_, info = result

    assert info == -2, f"Expected info=-2 for invalid COMPQ, got {info}"


def test_tg01hd_invalid_compz():
    """Test TG01HD with invalid COMPZ parameter."""
    n, m, p = 2, 1, 1
    tol = 0.0

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')

    result = tg01hd('C', 'I', 'X', n, m, p, a, e, b, c, tol)
    *_, info = result

    assert info == -3, f"Expected info=-3 for invalid COMPZ, got {info}"


def test_tg01hd_invalid_tol():
    """Test TG01HD with invalid TOL parameter (TOL >= 1)."""
    n, m, p = 2, 1, 1
    tol = 1.0

    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')

    result = tg01hd('C', 'I', 'I', n, m, p, a, e, b, c, tol)
    *_, info = result

    assert info == -23, f"Expected info=-23 for invalid TOL, got {info}"


def test_tg01hd_quick_return_n_zero():
    """Test TG01HD quick return with N=0."""
    n, m, p = 0, 2, 2
    tol = 0.0

    a = np.zeros((0, 0), dtype=np.float64, order='F')
    e = np.zeros((0, 0), dtype=np.float64, order='F')
    b = np.zeros((0, m), dtype=np.float64, order='F')
    c = np.zeros((p, 0), dtype=np.float64, order='F')

    result = tg01hd('C', 'I', 'I', n, m, p, a, e, b, c, tol)
    a_out, e_out, b_out, c_out, q, z, ncont, niucon, nrblck, rtau, info = result

    assert info == 0, f"TG01HD failed with info={info}"
    assert ncont == 0, f"Expected NCONT=0 for N=0, got {ncont}"


def test_tg01hd_rtau_sum_equals_ncont():
    """Test that sum of RTAU equals NCONT.

    Mathematical property: Sum of staircase block dimensions = controllable subspace dimension.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 6, 2, 2
    tol = 1e-10

    a = np.array([
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.eye(n, dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ], dtype=np.float64, order='F')

    c = np.ones((p, n), dtype=np.float64, order='F')

    result = tg01hd('C', 'I', 'I', n, m, p, a, e, b, c, tol)
    a_out, e_out, b_out, c_out, q, z, ncont, niucon, nrblck, rtau, info = result

    assert info == 0, f"TG01HD failed with info={info}"

    if nrblck > 0:
        rtau_sum = sum(rtau[:nrblck])
        assert rtau_sum == ncont, f"Sum of RTAU ({rtau_sum}) should equal NCONT ({ncont})"
