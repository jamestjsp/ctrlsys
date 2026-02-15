"""
Tests for SB06ND: Minimum norm feedback matrix for deadbeat control.

SB06ND constructs the minimum norm feedback matrix F to perform
"deadbeat control" on a (A,B)-pair that has been reduced to staircase
form by AB01OD, such that R = A + BFU' is nilpotent.
"""

import numpy as np
import pytest


def test_sb06nd_html_doc_example():
    """
    Test from SLICOT HTML documentation workflow.

    Uses AB01OD to reduce (A,B) to staircase form, then SB06ND for deadbeat control.

    Input: 5x5 state matrix A, 5x2 input matrix B from HTML doc.
    Validates that result is nilpotent (all eigenvalues = 0).
    """
    from ctrlsys import ab01od, sb06nd

    n = 5
    m = 2
    tol = 0.0

    # A is column-major: ((A(I,J), I = 1,N), J = 1,N)
    a_data = np.array([
        -17.0, 23.0, 34.0, 10.0, 11.0,
        24.0, -35.0, 26.0, 12.0, 18.0,
        41.0, 27.0, -13.0, 19.0, 25.0,
        68.0, 14.0, 20.0, -21.0, 52.0,
        15.0, 16.0, 22.0, 63.0, -29.0
    ], dtype=float)
    a = a_data.reshape((n, n), order='F')

    # B is row-major in doc: ((B(I,J), J = 1,M), I = 1,N)
    b_data = np.array([
        -31.0, 14.0,
        74.0, -69.0,
        -59.0, 16.0,
        16.0, -25.0,
        -25.0, 36.0
    ], dtype=float)
    b = b_data.reshape((n, m), order='C')
    b = np.asfortranarray(b)

    # Call AB01OD: stages, jobu, jobv, a, b, tol
    result = ab01od('A', 'N', 'N', a, b, tol)
    a_stair, b_stair, u_dummy, v_dummy, ncont, kmax, kstair, info_ab = result

    assert info_ab == 0, f"AB01OD returned info={info_ab}"
    assert ncont == 5, "Expected fully controllable system"

    # Initialize U as identity (as in example when JOBU='N')
    u = np.eye(n, order='F', dtype=float)

    # Call SB06ND
    a_out, b_out, u_out, f, info = sb06nd(n, m, kmax, a_stair, b_stair, kstair[:kmax], u)

    assert info == 0, f"SB06ND returned info={info}"
    assert f.shape == (m, n)

    # Validate nilpotent property: all eigenvalues of closed-loop are zero
    eigvals = np.linalg.eigvals(a_out)
    np.testing.assert_allclose(np.abs(eigvals), 0.0, atol=1e-10,
                               err_msg="Closed-loop matrix is not nilpotent")


def test_sb06nd_nilpotent_property():
    """
    Validate nilpotent property with random controllable system.

    For deadbeat control, all eigenvalues of closed-loop must be zero.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab01od, sb06nd

    np.random.seed(42)

    # Generate a random controllable system
    n = 4
    m = 2

    # Random A and B matrices
    a = np.random.randn(n, n).astype(float)
    a = np.asfortranarray(a)
    b = np.random.randn(n, m).astype(float)
    b = np.asfortranarray(b)

    # First reduce to staircase form with AB01OD
    result = ab01od('A', 'N', 'N', a, b, 0.0)
    a_stair, b_stair, u_dummy, v_dummy, ncont, kmax, kstair, info_ab = result

    assert info_ab == 0, f"AB01OD returned info={info_ab}"

    if ncont == 0 or kmax == 0:
        pytest.skip("System not controllable enough for deadbeat control")

    # Initialize U as identity
    u = np.eye(n, order='F', dtype=float)

    # Call SB06ND
    a_out, b_out, u_out, f, info = sb06nd(n, m, kmax, a_stair.copy(), b_stair.copy(),
                                           kstair[:kmax], u)

    assert info == 0, f"SB06ND returned info={info}"

    # Check eigenvalues of A_out are all zero (nilpotent)
    eigvals = np.linalg.eigvals(a_out)
    np.testing.assert_allclose(np.abs(eigvals), 0.0, atol=1e-10,
                               err_msg="Closed-loop matrix is not nilpotent")


def test_sb06nd_simple_siso():
    """
    Test with simple SISO (single-input) system.

    Uses AB01OD to get proper staircase form.
    """
    from ctrlsys import ab01od, sb06nd

    n = 2
    m = 1

    # Simple controllable SISO system
    a = np.array([
        [0.0, 1.0],
        [-1.0, -1.0]
    ], order='F', dtype=float)

    b = np.array([
        [0.0],
        [1.0]
    ], order='F', dtype=float)

    # Reduce to staircase form
    result = ab01od('A', 'N', 'N', a, b, 0.0)
    a_stair, b_stair, u_dummy, v_dummy, ncont, kmax, kstair, info_ab = result

    assert info_ab == 0
    assert ncont == 2, "Expected fully controllable"

    u = np.eye(n, order='F', dtype=float)

    a_out, b_out, u_out, f, info = sb06nd(n, m, kmax, a_stair, b_stair, kstair[:kmax], u)

    assert info == 0
    assert f.shape == (m, n)

    # Verify nilpotent property
    eigvals = np.linalg.eigvals(a_out)
    np.testing.assert_allclose(np.abs(eigvals), 0.0, atol=1e-10)


def test_sb06nd_quick_return_n_zero():
    """Test quick return when n=0."""
    from ctrlsys import sb06nd

    n = 0
    m = 2
    kmax = 0
    kstair = np.array([], dtype=np.int32)

    a = np.array([], order='F', dtype=float).reshape(0, 0)
    b = np.array([], order='F', dtype=float).reshape(0, m)
    u = np.array([], order='F', dtype=float).reshape(0, 0)

    a_out, b_out, u_out, f, info = sb06nd(n, m, kmax, a, b, kstair, u)

    assert info == 0
    assert f.shape == (m, 0)


def test_sb06nd_quick_return_m_zero():
    """Test quick return when m=0."""
    from ctrlsys import sb06nd

    n = 3
    m = 0
    kmax = 0
    kstair = np.array([], dtype=np.int32)

    a = np.eye(n, order='F', dtype=float)
    b = np.array([], order='F', dtype=float).reshape(n, 0)
    u = np.eye(n, order='F', dtype=float)

    a_out, b_out, u_out, f, info = sb06nd(n, m, kmax, a, b, kstair, u)

    assert info == 0
    assert f.shape == (0, n)


def test_sb06nd_invalid_n():
    """Test error handling for invalid n < 0."""
    from ctrlsys import sb06nd

    n = -1
    m = 2
    kmax = 0
    kstair = np.array([], dtype=np.int32)

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0, 0.0]], order='F', dtype=float)
    u = np.array([[1.0]], order='F', dtype=float)

    with pytest.raises(ValueError, match="n must be >= 0"):
        sb06nd(n, m, kmax, a, b, kstair, u)


def test_sb06nd_invalid_m():
    """Test error handling for invalid m < 0."""
    from ctrlsys import sb06nd

    n = 2
    m = -1
    kmax = 0
    kstair = np.array([], dtype=np.int32)

    a = np.eye(n, order='F', dtype=float)
    b = np.array([], order='F', dtype=float).reshape(n, 0)
    u = np.eye(n, order='F', dtype=float)

    with pytest.raises(ValueError, match="m must be >= 0"):
        sb06nd(n, m, kmax, a, b, kstair, u)


def test_sb06nd_invalid_kmax():
    """Test error handling for invalid kmax > n."""
    from ctrlsys import sb06nd

    n = 2
    m = 1
    kmax = 5  # Invalid: kmax > n
    kstair = np.array([1, 1], dtype=np.int32)

    a = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    with pytest.raises(ValueError, match="kmax must satisfy 0 <= kmax <= n"):
        sb06nd(n, m, kmax, a, b, kstair, u)
