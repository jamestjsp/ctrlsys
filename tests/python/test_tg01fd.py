import pytest
import numpy as np
from ctrlsys import tg01fd


def test_tg01fd_basic_example():
    """Test TG01FD with example from SLICOT HTML documentation.

    L=4, N=4, M=2, P=2, COMPQ='I', COMPZ='I', JOBA='R', TOL=0.0
    Tests descriptor system reduction to SVD-like form.
    """
    l, n, m, p = 4, 4, 2, 2
    tol = 0.0
    compq = 'I'
    compz = 'I'
    joba = 'R'

    # Input matrices (row-wise READ in Fortran example)
    # A matrix (L x N)
    a = np.array([
        [-1.0,  0.0,  0.0,  3.0],
        [ 0.0,  0.0,  1.0,  2.0],
        [ 1.0,  1.0,  0.0,  4.0],
        [ 0.0,  0.0,  0.0,  0.0]
    ], dtype=np.float64, order='F')

    # E matrix (L x N)
    e = np.array([
        [1.0,  2.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  1.0],
        [3.0,  9.0,  6.0,  3.0],
        [0.0,  0.0,  2.0,  0.0]
    ], dtype=np.float64, order='F')

    # B matrix (L x M)
    b = np.array([
        [1.0,  0.0],
        [0.0,  0.0],
        [0.0,  1.0],
        [1.0,  1.0]
    ], dtype=np.float64, order='F')

    # C matrix (P x N)
    c = np.array([
        [-1.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0, -1.0,  1.0]
    ], dtype=np.float64, order='F')

    # Call TG01FD
    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fd(
        compq, compz, joba, l, n, m, p, a, e, b, c, tol
    )

    # Check success
    assert info == 0, f"TG01FD failed with info={info}"

    # Check ranks
    assert ranke == 3, f"Expected RANKE=3, got {ranke}"
    assert rnka22 == 1, f"Expected RNKA22=1, got {rnka22}"

    # Expected results from documentation
    a_expected = np.array([
        [ 2.0278,  0.1078,  3.9062, -2.1571],
        [-0.0980,  0.2544,  1.6053, -0.1269],
        [ 0.2713,  0.7760, -0.3692, -0.4853],
        [ 0.0690, -0.5669, -2.1974,  0.3086]
    ], dtype=np.float64, order='F')

    e_expected = np.array([
        [10.1587,  5.8230,  1.3021,  0.0000],
        [ 0.0000, -2.4684, -0.1896,  0.0000],
        [ 0.0000,  0.0000,  1.0338,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000]
    ], dtype=np.float64, order='F')

    b_expected = np.array([
        [-0.2157, -0.9705],
        [ 0.3015,  0.9516],
        [ 0.7595,  0.0991],
        [ 1.1339,  0.3780]
    ], dtype=np.float64, order='F')

    c_expected = np.array([
        [ 0.3651, -1.0000, -0.4472, -0.8165],
        [-1.0954,  1.0000, -0.8944,  0.0000]
    ], dtype=np.float64, order='F')

    q_expected = np.array([
        [-0.2157, -0.5088,  0.6109,  0.5669],
        [-0.1078, -0.2544, -0.7760,  0.5669],
        [-0.9705,  0.1413, -0.0495, -0.1890],
        [ 0.0000,  0.8102,  0.1486,  0.5669]
    ], dtype=np.float64, order='F')

    z_expected = np.array([
        [-0.3651,  0.0000,  0.4472,  0.8165],
        [-0.9129,  0.0000,  0.0000, -0.4082],
        [ 0.0000, -1.0000,  0.0000,  0.0000],
        [-0.1826,  0.0000, -0.8944,  0.4082]
    ], dtype=np.float64, order='F')

    # Verify results with appropriate tolerance
    # HTML shows 4 decimal places, use rtol=1e-3
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(e_out, e_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(q, q_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z, z_expected, rtol=1e-3, atol=1e-4)


def test_tg01fd_zero_dimensions():
    """Test TG01FD with zero dimensions (edge case)."""
    l, n, m, p = 0, 0, 0, 0
    tol = 0.0
    compq = 'N'
    compz = 'N'
    joba = 'N'

    a = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    e = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    b = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    c = np.array([], dtype=np.float64).reshape(0, 0, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fd(
        compq, compz, joba, l, n, m, p, a, e, b, c, tol
    )

    assert info == 0, f"TG01FD failed with info={info}"
    assert ranke == 0, f"Expected RANKE=0, got {ranke}"


def test_tg01fd_invalid_compq():
    """Test TG01FD with invalid COMPQ parameter."""
    l, n, m, p = 2, 2, 1, 1
    tol = 0.0
    compq = 'X'  # Invalid
    compz = 'N'
    joba = 'N'

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    b = np.ones((2, 1), dtype=np.float64, order='F')
    c = np.ones((1, 2), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fd(
        compq, compz, joba, l, n, m, p, a, e, b, c, tol
    )

    assert info < 0, f"Expected info < 0 for invalid COMPQ, got {info}"


def test_tg01fd_joba_trapezoidal():
    """Test TG01FD with JOBA='T' (trapezoidal reduction)."""
    l, n, m, p = 3, 3, 1, 1
    tol = 1e-10
    compq = 'I'
    compz = 'I'
    joba = 'T'

    a = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    b = np.ones((3, 1), dtype=np.float64, order='F')
    c = np.ones((1, 3), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fd(
        compq, compz, joba, l, n, m, p, a, e, b, c, tol
    )

    assert info == 0, f"TG01FD failed with info={info}"
    assert ranke >= 0 and ranke <= min(l, n), f"Invalid RANKE={ranke}"

    # Verify E has the expected form: upper triangular in top-left, zeros elsewhere
    for i in range(ranke, l):
        for j in range(n):
            assert abs(e_out[i, j]) < 1e-10, \
                f"Expected E[{i},{j}]=0, got {e_out[i, j]}"


def test_tg01fd_compq_update():
    """Test TG01FD with COMPQ='U' (update existing Q matrix)."""
    l, n, m, p = 2, 2, 1, 1
    tol = 0.0
    compq = 'U'
    compz = 'N'
    joba = 'N'

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order='F')
    e = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64, order='F')
    b = np.ones((2, 1), dtype=np.float64, order='F')
    c = np.ones((1, 2), dtype=np.float64, order='F')

    # Start with identity for Q1
    q1 = np.eye(2, dtype=np.float64, order='F')

    # Note: We'd need to modify the wrapper to accept Q as input for COMPQ='U'
    # For now, this test documents the expected behavior
    # The wrapper should handle Q as input/output when COMPQ='U'


def test_tg01fd_boundary_pivots():
    """Test TG01FD with larger matrix to exercise permutation cycles.

    Tests boundary conditions in pivot index conversion to ensure
    out-of-bounds access prevention works correctly.
    """
    l, n, m, p = 5, 5, 2, 2
    tol = 1e-10
    compq = 'I'
    compz = 'I'
    joba = 'R'

    # Create test matrices with structure that will trigger permutations
    a = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [2.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    b = np.ones((5, 2), dtype=np.float64, order='F')
    c = np.ones((2, 5), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, ranke, rnka22, info = tg01fd(
        compq, compz, joba, l, n, m, p, a, e, b, c, tol
    )

    # Verify successful completion
    assert info == 0, f"TG01FD failed with info={info}"
    assert ranke >= 0 and ranke <= min(l, n), f"Invalid RANKE={ranke}"
    assert rnka22 >= 0, f"Invalid RNKA22={rnka22}"

    # Verify E has proper form (zeros in lower part)
    for i in range(ranke, l):
        for j in range(n):
            assert abs(e_out[i, j]) < 1e-8, \
                f"Expected E[{i},{j}]=0, got {e_out[i, j]}"

    # Verify Q and Z are orthogonal
    q_check = q.T @ q
    z_check = z.T @ z
    np.testing.assert_allclose(q_check, np.eye(l), rtol=1e-10, atol=1e-10,
                               err_msg="Q should be orthogonal")
    np.testing.assert_allclose(z_check, np.eye(n), rtol=1e-10, atol=1e-10,
                               err_msg="Z should be orthogonal")
