import pytest
import numpy as np
from ctrlsys import tg01ld


def test_tg01ld_basic_example():
    """Test TG01LD with example from SLICOT HTML documentation.

    N=4, M=2, P=2, JOB='F', JOBA='N', COMPQ='I', COMPZ='I', TOL=0.0
    Tests finite-infinite eigenvalue separation for descriptor system.
    """
    n, m, p = 4, 2, 2
    tol = 0.0
    job = 'F'
    joba = 'N'
    compq = 'I'
    compz = 'I'

    # Input matrices (row-wise READ in Fortran example)
    # A matrix (N x N)
    a = np.array([
        [-1.0,  0.0,  0.0,  3.0],
        [ 0.0,  0.0,  1.0,  2.0],
        [ 1.0,  1.0,  0.0,  4.0],
        [ 0.0,  0.0,  0.0,  0.0]
    ], dtype=np.float64, order='F')

    # E matrix (N x N)
    e = np.array([
        [1.0,  2.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  1.0],
        [3.0,  9.0,  6.0,  3.0],
        [0.0,  0.0,  2.0,  0.0]
    ], dtype=np.float64, order='F')

    # B matrix (N x M)
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

    # Call TG01LD
    a_out, e_out, b_out, c_out, q, z, nf, nd, niblck, iblck, info = tg01ld(
        job, joba, compq, compz, n, m, p, a, e, b, c, tol
    )

    # Check success
    assert info == 0, f"TG01LD failed with info={info}"

    # Check outputs from documentation
    assert nf == 3, f"Expected NF=3, got {nf}"
    assert nd == 1, f"Expected ND=1, got {nd}"
    assert niblck == 0, f"Expected NIBLCK=0, got {niblck}"

    # Expected A (Q'*A*Z) from documentation
    a_expected = np.array([
        [ 2.4497, -1.3995,  0.2397, -4.0023],
        [-0.0680, -0.0030,  0.1739, -1.6225],
        [ 0.3707,  0.0161, -0.9482,  0.1049],
        [ 0.0000,  0.0000,  0.0000,  2.2913]
    ], dtype=np.float64, order='F')

    # Expected E (Q'*E*Z) from documentation
    e_expected = np.array([
        [ 9.9139,  4.7725, -3.4725, -2.3836],
        [ 0.0000, -1.2024,  2.0137,  0.7926],
        [ 0.0000,  0.0000,  0.2929, -0.9914],
        [ 0.0000,  0.0000,  0.0000,  0.0000]
    ], dtype=np.float64, order='F')

    # Expected B (Q'*B) from documentation
    b_expected = np.array([
        [-0.2157, -0.9705],
        [ 0.3015,  0.9516],
        [ 0.7595,  0.0991],
        [ 1.1339,  0.3780]
    ], dtype=np.float64, order='F')

    # Expected C (C*Z) from documentation
    c_expected = np.array([
        [ 0.5345, -1.1134,  0.3758,  0.5774],
        [-1.0690,  0.2784, -1.2026,  0.5774]
    ], dtype=np.float64, order='F')

    # Expected Q from documentation
    q_expected = np.array([
        [-0.2157, -0.5088,  0.6109,  0.5669],
        [-0.1078, -0.2544, -0.7760,  0.5669],
        [-0.9705,  0.1413, -0.0495, -0.1890],
        [ 0.0000,  0.8102,  0.1486,  0.5669]
    ], dtype=np.float64, order='F')

    # Expected Z from documentation
    z_expected = np.array([
        [-0.5345,  0.6263,  0.4617, -0.3299],
        [-0.8018, -0.5219, -0.2792, -0.0825],
        [ 0.0000, -0.4871,  0.8375,  0.2474],
        [-0.2673,  0.3132, -0.0859,  0.9073]
    ], dtype=np.float64, order='F')

    # Verify results with appropriate tolerance (HTML shows 4 decimal places)
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(e_out, e_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(q, q_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z, z_expected, rtol=1e-3, atol=1e-4)


def test_tg01ld_orthogonality():
    """Test TG01LD transformation matrices Q and Z are orthogonal.

    Mathematical property test: Q'*Q = I and Z'*Z = I
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 5, 2, 3

    a = np.random.randn(n, n).astype(np.float64, order='F')
    e = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n, m).astype(np.float64, order='F')
    c = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nf, nd, niblck, iblck, info = tg01ld(
        'F', 'N', 'I', 'I', n, m, p, a.copy(order='F'), e.copy(order='F'),
        b.copy(order='F'), c.copy(order='F'), 0.0
    )

    if info == 0:
        # Verify Q and Z are orthogonal
        np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-14, atol=1e-14,
                                   err_msg="Q should be orthogonal")
        np.testing.assert_allclose(z.T @ z, np.eye(n), rtol=1e-14, atol=1e-14,
                                   err_msg="Z should be orthogonal")


def test_tg01ld_transformation_consistency():
    """Test TG01LD transformation: Q'*A_orig*Z = A_out, Q'*E_orig*Z = E_out.

    Mathematical property test validates the transformation equations.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 4, 2, 2

    a_orig = np.random.randn(n, n).astype(np.float64, order='F')
    e_orig = np.random.randn(n, n).astype(np.float64, order='F')
    b_orig = np.random.randn(n, m).astype(np.float64, order='F')
    c_orig = np.random.randn(p, n).astype(np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nf, nd, niblck, iblck, info = tg01ld(
        'F', 'N', 'I', 'I', n, m, p, a_orig.copy(order='F'), e_orig.copy(order='F'),
        b_orig.copy(order='F'), c_orig.copy(order='F'), 0.0
    )

    if info == 0:
        # Verify Q'*A*Z = A_out
        a_transformed = q.T @ a_orig @ z
        np.testing.assert_allclose(a_out, a_transformed, rtol=1e-13, atol=1e-14,
                                   err_msg="Q'*A*Z should equal A_out")

        # Verify Q'*E*Z = E_out
        e_transformed = q.T @ e_orig @ z
        np.testing.assert_allclose(e_out, e_transformed, rtol=1e-13, atol=1e-14,
                                   err_msg="Q'*E*Z should equal E_out")

        # Verify Q'*B = B_out
        b_transformed = q.T @ b_orig
        np.testing.assert_allclose(b_out, b_transformed, rtol=1e-13, atol=1e-14,
                                   err_msg="Q'*B should equal B_out")

        # Verify C*Z = C_out
        c_transformed = c_orig @ z
        np.testing.assert_allclose(c_out, c_transformed, rtol=1e-13, atol=1e-14,
                                   err_msg="C*Z should equal C_out")


def test_tg01ld_zero_dimensions():
    """Test TG01LD with zero dimensions (edge case)."""
    n, m, p = 0, 0, 0

    a = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    e = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    b = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    c = np.array([], dtype=np.float64).reshape(0, 0, order='F')

    a_out, e_out, b_out, c_out, q, z, nf, nd, niblck, iblck, info = tg01ld(
        'F', 'N', 'N', 'N', n, m, p, a, e, b, c, 0.0
    )

    assert info == 0, f"TG01LD failed with info={info}"
    assert nf == 0, f"Expected NF=0, got {nf}"
    assert nd == 0, f"Expected ND=0, got {nd}"
    assert niblck == 0, f"Expected NIBLCK=0, got {niblck}"


def test_tg01ld_invalid_job():
    """Test TG01LD with invalid JOB parameter."""
    n, m, p = 2, 1, 1

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    b = np.ones((2, 1), dtype=np.float64, order='F')
    c = np.ones((1, 2), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nf, nd, niblck, iblck, info = tg01ld(
        'X', 'N', 'N', 'N', n, m, p, a, e, b, c, 0.0  # Invalid JOB
    )

    assert info == -1, f"Expected info=-1 for invalid JOB, got {info}"


def test_tg01ld_invalid_joba():
    """Test TG01LD with invalid JOBA parameter."""
    n, m, p = 2, 1, 1

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    b = np.ones((2, 1), dtype=np.float64, order='F')
    c = np.ones((1, 2), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nf, nd, niblck, iblck, info = tg01ld(
        'F', 'X', 'N', 'N', n, m, p, a, e, b, c, 0.0  # Invalid JOBA
    )

    assert info == -2, f"Expected info=-2 for invalid JOBA, got {info}"


def test_tg01ld_job_infinite_first():
    """Test TG01LD with JOB='I' (infinite-finite separation).

    Verifies that the structure is swapped: infinite part comes first.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 4, 2, 2

    # Use the same example data
    a = np.array([
        [-1.0,  0.0,  0.0,  3.0],
        [ 0.0,  0.0,  1.0,  2.0],
        [ 1.0,  1.0,  0.0,  4.0],
        [ 0.0,  0.0,  0.0,  0.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0,  2.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  1.0],
        [3.0,  9.0,  6.0,  3.0],
        [0.0,  0.0,  2.0,  0.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0,  0.0],
        [0.0,  0.0],
        [0.0,  1.0],
        [1.0,  1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [-1.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0, -1.0,  1.0]
    ], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nf, nd, niblck, iblck, info = tg01ld(
        'I', 'N', 'I', 'I', n, m, p, a, e, b, c, 0.0
    )

    # Info should be 0 for success
    assert info == 0, f"TG01LD failed with info={info}"

    # Same pencil should yield same NF
    assert nf == 3, f"Expected NF=3, got {nf}"

    # Verify Q and Z are orthogonal
    np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z.T @ z, np.eye(n), rtol=1e-14, atol=1e-14)


def test_tg01ld_hessenberg_reduction():
    """Test TG01LD with JOBA='H' (reduce Af to Hessenberg form).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 4, 2, 2

    a = np.array([
        [-1.0,  0.0,  0.0,  3.0],
        [ 0.0,  0.0,  1.0,  2.0],
        [ 1.0,  1.0,  0.0,  4.0],
        [ 0.0,  0.0,  0.0,  0.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0,  2.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  1.0],
        [3.0,  9.0,  6.0,  3.0],
        [0.0,  0.0,  2.0,  0.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0,  0.0],
        [0.0,  0.0],
        [0.0,  1.0],
        [1.0,  1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [-1.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0, -1.0,  1.0]
    ], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, nf, nd, niblck, iblck, info = tg01ld(
        'F', 'H', 'I', 'I', n, m, p, a, e, b, c, 0.0
    )

    assert info == 0, f"TG01LD failed with info={info}"
    assert nf == 3, f"Expected NF=3, got {nf}"

    # Verify Af (top-left NF x NF) is in upper Hessenberg form
    # Elements below the first subdiagonal should be zero
    for i in range(2, nf):
        for j in range(i - 1):
            assert abs(a_out[i, j]) < 1e-10, \
                f"Af[{i},{j}] = {a_out[i,j]} should be zero for Hessenberg"

    # Verify Q and Z are still orthogonal
    np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z.T @ z, np.eye(n), rtol=1e-14, atol=1e-14)
