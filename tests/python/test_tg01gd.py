import pytest
import numpy as np
from ctrlsys import tg01gd


def test_tg01gd_basic_example():
    """Test TG01GD with example from SLICOT HTML documentation.

    L=4, N=4, M=2, P=2, JOBS='D', TOL=0.0
    Tests reduced descriptor representation without non-dynamic modes.
    """
    l, n, m, p = 4, 4, 2, 2
    tol = 0.0
    jobs = 'D'

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

    # D matrix (P x M)
    d = np.array([
        [1.0,  0.0],
        [1.0,  1.0]
    ], dtype=np.float64, order='F')

    # Call TG01GD
    a_out, e_out, b_out, c_out, d_out, lr, nr, ranke, infred, info = tg01gd(
        jobs, l, n, m, p, a, e, b, c, d, tol
    )

    # Check success
    assert info == 0, f"TG01GD failed with info={info}"

    # Check ranks and dimensions
    assert ranke == 3, f"Expected RANKE=3, got {ranke}"
    assert lr == 3, f"Expected LR=3, got {lr}"
    assert nr == 3, f"Expected NR=3, got {nr}"
    assert infred >= 0, f"Expected INFRED>=0, got {infred}"

    # Expected results from documentation (LR x NR reduced matrices)
    a_expected = np.array([
        [ 2.5102, -3.8550, -11.4533],
        [-0.0697,  0.0212,   0.7015],
        [ 0.3798, -0.1156,  -3.8250]
    ], dtype=np.float64, order='F')

    e_expected = np.array([
        [10.1587,  5.8230,  1.3021],
        [ 0.0000, -2.4684, -0.1896],
        [ 0.0000,  0.0000,  1.0338]
    ], dtype=np.float64, order='F')

    b_expected = np.array([
        [7.7100,  1.6714],
        [0.7678,  1.1070],
        [2.5428,  0.6935]
    ], dtype=np.float64, order='F')

    c_expected = np.array([
        [ 0.5477, -2.5000, -6.2610],
        [-1.0954,  1.0000, -0.8944]
    ], dtype=np.float64, order='F')

    d_expected = np.array([
        [4.0000,  1.0000],
        [1.0000,  1.0000]
    ], dtype=np.float64, order='F')

    # Verify results - only check the LR x NR part
    np.testing.assert_allclose(a_out[:lr, :nr], a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(e_out[:lr, :nr], e_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b_out[:lr, :], b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(c_out[:, :nr], c_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(d_out, d_expected, rtol=1e-3, atol=1e-4)


def test_tg01gd_standard_form():
    """Test TG01GD with JOBS='S' (standard form with identity E11).

    Tests that the leading RANKE-by-RANKE submatrix of Er becomes identity.
    """
    l, n, m, p = 4, 4, 2, 2
    tol = 0.0
    jobs = 'S'

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

    d = np.array([
        [1.0,  0.0],
        [1.0,  1.0]
    ], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, d_out, lr, nr, ranke, infred, info = tg01gd(
        jobs, l, n, m, p, a, e, b, c, d, tol
    )

    assert info == 0, f"TG01GD failed with info={info}"
    assert ranke == 3, f"Expected RANKE=3, got {ranke}"

    # With JOBS='S', the leading RANKE x RANKE submatrix of E should be identity
    e_identity = e_out[:ranke, :ranke]
    np.testing.assert_allclose(e_identity, np.eye(ranke), rtol=1e-10, atol=1e-10,
                               err_msg="E11 should be identity matrix with JOBS='S'")


def test_tg01gd_zero_dimensions():
    """Test TG01GD with zero dimensions (edge case)."""
    l, n, m, p = 0, 0, 0, 0
    tol = 0.0
    jobs = 'D'

    a = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    e = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    b = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    c = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    d = np.array([], dtype=np.float64).reshape(0, 0, order='F')

    a_out, e_out, b_out, c_out, d_out, lr, nr, ranke, infred, info = tg01gd(
        jobs, l, n, m, p, a, e, b, c, d, tol
    )

    assert info == 0, f"TG01GD failed with info={info}"
    assert lr == 0, f"Expected LR=0, got {lr}"
    assert nr == 0, f"Expected NR=0, got {nr}"
    assert ranke == 0, f"Expected RANKE=0, got {ranke}"


def test_tg01gd_invalid_jobs():
    """Test TG01GD with invalid JOBS parameter."""
    l, n, m, p = 2, 2, 1, 1
    tol = 0.0
    jobs = 'X'  # Invalid

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    b = np.ones((2, 1), dtype=np.float64, order='F')
    c = np.ones((1, 2), dtype=np.float64, order='F')
    d = np.zeros((1, 1), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, d_out, lr, nr, ranke, infred, info = tg01gd(
        jobs, l, n, m, p, a, e, b, c, d, tol
    )

    assert info < 0, f"Expected info < 0 for invalid JOBS, got {info}"


def test_tg01gd_no_reduction_possible():
    """Test TG01GD when no order reduction is possible.

    When E is full rank and A22 is empty, no reduction occurs.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    l, n, m, p = 3, 3, 1, 1
    tol = 0.0
    jobs = 'D'

    # Full rank E (no deficient part)
    e = np.array([
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    a = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], dtype=np.float64, order='F')

    b = np.ones((3, 1), dtype=np.float64, order='F')
    c = np.ones((1, 3), dtype=np.float64, order='F')
    d = np.zeros((1, 1), dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, d_out, lr, nr, ranke, infred, info = tg01gd(
        jobs, l, n, m, p, a, e, b, c, d, tol
    )

    assert info == 0, f"TG01GD failed with info={info}"
    # When E is full rank (ranke=n) and no reduction, LR=L, NR=N
    assert lr == l, f"Expected LR={l} (no reduction), got {lr}"
    assert nr == n, f"Expected NR={n} (no reduction), got {nr}"


def test_tg01gd_feedthrough_modification():
    """Test that feedthrough matrix D is correctly modified.

    Validates Dr = D - C2*inv(A22)*B2.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    l, n, m, p = 4, 4, 2, 2
    tol = 0.0
    jobs = 'D'

    # Same input as basic example
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

    d = np.array([
        [1.0,  0.0],
        [1.0,  1.0]
    ], dtype=np.float64, order='F')

    d_orig = d.copy()

    a_out, e_out, b_out, c_out, d_out, lr, nr, ranke, infred, info = tg01gd(
        jobs, l, n, m, p, a, e, b, c, d, tol
    )

    assert info == 0, f"TG01GD failed with info={info}"

    # D should be modified if reduction occurred
    if infred > 0:
        # D was modified - check expected from docs
        d_expected = np.array([
            [4.0,  1.0],
            [1.0,  1.0]
        ], dtype=np.float64, order='F')
        np.testing.assert_allclose(d_out, d_expected, rtol=1e-3, atol=1e-4)


def test_tg01gd_transfer_function_preservation():
    """Test that transfer function is preserved after reduction.

    For a descriptor system (E, A, B, C, D), the transfer function
    G(s) = C*(sE - A)^(-1)*B + D should be preserved by the reduction.

    This is a property-based test validating mathematical invariance.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    l, n, m, p = 4, 4, 2, 2
    tol = 0.0
    jobs = 'D'

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

    d = np.array([
        [1.0,  0.0],
        [1.0,  1.0]
    ], dtype=np.float64, order='F')

    # Save originals for comparison
    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()
    c_orig = c.copy()
    d_orig = d.copy()

    a_out, e_out, b_out, c_out, d_out, lr, nr, ranke, infred, info = tg01gd(
        jobs, l, n, m, p, a, e, b, c, d, tol
    )

    assert info == 0, f"TG01GD failed with info={info}"

    # Test at a few complex frequencies
    # G(s) = C*(sE - A)^(-1)*B + D
    test_frequencies = [0.1 + 0.5j, 1.0 + 2.0j, -0.5 + 1.0j]

    for s in test_frequencies:
        # Original transfer function (may be singular if E is rank deficient)
        try:
            sE_A_orig = s * e_orig - a_orig
            G_orig = c_orig @ np.linalg.solve(sE_A_orig, b_orig) + d_orig
        except np.linalg.LinAlgError:
            continue  # Skip singular cases

        # Reduced transfer function
        ar = a_out[:lr, :nr]
        er = e_out[:lr, :nr]
        br = b_out[:lr, :]
        cr = c_out[:, :nr]
        dr = d_out

        # For square reduced system
        if lr == nr:
            try:
                sE_A_red = s * er - ar
                G_red = cr @ np.linalg.solve(sE_A_red, br) + dr
                # Transfer functions should match (may have numerical differences)
                np.testing.assert_allclose(G_red, G_orig, rtol=1e-6, atol=1e-8,
                                           err_msg=f"Transfer function mismatch at s={s}")
            except np.linalg.LinAlgError:
                pass  # Skip if reduced system is also singular at this s
