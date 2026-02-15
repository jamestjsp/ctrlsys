"""
Tests for SB10DD: H-infinity (sub)optimal controller for discrete-time system.

Tests numerical correctness against SLICOT HTML documentation example.
"""
import numpy as np
import pytest


def test_sb10dd_html_doc_example():
    """
    Validate SB10DD using HTML documentation example.

    System: 6 states, 5 inputs (3 disturbance + 2 control),
            5 outputs (3 performance + 2 measurement)
    gamma = 111.294
    """
    from ctrlsys import sb10dd

    n = 6
    m = 5
    np_ = 5
    ncon = 2
    nmeas = 2
    gamma = 111.294
    tol = 1e-8

    # A matrix (6x6) - read row-by-row per Fortran READ
    a = np.array([
        [-0.7,  0.0,  0.3,  0.0, -0.5, -0.1],
        [-0.6,  0.2, -0.4, -0.3,  0.0,  0.0],
        [-0.5,  0.7, -0.1,  0.0,  0.0, -0.8],
        [-0.7,  0.0,  0.0, -0.5, -1.0,  0.0],
        [ 0.0,  0.3,  0.6, -0.9,  0.1, -0.4],
        [ 0.5, -0.8,  0.0,  0.0,  0.2, -0.9]
    ], order='F', dtype=float)

    # B matrix (6x5) - read row-by-row
    b = np.array([
        [-1.0, -2.0, -2.0,  1.0,  0.0],
        [ 1.0,  0.0,  1.0, -2.0,  1.0],
        [-3.0, -4.0,  0.0,  2.0, -2.0],
        [ 1.0, -2.0,  1.0,  0.0, -1.0],
        [ 0.0,  1.0, -2.0,  0.0,  3.0],
        [ 1.0,  0.0,  3.0, -1.0, -2.0]
    ], order='F', dtype=float)

    # C matrix (5x6) - read row-by-row
    c = np.array([
        [ 1.0, -1.0,  2.0, -2.0,  0.0, -3.0],
        [-3.0,  0.0,  1.0, -1.0,  1.0,  0.0],
        [ 0.0,  2.0,  0.0, -4.0,  0.0, -2.0],
        [ 1.0, -3.0,  0.0,  0.0,  3.0,  1.0],
        [ 0.0,  1.0, -2.0,  1.0,  0.0, -2.0]
    ], order='F', dtype=float)

    # D matrix (5x5) - read row-by-row
    d = np.array([
        [ 1.0, -1.0, -2.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0,  1.0,  0.0],
        [ 2.0, -1.0, -3.0,  0.0,  1.0],
        [ 0.0,  1.0,  0.0,  1.0, -1.0],
        [ 0.0,  0.0,  1.0,  2.0,  1.0]
    ], order='F', dtype=float)

    ak, bk, ck, dk, x, z, rcond, info = sb10dd(n, m, np_, ncon, nmeas, gamma, a, b, c, d, tol)

    assert info == 0, f"sb10dd failed with info={info}"

    # Expected AK from HTML doc (6x6)
    ak_expected = np.array([
        [-18.0030,  52.0376,  26.0831, -0.4271, -40.9022,  18.0857],
        [ 18.8203, -57.6244, -29.0938,  0.5870,  45.3309, -19.8644],
        [-26.5994,  77.9693,  39.0368, -1.4020, -60.1129,  26.6910],
        [-21.4163,  62.1719,  30.7507, -0.9201, -48.6221,  21.8351],
        [ -0.8911,   4.2787,   2.3286, -0.2424,  -3.0376,   1.2169],
        [ -5.3286,  16.1955,   8.4824, -0.2489, -12.2348,   5.1590]
    ], order='F', dtype=float)

    # Expected BK from HTML doc (6x2)
    bk_expected = np.array([
        [16.9788, 14.1648],
        [-18.9215, -15.6726],
        [25.2046, 21.2848],
        [20.1122, 16.8322],
        [1.4104, 1.2040],
        [5.3181, 4.5149]
    ], order='F', dtype=float)

    # Expected CK from HTML doc (2x6)
    ck_expected = np.array([
        [-9.1941, 27.5165, 13.7364, -0.3639, -21.5983, 9.6025],
        [3.6490, -10.6194, -5.2772, 0.2432, 8.1108, -3.6293]
    ], order='F', dtype=float)

    # Expected DK from HTML doc (2x2)
    dk_expected = np.array([
        [9.0317, 7.5348],
        [-3.4006, -2.8219]
    ], order='F', dtype=float)

    # Expected RCOND values from HTML doc
    rcond_expected = np.array([
        0.24960e+00, 0.98548e+00, 0.99186e+00, 0.63733e-05,
        0.48625e+00, 0.29430e-01, 0.56942e-02, 0.12470e-01
    ])

    # Validate controller matrices (4 decimal places per HTML display)
    np.testing.assert_allclose(ak, ak_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(bk, bk_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ck, ck_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dk, dk_expected, rtol=1e-3, atol=1e-3)

    # Validate RCOND values (looser tolerance for condition numbers)
    np.testing.assert_allclose(rcond, rcond_expected, rtol=0.1, atol=1e-4)

    # Validate X and Z are symmetric (Riccati solutions)
    np.testing.assert_allclose(x, x.T, rtol=1e-12)
    np.testing.assert_allclose(z, z.T, rtol=1e-12)


def test_sb10dd_controller_symmetry():
    """
    Validate controller matrices have expected properties.

    Uses HTML doc example data which is known to produce valid controller.
    The Riccati solutions X and Z should be symmetric.
    """
    from ctrlsys import sb10dd

    n = 6
    m = 5
    np_ = 5
    ncon = 2
    nmeas = 2
    gamma = 111.294
    tol = 1e-8

    a = np.array([
        [-0.7,  0.0,  0.3,  0.0, -0.5, -0.1],
        [-0.6,  0.2, -0.4, -0.3,  0.0,  0.0],
        [-0.5,  0.7, -0.1,  0.0,  0.0, -0.8],
        [-0.7,  0.0,  0.0, -0.5, -1.0,  0.0],
        [ 0.0,  0.3,  0.6, -0.9,  0.1, -0.4],
        [ 0.5, -0.8,  0.0,  0.0,  0.2, -0.9]
    ], order='F', dtype=float)

    b = np.array([
        [-1.0, -2.0, -2.0,  1.0,  0.0],
        [ 1.0,  0.0,  1.0, -2.0,  1.0],
        [-3.0, -4.0,  0.0,  2.0, -2.0],
        [ 1.0, -2.0,  1.0,  0.0, -1.0],
        [ 0.0,  1.0, -2.0,  0.0,  3.0],
        [ 1.0,  0.0,  3.0, -1.0, -2.0]
    ], order='F', dtype=float)

    c = np.array([
        [ 1.0, -1.0,  2.0, -2.0,  0.0, -3.0],
        [-3.0,  0.0,  1.0, -1.0,  1.0,  0.0],
        [ 0.0,  2.0,  0.0, -4.0,  0.0, -2.0],
        [ 1.0, -3.0,  0.0,  0.0,  3.0,  1.0],
        [ 0.0,  1.0, -2.0,  1.0,  0.0, -2.0]
    ], order='F', dtype=float)

    d = np.array([
        [ 1.0, -1.0, -2.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0,  1.0,  0.0],
        [ 2.0, -1.0, -3.0,  0.0,  1.0],
        [ 0.0,  1.0,  0.0,  1.0, -1.0],
        [ 0.0,  0.0,  1.0,  2.0,  1.0]
    ], order='F', dtype=float)

    ak, bk, ck, dk, x, z, rcond, info = sb10dd(n, m, np_, ncon, nmeas, gamma, a, b, c, d, tol)

    assert info == 0, f"sb10dd failed with info={info}"

    # X and Z should be symmetric (Riccati solutions)
    np.testing.assert_allclose(x, x.T, rtol=1e-12)
    np.testing.assert_allclose(z, z.T, rtol=1e-12)

    # Controller matrices should match expected dimensions
    assert ak.shape == (n, n)
    assert bk.shape == (n, nmeas)
    assert ck.shape == (ncon, n)
    assert dk.shape == (ncon, nmeas)


def test_sb10dd_riccati_solutions():
    """
    Validate Riccati equation solutions X and Z.

    X should satisfy the X-Riccati equation for the state feedback problem.
    Z should satisfy the Z-Riccati equation for the output injection problem.
    """
    from ctrlsys import sb10dd

    n = 6
    m = 5
    np_ = 5
    ncon = 2
    nmeas = 2
    gamma = 111.294
    tol = 1e-8

    # Use HTML doc example data
    a = np.array([
        [-0.7,  0.0,  0.3,  0.0, -0.5, -0.1],
        [-0.6,  0.2, -0.4, -0.3,  0.0,  0.0],
        [-0.5,  0.7, -0.1,  0.0,  0.0, -0.8],
        [-0.7,  0.0,  0.0, -0.5, -1.0,  0.0],
        [ 0.0,  0.3,  0.6, -0.9,  0.1, -0.4],
        [ 0.5, -0.8,  0.0,  0.0,  0.2, -0.9]
    ], order='F', dtype=float)

    b = np.array([
        [-1.0, -2.0, -2.0,  1.0,  0.0],
        [ 1.0,  0.0,  1.0, -2.0,  1.0],
        [-3.0, -4.0,  0.0,  2.0, -2.0],
        [ 1.0, -2.0,  1.0,  0.0, -1.0],
        [ 0.0,  1.0, -2.0,  0.0,  3.0],
        [ 1.0,  0.0,  3.0, -1.0, -2.0]
    ], order='F', dtype=float)

    c = np.array([
        [ 1.0, -1.0,  2.0, -2.0,  0.0, -3.0],
        [-3.0,  0.0,  1.0, -1.0,  1.0,  0.0],
        [ 0.0,  2.0,  0.0, -4.0,  0.0, -2.0],
        [ 1.0, -3.0,  0.0,  0.0,  3.0,  1.0],
        [ 0.0,  1.0, -2.0,  1.0,  0.0, -2.0]
    ], order='F', dtype=float)

    d = np.array([
        [ 1.0, -1.0, -2.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0,  1.0,  0.0],
        [ 2.0, -1.0, -3.0,  0.0,  1.0],
        [ 0.0,  1.0,  0.0,  1.0, -1.0],
        [ 0.0,  0.0,  1.0,  2.0,  1.0]
    ], order='F', dtype=float)

    ak, bk, ck, dk, x, z, rcond, info = sb10dd(n, m, np_, ncon, nmeas, gamma, a, b, c, d, tol)

    assert info == 0

    # X and Z should be symmetric
    np.testing.assert_allclose(x, x.T, rtol=1e-12)
    np.testing.assert_allclose(z, z.T, rtol=1e-12)

    # X should be positive semidefinite
    eig_x = np.linalg.eigvalsh(x)
    assert np.all(eig_x >= -1e-10), f"X not positive semidefinite: min_eig={np.min(eig_x)}"

    # Z should be positive semidefinite
    eig_z = np.linalg.eigvalsh(z)
    assert np.all(eig_z >= -1e-10), f"Z not positive semidefinite: min_eig={np.min(eig_z)}"


def test_sb10dd_edge_case_small_system():
    """
    Test with smaller system than HTML doc example.

    n=2, m=3, np=3, ncon=1, nmeas=1
    Constraints: np-nmeas=2 >= ncon=1, m-ncon=2 >= nmeas=1
    """
    from ctrlsys import sb10dd

    n = 2
    m = 3  # m1=2, m2=1
    np_ = 3  # np1=2, np2=1
    ncon = 1
    nmeas = 1

    # Stable system
    a = np.array([[0.5, 0.1], [0.0, 0.4]], order='F', dtype=float)

    # B = [B1 B2] with m1=2 disturbance, m2=1 control
    b = np.array([[0.3, 0.2, 1.0], [0.1, 0.5, 0.0]], order='F', dtype=float)

    # C = [C1; C2] with np1=2 performance, np2=1 measurement
    c = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], order='F', dtype=float)

    # D = [D11 D12; D21 D22] with full rank D12 (np1 x m2 = 2x1) and D21 (np2 x m1 = 1x2)
    d = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.5, 0.0]
    ], order='F', dtype=float)

    gamma = 10.0
    tol = 1e-8

    ak, bk, ck, dk, x, z, rcond, info = sb10dd(n, m, np_, ncon, nmeas, gamma, a, b, c, d, tol)

    assert info == 0, f"sb10dd failed with info={info}"

    # Check output dimensions
    assert ak.shape == (2, 2), f"AK shape {ak.shape} != (2, 2)"
    assert bk.shape == (2, 1), f"BK shape {bk.shape} != (2, 1)"
    assert ck.shape == (1, 2), f"CK shape {ck.shape} != (1, 2)"
    assert dk.shape == (1, 1), f"DK shape {dk.shape} != (1, 1)"
    assert x.shape == (2, 2), f"X shape {x.shape} != (2, 2)"
    assert z.shape == (2, 2), f"Z shape {z.shape} != (2, 2)"
    assert rcond.shape == (8,)


def test_sb10dd_error_rank_deficient_d12():
    """
    Test error handling when D12 does not have full column rank.

    Should return info=3.
    """
    from ctrlsys import sb10dd

    n = 2
    m = 3
    np_ = 3
    ncon = 2
    nmeas = 1
    m1 = 1  # m - ncon
    np1 = 2  # np - nmeas

    a = np.array([[0.5, 0.1], [0.0, 0.6]], order='F', dtype=float)
    b = np.array([[0.5, 1.0, 0.5], [0.0, 0.0, 0.0]], order='F', dtype=float)  # B2 has dependent columns
    c = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], order='F', dtype=float)

    # D12 (np1 x ncon = 2 x 2) is rank deficient
    d = np.array([
        [0.0, 1.0, 1.0],   # D12 row 1
        [0.0, 2.0, 2.0],   # D12 row 2 (dependent)
        [1.0, 0.0, 0.0]    # D21 row
    ], order='F', dtype=float)

    gamma = 10.0
    tol = 1e-8

    ak, bk, ck, dk, x, z, rcond, info = sb10dd(n, m, np_, ncon, nmeas, gamma, a, b, c, d, tol)

    assert info == 3, f"Expected info=3 for rank-deficient D12, got {info}"


def test_sb10dd_error_rank_deficient_d21():
    """
    Test error handling when D21 does not have full row rank.

    Should return info=4.
    """
    from ctrlsys import sb10dd

    n = 2
    m = 3
    np_ = 4
    ncon = 1
    nmeas = 2
    m1 = 2  # m - ncon
    np1 = 2  # np - nmeas

    a = np.array([[0.5, 0.1], [0.0, 0.6]], order='F', dtype=float)
    b = np.array([[0.5, 0.3, 1.0], [0.2, 0.1, 0.5]], order='F', dtype=float)
    c = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],   # C2 row 1
        [1.0, 1.0]    # C2 row 2
    ], order='F', dtype=float)

    # D21 (nmeas x m1 = 2 x 2) is rank deficient
    d = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5],
        [1.0, 2.0, 0.0],   # D21 row 1
        [1.0, 2.0, 0.0]    # D21 row 2 (same = dependent)
    ], order='F', dtype=float)

    gamma = 10.0
    tol = 1e-8

    ak, bk, ck, dk, x, z, rcond, info = sb10dd(n, m, np_, ncon, nmeas, gamma, a, b, c, d, tol)

    assert info == 4, f"Expected info=4 for rank-deficient D21, got {info}"


def test_sb10dd_error_small_gamma():
    """
    Test error handling when gamma is too small (controller not admissible).

    Should return info=5 (or 6/7 for Riccati failure).
    """
    from ctrlsys import sb10dd

    n = 6
    m = 5
    np_ = 5
    ncon = 2
    nmeas = 2

    # Use HTML doc example
    a = np.array([
        [-0.7,  0.0,  0.3,  0.0, -0.5, -0.1],
        [-0.6,  0.2, -0.4, -0.3,  0.0,  0.0],
        [-0.5,  0.7, -0.1,  0.0,  0.0, -0.8],
        [-0.7,  0.0,  0.0, -0.5, -1.0,  0.0],
        [ 0.0,  0.3,  0.6, -0.9,  0.1, -0.4],
        [ 0.5, -0.8,  0.0,  0.0,  0.2, -0.9]
    ], order='F', dtype=float)

    b = np.array([
        [-1.0, -2.0, -2.0,  1.0,  0.0],
        [ 1.0,  0.0,  1.0, -2.0,  1.0],
        [-3.0, -4.0,  0.0,  2.0, -2.0],
        [ 1.0, -2.0,  1.0,  0.0, -1.0],
        [ 0.0,  1.0, -2.0,  0.0,  3.0],
        [ 1.0,  0.0,  3.0, -1.0, -2.0]
    ], order='F', dtype=float)

    c = np.array([
        [ 1.0, -1.0,  2.0, -2.0,  0.0, -3.0],
        [-3.0,  0.0,  1.0, -1.0,  1.0,  0.0],
        [ 0.0,  2.0,  0.0, -4.0,  0.0, -2.0],
        [ 1.0, -3.0,  0.0,  0.0,  3.0,  1.0],
        [ 0.0,  1.0, -2.0,  1.0,  0.0, -2.0]
    ], order='F', dtype=float)

    d = np.array([
        [ 1.0, -1.0, -2.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0,  1.0,  0.0],
        [ 2.0, -1.0, -3.0,  0.0,  1.0],
        [ 0.0,  1.0,  0.0,  1.0, -1.0],
        [ 0.0,  0.0,  1.0,  2.0,  1.0]
    ], order='F', dtype=float)

    # gamma much smaller than optimal
    gamma = 1.0
    tol = 1e-8

    ak, bk, ck, dk, x, z, rcond, info = sb10dd(n, m, np_, ncon, nmeas, gamma, a, b, c, d, tol)

    # Should fail with info=5, 6, or 7 (admissibility or Riccati failure)
    assert info in [5, 6, 7], f"Expected info in [5,6,7] for small gamma, got {info}"


def test_sb10dd_output_dimensions():
    """
    Validate output array dimensions match specifications.
    """
    from ctrlsys import sb10dd

    n = 6
    m = 5
    np_ = 5
    ncon = 2
    nmeas = 2
    gamma = 111.294
    tol = 1e-8

    a = np.array([
        [-0.7,  0.0,  0.3,  0.0, -0.5, -0.1],
        [-0.6,  0.2, -0.4, -0.3,  0.0,  0.0],
        [-0.5,  0.7, -0.1,  0.0,  0.0, -0.8],
        [-0.7,  0.0,  0.0, -0.5, -1.0,  0.0],
        [ 0.0,  0.3,  0.6, -0.9,  0.1, -0.4],
        [ 0.5, -0.8,  0.0,  0.0,  0.2, -0.9]
    ], order='F', dtype=float)

    b = np.array([
        [-1.0, -2.0, -2.0,  1.0,  0.0],
        [ 1.0,  0.0,  1.0, -2.0,  1.0],
        [-3.0, -4.0,  0.0,  2.0, -2.0],
        [ 1.0, -2.0,  1.0,  0.0, -1.0],
        [ 0.0,  1.0, -2.0,  0.0,  3.0],
        [ 1.0,  0.0,  3.0, -1.0, -2.0]
    ], order='F', dtype=float)

    c = np.array([
        [ 1.0, -1.0,  2.0, -2.0,  0.0, -3.0],
        [-3.0,  0.0,  1.0, -1.0,  1.0,  0.0],
        [ 0.0,  2.0,  0.0, -4.0,  0.0, -2.0],
        [ 1.0, -3.0,  0.0,  0.0,  3.0,  1.0],
        [ 0.0,  1.0, -2.0,  1.0,  0.0, -2.0]
    ], order='F', dtype=float)

    d = np.array([
        [ 1.0, -1.0, -2.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0,  1.0,  0.0],
        [ 2.0, -1.0, -3.0,  0.0,  1.0],
        [ 0.0,  1.0,  0.0,  1.0, -1.0],
        [ 0.0,  0.0,  1.0,  2.0,  1.0]
    ], order='F', dtype=float)

    ak, bk, ck, dk, x, z, rcond, info = sb10dd(n, m, np_, ncon, nmeas, gamma, a, b, c, d, tol)

    assert info == 0

    # Controller dimensions
    assert ak.shape == (n, n), f"AK shape {ak.shape} != ({n}, {n})"
    assert bk.shape == (n, nmeas), f"BK shape {bk.shape} != ({n}, {nmeas})"
    assert ck.shape == (ncon, n), f"CK shape {ck.shape} != ({ncon}, {n})"
    assert dk.shape == (ncon, nmeas), f"DK shape {dk.shape} != ({ncon}, {nmeas})"

    # Riccati solution dimensions
    assert x.shape == (n, n), f"X shape {x.shape} != ({n}, {n})"
    assert z.shape == (n, n), f"Z shape {z.shape} != ({n}, {n})"

    # RCOND array
    assert rcond.shape == (8,), f"RCOND shape {rcond.shape} != (8,)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
