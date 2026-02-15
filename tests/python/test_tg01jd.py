"""
Tests for tg01jd - Irreducible descriptor representation.

TG01JD finds a reduced (controllable, observable, or irreducible)
descriptor representation (Ar-lambda*Er,Br,Cr) for an original
descriptor representation (A-lambda*E,B,C).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tg01jd_html_example():
    """
    Test TG01JD using data from SLICOT HTML documentation.

    System: 9x9 descriptor system with 2 inputs, 2 outputs.
    JOB='I', SYSTYP='R', EQUIL='N'
    Expected: Reduced to 7th order, with 2 eigenvalues eliminated in Phase 3.
    """
    from ctrlsys import tg01jd

    n, m, p = 9, 2, 2
    tol = 0.0

    # A matrix (row-wise from HTML doc)
    a = np.array([
        [-2, -3,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0, -2, -3,  0,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1],
    ], order='F', dtype=float)

    # E matrix (row-wise from HTML doc)
    e = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
    ], order='F', dtype=float)

    # B matrix (row-wise from HTML doc)
    b = np.array([
        [ 1,  0],
        [ 0,  0],
        [ 0,  1],
        [ 0,  0],
        [-1,  0],
        [ 0,  0],
        [ 0, -1],
        [ 0,  0],
        [ 0,  0],
    ], order='F', dtype=float)

    # C matrix (row-wise from HTML doc)
    c = np.array([
        [1, 0, 1, -3, 0, 1, 0, 2, 0],
        [0, 1, 1,  3, 0, 1, 0, 0, 1],
    ], order='F', dtype=float)

    result = tg01jd('I', 'R', 'N', a, e, b, c, tol)

    # Unpack results
    a_r, e_r, b_r, c_r, nr, infred, iwork, info = result

    assert info == 0, f"TG01JD returned info={info}"
    assert nr == 7, f"Expected nr=7, got {nr}"

    # Validate INFRED structure
    # INFRED(k) >= 0 means Phase k was performed with reduction
    # INFRED(k) = -1 means Phase k had no reduction (or was skipped)
    # For SYSTYP='R' (rational), all 4 phases are attempted
    # This system has: no uncontrollable eigenvalues, 2 unobservable eigenvalues
    # Phase 3 should eliminate 2 eigenvalues
    assert infred[2] == 2, f"Expected 2 eliminated in Phase 3, got {infred[2]}"
    # Other phases may or may not show reduction (depends on lspace condition)

    # Verify transfer function preservation instead of exact matrix values.
    # Different orthogonal transformations produce mathematically equivalent results,
    # so matrix values may differ from HTML doc while transfer function is preserved.

    # Extract reduced system
    a_red = a_r[:nr, :nr]
    e_red = e_r[:nr, :nr]
    b_red = b_r[:nr, :m]
    c_red = c_r[:p, :nr]

    # Test transfer function G(s) = C * inv(sE - A) * B at several frequencies
    test_freqs = [1.0, 2.0 + 1j, -0.5 + 2j, 0.1j, 5.0]

    for s in test_freqs:
        # Original system
        try:
            G_orig = c @ np.linalg.solve(s * e - a, b)
        except np.linalg.LinAlgError:
            continue

        # Reduced system
        try:
            G_red = c_red @ np.linalg.solve(s * e_red - a_red, b_red)
        except np.linalg.LinAlgError:
            continue

        # Transfer functions must match (machine precision)
        assert_allclose(G_red, G_orig, rtol=1e-10, atol=1e-12,
                        err_msg=f"Transfer function mismatch at s={s}")


def test_tg01jd_controllable_only():
    """
    Test TG01JD with JOB='C' (controllable part only).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tg01jd

    np.random.seed(42)
    n, m, p = 4, 2, 2

    # Create a simple controllable system
    a = np.array([
        [1, 2, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 2, 1],
        [0, 0, 0, 2],
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    b = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ], order='F', dtype=float)

    c = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ], order='F', dtype=float)

    result = tg01jd('C', 'R', 'N', a, e, b, c, 0.0)
    a_r, e_r, b_r, c_r, nr, infred, iwork, info = result

    assert info == 0, f"TG01JD returned info={info}"
    # System is fully controllable, so nr should equal n
    assert nr == n, f"Expected nr={n}, got {nr}"


def test_tg01jd_observable_only():
    """
    Test TG01JD with JOB='O' (observable part only).

    Uses a verifiably observable system with distinct eigenvalues.
    Observability matrix [C; CA; CA^2; ...] should have full rank.
    """
    from ctrlsys import tg01jd

    n, m, p = 4, 2, 2

    # Observable canonical form with distinct eigenvalues
    a = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-24, -50, -35, -10],  # Characteristic poly: s^4 + 10s^3 + 35s^2 + 50s + 24
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    b = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ], order='F', dtype=float)

    # Output from all states
    c = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], order='F', dtype=float)

    # Verify observability: O = [C; CA; CA^2; CA^3]
    obs_mat = np.vstack([c, c @ a, c @ (a @ a), c @ (a @ a @ a)])
    obs_rank = np.linalg.matrix_rank(obs_mat)
    assert obs_rank == n, f"Test setup error: observability rank is {obs_rank}, not {n}"

    result = tg01jd('O', 'R', 'N', a, e, b, c, 0.0)
    a_r, e_r, b_r, c_r, nr, infred, iwork, info = result

    assert info == 0, f"TG01JD returned info={info}"
    # System is observable, nr should be n (no reduction)
    assert nr == n, f"Expected nr={n}, got {nr}"


def test_tg01jd_with_scaling():
    """
    Test TG01JD with EQUIL='S' (perform scaling).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tg01jd

    np.random.seed(456)
    n, m, p = 3, 1, 1

    # Create a poorly scaled system
    a = np.array([
        [1e6, 1e-6, 0],
        [1e-6, 1e6, 0],
        [0, 0, 1],
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    b = np.array([
        [1e3],
        [1e-3],
        [1],
    ], order='F', dtype=float)

    c = np.array([
        [1e3, 1e-3, 1],
    ], order='F', dtype=float)

    result = tg01jd('I', 'R', 'S', a, e, b, c, 0.0)
    a_r, e_r, b_r, c_r, nr, infred, iwork, info = result

    assert info == 0, f"TG01JD returned info={info}"


def test_tg01jd_systyp_standard():
    """
    Test TG01JD with SYSTYP='S' (proper/standard transfer function).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tg01jd

    np.random.seed(789)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    result = tg01jd('I', 'S', 'N', a, e, b, c, 0.0)
    a_r, e_r, b_r, c_r, nr, infred, iwork, info = result

    assert info == 0, f"TG01JD returned info={info}"
    assert nr >= 0 and nr <= n


def test_tg01jd_systyp_polynomial():
    """
    Test TG01JD with SYSTYP='P' (polynomial transfer function).

    Random seed: 101112 (for reproducibility)
    """
    from ctrlsys import tg01jd

    np.random.seed(101112)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    result = tg01jd('I', 'P', 'N', a, e, b, c, 0.0)
    a_r, e_r, b_r, c_r, nr, infred, iwork, info = result

    assert info == 0, f"TG01JD returned info={info}"
    assert nr >= 0 and nr <= n


def test_tg01jd_edge_zero_system():
    """
    Test TG01JD with n=0 (edge case - quick return).
    """
    from ctrlsys import tg01jd

    n, m, p = 0, 2, 2

    a = np.zeros((0, 0), order='F', dtype=float)
    e = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, m), order='F', dtype=float)
    c = np.zeros((p, 0), order='F', dtype=float)

    result = tg01jd('I', 'R', 'N', a, e, b, c, 0.0)
    a_r, e_r, b_r, c_r, nr, infred, iwork, info = result

    assert info == 0, f"TG01JD returned info={info}"
    assert nr == 0


def test_tg01jd_error_invalid_job():
    """
    Test TG01JD with invalid JOB parameter.
    """
    from ctrlsys import tg01jd

    n, m, p = 3, 1, 1
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    result = tg01jd('X', 'R', 'N', a, e, b, c, 0.0)
    info = result[-1]

    assert info == -1, f"Expected info=-1 for invalid JOB, got {info}"


def test_tg01jd_error_invalid_systyp():
    """
    Test TG01JD with invalid SYSTYP parameter.
    """
    from ctrlsys import tg01jd

    n, m, p = 3, 1, 1
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    result = tg01jd('I', 'X', 'N', a, e, b, c, 0.0)
    info = result[-1]

    assert info == -2, f"Expected info=-2 for invalid SYSTYP, got {info}"


def test_tg01jd_error_invalid_equil():
    """
    Test TG01JD with invalid EQUIL parameter.
    """
    from ctrlsys import tg01jd

    n, m, p = 3, 1, 1
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    result = tg01jd('I', 'R', 'X', a, e, b, c, 0.0)
    info = result[-1]

    assert info == -3, f"Expected info=-3 for invalid EQUIL, got {info}"


def test_tg01jd_transfer_function_preservation():
    """
    Test that reduced system preserves transfer function.

    The irreducible representation should have the same transfer function
    as the original system at random evaluation points.

    Random seed: 2024 (for reproducibility)
    """
    from ctrlsys import tg01jd

    np.random.seed(2024)
    n, m, p = 5, 2, 2

    # Create original system
    a = np.diag([1, 2, 3, 4, 5]).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    result = tg01jd('I', 'R', 'N', a, e, b, c, 0.0)
    a_r, e_r, b_r, c_r, nr, infred, iwork, info = result

    assert info == 0

    # Evaluate transfer function at test frequencies
    test_freqs = [0.1j, 1.0j, 10.0j, 0.5 + 0.5j]

    for s in test_freqs:
        # Original: G(s) = C * (s*E - A)^{-1} * B
        try:
            G_orig = c_orig @ np.linalg.solve(s * e_orig - a_orig, b_orig)
        except np.linalg.LinAlgError:
            continue

        # Reduced: Gr(s) = Cr * (s*Er - Ar)^{-1} * Br
        a_red = a_r[:nr, :nr]
        e_red = e_r[:nr, :nr]
        b_red = b_r[:nr, :m]
        c_red = c_r[:p, :nr]

        try:
            G_red = c_red @ np.linalg.solve(s * e_red - a_red, b_red)
        except np.linalg.LinAlgError:
            continue

        # Check transfer function equality
        assert_allclose(G_red, G_orig, rtol=1e-10, atol=1e-12)
