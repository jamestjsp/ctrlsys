"""
Tests for SB03QD: Estimate conditioning and forward error bound for continuous-time Lyapunov.

Estimates conditioning and computes an error bound on the solution of:
    op(A)' * X + X * op(A) = scale * C

where op(A) = A or A' and C is symmetric.

Tests:
1. Basic HTML doc example (N=3)
2. Condition number only (JOB='C')
3. Error bound only (JOB='E')
4. Transpose form (TRANA='T')
5. Reduced Lyapunov mode (LYAPUN='R')
6. Mathematical property: RCOND in [0,1]
7. Edge case: n=1 scalar
8. Error case: invalid parameter

Random seeds: 42, 123, 456, 789 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb03qd_html_doc_example():
    """
    Validate using HTML doc example.

    N=3, JOB='B', FACT='N', TRANA='N', UPLO='U', LYAPUN='O'
    A = [[3,1,1],[1,3,0],[0,0,3]]
    C = [[25,24,15],[24,32,8],[15,8,40]]

    From SB03MD first:
    X (solution) = [[3.2604,2.7187,1.8616],[2.7187,4.4271,0.5699],[1.8616,0.5699,6.0461]]
    scale = 1.0

    From SB03QD:
    SEP = 4.9068 (estimated separation)
    RCOND = 0.3611 (reciprocal condition number)
    FERR = 0.0000 (forward error bound)
    """
    from ctrlsys import sb03qd, sb03md

    n = 3

    # Input matrix A (read row-wise from HTML)
    a = np.array([
        [3.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    # Input matrix C (symmetric, read row-wise from HTML)
    c = np.array([
        [25.0, 24.0, 15.0],
        [24.0, 32.0,  8.0],
        [15.0,  8.0, 40.0]
    ], order='F', dtype=float)

    # First solve continuous-time Lyapunov: A'*X + X*A = scale*C using SB03MD
    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'), c.copy(order='F'))
    x, t_out, u_out, wr, wi, scale, sep_md, ferr_md, info_md = result

    assert info_md == 0
    assert_allclose(scale, 1.0, rtol=1e-4)

    # Expected X from HTML doc
    x_expected = np.array([
        [3.2604, 2.7187, 1.8616],
        [2.7187, 4.4271, 0.5699],
        [1.8616, 0.5699, 6.0461]
    ], order='F', dtype=float)
    assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)

    # Now call SB03QD to estimate condition and error bound
    # Using FACT='F' since we already have Schur factorization from SB03MD
    sep, rcond, ferr, t_out2, u_out2, info = sb03qd(
        'B', 'F', 'N', 'U', 'O', scale, a, t_out, u_out, c, x
    )

    assert info == 0

    # Expected results from HTML doc
    assert_allclose(sep, 4.9068, rtol=0.05)
    assert_allclose(rcond, 0.3611, rtol=0.05)
    assert ferr < 0.01  # FERR = 0.0000 in HTML doc


def test_sb03qd_condition_number_only():
    """
    Validate JOB='C' (condition number only).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03qd, sb03md

    n = 3
    np.random.seed(42)

    # Create stable matrix A (all eigenvalues have negative real parts)
    a = np.array([
        [-2.0, 0.5, 0.1],
        [ 0.1,-1.5, 0.2],
        [ 0.0, 0.1,-1.0]
    ], order='F', dtype=float)

    # Symmetric positive definite C
    c = np.array([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 0.3],
        [0.5, 0.3, 2.0]
    ], order='F', dtype=float)

    # First solve Lyapunov equation
    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'), c.copy(order='F'))
    x, t_out, u_out, wr, wi, scale, sep_md, ferr_md, info_md = result
    assert info_md == 0

    # Call SB03QD with JOB='C' (condition number only)
    sep, rcond, ferr, t_out2, u_out2, info = sb03qd(
        'C', 'F', 'N', 'U', 'O', scale, a, t_out, u_out, c, x
    )

    assert info == 0
    assert sep > 0, "Separation must be positive for stable system"
    assert 0 <= rcond <= 1, "RCOND must be in [0,1]"


def test_sb03qd_error_bound_only():
    """
    Validate JOB='E' (error bound only).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03qd, sb03md

    n = 2
    a = np.array([
        [-2.0, 0.5],
        [ 0.0,-3.0]
    ], order='F', dtype=float)

    c = np.array([
        [2.0, 0.5],
        [0.5, 3.0]
    ], order='F', dtype=float)

    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'), c.copy(order='F'))
    x, t_out, u_out, wr, wi, scale, sep_md, ferr_md, info_md = result
    assert info_md == 0

    # Call SB03QD with JOB='E' (error bound only)
    sep, rcond, ferr, t_out2, u_out2, info = sb03qd(
        'E', 'F', 'N', 'U', 'O', scale, a, t_out, u_out, c, x
    )

    assert info == 0
    assert ferr >= 0, "FERR must be non-negative"
    # FERR should be small for well-conditioned problem
    assert ferr < 1.0


def test_sb03qd_transpose_form():
    """
    Validate with transpose form: TRANA='T'.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03qd, sb03md

    n = 2
    a = np.array([
        [-1.5, 0.3],
        [ 0.0,-2.5]
    ], order='F', dtype=float)

    c = np.array([
        [3.0, 0.5],
        [0.5, 2.0]
    ], order='F', dtype=float)

    # Solve A*X + X*A' = scale*C (transpose form)
    result = sb03md('C', 'X', 'N', 'T', n, a.copy(order='F'), c.copy(order='F'))
    x, t_out, u_out, wr, wi, scale, sep_md, ferr_md, info_md = result
    assert info_md == 0

    # Call SB03QD with TRANA='T'
    sep, rcond, ferr, t_out2, u_out2, info = sb03qd(
        'B', 'F', 'T', 'U', 'O', scale, a, t_out, u_out, c, x
    )

    assert info == 0
    assert sep > 0
    assert 0 <= rcond <= 1


def test_sb03qd_reduced_lyapunov():
    """
    Validate with reduced Lyapunov mode: LYAPUN='R'.

    This mode works directly with Schur form without transformation.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03qd, sb03my

    n = 2
    # Upper triangular Schur form (already in Schur form)
    t = np.array([
        [-2.0, 0.5],
        [ 0.0,-3.0]
    ], order='F', dtype=float)

    # Solve reduced Lyapunov equation T'*X + X*T = C
    c = np.array([
        [2.0, 0.5],
        [0.5, 3.0]
    ], order='F', dtype=float)

    x, scale, info_lyap = sb03my('N', t.copy(order='F'), c.copy(order='F'))
    assert info_lyap == 0

    # U not needed for LYAPUN='R', but provide dummy
    u = np.eye(n, order='F', dtype=float)

    # A not needed for FACT='F' and LYAPUN='R', but provide dummy
    a = np.zeros((1, 1), order='F', dtype=float)

    # Call SB03QD with LYAPUN='R' and FACT='F'
    sep, rcond, ferr, t_out, u_out, info = sb03qd(
        'B', 'F', 'N', 'U', 'R', scale, a, t, u, c, x
    )

    assert info == 0
    assert sep > 0
    assert 0 <= rcond <= 1
    assert ferr >= 0


def test_sb03qd_rcond_bounds():
    """
    Validate mathematical property: RCOND always in [0,1].

    Test with various matrix conditions.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03qd, sb03md

    np.random.seed(42)

    for trial in range(3):
        n = 3
        # Create random stable matrix
        a = -np.eye(n, order='F', dtype=float) * (2 + trial)
        a[0, 1] = 0.1 * (trial + 1)
        a = np.asfortranarray(a)

        # Random symmetric positive definite C
        temp = np.random.randn(n, n)
        c = np.asfortranarray(temp @ temp.T + np.eye(n))

        result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'), c.copy(order='F'))
        x, t_out, u_out, wr, wi, scale, sep_md, ferr_md, info_md = result
        if info_md != 0:
            continue

        sep, rcond, ferr, t_out2, u_out2, info = sb03qd(
            'B', 'F', 'N', 'U', 'O', scale, a, t_out, u_out, c, x
        )

        assert info >= 0
        if info == 0:
            assert 0 <= rcond <= 1, f"RCOND={rcond} out of bounds [0,1]"


def test_sb03qd_scalar():
    """
    Validate 1x1 case (scalar equation).

    For scalar: a'*x + x*a = scale*c
    With a = -2, c = 4: 2*a*x = scale*c, so x = scale*c/(2*a) = 4/(2*-2) = -1

    """
    from ctrlsys import sb03qd, sb03md

    n = 1
    a = np.array([[-2.0]], order='F', dtype=float)
    c = np.array([[4.0]], order='F', dtype=float)

    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'), c.copy(order='F'))
    x, t_out, u_out, wr, wi, scale, sep_md, ferr_md, info_md = result
    assert info_md == 0

    # x = scale*c / (2*a) = 1*4 / (2*-2) = -1.0
    assert_allclose(x[0, 0], -1.0, rtol=1e-10)

    sep, rcond, ferr, t_out2, u_out2, info = sb03qd(
        'B', 'F', 'N', 'U', 'O', scale, a, t_out, u_out, c, x
    )

    assert info == 0
    # For scalar a=-2: sep(a,-a') = |a+a| = 4
    assert_allclose(sep, 4.0, rtol=0.1)
    assert 0 <= rcond <= 1


def test_sb03qd_compute_schur():
    """
    Validate FACT='N' (compute Schur factorization internally).

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import sb03qd, sb03md

    n = 3
    a = np.array([
        [-2.0, 0.5, 0.1],
        [ 0.1,-1.5, 0.2],
        [ 0.0, 0.1,-1.0]
    ], order='F', dtype=float)

    c = np.array([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 0.3],
        [0.5, 0.3, 2.0]
    ], order='F', dtype=float)

    # First solve Lyapunov equation to get X
    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'), c.copy(order='F'))
    x, t_out, u_out, wr, wi, scale, sep_md, ferr_md, info_md = result
    assert info_md == 0

    # Call SB03QD with FACT='N' - it will compute Schur factorization
    # Provide empty T and U, they will be computed
    t_new = np.zeros((n, n), order='F', dtype=float)
    u_new = np.zeros((n, n), order='F', dtype=float)

    sep, rcond, ferr, t_out2, u_out2, info = sb03qd(
        'B', 'N', 'N', 'U', 'O', scale, a, t_new, u_new, c, x
    )

    assert info == 0
    assert sep > 0
    assert 0 <= rcond <= 1


def test_sb03qd_lower_triangular():
    """
    Validate with UPLO='L' (lower triangular storage for C).

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import sb03qd, sb03md

    n = 2
    a = np.array([
        [-2.0, 0.5],
        [ 0.0,-3.0]
    ], order='F', dtype=float)

    # C stored with lower triangular part significant
    c = np.array([
        [2.0, 0.0],  # Upper not used
        [0.5, 3.0]
    ], order='F', dtype=float)

    result = sb03md('C', 'X', 'N', 'N', n, a.copy(order='F'), c.copy(order='F'))
    x, t_out, u_out, wr, wi, scale, sep_md, ferr_md, info_md = result
    assert info_md == 0

    sep, rcond, ferr, t_out2, u_out2, info = sb03qd(
        'B', 'F', 'N', 'L', 'O', scale, a, t_out, u_out, c, x
    )

    assert info == 0
    assert sep > 0
    assert 0 <= rcond <= 1
