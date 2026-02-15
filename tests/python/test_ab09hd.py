"""
Tests for AB09HD - Stochastic balancing based model reduction.

AB09HD computes a reduced order model (Ar,Br,Cr,Dr) for an original
state-space representation (A,B,C,D) by using the stochastic balancing
approach with Balance & Truncate or Singular Perturbation Approximation
methods for the ALPHA-stable part of the system.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_ab09hd_html_doc_example():
    """
    Test using the example from SLICOT HTML documentation.

    Continuous-time system, balancing-free B&T, automatic order selection.
    N=7, M=2, P=3, BETA=1.0, TOL1=0.1, DICO='C', JOB='F', EQUIL='N', ORDSEL='A'

    NOTE: This test is skipped because eigenvalue validation fails. The HSV values
    are computed correctly, but the reduced state matrix has incorrect eigenvalues
    (including one unstable eigenvalue). This may indicate a bug in ab09ix or ab09hy
    for this specific input configuration.
    """
    from ctrlsys import ab09hd

    n, m, p = 7, 2, 3

    a = np.array([
        [-0.04165,  0.0000,  4.9200, -4.9200,  0.0000,  0.0000,  0.0000],
        [-5.2100, -12.500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  3.3300, -3.3300,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.5450,  0.0000,  0.0000,  0.0000, -0.5450,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  4.9200, -0.04165, 0.0000,  4.9200],
        [ 0.0000,  0.0000,  0.0000,  0.0000, -5.2100, -12.500,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  3.3300, -3.3300],
    ], order='F', dtype=np.float64)

    b = np.array([
        [ 0.0000,  0.0000],
        [12.500,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.0000, 12.500],
        [ 0.0000,  0.0000],
    ], order='F', dtype=np.float64)

    c = np.array([
        [1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000],
        [0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
    ], order='F', dtype=np.float64)

    d = np.array([
        [0.0000,  0.0000],
        [0.0000,  0.0000],
        [0.0000,  0.0000],
    ], order='F', dtype=np.float64)

    nr_in = 0
    alpha = 0.0
    beta = 1.0
    tol1 = 0.1
    tol2 = 0.0

    ar, br, cr, dr, nr_out, ns, hsv, iwarn, info = ab09hd(
        'C', 'F', 'N', 'A', n, m, p, nr_in, alpha, beta,
        a, b, c, d, tol1, tol2
    )

    assert info == 0, f"AB09HD failed with info={info}"
    assert nr_out == 5, f"Expected reduced order 5, got {nr_out}"
    assert ns == 7, f"Expected ns=7 (all stable), got {ns}"

    hsv_expected = np.array([0.8803, 0.8506, 0.8038, 0.4494, 0.3973, 0.0214, 0.0209])
    assert_allclose(hsv[:ns], hsv_expected, rtol=1e-3, atol=1e-4)

    ar_expected = np.array([
        [ 1.2729,  0.0000,  6.5947,  0.0000, -3.4229],
        [ 0.0000,  0.8169,  0.0000,  2.4821,  0.0000],
        [-2.9889,  0.0000, -2.9028,  0.0000, -0.3692],
        [ 0.0000, -3.3921,  0.0000, -3.1126,  0.0000],
        [-1.4767,  0.0000, -2.0339,  0.0000, -0.6107],
    ], order='F', dtype=np.float64)

    br_expected = np.array([
        [ 0.1331, -0.1331],
        [-0.0862, -0.0862],
        [-2.6777,  2.6777],
        [-3.5767, -3.5767],
        [-2.3033,  2.3033],
    ], order='F', dtype=np.float64)

    cr_expected = np.array([
        [-0.6907, -0.6882,  0.0779,  0.0958, -0.0038],
        [ 0.0676,  0.0000,  0.6532,  0.0000, -0.7522],
        [ 0.6907, -0.6882, -0.0779,  0.0958,  0.0038],
    ], order='F', dtype=np.float64)

    dr_expected = np.array([
        [0.0000,  0.0000],
        [0.0000,  0.0000],
        [0.0000,  0.0000],
    ], order='F', dtype=np.float64)

    # Validate eigenvalues of the reduced system match (eigenvalues are preserved
    # under state-space transformations, so this is the correct invariant to check)
    eig_computed = np.linalg.eigvals(ar[:nr_out, :nr_out])
    eig_expected = np.linalg.eigvals(ar_expected)

    # Sort by real part, then imaginary part
    def sort_key(ev):
        return (ev.real, ev.imag)

    eig_computed_sorted = np.array(sorted(eig_computed, key=sort_key))
    eig_expected_sorted = np.array(sorted(eig_expected, key=sort_key))

    assert_allclose(eig_computed_sorted.real, eig_expected_sorted.real, rtol=1e-3, atol=1e-4)
    assert_allclose(np.abs(eig_computed_sorted.imag), np.abs(eig_expected_sorted.imag), rtol=1e-3, atol=1e-4)

    # Validate that all eigenvalues are stable (negative real part for continuous-time)
    for ev in eig_computed:
        assert ev.real < 0, f"Reduced system should be stable but has eigenvalue {ev}"

    # Validate D matrix is preserved
    assert_allclose(dr[:p, :m], dr_expected, rtol=1e-3, atol=1e-4)


def test_ab09hd_fixed_order():
    """
    Test with fixed order selection (ORDSEL='F').

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab09hd

    np.random.seed(42)

    n, m, p = 4, 2, 2

    a = np.array([
        [-2.0,  0.5,  0.0,  0.0],
        [ 0.0, -1.5,  0.3,  0.0],
        [ 0.0,  0.0, -1.0,  0.2],
        [ 0.0,  0.0,  0.0, -0.5],
    ], order='F', dtype=np.float64)

    b = np.array([
        [1.0,  0.0],
        [0.5,  0.5],
        [0.0,  1.0],
        [0.2,  0.3],
    ], order='F', dtype=np.float64)

    c = np.array([
        [1.0,  0.5,  0.2,  0.1],
        [0.0,  1.0,  0.5,  0.2],
    ], order='F', dtype=np.float64)

    d = np.array([
        [0.1,  0.0],
        [0.0,  0.1],
    ], order='F', dtype=np.float64)

    nr_in = 2
    alpha = 0.0
    beta = 1.0
    tol1 = 0.0
    tol2 = 0.0

    ar, br, cr, dr, nr_out, ns, hsv, iwarn, info = ab09hd(
        'C', 'B', 'N', 'F', n, m, p, nr_in, alpha, beta,
        a, b, c, d, tol1, tol2
    )

    assert info == 0, f"AB09HD failed with info={info}"
    assert nr_out <= n, f"Reduced order {nr_out} should be <= {n}"
    assert ns == n, f"Expected ns={n} (all stable), got {ns}"
    assert len(hsv) >= ns

    for i in range(ns - 1):
        assert hsv[i] >= hsv[i + 1], "HSV should be in decreasing order"


def test_ab09hd_discrete_time():
    """
    Test discrete-time system (DICO='D').

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab09hd

    np.random.seed(123)

    n, m, p = 3, 1, 1

    a = np.array([
        [0.5,  0.1,  0.0],
        [0.0,  0.3,  0.1],
        [0.0,  0.0,  0.2],
    ], order='F', dtype=np.float64)

    b = np.array([
        [1.0],
        [0.5],
        [0.2],
    ], order='F', dtype=np.float64)

    c = np.array([
        [1.0,  0.5,  0.2],
    ], order='F', dtype=np.float64)

    d = np.array([
        [0.1],
    ], order='F', dtype=np.float64)

    nr_in = 0
    alpha = 0.9
    beta = 1.0
    tol1 = 0.0
    tol2 = 0.0

    ar, br, cr, dr, nr_out, ns, hsv, iwarn, info = ab09hd(
        'D', 'B', 'N', 'A', n, m, p, nr_in, alpha, beta,
        a, b, c, d, tol1, tol2
    )

    assert info == 0, f"AB09HD failed with info={info}"
    assert nr_out <= n
    assert ns == n


def test_ab09hd_spa_method():
    """
    Test Singular Perturbation Approximation method (JOB='S').

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import ab09hd

    np.random.seed(456)

    n, m, p = 4, 2, 2

    a = np.array([
        [-1.0,  0.2,  0.1,  0.0],
        [ 0.0, -2.0,  0.3,  0.0],
        [ 0.0,  0.0, -3.0,  0.2],
        [ 0.0,  0.0,  0.0, -4.0],
    ], order='F', dtype=np.float64)

    b = np.array([
        [1.0,  0.5],
        [0.0,  1.0],
        [0.5,  0.0],
        [0.2,  0.3],
    ], order='F', dtype=np.float64)

    c = np.array([
        [1.0,  0.0,  0.5,  0.2],
        [0.0,  1.0,  0.0,  0.3],
    ], order='F', dtype=np.float64)

    d = np.array([
        [0.1,  0.0],
        [0.0,  0.1],
    ], order='F', dtype=np.float64)

    nr_in = 2
    alpha = 0.0
    beta = 1.0
    tol1 = 0.0
    tol2 = 0.0

    ar, br, cr, dr, nr_out, ns, hsv, iwarn, info = ab09hd(
        'C', 'S', 'N', 'F', n, m, p, nr_in, alpha, beta,
        a, b, c, d, tol1, tol2
    )

    assert info == 0, f"AB09HD failed with info={info}"
    assert nr_out <= n
    assert ns == n


def test_ab09hd_with_equilibration():
    """
    Test with equilibration enabled (EQUIL='S').

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import ab09hd

    np.random.seed(789)

    n, m, p = 3, 2, 2

    a = np.array([
        [-1.0,  100.0,  0.0],
        [ 0.0,   -2.0,  0.1],
        [ 0.0,    0.0, -3.0],
    ], order='F', dtype=np.float64)

    b = np.array([
        [100.0,  0.0],
        [  0.0,  1.0],
        [  0.0,  0.1],
    ], order='F', dtype=np.float64)

    c = np.array([
        [0.01,  1.0,  0.0],
        [0.0,   0.0,  1.0],
    ], order='F', dtype=np.float64)

    d = np.array([
        [0.1,  0.0],
        [0.0,  0.1],
    ], order='F', dtype=np.float64)

    nr_in = 0
    alpha = 0.0
    beta = 1.0
    tol1 = 0.0
    tol2 = 0.0

    ar, br, cr, dr, nr_out, ns, hsv, iwarn, info = ab09hd(
        'C', 'B', 'S', 'A', n, m, p, nr_in, alpha, beta,
        a, b, c, d, tol1, tol2
    )

    assert info == 0, f"AB09HD failed with info={info}"
    assert nr_out <= n
    assert ns == n


def test_ab09hd_beta_zero():
    """
    Test pure relative error method (BETA=0).
    Requires rank(D) = P.

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import ab09hd

    np.random.seed(999)

    n, m, p = 3, 2, 2

    a = np.array([
        [-1.0,  0.2,  0.0],
        [ 0.0, -2.0,  0.1],
        [ 0.0,  0.0, -3.0],
    ], order='F', dtype=np.float64)

    b = np.array([
        [1.0,  0.0],
        [0.0,  1.0],
        [0.5,  0.5],
    ], order='F', dtype=np.float64)

    c = np.array([
        [1.0,  0.5,  0.2],
        [0.0,  1.0,  0.5],
    ], order='F', dtype=np.float64)

    d = np.array([
        [1.0,  0.0],
        [0.0,  1.0],
    ], order='F', dtype=np.float64)

    nr_in = 0
    alpha = 0.0
    beta = 0.0
    tol1 = 0.0
    tol2 = 0.0

    ar, br, cr, dr, nr_out, ns, hsv, iwarn, info = ab09hd(
        'C', 'B', 'N', 'A', n, m, p, nr_in, alpha, beta,
        a, b, c, d, tol1, tol2
    )

    assert info == 0, f"AB09HD failed with info={info}"
    assert nr_out <= n
    assert ns == n


def test_ab09hd_zero_dimension():
    """
    Test quick return for zero dimensions.
    """
    from ctrlsys import ab09hd

    n, m, p = 0, 2, 2

    a = np.zeros((1, 1), order='F', dtype=np.float64)
    b = np.zeros((1, m), order='F', dtype=np.float64)
    c = np.zeros((p, 1), order='F', dtype=np.float64)
    d = np.zeros((p, m), order='F', dtype=np.float64)

    nr_in = 0
    alpha = 0.0
    beta = 1.0
    tol1 = 0.0
    tol2 = 0.0

    ar, br, cr, dr, nr_out, ns, hsv, iwarn, info = ab09hd(
        'C', 'B', 'N', 'A', n, m, p, nr_in, alpha, beta,
        a, b, c, d, tol1, tol2
    )

    assert info == 0
    assert nr_out == 0
    assert ns == 0


def test_ab09hd_invalid_dico():
    """
    Test error handling for invalid DICO parameter.
    """
    from ctrlsys import ab09hd

    n, m, p = 2, 1, 1

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.5]], order='F', dtype=np.float64)
    c = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    d = np.array([[0.1]], order='F', dtype=np.float64)

    with pytest.raises(ValueError):
        ab09hd('X', 'B', 'N', 'A', n, m, p, 0, 0.0, 1.0, a, b, c, d, 0.0, 0.0)


def test_ab09hd_invalid_job():
    """
    Test error handling for invalid JOB parameter.
    """
    from ctrlsys import ab09hd

    n, m, p = 2, 1, 1

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.5]], order='F', dtype=np.float64)
    c = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    d = np.array([[0.1]], order='F', dtype=np.float64)

    with pytest.raises(ValueError):
        ab09hd('C', 'X', 'N', 'A', n, m, p, 0, 0.0, 1.0, a, b, c, d, 0.0, 0.0)


def test_ab09hd_hsv_decreasing():
    """
    Validate mathematical property: Hankel singular values are in decreasing order.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import ab09hd

    np.random.seed(111)

    n, m, p = 5, 2, 2

    a = np.diag([-0.5, -1.0, -1.5, -2.0, -2.5]).astype(np.float64, order='F')
    a[0, 1] = 0.1
    a[1, 2] = 0.2
    a[2, 3] = 0.1
    a[3, 4] = 0.15
    a = np.asfortranarray(a)

    b = np.array([
        [1.0,  0.0],
        [0.5,  0.5],
        [0.0,  1.0],
        [0.3,  0.2],
        [0.1,  0.4],
    ], order='F', dtype=np.float64)

    c = np.array([
        [1.0,  0.5,  0.2,  0.1,  0.05],
        [0.0,  1.0,  0.5,  0.2,  0.1],
    ], order='F', dtype=np.float64)

    d = np.array([
        [0.1,  0.0],
        [0.0,  0.1],
    ], order='F', dtype=np.float64)

    nr_in = 0
    alpha = 0.0
    beta = 1.0
    tol1 = 0.0
    tol2 = 0.0

    ar, br, cr, dr, nr_out, ns, hsv, iwarn, info = ab09hd(
        'C', 'B', 'N', 'A', n, m, p, nr_in, alpha, beta,
        a, b, c, d, tol1, tol2
    )

    assert info == 0, f"AB09HD failed with info={info}"

    for i in range(ns - 1):
        assert hsv[i] >= hsv[i + 1] - 1e-14, \
            f"HSV[{i}]={hsv[i]} should be >= HSV[{i+1}]={hsv[i+1]}"


def test_ab09hd_reduced_system_stable():
    """
    Validate mathematical property: reduced system eigenvalues remain stable.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import ab09hd

    np.random.seed(222)

    n, m, p = 4, 2, 2

    a = np.array([
        [-1.0,  0.2,  0.0,  0.0],
        [ 0.0, -1.5,  0.1,  0.0],
        [ 0.0,  0.0, -2.0,  0.1],
        [ 0.0,  0.0,  0.0, -2.5],
    ], order='F', dtype=np.float64)

    b = np.array([
        [1.0,  0.0],
        [0.5,  0.5],
        [0.0,  1.0],
        [0.2,  0.3],
    ], order='F', dtype=np.float64)

    c = np.array([
        [1.0,  0.5,  0.2,  0.1],
        [0.0,  1.0,  0.5,  0.2],
    ], order='F', dtype=np.float64)

    d = np.array([
        [0.1,  0.0],
        [0.0,  0.1],
    ], order='F', dtype=np.float64)

    nr_in = 2
    alpha = 0.0
    beta = 1.0
    tol1 = 0.0
    tol2 = 0.0

    ar, br, cr, dr, nr_out, ns, hsv, iwarn, info = ab09hd(
        'C', 'B', 'N', 'F', n, m, p, nr_in, alpha, beta,
        a, b, c, d, tol1, tol2
    )

    assert info == 0, f"AB09HD failed with info={info}"

    if nr_out > 0:
        ar_reduced = ar[:nr_out, :nr_out]
        eigvals = np.linalg.eigvals(ar_reduced)

        for ev in eigvals:
            assert ev.real < 0, \
                f"Reduced system should be stable but has eigenvalue {ev}"
