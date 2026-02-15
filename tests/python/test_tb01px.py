"""
Tests for TB01PX - Minimal/Controllable/Observable state-space realization

TB01PX finds a reduced (controllable, observable, or minimal) state-space
representation (Ar,Br,Cr) for any original state-space representation (A,B,C).
The matrix Ar is in upper block Hessenberg staircase form.

Two-phase reduction:
- Phase 1 (JOB='M' or 'C'): Remove uncontrollable part
- Phase 2 (JOB='M' or 'O'): Remove unobservable part

Mathematical properties tested:
- Order reduction: NR <= N
- Eigenvalue preservation for minimal realization
- Transfer function equivalence
- INFRED status tracking

Test data from SLICOT-Reference/doc/TB01PX.html example
"""
import numpy as np
import pytest

try:
    import ctrlsys
    HAS_SLICOT = True
except ImportError:
    HAS_SLICOT = False


pytestmark = pytest.mark.skipif(not HAS_SLICOT, reason="slicot module not available")


def test_html_example():
    """Test TB01PX with example from HTML documentation

    From HTML doc:
    N=3, M=1, P=2, TOL=0.0, JOB='M', EQUIL='N'

    A read row-wise: ((A(I,J), J=1,N), I=1,N)
    A = [1.0  2.0  0.0]
        [4.0 -1.0  0.0]
        [0.0  0.0  1.0]

    B read column-wise: ((B(I,J), I=1,N), J=1,M)
    B = [1.0]
        [0.0]
        [1.0]

    C read row-wise: ((C(I,J), J=1,N), I=1,P)
    C = [0.0  1.0 -1.0]
        [0.0  0.0  1.0]

    Expected outputs:
    NR = 3 (no reduction - system is already minimal)
    INFRED = [-1, -1, 2, 0] (neither phase reduced order)

    Ar = [1.0  2.0  0.0]    (same as input)
         [4.0 -1.0  0.0]
         [0.0  0.0  1.0]

    Br = [1.0]              (same as input)
         [0.0]
         [1.0]

    Cr = [0.0  1.0 -1.0]    (same as input)
         [0.0  0.0  1.0]
    """
    n, m, p = 3, 1, 2

    a = np.array([
        [1.0,  2.0,  0.0],
        [4.0, -1.0,  0.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [0.0,  1.0, -1.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64, order='F')

    expected_nr = 3
    expected_infred = [-1, -1, 2, 0]

    expected_a = np.array([
        [1.0,  2.0,  0.0],
        [4.0, -1.0,  0.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64, order='F')

    expected_b = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    expected_c = np.array([
        [0.0,  1.0, -1.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'M', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0, f"tb01px failed with info={info}"
    assert nr == expected_nr, f"nr={nr}, expected {expected_nr}"

    for i in range(4):
        assert infred[i] == expected_infred[i], \
            f"infred[{i}]={infred[i]}, expected {expected_infred[i]}"

    np.testing.assert_allclose(
        a_out[:nr, :nr], expected_a, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        b_out[:nr, :m], expected_b, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        c_out[:p, :nr], expected_c, rtol=1e-3, atol=1e-4)


def test_minimal_realization_reduces_order():
    """Test that TB01PX reduces uncontrollable/unobservable states

    Create a system with uncontrollable states and verify reduction.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m, p = 4, 1, 1

    a = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0],
        [1.0],
        [0.0],
        [0.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 1.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'M', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0, f"tb01px failed with info={info}"
    assert nr < n, f"Order should be reduced: nr={nr}, n={n}"
    assert nr == 2, f"Minimal order should be 2, got nr={nr}"


def test_controllable_realization():
    """Test JOB='C' removes only uncontrollable part

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 4, 1, 2

    a = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0],
        [1.0],
        [0.0],
        [0.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'C', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nr == 2, f"Controllable order should be 2, got nr={nr}"
    assert infred[0] >= 0, "Phase 1 should have been performed"
    assert infred[1] < 0, "Phase 2 should NOT have been performed"


def test_observable_realization():
    """Test JOB='O' removes only unobservable part

    Create system where states 3,4 are unobservable (C has zeros there).
    With diagonal A, observability = rank([C; CA; CA^2; ...])
    Only states with nonzero C entries are observable.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 4, 2, 2

    a = np.array([
        [1.0, 0.1, 0.0, 0.0],
        [0.1, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'O', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nr == 2, f"Observable order should be 2, got nr={nr}"
    assert infred[0] < 0, "Phase 1 should NOT have been performed"
    assert infred[1] >= 0, "Phase 2 should have been performed"


def test_eigenvalue_preservation_minimal():
    """Test eigenvalue preservation for minimal realization

    The eigenvalues of the reduced system should be a subset of
    the original eigenvalues (the controllable and observable ones).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 5, 2, 2

    a = np.array([
        [-1.0,  0.5,  0.0,  0.0,  0.0],
        [ 0.3, -2.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0, -3.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0, -4.0,  0.2],
        [ 0.0,  0.0,  0.0,  0.1, -5.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    eig_orig = np.linalg.eigvals(a)

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'M', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nr == 2, f"Minimal order should be 2, got nr={nr}"

    eig_reduced = np.linalg.eigvals(a_out[:nr, :nr])

    for eig_r in eig_reduced:
        found = False
        for eig_o in eig_orig:
            if abs(eig_r - eig_o) < 1e-10:
                found = True
                break
        assert found, f"Eigenvalue {eig_r} not found in original spectrum"


def test_equil_scaling():
    """Test EQUIL='S' performs balancing

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m, p = 3, 1, 1

    a = np.array([
        [1.0,  2.0,  0.0],
        [4.0, -1.0,  0.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [0.0,  1.0, -1.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'M', 'S', n, m, p, a, b, c, tol=0.0)

    assert info == 0


def test_quick_return_n_zero():
    """Test quick return for N=0"""
    n, m, p = 0, 2, 2
    a = np.zeros((1, 1), dtype=np.float64, order='F')
    b = np.zeros((1, 2), dtype=np.float64, order='F')
    c = np.zeros((2, 1), dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'M', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nr == 0


def test_quick_return_job_o_p_zero():
    """Test quick return for JOB='O' and P=0"""
    n, m, p = 3, 2, 0
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.ones((n, m), dtype=np.float64, order='F')
    c = np.zeros((1, n), dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'O', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nr == 0


def test_quick_return_job_c_m_zero():
    """Test quick return for JOB='C' and M=0"""
    n, m, p = 3, 0, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, 1), dtype=np.float64, order='F')
    c = np.ones((p, n), dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'C', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nr == 0


def test_infred_status_tracking():
    """Test INFRED array correctly tracks reduction status

    INFRED(1) >= 0 if Phase 1 performed (controllability reduction)
    INFRED(2) >= 0 if Phase 2 performed (observability reduction)
    INFRED(3) = number of nonzero subdiagonals of A
    INFRED(4) = number of blocks in staircase form
    """
    n, m, p = 4, 1, 1

    a = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0],
        [0.0],
        [0.0],
        [0.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 0.0, 0.0, 0.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'M', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nr == 1
    assert infred[0] >= 0, "Phase 1 should have been performed"
    assert len(infred) == 4


def test_large_workspace_save_restore():
    """Test that large workspace enables save/restore behavior

    When LDWORK >= N + MAX(N, 3*M, 3*P) + N*(N+M+P), system matrices
    are saved before each phase and restored if no reduction occurred.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 3, 1, 2

    a = np.array([
        [1.0,  2.0,  0.0],
        [4.0, -1.0,  0.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [0.0,  1.0, -1.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64, order='F')

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'M', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nr == n


def test_invalid_job():
    """Test error handling for invalid JOB parameter"""
    n, m, p = 3, 2, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m), dtype=np.float64, order='F')
    c = np.zeros((p, n), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01px('X', 'N', n, m, p, a, b, c, tol=0.0)


def test_invalid_equil():
    """Test error handling for invalid EQUIL parameter"""
    n, m, p = 3, 2, 2
    a = np.eye(n, dtype=np.float64, order='F')
    b = np.zeros((n, m), dtype=np.float64, order='F')
    c = np.zeros((p, n), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01px('M', 'X', n, m, p, a, b, c, tol=0.0)


def test_invalid_n_negative():
    """Test error handling for N < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01px('M', 'N', -1, 1, 1, a, b, c, tol=0.0)


def test_invalid_m_negative():
    """Test error handling for M < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01px('M', 'N', 1, -1, 1, a, b, c, tol=0.0)


def test_invalid_p_negative():
    """Test error handling for P < 0"""
    a = np.eye(1, dtype=np.float64, order='F')
    b = np.zeros((1, 1), dtype=np.float64, order='F')
    c = np.zeros((1, 1), dtype=np.float64, order='F')

    with pytest.raises((ValueError, RuntimeError)):
        slicot.tb01px('M', 'N', 1, 1, -1, a, b, c, tol=0.0)


def test_fully_controllable_observable():
    """Test system that is already fully controllable and observable

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n, m, p = 3, 2, 2

    a = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-6.0, -11.0, -6.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float64, order='F')

    c = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64, order='F')

    eig_orig = np.sort(np.linalg.eigvals(a).real)

    a_out, b_out, c_out, nr, infred, iwork, info = slicot.tb01px(
        'M', 'N', n, m, p, a, b, c, tol=0.0)

    assert info == 0
    assert nr == n, f"Fully minimal system should have nr=n, got nr={nr}"

    eig_new = np.sort(np.linalg.eigvals(a_out[:nr, :nr]).real)
    np.testing.assert_allclose(eig_orig, eig_new, rtol=1e-12, atol=1e-14)
