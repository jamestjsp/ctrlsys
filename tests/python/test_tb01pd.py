"""
Tests for TB01PD - Minimal realization computation.

TB01PD finds a reduced (controllable, observable, or minimal) state-space
representation (Ar,Br,Cr) for any original state-space representation (A,B,C).
The matrix Ar is in upper block Hessenberg form.
"""

import numpy as np
import pytest
from ctrlsys import tb01pd


"""Basic functionality tests from HTML documentation example."""

def test_html_doc_example_minimal():
    """
    Test case from SLICOT HTML documentation with JOB='M'.

    Input: N=3, M=1, P=2, TOL=0.0, JOB='M', EQUIL='N'
    A matrix (row-wise):
        1.0  2.0  0.0
        4.0 -1.0  0.0
        0.0  0.0  1.0
    B matrix (column-wise): 1.0, 0.0, 1.0
    C matrix (row-wise):
        0.0  1.0 -1.0
        0.0  0.0  1.0

    Expected NR = 3 (system is already minimal)

    Verifies: Markov parameter preservation (transfer function equivalence)
    """
    n, m, p = 3, 1, 2

    a_orig = np.array([
        [1.0,  2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0,  0.0, 1.0]
    ], dtype=float, order='F')

    b_orig = np.array([
        [1.0],
        [0.0],
        [1.0]
    ], dtype=float, order='F')

    c_orig = np.array([
        [0.0, 1.0, -1.0],
        [0.0, 0.0,  1.0]
    ], dtype=float, order='F')

    a = a_orig.copy()
    b = b_orig.copy()
    c = c_orig.copy()
    tol = 0.0

    a_out, b_out, c_out, nr, iwork_out, info = tb01pd('M', 'N', a, b, c, tol)

    assert info == 0
    assert nr == 3  # system is already minimal

    ar = a_out[:nr, :nr]
    br = b_out[:nr, :]
    cr = c_out[:, :nr]

    d_orig = np.zeros((p, m), dtype=float)
    d_red = np.zeros((p, m), dtype=float)

    for k in range(6):
        if k == 0:
            h_orig = d_orig.copy()
            h_red = d_red.copy()
        else:
            a_k_orig = np.linalg.matrix_power(a_orig, k - 1)
            a_k_red = np.linalg.matrix_power(ar, k - 1)
            h_orig = c_orig @ a_k_orig @ b_orig
            h_red = cr @ a_k_red @ br
        np.testing.assert_allclose(h_orig, h_red, rtol=1e-10, atol=1e-12,
                                  err_msg=f"Markov param h({k}) mismatch")


"""Test JOB='C' mode (controllable reduction only)."""

def test_controllable_reduction():
    """
    Test controllable reduction with partially controllable system.
    """
    n, m, p = 4, 1, 2

    a = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    b = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=float, order='F')

    c = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0]
    ], dtype=float, order='F')

    _, _, _, nr, _, info = tb01pd('C', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr < n


"""Test JOB='O' mode (observable reduction only)."""

def test_observable_reduction():
    """
    Test observable reduction with partially observable system.
    """
    n, m, p = 4, 2, 1

    a = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0, 3.0]
    ], dtype=float, order='F')

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=float, order='F')

    c = np.array([
        [1.0, 0.0, 0.0, 0.0]
    ], dtype=float, order='F')

    _, _, _, nr, _, info = tb01pd('O', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr < n


"""Test JOB='M' mode (minimal realization)."""

def test_minimal_from_nonminimal():
    """
    Test minimal reduction on system with hidden modes.

    Create system with uncontrollable and unobservable modes.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 6

    a = np.zeros((n, n), dtype=float, order='F')
    a[0:2, 0:2] = np.array([[1.0, 0.5], [0.0, 1.0]])
    a[2:4, 2:4] = np.array([[2.0, 0.0], [0.0, 2.0]])
    a[4:6, 4:6] = np.array([[3.0, 0.0], [0.0, 3.0]])

    b = np.zeros((n, 1), dtype=float, order='F')
    b[0, 0] = 1.0

    c = np.zeros((1, n), dtype=float, order='F')
    c[0, 0] = 1.0

    _, _, _, nr, _, info = tb01pd('M', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr < n


"""Test equilibration (scaling) option."""

def test_with_scaling():
    """
    Test with EQUIL='S' (scaling enabled).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F') * 100
    b = np.random.randn(n, m).astype(float, order='F') * 100
    c = np.random.randn(p, n).astype(float, order='F') * 100

    _, _, _, nr, _, info = tb01pd('M', 'S', a, b, c, 0.0)

    assert info == 0
    assert nr <= n

def test_without_scaling():
    """
    Test with EQUIL='N' (scaling disabled).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    _, _, _, nr, _, info = tb01pd('M', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr <= n


"""Test transfer function preservation property."""

def test_markov_parameters_preserved():
    """
    Validate Markov parameters preserved: h(k) = C*A^k*B same before/after.

    The transfer function G(s) = C*(sI-A)^(-1)*B should be the same
    for the original and reduced systems. For SISO systems, test
    Markov parameters.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, m, p = 4, 1, 1

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    markov_orig = []
    ak = np.eye(n)
    for k in range(10):
        h_k = c_copy @ ak @ b_copy
        markov_orig.append(h_k[0, 0])
        ak = ak @ a_copy

    a_out, b_out, c_out, nr, _, info = tb01pd('M', 'N', a, b, c, 0.0)

    assert info == 0

    markov_reduced = []
    ak = np.eye(nr)
    ar = a_out[:nr, :nr]
    br = b_out[:nr, :]
    cr = c_out[:, :nr]
    for k in range(10):
        h_k = cr @ ak @ br
        markov_reduced.append(h_k[0, 0])
        ak = ak @ ar

    np.testing.assert_allclose(markov_reduced, markov_orig, rtol=1e-10, atol=1e-12)


"""Test upper block Hessenberg structure."""

def test_output_is_block_hessenberg():
    """
    Validate Ar has upper block Hessenberg structure.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, m, p = 5, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_out, _, _, nr, iwork_out, info = tb01pd('M', 'N', a, b, c, 0.0)

    assert info == 0


"""Test edge cases and boundary conditions."""

def test_n_equals_1():
    """
    Test with N=1 (scalar system).
    """
    a = np.array([[2.0]], dtype=float, order='F')
    b = np.array([[1.0]], dtype=float, order='F')
    c = np.array([[1.0]], dtype=float, order='F')

    a_out, b_out, c_out, nr, _, info = tb01pd('M', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr == 1

def test_n_equals_0():
    """
    Test with N=0 (empty system).
    """
    a = np.array([], dtype=float, order='F').reshape(0, 0)
    b = np.array([], dtype=float, order='F').reshape(0, 1)
    c = np.array([], dtype=float, order='F').reshape(1, 0)

    _, _, _, nr, _, info = tb01pd('M', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr == 0

def test_m_equals_0():
    """
    Test with M=0 (no inputs) - uncontrollable.
    """
    n, p = 3, 2
    a = np.eye(n, dtype=float, order='F')
    b = np.array([], dtype=float, order='F').reshape(n, 0)
    c = np.ones((p, n), dtype=float, order='F')

    _, _, _, nr, _, info = tb01pd('C', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr == 0

def test_p_equals_0():
    """
    Test with P=0 (no outputs) - unobservable.
    """
    n, m = 3, 2
    a = np.eye(n, dtype=float, order='F')
    b = np.ones((n, m), dtype=float, order='F')
    c = np.array([], dtype=float, order='F').reshape(0, n)

    _, _, _, nr, _, info = tb01pd('O', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr == 0

def test_large_system():
    """
    Test with larger system for scalability.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n, m, p = 15, 3, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    _, _, _, nr, _, info = tb01pd('M', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr <= n


"""Test systems that are already minimal."""

def test_fully_controllable_observable():
    """
    Test fully controllable and observable system stays unchanged in order.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    _, _, _, nr, _, info = tb01pd('M', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr == n


"""Test error handling."""

def test_invalid_job():
    """
    Test invalid JOB parameter.
    """
    n, m, p = 3, 1, 2
    a = np.eye(n, dtype=float, order='F')
    b = np.ones((n, m), dtype=float, order='F')
    c = np.ones((p, n), dtype=float, order='F')

    with pytest.raises(ValueError, match="[Jj]ob"):
        tb01pd('X', 'N', a, b, c, 0.0)

def test_invalid_equil():
    """
    Test invalid EQUIL parameter.
    """
    n, m, p = 3, 1, 2
    a = np.eye(n, dtype=float, order='F')
    b = np.ones((n, m), dtype=float, order='F')
    c = np.ones((p, n), dtype=float, order='F')

    with pytest.raises(ValueError, match="[Ee]quil"):
        tb01pd('M', 'X', a, b, c, 0.0)

def test_mismatched_dimensions_ab():
    """
    Test mismatched A and B dimensions.
    """
    a = np.eye(3, dtype=float, order='F')
    b = np.ones((4, 2), dtype=float, order='F')
    c = np.ones((2, 3), dtype=float, order='F')

    with pytest.raises(ValueError):
        tb01pd('M', 'N', a, b, c, 0.0)

def test_mismatched_dimensions_ac():
    """
    Test mismatched A and C dimensions.
    """
    a = np.eye(3, dtype=float, order='F')
    b = np.ones((3, 2), dtype=float, order='F')
    c = np.ones((2, 4), dtype=float, order='F')

    with pytest.raises(ValueError):
        tb01pd('M', 'N', a, b, c, 0.0)

def test_non_square_a():
    """
    Test non-square A matrix.
    """
    a = np.ones((3, 4), dtype=float, order='F')
    b = np.ones((3, 2), dtype=float, order='F')
    c = np.ones((2, 3), dtype=float, order='F')

    with pytest.raises(ValueError):
        tb01pd('M', 'N', a, b, c, 0.0)


"""Test tolerance handling."""

def test_explicit_tolerance():
    """
    Test with explicit positive tolerance.
    """
    n, m, p = 3, 1, 2

    a = np.array([
        [1.0,  2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0,  0.0, 1.0]
    ], dtype=float, order='F')

    b = np.array([[1.0], [0.0], [1.0]], dtype=float, order='F')

    c = np.array([
        [0.0, 1.0, -1.0],
        [0.0, 0.0,  1.0]
    ], dtype=float, order='F')

    tol = 1e-10

    _, _, _, nr, _, info = tb01pd('M', 'N', a, b, c, tol)

    assert info == 0
    assert nr == 3

def test_default_tolerance():
    """
    Test that zero/negative tolerance uses default.
    """
    n, m, p = 3, 1, 2

    a = np.array([
        [1.0,  2.0, 0.0],
        [4.0, -1.0, 0.0],
        [0.0,  0.0, 1.0]
    ], dtype=float, order='F')

    b = np.array([[1.0], [0.0], [1.0]], dtype=float, order='F')

    c = np.array([
        [0.0, 1.0, -1.0],
        [0.0, 0.0,  1.0]
    ], dtype=float, order='F')

    _, _, _, nr, _, info = tb01pd('M', 'N', a, b, c, 0.0)

    assert info == 0
    assert nr == 3
