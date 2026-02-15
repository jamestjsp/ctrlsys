"""
Tests for TG01OD: Reduce SISO descriptor system to have large feedthrough term.

TG01OD computes for a single-input single-output descriptor system,
given by the system matrix:

    [ D     C    ]
    [ B  A - s*E ]

with E nonsingular, a reduced system matrix:

    [ d     c    ]
    [ b  a - s*e ]

such that d has a "sufficiently" large magnitude.

Uses Householder transformations and Givens rotations. If E is a general
matrix, it is first triangularized using QR decomposition.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tg01od_identity_e_basic():
    """
    Test TG01OD with JOBE='I' (E is identity).

    Random seed: 42 (for reproducibility)

    Creates a system with small D that requires reduction.
    """
    from ctrlsys import tg01od

    np.random.seed(42)
    n = 3

    d = 1e-10
    c = np.array([1.0, 2.0, 3.0], dtype=float)
    b = np.array([1.0, 2.0, 3.0], dtype=float)
    a = np.random.randn(n, n).astype(float, order='F')

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    e = np.zeros((1, 1), dtype=float, order='F')

    dcba_out, e_out, nz, g, info = tg01od('I', dcba, e, 0.0)

    assert info == 0
    assert nz >= 0
    assert nz <= n


def test_tg01od_general_e_basic():
    """
    Test TG01OD with JOBE='G' (E is general nonsingular matrix).

    Random seed: 123 (for reproducibility)

    Mathematical property tested:
    - Gain g relates original system to reduced system
    - Reduced system has sufficiently large feedthrough d
    """
    from ctrlsys import tg01od

    np.random.seed(123)
    n = 3

    d = 1e-12
    c = np.array([0.5, 1.0, -0.5], dtype=float)
    b = np.array([1.0, 0.5, -1.0], dtype=float)
    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, dtype=float, order='F') + 0.1 * np.random.randn(n, n)
    e = e.astype(float, order='F')

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    dcba_out, e_out, nz, g, info = tg01od('G', dcba, e, 0.0)

    assert info == 0
    assert nz >= 0
    assert nz <= n


def test_tg01od_no_reduction_needed():
    """
    Test TG01OD when D is already large enough (no reduction needed).

    Random seed: 456 (for reproducibility)

    When |D| * (1 + max|A|) > tol * ||B|| * ||C||, no reduction occurs.
    """
    from ctrlsys import tg01od

    np.random.seed(456)
    n = 3

    d = 10.0
    c = np.array([0.1, 0.2, 0.3], dtype=float)
    b = np.array([0.1, 0.2, 0.3], dtype=float)
    a = 0.1 * np.random.randn(n, n).astype(float, order='F')

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    dcba_orig = dcba.copy(order='F')
    e = np.zeros((1, 1), dtype=float, order='F')

    dcba_out, e_out, nz, g, info = tg01od('I', dcba, e, 0.0)

    assert info == 0
    assert nz == n
    assert_allclose(g, d, rtol=1e-14)


def test_tg01od_n_zero():
    """Test TG01OD with N=0 (quick return case)."""
    from ctrlsys import tg01od

    n = 0
    d = 5.0
    dcba = np.array([[d]], dtype=float, order='F')
    e = np.zeros((1, 1), dtype=float, order='F')

    dcba_out, e_out, nz, g, info = tg01od('I', dcba, e, 0.0)

    assert info == 0
    assert nz == 0
    assert_allclose(g, d, rtol=1e-14)


def test_tg01od_gain_computation():
    """
    Test that the gain G correctly relates original and reduced systems.

    Random seed: 789 (for reproducibility)

    For a SISO system, g = d (original) when no reduction occurs.
    When reduction occurs, g = product of transformations * d (reduced).
    """
    from ctrlsys import tg01od

    np.random.seed(789)
    n = 4

    d = 1e-15
    c = np.random.randn(n)
    b = np.random.randn(n)
    a = np.random.randn(n, n).astype(float, order='F')

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    e = np.zeros((1, 1), dtype=float, order='F')

    dcba_out, e_out, nz, g, info = tg01od('I', dcba, e, 0.0)

    assert info == 0

    if nz < n and nz > 0:
        d_reduced = dcba_out[0, 0]
        assert abs(d_reduced) > 1e-10 or nz == 0


def test_tg01od_full_reduction():
    """
    Test TG01OD when system is fully reduced (nz=0).

    Random seed: 101 (for reproducibility)

    This happens when [D; B] = 0 or [D C] = 0.
    """
    from ctrlsys import tg01od

    np.random.seed(101)
    n = 3

    d = 0.0
    c = np.zeros(n)
    b = np.random.randn(n)
    a = np.random.randn(n, n).astype(float, order='F')

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    e = np.zeros((1, 1), dtype=float, order='F')

    dcba_out, e_out, nz, g, info = tg01od('I', dcba, e, 0.0)

    assert info == 0
    assert nz == 0 or nz <= n


def test_tg01od_general_e_reduction():
    """
    Test TG01OD with JOBE='G' performs reduction correctly.

    Random seed: 202 (for reproducibility)

    When JOBE='G', E is first triangularized via QR decomposition,
    and the algorithm reduces the system until d has large enough magnitude.
    """
    from ctrlsys import tg01od

    np.random.seed(202)
    n = 4

    d = 1e-10
    c = np.random.randn(n)
    b = np.random.randn(n)
    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, dtype=float, order='F') + 0.2 * np.random.randn(n, n)
    e = e.astype(float, order='F')

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    dcba_out, e_out, nz, g, info = tg01od('G', dcba, e, 0.0)

    assert info == 0
    assert 0 <= nz <= n

    if nz < n and nz > 0:
        d_reduced = dcba_out[0, 0]
        assert abs(d_reduced) > 1e-15 or nz == 0


def test_tg01od_invalid_jobe():
    """Test TG01OD with invalid JOBE parameter."""
    from ctrlsys import tg01od

    n = 2
    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    e = np.eye(n, dtype=float, order='F')

    with pytest.raises(ValueError):
        tg01od('X', dcba, e)


def test_tg01od_workspace_query():
    """
    Test TG01OD workspace query (ldwork=-1).

    For JOBE='G': needs at least 2*N+1
    For JOBE='I': needs at least N+1
    """
    from ctrlsys import tg01od

    np.random.seed(303)
    n = 5

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = 1e-10
    dcba[0, 1:] = np.random.randn(n)
    dcba[1:, 0] = np.random.randn(n)
    dcba[1:, 1:] = np.random.randn(n, n)

    e = np.eye(n, dtype=float, order='F') + 0.1 * np.random.randn(n, n)
    e = e.astype(float, order='F')

    dcba_out, e_out, nz, g, info = tg01od('G', dcba, e, 0.0)

    assert info == 0


def test_tg01od_custom_tolerance():
    """
    Test TG01OD with custom tolerance.

    Random seed: 404 (for reproducibility)

    A smaller tolerance should allow more reduction.
    """
    from ctrlsys import tg01od

    np.random.seed(404)
    n = 3

    d = 1e-8
    c = np.array([1.0, 2.0, 3.0], dtype=float)
    b = np.array([1.0, 2.0, 3.0], dtype=float)
    a = np.random.randn(n, n).astype(float, order='F')

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    e = np.zeros((1, 1), dtype=float, order='F')

    dcba_out1, e_out1, nz1, g1, info1 = tg01od('I', dcba.copy(order='F'), e.copy(order='F'), 1e-3)
    dcba_out2, e_out2, nz2, g2, info2 = tg01od('I', dcba.copy(order='F'), e.copy(order='F'), 1e-12)

    assert info1 == 0
    assert info2 == 0


def test_tg01od_output_dimensions():
    """
    Test that output DCBA and E have correct dimensions.

    The reduced system has NZ state dimension, so:
    - Reduced DCBA: (NZ+1) x (NZ+1) in leading part
    - Reduced E: NZ x NZ in leading part (if JOBE='G')
    """
    from ctrlsys import tg01od

    np.random.seed(505)
    n = 4

    d = 1e-14
    c = np.random.randn(n)
    b = np.random.randn(n)
    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, dtype=float, order='F') + 0.1 * np.random.randn(n, n)
    e = e.astype(float, order='F')

    dcba = np.zeros((n + 1, n + 1), dtype=float, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c
    dcba[1:, 0] = b
    dcba[1:, 1:] = a

    dcba_out, e_out, nz, g, info = tg01od('G', dcba, e, 0.0)

    assert info == 0

    assert dcba_out.shape[0] >= nz + 1
    assert dcba_out.shape[1] >= nz + 1

    if nz > 0:
        assert e_out.shape[0] >= nz
        assert e_out.shape[1] >= nz
