"""
Tests for MB04TX: Separate pencils s*E(eps)-A(eps) and s*E(inf)-A(inf).

MB04TX separates the epsilon and infinite parts of the pencil
s*E(eps,inf)-A(eps,inf) in staircase form using Givens rotations.

No HTML documentation available, so tests use synthetic data with
mathematical property validation.
"""

import numpy as np
import pytest
from ctrlsys import mb04tx


def test_mb04tx_basic():
    """
    Test basic functionality with 2 blocks.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    # Setup: 2 blocks with dimensions nu=[2,1], mu=[3,2]
    # Total rows = 2+1 = 3, total cols = 3+2 = 5
    nblcks = 2
    inuk = np.array([2, 1], dtype=np.int32)
    imuk = np.array([3, 2], dtype=np.int32)

    m = np.sum(inuk)  # 3
    n = np.sum(imuk)  # 5

    # Create staircase form matrices
    a = np.zeros((m, n), dtype=float, order='F')
    e = np.zeros((m, n), dtype=float, order='F')

    # Fill with staircase structure
    # Block 1: nu[0]=2 rows, mu[0]=3 cols
    # Block 2: nu[1]=1 row, mu[1]=2 cols (shifted)
    a[0, 0] = 1.0
    a[0, 1] = 2.0
    a[0, 2] = 3.0
    a[1, 0] = 0.5
    a[1, 1] = 1.5
    a[1, 2] = 2.5
    a[2, 3] = 1.0
    a[2, 4] = 2.0

    e[0, 1] = 1.0
    e[0, 2] = 2.0
    e[0, 3] = 3.0
    e[1, 2] = 0.5
    e[1, 3] = 1.5
    e[1, 4] = 2.5
    e[2, 4] = 1.0

    q = np.eye(m, dtype=float, order='F')
    z = np.eye(n, dtype=float, order='F')

    # Call routine
    a_out, e_out, q_out, z_out, nblcks_out, inuk_out, imuk_out, mnei, info = mb04tx(
        a, e, inuk, imuk, q=q, z=z, updatq=True, updatz=True
    )

    assert info == 0
    assert mnei.shape == (4,)

    # MNEI contains dimensions of separated pencils
    meps, neps, minf, ninf = mnei
    assert meps >= 0
    assert neps >= 0
    assert minf >= 0
    assert ninf >= 0

    # Total dimensions should be preserved
    assert meps + minf <= m
    assert neps + ninf <= n


def test_mb04tx_orthogonality():
    """
    Test that Q and Z remain orthogonal after transformation.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    # Setup: Simple 2-block structure
    nblcks = 2
    inuk = np.array([2, 1], dtype=np.int32)
    imuk = np.array([2, 1], dtype=np.int32)

    m = np.sum(inuk)
    n = np.sum(imuk)

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    q = np.eye(m, dtype=float, order='F')
    z = np.eye(n, dtype=float, order='F')

    a_out, e_out, q_out, z_out, nblcks_out, inuk_out, imuk_out, mnei, info = mb04tx(
        a, e, inuk, imuk, q=q, z=z, updatq=True, updatz=True
    )

    assert info == 0

    # Q should remain orthogonal: Q^T * Q = I
    qtq = q_out.T @ q_out
    np.testing.assert_allclose(qtq, np.eye(m), rtol=1e-14, atol=1e-14)

    # Z should remain orthogonal: Z^T * Z = I
    ztz = z_out.T @ z_out
    np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb04tx_transformation_consistency():
    """
    Test that A_out = Q^T * A_orig * Z (transformation is consistent).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    nblcks = 2
    inuk = np.array([2, 1], dtype=np.int32)
    imuk = np.array([2, 1], dtype=np.int32)

    m = np.sum(inuk)
    n = np.sum(imuk)

    a_orig = np.random.randn(m, n).astype(float, order='F')
    e_orig = np.random.randn(m, n).astype(float, order='F')

    a = a_orig.copy(order='F')
    e = e_orig.copy(order='F')

    q = np.eye(m, dtype=float, order='F')
    z = np.eye(n, dtype=float, order='F')

    a_out, e_out, q_out, z_out, nblcks_out, inuk_out, imuk_out, mnei, info = mb04tx(
        a, e, inuk, imuk, q=q, z=z, updatq=True, updatz=True
    )

    assert info == 0

    # Verify: A_out = Q^T * A_orig * Z
    a_expected = q_out.T @ a_orig @ z_out
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-13, atol=1e-14)

    # Verify: E_out = Q^T * E_orig * Z
    e_expected = q_out.T @ e_orig @ z_out
    np.testing.assert_allclose(e_out, e_expected, rtol=1e-13, atol=1e-14)


def test_mb04tx_no_update_q_z():
    """
    Test with updatq=False and updatz=False.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    nblcks = 2
    inuk = np.array([2, 1], dtype=np.int32)
    imuk = np.array([2, 1], dtype=np.int32)

    m = np.sum(inuk)
    n = np.sum(imuk)

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    # No Q/Z update
    a_out, e_out, q_out, z_out, nblcks_out, inuk_out, imuk_out, mnei, info = mb04tx(
        a, e, inuk, imuk, updatq=False, updatz=False
    )

    assert info == 0
    assert mnei.shape == (4,)

    # Q and Z should be None when not updated
    assert q_out is None
    assert z_out is None


def test_mb04tx_single_block():
    """
    Test with single block (NBLCKS=1).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    nblcks = 1
    inuk = np.array([2], dtype=np.int32)
    imuk = np.array([3], dtype=np.int32)

    m = np.sum(inuk)
    n = np.sum(imuk)

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    q = np.eye(m, dtype=float, order='F')
    z = np.eye(n, dtype=float, order='F')

    a_out, e_out, q_out, z_out, nblcks_out, inuk_out, imuk_out, mnei, info = mb04tx(
        a, e, inuk, imuk, q=q, z=z, updatq=True, updatz=True
    )

    assert info == 0

    # Orthogonality preserved
    np.testing.assert_allclose(q_out.T @ q_out, np.eye(m), rtol=1e-14)
    np.testing.assert_allclose(z_out.T @ z_out, np.eye(n), rtol=1e-14)


def test_mb04tx_empty_case():
    """
    Test edge case with m=0 or n=0 (quick return).
    """
    nblcks = 0
    inuk = np.array([], dtype=np.int32)
    imuk = np.array([], dtype=np.int32)

    m = 0
    n = 0

    a = np.zeros((1, 1), dtype=float, order='F')  # Dummy
    e = np.zeros((1, 1), dtype=float, order='F')  # Dummy

    # Should handle gracefully with quick return
    a_out, e_out, q_out, z_out, nblcks_out, inuk_out, imuk_out, mnei, info = mb04tx(
        a, e, inuk, imuk, m=0, n=0, updatq=False, updatz=False
    )

    assert info == 0
    # MNEI should be all zeros for quick return
    assert np.all(mnei == 0)
