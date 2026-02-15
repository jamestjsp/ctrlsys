"""
Tests for MC03NY: Minimal polynomial basis for right nullspace of staircase pencil.

MC03NY computes a minimal basis of the right nullspace of the subpencil
s*E(eps)-A(eps) which must be in staircase form. This is a helper routine
for MC03ND.

The basis vectors are represented by matrix V(s) with polynomial coefficients
stored in VEPS.
"""

import numpy as np
import pytest
from ctrlsys import mc03ny


def test_mc03ny_single_block():
    """
    Test with single block: nblcks=1.

    For single block, VEPS has nrv=mu(1) rows and ncv=1*(mu(1)-nu(1)) columns.
    The result is V11,0 = [I; O] where I is identity of size mu-nu.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    nblcks = 1
    mu = np.array([3], dtype=np.int32)
    nu = np.array([2], dtype=np.int32)
    nra = np.sum(nu)
    nca = np.sum(mu)

    a = np.random.randn(nra, nca).astype(float, order='F')
    e = np.random.randn(nra, nca).astype(float, order='F')

    a[0, 1] = 2.0
    a[1, 1] = 0.0
    a[1, 2] = 3.0

    ncv = 1 * (mu[0] - nu[0])

    veps, imuk_out, info = mc03ny(nblcks, nra, nca, a, e, mu.copy(), nu)

    assert info == 0
    assert veps.shape == (nca, ncv)

    v11_0_expected = np.array([[1.0], [0.0], [0.0]], order='F')
    np.testing.assert_allclose(veps, v11_0_expected, rtol=1e-14, atol=1e-14)


def test_mc03ny_two_blocks():
    """
    Test with two blocks: nblcks=2.

    For two blocks:
      nrv = mu(1) + mu(2)
      ncv = 1*(mu(1)-nu(1)) + 2*(mu(2)-nu(2))

    The VEPS matrix structure for n=2:
        sizes:    m1-n1   m2-n2   m2-n2
        m1  { | V11,0 || V12,0 | V12,1 ||
        m2  { |       || V22,0 |       ||

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    nblcks = 2
    mu = np.array([2, 2], dtype=np.int32)
    nu = np.array([1, 1], dtype=np.int32)
    nra = np.sum(nu)
    nca = np.sum(mu)

    a = np.zeros((nra, nca), order='F')
    e = np.zeros((nra, nca), order='F')

    a[0, 0] = 0.0
    a[0, 1] = 1.0
    a[0, 2] = 0.5
    a[0, 3] = 0.0
    a[1, 2] = 0.0
    a[1, 3] = 2.0

    e[0, 2] = 1.0
    e[0, 3] = 0.0
    e[1, 2] = 0.0
    e[1, 3] = 1.0

    ncv = 1 * (mu[0] - nu[0]) + 2 * (mu[1] - nu[1])
    assert ncv == 1 + 2 == 3

    veps, imuk_out, info = mc03ny(nblcks, nra, nca, a, e, mu.copy(), nu)

    assert info == 0
    assert veps.shape == (nca, ncv)
    np.testing.assert_array_equal(imuk_out, mu)


def test_mc03ny_identity_pencil():
    """
    Test with identity-like staircase pencil.

    Uses a simple diagonal structure that satisfies full row rank requirement.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    nblcks = 2
    mu = np.array([2, 2], dtype=np.int32)
    nu = np.array([1, 1], dtype=np.int32)
    nra = 2
    nca = 4

    a = np.zeros((nra, nca), order='F')
    a[0, 1] = 1.0
    a[1, 3] = 1.0

    e = np.zeros((nra, nca), order='F')
    e[0, 2] = 1.0
    e[0, 3] = 0.5
    e[1, 2] = 0.0
    e[1, 3] = 1.0

    veps, imuk_out, info = mc03ny(nblcks, nra, nca, a, e, mu.copy(), nu)

    assert info == 0
    assert veps.shape[0] == nca
    np.testing.assert_array_equal(imuk_out, mu)


def test_mc03ny_quick_return_zero_blocks():
    """Test quick return when nblcks=0."""
    nblcks = 0
    nra = 0
    nca = 0

    a = np.zeros((1, 1), order='F')
    e = np.zeros((1, 1), order='F')
    mu = np.array([], dtype=np.int32)
    nu = np.array([], dtype=np.int32)

    veps, imuk_out, info = mc03ny(nblcks, nra, nca, a, e, mu, nu)

    assert info == 0


def test_mc03ny_quick_return_zero_rows():
    """Test quick return when nra=0."""
    nblcks = 1
    nra = 0
    nca = 2

    a = np.zeros((1, nca), order='F')
    e = np.zeros((1, nca), order='F')
    mu = np.array([2], dtype=np.int32)
    nu = np.array([0], dtype=np.int32)

    veps, imuk_out, info = mc03ny(nblcks, nra, nca, a, e, mu, nu)

    assert info == 0


def test_mc03ny_invalid_nblcks():
    """Test error handling for negative nblcks."""
    nblcks = -1
    nra = 2
    nca = 3

    a = np.zeros((nra, nca), order='F')
    e = np.zeros((nra, nca), order='F')
    mu = np.array([3], dtype=np.int32)
    nu = np.array([2], dtype=np.int32)

    veps, imuk_out, info = mc03ny(nblcks, nra, nca, a, e, mu, nu)

    assert info == -1


def test_mc03ny_invalid_nra():
    """Test error handling for negative nra."""
    nblcks = 1
    nra = -1
    nca = 3

    a = np.zeros((1, nca), order='F')
    e = np.zeros((1, nca), order='F')
    mu = np.array([3], dtype=np.int32)
    nu = np.array([2], dtype=np.int32)

    veps, imuk_out, info = mc03ny(nblcks, nra, nca, a, e, mu, nu)

    assert info == -2


def test_mc03ny_invalid_nca():
    """Test error handling for negative nca.

    The routine correctly detects the error and calls XERBLA (seen in stdout),
    but the wrapper also raises ValueError due to negative array dimensions.
    """
    nblcks = 1
    nra = 2
    nca = -1

    a = np.zeros((nra, 1), order='F')
    e = np.zeros((nra, 1), order='F')
    mu = np.array([3], dtype=np.int32)
    nu = np.array([2], dtype=np.int32)

    with pytest.raises(ValueError, match="negative dimensions"):
        mc03ny(nblcks, nra, nca, a, e, mu, nu)


def test_mc03ny_imuk_restored():
    """
    Verify that IMUK is restored to original values after computation.

    The routine modifies IMUK internally (cumulative sums) but must restore
    the original values on exit.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    nblcks = 3
    mu_orig = np.array([2, 3, 2], dtype=np.int32)
    nu = np.array([1, 2, 1], dtype=np.int32)
    nra = np.sum(nu)
    nca = np.sum(mu_orig)

    a = np.random.randn(nra, nca).astype(float, order='F')
    e = np.random.randn(nra, nca).astype(float, order='F')

    for i in range(nra):
        for j in range(nra):
            a[i, j + (nca - nra)] = (1.0 if i == j else 0.0) + 0.1 * a[i, j + (nca - nra)]

    mu_input = mu_orig.copy()
    veps, imuk_out, info = mc03ny(nblcks, nra, nca, a, e, mu_input, nu)

    np.testing.assert_array_equal(imuk_out, mu_orig)
