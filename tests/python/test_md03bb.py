"""
Tests for MD03BB - Levenberg-Marquardt trust-region step.

MD03BB computes the Levenberg-Marquardt parameter and the step x that
solves A*x = b, sqrt(PAR)*D*x = 0 in the least squares sense, where
A*P = Q*R (QR with column pivoting), D = diag(diag).

Key output relationships:
- x: the least squares solution
- rx = -R * P^T * x  (note the negative sign per SLICOT doc)
- Either PAR=0 and ||D*x|| - delta <= 0.1*delta,
  or PAR>0 and |  ||D*x|| - delta | <= 0.1*delta
"""
import numpy as np
import pytest
from ctrlsys import md03bb


def test_md03bb_basic():
    """
    Validate MD03BB via wrapper.
    """
    np.random.seed(42)
    n = 3
    ipar = np.array([0], dtype=np.int32)

    r = np.triu(np.random.rand(n, n))
    for i in range(n): r[i, i] += 2.0
    r_in = np.asfortranarray(r)

    ipvt = np.arange(1, n+1, dtype=np.int32)
    diag = np.ones(n)
    qtb = np.random.rand(n)
    delta = 1.0
    par = 0.0
    ranks = np.array([n], dtype=np.int32)
    tol = 0.0

    r_out, par_out, ranks_out, x, rx, info = md03bb('N', n, ipar, r_in, ipvt, diag, qtb, delta, par, ranks, tol)

    assert info == 0
    assert par_out >= 0.0
    assert x.shape == (n,)
    assert rx.shape == (n,)
    assert ranks_out[0] <= n


def test_known_solution_identity():
    """
    Known solution: R=I, ipvt=identity, diag=I, qtb=b.
    With large delta (unconstrained), x should equal R^{-1}*qtb = qtb.
    rx = -R*P^T*x = -I*x = -qtb.
    """
    n = 3
    ipar = np.array([0], dtype=np.int32)

    r = np.eye(n, dtype=float, order='F')
    ipvt = np.arange(1, n+1, dtype=np.int32)
    diag = np.ones(n, dtype=float)
    qtb = np.array([1.0, 2.0, 3.0], dtype=float)
    delta = 100.0
    par = 0.0
    ranks = np.array([n], dtype=np.int32)

    r_out, par_out, ranks_out, x, rx, info = md03bb('N', n, ipar, r.copy(order='F'), ipvt, diag, qtb, delta, par, ranks.copy(), 0.0)

    assert info == 0
    np.testing.assert_allclose(x, qtb, atol=1e-10)
    np.testing.assert_allclose(rx, -qtb, atol=1e-10)
    assert par_out < 1e-10


def test_trust_region_constraint():
    """
    Verify ||D*x|| is close to delta when constraint is active.
    Per SLICOT doc: |  ||D*x|| - delta | <= 0.1*delta when PAR > 0.
    """
    n = 3
    ipar = np.array([0], dtype=np.int32)

    r = np.array([[3.0, 1.0, 0.5],
                  [0.0, 2.0, 0.3],
                  [0.0, 0.0, 1.5]], dtype=float, order='F')
    ipvt = np.arange(1, n+1, dtype=np.int32)
    diag = np.array([1.0, 1.0, 1.0], dtype=float)
    qtb = np.array([10.0, 20.0, 30.0], dtype=float)
    delta = 0.5
    par = 0.0
    ranks = np.array([n], dtype=np.int32)

    r_out, par_out, ranks_out, x, rx, info = md03bb('N', n, ipar, r.copy(order='F'), ipvt, diag, qtb, delta, par, ranks.copy(), 0.0)

    assert info == 0
    dx_norm = np.linalg.norm(diag * x)
    assert abs(dx_norm - delta) <= 0.1 * delta
    assert par_out > 0.0


def test_rx_equals_neg_r_pt_x():
    """
    Verify rx = -R * P^T * x output relationship.
    With identity permutation, rx = -R * x.
    """
    n = 3
    ipar = np.array([0], dtype=np.int32)

    r = np.array([[2.0, 0.5, 0.1],
                  [0.0, 3.0, 0.2],
                  [0.0, 0.0, 1.0]], dtype=float, order='F')
    ipvt = np.arange(1, n+1, dtype=np.int32)
    diag = np.ones(n, dtype=float)
    qtb = np.array([1.0, 2.0, 3.0], dtype=float)
    delta = 100.0
    par = 0.0
    ranks = np.array([n], dtype=np.int32)

    r_out, par_out, ranks_out, x, rx, info = md03bb('N', n, ipar, r.copy(order='F'), ipvt, diag, qtb, delta, par, ranks.copy(), 0.0)

    assert info == 0
    expected_rx = -(r @ x)
    np.testing.assert_allclose(rx, expected_rx, atol=1e-10)


def test_scaled_diagonal():
    """
    Test with non-uniform diagonal scaling.
    The trust region is ||D*x|| approx delta.
    """
    n = 2
    ipar = np.array([0], dtype=np.int32)

    r = np.array([[4.0, 1.0],
                  [0.0, 3.0]], dtype=float, order='F')
    ipvt = np.array([1, 2], dtype=np.int32)
    diag = np.array([2.0, 0.5], dtype=float)
    qtb = np.array([10.0, 10.0], dtype=float)
    delta = 1.0
    par = 0.0
    ranks = np.array([n], dtype=np.int32)

    r_out, par_out, ranks_out, x, rx, info = md03bb('N', n, ipar, r.copy(order='F'), ipvt, diag, qtb, delta, par, ranks.copy(), 0.0)

    assert info == 0
    dx_norm = np.linalg.norm(diag * x)
    assert abs(dx_norm - delta) <= 0.1 * delta + 1e-10


def test_with_permutation():
    """
    Test with non-trivial column permutation.
    rx = -R * P^T * x where P^T reorders x by ipvt.
    """
    n = 3
    ipar = np.array([0], dtype=np.int32)

    r = np.array([[5.0, 1.0, 0.5],
                  [0.0, 4.0, 0.3],
                  [0.0, 0.0, 3.0]], dtype=float, order='F')
    ipvt = np.array([3, 1, 2], dtype=np.int32)
    diag = np.ones(n, dtype=float)
    qtb = np.array([1.0, 2.0, 3.0], dtype=float)
    delta = 100.0
    par = 0.0
    ranks = np.array([n], dtype=np.int32)

    r_out, par_out, ranks_out, x, rx, info = md03bb('N', n, ipar, r.copy(order='F'), ipvt, diag, qtb, delta, par, ranks.copy(), 0.0)

    assert info == 0
    assert np.all(np.isfinite(x))
    perm_x = np.zeros(n)
    for i in range(n):
        perm_x[i] = x[ipvt[i] - 1]
    expected_rx = -(r @ perm_x)
    np.testing.assert_allclose(rx, expected_rx, atol=1e-10)


def test_residual_optimality():
    """
    For unconstrained case (large delta), the solution should satisfy
    R*P^T*x = qtb, so rx = -R*P^T*x = -qtb.
    """
    n = 2
    ipar = np.array([0], dtype=np.int32)

    r = np.array([[3.0, 1.0],
                  [0.0, 2.0]], dtype=float, order='F')
    ipvt = np.array([1, 2], dtype=np.int32)
    diag = np.ones(n, dtype=float)
    qtb = np.array([5.0, 4.0], dtype=float)
    delta = 100.0
    par = 0.0
    ranks = np.array([n], dtype=np.int32)

    r_out, par_out, ranks_out, x, rx, info = md03bb('N', n, ipar, r.copy(order='F'), ipvt, diag, qtb, delta, par, ranks.copy(), 0.0)

    assert info == 0
    np.testing.assert_allclose(rx, -qtb, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
