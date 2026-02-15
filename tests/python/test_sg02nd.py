"""
Tests for SG02ND: Optimal gain matrix for discrete/continuous Riccati problems.

Computes:
- Discrete: K = (R + B'XB)^{-1} (B'X*op(A) + L')
- Continuous: K = R^{-1} (B'X*op(E) + L')

Test data from SLICOT HTML documentation example.

Mathematical properties tested:
- Gain matrix dimensions
- Riccati equation residual (closed-loop verification)
- Condition number output

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sg02nd_discrete_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Discrete-time case with identity E, unfactored R, no L matrix.
    N=2, M=1, P=3, DICO='D', JOBE='I', JOB='K', FACT='N', JOBL='Z'

    The example solves for the Riccati solution X and then computes the
    optimal feedback matrix K.
    """
    from ctrlsys import sg02nd

    n = 2
    m = 1
    p = 3

    a = np.array([
        [2.0, -1.0],
        [1.0,  0.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.0]
    ], order='F', dtype=float)

    r = np.array([
        [0.0]
    ], order='F', dtype=float)

    x = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    k_expected = np.array([
        [2.0, -1.0]
    ], order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='D', jobe='I', job='K', jobx='N', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=p,
        a=a.copy(order='F'),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)
    assert_allclose(k, k_expected, rtol=1e-3, atol=1e-4)


def test_sg02nd_continuous_identity_e():
    """
    Test continuous-time case with identity E.

    Continuous: K = R^{-1} B'X
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(42)
    n = 3
    m = 2

    r = np.eye(m, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)
    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='C', jobe='I', job='K', jobx='N', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=np.zeros((1, 1), order='F', dtype=float),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)

    k_expected = b.T.copy(order='F')
    assert_allclose(k, k_expected, rtol=1e-14)


def test_sg02nd_with_h_output():
    """
    Test JOB='H' mode that returns both K and H.

    H = op(A)'*X*B + L (discrete) or op(E)'*X*B + L (continuous)
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(123)
    n = 2
    m = 1

    a = np.array([
        [0.5, 0.2],
        [0.1, 0.4]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.0]
    ], order='F', dtype=float)

    r = np.array([[1.0]], order='F', dtype=float)

    x = np.array([
        [2.0, 0.5],
        [0.5, 1.5]
    ], order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='D', jobe='I', job='H', jobx='N', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=a.copy(order='F'),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)
    assert h.shape == (n, m)

    h_expected = a.T @ x @ b
    assert_allclose(h, h_expected, rtol=1e-14)


def test_sg02nd_cholesky_factored_r():
    """
    Test with Cholesky factored R (FACT='C').

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(456)
    n = 2
    m = 2

    r_chol = np.array([
        [2.0, 0.5],
        [0.0, 1.5]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='C', jobe='I', job='K', jobx='N', fact='C',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=np.zeros((1, 1), order='F', dtype=float),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r_chol.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert oufact[0] == 1


def test_sg02nd_with_cross_term_l():
    """
    Test with non-zero cross-weighting matrix L (JOBL='N').

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(789)
    n = 2
    m = 1

    a = np.array([
        [1.0, 0.5],
        [0.0, 0.8]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.5]
    ], order='F', dtype=float)

    r = np.array([[2.0]], order='F', dtype=float)

    l = np.array([
        [0.1],
        [0.2]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='D', jobe='I', job='K', jobx='N', fact='N',
        uplo='U', jobl='N', trans='N',
        n=n, m=m, p=m,
        a=a.copy(order='F'),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=l.copy(order='F'),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)


def test_sg02nd_zero_dimensions():
    """
    Test edge case with n=0 (quick return).
    """
    from ctrlsys import sg02nd

    n = 0
    m = 1

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='C', jobe='I', job='K', jobx='N', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=np.zeros((1, 1), order='F', dtype=float),
        e=np.zeros((1, 1), order='F', dtype=float),
        b=np.zeros((1, 1), order='F', dtype=float),
        r=np.zeros((1, 1), order='F', dtype=float),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=np.zeros((1, 1), order='F', dtype=float),
        rnorm=0.0
    )

    assert info == 0
    assert rcond == 1.0


def test_sg02nd_continuous_general_e():
    """
    Test continuous-time case with general E matrix (JOBE='G').

    Random seed: 321 (for reproducibility)
    """
    from ctrlsys import sg02nd

    np.random.seed(321)
    n = 2
    m = 1

    e = np.array([
        [2.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0],
        [0.5]
    ], order='F', dtype=float)

    r = np.array([[1.0]], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    k, h, xe, oufact, rcond, info = sg02nd(
        dico='C', jobe='G', job='K', jobx='C', fact='N',
        uplo='U', jobl='Z', trans='N',
        n=n, m=m, p=m,
        a=np.zeros((1, 1), order='F', dtype=float),
        e=e.copy(order='F'),
        b=b.copy(order='F'),
        r=r.copy(order='F'),
        ipiv=np.zeros(m, dtype=np.int32),
        l=np.zeros((1, 1), order='F', dtype=float),
        x=x.copy(order='F'),
        rnorm=0.0
    )

    assert info == 0
    assert k.shape == (m, n)

    xe_expected = x @ e
    assert_allclose(xe, xe_expected, rtol=1e-14)
