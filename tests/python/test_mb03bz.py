"""
Tests for MB03BZ: Complex periodic QZ algorithm for eigenvalues of generalized matrix products.

MB03BZ computes eigenvalues of the complex generalized matrix product:
    A(:,:,1)^S(1) * A(:,:,2)^S(2) * ... * A(:,:,K)^S(K),  S(1) = 1

where A(:,:,1) is upper Hessenberg and A(:,:,i), i=2,...,K are upper triangular.
Can optionally reduce to periodic Schur form.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb03bz_html_example():
    """
    Test MB03BZ using data from SLICOT HTML documentation.

    Test parameters:
    - K = 3 factors
    - N = 4 matrix size
    - ILO = 1, IHI = 4
    - JOB = 'S' (compute Schur form)
    - COMPQ = 'I' (initialize Q to identity)
    - S = [1, -1, 1] signatures
    """
    from ctrlsys import mb03bz

    k = 3
    n = 4
    ilo = 1
    ihi = 4

    a1 = np.array([
        [0.8637+0.9326j, 0.8819+0.4850j, 0.5920+0.8826j, 0.8991+0.9040j],
        [0.6994+0.8588j, 0.9527+0.2672j, 0.5087+0.0621j, 0.9653+0.5715j],
        [0.0+0.0j,       0.1561+0.1898j, 0.9514+0.9266j, 0.6582+0.3102j],
        [0.0+0.0j,       0.0+0.0j,       0.8649+0.1265j, 0.1701+0.0013j]
    ], dtype=np.complex128, order='F')

    a2 = np.array([
        [0.5113+0.7375j, 0.6869+0.7692j, 0.7812+0.1467j, 0.7216+0.9498j],
        [0.0+0.0j,       0.1319+0.9137j, 0.5879+0.0201j, 0.9834+0.0549j],
        [0.0+0.0j,       0.0+0.0j,       0.7711+0.2422j, 0.9468+0.3280j],
        [0.0+0.0j,       0.0+0.0j,       0.0+0.0j,       0.2219+0.3971j]
    ], dtype=np.complex128, order='F')

    a3 = np.array([
        [0.0158+0.4042j, 0.0082+0.2033j, 0.1028+0.9913j, 0.6954+0.1987j],
        [0.0+0.0j,       0.5066+0.4587j, 0.1060+0.6949j, 0.5402+0.0970j],
        [0.0+0.0j,       0.0+0.0j,       0.4494+0.3700j, 0.8492+0.4882j],
        [0.0+0.0j,       0.0+0.0j,       0.0+0.0j,       0.2110+0.5824j]
    ], dtype=np.complex128, order='F')

    a = np.zeros((n, n, k), dtype=np.complex128, order='F')
    a[:, :, 0] = a1
    a[:, :, 1] = a2
    a[:, :, 2] = a3

    s = np.array([1, -1, 1], dtype=np.int32)

    a_out, q_out, alpha, beta, scal, info = mb03bz('S', 'I', k, n, ilo, ihi, s, a)

    assert info == 0

    alpha_expected = np.array([
        0.6290+1.0715j,
        -0.2992+1.1797j,
        -1.0195-1.0290j,
        -1.1523+0.0326j
    ], dtype=np.complex128)

    beta_expected = np.array([
        1.0+0.0j,
        1.0+0.0j,
        1.0+0.0j,
        1.0+0.0j
    ], dtype=np.complex128)

    scal_expected = np.array([0, -1, -2, -3], dtype=np.int32)

    assert_allclose(alpha, alpha_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(beta, beta_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(scal, scal_expected)

    for l in range(k):
        for i in range(n):
            for j in range(i):
                assert abs(a_out[i, j, l]) < 1e-10, f"a_out[{i},{j},{l}] should be zero"

    for l in range(1, k):
        for i in range(n):
            assert a_out[i, i, l].imag == pytest.approx(0, abs=1e-10), \
                f"a_out[{i},{i},{l}] diagonal should be real"
            assert a_out[i, i, l].real >= -1e-10, \
                f"a_out[{i},{i},{l}] diagonal should be non-negative"

    for l in range(k):
        qh = np.conj(q_out[:, :, l].T)
        identity_check = qh @ q_out[:, :, l]
        assert_allclose(identity_check, np.eye(n), rtol=1e-10, atol=1e-10)


def test_mb03bz_eigenvalues_only():
    """
    Test MB03BZ with JOB='E' (eigenvalues only).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03bz

    np.random.seed(42)

    k = 2
    n = 3
    ilo = 1
    ihi = 3

    a1 = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        for j in range(i, n):
            a1[i, j] = np.random.randn() + 1j * np.random.randn()
    a1[1, 0] = np.random.randn() + 1j * np.random.randn()

    a2 = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        for j in range(i, n):
            a2[i, j] = np.random.randn() + 1j * np.random.randn()
    np.fill_diagonal(a2, np.abs(np.diag(a2)) + 0.1)

    a = np.zeros((n, n, k), dtype=np.complex128, order='F')
    a[:, :, 0] = a1
    a[:, :, 1] = a2

    s = np.array([1, 1], dtype=np.int32)

    a_out, q_out, alpha, beta, scal, info = mb03bz('E', 'N', k, n, ilo, ihi, s, a)

    assert info == 0
    assert alpha.shape == (n,)
    assert beta.shape == (n,)
    assert scal.shape == (n,)


def test_mb03bz_n_equals_zero():
    """
    Test MB03BZ with N=0 (edge case - quick return).
    """
    from ctrlsys import mb03bz

    k = 2
    n = 0
    ilo = 1
    ihi = 0

    a = np.zeros((1, 1, k), dtype=np.complex128, order='F')
    s = np.array([1, 1], dtype=np.int32)

    a_out, q_out, alpha, beta, scal, info = mb03bz('S', 'I', k, n, ilo, ihi, s, a)

    assert info == 0


def test_mb03bz_single_factor():
    """
    Test MB03BZ with K=1 (single factor).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03bz

    np.random.seed(123)

    k = 1
    n = 3
    ilo = 1
    ihi = 3

    a1 = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        for j in range(i, n):
            a1[i, j] = np.random.randn() + 1j * np.random.randn()
    a1[1, 0] = np.random.randn() + 1j * np.random.randn()

    a = np.zeros((n, n, k), dtype=np.complex128, order='F')
    a[:, :, 0] = a1

    s = np.array([1], dtype=np.int32)

    a_out, q_out, alpha, beta, scal, info = mb03bz('S', 'I', k, n, ilo, ihi, s, a)

    assert info == 0

    for i in range(n):
        for j in range(i):
            assert abs(a_out[i, j, 0]) < 1e-10


def test_mb03bz_compq_v():
    """
    Test MB03BZ with COMPQ='V' (update given Q matrices).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03bz

    np.random.seed(456)

    k = 2
    n = 3
    ilo = 1
    ihi = 3

    a1 = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        for j in range(i, n):
            a1[i, j] = np.random.randn() + 1j * np.random.randn()
    a1[1, 0] = np.random.randn() + 1j * np.random.randn()

    a2 = np.zeros((n, n), dtype=np.complex128, order='F')
    for i in range(n):
        for j in range(i, n):
            a2[i, j] = np.random.randn() + 1j * np.random.randn()
    np.fill_diagonal(a2, np.abs(np.diag(a2)) + 0.1)

    a = np.zeros((n, n, k), dtype=np.complex128, order='F')
    a[:, :, 0] = a1
    a[:, :, 1] = a2

    s = np.array([1, 1], dtype=np.int32)

    q = np.zeros((n, n, k), dtype=np.complex128, order='F')
    for l in range(k):
        q[:, :, l] = np.eye(n, dtype=np.complex128)

    a_out, q_out, alpha, beta, scal, info = mb03bz('S', 'V', k, n, ilo, ihi, s, a, q)

    assert info == 0

    for l in range(k):
        qh = np.conj(q_out[:, :, l].T)
        identity_check = qh @ q_out[:, :, l]
        assert_allclose(identity_check, np.eye(n), rtol=1e-10, atol=1e-10)


def test_mb03bz_invalid_job():
    """
    Test MB03BZ with invalid JOB parameter (error handling).
    """
    from ctrlsys import mb03bz

    k = 2
    n = 3
    ilo = 1
    ihi = 3

    a = np.zeros((n, n, k), dtype=np.complex128, order='F')
    s = np.array([1, 1], dtype=np.int32)

    a_out, q_out, alpha, beta, scal, info = mb03bz('X', 'N', k, n, ilo, ihi, s, a)

    assert info == -1


def test_mb03bz_invalid_compq():
    """
    Test MB03BZ with invalid COMPQ parameter (error handling).
    """
    from ctrlsys import mb03bz

    k = 2
    n = 3
    ilo = 1
    ihi = 3

    a = np.zeros((n, n, k), dtype=np.complex128, order='F')
    s = np.array([1, 1], dtype=np.int32)

    a_out, q_out, alpha, beta, scal, info = mb03bz('S', 'X', k, n, ilo, ihi, s, a)

    assert info == -2


def test_mb03bz_invalid_s():
    """
    Test MB03BZ with invalid S array (S(1) must be 1).
    """
    from ctrlsys import mb03bz

    k = 2
    n = 3
    ilo = 1
    ihi = 3

    a = np.zeros((n, n, k), dtype=np.complex128, order='F')
    s = np.array([-1, 1], dtype=np.int32)

    a_out, q_out, alpha, beta, scal, info = mb03bz('S', 'N', k, n, ilo, ihi, s, a)

    assert info == -7
