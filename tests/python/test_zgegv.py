"""
Tests for ZGEGV - Generalized eigenvalue problem solver for complex matrices.

ZGEGV computes the eigenvalues and, optionally, the left and/or right
eigenvectors of a complex matrix pair (A,B).

Given A*x = lambda*B*x, returns alpha and beta such that lambda = alpha/beta.

This routine is deprecated and replaced by ZGGEV.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_zgegv_basic_eigenvalues():
    """
    Test ZGEGV basic eigenvalue computation without eigenvectors.

    Uses simple 2x2 complex matrices with known generalized eigenvalues.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import zgegv

    np.random.seed(42)

    n = 2
    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = np.eye(n, dtype=np.complex128, order="F")

    a_orig = a.copy()
    eigs_expected = np.linalg.eigvals(a_orig)

    alpha, beta, vl, vr, info = zgegv("N", "N", a, b)

    assert info == 0
    assert alpha.shape == (n,)
    assert beta.shape == (n,)

    eigs_computed = []
    for i in range(n):
        if abs(beta[i]) > 1e-14:
            eigs_computed.append(alpha[i] / beta[i])

    eigs_computed = np.array(sorted(eigs_computed, key=lambda x: (x.real, x.imag)))
    eigs_expected = np.array(sorted(eigs_expected, key=lambda x: (x.real, x.imag)))

    assert_allclose(eigs_computed, eigs_expected, rtol=1e-12, atol=1e-13)


def test_zgegv_with_right_eigenvectors():
    """
    Test ZGEGV with right eigenvector computation.

    Validates A*x = lambda*B*x for right eigenvectors.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import zgegv

    np.random.seed(123)
    n = 3

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = (np.eye(n) + 0.1 * (np.random.randn(n, n) + 1j * np.random.randn(n, n))).astype(
        np.complex128, order="F"
    )

    a_orig = a.copy()
    b_orig = b.copy()

    alpha, beta, vl, vr, info = zgegv("N", "V", a, b)

    assert info == 0
    assert vr.shape == (n, n)

    for j in range(n):
        if abs(beta[j]) > 1e-14:
            lam = alpha[j] / beta[j]
            x = vr[:, j]
            lhs = a_orig @ x
            rhs = lam * (b_orig @ x)
            assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


def test_zgegv_with_left_eigenvectors():
    """
    Test ZGEGV with left eigenvector computation.

    Validates u^H*A = lambda*u^H*B for left eigenvectors.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import zgegv

    np.random.seed(456)
    n = 3

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = np.eye(n, dtype=np.complex128, order="F")

    a_orig = a.copy()
    b_orig = b.copy()

    alpha, beta, vl, vr, info = zgegv("V", "N", a, b)

    assert info == 0
    assert vl.shape == (n, n)

    for j in range(n):
        if abs(beta[j]) > 1e-14:
            lam = alpha[j] / beta[j]
            u = vl[:, j].conj()
            lhs = u @ a_orig
            rhs = lam * (u @ b_orig)
            assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


def test_zgegv_both_eigenvectors():
    """
    Test ZGEGV with both left and right eigenvector computation.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import zgegv

    np.random.seed(789)
    n = 4

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = (np.eye(n) + 0.1 * (np.random.randn(n, n) + 1j * np.random.randn(n, n))).astype(
        np.complex128, order="F"
    )

    a_orig = a.copy()
    b_orig = b.copy()

    alpha, beta, vl, vr, info = zgegv("V", "V", a, b)

    assert info == 0
    assert vl.shape == (n, n)
    assert vr.shape == (n, n)

    for j in range(n):
        if abs(beta[j]) > 1e-14:
            lam = alpha[j] / beta[j]

            x = vr[:, j]
            residual_r = np.linalg.norm(a_orig @ x - lam * b_orig @ x) / np.linalg.norm(x)
            assert residual_r < 1e-10, f"Right eigenvector residual too large: {residual_r}"

            u = vl[:, j].conj()
            residual_l = np.linalg.norm(u @ a_orig - lam * u @ b_orig) / np.linalg.norm(u)
            assert residual_l < 1e-10, f"Left eigenvector residual too large: {residual_l}"


def test_zgegv_identity():
    """
    Test ZGEGV with identity matrices.

    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import zgegv

    n = 5
    a = np.eye(n, dtype=np.complex128, order="F")
    b = np.eye(n, dtype=np.complex128, order="F")

    alpha, beta, vl, vr, info = zgegv("N", "N", a, b)

    assert info == 0

    for i in range(n):
        if abs(beta[i]) > 1e-14:
            eig = alpha[i] / beta[i]
            assert_allclose(eig, 1.0 + 0j, rtol=1e-12, atol=1e-13)


def test_zgegv_singular_b():
    """
    Test ZGEGV with singular B (infinite eigenvalue case).

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import zgegv

    n = 3
    a = np.diag([1.0 + 0j, 2.0 + 0j, 3.0 + 0j]).astype(np.complex128, order="F")
    b = np.diag([1.0 + 0j, 1.0 + 0j, 0.0 + 0j]).astype(np.complex128, order="F")

    alpha, beta, vl, vr, info = zgegv("N", "N", a, b)

    assert info == 0

    inf_count = 0
    for i in range(n):
        if abs(beta[i]) < 1e-14 and abs(alpha[i]) > 1e-14:
            inf_count += 1

    assert inf_count >= 1


def test_zgegv_n_equals_1():
    """
    Edge case: n=1 (scalar case).
    """
    from ctrlsys import zgegv

    a = np.array([[2.0 + 1j]], dtype=np.complex128, order="F")
    b = np.array([[1.0 + 0j]], dtype=np.complex128, order="F")

    alpha, beta, vl, vr, info = zgegv("V", "V", a, b)

    assert info == 0

    if abs(beta[0]) > 1e-14:
        eig = alpha[0] / beta[0]
        assert_allclose(eig, 2.0 + 1j, rtol=1e-13, atol=1e-14)


def test_zgegv_error_invalid_jobvl():
    """
    Test ZGEGV error handling for invalid JOBVL parameter.
    """
    from ctrlsys import zgegv

    n = 2
    a = np.eye(n, dtype=np.complex128, order="F")
    b = np.eye(n, dtype=np.complex128, order="F")

    alpha, beta, vl, vr, info = zgegv("X", "N", a, b)

    assert info == -1


def test_zgegv_error_invalid_jobvr():
    """
    Test ZGEGV error handling for invalid JOBVR parameter.
    """
    from ctrlsys import zgegv

    n = 2
    a = np.eye(n, dtype=np.complex128, order="F")
    b = np.eye(n, dtype=np.complex128, order="F")

    alpha, beta, vl, vr, info = zgegv("N", "X", a, b)

    assert info == -2


def test_zgegv_eigenvalue_consistency():
    """
    Test that eigenvalues from ZGEGV match numpy for B=I.

    Mathematical property: When B=I, generalized eigenvalues reduce to
    standard eigenvalues of A.

    Random seed: 404 (for reproducibility)
    """
    from ctrlsys import zgegv

    np.random.seed(404)
    n = 5

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = np.eye(n, dtype=np.complex128, order="F")

    np_eigs = np.linalg.eigvals(a.copy())

    alpha, beta, vl, vr, info = zgegv("N", "N", a, b)
    assert info == 0

    zgegv_eigs = []
    for i in range(n):
        if abs(beta[i]) > 1e-14:
            zgegv_eigs.append(alpha[i] / beta[i])

    zgegv_eigs_sorted = sorted(zgegv_eigs, key=lambda x: (x.real, x.imag))
    np_eigs_sorted = sorted(np_eigs, key=lambda x: (x.real, x.imag))

    for ze, ne in zip(zgegv_eigs_sorted, np_eigs_sorted):
        assert_allclose(ze, ne, rtol=1e-10, atol=1e-12)
