"""
Tests for DGEGV - Generalized eigenvalue problem solver.

DGEGV computes the eigenvalues and, optionally, the left and/or right
eigenvectors of a real matrix pair (A,B).

Given A*x = lambda*B*x, returns alpha and beta such that lambda = alpha/beta.

This routine is deprecated and replaced by DGGEV.
"""
import numpy as np
import pytest


def test_dgegv_basic_eigenvalues():
    """
    Test DGEGV basic eigenvalue computation without eigenvectors.

    Uses simple 2x2 matrices with known generalized eigenvalues.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import dgegv

    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    eigs_expected = np.linalg.eigvals(a.copy())

    alphar, alphai, beta, vl, vr, info = dgegv('N', 'N', a, b)

    assert info == 0
    assert alphar.shape == (2,)
    assert alphai.shape == (2,)
    assert beta.shape == (2,)

    eigs_computed = []
    for i in range(2):
        if beta[i] != 0:
            eigs_computed.append(complex(alphar[i], alphai[i]) / beta[i])

    eigs_computed = np.array(sorted(eigs_computed, key=lambda x: x.real))
    eigs_expected = np.array(sorted(eigs_expected, key=lambda x: x.real))

    np.testing.assert_allclose(
        np.array([e.real for e in eigs_computed]),
        np.array([e.real for e in eigs_expected]),
        rtol=1e-12
    )


def test_dgegv_with_right_eigenvectors():
    """
    Test DGEGV with right eigenvector computation.

    Validates A*x = lambda*B*x for right eigenvectors.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import dgegv

    np.random.seed(123)
    n = 3

    a = np.array([
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    b_orig = b.copy()

    alphar, alphai, beta, vl, vr, info = dgegv('N', 'V', a, b)

    assert info == 0
    assert vr.shape == (n, n)

    for j in range(n):
        if alphai[j] == 0.0 and beta[j] != 0.0:
            lam = alphar[j] / beta[j]
            x = vr[:, j]
            lhs = a_orig @ x
            rhs = lam * (b_orig @ x)
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


def test_dgegv_with_left_eigenvectors():
    """
    Test DGEGV with left eigenvector computation.

    Validates u^H*A = lambda*u^H*B for left eigenvectors.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import dgegv

    n = 3
    a = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 1.0],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()
    b_orig = b.copy()

    alphar, alphai, beta, vl, vr, info = dgegv('V', 'N', a, b)

    assert info == 0
    assert vl.shape == (n, n)

    for j in range(n):
        if alphai[j] == 0.0 and beta[j] != 0.0:
            lam = alphar[j] / beta[j]
            u = vl[:, j]
            lhs = u @ a_orig
            rhs = lam * (u @ b_orig)
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


def test_dgegv_both_eigenvectors():
    """
    Test DGEGV with both left and right eigenvector computation.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import dgegv

    n = 4
    np.random.seed(789)

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.eye(n, order='F', dtype=float) + 0.1 * np.random.randn(n, n).astype(float, order='F')

    a_orig = a.copy()
    b_orig = b.copy()

    alphar, alphai, beta, vl, vr, info = dgegv('V', 'V', a, b)

    assert info == 0
    assert vl.shape == (n, n)
    assert vr.shape == (n, n)

    j = 0
    while j < n:
        if alphai[j] == 0.0:
            if beta[j] != 0.0:
                lam = alphar[j] / beta[j]
                x = vr[:, j]
                residual_r = np.linalg.norm(a_orig @ x - lam * b_orig @ x) / np.linalg.norm(x)
                assert residual_r < 1e-10, f"Right eigenvector residual too large: {residual_r}"

                u = vl[:, j]
                residual_l = np.linalg.norm(u @ a_orig - lam * u @ b_orig) / np.linalg.norm(u)
                assert residual_l < 1e-10, f"Left eigenvector residual too large: {residual_l}"
            j += 1
        else:
            j += 2


def test_dgegv_identity():
    """
    Test DGEGV with identity matrices.

    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import dgegv

    n = 5
    a = np.eye(n, order='F', dtype=float)
    b = np.eye(n, order='F', dtype=float)

    alphar, alphai, beta, vl, vr, info = dgegv('N', 'N', a, b)

    assert info == 0

    for i in range(n):
        if beta[i] != 0:
            eig = alphar[i] / beta[i]
            np.testing.assert_allclose(eig, 1.0, rtol=1e-12)


def test_dgegv_singular_b():
    """
    Test DGEGV with singular B (infinite eigenvalue case).

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import dgegv

    n = 3
    a = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ], order='F', dtype=float)

    alphar, alphai, beta, vl, vr, info = dgegv('N', 'N', a, b)

    assert info == 0

    inf_count = 0
    for i in range(n):
        if beta[i] == 0.0 and (alphar[i] != 0.0 or alphai[i] != 0.0):
            inf_count += 1

    assert inf_count >= 1


def test_dgegv_complex_eigenvalues():
    """
    Test DGEGV with matrix pair having complex conjugate eigenvalues.

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import dgegv

    a = np.array([
        [0.0, -1.0],
        [1.0, 0.0]
    ], order='F', dtype=float)

    b = np.eye(2, order='F', dtype=float)

    alphar, alphai, beta, vl, vr, info = dgegv('V', 'V', a, b)

    assert info == 0

    has_complex = any(alphai[i] != 0.0 for i in range(2))
    assert has_complex, "Should have complex eigenvalues (i, -i)"


def test_dgegv_error_invalid_jobvl():
    """
    Test DGEGV error handling for invalid JOBVL parameter.
    """
    from ctrlsys import dgegv

    n = 2
    a = np.eye(n, order='F', dtype=float)
    b = np.eye(n, order='F', dtype=float)

    with pytest.raises((ValueError, RuntimeError)):
        dgegv('X', 'N', a, b)


def test_dgegv_eigenvalue_consistency():
    """
    Test that eigenvalues from DGEGV match scipy.linalg.eig for B=I.

    Mathematical property: When B=I, generalized eigenvalues reduce to
    standard eigenvalues of A.

    Random seed: 404 (for reproducibility)
    """
    from ctrlsys import dgegv

    np.random.seed(404)
    n = 5

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.eye(n, order='F', dtype=float)

    np_eigs = np.linalg.eigvals(a.copy())

    alphar, alphai, beta, vl, vr, info = dgegv('N', 'N', a, b)
    assert info == 0

    dgegv_eigs = []
    for i in range(n):
        if beta[i] != 0:
            dgegv_eigs.append(complex(alphar[i], alphai[i]) / beta[i])

    dgegv_eigs_sorted = sorted(dgegv_eigs, key=lambda x: (x.real, abs(x.imag)))
    np_eigs_sorted = sorted(np_eigs, key=lambda x: (x.real, abs(x.imag)))

    for de, ne in zip(dgegv_eigs_sorted, np_eigs_sorted):
        np.testing.assert_allclose(de.real, ne.real, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(abs(de.imag), abs(ne.imag), rtol=1e-10, atol=1e-12)
