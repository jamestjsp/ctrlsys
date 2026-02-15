"""
Tests for ZGEGS - Generalized Schur decomposition for complex matrices.

ZGEGS computes the eigenvalues, Schur form, and optionally left/right Schur
vectors of a complex matrix pair (A, B).

Given matrices A and B, computes:
    A = Q * S * Z^H
    B = Q * T * Z^H

where Q and Z are unitary and S and T are upper triangular.
Eigenvalues are alpha(j)/beta(j) where alpha = diag(S), beta = diag(T).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_zgegs_basic():
    """
    Basic test with 2x2 complex matrices.

    Tests that generalized Schur decomposition produces upper triangular
    S and T with correct eigenvalue relationship.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import zgegs

    np.random.seed(42)

    n = 2
    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )

    a_orig = a.copy()
    b_orig = b.copy()

    s, t, alpha, beta, vsl, vsr, info = zgegs("V", "V", a, b)

    assert info == 0

    assert s.shape == (n, n)
    assert t.shape == (n, n)
    assert alpha.shape == (n,)
    assert beta.shape == (n,)
    assert vsl.shape == (n, n)
    assert vsr.shape == (n, n)

    for j in range(n):
        for i in range(j + 1, n):
            assert abs(s[i, j]) < 1e-12, f"S not upper triangular at ({i},{j})"
            assert abs(t[i, j]) < 1e-12, f"T not upper triangular at ({i},{j})"

    assert_allclose(vsl @ vsl.conj().T, np.eye(n), rtol=1e-13, atol=1e-14)
    assert_allclose(vsr @ vsr.conj().T, np.eye(n), rtol=1e-13, atol=1e-14)

    a_reconstructed = vsl @ s @ vsr.conj().T
    b_reconstructed = vsl @ t @ vsr.conj().T

    assert_allclose(a_reconstructed, a_orig, rtol=1e-12, atol=1e-13)
    assert_allclose(b_reconstructed, b_orig, rtol=1e-12, atol=1e-13)

    for j in range(n):
        assert_allclose(alpha[j], s[j, j], rtol=1e-14)
        assert_allclose(beta[j], t[j, j], rtol=1e-14)


def test_zgegs_eigenvalue_preservation():
    """
    Validate eigenvalue preservation: generalized eigenvalues should match.

    The generalized eigenvalues lambda_j = alpha_j / beta_j (when beta_j != 0)
    should satisfy det(A - lambda_j * B) = 0.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import zgegs

    np.random.seed(123)

    n = 3
    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )

    a_orig = a.copy()
    b_orig = b.copy()

    s, t, alpha, beta, vsl, vsr, info = zgegs("V", "V", a, b)
    assert info == 0

    eig_orig = np.linalg.eigvals(np.linalg.solve(b_orig, a_orig))

    eig_computed = []
    for j in range(n):
        if abs(beta[j]) > 1e-14:
            eig_computed.append(alpha[j] / beta[j])

    eig_orig_sorted = sorted(eig_orig, key=lambda x: (x.real, x.imag))
    eig_computed_sorted = sorted(eig_computed, key=lambda x: (x.real, x.imag))

    assert len(eig_computed_sorted) == n
    for i in range(n):
        assert_allclose(eig_computed_sorted[i], eig_orig_sorted[i], rtol=1e-10, atol=1e-12)


def test_zgegs_no_schur_vectors():
    """
    Test with JOBVSL='N' and JOBVSR='N' (no Schur vectors computed).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import zgegs

    np.random.seed(456)

    n = 3
    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )

    s, t, alpha, beta, vsl, vsr, info = zgegs("N", "N", a, b)

    assert info == 0
    assert s.shape == (n, n)
    assert t.shape == (n, n)
    assert alpha.shape == (n,)
    assert beta.shape == (n,)

    for j in range(n):
        for i in range(j + 1, n):
            assert abs(s[i, j]) < 1e-12, f"S not upper triangular at ({i},{j})"
            assert abs(t[i, j]) < 1e-12, f"T not upper triangular at ({i},{j})"


def test_zgegs_left_vectors_only():
    """
    Test with JOBVSL='V' and JOBVSR='N' (only left Schur vectors).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import zgegs

    np.random.seed(789)

    n = 3
    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )

    s, t, alpha, beta, vsl, vsr, info = zgegs("V", "N", a, b)

    assert info == 0
    assert vsl.shape == (n, n)

    assert_allclose(vsl @ vsl.conj().T, np.eye(n), rtol=1e-13, atol=1e-14)


def test_zgegs_right_vectors_only():
    """
    Test with JOBVSL='N' and JOBVSR='V' (only right Schur vectors).

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import zgegs

    np.random.seed(888)

    n = 3
    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )
    b = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
        np.complex128, order="F"
    )

    s, t, alpha, beta, vsl, vsr, info = zgegs("N", "V", a, b)

    assert info == 0
    assert vsr.shape == (n, n)

    assert_allclose(vsr @ vsr.conj().T, np.eye(n), rtol=1e-13, atol=1e-14)


def test_zgegs_identity_matrices():
    """
    Test with identity matrices - eigenvalues should all be 1.
    """
    from ctrlsys import zgegs

    n = 3
    a = np.eye(n, dtype=np.complex128, order="F")
    b = np.eye(n, dtype=np.complex128, order="F")

    s, t, alpha, beta, vsl, vsr, info = zgegs("V", "V", a, b)

    assert info == 0

    for j in range(n):
        if abs(beta[j]) > 1e-14:
            eig = alpha[j] / beta[j]
            assert_allclose(eig, 1.0 + 0j, rtol=1e-13, atol=1e-14)


def test_zgegs_n_equals_1():
    """
    Edge case: n=1 (scalar case).
    """
    from ctrlsys import zgegs

    a = np.array([[2.0 + 1j]], dtype=np.complex128, order="F")
    b = np.array([[1.0 + 0j]], dtype=np.complex128, order="F")

    s, t, alpha, beta, vsl, vsr, info = zgegs("V", "V", a, b)

    assert info == 0
    assert s.shape == (1, 1)
    assert t.shape == (1, 1)

    if abs(beta[0]) > 1e-14:
        eig = alpha[0] / beta[0]
        assert_allclose(eig, 2.0 + 1j, rtol=1e-13, atol=1e-14)


def test_zgegs_invalid_jobvsl():
    """
    Error test: invalid JOBVSL parameter.
    """
    from ctrlsys import zgegs

    n = 2
    a = np.eye(n, dtype=np.complex128, order="F")
    b = np.eye(n, dtype=np.complex128, order="F")

    s, t, alpha, beta, vsl, vsr, info = zgegs("X", "V", a, b)

    assert info == -1


def test_zgegs_invalid_jobvsr():
    """
    Error test: invalid JOBVSR parameter.
    """
    from ctrlsys import zgegs

    n = 2
    a = np.eye(n, dtype=np.complex128, order="F")
    b = np.eye(n, dtype=np.complex128, order="F")

    s, t, alpha, beta, vsl, vsr, info = zgegs("V", "X", a, b)

    assert info == -2
