"""
Tests for MB03YT - Periodic Schur factorization of 2x2 matrix pair.

MB03YT computes the periodic Schur factorization of a real 2x2 matrix pair (A,B)
where B is upper triangular.

Test data sources:
- Mathematical properties of periodic Schur form
- Known special cases
"""

import numpy as np
import pytest

from ctrlsys import mb03yt


def test_mb03yt_real_eigenvalues():
    """
    Test with matrix pair having real eigenvalues.
    """
    a = np.array([
        [4.0, 1.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a_out, b_out, alphar, alphai, beta, csl, snl, csr, snr = mb03yt(a, b)

    assert abs(alphai[0]) < 1e-10, "Should have real eigenvalues"
    assert abs(alphai[1]) < 1e-10, "Should have real eigenvalues"


def test_mb03yt_complex_eigenvalues():
    """
    Test with matrix pair having complex conjugate eigenvalues.
    """
    a = np.array([
        [0.0, -1.0],
        [1.0, 0.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a_out, b_out, alphar, alphai, beta, csl, snl, csr, snr = mb03yt(a, b)

    assert abs(alphai[0]) > 0, "Should have complex eigenvalues"
    assert abs(alphai[0] + alphai[1]) < 1e-10, "Imaginary parts should be conjugate"


def test_mb03yt_rotation_orthogonality():
    """
    Test that rotation matrices are orthogonal and reconstruct A_out, B_out.
    """
    np.random.seed(42)
    a = np.random.randn(2, 2).astype(float, order='F')
    b = np.array([
        [np.random.randn(), np.random.randn()],
        [0.0, np.random.randn()]
    ], order='F', dtype=float)

    a_orig = a.copy()
    b_orig = b.copy()

    a_out, b_out, alphar, alphai, beta, csl, snl, csr, snr = mb03yt(a, b)

    assert abs(csl**2 + snl**2 - 1.0) < 1e-14, "Left rotation not orthogonal"
    assert abs(csr**2 + snr**2 - 1.0) < 1e-14, "Right rotation not orthogonal"

    Ql = np.array([[csl, snl], [-snl, csl]])
    Qr = np.array([[csr, -snr], [snr, csr]])
    np.testing.assert_allclose(Ql @ a_orig @ Qr, a_out, atol=1e-12)

    Ql_b = np.array([[csr, snr], [-snr, csr]])
    Qr_b = np.array([[csl, -snl], [snl, csl]])
    np.testing.assert_allclose(Ql_b @ b_orig @ Qr_b, b_out, atol=1e-12)


def test_mb03yt_diagonal_b():
    """
    Test with diagonal B matrix.
    """
    a = np.array([
        [2.0, 1.0],
        [1.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    a_out, b_out, alphar, alphai, beta, csl, snl, csr, snr = mb03yt(a, b)

    assert len(beta) == 2


def test_mb03yt_identity_b():
    """
    Test with identity B - eigenvalues should match A's eigenvalues.
    """
    a = np.array([
        [1.0, 4.0],
        [0.0, 2.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    a_out, b_out, alphar, alphai, beta, csl, snl, csr, snr = mb03yt(a, b)

    if abs(alphai[0]) < 1e-10:
        eig1 = alphar[0] / beta[0] if abs(beta[0]) > 1e-15 else 0
        eig2 = alphar[1] / beta[1] if abs(beta[1]) > 1e-15 else 0
        assert abs(eig1 - 1.0) < 1e-10 or abs(eig1 - 2.0) < 1e-10
        assert abs(eig2 - 1.0) < 1e-10 or abs(eig2 - 2.0) < 1e-10


def test_mb03yt_random():
    """
    Test with random matrices.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    for _ in range(5):
        a = np.random.randn(2, 2).astype(float, order='F')
        b = np.array([
            [np.random.randn() + 0.5, np.random.randn()],
            [0.0, np.random.randn() + 0.5]
        ], order='F', dtype=float)

        a_out, b_out, alphar, alphai, beta, csl, snl, csr, snr = mb03yt(a, b)

        assert abs(csl**2 + snl**2 - 1.0) < 1e-14
        assert abs(csr**2 + snr**2 - 1.0) < 1e-14
