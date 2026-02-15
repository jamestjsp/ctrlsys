"""
Tests for mb03vw: Periodic Hessenberg-triangular reduction (unblocked).

Reduces a generalized matrix product A(:,:,1)^S(1) * ... * A(:,:,K)^S(K)
to upper Hessenberg-triangular form.
"""

import numpy as np
import pytest


def test_mb03vw_basic():
    """
    Basic functionality test with K=2 matrices, signature [1, -1].

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03vw

    np.random.seed(42)
    n = 4
    k = 2
    ilo = 1
    ihi = n

    # Create random matrices in column-major order
    a = np.random.randn(n, n, k).astype(float, order='F')

    # Signatures: first positive, second negative
    s = np.array([1, -1], dtype=np.int32)

    # Initial H value (will be adjusted by routine if needed)
    h = 1

    # COMPQ = 'I' to initialize Q matrices
    a_out, q_out, h_out, info = mb03vw('I', None, 'N', n, k, h, ilo, ihi, s, a)

    assert info == 0, f"mb03vw failed with info={info}"

    # H-th matrix (h_out) should be upper Hessenberg
    # Other matrices should be upper triangular

    # Check upper triangular: below subdiagonal should be zero
    for i in range(k):
        if i + 1 != h_out:
            # Should be upper triangular - subdiagonals zero
            for row in range(2, n):
                for col in range(row - 1):
                    assert abs(a_out[row, col, i]) < 1e-14, \
                        f"Matrix {i} not upper triangular at ({row},{col})"

    # Check upper Hessenberg for H-th matrix: below first subdiagonal zero
    h_idx = h_out - 1
    for row in range(2, n):
        for col in range(row - 1):
            assert abs(a_out[row, col, h_idx]) < 1e-14, \
                f"Hessenberg matrix has nonzero at ({row},{col})"

    # Q matrices should be orthogonal
    for i in range(k):
        qt_q = q_out[:, :, i].T @ q_out[:, :, i]
        np.testing.assert_allclose(qt_q, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03vw_single_matrix():
    """
    Test with K=1 (single matrix) - edge case.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03vw

    np.random.seed(123)
    n = 3
    k = 1
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, k).astype(float, order='F')
    s = np.array([1], dtype=np.int32)
    h = 1

    a_out, q_out, h_out, info = mb03vw('I', None, 'N', n, k, h, ilo, ihi, s, a)

    assert info == 0
    assert h_out == 1

    # Single matrix becomes upper Hessenberg
    for row in range(2, n):
        for col in range(row - 1):
            assert abs(a_out[row, col, 0]) < 1e-14


def test_mb03vw_no_q_accumulation():
    """
    Test COMPQ = 'N' (no Q accumulation).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03vw

    np.random.seed(456)
    n = 3
    k = 2
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, k).astype(float, order='F')
    s = np.array([1, 1], dtype=np.int32)
    h = 1

    a_out, q_out, h_out, info = mb03vw('N', None, 'N', n, k, h, ilo, ihi, s, a)

    assert info == 0
    # q_out should be None or empty when COMPQ='N'


def test_mb03vw_compq_update():
    """
    Test COMPQ = 'U' (update existing Q matrices).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03vw

    np.random.seed(789)
    n = 3
    k = 2
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, k).astype(float, order='F')
    s = np.array([1, -1], dtype=np.int32)
    h = 1

    # Initialize Q to identity matrices
    q_in = np.zeros((n, n, k), dtype=float, order='F')
    for i in range(k):
        q_in[:, :, i] = np.eye(n)

    a_out, q_out, h_out, info = mb03vw('U', None, 'N', n, k, h, ilo, ihi, s, a, q=q_in)

    assert info == 0

    # Q matrices should remain orthogonal after update
    for i in range(k):
        qt_q = q_out[:, :, i].T @ q_out[:, :, i]
        np.testing.assert_allclose(qt_q, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03vw_triu_all():
    """
    Test TRIU = 'A' (all possible matrices triangularized in first stage).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb03vw

    np.random.seed(111)
    n = 4
    k = 3
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, k).astype(float, order='F')
    s = np.array([1, 1, -1], dtype=np.int32)
    h = 1

    a_out, q_out, h_out, info = mb03vw('I', None, 'A', n, k, h, ilo, ihi, s, a)

    assert info == 0

    # All matrices except H-th should be upper triangular
    for i in range(k):
        if i + 1 != h_out:
            for row in range(2, n):
                for col in range(row - 1):
                    assert abs(a_out[row, col, i]) < 1e-14


def test_mb03vw_partial_range():
    """
    Test with ILO > 1 and IHI < N (partial reduction).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb03vw

    np.random.seed(222)
    n = 5
    k = 2
    ilo = 2
    ihi = 4

    # Create matrices already triangular outside ILO:IHI
    a = np.zeros((n, n, k), dtype=float, order='F')
    for i in range(k):
        a[:, :, i] = np.triu(np.random.randn(n, n))
        # Make it general in ILO:IHI region
        a[ilo-1:ihi, ilo-1:ihi, i] = np.random.randn(ihi-ilo+1, ihi-ilo+1)

    s = np.array([1, 1], dtype=np.int32)
    h = 1

    a_out, q_out, h_out, info = mb03vw('I', None, 'N', n, k, h, ilo, ihi, s, a)

    assert info == 0


def test_mb03vw_ilo_equals_ihi():
    """
    Test ILO = IHI (quick return, all already triangular).

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb03vw

    np.random.seed(333)
    n = 4
    k = 2
    ilo = 2
    ihi = 2  # Same as ILO

    a = np.random.randn(n, n, k).astype(float, order='F')
    s = np.array([1, -1], dtype=np.int32)
    h = 1

    a_out, q_out, h_out, info = mb03vw('I', None, 'N', n, k, h, ilo, ihi, s, a)

    assert info == 0


def test_mb03vw_zero_dimensions():
    """
    Test N=0 edge case (quick return).
    """
    from ctrlsys import mb03vw

    # Test N=0
    n = 0
    k = 2
    ilo = 1
    ihi = 0

    a = np.zeros((0, 0, k), dtype=float, order='F')
    s = np.array([1, -1], dtype=np.int32)
    h = 1

    a_out, q_out, h_out, info = mb03vw('I', None, 'N', n, k, h, ilo, ihi, s, a)
    assert info == 0


def test_mb03vw_similarity_preservation():
    """
    Test that the transformation preserves the product's eigenvalues.

    For S(i)=1: Q_i * A_i * Q_{i+1}^T = A_i_new
    For S(i)=-1: Q_{i+1} * A_i * Q_i^T = A_i_new

    The product eigenvalues should be preserved.

    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb03vw

    np.random.seed(444)
    n = 3
    k = 2
    ilo = 1
    ihi = n

    a_orig = np.random.randn(n, n, k).astype(float, order='F')
    s = np.array([1, 1], dtype=np.int32)
    h = 1

    # Compute original product eigenvalues
    prod_orig = a_orig[:, :, 0].copy()
    for i in range(1, k):
        prod_orig = prod_orig @ a_orig[:, :, i]
    eig_orig = np.linalg.eigvals(prod_orig)

    a_out, q_out, h_out, info = mb03vw('I', None, 'N', n, k, h, ilo, ihi, s, a_orig)

    assert info == 0

    # Compute new product eigenvalues
    prod_new = a_out[:, :, 0].copy()
    for i in range(1, k):
        prod_new = prod_new @ a_out[:, :, i]
    eig_new = np.linalg.eigvals(prod_new)

    # Sort eigenvalues by real then imaginary for comparison
    eig_orig_sorted = sorted(eig_orig, key=lambda x: (x.real, x.imag))
    eig_new_sorted = sorted(eig_new, key=lambda x: (x.real, x.imag))

    np.testing.assert_allclose(
        [e.real for e in eig_orig_sorted],
        [e.real for e in eig_new_sorted],
        rtol=1e-10, atol=1e-14
    )
    np.testing.assert_allclose(
        [e.imag for e in eig_orig_sorted],
        [e.imag for e in eig_new_sorted],
        rtol=1e-10, atol=1e-14
    )


def test_mb03vw_invalid_compq():
    """
    Test invalid COMPQ parameter.
    """
    from ctrlsys import mb03vw

    n = 3
    k = 2
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, k).astype(float, order='F')
    s = np.array([1, -1], dtype=np.int32)
    h = 1

    with pytest.raises((ValueError, RuntimeError)):
        mb03vw('X', None, 'N', n, k, h, ilo, ihi, s, a)


def test_mb03vw_invalid_triu():
    """
    Test invalid TRIU parameter.
    """
    from ctrlsys import mb03vw

    n = 3
    k = 2
    ilo = 1
    ihi = n

    a = np.random.randn(n, n, k).astype(float, order='F')
    s = np.array([1, -1], dtype=np.int32)
    h = 1

    with pytest.raises((ValueError, RuntimeError)):
        mb03vw('I', None, 'X', n, k, h, ilo, ihi, s, a)
