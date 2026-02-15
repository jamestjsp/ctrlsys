"""
Tests for MB03BD: Eigenvalues of periodic Hessenberg matrix product.

Computes eigenvalues of generalized matrix product using double-shift periodic
QZ algorithm. Can reduce to periodic Schur form.

Eigenvalue format: (ALPHAR + ALPHAI*sqrt(-1))/BETA * BASE^SCAL

Tests:
1. Basic case from HTML docs (K=3, N=3)
2. Single factor case (K=1)
3. Mathematical property: eigenvalue trace preservation
4. Edge case: N=0 (quick return)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def test_mb03bd_html_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Input: K=3 factors, N=3, H=2 (Hessenberg index), JOB='S', COMPQ='I'
    Matrices: A[0] upper triangular, A[1] upper Hessenberg, A[2] upper triangular
    Signatures: S = [-1, 1, -1]

    Expected eigenvalues:
    (0.3230 + 0.5694j)/1.0 * 2^0 = 0.3230 + 0.5694j
    (0.3230 - 0.5694j)/1.0 * 2^0 = 0.3230 - 0.5694j
    (-0.8752)/1.0 * 2^(-1) = -0.4376
    """
    from ctrlsys import mb03bd

    k = 3
    n = 3
    h = 2
    ilo = 1
    ihi = 3
    s = np.array([-1, 1, -1], dtype=np.int32)

    # Matrix A[0] (upper triangular, read row-by-row)
    a0 = np.array([
        [2.0, 0.0, 1.0],
        [0.0, -2.0, -1.0],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    # Matrix A[1] (upper Hessenberg)
    a1 = np.array([
        [1.0, 2.0, 0.0],
        [4.0, -1.0, 3.0],
        [0.0, 3.0, 1.0]
    ], order='F', dtype=float)

    # Matrix A[2] (upper triangular)
    a2 = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 4.0, -1.0],
        [0.0, 0.0, -2.0]
    ], order='F', dtype=float)

    # Stack into 3D array (N x N x K)
    a = np.zeros((n, n, k), order='F', dtype=float)
    a[:, :, 0] = a0
    a[:, :, 1] = a1
    a[:, :, 2] = a2

    # Call mb03bd
    a_out, q_out, alphar, alphai, beta, scal, iwarn, info = mb03bd(
        'S', 'C', 'I', k, n, h, ilo, ihi, s, a
    )

    assert info == 0, f"MB03BD returned info={info}"
    assert iwarn == 0, f"MB03BD returned iwarn={iwarn}"

    # Expected eigenvalue representations from HTML docs (rtol=1e-3 for 4-decimal precision)
    expected_alphar = np.array([0.3230, 0.3230, -0.8752])
    expected_alphai = np.array([0.5694, -0.5694, 0.0])
    expected_beta = np.array([1.0, 1.0, 1.0])
    expected_scal = np.array([0, 0, -1], dtype=np.int32)

    # Verify eigenvalues
    # Compute complex eigenvalues from alphar/alphai/beta/scal
    # Note: slices 2^scal so we need base=2
    base = 2.0
    
    eig = []
    i = 0
    while i < n:
        if alphai[i] == 0:
            # Real
            val = (alphar[i] / beta[i]) * (base ** scal[i])
            eig.append(val)
            i += 1
        else:
            # Complex pair
            val = (complex(alphar[i], alphai[i]) / beta[i]) * (base ** scal[i])
            eig.append(val)
            eig.append(val.conjugate())
            i += 2
            
    eig = np.array(sorted(eig, key=lambda x: (x.real, x.imag)))
    expected = np.array(sorted([0.3230 + 0.5694j, 0.3230 - 0.5694j, -0.4376], key=lambda x: (x.real, x.imag)))
    
    assert_allclose(eig, expected, rtol=1e-3)


def test_mb03bd_single_factor():
    """
    Validate K=1 (single matrix eigenvalue problem).

    Single Hessenberg matrix - eigenvalues should match numpy.linalg.eigvals.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03bd

    np.random.seed(42)
    n = 4
    k = 1
    h = 1
    ilo = 1
    ihi = n
    s = np.array([1], dtype=np.int32)

    # Create upper Hessenberg matrix
    a = np.zeros((n, n, k), order='F', dtype=float)
    # Start with random, then zero below subdiagonal
    a[:, :, 0] = np.random.randn(n, n)
    for i in range(2, n):
        for j in range(i - 1):
            a[i, j, 0] = 0.0

    # Compute eigenvalues with numpy for comparison
    eig_np = np.linalg.eigvals(a[:, :, 0])

    # Call mb03bd (eigenvalues only)
    a_out, q_out, alphar, alphai, beta, scal, iwarn, info = mb03bd(
        'E', 'C', 'N', k, n, h, ilo, ihi, s, a.copy()
    )

    assert info == 0, f"MB03BD returned info={info}"

    # Reconstruct eigenvalues: (alphar + alphai*j) / beta * 2^scal
    base = 2.0
    eig_slicot = np.where(
        beta != 0,
        (alphar + 1j * alphai) / beta * (base ** scal),
        np.inf
    )

    # Sort by real part, then imaginary part
    eig_np_sorted = sorted(eig_np, key=lambda x: (x.real, x.imag))
    eig_slicot_sorted = sorted(eig_slicot, key=lambda x: (x.real, x.imag))

    assert_allclose(
        np.array(eig_slicot_sorted),
        np.array(eig_np_sorted),
        rtol=1e-10
    )


def test_mb03bd_eigenvalue_product_trace():
    """
    Validate mathematical property: trace of matrix product equals sum of eigenvalues.

    For product P = A_1^{s_1} * A_2^{s_2} * ... * A_k^{s_k}:
    trace(P) = sum(eigenvalues)

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03bd

    np.random.seed(123)
    n = 3
    k = 2
    h = 1
    ilo = 1
    ihi = n
    s = np.array([1, 1], dtype=np.int32)

    # Create matrices (A[0] Hessenberg, A[1] triangular)
    a = np.zeros((n, n, k), order='F', dtype=float)

    # A[0]: Hessenberg
    a[:, :, 0] = np.random.randn(n, n)
    for i in range(2, n):
        for j in range(i - 1):
            a[i, j, 0] = 0.0

    # A[1]: Upper triangular
    a[:, :, 1] = np.triu(np.random.randn(n, n))

    # Compute product P = A[0] * A[1] (both have positive signature)
    product = a[:, :, 0] @ a[:, :, 1]
    trace_product = np.trace(product)

    # Call mb03bd
    a_out, q_out, alphar, alphai, beta, scal, iwarn, info = mb03bd(
        'E', 'C', 'N', k, n, h, ilo, ihi, s, a.copy()
    )

    assert info == 0

    # Sum of eigenvalues = trace
    base = 2.0
    eig_sum = 0.0
    for i in range(n):
        if beta[i] != 0:
            eig_sum += (alphar[i] / beta[i]) * (base ** scal[i])

    assert_allclose(eig_sum, trace_product, rtol=1e-10)


def test_mb03bd_schur_form_output():
    """
    Validate JOB='S' produces quasi-triangular output.

    After JOB='S': A[:,:,H] should be quasi-triangular (upper triangular
    with possible 2x2 blocks on diagonal for complex eigenvalues).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03bd

    np.random.seed(456)
    n = 4
    k = 2
    h = 1
    ilo = 1
    ihi = n
    s = np.array([1, 1], dtype=np.int32)

    a = np.zeros((n, n, k), order='F', dtype=float)
    a[:, :, 0] = np.random.randn(n, n)
    for i in range(2, n):
        for j in range(i - 1):
            a[i, j, 0] = 0.0
    a[:, :, 1] = np.triu(np.random.randn(n, n))

    # Call with JOB='S' for Schur form
    a_out, q_out, alphar, alphai, beta, scal, iwarn, info = mb03bd(
        'S', 'C', 'I', k, n, h, ilo, ihi, s, a.copy()
    )

    assert info == 0
    assert iwarn == 0

    # Check that A[H] is quasi-triangular
    # (Checking if elements below 1st subdiagonal are zero)
    ah = a_out[:, :, h - 1] # H is 1-based index
    lower_tri = np.tril(ah, -2) # Elements below subdiagonal
    assert_allclose(lower_tri, 0, atol=1e-10)


def test_mb03bd_orthogonal_factors():
    """
    Validate COMPQ='I' produces orthogonal Q matrices.

    Mathematical property: Q^T * Q = I for each Q factor.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03bd

    np.random.seed(789)
    n = 3
    k = 2
    h = 1
    ilo = 1
    ihi = n
    s = np.array([1, 1], dtype=np.int32)

    a = np.zeros((n, n, k), order='F', dtype=float)
    a[:, :, 0] = np.random.randn(n, n)
    for i in range(2, n):
        for j in range(i - 1):
            a[i, j, 0] = 0.0
    a[:, :, 1] = np.triu(np.random.randn(n, n))

    # Call with COMPQ='I'
    a_out, q_out, alphar, alphai, beta, scal, iwarn, info = mb03bd(
        'S', 'C', 'I', k, n, h, ilo, ihi, s, a.copy()
    )

    assert info == 0

    # Verify orthogonality of each Q factor
    for i in range(k):
        q_i = q_out[:, :, i]
        qtq = q_i.T @ q_i
        assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb03bd_zero_dimension():
    """
    Validate N=0 edge case (quick return).
    """
    from ctrlsys import mb03bd

    k = 2
    n = 0
    h = 1
    ilo = 1
    ihi = 0
    s = np.array([1, 1], dtype=np.int32)

    a = np.zeros((1, 1, k), order='F', dtype=float)

    a_out, q_out, alphar, alphai, beta, scal, iwarn, info = mb03bd(
        'E', 'C', 'N', k, n, h, ilo, ihi, s, a
    )

    assert info == 0
    assert len(alphar) == 0
    assert len(alphai) == 0



def test_mb03bd_negative_signature():
    """
    Validate with negative signatures (inverse matrices in product).

    Product: A[0]^(-1) * A[1] * A[2]^(-1)
    S = [-1, 1, -1]

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb03bd

    np.random.seed(999)
    n = 3
    k = 3
    h = 2
    ilo = 1
    ihi = n
    s = np.array([-1, 1, -1], dtype=np.int32)

    a = np.zeros((n, n, k), order='F', dtype=float)

    # A[0]: Upper triangular (invertible - nonzero diagonal)
    a[:, :, 0] = np.triu(np.random.randn(n, n))
    np.fill_diagonal(a[:, :, 0], np.abs(np.diag(a[:, :, 0])) + 1.0)

    # A[1]: Hessenberg (H=2)
    a[:, :, 1] = np.random.randn(n, n)
    for i in range(2, n):
        for j in range(i - 1):
            a[i, j, 1] = 0.0

    # A[2]: Upper triangular (invertible)
    a[:, :, 2] = np.triu(np.random.randn(n, n))
    np.fill_diagonal(a[:, :, 2], np.abs(np.diag(a[:, :, 2])) + 1.0)

    a_out, q_out, alphar, alphai, beta, scal, iwarn, info = mb03bd(
        'E', 'C', 'N', k, n, h, ilo, ihi, s, a.copy()
    )

    assert info == 0

    # Compute actual product: A[0]^(-1) * A[1] * A[2]^(-1)
    product = np.linalg.inv(a[:, :, 0]) @ a[:, :, 1] @ np.linalg.inv(a[:, :, 2])
    eig_np = np.linalg.eigvals(product)

    # Reconstruct eigenvalues
    base = 2.0
    eig_slicot = []
    for i in range(n):
        if beta[i] != 0:
            eig_ctrlsys.append((alphar[i] + 1j * alphai[i]) / beta[i] * (base ** scal[i]))
        else:
            eig_ctrlsys.append(np.inf)
    eig_slicot = np.array(eig_slicot)

    # Sort and compare
    eig_np_sorted = sorted(eig_np, key=lambda x: (x.real, x.imag))
    eig_slicot_sorted = sorted(eig_slicot, key=lambda x: (x.real, x.imag))

    assert_allclose(
        np.array(eig_slicot_sorted),
        np.array(eig_np_sorted),
        rtol=1e-8
    )
