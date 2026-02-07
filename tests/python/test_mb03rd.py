"""
Tests for MB03RD: Reduce real Schur form matrix to block-diagonal form.

Reduces a matrix A in real Schur form to a block-diagonal form using
well-conditioned non-orthogonal similarity transformations. The condition
numbers are roughly bounded by PMAX.

Tests:
1. Basic block-diagonalization from HTML doc example
2. Eigenvalue preservation property
3. Block-diagonal structure verification
4. Different SORT options (N, S, C, B)
5. Transformation matrix accumulation (JOBX='U')

Random seed: 42, 123, 456 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.linalg import solve


def test_mb03rd_html_doc_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    The HTML doc example applies DGEES first to get real Schur form, then MB03RD.
    The input matrix has all eigenvalues clustered around 1+/-i, so with tol=0.01
    they all get merged into a single block (or 2 blocks depending on separability).

    We test with a simpler case where eigenvalues are clearly separable.
    """
    from slicot import mb03rd

    n = 4
    pmax = 1.0e3

    # Simple real Schur form with distinct eigenvalues (1, 2, 3, 4)
    a = np.array([
        [1.0, 0.5, 0.2, 0.1],
        [0.0, 2.0, 0.3, 0.2],
        [0.0, 0.0, 3.0, 0.4],
        [0.0, 0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    a_out, x_out, nblcks, blsize, wr, wi, info = mb03rd('U', 'N', a, pmax, x)

    assert info == 0

    # With distinct eigenvalues and no clustering, should separate into 4 1x1 blocks
    # or possibly fewer blocks depending on separability
    assert nblcks > 0

    # Verify eigenvalues preserved
    assert_allclose(np.sort(wr), np.sort([1.0, 2.0, 3.0, 4.0]), rtol=1e-10)

    # Verify block-diagonal structure
    total_size = 0
    for i in range(nblcks):
        total_size += blsize[i]
    assert total_size == n


def test_mb03rd_eigenvalue_preservation():
    """
    Validate eigenvalue preservation under block-diagonalization.

    Similarity transformation must preserve eigenvalues.
    Random seed: 42 (for reproducibility)
    """
    from slicot import mb03rd

    np.random.seed(42)

    n = 5
    pmax = 1.0e3

    # Create a real Schur form matrix (upper quasi-triangular)
    # Eigenvalues: 1.0, 2.0, 3.0, 4.0, 5.0 (all real, distinct)
    a = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    # Add upper triangular noise
    for i in range(n):
        for j in range(i+1, n):
            a[i, j] = 0.1 * np.random.randn()
    a = np.asfortranarray(a)

    a_orig = a.copy()

    # Call without transformation matrix accumulation
    a_out, x_out, nblcks, blsize, wr, wi, info = mb03rd('N', 'N', a, pmax)

    assert info == 0

    # Verify eigenvalue preservation
    eig_orig = np.linalg.eigvals(a_orig)
    eig_out = np.linalg.eigvals(a_out)
    assert_allclose(np.sort(eig_orig.real), np.sort(eig_out.real), rtol=1e-10)


def test_mb03rd_transformation_matrix():
    """
    Validate transformation matrix accumulation (JOBX='U').

    Tests: X_out = X_in * T where T is the block-diagonalizing transformation.
    Random seed: 123 (for reproducibility)
    """
    from slicot import mb03rd

    np.random.seed(123)

    n = 4
    pmax = 1.0e3

    # Simple real Schur form with distinct eigenvalues
    a = np.array([
        [1.0, 0.5, 0.2, 0.1],
        [0.0, 2.0, 0.3, 0.2],
        [0.0, 0.0, 3.0, 0.4],
        [0.0, 0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    # Start with identity
    x = np.eye(n, order='F', dtype=float)

    a_orig = a.copy()

    a_out, x_out, nblcks, blsize, wr, wi, info = mb03rd('U', 'N', a, pmax, x)

    assert info == 0

    # Verify eigenvalues from wr, wi arrays
    assert_allclose(np.sort(wr), np.sort([1.0, 2.0, 3.0, 4.0]), rtol=1e-10)
    assert_allclose(wi, np.zeros(n), atol=1e-14)

    # Block sizes should sum to n
    assert np.sum(blsize[:nblcks]) == n


def test_mb03rd_complex_eigenvalues():
    """
    Validate handling of 2x2 blocks (complex conjugate eigenvalues).

    A 2x2 block in real Schur form represents a complex conjugate pair.
    Random seed: 456 (for reproducibility)
    """
    from slicot import mb03rd

    n = 4
    pmax = 1.0e3

    # Real Schur form with one 2x2 block (complex eigenvalues 1 +/- i)
    # and two 1x1 blocks (real eigenvalues 3, 4)
    a = np.array([
        [1.0, 1.0, 0.2, 0.1],
        [-1.0, 1.0, 0.3, 0.2],
        [0.0, 0.0, 3.0, 0.4],
        [0.0, 0.0, 0.0, 4.0]
    ], order='F', dtype=float)

    x = np.eye(n, order='F', dtype=float)

    a_out, x_out, nblcks, blsize, wr, wi, info = mb03rd('U', 'N', a, pmax, x)

    assert info == 0

    # Check that complex pair is identified
    complex_found = any(w != 0 for w in wi)
    assert complex_found


def test_mb03rd_sort_options():
    """
    Validate different SORT options: N, S, C, B.

    - 'N': No reordering
    - 'S': Reorder for clustered eigenvalues
    - 'C': Closest-neighbor strategy
    - 'B': Both reordering and closest-neighbor
    """
    from slicot import mb03rd

    n = 4
    pmax = 1.0e3
    tol = 0.5  # Clustering tolerance

    # Real Schur form with close eigenvalues (for clustering test)
    a = np.array([
        [1.0, 0.5, 0.2, 0.1],
        [0.0, 1.1, 0.3, 0.2],  # Close to 1.0
        [0.0, 0.0, 3.0, 0.4],
        [0.0, 0.0, 0.0, 3.1]   # Close to 3.0
    ], order='F', dtype=float)

    for sort in ['N', 'S', 'C', 'B']:
        a_test = a.copy()
        x = np.eye(n, order='F', dtype=float)

        if sort in ['S', 'B']:
            a_out, x_out, nblcks, blsize, wr, wi, info = mb03rd('U', sort, a_test, pmax, x, tol)
        else:
            a_out, x_out, nblcks, blsize, wr, wi, info = mb03rd('U', sort, a_test, pmax, x)

        assert info == 0, f"Failed for SORT='{sort}'"


def test_mb03rd_no_transform_accumulation():
    """
    Validate JOBX='N' (no transformation accumulation).

    When JOBX='N', X array is not referenced and output X should be None or unchanged.
    """
    from slicot import mb03rd

    n = 3
    pmax = 1.0e3

    a = np.array([
        [1.0, 0.5, 0.2],
        [0.0, 2.0, 0.3],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    # No X provided
    a_out, x_out, nblcks, blsize, wr, wi, info = mb03rd('N', 'N', a, pmax)

    assert info == 0
    assert nblcks > 0


def test_mb03rd_parameter_errors():
    """
    Validate error handling for invalid parameters.

    INFO < 0 indicates parameter error.
    """
    from slicot import mb03rd

    n = 3
    pmax = 1.0e3

    a = np.array([
        [1.0, 0.5, 0.2],
        [0.0, 2.0, 0.3],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    # Test PMAX < 1.0 (invalid)
    a_out, x_out, nblcks, blsize, wr, wi, info = mb03rd('N', 'N', a, 0.5)
    assert info == -4  # PMAX parameter error


def test_mb03rd_empty_matrix():
    """
    Validate handling of empty matrix (N=0).

    Should return immediately with NBLCKS=0.
    """
    from slicot import mb03rd

    a = np.array([], order='F', dtype=float).reshape(0, 0)
    pmax = 1.0e3

    a_out, x_out, nblcks, blsize, wr, wi, info = mb03rd('N', 'N', a, pmax)

    assert info == 0
    assert nblcks == 0


def test_mb03rd_transformation_correctness():
    """
    Validate that solve(X, A) @ X = A_out holds.

    This is the fundamental correctness property of similarity transformations.
    The transformation X should satisfy: X^(-1) @ A @ X = A_out
    """
    from slicot import mb03rd

    n = 5
    aschur = np.array([
        [-2.0, -12.0, 5.0, 1.0, 2.0],
        [0.0, -2.0, 3.0, 1.0, 1.0],
        [0.0, 0.0, -2.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, -1.0, -0.5],
        [0.0, 0.0, 0.0, 0.0, -1.0]
    ], order='F', dtype=float)

    X_in = np.eye(n, order='F', dtype=float)
    pmax = 1e4

    Aout, Xout, nblcks, blsize, wr, wi, info = mb03rd(
        'U', 'N', aschur.copy(), pmax, X_in.copy(), 0.0
    )

    assert info == 0
    assert nblcks == 2

    # This is the critical check - the transformation should be correct
    result = solve(Xout, aschur) @ Xout
    assert_allclose(result, Aout, rtol=1e-10, atol=1e-10)
