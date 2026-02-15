"""
Tests for MB04FD: Eigenvalues and orthogonal decomposition of a real
skew-Hamiltonian/skew-Hamiltonian pencil.

Computes eigenvalues of aS - bT where:
  S = [[A, D], [E, A']] with D, E skew-symmetric
  T = [[B, F], [G, B']] with F, G skew-symmetric

Tests:
1. Basic case from HTML docs (N=8, JOB='T', COMPQ='I')
2. Edge case: N=0 (quick return)
3. Mathematical property: orthogonality of Q matrix
4. Eigenvalues only mode (JOB='E')
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mb04fd_html_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Input: N=8 (even), JOB='T', COMPQ='I'
    Matrices: A (4x4), DE (4x5), B (4x4), FG (4x5)

    Data parsed from HTML doc (read row-by-row).
    """
    from ctrlsys import mb04fd

    n = 8
    m = n // 2  # m = 4

    # Matrix A (4x4, read row-by-row from HTML)
    a = np.array([
        [0.8147, 0.6323, 0.9575, 0.9571],
        [0.9057, 0.0975, 0.9648, 0.4853],
        [0.1269, 0.2784, 0.1576, 0.8002],
        [0.9133, 0.5468, 0.9705, 0.1418]
    ], order='F', dtype=float)

    # Matrix DE (4x5): column 1 = strictly lower E, columns 2-5 = strictly upper D
    # Row-wise from HTML:
    # 0.4217  0.6557  0.6787  0.6554  0.2769
    # 0.9157  0.0357  0.7577  0.1711  0.0461
    # 0.7922  0.8491  0.7431  0.7060  0.0971
    # 0.9594  0.9339  0.3922  0.0318  0.8234
    de = np.array([
        [0.4217, 0.6557, 0.6787, 0.6554, 0.2769],
        [0.9157, 0.0357, 0.7577, 0.1711, 0.0461],
        [0.7922, 0.8491, 0.7431, 0.7060, 0.0971],
        [0.9594, 0.9339, 0.3922, 0.0318, 0.8234]
    ], order='F', dtype=float)

    # Matrix B (4x4, read row-by-row from HTML)
    b = np.array([
        [0.6948, 0.4387, 0.1868, 0.7093],
        [0.3170, 0.3815, 0.4897, 0.7546],
        [0.9502, 0.7655, 0.4455, 0.2760],
        [0.0344, 0.7951, 0.6463, 0.6797]
    ], order='F', dtype=float)

    # Matrix FG (4x5): column 1 = strictly lower G, columns 2-5 = strictly upper F
    # Row-wise from HTML:
    # 0.6550  0.9597  0.7512  0.8909  0.1492
    # 0.1626  0.3403  0.2550  0.9592  0.2575
    # 0.1189  0.5852  0.5059  0.5472  0.8407
    # 0.4983  0.2238  0.6990  0.1386  0.2542
    fg = np.array([
        [0.6550, 0.9597, 0.7512, 0.8909, 0.1492],
        [0.1626, 0.3403, 0.2550, 0.9592, 0.2575],
        [0.1189, 0.5852, 0.5059, 0.5472, 0.8407],
        [0.4983, 0.2238, 0.6990, 0.1386, 0.2542]
    ], order='F', dtype=float)

    # Expected A output from HTML (upper triangular after JOB='T')
    a_expected = np.array([
        [0.0550, -0.3064, 0.1543, 0.2170],
        [0.0000,  1.2189, 0.3267, -1.3622],
        [0.0000,  0.0000, 0.7734, -0.6215],
        [0.0000,  0.0000, 0.0000,  2.5172]
    ], order='F', dtype=float)

    # Expected B output from HTML (upper quasi-triangular)
    b_expected = np.array([
        [0.8482, -0.4425, -0.3643, 0.8333],
        [0.0000, -0.5919, -0.0987, -0.7923],
        [0.0000,  0.0000, 1.1021, 0.1926],
        [0.0000,  0.0000, 0.0000, 1.9788]
    ], order='F', dtype=float)

    # Expected eigenvalues from HTML
    alphar_expected = np.array([0.8482, -0.5919, 1.1021, 1.9788])
    alphai_expected = np.array([0.0000, 0.0000, 0.0000, 0.0000])
    beta_expected = np.array([0.0550, 1.2189, 0.7734, 2.5172])

    # Call mb04fd with JOB='T', COMPQ='I'
    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fd(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04FD returned info={info}"

    # Verify A is upper triangular
    for i in range(1, m):
        for j in range(i):
            assert abs(a_out[i, j]) < 1e-10, f"A[{i},{j}]={a_out[i,j]} should be zero"

    # Verify B is upper triangular
    for i in range(1, m):
        for j in range(i):
            assert abs(b_out[i, j]) < 1e-10, f"B[{i},{j}]={b_out[i,j]} should be zero"

    # Check eigenvalues (alphar/beta ratios)
    # Eigenvalue ordering may differ from reference, so compare sorted values
    eigenvalues_expected = np.sort(alphar_expected / beta_expected)
    eigenvalues_actual = np.sort(alphar / beta)
    assert_allclose(eigenvalues_actual, eigenvalues_expected, rtol=1e-2, atol=1e-3)
    # All eigenvalues should be real for this example
    assert_allclose(alphai, alphai_expected, atol=1e-4)

    # Verify Q is orthogonal: Q'*Q = I
    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb04fd_q_orthogonality():
    """
    Validate mathematical property: Q should be orthogonal (Q'*Q = I).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04fd

    np.random.seed(42)
    n = 8
    m = n // 2

    # Random A matrix
    a = np.random.randn(m, m).astype(float, order='F')

    # Random B matrix
    b = np.random.randn(m, m).astype(float, order='F')

    # DE with skew-symmetric structure
    de = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            de[i, j] = np.random.randn()  # E (strictly lower)
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            de[i, j] = np.random.randn()  # D (strictly upper starting col 2)

    # FG with skew-symmetric structure
    fg = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            fg[i, j] = np.random.randn()  # G (strictly lower)
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            fg[i, j] = np.random.randn()  # F (strictly upper starting col 2)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fd(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04FD returned info={info}"

    # Q should be orthogonal: Q' * Q = I
    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)

    # Q * Q' = I
    qqt = q_out @ q_out.T
    assert_allclose(qqt, np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb04fd_eigenvalues_only():
    """
    Test JOB='E' mode: compute eigenvalues only.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04fd

    np.random.seed(456)
    n = 6
    m = n // 2

    # Random matrices
    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')

    de = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            de[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            de[i, j] = np.random.randn()

    fg = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            fg[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            fg[i, j] = np.random.randn()

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fd(
        'E', 'N', a.copy(), de.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04FD returned info={info}"
    assert len(alphar) == m
    assert len(alphai) == m
    assert len(beta) == m


def test_mb04fd_n_zero():
    """
    Edge case: N=0 should return immediately with info=0.
    """
    from ctrlsys import mb04fd

    n = 0
    a = np.array([], order='F', dtype=float).reshape(0, 0)
    de = np.array([], order='F', dtype=float).reshape(0, 1)
    b = np.array([], order='F', dtype=float).reshape(0, 0)
    fg = np.array([], order='F', dtype=float).reshape(0, 1)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fd(
        'T', 'I', a, de, b, fg
    )

    assert info == 0, f"MB04FD returned info={info}"
    assert len(alphar) == 0
    assert len(alphai) == 0
    assert len(beta) == 0


def test_mb04fd_invalid_n_odd():
    """
    Test error handling: N must be even. N=5 should return info=-3.
    """
    from ctrlsys import mb04fd

    np.random.seed(888)
    n = 5
    m = n // 2 + 1  # Need to make arrays fit n
    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    de = np.zeros((m, m + 1), order='F', dtype=float)
    fg = np.zeros((m, m + 1), order='F', dtype=float)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fd(
        'T', 'I', a, de, b, fg, n=n
    )

    assert info == -3, f"Expected info=-3 for odd N, got info={info}"


def test_mb04fd_compq_update():
    """
    Test COMPQ='U' mode: update existing Q0 matrix.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04fd

    np.random.seed(789)
    n = 6
    m = n // 2

    # Random matrices
    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')

    de = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            de[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            de[i, j] = np.random.randn()

    fg = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            fg[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            fg[i, j] = np.random.randn()

    # Initial Q0 = identity
    q0 = np.eye(n, order='F', dtype=float)

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fd(
        'T', 'U', a.copy(), de.copy(), b.copy(), fg.copy(), q=q0.copy()
    )

    assert info == 0, f"MB04FD returned info={info}"

    # Q should be orthogonal: Q' * Q = I
    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb04fd_eigenvalue_structure():
    """
    Validate mathematical property: eigenvalues occur in pairs.

    For skew-Hamiltonian/skew-Hamiltonian pencils, every eigenvalue
    has even multiplicity. We test that eigenvalue count is correct.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import mb04fd

    np.random.seed(555)
    n = 8
    m = n // 2

    # Random matrices
    a = np.random.randn(m, m).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')

    de = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            de[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            de[i, j] = np.random.randn()

    fg = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            fg[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            fg[i, j] = np.random.randn()

    a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, iwork, info = mb04fd(
        'T', 'I', a.copy(), de.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04FD returned info={info}"

    # Each eigenvalue stored once (due to even multiplicity),
    # so we have N/2 eigenvalues for N-by-N pencil
    assert len(alphar) == m
    assert len(alphai) == m
    assert len(beta) == m

    # Complex eigenvalues should come in conjugate pairs (alphai[j] > 0 means pair)
    j = 0
    while j < m:
        if alphai[j] != 0:
            # Complex conjugate pair
            assert j + 1 < m, "Complex eigenvalue pair incomplete"
            assert_allclose(alphar[j], alphar[j + 1], rtol=1e-14)
            assert_allclose(alphai[j], -alphai[j + 1], rtol=1e-14)
            j += 2
        else:
            j += 1
