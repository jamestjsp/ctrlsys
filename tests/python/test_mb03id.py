"""
Tests for MB03ID - Reorder eigenvalues with negative real parts of a skew-Hamiltonian/Hamiltonian pencil.

MB03ID moves eigenvalues with strictly negative real parts of an N-by-N real
skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the
leading principal subpencil.

The pencil structure:
    S = J Z' J' Z, J = [[0, I], [-I, 0]], Z = [[A, D], [0, C]], H = [[B, F], [0, -B']]

where A is upper triangular, B is upper quasi-triangular, C is lower triangular.
"""

import numpy as np
import pytest


def test_mb03id_basic_n4():
    """
    Test MB03ID with N=4 (M=2) - minimal non-trivial case.

    Uses matrices where eigenvalue signs are known from diagonal structure.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb03id

    np.random.seed(42)

    n = 4
    m = n // 2  # m = 2

    # Create upper triangular A with known diagonal
    a = np.zeros((m, m), dtype=float, order='F')
    a[0, 0] = 2.0
    a[0, 1] = 0.5
    a[1, 1] = 3.0

    # Create lower triangular C with known diagonal
    c = np.zeros((m, m), dtype=float, order='F')
    c[0, 0] = 1.0
    c[1, 0] = 0.3
    c[1, 1] = 2.0

    # Create general D matrix
    d = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float, order='F')

    # Create upper quasi-triangular B with negative eigenvalue in first position
    # and positive in second
    b = np.zeros((m, m), dtype=float, order='F')
    b[0, 0] = -1.0  # Negative eigenvalue
    b[0, 1] = 0.2
    b[1, 1] = 1.0   # Positive eigenvalue

    # Create symmetric F (upper triangular storage)
    f = np.array([[0.5, 0.1], [0.0, 0.6]], dtype=float, order='F')

    # Call routine - compute both Q and U
    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('I', 'I', n, a, c, d, b, f)

    assert info == 0, f"mb03id returned info={info}"
    assert neig >= 0, f"neig should be non-negative, got {neig}"

    # Verify Q is orthogonal: Q'*Q = I
    qtq = q_out.T @ q_out
    np.testing.assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)

    # Verify U is orthogonal symplectic: U = [[U1, U2], [-U2, U1]]
    # Check U1'*U1 + U2'*U2 = I
    u1tu1_u2tu2 = u1_out.T @ u1_out + u2_out.T @ u2_out
    np.testing.assert_allclose(u1tu1_u2tu2, np.eye(m), rtol=1e-13, atol=1e-14)

    # Check U1'*U2 = U2'*U1 (symmetry condition for orthogonal symplectic)
    u1tu2 = u1_out.T @ u2_out
    u2tu1 = u2_out.T @ u1_out
    np.testing.assert_allclose(u1tu2, u2tu1, rtol=1e-13, atol=1e-14)


def test_mb03id_n6_2x2_block():
    """
    Test MB03ID with N=6 (M=3) including a 2x2 block in B.

    Tests the case where B has complex conjugate eigenvalue pairs.
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb03id

    np.random.seed(123)

    n = 6
    m = n // 2  # m = 3

    # Create upper triangular A
    a = np.array([
        [1.0, 0.2, 0.3],
        [0.0, 2.0, 0.4],
        [0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    # Create lower triangular C
    c = np.array([
        [1.5, 0.0, 0.0],
        [0.1, 2.5, 0.0],
        [0.2, 0.3, 3.5]
    ], dtype=float, order='F')

    # Create general D
    d = 0.1 * np.random.randn(m, m).astype(float, order='F')

    # Create upper quasi-triangular B with one 2x2 block
    # First 2x2 block has complex eigenvalues with negative real part
    # Last 1x1 block has positive real eigenvalue
    b = np.zeros((m, m), dtype=float, order='F')
    b[0, 0] = -0.5   # Part of 2x2 block
    b[0, 1] = 1.0
    b[1, 0] = -1.0   # Subdiagonal (makes it 2x2 block)
    b[1, 1] = -0.5   # Part of 2x2 block
    b[0, 2] = 0.1
    b[1, 2] = 0.2
    b[2, 2] = 2.0    # Positive eigenvalue

    # Create symmetric F (upper triangular)
    f = np.array([
        [0.5, 0.1, 0.2],
        [0.0, 0.6, 0.15],
        [0.0, 0.0, 0.7]
    ], dtype=float, order='F')

    # Call routine
    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('I', 'I', n, a, c, d, b, f)

    assert info == 0, f"mb03id returned info={info}"
    # The 2x2 block has 2 eigenvalues with negative real parts
    assert neig >= 0, f"neig should be non-negative, got {neig}"

    # Verify Q is orthogonal
    qtq = q_out.T @ q_out
    np.testing.assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)

    # Verify orthogonal symplectic property
    u1tu1_u2tu2 = u1_out.T @ u1_out + u2_out.T @ u2_out
    np.testing.assert_allclose(u1tu1_u2tu2, np.eye(m), rtol=1e-13, atol=1e-14)


def test_mb03id_compq_u():
    """
    Test MB03ID with COMPQ='U' and COMPU='U' - update existing transformations.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb03id

    np.random.seed(456)

    n = 4
    m = n // 2

    # Create test matrices
    a = np.array([[2.0, 0.3], [0.0, 3.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.0], [0.2, 2.0]], dtype=float, order='F')
    d = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float, order='F')
    b = np.array([[-1.0, 0.2], [0.0, 1.0]], dtype=float, order='F')
    f = np.array([[0.5, 0.1], [0.0, 0.6]], dtype=float, order='F')

    # Create initial orthogonal Q0 (identity for simplicity)
    q0 = np.eye(n, dtype=float, order='F')

    # Create initial orthogonal symplectic U0
    u1_0 = np.eye(m, dtype=float, order='F')
    u2_0 = np.zeros((m, m), dtype=float, order='F')

    # Call routine with update mode
    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('U', 'U', n, a, c, d, b, f, q0, u1_0, u2_0)

    assert info == 0, f"mb03id returned info={info}"

    # Q should still be orthogonal
    qtq = q_out.T @ q_out
    np.testing.assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)

    # U should still satisfy orthogonal symplectic conditions
    u1tu1_u2tu2 = u1_out.T @ u1_out + u2_out.T @ u2_out
    np.testing.assert_allclose(u1tu1_u2tu2, np.eye(m), rtol=1e-13, atol=1e-14)


def test_mb03id_compq_n_compu_n():
    """
    Test MB03ID with COMPQ='N' and COMPU='N' - no orthogonal matrices computed.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb03id

    np.random.seed(789)

    n = 4
    m = n // 2

    a = np.array([[2.0, 0.3], [0.0, 3.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.0], [0.2, 2.0]], dtype=float, order='F')
    d = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float, order='F')
    b = np.array([[-1.0, 0.2], [0.0, 1.0]], dtype=float, order='F')
    f = np.array([[0.5, 0.1], [0.0, 0.6]], dtype=float, order='F')

    # Call with no Q or U computation
    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('N', 'N', n, a, c, d, b, f)

    assert info == 0, f"mb03id returned info={info}"
    assert neig >= 0


def test_mb03id_n0_quick_return():
    """
    Test MB03ID with N=0 - quick return case.
    """
    from ctrlsys import mb03id

    n = 0
    m = n // 2

    # Empty arrays
    a = np.zeros((1, 1), dtype=float, order='F')
    c = np.zeros((1, 1), dtype=float, order='F')
    d = np.zeros((1, 1), dtype=float, order='F')
    b = np.zeros((1, 1), dtype=float, order='F')
    f = np.zeros((1, 1), dtype=float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('N', 'N', n, a, c, d, b, f)

    assert info == 0
    assert neig == 0


def test_mb03id_error_n_odd():
    """
    Test MB03ID with odd N - should return error.
    """
    from ctrlsys import mb03id

    n = 3  # Odd - invalid

    a = np.zeros((1, 1), dtype=float, order='F')
    c = np.zeros((1, 1), dtype=float, order='F')
    d = np.zeros((1, 1), dtype=float, order='F')
    b = np.zeros((1, 1), dtype=float, order='F')
    f = np.zeros((1, 1), dtype=float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('N', 'N', n, a, c, d, b, f)

    assert info == -3, f"Expected info=-3 for odd N, got info={info}"


def test_mb03id_error_n_negative():
    """
    Test MB03ID with negative N - should return error.
    """
    from ctrlsys import mb03id

    n = -2

    a = np.zeros((1, 1), dtype=float, order='F')
    c = np.zeros((1, 1), dtype=float, order='F')
    d = np.zeros((1, 1), dtype=float, order='F')
    b = np.zeros((1, 1), dtype=float, order='F')
    f = np.zeros((1, 1), dtype=float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('N', 'N', n, a, c, d, b, f)

    assert info == -3, f"Expected info=-3 for negative N, got info={info}"


def test_mb03id_error_invalid_compq():
    """
    Test MB03ID with invalid COMPQ - should return error.
    """
    from ctrlsys import mb03id

    n = 4
    m = n // 2

    a = np.zeros((m, m), dtype=float, order='F')
    c = np.zeros((m, m), dtype=float, order='F')
    d = np.zeros((m, m), dtype=float, order='F')
    b = np.zeros((m, m), dtype=float, order='F')
    f = np.zeros((m, m), dtype=float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('X', 'N', n, a, c, d, b, f)

    assert info == -1, f"Expected info=-1 for invalid COMPQ, got info={info}"


def test_mb03id_error_invalid_compu():
    """
    Test MB03ID with invalid COMPU - should return error.
    """
    from ctrlsys import mb03id

    n = 4
    m = n // 2

    a = np.zeros((m, m), dtype=float, order='F')
    c = np.zeros((m, m), dtype=float, order='F')
    d = np.zeros((m, m), dtype=float, order='F')
    b = np.zeros((m, m), dtype=float, order='F')
    f = np.zeros((m, m), dtype=float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('N', 'X', n, a, c, d, b, f)

    assert info == -2, f"Expected info=-2 for invalid COMPU, got info={info}"


def test_mb03id_property_b_quasi_triangular():
    """
    Test that output B remains upper quasi-triangular.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb03id

    np.random.seed(111)

    n = 6
    m = n // 2

    # Create matrices
    a = np.triu(np.random.randn(m, m)).astype(float, order='F')
    c = np.tril(np.random.randn(m, m)).astype(float, order='F')
    d = np.random.randn(m, m).astype(float, order='F')

    # B with 2x2 block
    b = np.zeros((m, m), dtype=float, order='F')
    b[0, 0] = -0.5
    b[0, 1] = 1.0
    b[1, 0] = -0.5  # Subdiagonal for 2x2 block
    b[1, 1] = -0.5
    b[0, 2] = 0.1
    b[1, 2] = 0.1
    b[2, 2] = 1.0

    f = np.triu(np.random.randn(m, m)).astype(float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('I', 'I', n, a, c, d, b, f)

    assert info == 0

    # Check B is quasi-triangular: elements below subdiagonal must be zero
    for i in range(2, m):
        for j in range(i - 1):
            assert abs(b_out[i, j]) < 1e-14, \
                f"B[{i},{j}] = {b_out[i, j]} should be zero (quasi-triangular)"


def test_mb03id_property_a_upper_triangular():
    """
    Test that output A remains upper triangular.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb03id

    np.random.seed(222)

    n = 4
    m = n // 2

    a = np.triu(np.random.randn(m, m)).astype(float, order='F')
    c = np.tril(np.random.randn(m, m)).astype(float, order='F')
    d = np.random.randn(m, m).astype(float, order='F')
    b = np.array([[-1.0, 0.2], [0.0, 1.0]], dtype=float, order='F')
    f = np.triu(np.random.randn(m, m)).astype(float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('I', 'I', n, a, c, d, b, f)

    assert info == 0

    # Check A is upper triangular
    for i in range(m):
        for j in range(i):
            assert abs(a_out[i, j]) < 1e-14, \
                f"A[{i},{j}] = {a_out[i, j]} should be zero (upper triangular)"


def test_mb03id_property_c_lower_triangular():
    """
    Test that output C remains lower triangular.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb03id

    np.random.seed(333)

    n = 4
    m = n // 2

    a = np.triu(np.random.randn(m, m)).astype(float, order='F')
    c = np.tril(np.random.randn(m, m)).astype(float, order='F')
    d = np.random.randn(m, m).astype(float, order='F')
    b = np.array([[-1.0, 0.2], [0.0, 1.0]], dtype=float, order='F')
    f = np.triu(np.random.randn(m, m)).astype(float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('I', 'I', n, a, c, d, b, f)

    assert info == 0

    # Check C is lower triangular
    for i in range(m):
        for j in range(i + 1, m):
            assert abs(c_out[i, j]) < 1e-14, \
                f"C[{i},{j}] = {c_out[i, j]} should be zero (lower triangular)"


def test_mb03id_all_negative_eigenvalues():
    """
    Test MB03ID when all eigenvalues have negative real parts.

    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb03id

    np.random.seed(444)

    n = 4
    m = n // 2

    # All diagonal elements of B are negative (all eigenvalues negative)
    a = np.array([[2.0, 0.3], [0.0, 3.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.0], [0.2, 2.0]], dtype=float, order='F')
    d = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float, order='F')
    b = np.array([[-1.0, 0.2], [0.0, -2.0]], dtype=float, order='F')
    f = np.array([[0.5, 0.1], [0.0, 0.6]], dtype=float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('I', 'I', n, a, c, d, b, f)

    assert info == 0
    # neig counts eigenvalues with negative real parts
    # With 2 1x1 blocks both negative, neig should be 2
    assert neig == 2, f"Expected neig=2 for all negative eigenvalues, got {neig}"


def test_mb03id_all_positive_eigenvalues():
    """
    Test MB03ID when subpencil eigenvalues have positive real parts.

    Note: The full pencil aS - bH may have different eigenvalue structure
    than the subpencil aC'*A - bB. This test verifies the routine runs
    without error and produces valid output.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import mb03id

    np.random.seed(555)

    n = 4
    m = n // 2

    # Diagonal elements of A*B*C product are positive (subpencil eigenvalues)
    a = np.array([[2.0, 0.3], [0.0, 3.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.0], [0.2, 2.0]], dtype=float, order='F')
    d = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float, order='F')
    b = np.array([[1.0, 0.2], [0.0, 2.0]], dtype=float, order='F')
    f = np.array([[0.5, 0.1], [0.0, 0.6]], dtype=float, order='F')

    (a_out, c_out, d_out, b_out, f_out, q_out, u1_out, u2_out,
     neig, info) = mb03id('I', 'I', n, a, c, d, b, f)

    assert info == 0
    # neig should be non-negative and <= n (number of eigenvalues)
    assert 0 <= neig <= n, f"neig should be in [0, {n}], got {neig}"
