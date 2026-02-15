"""Tests for tb01nd - Observer Hessenberg form reduction."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tb01nd_upper_hessenberg_html_example():
    """
    Test upper observer Hessenberg form using HTML doc example.

    N=5, P=3, JOBU='N', UPLO='U'
    Input A: 5x5, C: 3x5
    Validates mathematical properties since Householder signs can vary.
    """
    from ctrlsys import tb01nd

    # Input A: 5x5 from HTML doc - read column-wise: ((A(I,J), I=1,N), J=1,N)
    a = np.array([
        [15.0, 20.0,  4.0,  5.0,  5.0],
        [21.0,  1.0,  1.0,  6.0, 11.0],
        [-3.0,  2.0,  7.0, 12.0, 17.0],
        [ 3.0,  8.0, 13.0, 13.0, -7.0],
        [ 9.0,  9.0, 14.0, -6.0, -1.0]
    ], order='F', dtype=float)

    # Input C: 3x5 from HTML doc - read row-wise: ((C(I,J), J=1,N), I=1,P)
    c = np.array([
        [ 7.0, -1.0,  3.0, -6.0, -3.0],
        [ 4.0,  5.0,  6.0, -2.0, -3.0],
        [ 9.0,  8.0,  5.0,  2.0,  1.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()
    n, p = 5, 3

    # Expected output from HTML doc (printed row-by-row)
    # Transformed A (upper observer Hessenberg form)
    a_expected = np.array([
        [  7.1637, -0.9691, -16.5046,  0.2869,  0.9205],
        [ -2.3285, 11.5431,  -8.7471,  3.4122, -3.7118],
        [-10.5440, -7.6032,  -0.3215,  3.6571, -0.4335],
        [ -3.6845,  5.6449,   0.5906,-15.6996, 17.4267],
        [  0.0000, -6.4260,   1.5591, 14.4317, 32.3143]
    ], order='F', dtype=float)

    # Transformed C
    c_expected = np.array([
        [ 0.0000,  0.0000,  7.6585,  5.2973, -4.1576],
        [ 0.0000,  0.0000,  0.0000,  5.8305, -7.4837],
        [ 0.0000,  0.0000,  0.0000,  0.0000,-13.2288]
    ], order='F', dtype=float)

    # Call tb01nd with JOBU='N' (don't compute U)
    a_out, c_out, u_out, info = tb01nd('N', 'U', a, c)

    assert info == 0

    # Verify against expected values from HTML doc (4 decimal places)
    assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)


def test_tb01nd_upper_with_u_matrix():
    """
    Test upper observer Hessenberg form with U matrix (JOBU='I').

    Validates mathematical properties: U'AU similarity and CU transformation.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tb01nd

    np.random.seed(42)
    n, p = 5, 3

    a = np.random.randn(n, n).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    a_orig = a.copy()
    c_orig = c.copy()

    # Call with JOBU='I' to get transformation matrix
    a_out, c_out, u_out, info = tb01nd('I', 'U', a, c)

    assert info == 0

    # Verify U is orthogonal: U^T @ U = I
    assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-14, atol=1e-14)

    # Verify similarity transformation: A_out = U' @ A_orig @ U
    assert_allclose(a_out, u_out.T @ a_orig @ u_out, rtol=1e-14, atol=1e-14)

    # Verify: C_out = C_orig @ U
    assert_allclose(c_out, c_orig @ u_out, rtol=1e-14, atol=1e-14)

    # Verify upper triangular structure of C (zeros in lower-left)
    # For UPLO='U': C has upper triangular form in last P columns
    # C(i,j)=0 for j < N-P+i for i=0..P-1
    # i=0: j<N-P=2 => C[0,0:2]=0
    # i=1: j<N-P+1=3 => C[1,0:3]=0
    # i=2: j<N-P+2=4 => C[2,0:4]=0
    for i in range(p):
        for j in range(n - p + i):
            assert_allclose(c_out[i, j], 0.0, atol=1e-14)

    # Verify eigenvalues preserved
    eig_before = np.linalg.eigvals(a_orig)
    eig_after = np.linalg.eigvals(a_out)
    assert_allclose(sorted(eig_before.real), sorted(eig_after.real), rtol=1e-13, atol=1e-14)


def test_tb01nd_lower_hessenberg():
    """
    Test lower observer Hessenberg form (UPLO='L').

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import tb01nd

    np.random.seed(123)
    n, p = 5, 3

    a = np.random.randn(n, n).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    a_orig = a.copy()
    c_orig = c.copy()

    # Call with JOBU='I', UPLO='L'
    a_out, c_out, u_out, info = tb01nd('I', 'L', a, c)

    assert info == 0

    # Verify U is orthogonal: U^T @ U = I
    assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-14, atol=1e-14)

    # Verify similarity transformation: A_out = U' @ A_orig @ U
    assert_allclose(a_out, u_out.T @ a_orig @ u_out, rtol=1e-14, atol=1e-14)

    # Verify: C_out = C_orig @ U
    assert_allclose(c_out, c_orig @ u_out, rtol=1e-14, atol=1e-14)

    # For lower observer Hessenberg with UPLO='L':
    # C has lower triangular structure in first P columns
    # C(i,j) = 0 for j > i
    for i in range(p):
        for j in range(i + 1, p):
            assert_allclose(c_out[i, j], 0.0, atol=1e-14)


def test_tb01nd_accumulate_u():
    """
    Test U matrix accumulation mode (JOBU='U').

    When JOBU='U', the given U is updated with the transformations.
    U_out = U_init @ Q where Q is the internal Hessenberg reduction transformation.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tb01nd

    np.random.seed(456)
    n, p = 5, 2

    a = np.random.randn(n, n).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    a_orig = a.copy()
    c_orig = c.copy()

    # Create initial orthogonal U from QR decomposition
    q, _ = np.linalg.qr(np.random.randn(n, n))
    u_init = np.asfortranarray(q.astype(float))
    u_init_copy = u_init.copy()

    # Call with JOBU='U' to update the given U
    a_out, c_out, u_out, info = tb01nd('U', 'U', a, c, u=u_init)

    assert info == 0

    # Verify U_out is still orthogonal
    assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-14, atol=1e-14)

    # Extract internal transformation Q = U_init^T @ U_out
    q_transform = u_init_copy.T @ u_out

    # Verify Q is orthogonal
    assert_allclose(q_transform.T @ q_transform, np.eye(n), rtol=1e-14, atol=1e-14)

    # Verify: A_out = Q^T @ A_orig @ Q
    assert_allclose(a_out, q_transform.T @ a_orig @ q_transform, rtol=1e-14, atol=1e-14)

    # Verify: C_out = C_orig @ Q
    assert_allclose(c_out, c_orig @ q_transform, rtol=1e-14, atol=1e-14)


def test_tb01nd_eigenvalue_preservation():
    """
    Validate eigenvalue preservation under similarity transformation.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tb01nd

    np.random.seed(789)
    n, p = 6, 3

    a = np.random.randn(n, n).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    # Compute eigenvalues before transformation
    eig_before = np.linalg.eigvals(a)

    a_out, c_out, u_out, info = tb01nd('N', 'U', a, c)

    assert info == 0

    # Compute eigenvalues after transformation
    eig_after = np.linalg.eigvals(a_out)

    # Eigenvalues should be preserved (sort for comparison)
    eig_before_sorted = np.sort(eig_before)
    eig_after_sorted = np.sort(eig_after)

    assert_allclose(eig_before_sorted.real, eig_after_sorted.real, rtol=1e-13, atol=1e-14)
    assert_allclose(eig_before_sorted.imag, eig_after_sorted.imag, rtol=1e-13, atol=1e-14)


def test_tb01nd_p_eq_n():
    """
    Test case where P == N (C becomes square upper triangular, A stays full).

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import tb01nd

    np.random.seed(888)
    n, p = 4, 4  # p == n

    a = np.random.randn(n, n).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')
    a_orig = a.copy()
    c_orig = c.copy()

    a_out, c_out, u_out, info = tb01nd('I', 'U', a, c)

    assert info == 0

    # Verify U is orthogonal
    assert_allclose(u_out.T @ u_out, np.eye(n), rtol=1e-14, atol=1e-14)

    # Verify similarity transformation
    assert_allclose(a_out, u_out.T @ a_orig @ u_out, rtol=1e-14, atol=1e-14)
    assert_allclose(c_out, c_orig @ u_out, rtol=1e-14, atol=1e-14)


def test_tb01nd_n_zero():
    """Test edge case with N=0 and P=0."""
    from ctrlsys import tb01nd

    a = np.empty((0, 0), order='F', dtype=float)
    c = np.empty((0, 0), order='F', dtype=float)  # P must be <= N, so P=0 when N=0

    a_out, c_out, u_out, info = tb01nd('I', 'U', a, c)

    assert info == 0
    assert a_out.shape == (0, 0)
    assert c_out.shape == (0, 0)


def test_tb01nd_p_zero():
    """Test edge case with P=0 - A should be unchanged."""
    from ctrlsys import tb01nd

    np.random.seed(111)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    c = np.empty((0, n), order='F', dtype=float)
    a_orig = a.copy()

    a_out, c_out, u_out, info = tb01nd('I', 'U', a, c)

    assert info == 0
    # When P=0, A is unchanged
    assert_allclose(a_out, a_orig, rtol=1e-14, atol=1e-14)
    # U should be identity when no transformation needed
    assert_allclose(u_out, np.eye(n), rtol=1e-14, atol=1e-14)


def test_tb01nd_invalid_jobu():
    """Test error handling for invalid JOBU parameter."""
    from ctrlsys import tb01nd

    a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    c = np.array([[1.0, 2.0]], order='F', dtype=float)

    with pytest.raises(ValueError):
        tb01nd('X', 'U', a, c)


def test_tb01nd_invalid_uplo():
    """Test error handling for invalid UPLO parameter."""
    from ctrlsys import tb01nd

    a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    c = np.array([[1.0, 2.0]], order='F', dtype=float)

    with pytest.raises(ValueError):
        tb01nd('N', 'X', a, c)


def test_tb01nd_p_greater_than_n():
    """Test error handling when P > N (invalid)."""
    from ctrlsys import tb01nd

    # P cannot exceed N per Fortran docs: 0 <= P <= N
    # However, looking at HTML doc this seems valid (trapezoidal C)
    # Let's check by just verifying it returns proper info
    # Actually, the Fortran has: IF( P.LT.0 .OR. P.GT.N )
    # So P > N should give INFO = -4

    n, p = 3, 5
    a = np.random.randn(n, n).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_out, c_out, u_out, info = tb01nd('N', 'U', a, c)

    # Fortran docs: 0 <= P <= N, so P > N should give error
    assert info == -4
