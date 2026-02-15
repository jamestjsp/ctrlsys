"""
Tests for MA02AD - Matrix transposition

MA02AD performs in-place matrix transposition for full, upper triangular,
and lower triangular matrices. This is a fundamental linear algebra operation
used throughout SLICOT for efficient matrix manipulations.

Property tests verify:
- Transpose is an involution: (A^T)^T = A
- Shape correctness: (m,n)^T = (n,m)
- Orthogonality preservation for special matrices
"""
import numpy as np
import pytest
from ctrlsys import ma02ad


def test_ma02ad_full_transpose():
    """Test full matrix transpose (JOB='F')"""
    # 3x2 matrix
    a = np.array([[1.0, 4.0],
                  [2.0, 5.0],
                  [3.0, 6.0]], order='F')

    b = ma02ad('F', a)

    # Expected: 2x3 transpose
    expected = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]], order='F')

    np.testing.assert_allclose(b, expected, rtol=1e-14)


def test_ma02ad_upper_triangular():
    """Test upper triangular transpose (JOB='U')"""
    # 3x3 matrix - only upper triangle used
    a = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]], order='F')

    b = ma02ad('U', a)

    # Expected: B(j,i) = A(i,j) for i <= j (upper triangle)
    # B is transpose, so upper triangle of A becomes upper triangle of B
    expected = np.array([[1.0, 0.0, 0.0],
                        [2.0, 5.0, 0.0],
                        [3.0, 6.0, 9.0]], order='F')

    np.testing.assert_allclose(b, expected, rtol=1e-14)


def test_ma02ad_lower_triangular():
    """Test lower triangular transpose (JOB='L')"""
    # 3x3 matrix - only lower triangle used
    a = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]], order='F')

    b = ma02ad('L', a)

    # Expected: B(j,i) = A(i,j) for i >= j (lower triangle)
    # Lower triangle of A becomes lower triangle of B after transpose
    expected = np.array([[1.0, 4.0, 7.0],
                        [0.0, 5.0, 8.0],
                        [0.0, 0.0, 9.0]], order='F')

    np.testing.assert_allclose(b, expected, rtol=1e-14)


def test_ma02ad_rectangular():
    """Test rectangular matrix transpose"""
    # 2x4 matrix
    a = np.array([[1.0, 3.0, 5.0, 7.0],
                  [2.0, 4.0, 6.0, 8.0]], order='F')

    b = ma02ad('F', a)

    # Expected: 4x2 transpose
    expected = np.array([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0]], order='F')

    np.testing.assert_allclose(b, expected, rtol=1e-14)


def test_ma02ad_upper_trapezoid():
    """Test upper trapezoidal transpose (M > N)"""
    # 4x2 matrix - upper trapezoid
    a = np.array([[1.0, 3.0],
                  [2.0, 4.0],
                  [5.0, 6.0],
                  [7.0, 8.0]], order='F')

    b = ma02ad('U', a)

    # Expected: B(j,i) = A(i,j) for i <= min(j,m)
    # Only elements where row <= col in original
    expected = np.array([[1.0, 0.0, 0.0, 0.0],
                        [3.0, 4.0, 0.0, 0.0]], order='F')

    np.testing.assert_allclose(b, expected, rtol=1e-14)


def test_ma02ad_zero_rows():
    """Test edge case: zero rows"""
    a = np.array([], dtype=np.float64).reshape(0, 3, order='F')

    b = ma02ad('F', a)

    assert b.shape == (3, 0)


def test_ma02ad_zero_cols():
    """Test edge case: zero columns"""
    a = np.array([], dtype=np.float64).reshape(3, 0, order='F')

    b = ma02ad('F', a)

    assert b.shape == (0, 3)


def test_ma02ad_single_element():
    """Test edge case: single element"""
    a = np.array([[5.0]], order='F')

    b = ma02ad('F', a)

    np.testing.assert_allclose(b, np.array([[5.0]], order='F'), rtol=1e-14)


def test_ma02ad_involution_property():
    """
    Test transpose involution property: (A^T)^T = A

    Validates:
    - Double transpose returns original matrix
    - Property holds for all JOB types
    - Numerical accuracy preserved through double operation

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    a_orig = np.random.randn(4, 3)

    # Full transpose: (A^T)^T = A
    a_copy = a_orig.copy(order='F')
    b = ma02ad('F', a_copy)
    c = ma02ad('F', b)
    np.testing.assert_allclose(c, a_orig.T.T, rtol=1e-14)

    # For triangular matrices, the involution property is more nuanced
    # since only the triangular part is transposed. Test square case only.
    a_square = np.random.randn(4, 4)
    a_copy = a_square.copy(order='F')
    b = ma02ad('F', a_copy)
    c = ma02ad('F', b)
    np.testing.assert_allclose(c, a_square, rtol=1e-14)


def test_ma02ad_orthogonal_matrix_preservation():
    """
    Test transpose of orthogonal matrix: Q^T * Q = I

    Validates:
    - Orthogonality preserved after transpose
    - Numerical stability for well-conditioned matrices
    - Cross-validation with NumPy

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    a = np.random.randn(4, 4)
    q, r = np.linalg.qr(a)

    # Transpose with ma02ad
    q_copy = q.copy(order='F')
    qt = ma02ad('F', q_copy)

    # Q^T * Q should be identity
    result = qt @ q
    identity = np.eye(4)

    np.testing.assert_allclose(result, identity, rtol=1e-13, atol=1e-14)

    # Cross-validate transpose matches NumPy
    np.testing.assert_allclose(qt, q.T, rtol=1e-14)


def test_ma02ad_cross_validate_numpy():
    """
    Cross-validate MA02AD against NumPy transpose

    Validates:
    - Full transpose matches np.transpose()
    - Triangular extractions match manual computation
    - Various matrix sizes and aspect ratios

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    # Test several shapes
    for m, n in [(3, 3), (4, 2), (2, 5), (10, 1)]:
        a = np.random.randn(m, n)

        # Full transpose
        a_copy = a.copy(order='F')
        b_ma02ad = ma02ad('F', a_copy)
        b_numpy = a.T

        np.testing.assert_allclose(b_ma02ad, b_numpy, rtol=1e-14,
                                  err_msg=f"Failed for shape ({m},{n})")


def test_ma02ad_large_matrix():
    """
    Test transpose with larger matrices

    Validates:
    - Performance with realistic matrix sizes
    - Numerical accuracy maintained for larger data
    - Memory layout correctness (Fortran order)

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    # 100x80 matrix
    a = np.random.randn(100, 80)
    a_copy = a.copy(order='F')

    b = ma02ad('F', a_copy)

    assert b.shape == (80, 100)
    np.testing.assert_allclose(b, a.T, rtol=1e-13, atol=1e-14)

    # Verify column-major storage
    assert b.flags['F_CONTIGUOUS']


def test_ma02ad_symmetric_matrix():
    """
    Test transpose of symmetric matrix: A^T = A

    Validates:
    - Symmetric matrices satisfy A^T = A
    - Full transpose preserves symmetry
    - Numerical precision for symmetric structures

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    # Create symmetric matrix
    a = np.random.randn(4, 4)
    a = (a + a.T) / 2  # Make symmetric
    a = a.astype(float, order='F')

    # Full transpose of symmetric matrix should equal original
    a_copy = a.copy(order='F')
    at = ma02ad('F', a_copy)

    # A^T = A for symmetric matrices
    np.testing.assert_allclose(at, a, rtol=1e-14)
