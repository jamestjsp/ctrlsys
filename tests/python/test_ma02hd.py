"""
Tests for MA02HD - Check if matrix equals scalar times identity-like matrix

MA02HD checks if A = DIAG*I, where I is an M-by-N matrix with ones on
the diagonal and zeros elsewhere.

Returns True if A = DIAG*I, False otherwise. If min(M,N) = 0, returns False.

Property tests verify:
- Identity matrix scaled by DIAG
- Zero matrix (DIAG=0)
- Non-identity matrices return False
- Upper/lower triangular checks
- Edge cases (empty, rectangular)
"""
import numpy as np
import pytest
from ctrlsys import ma02hd


def test_ma02hd_identity_basic():
    """
    Test basic identity matrix check with DIAG=1.0.

    A 3x3 identity matrix should return True when DIAG=1.0.
    """
    a = np.eye(3, order='F')

    result = ma02hd('A', a, 1.0)

    assert result is True


def test_ma02hd_scaled_identity():
    """
    Test scaled identity matrix with DIAG=2.5.

    2.5 * I_3 should return True.
    """
    diag = 2.5
    a = diag * np.eye(3, order='F')

    result = ma02hd('A', a, diag)

    assert result is True


def test_ma02hd_zero_matrix():
    """
    Test zero matrix with DIAG=0.0.

    A zero 4x4 matrix equals 0*I, should return True.
    """
    a = np.zeros((4, 4), order='F')

    result = ma02hd('A', a, 0.0)

    assert result is True


def test_ma02hd_not_identity():
    """
    Test matrix that is NOT a scaled identity.

    A general matrix should return False.
    """
    a = np.array([[1.0, 2.0],
                  [3.0, 4.0]], order='F')

    result = ma02hd('A', a, 1.0)

    assert result is False


def test_ma02hd_wrong_diagonal():
    """
    Test identity matrix with wrong DIAG value.

    I_3 with DIAG=2.0 should return False (diagonal is 1, not 2).
    """
    a = np.eye(3, order='F')

    result = ma02hd('A', a, 2.0)

    assert result is False


def test_ma02hd_nonzero_offdiagonal():
    """
    Test matrix with correct diagonal but nonzero off-diagonal.
    """
    a = np.array([[1.0, 0.1, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], order='F')

    result = ma02hd('A', a, 1.0)

    assert result is False


def test_ma02hd_rectangular_tall():
    """
    Test rectangular tall matrix (m > n).

    A 4x2 matrix with DIAG*I structure.
    I = [[1, 0],
         [0, 1],
         [0, 0],
         [0, 0]]
    """
    diag = 3.0
    a = np.array([[3.0, 0.0],
                  [0.0, 3.0],
                  [0.0, 0.0],
                  [0.0, 0.0]], order='F')

    result = ma02hd('A', a, diag)

    assert result is True


def test_ma02hd_rectangular_wide():
    """
    Test rectangular wide matrix (m < n).

    A 2x4 matrix with DIAG*I structure.
    I = [[1, 0, 0, 0],
         [0, 1, 0, 0]]
    """
    diag = -1.5
    a = np.array([[-1.5, 0.0, 0.0, 0.0],
                  [0.0, -1.5, 0.0, 0.0]], order='F')

    result = ma02hd('A', a, diag)

    assert result is True


def test_ma02hd_upper_triangular_true():
    """
    Test upper triangular check with JOB='U'.

    Only checks upper triangle. Lower triangle can have any values.
    """
    diag = 1.0
    a = np.array([[1.0, 0.0, 0.0],
                  [99.0, 1.0, 0.0],
                  [99.0, 99.0, 1.0]], order='F')

    result = ma02hd('U', a, diag)

    assert result is True


def test_ma02hd_upper_triangular_false():
    """
    Test upper triangular check that fails.

    Upper triangle has non-zero off-diagonal.
    """
    a = np.array([[1.0, 0.5, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], order='F')

    result = ma02hd('U', a, 1.0)

    assert result is False


def test_ma02hd_lower_triangular_true():
    """
    Test lower triangular check with JOB='L'.

    Only checks lower triangle. Upper triangle can have any values.
    """
    diag = 2.0
    a = np.array([[2.0, 99.0, 99.0],
                  [0.0, 2.0, 99.0],
                  [0.0, 0.0, 2.0]], order='F')

    result = ma02hd('L', a, diag)

    assert result is True


def test_ma02hd_lower_triangular_false():
    """
    Test lower triangular check that fails.

    Lower triangle has non-zero off-diagonal.
    """
    a = np.array([[1.0, 0.0, 0.0],
                  [0.5, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], order='F')

    result = ma02hd('L', a, 1.0)

    assert result is False


def test_ma02hd_empty_returns_false():
    """
    Test that min(m,n)=0 returns False.

    Empty matrix case.
    """
    a = np.zeros((0, 3), order='F')

    result = ma02hd('A', a, 1.0)

    assert result is False


def test_ma02hd_single_element_match():
    """
    Test single element matrix that matches DIAG.
    """
    a = np.array([[5.0]], order='F')

    result = ma02hd('A', a, 5.0)

    assert result is True


def test_ma02hd_single_element_no_match():
    """
    Test single element matrix that doesn't match DIAG.
    """
    a = np.array([[5.0]], order='F')

    result = ma02hd('A', a, 3.0)

    assert result is False


def test_ma02hd_negative_diag():
    """
    Test negative diagonal value.
    """
    diag = -2.0
    a = diag * np.eye(4, order='F')

    result = ma02hd('A', a, diag)

    assert result is True


def test_ma02hd_large_matrix():
    """
    Test larger identity-like matrix.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 50
    diag = 7.5
    a = diag * np.eye(n, order='F')

    result = ma02hd('A', a, diag)

    assert result is True


def test_ma02hd_large_matrix_with_perturbation():
    """
    Test that even tiny perturbation causes failure.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 20
    diag = 1.0
    a = diag * np.eye(n, order='F')
    a[5, 10] = 1e-100

    result = ma02hd('A', a, diag)

    assert result is False


def test_ma02hd_upper_rectangular_tall():
    """
    Test upper triangular check on tall rectangular matrix.

    4x2 matrix, upper check only looks at upper trapezoid.
    """
    a = np.array([[1.0, 0.0],
                  [99.0, 1.0],
                  [99.0, 99.0],
                  [99.0, 99.0]], order='F')

    result = ma02hd('U', a, 1.0)

    assert result is True


def test_ma02hd_lower_rectangular_wide():
    """
    Test lower triangular check on wide rectangular matrix.

    2x4 matrix, lower check only looks at lower trapezoid.
    """
    a = np.array([[1.0, 99.0, 99.0, 99.0],
                  [0.0, 1.0, 99.0, 99.0]], order='F')

    result = ma02hd('L', a, 1.0)

    assert result is True


def test_ma02hd_involution_property():
    """
    Test mathematical property: if A=DIAG*I, then check returns True,
    and if modified, check returns False.

    This validates both positive and negative cases.
    """
    n = 5
    diag = 3.14159

    a = diag * np.eye(n, order='F')
    assert ma02hd('A', a, diag) is True

    a[2, 3] = 0.001
    assert ma02hd('A', a, diag) is False

    a[2, 3] = 0.0
    assert ma02hd('A', a, diag) is True
