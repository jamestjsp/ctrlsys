"""
Tests for MA02HZ - Check if complex matrix equals scalar times identity-like matrix

MA02HZ checks if A = DIAG*I, where I is an M-by-N matrix with ones on
the diagonal and zeros elsewhere. A and DIAG are complex.

Returns True if A = DIAG*I, False otherwise. If min(M,N) = 0, returns False.

Property tests verify:
- Complex identity matrix scaled by complex DIAG
- Zero matrix (DIAG=0)
- Non-identity matrices return False
- Upper/lower triangular checks
- Edge cases (empty, rectangular)
"""
import numpy as np
import pytest
from ctrlsys import ma02hz


def test_ma02hz_identity_basic():
    """
    Test basic identity matrix check with DIAG=1+0j.

    A 3x3 identity matrix should return True when DIAG=1.0.
    """
    a = np.eye(3, dtype=complex, order='F')

    result = ma02hz('A', a, 1.0+0j)

    assert result is True


def test_ma02hz_complex_scaled_identity():
    """
    Test complex scaled identity matrix with DIAG=2.5+1.5j.

    (2.5+1.5j) * I_3 should return True.
    """
    diag = 2.5 + 1.5j
    a = diag * np.eye(3, dtype=complex, order='F')

    result = ma02hz('A', a, diag)

    assert result is True


def test_ma02hz_pure_imaginary_diag():
    """
    Test pure imaginary diagonal value.

    3j * I_4 should return True.
    """
    diag = 3j
    a = diag * np.eye(4, dtype=complex, order='F')

    result = ma02hz('A', a, diag)

    assert result is True


def test_ma02hz_zero_matrix():
    """
    Test zero matrix with DIAG=0.0.

    A zero 4x4 matrix equals 0*I, should return True.
    """
    a = np.zeros((4, 4), dtype=complex, order='F')

    result = ma02hz('A', a, 0.0+0j)

    assert result is True


def test_ma02hz_not_identity():
    """
    Test matrix that is NOT a scaled identity.

    A general matrix should return False.
    """
    a = np.array([[1.0+1j, 2.0-0.5j],
                  [3.0+0.2j, 4.0-1j]], dtype=complex, order='F')

    result = ma02hz('A', a, 1.0+0j)

    assert result is False


def test_ma02hz_wrong_diagonal():
    """
    Test identity matrix with wrong DIAG value.

    I_3 with DIAG=2.0+0j should return False (diagonal is 1, not 2).
    """
    a = np.eye(3, dtype=complex, order='F')

    result = ma02hz('A', a, 2.0+0j)

    assert result is False


def test_ma02hz_wrong_imaginary_part():
    """
    Test identity matrix with DIAG that has wrong imaginary part.

    I_3 with DIAG=1.0+0.1j should return False.
    """
    a = np.eye(3, dtype=complex, order='F')

    result = ma02hz('A', a, 1.0+0.1j)

    assert result is False


def test_ma02hz_nonzero_offdiagonal():
    """
    Test matrix with correct diagonal but nonzero off-diagonal.
    """
    a = np.array([[1.0+0j, 0.1j, 0.0],
                  [0.0, 1.0+0j, 0.0],
                  [0.0, 0.0, 1.0+0j]], dtype=complex, order='F')

    result = ma02hz('A', a, 1.0+0j)

    assert result is False


def test_ma02hz_rectangular_tall():
    """
    Test rectangular tall matrix (m > n).

    A 4x2 matrix with DIAG*I structure.
    I = [[1, 0],
         [0, 1],
         [0, 0],
         [0, 0]]
    """
    diag = 3.0 - 1j
    i_mat = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [0.0, 0.0],
                      [0.0, 0.0]], dtype=complex, order='F')
    a = diag * i_mat

    result = ma02hz('A', a, diag)

    assert result is True


def test_ma02hz_rectangular_wide():
    """
    Test rectangular wide matrix (m < n).

    A 2x4 matrix with DIAG*I structure.
    I = [[1, 0, 0, 0],
         [0, 1, 0, 0]]
    """
    diag = -1.5 + 2j
    i_mat = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]], dtype=complex, order='F')
    a = diag * i_mat

    result = ma02hz('A', a, diag)

    assert result is True


def test_ma02hz_upper_triangular_true():
    """
    Test upper triangular check with JOB='U'.

    Only checks upper triangle. Lower triangle can have any values.
    """
    diag = 1.0 + 0j
    a = np.array([[1.0+0j, 0.0, 0.0],
                  [99.0+5j, 1.0+0j, 0.0],
                  [99.0-3j, 99.0+1j, 1.0+0j]], dtype=complex, order='F')

    result = ma02hz('U', a, diag)

    assert result is True


def test_ma02hz_upper_triangular_false():
    """
    Test upper triangular check that fails.

    Upper triangle has non-zero off-diagonal.
    """
    a = np.array([[1.0+0j, 0.5j, 0.0],
                  [0.0, 1.0+0j, 0.0],
                  [0.0, 0.0, 1.0+0j]], dtype=complex, order='F')

    result = ma02hz('U', a, 1.0+0j)

    assert result is False


def test_ma02hz_lower_triangular_true():
    """
    Test lower triangular check with JOB='L'.

    Only checks lower triangle. Upper triangle can have any values.
    """
    diag = 2.0 - 0.5j
    a = np.array([[2.0-0.5j, 99.0+1j, 99.0-2j],
                  [0.0, 2.0-0.5j, 99.0+3j],
                  [0.0, 0.0, 2.0-0.5j]], dtype=complex, order='F')

    result = ma02hz('L', a, diag)

    assert result is True


def test_ma02hz_lower_triangular_false():
    """
    Test lower triangular check that fails.

    Lower triangle has non-zero off-diagonal.
    """
    a = np.array([[1.0+0j, 0.0, 0.0],
                  [0.5-0.2j, 1.0+0j, 0.0],
                  [0.0, 0.0, 1.0+0j]], dtype=complex, order='F')

    result = ma02hz('L', a, 1.0+0j)

    assert result is False


def test_ma02hz_empty_returns_false():
    """
    Test that min(m,n)=0 returns False.

    Empty matrix case.
    """
    a = np.zeros((0, 3), dtype=complex, order='F')

    result = ma02hz('A', a, 1.0+0j)

    assert result is False


def test_ma02hz_single_element_match():
    """
    Test single element matrix that matches DIAG.
    """
    diag = 5.0 + 2.5j
    a = np.array([[diag]], dtype=complex, order='F')

    result = ma02hz('A', a, diag)

    assert result is True


def test_ma02hz_single_element_no_match():
    """
    Test single element matrix that doesn't match DIAG.
    """
    a = np.array([[5.0 + 1j]], dtype=complex, order='F')

    result = ma02hz('A', a, 3.0 + 0j)

    assert result is False


def test_ma02hz_negative_real_diag():
    """
    Test negative real diagonal value.
    """
    diag = -2.0 + 0j
    a = diag * np.eye(4, dtype=complex, order='F')

    result = ma02hz('A', a, diag)

    assert result is True


def test_ma02hz_large_matrix():
    """
    Test larger identity-like matrix.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 50
    diag = 7.5 - 3.2j
    a = diag * np.eye(n, dtype=complex, order='F')

    result = ma02hz('A', a, diag)

    assert result is True


def test_ma02hz_large_matrix_with_perturbation():
    """
    Test that even tiny perturbation causes failure.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 20
    diag = 1.0 + 0j
    a = diag * np.eye(n, dtype=complex, order='F')
    a[5, 10] = 1e-100 + 0j

    result = ma02hz('A', a, diag)

    assert result is False


def test_ma02hz_upper_rectangular_tall():
    """
    Test upper triangular check on tall rectangular matrix.

    4x2 matrix, upper check only looks at upper trapezoid.
    """
    a = np.array([[1.0+0j, 0.0],
                  [99.0+1j, 1.0+0j],
                  [99.0-2j, 99.0+3j],
                  [99.0, 99.0-1j]], dtype=complex, order='F')

    result = ma02hz('U', a, 1.0+0j)

    assert result is True


def test_ma02hz_lower_rectangular_wide():
    """
    Test lower triangular check on wide rectangular matrix.

    2x4 matrix, lower check only looks at lower trapezoid.
    """
    a = np.array([[1.0+0j, 99.0+1j, 99.0-2j, 99.0+3j],
                  [0.0, 1.0+0j, 99.0, 99.0-1j]], dtype=complex, order='F')

    result = ma02hz('L', a, 1.0+0j)

    assert result is True


def test_ma02hz_involution_property():
    """
    Test mathematical property: if A=DIAG*I, then check returns True,
    and if modified, check returns False.

    This validates both positive and negative cases.
    """
    n = 5
    diag = 3.14159 - 2.71828j

    a = diag * np.eye(n, dtype=complex, order='F')
    assert ma02hz('A', a, diag) is True

    a[2, 3] = 0.001 + 0.001j
    assert ma02hz('A', a, diag) is False

    a[2, 3] = 0.0
    assert ma02hz('A', a, diag) is True


def test_ma02hz_conjugate_diagonal():
    """
    Test that conjugate of DIAG is not the same.

    If A = DIAG*I, then A != conj(DIAG)*I.
    """
    diag = 2.0 + 3.0j
    a = diag * np.eye(3, dtype=complex, order='F')

    assert ma02hz('A', a, diag) is True
    assert ma02hz('A', a, np.conj(diag)) is False


def test_ma02hz_unit_circle_diag():
    """
    Test diagonal on unit circle.

    DIAG = exp(i*pi/4) = (sqrt(2)/2)(1 + i)
    """
    diag = np.exp(1j * np.pi / 4)
    a = diag * np.eye(4, dtype=complex, order='F')

    result = ma02hz('A', a, diag)

    assert result is True


def test_ma02hz_hermitian_scaled_identity():
    """
    Test that scaled identity with real DIAG is Hermitian.

    Mathematical property: If A = d*I with d real, then A = A^H.
    """
    diag = 5.0 + 0j
    n = 4
    a = diag * np.eye(n, dtype=complex, order='F')

    result = ma02hz('A', a, diag)
    assert result is True

    np.testing.assert_allclose(a, np.conj(a.T), rtol=1e-14)
