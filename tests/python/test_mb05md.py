"""
Tests for MB05MD: Matrix exponential for a real non-defective matrix.

Computes exp(A*delta) where A is a real N-by-N non-defective matrix
with real or complex eigenvalues and delta is a scalar value.

Uses eigenvalue/eigenvector decomposition technique (Moler-Van Loan "Method 15").
Returns eigenvalues, eigenvectors of A, and intermediate matrix Y.
exp(A*delta) = V*Y where V is the eigenvector matrix.
"""
import numpy as np
import pytest
from ctrlsys import mb05md


"""Basic functionality tests using SLICOT HTML doc example."""

def test_html_doc_example():
    """
    Validate 4x4 example from SLICOT HTML documentation.

    Input: 4x4 matrix A, delta=1.0, BALANC='N'
    Data from MB05MD.html (read row-wise: ((A(I,J), J=1,N), I=1,N))
    Expected output with 4-digit precision.
    
    Note: Eigenvalue ordering is implementation-dependent in LAPACK.
    We verify:
    1. exp(A*delta) matches documented result
    2. Eigenvalues match (unordered)
    3. exp(A*delta) = V*Y relationship holds
    4. V contains valid eigenvectors of A
    5. Eigenvectors match documented values (up to ordering/sign)
    """
    n = 4
    delta = 1.0

    a = np.array([
        [0.5, 0.0, 2.3, -2.6],
        [0.0, 0.5, -1.4, -0.7],
        [2.3, -1.4, 0.5, 0.0],
        [-2.6, -0.7, 0.0, 0.5],
    ], order='F', dtype=float)

    a_copy = a.copy()

    exp_a_delta_expected = np.array([
        [26.8551, -3.2824, 18.7409, -19.4430],
        [-3.2824, 4.3474, -5.1848, 0.2700],
        [18.7409, -5.1848, 15.6012, -11.7228],
        [-19.4430, 0.2700, -11.7228, 15.6012],
    ], order='F', dtype=float)

    valr_expected = np.array([-3.0, 4.0, -1.0, 2.0])
    vali_expected = np.array([0.0, 0.0, 0.0, 0.0])

    v_expected = np.array([
        [-0.7000, 0.7000, 0.1000, -0.1000],
        [0.1000, -0.1000, 0.7000, -0.7000],
        [0.5000, 0.5000, 0.5000, 0.5000],
        [-0.5000, -0.5000, 0.5000, 0.5000],
    ], order='F', dtype=float)

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == 0
    
    # Test 1: exp(A*delta) matches expected output from documentation
    np.testing.assert_allclose(exp_a_delta, exp_a_delta_expected, rtol=1e-3, atol=1e-4)

    # Test 2: Eigenvalues match (unordered, since ordering is implementation-dependent)
    eig_computed = valr + 1j * vali
    eig_expected = valr_expected + 1j * vali_expected
    np.testing.assert_allclose(
        sorted(eig_computed, key=lambda x: (x.real, x.imag)),
        sorted(eig_expected, key=lambda x: (x.real, x.imag)),
        rtol=1e-3, atol=1e-4
    )

    # Test 3: exp(A*delta) = V*Y (fundamental relationship)
    np.testing.assert_allclose(exp_a_delta, v @ y, rtol=1e-12, atol=1e-14)

    # Test 4: V contains eigenvectors of A (A*v[:, i] = eigenvalue[i] * v[:, i])
    for i in range(n):
        if vali[i] == 0:  # Real eigenvalue
            residual = a_copy @ v[:, i] - valr[i] * v[:, i]
            assert np.linalg.norm(residual) < 1e-12, \
                f"Eigenvector {i} failed: ||A*v - λ*v|| = {np.linalg.norm(residual)}"

    # Test 5: Eigenvectors match documentation (accounting for ordering and sign ambiguity)
    # Create mapping from computed to expected eigenvalues
    mapping = {}
    for i, ev_comp in enumerate(valr):
        for j, ev_exp in enumerate(valr_expected):
            if abs(ev_comp - ev_exp) < 1e-10 and j not in mapping.values():
                mapping[i] = j
                break
    
    # Verify eigenvectors match after reordering (up to sign)
    for i in range(n):
        j = mapping[i]
        v_comp = v[:, i]
        v_exp = v_expected[:, j]
        # Eigenvectors can differ by sign, so check both
        assert (np.allclose(v_comp, v_exp, rtol=1e-3, atol=1e-4) or
                np.allclose(v_comp, -v_exp, rtol=1e-3, atol=1e-4)), \
               f"Eigenvector {i} doesn't match expected column {j} (λ={valr[i]})"


"""Edge case tests for MB05MD."""

def test_n_equals_zero():
    """Test with n=0 (empty system)."""
    n = 0
    delta = 1.0
    a = np.array([], order='F', dtype=float).reshape(0, 0)

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == 0
    assert exp_a_delta.shape == (0, 0)
    assert v.shape == (0, 0)
    assert y.shape == (0, 0)
    assert valr.shape == (0,)
    assert vali.shape == (0,)

def test_delta_equals_zero():
    """
    Test with delta=0.

    exp(A*0) = I (identity matrix)

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3
    delta = 0.0

    a = 0.1 * np.random.randn(n, n).astype(float, order='F')

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a_delta, np.eye(n), rtol=1e-12, atol=1e-14)

def test_n_equals_one():
    """
    Test scalar case (n=1).

    For A = [a], exp(A*delta) = exp(a*delta)
    """
    n = 1
    delta = 0.5
    a_val = 2.0
    a = np.array([[a_val]], order='F', dtype=float)

    exp_expected = np.exp(a_val * delta)

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a_delta[0, 0], exp_expected, rtol=1e-14)
    np.testing.assert_allclose(valr[0], a_val, rtol=1e-14)
    np.testing.assert_allclose(vali[0], 0.0, atol=1e-14)

def test_identity_matrix():
    """
    Test identity matrix.

    exp(I*delta) = exp(delta)*I
    """
    n = 3
    delta = 2.0
    a = np.eye(n, order='F', dtype=float)

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == 0
    exp_expected = np.exp(delta) * np.eye(n)
    np.testing.assert_allclose(exp_a_delta, exp_expected, rtol=1e-13)
    np.testing.assert_allclose(valr, np.ones(n), rtol=1e-14)
    np.testing.assert_allclose(vali, np.zeros(n), atol=1e-14)


"""Mathematical property tests for numerical correctness."""

def test_exp_a_delta_equals_v_times_y():
    """
    Validate exp(A*delta) = V*Y.

    This is the fundamental relationship computed by MB05MD.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    delta = 0.5

    a = np.random.randn(n, n).astype(float, order='F')

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == 0
    exp_reconstructed = v @ y
    np.testing.assert_allclose(exp_a_delta, exp_reconstructed, rtol=1e-13, atol=1e-14)

def test_eigenvalue_preservation():
    """
    Validate eigenvalues match numpy.linalg.eigvals.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 4

    a = np.random.randn(n, n).astype(float, order='F')
    a_copy = a.copy()

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, 1.0, a)

    assert info == 0

    eig_numpy = np.linalg.eigvals(a_copy)
    eig_computed = valr + 1j * vali

    np.testing.assert_allclose(
        sorted(eig_numpy, key=lambda x: (x.real, x.imag)),
        sorted(eig_computed, key=lambda x: (x.real, x.imag)),
        rtol=1e-12
    )

def test_exp_property_exp_a_plus_b():
    """
    Validate exp(A*2*delta) = exp(A*delta) * exp(A*delta) for commuting matrices.

    For a single matrix A: exp(2A) = (exp(A))^2

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3
    delta = 0.25

    a = np.random.randn(n, n).astype(float, order='F')

    exp_a_delta, v1, y1, valr1, vali1, info1 = mb05md('N', n, delta, a.copy())
    assert info1 == 0

    exp_a_2delta, v2, y2, valr2, vali2, info2 = mb05md('N', n, 2 * delta, a.copy())
    assert info2 == 0

    exp_squared = exp_a_delta @ exp_a_delta
    np.testing.assert_allclose(exp_a_2delta, exp_squared, rtol=1e-12)

def test_diagonal_matrix():
    """
    Test with diagonal matrix (exact closed form).

    For diagonal A = diag(a1, a2, ...):
    exp(A*delta) = diag(exp(a1*delta), exp(a2*delta), ...)

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n = 4
    delta = 0.3

    diag_vals = np.random.randn(n)
    a = np.diag(diag_vals).astype(float, order='F')

    exp_expected = np.diag(np.exp(diag_vals * delta))

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a_delta, exp_expected, rtol=1e-13)
    np.testing.assert_allclose(sorted(valr), sorted(diag_vals), rtol=1e-14)
    np.testing.assert_allclose(vali, np.zeros(n), atol=1e-14)

def test_complex_eigenvalues():
    """
    Test matrix with complex conjugate eigenvalues.

    For rotation matrix [[cos(t), -sin(t)], [sin(t), cos(t)]]:
    eigenvalues are exp(+-i*t), so A = i*[[0, -1], [1, 0]] (scaled)

    Random seed: 999 (for reproducibility)
    """
    n = 2
    delta = 1.0
    omega = 1.0

    a = omega * np.array([
        [0.0, -1.0],
        [1.0, 0.0],
    ], order='F', dtype=float)

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == 0
    np.testing.assert_allclose(exp_a_delta[0, 0], np.cos(omega * delta), rtol=1e-14)
    np.testing.assert_allclose(exp_a_delta[1, 1], np.cos(omega * delta), rtol=1e-14)
    np.testing.assert_allclose(np.abs(exp_a_delta[0, 1]), np.sin(omega * delta), rtol=1e-14)
    np.testing.assert_allclose(np.abs(exp_a_delta[1, 0]), np.sin(omega * delta), rtol=1e-14)
    assert exp_a_delta[0, 1] == -exp_a_delta[1, 0]
    np.testing.assert_allclose(valr, [0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(np.abs(vali), [omega, omega], rtol=1e-14)

def test_nilpotent_matrix_is_defective():
    """
    Test that nilpotent matrix is correctly detected as defective.

    Nilpotent matrices have repeated zero eigenvalues with insufficient
    eigenvectors, making them defective. MB05MD should return info=n+2.
    """
    n = 3
    delta = 1.0

    a = np.array([
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], order='F', dtype=float)

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info in [n + 1, n + 2]


"""Tests for diagonal scaling (BALANC='S')."""

def test_with_scaling():
    """
    Test with BALANC='S' for badly scaled matrix.

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 4
    delta = 0.1

    d = np.diag([1e-3, 1.0, 1e2, 1e4])
    a_balanced = 0.1 * np.random.randn(n, n)
    a = d @ a_balanced @ np.linalg.inv(d)
    a = np.asfortranarray(a)

    exp_a_delta_s, v_s, y_s, valr_s, vali_s, info_s = mb05md('S', n, delta, a.copy())
    exp_a_delta_n, v_n, y_n, valr_n, vali_n, info_n = mb05md('N', n, delta, a.copy())

    assert info_s == 0
    assert info_n == 0

    eig_s = valr_s + 1j * vali_s
    eig_n = valr_n + 1j * vali_n
    np.testing.assert_allclose(
        sorted(eig_s, key=lambda x: (x.real, x.imag)),
        sorted(eig_n, key=lambda x: (x.real, x.imag)),
        rtol=1e-5
    )

def test_no_scaling():
    """
    Test with BALANC='N' (no scaling).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 3
    delta = 0.2

    a = np.random.randn(n, n).astype(float, order='F')
    a_copy = a.copy()

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == 0

    eig_numpy = np.linalg.eigvals(a_copy)
    eig_computed = valr + 1j * vali

    np.testing.assert_allclose(
        sorted(eig_numpy, key=lambda x: (x.real, x.imag)),
        sorted(eig_computed, key=lambda x: (x.real, x.imag)),
        rtol=1e-14
    )


"""Error handling tests."""

def test_invalid_balanc():
    """Test that invalid BALANC returns info=-1."""
    n = 2
    delta = 1.0
    a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)

    exp_a_delta, v, y, valr, vali, info = mb05md('X', n, delta, a)

    assert info == -1

def test_negative_n():
    """Test that negative n returns info=-2."""
    delta = 1.0
    a = np.array([[1.0]], order='F', dtype=float)

    exp_a_delta, v, y, valr, vali, info = mb05md('N', -1, delta, a)

    assert info == -2

def test_defective_matrix():
    """
    Test defective matrix returns info=N+2.

    A defective matrix has repeated eigenvalues but insufficient eigenvectors.
    """
    n = 2
    delta = 1.0

    a = np.array([
        [1.0, 1.0],
        [0.0, 1.0],
    ], order='F', dtype=float)

    exp_a_delta, v, y, valr, vali, info = mb05md('N', n, delta, a)

    assert info == n + 2
