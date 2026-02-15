"""
Tests for AB04MD - Bilinear transformation of state-space systems.

Converts between discrete-time and continuous-time representations.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_ab04md_continuous_to_discrete():
    """
    Test continuous->discrete transformation using SLICOT HTML doc example.

    Input (continuous-time): N=2, M=2, P=2, alpha=1, beta=1
    A = [[1.0, 0.5], [0.5, 1.0]]
    B = [[0.0, -1.0], [1.0, 0.0]]  (column-major: read column by column)
    C = [[-1.0, 0.0], [0.0, 1.0]]
    D = [[1.0, 0.0], [0.0, -1.0]]

    Expected output (discrete-time):
    A = [[-1.0, -4.0], [-4.0, -1.0]]
    B = [[2.8284, 0.0], [0.0, -2.8284]]
    C = [[0.0, 2.8284], [-2.8284, 0.0]]
    D = [[-1.0, 0.0], [0.0, -3.0]]
    """
    from ctrlsys import ab04md

    # Input matrices (column-major order)
    # Note: Fortran reads column-by-column from data file:
    # B data: 0.0, -1.0, 1.0, 0.0 -> B(:,1)=[0,-1], B(:,2)=[1,0]
    a = np.array([[1.0, 0.5],
                  [0.5, 1.0]], order='F', dtype=float)
    b = np.array([[0.0, 1.0],
                  [-1.0, 0.0]], order='F', dtype=float)
    c = np.array([[-1.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)
    d = np.array([[1.0, 0.0],
                  [0.0, -1.0]], order='F', dtype=float)

    # Expected outputs from HTML doc
    a_expected = np.array([[-1.0, -4.0],
                           [-4.0, -1.0]], order='F', dtype=float)
    b_expected = np.array([[2.8284, 0.0],
                           [0.0, -2.8284]], order='F', dtype=float)
    c_expected = np.array([[0.0, 2.8284],
                           [-2.8284, 0.0]], order='F', dtype=float)
    d_expected = np.array([[-1.0, 0.0],
                           [0.0, -3.0]], order='F', dtype=float)

    # TYPE='C' means continuous->discrete
    a_out, b_out, c_out, d_out, info = ab04md('C', a, b, c, d, alpha=1.0, beta=1.0)

    assert info == 0
    assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(d_out, d_expected, rtol=1e-3, atol=1e-4)


def test_ab04md_discrete_to_continuous():
    """
    Test discrete->continuous transformation (inverse of the HTML doc example).

    If we apply discrete->continuous to the outputs from the previous test,
    we should get back the original inputs.

    Random seed: 42 (not used, deterministic test)
    """
    from ctrlsys import ab04md

    # Start with discrete-time system (output from previous test)
    # Use exact sqrt(2) values for precision
    sqrt2 = np.sqrt(2.0)
    a = np.array([[-1.0, -4.0],
                  [-4.0, -1.0]], order='F', dtype=float)
    b = np.array([[2*sqrt2, 0.0],
                  [0.0, -2*sqrt2]], order='F', dtype=float)
    c = np.array([[0.0, 2*sqrt2],
                  [-2*sqrt2, 0.0]], order='F', dtype=float)
    d = np.array([[-1.0, 0.0],
                  [0.0, -3.0]], order='F', dtype=float)

    # Expected: original continuous-time system (with corrected B ordering)
    a_expected = np.array([[1.0, 0.5],
                           [0.5, 1.0]], order='F', dtype=float)
    b_expected = np.array([[0.0, 1.0],
                           [-1.0, 0.0]], order='F', dtype=float)
    c_expected = np.array([[-1.0, 0.0],
                           [0.0, 1.0]], order='F', dtype=float)
    d_expected = np.array([[1.0, 0.0],
                           [0.0, -1.0]], order='F', dtype=float)

    # TYPE='D' means discrete->continuous
    a_out, b_out, c_out, d_out, info = ab04md('D', a, b, c, d, alpha=1.0, beta=1.0)

    assert info == 0
    assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(d_out, d_expected, rtol=1e-3, atol=1e-4)


def test_ab04md_roundtrip():
    """
    Test that continuous->discrete->continuous is identity.

    Mathematical property: bilinear transformation is invertible.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab04md

    np.random.seed(42)
    n, m, p = 3, 2, 2

    # Generate random stable continuous-time system
    # Make A have negative eigenvalues for stability
    a_orig = np.random.randn(n, n).astype(float, order='F')
    a_orig = a_orig - 3.0 * np.eye(n, order='F')  # Shift eigenvalues left
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(p, n).astype(float, order='F')
    d_orig = np.random.randn(p, m).astype(float, order='F')

    # Store copies for comparison
    a0, b0, c0, d0 = a_orig.copy(), b_orig.copy(), c_orig.copy(), d_orig.copy()

    # Continuous -> Discrete
    a1, b1, c1, d1, info1 = ab04md('C', a0, b0, c0, d0, alpha=1.0, beta=1.0)
    assert info1 == 0

    # Discrete -> Continuous
    a2, b2, c2, d2, info2 = ab04md('D', a1, b1, c1, d1, alpha=1.0, beta=1.0)
    assert info2 == 0

    # Should match original (machine precision)
    assert_allclose(a2, a_orig, rtol=1e-13, atol=1e-14)
    assert_allclose(b2, b_orig, rtol=1e-13, atol=1e-14)
    assert_allclose(c2, c_orig, rtol=1e-13, atol=1e-14)
    assert_allclose(d2, d_orig, rtol=1e-13, atol=1e-14)


def test_ab04md_eigenvalue_mapping():
    """
    Test eigenvalue mapping property of bilinear transformation.

    For continuous->discrete with alpha=beta=1:
    z = (1 + s) / (1 - s)

    This maps left half-plane (Re(s) < 0) to unit disk (|z| < 1).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab04md

    np.random.seed(123)
    n, m, p = 4, 1, 1

    # Create stable continuous-time system (all eigenvalues in LHP)
    # Use a Schur-form matrix with known negative real eigenvalues
    eigs_cont = np.array([-0.5, -1.0, -2.0, -3.0])
    a_cont = np.diag(eigs_cont).astype(float, order='F')
    b_cont = np.ones((n, m), order='F', dtype=float)
    c_cont = np.ones((p, n), order='F', dtype=float)
    d_cont = np.zeros((p, m), order='F', dtype=float)

    # Transform to discrete-time
    a_disc, b_disc, c_disc, d_disc, info = ab04md('C', a_cont, b_cont, c_cont, d_cont, alpha=1.0, beta=1.0)
    assert info == 0

    # Compute discrete eigenvalues
    eigs_disc = np.linalg.eigvals(a_disc)

    # Expected discrete eigenvalues: z = (1 + s) / (1 - s)
    eigs_disc_expected = (1.0 + eigs_cont) / (1.0 - eigs_cont)

    # Sort for comparison
    assert_allclose(sorted(eigs_disc.real), sorted(eigs_disc_expected.real), rtol=1e-13, atol=1e-14)

    # All discrete eigenvalues should be inside unit circle
    assert np.all(np.abs(eigs_disc) < 1.0)


def test_ab04md_transfer_function_equivalence():
    """
    Test that transfer function is preserved under bilinear transformation.

    For SISO system, H(s) = D + C(sI-A)^{-1}B should equal
    H(z) evaluated at z = (beta+s)/(beta-s)*alpha

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import ab04md

    np.random.seed(456)
    n, m, p = 2, 1, 1
    alpha, beta = 1.0, 1.0

    # Simple stable continuous-time system
    a_cont_orig = np.array([[-1.0, 0.0],
                            [0.0, -2.0]], order='F', dtype=float)
    b_cont_orig = np.array([[1.0],
                            [1.0]], order='F', dtype=float)
    c_cont_orig = np.array([[1.0, 1.0]], order='F', dtype=float)
    d_cont_orig = np.array([[0.5]], order='F', dtype=float)

    # Transform (use copies since ab04md modifies in place)
    a_disc, b_disc, c_disc, d_disc, info = ab04md('C',
                                                   a_cont_orig.copy(),
                                                   b_cont_orig.copy(),
                                                   c_cont_orig.copy(),
                                                   d_cont_orig.copy(),
                                                   alpha=alpha, beta=beta)
    assert info == 0

    # Evaluate transfer function at several points and verify equivalence
    # Pick s values that map to valid z values
    s_vals = [0.1j, 0.5j, 1.0j, 2.0j]

    for s in s_vals:
        # Continuous transfer function: H_c(s) = D + C(sI-A)^{-1}B
        I = np.eye(n, dtype=complex)
        H_cont = d_cont_orig + c_cont_orig @ np.linalg.solve(s * I - a_cont_orig, b_cont_orig)

        # Map s to z using bilinear transform: z = alpha*(beta+s)/(beta-s)
        z = alpha * (beta + s) / (beta - s)

        # Discrete transfer function: H_d(z) = D + C(zI-A)^{-1}B
        H_disc = d_disc + c_disc @ np.linalg.solve(z * I - a_disc, b_disc)

        # Should be equal
        assert_allclose(H_cont, H_disc, rtol=1e-12, atol=1e-13)


def test_ab04md_zero_dimension():
    """
    Test edge case: n=0 (empty state matrix).

    Should return immediately with info=0.
    """
    from ctrlsys import ab04md

    # N=0 system (static gain only)
    m, p = 2, 2
    a = np.array([], order='F', dtype=float).reshape(0, 0)
    b = np.array([], order='F', dtype=float).reshape(0, m)
    c = np.array([], order='F', dtype=float).reshape(p, 0)
    d = np.array([[1.0, 2.0],
                  [3.0, 4.0]], order='F', dtype=float)

    # Should handle empty matrices gracefully
    a_out, b_out, c_out, d_out, info = ab04md('C', a, b, c, d, alpha=1.0, beta=1.0)

    assert info == 0
    assert a_out.shape == (0, 0)
    assert b_out.shape == (0, m)
    assert c_out.shape == (p, 0)
    # D unchanged for n=0 case (no transformation needed)


def test_ab04md_custom_alpha_beta():
    """
    Test with non-unit alpha and beta values.

    The transformation should scale correctly with different parameters.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import ab04md

    np.random.seed(789)
    n, m, p = 2, 1, 1
    alpha, beta = 2.0, 0.5

    # Simple system
    a = np.array([[-1.0, 0.0],
                  [0.0, -0.5]], order='F', dtype=float)
    b = np.array([[1.0],
                  [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 0.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    # Store original
    a_orig = a.copy()

    # Transform continuous -> discrete
    a_out, b_out, c_out, d_out, info = ab04md('C', a, b, c, d, alpha=alpha, beta=beta)
    assert info == 0

    # Verify eigenvalue mapping: z = alpha*(beta+s)/(beta-s)
    eigs_cont = np.linalg.eigvals(a_orig)
    eigs_disc = np.linalg.eigvals(a_out)
    eigs_disc_expected = alpha * (beta + eigs_cont) / (beta - eigs_cont)

    assert_allclose(sorted(eigs_disc.real), sorted(eigs_disc_expected.real), rtol=1e-12, atol=1e-13)


def test_ab04md_error_invalid_type():
    """
    Test error handling for invalid TYPE parameter.
    """
    from ctrlsys import ab04md

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    with pytest.raises(ValueError):
        ab04md('X', a, b, c, d, alpha=1.0, beta=1.0)


def test_ab04md_error_zero_alpha():
    """
    Test error handling for alpha=0.
    """
    from ctrlsys import ab04md

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    with pytest.raises(ValueError):
        ab04md('C', a, b, c, d, alpha=0.0, beta=1.0)


def test_ab04md_error_zero_beta():
    """
    Test error handling for beta=0.
    """
    from ctrlsys import ab04md

    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    c = np.array([[1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    with pytest.raises(ValueError):
        ab04md('C', a, b, c, d, alpha=1.0, beta=0.0)


def test_ab04md_singular_matrix_discrete_to_continuous():
    """
    Test error when (alpha*I + A) is singular during discrete->continuous.

    If A has eigenvalue -alpha, the transformation fails.
    """
    from ctrlsys import ab04md

    alpha = 1.0
    # A has eigenvalue -1, so (alpha*I + A) = (I + A) is singular
    a = np.array([[-1.0, 0.0],
                  [0.0, 0.5]], order='F', dtype=float)
    b = np.array([[1.0],
                  [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    a_out, b_out, c_out, d_out, info = ab04md('D', a, b, c, d, alpha=alpha, beta=1.0)
    assert info == 1  # Matrix (ALPHA*I + A) is singular


def test_ab04md_singular_matrix_continuous_to_discrete():
    """
    Test error when (beta*I - A) is singular during continuous->discrete.

    If A has eigenvalue beta, the transformation fails.
    """
    from ctrlsys import ab04md

    beta = 1.0
    # A has eigenvalue 1, so (beta*I - A) = (I - A) is singular
    a = np.array([[1.0, 0.0],
                  [0.0, -0.5]], order='F', dtype=float)
    b = np.array([[1.0],
                  [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    a_out, b_out, c_out, d_out, info = ab04md('C', a, b, c, d, alpha=1.0, beta=beta)
    assert info == 2  # Matrix (BETA*I - A) is singular
