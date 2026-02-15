"""
Tests for TB01ID - System Matrix Balancing.

Balances the system matrix S = [A B; C 0] using diagonal similarity transformation.
"""
import numpy as np
import pytest


def test_tb01id_basic_html_example():
    """
    Test using HTML doc example data.

    N=5, M=2, P=5, JOB='A', MAXRED=0.0 (uses default 10.0)
    """
    from ctrlsys import tb01id

    n, m, p = 5, 2, 5

    # Input A matrix (5x5) from HTML doc
    a = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [-1.58e6, -1.257e3, 0.0, 0.0, 0.0],
        [3.541e14, 0.0, -1.434e3, 0.0, -5.33e11],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.863e4, -1.482]
    ], order='F', dtype=float)

    # Input B matrix (5x2) from HTML doc
    b = np.array([
        [0.0, 0.0],
        [1.103e2, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 8.333e-3]
    ], order='F', dtype=float)

    # Input C matrix (5x5) from HTML doc
    c = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [6.664e-1, 0.0, -6.2e-13, 0.0, 0.0],
        [0.0, 0.0, -1.0e-3, 1.896e6, 1.508e2]
    ], order='F', dtype=float)

    # Expected balanced A from HTML doc
    a_expected = np.array([
        [0.0, 1.0e4, 0.0, 0.0, 0.0],
        [-1.58e2, -1.257e3, 0.0, 0.0, 0.0],
        [3.541e4, 0.0, -1.434e3, 0.0, -5.33e2],
        [0.0, 0.0, 0.0, 0.0, 1.0e2],
        [0.0, 0.0, 0.0, -1.863e2, -1.482]
    ], order='F', dtype=float)

    # Expected balanced B from HTML doc
    b_expected = np.array([
        [0.0, 0.0],
        [1.103e3, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 8.333e1]
    ], order='F', dtype=float)

    # Expected balanced C from HTML doc
    c_expected = np.array([
        [1.0e-5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0e5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0e-6, 0.0],
        [6.664e-6, 0.0, -6.2e-8, 0.0, 0.0],
        [0.0, 0.0, -1.0e2, 1.896, 1.508e-2]
    ], order='F', dtype=float)

    # Expected scale factors from HTML doc
    scale_expected = np.array([1.0e-5, 1.0e-1, 1.0e5, 1.0e-6, 1.0e-4], dtype=float)

    # MAXRED output from HTML doc
    maxred_expected = 0.3488e10

    maxred_in = 0.0  # Use default 10.0

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a, b, c, maxred_in)

    assert info == 0

    # Validate output matrices (HTML doc has 4-digit precision)
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-10)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-10)
    np.testing.assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-10)
    np.testing.assert_allclose(scale, scale_expected, rtol=1e-3, atol=1e-10)
    np.testing.assert_allclose(maxred_out, maxred_expected, rtol=1e-2)


def test_tb01id_similarity_transform_property():
    """
    Validate mathematical property: D^{-1} A D is equivalent to original A
    under the returned scaling.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tb01id

    np.random.seed(42)
    n, m, p = 4, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    # Save original for verification
    a_orig = a.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

    # Verify transformation property: a_out = D^{-1} * a_orig * D
    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)

    a_reconstructed = d_inv @ a_orig @ d
    np.testing.assert_allclose(a_out, a_reconstructed, rtol=1e-14)

    # Verify b_out = D^{-1} * b_orig
    b_reconstructed = d_inv @ b_orig
    np.testing.assert_allclose(b_out, b_reconstructed, rtol=1e-14)

    # Verify c_out = c_orig * D
    c_reconstructed = c_orig @ d
    np.testing.assert_allclose(c_out, c_reconstructed, rtol=1e-14)


def test_tb01id_eigenvalue_preservation():
    """
    Validate eigenvalue preservation under similarity transformation.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import tb01id

    np.random.seed(123)
    n, m, p = 5, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    eig_before = np.linalg.eigvals(a)

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

    eig_after = np.linalg.eigvals(a_out)

    # Sort by real part then imaginary for comparison
    eig_before_sorted = sorted(eig_before, key=lambda x: (x.real, x.imag))
    eig_after_sorted = sorted(eig_after, key=lambda x: (x.real, x.imag))

    np.testing.assert_allclose(
        np.array(eig_before_sorted),
        np.array(eig_after_sorted),
        rtol=1e-13
    )


def test_tb01id_job_n_a_only():
    """
    Test JOB='N': Only A matrix is balanced (B and C not involved).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tb01id

    np.random.seed(456)
    n, m, p = 4, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_orig = a.copy()

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('N', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

    # Verify transformation property for A only
    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)
    a_reconstructed = d_inv @ a_orig @ d
    np.testing.assert_allclose(a_out, a_reconstructed, rtol=1e-14)


def test_tb01id_job_b():
    """
    Test JOB='B': B and A matrices are involved in balancing.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tb01id

    np.random.seed(789)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_orig = a.copy()
    b_orig = b.copy()

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('B', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

    # Verify transformation properties
    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)

    a_reconstructed = d_inv @ a_orig @ d
    np.testing.assert_allclose(a_out, a_reconstructed, rtol=1e-14)

    b_reconstructed = d_inv @ b_orig
    np.testing.assert_allclose(b_out, b_reconstructed, rtol=1e-14)


def test_tb01id_job_c():
    """
    Test JOB='C': C and A matrices are involved in balancing.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import tb01id

    np.random.seed(111)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_orig = a.copy()
    c_orig = c.copy()

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('C', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

    # Verify transformation properties
    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)

    a_reconstructed = d_inv @ a_orig @ d
    np.testing.assert_allclose(a_out, a_reconstructed, rtol=1e-14)

    c_reconstructed = c_orig @ d
    np.testing.assert_allclose(c_out, c_reconstructed, rtol=1e-14)


def test_tb01id_mimo_balancing():
    """
    MIMO system balancing: 5-state, 3-input, 4-output.
    Verify eigenvalue preservation and transformation properties.
    """
    from ctrlsys import tb01id

    np.random.seed(700)
    n, m, p = 5, 3, 4

    a = np.array([
        [1.0, 1e4, 0.0, 0.0, 0.0],
        [1e-4, 1.0, 1e3, 0.0, 0.0],
        [0.0, 1e-3, 1.0, 1e2, 0.0],
        [0.0, 0.0, 1e-2, 1.0, 1e1],
        [0.0, 0.0, 0.0, 1e-1, 1.0]
    ], order='F', dtype=float)

    b = np.asfortranarray(np.random.randn(n, m))
    c = np.asfortranarray(np.random.randn(p, n))

    a_orig = a.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    eig_before = np.linalg.eigvals(a_orig)

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

    eig_after = np.linalg.eigvals(a_out)
    eig_before_sorted = np.sort(np.abs(eig_before))
    eig_after_sorted = np.sort(np.abs(eig_after))
    np.testing.assert_allclose(eig_before_sorted, eig_after_sorted, rtol=1e-8, atol=1e-14)

    d = np.diag(scale)
    d_inv = np.diag(1.0 / scale)
    np.testing.assert_allclose(a_out, d_inv @ a_orig @ d, rtol=1e-14)
    np.testing.assert_allclose(b_out, d_inv @ b_orig, rtol=1e-14)
    np.testing.assert_allclose(c_out, c_orig @ d, rtol=1e-14)


def test_tb01id_n_zero():
    """
    Test edge case: n=0 (empty system).
    """
    from ctrlsys import tb01id

    a = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, 2), order='F', dtype=float)
    c = np.zeros((3, 0), order='F', dtype=float)

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a, b, c, 0.0)

    assert info == 0
    assert scale.shape == (0,)


def test_tb01id_m_zero():
    """
    Test edge case: m=0 (no inputs).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import tb01id

    np.random.seed(222)
    n, m, p = 3, 0, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.zeros((n, 0), order='F', dtype=float)
    c = np.random.randn(p, n).astype(float, order='F')

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a, b, c, 0.0)

    assert info == 0
    assert b_out.shape == (n, 0)


def test_tb01id_p_zero():
    """
    Test edge case: p=0 (no outputs).

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import tb01id

    np.random.seed(333)
    n, m, p = 3, 2, 0

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.zeros((0, n), order='F', dtype=float)

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a, b, c, 0.0)

    assert info == 0
    assert c_out.shape == (0, n)


def test_tb01id_zero_matrix():
    """
    Test edge case: All zero matrices - should return immediately.
    """
    from ctrlsys import tb01id

    n, m, p = 3, 2, 2

    a = np.zeros((n, n), order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)
    c = np.zeros((p, n), order='F', dtype=float)

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a, b, c, 0.0)

    assert info == 0
    # Scale should be all 1.0 (no balancing done)
    np.testing.assert_array_equal(scale, np.ones(n))


def test_tb01id_error_invalid_job():
    """
    Test error handling: Invalid JOB parameter.
    """
    from ctrlsys import tb01id

    n, m, p = 2, 1, 1
    a = np.zeros((n, n), order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)
    c = np.zeros((p, n), order='F', dtype=float)

    with pytest.raises(ValueError):
        tb01id('X', a, b, c, 0.0)


def test_tb01id_error_maxred():
    """
    Test error handling: MAXRED in (0, 1) range (invalid).
    """
    from ctrlsys import tb01id

    n, m, p = 2, 1, 1
    a = np.zeros((n, n), order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)
    c = np.zeros((p, n), order='F', dtype=float)

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a, b, c, 0.5)

    assert info == -5


def test_tb01id_maxred_custom():
    """
    Test custom MAXRED value.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import tb01id

    np.random.seed(555)
    n, m, p = 3, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    # Use custom MAXRED > 1
    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a.copy(), b.copy(), c.copy(), 5.0)

    assert info == 0
    # MAXRED should be updated to ratio of norms (>= 1 typically)
    assert maxred_out >= 1.0 or np.isclose(maxred_out, 1.0, rtol=1e-10)


def test_tb01id_norm_reduction():
    """
    Test that balancing reduces or maintains 1-norm.

    Random seed: 666 (for reproducibility)
    """
    from ctrlsys import tb01id

    np.random.seed(666)
    n, m, p = 4, 2, 3

    # Create ill-conditioned system with large norm differences
    a = np.array([
        [1.0, 1e6, 0.0, 0.0],
        [1e-6, 1.0, 1e6, 0.0],
        [0.0, 1e-6, 1.0, 1e6],
        [0.0, 0.0, 1e-6, 1.0]
    ], order='F', dtype=float)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    # Compute 1-norm of original system matrix
    s_orig = np.block([[a, b], [c, np.zeros((p, m))]])
    norm_orig = np.max(np.sum(np.abs(s_orig), axis=0))

    a_out, b_out, c_out, maxred_out, scale, info = tb01id('A', a.copy(), b.copy(), c.copy(), 0.0)

    assert info == 0

    # Compute 1-norm of balanced system
    s_bal = np.block([[a_out, b_out], [c_out, np.zeros((p, m))]])
    norm_bal = np.max(np.sum(np.abs(s_bal), axis=0))

    # MAXRED should equal ratio of norms
    np.testing.assert_allclose(maxred_out, norm_orig / norm_bal, rtol=1e-10)
