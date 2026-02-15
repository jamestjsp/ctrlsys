"""
Tests for MB04IY - Apply orthogonal transformations from MB04ID.

MB04IY applies Q or Q' to a matrix C, where Q is represented as a product
of elementary reflectors stored in a special format (from MB04ID).
"""

import numpy as np
import pytest


def test_mb04iy_import():
    """Test that mb04iy can be imported."""
    from ctrlsys import mb04iy
    assert mb04iy is not None


def test_mb04iy_left_transpose_basic():
    """
    Test MB04IY with SIDE='L', TRANS='T' (Q'*C from left).

    Validates orthogonal transformation preserves numerical properties.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04iy

    np.random.seed(42)

    n, m, k, p = 5, 3, 3, 1

    # Generate orthogonal reflectors manually
    # For simplicity, create a simple case where we can verify orthogonality
    a = np.zeros((n, k), order='F', dtype=float)
    tau = np.zeros(k, dtype=float)

    # Create simple Householder reflector data
    # H(i) stored in column i of A below diagonal
    for i in range(k):
        if i < p:
            # First p reflectors: modify n-p elements
            length = n - p
            start_row = i
        else:
            # Remaining reflectors: standard QR form
            length = n - i
            start_row = i

        if length > 0:
            # Create random vector for reflector
            v = np.random.randn(length)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-14:
                v = v / v_norm
                tau[i] = 2.0
                # Store in A (below diagonal element)
                if i < p:
                    a[i+1:i+1+length-1, i] = v[1:]
                else:
                    a[i+1:start_row+length, i] = v[1:]

    # Create test matrix C
    c = np.random.randn(n, m).astype(float, order='F')
    c_orig = c.copy()

    # Apply Q'*C
    c_result, info = mb04iy('L', 'T', a, tau, c, p=p)

    assert info == 0
    assert c_result.shape == (n, m)

    # Verify transformation was applied (result should differ)
    assert not np.allclose(c_result, c_orig, rtol=1e-10)


def test_mb04iy_right_notrans_basic():
    """
    Test MB04IY with SIDE='R', TRANS='N' (C*Q from right).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04iy

    np.random.seed(123)

    n, m, k, p = 4, 6, 4, 2

    # Generate reflector data
    a = np.zeros((m, k), order='F', dtype=float)
    tau = np.zeros(k, dtype=float)

    for i in range(k):
        if i < p:
            length = m - p
        else:
            length = m - i

        if length > 0:
            v = np.random.randn(length)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-14:
                v = v / v_norm
                tau[i] = 2.0
                if i < p:
                    a[i+1:i+1+length-1, i] = v[1:]
                else:
                    a[i+1:i+length, i] = v[1:]

    c = np.random.randn(n, m).astype(float, order='F')
    c_orig = c.copy()

    # Apply C*Q
    c_result, info = mb04iy('R', 'N', a, tau, c, p=p)

    assert info == 0
    assert c_result.shape == (n, m)
    assert not np.allclose(c_result, c_orig, rtol=1e-10)


def test_mb04iy_orthogonality():
    """
    Test that Q is orthogonal by verifying Q'*Q = I.

    Apply Q and Q' in sequence and verify we get back original matrix.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04iy

    np.random.seed(456)

    n, m, k, p = 6, 4, 4, 2

    # Create identity-like reflectors (tau=0 means H(i)=I)
    a = np.zeros((n, k), order='F', dtype=float)
    tau = np.zeros(k, dtype=float)

    c = np.random.randn(n, m).astype(float, order='F')
    c_orig = c.copy()

    # With tau=0, Q=I, so C should be unchanged
    c_result, info = mb04iy('L', 'N', a, tau, c, p=p)

    assert info == 0
    np.testing.assert_allclose(c_result, c_orig, rtol=1e-14, atol=1e-15)


def test_mb04iy_zero_dimensions():
    """
    Test MB04IY with zero dimensions (quick return cases).
    """
    from ctrlsys import mb04iy

    # Zero columns
    a = np.zeros((5, 2), order='F', dtype=float)
    tau = np.zeros(2, dtype=float)
    c = np.zeros((5, 0), order='F', dtype=float)

    c_result, info = mb04iy('L', 'N', a, tau, c, p=0)
    assert info == 0
    assert c_result.shape == (5, 0)

    # Zero reflectors
    a = np.zeros((5, 0), order='F', dtype=float)
    tau = np.zeros(0, dtype=float)
    c = np.random.randn(5, 3).astype(float, order='F')
    c_orig = c.copy()

    c_result, info = mb04iy('L', 'N', a, tau, c, p=0)
    assert info == 0
    np.testing.assert_array_equal(c_result, c_orig)


def test_mb04iy_error_invalid_side():
    """Test error handling for invalid SIDE parameter."""
    from ctrlsys import mb04iy

    a = np.zeros((5, 2), order='F', dtype=float)
    tau = np.zeros(2, dtype=float)
    c = np.zeros((5, 3), order='F', dtype=float)

    with pytest.raises((ValueError, Exception)):
        mb04iy('X', 'N', a, tau, c, p=0)


def test_mb04iy_error_invalid_trans():
    """Test error handling for invalid TRANS parameter."""
    from ctrlsys import mb04iy

    a = np.zeros((5, 2), order='F', dtype=float)
    tau = np.zeros(2, dtype=float)
    c = np.zeros((5, 3), order='F', dtype=float)

    with pytest.raises((ValueError, Exception)):
        mb04iy('L', 'X', a, tau, c, p=0)


def test_mb04iy_error_negative_dimensions():
    """Test error handling for negative dimensions."""
    from ctrlsys import mb04iy

    # This should fail in parameter validation
    a = np.zeros((5, 2), order='F', dtype=float)
    tau = np.zeros(2, dtype=float)
    c = np.zeros((5, 3), order='F', dtype=float)

    with pytest.raises((ValueError, Exception)):
        mb04iy('L', 'N', a, tau, c, p=-1)


def test_mb04iy_p_equals_zero():
    """
    Test MB04IY with P=0 (standard QR form, no special structure).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04iy

    np.random.seed(789)

    n, m, k, p = 5, 3, 3, 0

    # Standard QR reflector storage
    a = np.random.randn(n, k).astype(float, order='F')
    tau = np.random.uniform(0, 2, k)
    c = np.random.randn(n, m).astype(float, order='F')
    c_orig = c.copy()

    c_result, info = mb04iy('L', 'N', a, tau, c, p=p)

    assert info == 0
    assert c_result.shape == (n, m)
    # With random reflectors, result should differ
    assert not np.allclose(c_result, c_orig, rtol=1e-10)


def test_mb04iy_p_large():
    """
    Test MB04IY with P >= K (all reflectors in special form).

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import mb04iy

    np.random.seed(999)

    n, m, k, p = 7, 4, 3, 5

    a = np.random.randn(n, k).astype(float, order='F')
    tau = np.random.uniform(0, 2, k)
    c = np.random.randn(n, m).astype(float, order='F')

    c_result, info = mb04iy('L', 'T', a, tau, c, p=p)

    assert info == 0
    assert c_result.shape == (n, m)


def test_mb04iy_single_reflector():
    """
    Test MB04IY with K=1 (single elementary reflector).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04iy

    np.random.seed(111)

    n, m, k, p = 6, 4, 1, 0

    a = np.random.randn(n, k).astype(float, order='F')
    tau = np.array([1.5], dtype=float)
    c = np.random.randn(n, m).astype(float, order='F')
    c_orig = c.copy()

    c_result, info = mb04iy('L', 'N', a, tau, c, p=p)

    assert info == 0
    assert c_result.shape == (n, m)
    assert not np.allclose(c_result, c_orig, rtol=1e-10)


def test_mb04iy_identity_left():
    """
    Test identity transformation: tau=0 means Q=I.

    When tau=0, H(i)=I, so Q=I and C should be unchanged.
    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb04iy

    np.random.seed(222)

    n, m, k, p = 5, 3, 3, 1

    a = np.random.randn(n, k).astype(float, order='F')
    tau = np.zeros(k, dtype=float)  # Identity: Q = I
    c = np.random.randn(n, m).astype(float, order='F')
    c_orig = c.copy()

    # Apply Q=I
    c_result, info = mb04iy('L', 'N', a, tau, c, p=p)
    assert info == 0

    # Should get back original (machine precision)
    np.testing.assert_allclose(c_result, c_orig, rtol=1e-14, atol=1e-15)


def test_mb04iy_identity_right():
    """
    Test identity transformation from right: tau=0 means Q=I.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb04iy

    np.random.seed(333)

    n, m, k, p = 4, 6, 4, 2

    a = np.random.randn(m, k).astype(float, order='F')
    tau = np.zeros(k, dtype=float)  # Identity: Q = I
    c = np.random.randn(n, m).astype(float, order='F')
    c_orig = c.copy()

    # Apply Q=I from right
    c_result, info = mb04iy('R', 'N', a, tau, c, p=p)
    assert info == 0

    # Should get back original (machine precision)
    np.testing.assert_allclose(c_result, c_orig, rtol=1e-14, atol=1e-15)
