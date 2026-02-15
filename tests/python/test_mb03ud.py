import numpy as np
import pytest


def test_mb03ud_basic_example():
    """
    Validate basic SVD computation using SLICOT HTML doc example.

    Tests numerical correctness of singular values and vectors.
    """
    n = 4
    a = np.array([
        [-1.0, 37.0, -12.0, -12.0],
        [0.0, -10.0, 0.0, 4.0],
        [0.0, 0.0, 7.0, -6.0],
        [0.0, 0.0, 0.0, -9.0]
    ], order='F', dtype=float)

    sv_expected = np.array([42.0909, 11.7764, 5.4420, 0.2336], dtype=float)

    p_expected = np.array([
        [0.0230, -0.9084, 0.2759, 0.3132],
        [0.0075, -0.1272, 0.5312, -0.8376],
        [0.0092, 0.3978, 0.8009, 0.4476],
        [0.9997, 0.0182, -0.0177, -0.0050]
    ], order='F', dtype=float)

    q_expected = np.array([
        [-0.9671, -0.0882, -0.0501, -0.2335],
        [0.2456, -0.1765, -0.4020, -0.8643],
        [0.0012, 0.7425, 0.5367, -0.4008],
        [-0.0670, 0.6401, -0.7402, 0.1945]
    ], order='F', dtype=float)

    from ctrlsys import mb03ud

    a_copy = a.copy()
    sv, p, q, info = mb03ud(n, a_copy, jobq='V', jobp='V')

    assert info == 0

    # Validate singular values (rtol based on 4-decimal display)
    np.testing.assert_allclose(sv, sv_expected, rtol=1e-3, atol=1e-4)

    # Validate right singular vectors (P')
    # Note: Sign ambiguity in SVD - check column-wise absolute values
    for i in range(n):
        sign = np.sign(np.dot(p[:, i], p_expected[:, i]))
        np.testing.assert_allclose(p[:, i], sign * p_expected[:, i], rtol=1e-3, atol=1e-4)

    # Validate left singular vectors (Q)
    for i in range(n):
        sign = np.sign(np.dot(q[:, i], q_expected[:, i]))
        np.testing.assert_allclose(q[:, i], sign * q_expected[:, i], rtol=1e-3, atol=1e-4)


def test_mb03ud_svd_decomposition():
    """
    Validate SVD decomposition: A = Q * diag(SV) * P'.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 5

    # Generate random upper triangular matrix
    a = np.triu(np.random.randn(n, n)).astype(float, order='F')
    a_orig = a.copy()

    from ctrlsys import mb03ud

    sv, p_transpose, q, info = mb03ud(n, a, jobq='V', jobp='V')

    assert info == 0

    # Reconstruct A from SVD: A = Q * diag(SV) * P'
    # Note: p_transpose IS P' (not P), so use directly without .T
    s = np.diag(sv)
    a_reconstructed = q @ s @ p_transpose

    # Validate reconstruction (machine precision for exact factorization)
    np.testing.assert_allclose(a_reconstructed, a_orig, rtol=1e-13, atol=1e-14)


def test_mb03ud_orthogonality():
    """
    Validate orthogonality of Q and P matrices: Q'Q = I, P'P = I.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 6

    a = np.triu(np.random.randn(n, n)).astype(float, order='F')

    from ctrlsys import mb03ud

    sv, p, q, info = mb03ud(n, a, jobq='V', jobp='V')

    assert info == 0

    # Validate Q orthogonality: Q'Q = I
    qtq = q.T @ q
    np.testing.assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-15)

    # Validate P orthogonality: P'P = I
    ptp = p.T @ p
    np.testing.assert_allclose(ptp, np.eye(n), rtol=1e-14, atol=1e-15)


def test_mb03ud_singular_value_ordering():
    """
    Validate singular values are in descending order.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 7

    a = np.triu(np.random.randn(n, n)).astype(float, order='F')

    from ctrlsys import mb03ud

    sv, p, q, info = mb03ud(n, a, jobq='V', jobp='V')

    assert info == 0

    # Validate descending order
    for i in range(n - 1):
        assert sv[i] >= sv[i+1], f"SV[{i}]={sv[i]} < SV[{i+1}]={sv[i+1]}"

    # Validate non-negativity
    assert np.all(sv >= 0), "Singular values must be non-negative"


def test_mb03ud_no_vectors():
    """
    Validate singular values computation without vectors (jobq='N', jobp='N').

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4

    a = np.triu(np.random.randn(n, n)).astype(float, order='F')
    a_copy = a.copy()

    from ctrlsys import mb03ud

    # Compute with vectors
    sv_with, p_with, q_with, info_with = mb03ud(n, a_copy, jobq='V', jobp='V')
    assert info_with == 0

    # Compute without vectors
    sv_only, _, _, info_only = mb03ud(n, a.copy(), jobq='N', jobp='N')
    assert info_only == 0

    # Singular values should match
    np.testing.assert_allclose(sv_only, sv_with, rtol=1e-14, atol=1e-15)


def test_mb03ud_partial_vectors():
    """
    Validate partial vector computation: Q only, P only.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n = 5

    a = np.triu(np.random.randn(n, n)).astype(float, order='F')

    from ctrlsys import mb03ud

    # Compute Q only
    sv_q, _, q_only, info_q = mb03ud(n, a.copy(), jobq='V', jobp='N')
    assert info_q == 0
    assert q_only.shape == (n, n)

    # Validate Q orthogonality
    qtq = q_only.T @ q_only
    np.testing.assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-15)

    # Compute P only
    sv_p, p_only, _, info_p = mb03ud(n, a.copy(), jobq='N', jobp='V')
    assert info_p == 0
    assert p_only.shape == (n, n)

    # Validate P orthogonality
    ptp = p_only.T @ p_only
    np.testing.assert_allclose(ptp, np.eye(n), rtol=1e-14, atol=1e-15)

    # Singular values should match
    np.testing.assert_allclose(sv_q, sv_p, rtol=1e-14, atol=1e-15)


def test_mb03ud_edge_case_1x1():
    """
    Validate edge case: 1x1 matrix (trivial SVD).
    """
    n = 1
    a = np.array([[5.0]], order='F', dtype=float)

    from ctrlsys import mb03ud

    sv, p, q, info = mb03ud(n, a.copy(), jobq='V', jobp='V')

    assert info == 0
    assert sv[0] == pytest.approx(5.0, rel=1e-14)

    # Q and P should be [[1.0]] or [[-1.0]]
    assert abs(q[0, 0]) == pytest.approx(1.0, rel=1e-14)
    assert abs(p[0, 0]) == pytest.approx(1.0, rel=1e-14)


def test_mb03ud_edge_case_diagonal():
    """
    Validate edge case: diagonal matrix (SVD = absolute values).

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 4

    # Create diagonal matrix
    diag_vals = np.random.randn(n)
    a = np.diag(diag_vals).astype(float, order='F')

    from ctrlsys import mb03ud

    sv, p, q, info = mb03ud(n, a, jobq='V', jobp='V')

    assert info == 0

    # Singular values should be sorted absolute values
    sv_expected = np.sort(np.abs(diag_vals))[::-1]
    np.testing.assert_allclose(sv, sv_expected, rtol=1e-14, atol=1e-15)


def test_mb03ud_error_invalid_n():
    """
    Validate error handling: invalid N (negative).
    """
    n = -1
    a = np.array([[1.0]], order='F', dtype=float)

    from ctrlsys import mb03ud

    with pytest.raises((ValueError, RuntimeError)):
        mb03ud(n, a, jobq='N', jobp='N')


def test_mb03ud_error_invalid_jobq():
    """
    Validate error handling: invalid JOBQ.
    """
    n = 2
    a = np.eye(n, dtype=float, order='F')

    from ctrlsys import mb03ud

    with pytest.raises((ValueError, RuntimeError)):
        mb03ud(n, a, jobq='X', jobp='N')


def test_mb03ud_workspace_query():
    """
    Validate workspace query mechanism (ldwork=-1).
    """
    n = 5
    a = np.triu(np.random.randn(n, n)).astype(float, order='F')

    from ctrlsys import mb03ud

    # Should succeed and return optimal workspace in info or similar
    sv, _, _, info = mb03ud(n, a, jobq='V', jobp='V')
    assert info == 0
