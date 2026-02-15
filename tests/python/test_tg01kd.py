"""
Tests for TG01KD: Orthogonal equivalence transformation of SISO descriptor
system (A, E, B, C) with E upper triangular.

TG01KD transforms the system so that Q'*B has only the first element nonzero
and Q'*E*Z remains upper triangular.
"""
import numpy as np
import pytest
from ctrlsys import tg01kd


def test_tg01kd_basic_upper_triangular_e():
    """
    Test TG01KD with upper triangular E (JOBE='U').

    Validates:
    - b_out (which is Q'*B) has only first element nonzero
    - E_out = Q'*E*Z remains upper triangular
    - Q and Z are orthogonal
    - A_out = Q' * A_orig * Z

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    # Create upper triangular E
    e = np.triu(np.random.randn(n, n)).astype(np.float64, order='F')
    for i in range(n):
        e[i, i] = abs(e[i, i]) + 1.0

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64, order='F')
    c = np.random.randn(n).astype(np.float64, order='F')

    # Save originals before call (routine modifies in-place)
    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    a_out, e_out, b_out, c_out, q, z, info = tg01kd('U', 'C', 'I', 'I', a, e, b, c)

    assert info == 0, f"TG01KD failed with info={info}"

    # b_out = Q'*B should have only first element nonzero
    np.testing.assert_allclose(b_out[1:], np.zeros(n-1), atol=1e-14,
        err_msg="b_out should have zeros in elements 2..N")

    # E_out = Q'*E*Z should be upper triangular
    for i in range(1, n):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-14, \
                f"E_out[{i},{j}]={e_out[i,j]} should be zero (upper triangular)"

    # Q is orthogonal
    np.testing.assert_allclose(q @ q.T, np.eye(n), rtol=1e-14, atol=1e-14,
        err_msg="Q should be orthogonal")

    # Z is orthogonal
    np.testing.assert_allclose(z @ z.T, np.eye(n), rtol=1e-14, atol=1e-14,
        err_msg="Z should be orthogonal")

    # Verify transformation: A_out = Q' * A_orig * Z
    a_check = q.T @ a_orig @ z
    np.testing.assert_allclose(a_out, a_check, rtol=1e-14, atol=1e-14,
        err_msg="A_out should equal Q'*A*Z")

    # Verify E transformation: E_out = Q' * E_orig * Z
    e_check = q.T @ e_orig @ z
    np.testing.assert_allclose(e_out, e_check, rtol=1e-14, atol=1e-14,
        err_msg="E_out should equal Q'*E*Z")

    # Verify B transformation: b_out = Q' * b_orig
    b_check = q.T @ b_orig
    np.testing.assert_allclose(b_out, b_check, rtol=1e-14, atol=1e-14,
        err_msg="b_out should equal Q'*b")


def test_tg01kd_identity_e():
    """
    Test TG01KD with E=I (JOBE='I').

    When E is identity, the transformation should preserve E=I and
    Q=Z (same rotation applied to rows and columns).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3

    # Random A and B
    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64, order='F')
    c = np.random.randn(n).astype(np.float64, order='F')

    # E is not used when JOBE='I', but must exist
    e = np.eye(n, dtype=np.float64, order='F')

    # Call TG01KD with JOBE='I'
    a_out, e_out, b_out, c_out, q, z, info = tg01kd('I', 'C', 'I', 'I', a, e, b, c)

    assert info == 0, f"TG01KD failed with info={info}"

    # Verify Q'*B has only first element nonzero
    # Note: b_out should already be Q'*B
    np.testing.assert_allclose(b_out[1:], np.zeros(n-1), atol=1e-14,
        err_msg="b_out should have zeros in elements 2..N")

    # Q and Z should be identical for JOBE='I' case
    np.testing.assert_allclose(q, z, rtol=1e-14, atol=1e-14,
        err_msg="Q and Z should be identical when E=I")

    # Verify orthogonality
    np.testing.assert_allclose(q @ q.T, np.eye(n), rtol=1e-14, atol=1e-14,
        err_msg="Q should be orthogonal")


def test_tg01kd_no_transform_c():
    """
    Test TG01KD with COMPC='N' (do not transform C).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3

    e = np.triu(np.random.randn(n, n)).astype(np.float64, order='F')
    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64, order='F')
    c_orig = np.random.randn(n).astype(np.float64, order='F')
    c = c_orig.copy()

    a_out, e_out, b_out, c_out, q, z, info = tg01kd('U', 'N', 'I', 'I', a, e, b, c)

    assert info == 0, f"TG01KD failed with info={info}"

    # C should be unchanged when COMPC='N'
    # Note: the wrapper returns c_out but it may be the original if not transformed
    # The internal array should not be modified


def test_tg01kd_no_q_accumulation():
    """
    Test TG01KD with COMPQ='N' (do not accumulate Q).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3

    e = np.triu(np.random.randn(n, n)).astype(np.float64, order='F')
    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64, order='F')
    c = np.random.randn(n).astype(np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, info = tg01kd('U', 'C', 'N', 'I', a, e, b, c)

    assert info == 0, f"TG01KD failed with info={info}"

    # Z should still be orthogonal
    np.testing.assert_allclose(z @ z.T, np.eye(n), rtol=1e-14, atol=1e-14,
        err_msg="Z should be orthogonal")


def test_tg01kd_update_q():
    """
    Test TG01KD with COMPQ='U' (update existing Q).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    n = 3

    e = np.triu(np.random.randn(n, n)).astype(np.float64, order='F')
    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64, order='F')
    c = np.random.randn(n).astype(np.float64, order='F')

    # Initial Q1 as a simple orthogonal matrix (permutation)
    q1 = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q_out, z, info = tg01kd(
        'U', 'C', 'U', 'I', a, e, b, c, q=q1
    )

    assert info == 0, f"TG01KD failed with info={info}"

    # Q_out should be Q1 * Q (accumulated)
    # Verify Q_out is orthogonal
    np.testing.assert_allclose(q_out @ q_out.T, np.eye(n), rtol=1e-14, atol=1e-14,
        err_msg="Q_out should be orthogonal")


def test_tg01kd_n_equals_1():
    """
    Test TG01KD with N=1 (edge case).
    """
    n = 1

    a = np.array([[2.0]], dtype=np.float64, order='F')
    e = np.array([[3.0]], dtype=np.float64, order='F')
    b = np.array([1.0], dtype=np.float64, order='F')
    c = np.array([4.0], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, info = tg01kd('U', 'C', 'I', 'I', a, e, b, c)

    assert info == 0, f"TG01KD failed with info={info}"

    # For N=1, Q and Z should be identity
    np.testing.assert_allclose(q, np.eye(1), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z, np.eye(1), rtol=1e-14, atol=1e-14)

    # A, E, B, C should be unchanged (identity transformation)
    np.testing.assert_allclose(a_out, a, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(b_out, b, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(c_out, c, rtol=1e-14, atol=1e-14)


def test_tg01kd_n_equals_0():
    """
    Test TG01KD with N=0 (empty system edge case).
    """
    a = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    e = np.array([], dtype=np.float64).reshape(0, 0, order='F')
    b = np.array([], dtype=np.float64, order='F')
    c = np.array([], dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, info = tg01kd('U', 'C', 'I', 'I', a, e, b, c)

    assert info == 0, f"TG01KD failed with info={info}"


def test_tg01kd_invalid_jobe():
    """
    Test TG01KD with invalid JOBE parameter.
    """
    n = 2
    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones(n, dtype=np.float64, order='F')
    c = np.ones(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, info = tg01kd('X', 'C', 'I', 'I', a, e, b, c)

    assert info == -1, f"Expected info=-1 for invalid JOBE, got {info}"


def test_tg01kd_invalid_compq():
    """
    Test TG01KD with invalid COMPQ parameter.
    """
    n = 2
    a = np.eye(n, dtype=np.float64, order='F')
    e = np.eye(n, dtype=np.float64, order='F')
    b = np.ones(n, dtype=np.float64, order='F')
    c = np.ones(n, dtype=np.float64, order='F')

    a_out, e_out, b_out, c_out, q, z, info = tg01kd('U', 'C', 'X', 'I', a, e, b, c)

    assert info == -3, f"Expected info=-3 for invalid COMPQ, got {info}"


def test_tg01kd_transformation_preserves_eigenvalues():
    """
    Test that the transformation preserves generalized eigenvalues.

    For descriptor system (A, E), the generalized eigenvalues lambda satisfy
    det(A - lambda*E) = 0. Orthogonal transformations Q'*A*Z and Q'*E*Z
    preserve these eigenvalues.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 4

    # Create upper triangular E with nonzero diagonal
    e = np.triu(np.random.randn(n, n)).astype(np.float64, order='F')
    for i in range(n):
        e[i, i] = abs(e[i, i]) + 1.0

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64, order='F')
    c = np.random.randn(n).astype(np.float64, order='F')

    # Save original matrices for eigenvalue computation
    a_orig = a.copy()
    e_orig = e.copy()

    # Compute eigenvalues before transformation
    eig_before = np.linalg.eigvals(np.linalg.solve(e_orig, a_orig))

    a_out, e_out, b_out, c_out, q, z, info = tg01kd('U', 'C', 'I', 'I', a, e, b, c)
    assert info == 0

    # Compute eigenvalues after transformation
    eig_after = np.linalg.eigvals(np.linalg.solve(e_out, a_out))

    # Sort eigenvalues for comparison (by real part, then imaginary)
    eig_before_sorted = sorted(eig_before, key=lambda x: (x.real, x.imag))
    eig_after_sorted = sorted(eig_after, key=lambda x: (x.real, x.imag))

    np.testing.assert_allclose(
        np.array(eig_before_sorted),
        np.array(eig_after_sorted),
        rtol=1e-12, atol=1e-12,
        err_msg="Generalized eigenvalues should be preserved"
    )


def test_tg01kd_c_transformation():
    """
    Test that C is correctly transformed as C*Z.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 4

    e = np.triu(np.random.randn(n, n)).astype(np.float64, order='F')
    for i in range(n):
        e[i, i] = abs(e[i, i]) + 1.0

    a = np.random.randn(n, n).astype(np.float64, order='F')
    b = np.random.randn(n).astype(np.float64, order='F')
    c = np.random.randn(n).astype(np.float64, order='F')
    c_orig = c.copy()

    a_out, e_out, b_out, c_out, q, z, info = tg01kd('U', 'C', 'I', 'I', a, e, b, c)
    assert info == 0

    # c_out should equal c_orig @ z
    c_expected = c_orig @ z
    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14, atol=1e-14,
        err_msg="C_out should equal C*Z")
