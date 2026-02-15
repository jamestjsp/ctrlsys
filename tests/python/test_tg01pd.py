"""
Tests for tg01pd - Bi-domain spectral splitting of descriptor system pencil.

TG01PD computes orthogonal transformation matrices Q and Z which reduce
the regular pole pencil A-lambda*E of descriptor system (A-lambda*E,B,C)
to generalized real Schur form with ordered generalized eigenvalues.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tg01pd_html_example():
    """
    Test TG01PD using data from SLICOT HTML documentation.

    System: 4x4 descriptor system with 2 inputs, 2 outputs.
    DICO='C', STDOM='S', JOBAE='G', COMPQ='I', COMPZ='I'
    NLOW=1, NSUP=4, ALPHA=-1.E-7
    Expected: 1 eigenvalue in the domain of interest.
    """
    from ctrlsys import tg01pd

    n, m, p = 4, 2, 2
    nlow, nsup = 1, 4
    alpha = -1e-7

    # A matrix (row-wise from HTML doc)
    a = np.array([
        [-1,  0,  0,  3],
        [ 0,  0,  1,  2],
        [ 1,  1,  0,  4],
        [ 0,  0,  0,  0],
    ], order='F', dtype=float)

    # E matrix (row-wise from HTML doc)
    e = np.array([
        [1, 2, 0, 0],
        [0, 1, 0, 1],
        [3, 9, 6, 3],
        [0, 0, 2, 0],
    ], order='F', dtype=float)

    # B matrix (row-wise from HTML doc)
    b = np.array([
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1],
    ], order='F', dtype=float)

    # C matrix (row-wise from HTML doc)
    c = np.array([
        [-1, 0,  1, 0],
        [ 0, 1, -1, 1],
    ], order='F', dtype=float)

    result = tg01pd('C', 'S', 'G', 'I', 'I', n, m, p, nlow, nsup, alpha, a, e, b, c)

    a_out, e_out, b_out, c_out, q, z, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"
    assert ndim == 1, f"Expected ndim=1, got {ndim}"

    # Verify orthogonality of Q and Z (platform-independent)
    assert_allclose(q.T @ q, np.eye(n), rtol=1e-13, atol=1e-14)
    assert_allclose(z.T @ z, np.eye(n), rtol=1e-13, atol=1e-14)

    # Verify A is quasi-upper triangular (Schur form)
    for i in range(2, n):
        for j in range(i - 1):
            assert abs(a_out[i, j]) < 1e-10, f"A not quasi-upper triangular at [{i},{j}]"

    # Verify E is upper triangular
    for i in range(1, n):
        for j in range(i):
            assert abs(e_out[i, j]) < 1e-10, f"E not upper triangular at [{i},{j}]"

    # Verify diagonal absolute values (sign-independent)
    a_diag_expected = np.array([1.6311, 0.4550, 2.6950, 0.0000])
    e_diag_expected = np.array([0.4484, 3.3099, 0.0000, 2.0000])
    assert_allclose(np.abs(np.diag(a_out)), a_diag_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(np.abs(np.diag(e_out)), e_diag_expected, rtol=1e-3, atol=1e-4)


def test_tg01pd_orthogonality():
    """
    Test orthogonality of Q and Z matrices.

    Mathematical property: Q'*Q = I and Z'*Z = I

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tg01pd

    np.random.seed(42)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float) + 0.1 * np.random.randn(n, n)
    e = e.astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    result = tg01pd('C', 'S', 'G', 'I', 'I', n, m, p, 1, n, 0.0, a, e, b, c)
    a_out, e_out, b_out, c_out, q, z, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"

    # Q'*Q = I
    qtq = q.T @ q
    assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14,
                    err_msg="Q is not orthogonal: Q'*Q != I")

    # Z'*Z = I
    ztz = z.T @ z
    assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14,
                    err_msg="Z is not orthogonal: Z'*Z != I")


def test_tg01pd_equivalence_transformation():
    """
    Test equivalence transformation property.

    Mathematical property:
    - Q'*A_orig*Z = A_out
    - Q'*E_orig*Z = E_out
    - Q'*B_orig = B_out
    - C_orig*Z = C_out

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import tg01pd

    np.random.seed(123)
    n, m, p = 5, 2, 3

    a_orig = np.random.randn(n, n).astype(float, order='F')
    e_orig = np.eye(n, order='F', dtype=float) + 0.2 * np.random.randn(n, n)
    e_orig = e_orig.astype(float, order='F')
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(p, n).astype(float, order='F')

    a = a_orig.copy()
    e = e_orig.copy()
    b = b_orig.copy()
    c = c_orig.copy()

    result = tg01pd('C', 'S', 'G', 'I', 'I', n, m, p, 1, n, 0.0, a, e, b, c)
    a_out, e_out, b_out, c_out, q, z, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"

    # Verify transformations
    assert_allclose(q.T @ a_orig @ z, a_out, rtol=1e-13, atol=1e-14,
                    err_msg="Q'*A*Z != A_out")
    assert_allclose(q.T @ e_orig @ z, e_out, rtol=1e-13, atol=1e-14,
                    err_msg="Q'*E*Z != E_out")
    assert_allclose(q.T @ b_orig, b_out, rtol=1e-13, atol=1e-14,
                    err_msg="Q'*B != B_out")
    assert_allclose(c_orig @ z, c_out, rtol=1e-13, atol=1e-14,
                    err_msg="C*Z != C_out")


def test_tg01pd_eigenvalue_preservation():
    """
    Test eigenvalue preservation under transformation.

    Mathematical property: eigenvalues of (A,E) = eigenvalues of (Q'AZ, Q'EZ)

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tg01pd

    np.random.seed(456)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float) + 0.1 * np.random.randn(n, n)
    e = e.astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    # Compute original eigenvalues
    eig_orig = np.linalg.eigvals(np.linalg.solve(e, a))
    eig_orig = np.sort_complex(eig_orig)

    result = tg01pd('C', 'S', 'G', 'I', 'I', n, m, p, 1, n, 0.0, a, e, b, c)
    a_out, e_out, b_out, c_out, q, z, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"

    # Compute eigenvalues from output
    eig_out = np.linalg.eigvals(np.linalg.solve(e_out, a_out))
    eig_out = np.sort_complex(eig_out)

    assert_allclose(eig_out, eig_orig, rtol=1e-10, atol=1e-12,
                    err_msg="Eigenvalues not preserved")


def test_tg01pd_discrete_time():
    """
    Test discrete-time system (DICO='D').

    For discrete-time, eigenvalues with |lambda| < alpha go in first block.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tg01pd

    np.random.seed(789)
    n, m, p = 4, 2, 2
    alpha = 1.0  # Unit circle

    # Create stable discrete-time system (eigenvalues inside unit circle)
    a = 0.5 * np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    result = tg01pd('D', 'S', 'G', 'I', 'I', n, m, p, 1, n, alpha, a, e, b, c)
    a_out, e_out, b_out, c_out, q, z, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"

    # All eigenvalues should be in stability domain for this system
    eig = np.linalg.eigvals(a)
    n_stable = np.sum(np.abs(eig) < alpha)
    assert ndim == n_stable, f"Expected ndim={n_stable}, got {ndim}"


def test_tg01pd_unstable_domain():
    """
    Test instability domain selection (STDOM='U').

    For STDOM='U', eigenvalues outside the domain are placed first.

    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import tg01pd

    np.random.seed(101)
    n, m, p = 4, 2, 2
    alpha = 0.0

    # Create system with 2 stable and 2 unstable eigenvalues
    a = np.diag([-1.0, -2.0, 1.0, 2.0]).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    # STDOM='U' selects unstable (Re > alpha = 0) eigenvalues
    result = tg01pd('C', 'U', 'G', 'I', 'I', n, m, p, 1, n, alpha, a, e, b, c)
    a_out, e_out, b_out, c_out, q, z, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"
    assert ndim == 2, f"Expected 2 unstable eigenvalues, got {ndim}"


def test_tg01pd_schur_form_input():
    """
    Test with input already in Schur form (JOBAE='S').

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import tg01pd

    np.random.seed(202)
    n, m, p = 4, 2, 2

    # Create upper quasi-triangular A (real Schur form)
    a = np.triu(np.random.randn(n, n)).astype(float, order='F')
    a[1, 0] = 0.0  # Ensure true upper triangular for simplicity

    # E must be upper triangular when JOBAE='S'
    e = np.triu(np.random.randn(n, n)).astype(float, order='F')
    # Make E nonsingular
    for i in range(n):
        if abs(e[i, i]) < 0.1:
            e[i, i] = 1.0

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    # For JOBAE='S', NLOW and NSUP define subpencil range
    nlow, nsup = 1, n

    result = tg01pd('C', 'S', 'S', 'I', 'I', n, m, p, nlow, nsup, 0.0, a, e, b, c)
    a_out, e_out, b_out, c_out, q, z, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"


def test_tg01pd_compq_update():
    """
    Test Q update mode (COMPQ='U') with JOBAE='S'.

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import tg01pd

    np.random.seed(303)
    n, m, p = 3, 1, 1

    # Real Schur form inputs
    a = np.triu(np.random.randn(n, n)).astype(float, order='F')
    e = np.triu(np.random.randn(n, n)).astype(float, order='F')
    for i in range(n):
        if abs(e[i, i]) < 0.1:
            e[i, i] = 1.0

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    # Initial orthogonal Q1
    q1 = np.eye(n, order='F', dtype=float)
    z1 = np.eye(n, order='F', dtype=float)

    result = tg01pd('C', 'S', 'S', 'U', 'U', n, m, p, 1, n, 0.0, a, e, b, c, q1, z1)
    a_out, e_out, b_out, c_out, q_out, z_out, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"

    # Q_out should still be orthogonal
    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14)


def test_tg01pd_edge_n0():
    """
    Test edge case: n=0 (quick return).
    """
    from ctrlsys import tg01pd

    n, m, p = 0, 2, 2

    a = np.zeros((0, 0), order='F', dtype=float)
    e = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, m), order='F', dtype=float)
    c = np.zeros((p, 0), order='F', dtype=float)

    result = tg01pd('C', 'S', 'G', 'I', 'I', n, m, p, 0, 0, 0.0, a, e, b, c)
    a_out, e_out, b_out, c_out, q, z, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"
    assert ndim == 0


def test_tg01pd_error_invalid_dico():
    """
    Test error: invalid DICO parameter.
    """
    from ctrlsys import tg01pd

    n, m, p = 3, 1, 1
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    result = tg01pd('X', 'S', 'G', 'I', 'I', n, m, p, 1, n, 0.0, a, e, b, c)
    info = result[-1]

    assert info == -1, f"Expected info=-1 for invalid DICO, got {info}"


def test_tg01pd_error_invalid_stdom():
    """
    Test error: invalid STDOM parameter.
    """
    from ctrlsys import tg01pd

    n, m, p = 3, 1, 1
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    result = tg01pd('C', 'X', 'G', 'I', 'I', n, m, p, 1, n, 0.0, a, e, b, c)
    info = result[-1]

    assert info == -2, f"Expected info=-2 for invalid STDOM, got {info}"


def test_tg01pd_error_invalid_jobae():
    """
    Test error: invalid JOBAE parameter.
    """
    from ctrlsys import tg01pd

    n, m, p = 3, 1, 1
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    result = tg01pd('C', 'S', 'X', 'I', 'I', n, m, p, 1, n, 0.0, a, e, b, c)
    info = result[-1]

    assert info == -3, f"Expected info=-3 for invalid JOBAE, got {info}"


def test_tg01pd_error_negative_alpha_discrete():
    """
    Test error: negative ALPHA for discrete-time system.
    """
    from ctrlsys import tg01pd

    n, m, p = 3, 1, 1
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)

    result = tg01pd('D', 'S', 'G', 'I', 'I', n, m, p, 1, n, -1.0, a, e, b, c)
    info = result[-1]

    assert info == -11, f"Expected info=-11 for negative alpha in discrete-time, got {info}"


def test_tg01pd_transfer_function_preservation():
    """
    Test transfer function preservation under transformation.

    G(s) = C * inv(s*E - A) * B must be preserved.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import tg01pd

    np.random.seed(555)
    n, m, p = 4, 2, 2

    a_orig = np.random.randn(n, n).astype(float, order='F')
    e_orig = np.eye(n, order='F', dtype=float) + 0.1 * np.random.randn(n, n)
    e_orig = e_orig.astype(float, order='F')
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(p, n).astype(float, order='F')

    a = a_orig.copy()
    e = e_orig.copy()
    b = b_orig.copy()
    c = c_orig.copy()

    result = tg01pd('C', 'S', 'G', 'I', 'I', n, m, p, 1, n, 0.0, a, e, b, c)
    a_out, e_out, b_out, c_out, q, z, ndim, alphar, alphai, beta, info = result

    assert info == 0, f"TG01PD returned info={info}"

    # Test transfer function at various frequencies
    test_freqs = [0.1j, 1.0j, 10.0j, 0.5 + 0.5j, -1.0 + 2.0j]

    for s in test_freqs:
        try:
            G_orig = c_orig @ np.linalg.solve(s * e_orig - a_orig, b_orig)
            G_out = c_out @ np.linalg.solve(s * e_out - a_out, b_out)
        except np.linalg.LinAlgError:
            continue

        assert_allclose(G_out, G_orig, rtol=1e-10, atol=1e-12,
                        err_msg=f"Transfer function mismatch at s={s}")
