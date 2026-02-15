"""
Tests for tg01wd - Reduction of descriptor system to generalized Schur form.

TG01WD reduces the pair (A,E) to a real generalized Schur form using an
orthogonal equivalence transformation (A,E) <-- (Q'*A*Z, Q'*E*Z) and applies
the transformation to B and C: B <-- Q'*B and C <-- C*Z.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tg01wd_basic():
    """
    Test basic functionality with a simple descriptor system.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tg01wd

    np.random.seed(42)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float) + 0.1 * np.random.randn(n, n)
    e = e.astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_out, e_out, b_out, c_out, q, z, alphar, alphai, beta, info = tg01wd(
        n, m, p, a, e, b, c
    )

    assert info == 0, f"TG01WD returned info={info}"

    # E should be upper triangular
    for j in range(n):
        for i in range(j + 1, n):
            assert abs(e_out[i, j]) < 1e-14, f"E[{i},{j}] = {e_out[i, j]} should be 0"

    # A should be upper quasi-triangular (elements below first subdiagonal = 0)
    for j in range(n):
        for i in range(j + 2, n):
            assert abs(a_out[i, j]) < 1e-14, f"A[{i},{j}] = {a_out[i, j]} should be 0"


def test_tg01wd_orthogonality():
    """
    Test orthogonality of Q and Z matrices.

    Mathematical property: Q'*Q = I and Z'*Z = I

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import tg01wd

    np.random.seed(123)
    n, m, p = 5, 2, 3

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float) + 0.2 * np.random.randn(n, n)
    e = e.astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_out, e_out, b_out, c_out, q, z, alphar, alphai, beta, info = tg01wd(
        n, m, p, a, e, b, c
    )

    assert info == 0, f"TG01WD returned info={info}"

    # Q'*Q = I
    qtq = q.T @ q
    assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14,
                    err_msg="Q is not orthogonal: Q'*Q != I")

    # Z'*Z = I
    ztz = z.T @ z
    assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14,
                    err_msg="Z is not orthogonal: Z'*Z != I")


def test_tg01wd_equivalence_transformation():
    """
    Test equivalence transformation property.

    Mathematical property:
    - Q'*A_orig*Z = A_out
    - Q'*E_orig*Z = E_out
    - Q'*B_orig = B_out
    - C_orig*Z = C_out

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tg01wd

    np.random.seed(456)
    n, m, p = 4, 2, 2

    a_orig = np.random.randn(n, n).astype(float, order='F')
    e_orig = np.eye(n, order='F', dtype=float) + 0.15 * np.random.randn(n, n)
    e_orig = e_orig.astype(float, order='F')
    b_orig = np.random.randn(n, m).astype(float, order='F')
    c_orig = np.random.randn(p, n).astype(float, order='F')

    a = a_orig.copy()
    e = e_orig.copy()
    b = b_orig.copy()
    c = c_orig.copy()

    a_out, e_out, b_out, c_out, q, z, alphar, alphai, beta, info = tg01wd(
        n, m, p, a, e, b, c
    )

    assert info == 0, f"TG01WD returned info={info}"

    # Verify transformations
    assert_allclose(q.T @ a_orig @ z, a_out, rtol=1e-13, atol=1e-14,
                    err_msg="Q'*A*Z != A_out")
    assert_allclose(q.T @ e_orig @ z, e_out, rtol=1e-13, atol=1e-14,
                    err_msg="Q'*E*Z != E_out")
    assert_allclose(q.T @ b_orig, b_out, rtol=1e-13, atol=1e-14,
                    err_msg="Q'*B != B_out")
    assert_allclose(c_orig @ z, c_out, rtol=1e-13, atol=1e-14,
                    err_msg="C*Z != C_out")


def test_tg01wd_eigenvalue_preservation():
    """
    Test eigenvalue preservation under transformation.

    Mathematical property: eigenvalues of (A,E) = eigenvalues of (Q'AZ, Q'EZ)

    Generalized eigenvalues are (ALPHAR + i*ALPHAI) / BETA.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tg01wd

    np.random.seed(789)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float) + 0.1 * np.random.randn(n, n)
    e = e.astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    # Compute original generalized eigenvalues using scipy
    from numpy.linalg import eigvals, solve
    try:
        eig_orig = eigvals(solve(e, a))
        eig_orig = np.sort_complex(eig_orig)
    except np.linalg.LinAlgError:
        pytest.skip("Original E is singular")

    a_out, e_out, b_out, c_out, q, z, alphar, alphai, beta, info = tg01wd(
        n, m, p, a, e, b, c
    )

    assert info == 0, f"TG01WD returned info={info}"

    # Compute eigenvalues from ALPHAR, ALPHAI, BETA
    eig_returned = np.zeros(n, dtype=complex)
    for j in range(n):
        if abs(beta[j]) > 1e-15:
            eig_returned[j] = complex(alphar[j], alphai[j]) / beta[j]
        else:
            eig_returned[j] = complex(np.inf, 0)
    eig_returned = np.sort_complex(eig_returned)

    # Filter out infinities for comparison
    finite_orig = eig_orig[np.isfinite(eig_orig)]
    finite_ret = eig_returned[np.isfinite(eig_returned)]

    # Sort by real part first, then by absolute imaginary part
    def sort_key(eig):
        return (eig.real, abs(eig.imag))

    sorted_orig = sorted(finite_orig, key=sort_key)
    sorted_ret = sorted(finite_ret, key=sort_key)

    for i in range(len(sorted_orig)):
        # Compare real parts
        assert abs(sorted_ret[i].real - sorted_orig[i].real) < 1e-10, \
            f"Real part mismatch at index {i}"
        # Compare imaginary magnitudes (conjugate pairs are equivalent)
        assert abs(abs(sorted_ret[i].imag) - abs(sorted_orig[i].imag)) < 1e-10, \
            f"Imaginary magnitude mismatch at index {i}"


def test_tg01wd_transfer_function_preservation():
    """
    Test transfer function preservation under transformation.

    G(s) = C * inv(s*E - A) * B must be preserved.

    Random seed: 101 (for reproducibility)
    """
    from ctrlsys import tg01wd

    np.random.seed(101)
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

    a_out, e_out, b_out, c_out, q, z, alphar, alphai, beta, info = tg01wd(
        n, m, p, a, e, b, c
    )

    assert info == 0, f"TG01WD returned info={info}"

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


def test_tg01wd_edge_n0():
    """
    Test edge case: n=0 (quick return).
    """
    from ctrlsys import tg01wd

    n, m, p = 0, 2, 2

    a = np.zeros((0, 0), order='F', dtype=float)
    e = np.zeros((0, 0), order='F', dtype=float)
    b = np.zeros((0, m), order='F', dtype=float)
    c = np.zeros((p, 0), order='F', dtype=float)

    a_out, e_out, b_out, c_out, q, z, alphar, alphai, beta, info = tg01wd(
        n, m, p, a, e, b, c
    )

    assert info == 0, f"TG01WD returned info={info}"
    assert a_out.shape == (0, 0)
    assert e_out.shape == (0, 0)
    assert b_out.shape == (0, m)
    assert c_out.shape == (p, 0)


def test_tg01wd_edge_m0():
    """
    Test edge case: m=0 (no inputs).

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import tg01wd

    np.random.seed(202)
    n, m, p = 3, 0, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float)
    b = np.zeros((n, 0), order='F', dtype=float)
    c = np.random.randn(p, n).astype(float, order='F')

    a_out, e_out, b_out, c_out, q, z, alphar, alphai, beta, info = tg01wd(
        n, m, p, a, e, b, c
    )

    assert info == 0, f"TG01WD returned info={info}"
    assert b_out.shape == (n, 0)


def test_tg01wd_edge_p0():
    """
    Test edge case: p=0 (no outputs).

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import tg01wd

    np.random.seed(303)
    n, m, p = 3, 2, 0

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.zeros((0, n), order='F', dtype=float)

    a_out, e_out, b_out, c_out, q, z, alphar, alphai, beta, info = tg01wd(
        n, m, p, a, e, b, c
    )

    assert info == 0, f"TG01WD returned info={info}"
    assert c_out.shape == (0, n)


def test_tg01wd_identity_e():
    """
    Test with E = identity matrix (standard state-space).

    Random seed: 404 (for reproducibility)
    """
    from ctrlsys import tg01wd

    np.random.seed(404)
    n, m, p = 4, 2, 2

    a = np.random.randn(n, n).astype(float, order='F')
    e = np.eye(n, order='F', dtype=float)
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    a_out, e_out, b_out, c_out, q, z, alphar, alphai, beta, info = tg01wd(
        n, m, p, a, e, b, c
    )

    assert info == 0, f"TG01WD returned info={info}"

    # E should be upper triangular
    for j in range(n):
        for i in range(j + 1, n):
            assert abs(e_out[i, j]) < 1e-14

    # BETA should be non-zero since E=I is nonsingular
    for j in range(n):
        assert abs(beta[j]) > 1e-10, f"BETA[{j}] should be non-zero"


def test_tg01wd_error_negative_n():
    """
    Test error: negative N parameter.
    """
    from ctrlsys import tg01wd

    n, m, p = -1, 2, 2

    a = np.zeros((1, 1), order='F', dtype=float)
    e = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, m), order='F', dtype=float)
    c = np.zeros((p, 1), order='F', dtype=float)

    result = tg01wd(n, m, p, a, e, b, c)
    info = result[-1]

    assert info == -1, f"Expected info=-1 for negative N, got {info}"


def test_tg01wd_error_negative_m():
    """
    Test error: negative M parameter.
    """
    from ctrlsys import tg01wd

    n, m, p = 2, -1, 2

    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.zeros((n, 1), order='F', dtype=float)
    c = np.zeros((p, n), order='F', dtype=float)

    result = tg01wd(n, m, p, a, e, b, c)
    info = result[-1]

    assert info == -2, f"Expected info=-2 for negative M, got {info}"


def test_tg01wd_error_negative_p():
    """
    Test error: negative P parameter.
    """
    from ctrlsys import tg01wd

    n, m, p = 2, 2, -1

    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)
    c = np.zeros((1, n), order='F', dtype=float)

    result = tg01wd(n, m, p, a, e, b, c)
    info = result[-1]

    assert info == -3, f"Expected info=-3 for negative P, got {info}"
