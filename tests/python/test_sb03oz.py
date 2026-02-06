"""
Tests for SB03OZ - Complex Lyapunov equation solver computing Cholesky factor.

Solves for X = op(U)^H * op(U) either the stable continuous-time Lyapunov equation:
    op(A)^H * X + X * op(A) = -scale^2 * op(B)^H * op(B)
or the convergent discrete-time Lyapunov equation:
    op(A)^H * X * op(A) - X = -scale^2 * op(B)^H * op(B)

where A is N-by-N complex, op(B) is M-by-N complex, U is upper triangular Cholesky factor.
"""
import numpy as np
import slicot


def make_stable_continuous_complex(n, seed=42):
    """
    Create an n-by-n complex matrix with stable eigenvalues (negative real parts).

    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    u, _ = np.linalg.qr(np.random.randn(n, n) + 1j*np.random.randn(n, n))
    eig = -np.random.rand(n) - 0.5 + 1j * np.random.randn(n) * 0.3
    a = u @ np.diag(eig) @ u.conj().T
    return np.asfortranarray(a)


def make_stable_discrete_complex(n, seed=42):
    """
    Create an n-by-n complex matrix with convergent eigenvalues (modulus < 1).

    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    u, _ = np.linalg.qr(np.random.randn(n, n) + 1j*np.random.randn(n, n))
    r = 0.3 + np.random.rand(n) * 0.5
    theta = np.random.rand(n) * 2 * np.pi
    eig = r * np.exp(1j * theta)
    a = u @ np.diag(eig) @ u.conj().T
    return np.asfortranarray(a)


def test_continuous_nofact_notrans():
    """
    Test continuous-time, compute Schur factorization, no transpose.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m = 3, 4

    a = make_stable_continuous_complex(n, seed=42)
    a_orig = a.copy()

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='N', trans='N',
                                   a=a, q=q, b=b_padded)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u.conj().T @ u

    rhs = -scale**2 * b_orig.conj().T @ b_orig
    residual = a_orig.conj().T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_continuous_transpose():
    """
    Test continuous-time with transpose (TRANS='C').

    Verifies routine completes successfully and X is positive semi-definite.
    Random seed: 100 (for reproducibility)
    """
    np.random.seed(100)
    n = 3

    a = make_stable_continuous_complex(n, seed=100)

    b = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    b = np.asfortranarray(b)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='N', trans='C',
                                   a=a, q=q, b=b)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(b_out[:n, :n])
    x = u @ u.conj().T

    eig = np.linalg.eigvalsh(x)
    assert all(e >= -1e-10 for e in eig), "X not positive semi-definite"


def test_discrete_nofact_notrans():
    """
    Test discrete-time, compute Schur factorization, no transpose.

    Equation: A^H * X * A - X = -scale^2 * B^H * B, X = U^H * U
    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n, m = 3, 4

    a = make_stable_discrete_complex(n, seed=200)
    a_orig = a.copy()

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='D', fact='N', trans='N',
                                   a=a, q=q, b=b_padded)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u.conj().T @ u

    rhs = -scale**2 * b_orig.conj().T @ b_orig
    residual = a_orig.conj().T @ x @ a_orig - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_discrete_transpose():
    """
    Test discrete-time with transpose (TRANS='C').

    Verifies routine completes successfully and X is positive semi-definite.
    Random seed: 300 (for reproducibility)
    """
    np.random.seed(300)
    n = 3

    a = make_stable_discrete_complex(n, seed=300)

    b = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    b = np.asfortranarray(b)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='D', fact='N', trans='C',
                                   a=a, q=q, b=b)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(b_out[:n, :n])
    x = u @ u.conj().T

    eig = np.linalg.eigvalsh(x)
    assert all(e >= -1e-10 for e in eig), "X not positive semi-definite"


def test_schur_provided():
    """
    Test with Schur factorization already provided (FACT='F').

    Random seed: 400 (for reproducibility)
    """
    np.random.seed(400)
    n, m = 3, 4

    s = np.array([
        [-1.0 + 0.2j, 0.3 - 0.1j, 0.2 + 0.1j],
        [0.0 + 0j, -1.5 + 0.3j, 0.4 - 0.2j],
        [0.0 + 0j, 0.0 + 0j, -2.0 - 0.5j]
    ], order='F', dtype=np.complex128)

    q = np.eye(n, order='F', dtype=np.complex128)

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    a_orig = q @ s @ q.conj().T

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='F', trans='N',
                                   a=s, q=q, b=b_padded)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u.conj().T @ u

    rhs = -scale**2 * b_orig.conj().T @ b_orig
    residual = a_orig.conj().T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_zero_rhs():
    """Test with zero RHS: U should be nearly zero."""
    n = 3

    a = np.array([
        [-1.0 + 0.2j, 0.3 - 0.1j, 0.2 + 0.1j],
        [0.0 + 0j, -1.5 + 0.3j, 0.4 - 0.2j],
        [0.0 + 0j, 0.0 + 0j, -2.0 - 0.5j]
    ], order='F', dtype=np.complex128)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    b = np.zeros((n, n), order='F', dtype=np.complex128)

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='N', trans='N',
                                   a=a, q=q, b=b)

    assert info == 0
    np.testing.assert_allclose(np.triu(b_out[:n, :n]), 0.0, atol=1e-14)


def test_n_zero():
    """Test N=0: quick return."""
    n, m = 0, 3

    a = np.zeros((1, 1), order='F', dtype=np.complex128)
    q = np.zeros((1, 1), order='F', dtype=np.complex128)
    b = np.zeros((m, 1), order='F', dtype=np.complex128)

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='N', trans='N',
                                   a=a, q=q, b=b)

    assert info == 0


def test_unstable_continuous():
    """Test unstable A (non-negative real eigenvalue) returns info=2."""
    n, m = 2, 2

    a = np.array([
        [1.0 + 0.2j, 0.0 + 0j],
        [0.0 + 0j, -1.0 + 0.3j]
    ], order='F', dtype=np.complex128)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    b = np.array([
        [1.0 + 0j, 0.0 + 0j],
        [0.0 + 0j, 1.0 + 0j]
    ], order='F', dtype=np.complex128)

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='N', trans='N',
                                   a=a, q=q, b=b)

    assert info == 2


def test_non_convergent_discrete():
    """Test non-convergent A (eigenvalue modulus >= 1) returns info=2."""
    n, m = 2, 2

    a = np.array([
        [1.5 + 0.3j, 0.0 + 0j],
        [0.0 + 0j, 0.5 + 0.2j]
    ], order='F', dtype=np.complex128)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    b = np.array([
        [1.0 + 0j, 0.0 + 0j],
        [0.0 + 0j, 1.0 + 0j]
    ], order='F', dtype=np.complex128)

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='D', fact='N', trans='N',
                                   a=a, q=q, b=b)

    assert info == 2


def test_invalid_dico():
    """Test invalid DICO parameter."""
    n, m = 2, 2

    a = np.array([[-1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], order='F', dtype=np.complex128)
    q = np.zeros((n, n), order='F', dtype=np.complex128)
    b = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], order='F', dtype=np.complex128)

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='X', fact='N', trans='N',
                                   a=a, q=q, b=b)

    assert info == -1


def test_invalid_fact():
    """Test invalid FACT parameter returns error."""
    n = 2

    a = np.array([[-1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], order='F', dtype=np.complex128)
    q = np.zeros((n, n), order='F', dtype=np.complex128)
    b = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], order='F', dtype=np.complex128)

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='X', trans='N',
                                   a=a, q=q, b=b)

    assert info == -2


def test_5x5_continuous():
    """
    Test larger 5x5 continuous-time.

    Random seed: 500 (for reproducibility)
    """
    np.random.seed(500)
    n, m = 5, 7

    a = make_stable_continuous_complex(n, seed=500)
    a_orig = a.copy()

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='N', trans='N',
                                   a=a, q=q, b=b_padded)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u.conj().T @ u

    rhs = -scale**2 * b_orig.conj().T @ b_orig
    residual = a_orig.conj().T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_positive_semidefinite():
    """
    Validate X = U^H * U is positive semi-definite (Hermitian positive).

    Random seed: 600 (for reproducibility)
    """
    np.random.seed(600)
    n, m = 4, 5

    a = make_stable_continuous_complex(n, seed=600)

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='N', trans='N',
                                   a=a, q=q, b=b_padded)

    if info == 0:
        u = np.triu(b_padded[:n, :n])
        x = u.conj().T @ u
        eig = np.linalg.eigvalsh(x)
        assert all(e >= -1e-10 for e in eig), "X not positive semi-definite"


def test_upper_triangular_output():
    """
    Validate output U remains upper triangular with real non-negative diagonal.

    Random seed: 700 (for reproducibility)
    """
    np.random.seed(700)
    n, m = 4, 5

    a = make_stable_continuous_complex(n, seed=700)

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='N', trans='N',
                                   a=a, q=q, b=b_padded)

    if info == 0:
        u = b_padded[:n, :n]
        assert np.allclose(np.tril(u, -1), 0, atol=1e-14), "U not upper triangular"
        for i in range(n):
            diag_val = u[i, i]
            assert abs(diag_val.imag) < 1e-14, f"Diagonal {i} not real"
            assert diag_val.real >= -1e-14, f"Diagonal {i} negative"


def test_eigenvalue_check():
    """
    Validate eigenvalues W returned match eigenvalues of A.

    Random seed: 800 (for reproducibility)
    """
    np.random.seed(800)
    n, m = 4, 5

    a = make_stable_continuous_complex(n, seed=800)
    a_orig = a.copy()

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    a_out, q_out, b_out, scale, w, info = slicot.sb03oz(dico='C', fact='N', trans='N',
                                   a=a, q=q, b=b_padded)

    if info == 0:
        expected_eig = np.linalg.eigvals(a_orig)
        np.testing.assert_allclose(sorted(w.real), sorted(expected_eig.real), rtol=1e-10)
        np.testing.assert_allclose(sorted(w.imag), sorted(expected_eig.imag), rtol=1e-10)
