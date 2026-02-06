"""
Tests for SB03OS - Complex triangular Lyapunov equation solver for Cholesky factor.

Solves for X = op(U)^H * op(U) either:
  Continuous: op(S)^H * X + X * op(S) = -scale^2 * op(R)^H * op(R)
  Discrete:   op(S)^H * X * op(S) - X = -scale^2 * op(R)^H * op(R)

where S and R are complex N-by-N upper triangular matrices.
U is the upper triangular Cholesky factor (overwrites R).
"""
import numpy as np
import pytest
import slicot


def make_stable_continuous_complex(n, seed=42):
    """
    Create an n-by-n complex upper triangular matrix with stable eigenvalues
    (negative real parts on diagonal).

    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    s = np.zeros((n, n), order='F', dtype=np.complex128)

    for i in range(n):
        s[i, i] = complex(-0.5 - np.random.rand(), np.random.randn() * 0.3)

    for i in range(n):
        for j in range(i+1, n):
            s[i, j] = complex(0.3 * np.random.randn(), 0.3 * np.random.randn())

    return s


def make_stable_discrete_complex(n, seed=42):
    """
    Create an n-by-n complex upper triangular matrix with convergent eigenvalues
    (modulus < 1 on diagonal).

    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    s = np.zeros((n, n), order='F', dtype=np.complex128)

    for i in range(n):
        r = 0.3 + np.random.rand() * 0.5
        theta = np.random.rand() * 2 * np.pi
        s[i, i] = r * np.exp(1j * theta)

    for i in range(n):
        for j in range(i+1, n):
            s[i, j] = complex(0.3 * np.random.randn(), 0.3 * np.random.randn())

    return s


def make_upper_triangular_complex(n, seed=42):
    """
    Create an n-by-n complex upper triangular matrix with real non-negative diagonal.

    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    r = np.zeros((n, n), order='F', dtype=np.complex128)

    for i in range(n):
        r[i, i] = abs(np.random.randn()) + 0.1

    for i in range(n):
        for j in range(i+1, n):
            r[i, j] = complex(np.random.randn(), np.random.randn())

    return r


def test_1x1_continuous_basic():
    """
    Test 1x1 continuous-time case.

    For n=1: s^H * x + x * s = -scale^2 * r^H * r
    With s = -2 + 0.5i, r = 1.0:
    (-2 - 0.5i)*x + x*(-2 + 0.5i) = -4*Re(s)*x = 8*x = -scale^2*1
    => x = scale^2/8, u = r/alpha where alpha = sqrt(-2*Re(s)) = 2
    """
    n = 1
    s = np.array([[-2.0 + 0.5j]], order='F', dtype=np.complex128)
    r = np.array([[1.0 + 0j]], order='F', dtype=np.complex128)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(False, False, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = r
    x = u.conj().T @ u

    rhs = -scale**2 * r_orig.conj().T @ r_orig
    residual = s.conj().T @ x + x @ s - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_2x2_continuous():
    """
    Test 2x2 continuous-time with complex upper triangular S.
    """
    n = 2
    s = np.array([
        [-1.0 + 0.3j, 0.5 - 0.2j],
        [0.0 + 0j, -2.0 - 0.4j]
    ], order='F', dtype=np.complex128)
    r = np.array([
        [1.0 + 0j, 0.3 + 0.1j],
        [0.0 + 0j, 0.8 + 0j]
    ], order='F', dtype=np.complex128)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(False, False, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(r)
    x = u.conj().T @ u

    rhs = -scale**2 * r_orig.conj().T @ r_orig
    residual = s.conj().T @ x + x @ s - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_3x3_continuous():
    """
    Test 3x3 continuous-time.

    Random seed: 100 (for reproducibility)
    """
    n = 3
    s = make_stable_continuous_complex(n, seed=100)
    r = make_upper_triangular_complex(n, seed=101)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(False, False, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(r)
    x = u.conj().T @ u

    rhs = -scale**2 * r_orig.conj().T @ r_orig
    residual = s.conj().T @ x + x @ s - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_continuous_transpose():
    """
    Test continuous-time with transpose (LTRANS=True).

    Equation: S*X + X*S^H = -scale^2 * R * R^H, X = U * U^H
    Random seed: 200 (for reproducibility)
    """
    n = 3
    s = make_stable_continuous_complex(n, seed=200)
    r = make_upper_triangular_complex(n, seed=201)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(False, True, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(r)
    x = u @ u.conj().T

    rhs = -scale**2 * r_orig @ r_orig.conj().T
    residual = s @ x + x @ s.conj().T - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_1x1_discrete():
    """
    Test 1x1 discrete-time case.

    s^H * x * s - x = -scale^2 * r^H * r
    """
    n = 1
    s = np.array([[0.5 + 0.3j]], order='F', dtype=np.complex128)
    r = np.array([[1.0 + 0j]], order='F', dtype=np.complex128)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(True, False, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = r
    x = u.conj().T @ u

    rhs = -scale**2 * r_orig.conj().T @ r_orig
    residual = s.conj().T @ x @ s - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_2x2_discrete():
    """
    Test 2x2 discrete-time.
    """
    n = 2
    s = np.array([
        [0.5 + 0.2j, 0.3 - 0.1j],
        [0.0 + 0j, 0.4 - 0.3j]
    ], order='F', dtype=np.complex128)
    r = np.array([
        [1.0 + 0j, 0.2 + 0.3j],
        [0.0 + 0j, 0.7 + 0j]
    ], order='F', dtype=np.complex128)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(True, False, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(r)
    x = u.conj().T @ u

    rhs = -scale**2 * r_orig.conj().T @ r_orig
    residual = s.conj().T @ x @ s - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_3x3_discrete():
    """
    Test 3x3 discrete-time.

    Random seed: 300 (for reproducibility)
    """
    n = 3
    s = make_stable_discrete_complex(n, seed=300)
    r = make_upper_triangular_complex(n, seed=301)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(True, False, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(r)
    x = u.conj().T @ u

    rhs = -scale**2 * r_orig.conj().T @ r_orig
    residual = s.conj().T @ x @ s - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_discrete_transpose():
    """
    Test discrete-time with transpose (LTRANS=True).

    Equation: S*X*S^H - X = -scale^2 * R * R^H, X = U * U^H
    Random seed: 400 (for reproducibility)
    """
    n = 3
    s = make_stable_discrete_complex(n, seed=400)
    r = make_upper_triangular_complex(n, seed=401)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(True, True, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(r)
    x = u @ u.conj().T

    rhs = -scale**2 * r_orig @ r_orig.conj().T
    residual = s @ x @ s.conj().T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_n_zero():
    """Test n=0 returns immediately with success."""
    n = 0
    s = np.zeros((1, 1), order='F', dtype=np.complex128)
    r = np.zeros((1, 1), order='F', dtype=np.complex128)

    s_out, r_out, scale, info = slicot.sb03os(False, False, n, s, r)

    assert info == 0
    assert scale == 1.0


def test_unstable_continuous_returns_info3():
    """
    Test that unstable S (non-negative real eigenvalue) returns info=3.
    """
    n = 1
    s = np.array([[1.0 + 0.2j]], order='F', dtype=np.complex128)
    r = np.array([[1.0 + 0j]], order='F', dtype=np.complex128)

    s_out, r_out, scale, info = slicot.sb03os(False, False, n, s, r)

    assert info == 3


def test_non_convergent_discrete_returns_info3():
    """
    Test that non-convergent S (eigenvalue modulus >= 1) returns info=3.
    """
    n = 1
    s = np.array([[1.5 + 0.3j]], order='F', dtype=np.complex128)
    r = np.array([[1.0 + 0j]], order='F', dtype=np.complex128)

    s_out, r_out, scale, info = slicot.sb03os(True, False, n, s, r)

    assert info == 3


def test_5x5_continuous():
    """
    Test larger 5x5 continuous-time.

    Random seed: 500 (for reproducibility)
    """
    n = 5
    s = make_stable_continuous_complex(n, seed=500)
    r = make_upper_triangular_complex(n, seed=501)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(False, False, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(r)
    x = u.conj().T @ u

    rhs = -scale**2 * r_orig.conj().T @ r_orig
    residual = s.conj().T @ x + x @ s - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_5x5_discrete_transpose():
    """
    Test larger 5x5 discrete-time with transpose.

    Random seed: 600 (for reproducibility)
    """
    n = 5
    s = make_stable_discrete_complex(n, seed=600)
    r = make_upper_triangular_complex(n, seed=601)
    r_orig = r.copy()

    s_out, r_out, scale, info = slicot.sb03os(True, True, n, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = np.triu(r)
    x = u @ u.conj().T

    rhs = -scale**2 * r_orig @ r_orig.conj().T
    residual = s @ x @ s.conj().T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_positive_semidefinite():
    """
    Validate X = U^H * U is positive semi-definite (Hermitian positive).

    Random seed: 700 (for reproducibility)
    """
    n = 4
    s = make_stable_continuous_complex(n, seed=700)
    r = make_upper_triangular_complex(n, seed=701)

    s_out, r_out, scale, info = slicot.sb03os(False, False, n, s, r)

    if info == 0:
        u = np.triu(r)
        x = u.conj().T @ u
        eig = np.linalg.eigvalsh(x)
        assert all(e >= -1e-10 for e in eig), "X not positive semi-definite"


def test_upper_triangular_output():
    """
    Validate output U remains upper triangular with real non-negative diagonal.

    Random seed: 800 (for reproducibility)
    """
    n = 4
    s = make_stable_continuous_complex(n, seed=800)
    r = make_upper_triangular_complex(n, seed=801)

    s_out, r_out, scale, info = slicot.sb03os(False, False, n, s, r)

    if info == 0:
        u = r
        assert np.allclose(np.tril(u, -1), 0, atol=1e-14), "U not upper triangular"
        for i in range(n):
            diag_val = u[i, i]
            assert diag_val.imag == pytest.approx(0.0, abs=1e-14), f"Diagonal {i} not real"
            assert diag_val.real >= -1e-14, f"Diagonal {i} negative"


def test_negative_n_returns_info_minus3():
    """Test that n < 0 returns info = -3."""
    n = -1
    s = np.zeros((1, 1), order='F', dtype=np.complex128)
    r = np.zeros((1, 1), order='F', dtype=np.complex128)

    s_out, r_out, scale, info = slicot.sb03os(False, False, n, s, r)

    assert info == -3
