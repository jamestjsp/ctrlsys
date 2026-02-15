"""
Tests for SB03OY - 2x2 Lyapunov equation solver for Cholesky factor.

SB03OY solves for the Cholesky factor U of X, where op(U)'*op(U) = X, either:
- Continuous-time: op(S)'*X + X*op(S) = -ISGN*scale^2*op(R)'*op(R)
- Discrete-time: op(S)'*X*op(S) - X = -ISGN*scale^2*op(R)'*op(R)

where S is 2x2 with complex conjugate eigenvalues, R is 2x2 upper triangular.
"""
import numpy as np
import ctrlsys


def make_stable_continuous_2x2(alpha, omega):
    """
    Create a 2x2 matrix with complex conjugate eigenvalues alpha +/- i*omega.
    For continuous-time stability: alpha < 0.

    Returns matrix: [[alpha, omega], [-omega, alpha]]
    which has eigenvalues alpha +/- i*omega.
    """
    return np.array([[alpha, omega], [-omega, alpha]], order='F', dtype=np.float64)


def make_stable_discrete_2x2(r, theta):
    """
    Create a 2x2 matrix with complex conjugate eigenvalues r*exp(+/-i*theta).
    For discrete-time stability (convergent): 0 < r < 1.

    Returns matrix: [[r*cos(theta), r*sin(theta)], [-r*sin(theta), r*cos(theta)]]
    which has eigenvalues r*exp(+/-i*theta).
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[r*c, r*s], [-r*s, r*c]], order='F', dtype=np.float64)


def test_sb03oy_continuous_basic():
    """
    Test continuous-time Lyapunov equation with no transpose.

    Solves: S'*X + X*S = -ISGN*scale^2*R'*R for Cholesky factor U where X = U'*U.

    Uses stable matrix S with eigenvalues -1 +/- 2i (negative real parts).
    """
    s = make_stable_continuous_2x2(-1.0, 2.0)
    s_orig = s.copy()
    r = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=np.float64)
    r_orig = r.copy()

    s_out, r_out, a, scale, info = slicot.sb03oy(False, False, 1, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = r
    assert u[1, 0] == 0.0 or abs(u[1, 0]) < 1e-14

    x = u.T @ u

    rhs = -1 * scale**2 * r_orig.T @ r_orig
    residual = s_orig.T @ x + x @ s_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_continuous_transpose():
    """
    Test continuous-time Lyapunov equation with transpose.

    Solves: S*X + X*S' = -ISGN*scale^2*R*R' for Cholesky factor U where X = U*U'.

    Uses stable matrix S with eigenvalues -0.5 +/- 1.5i.
    """
    s = make_stable_continuous_2x2(-0.5, 1.5)
    s_orig = s.copy()
    r = np.array([[2.0, 1.0], [0.0, 1.5]], order='F', dtype=np.float64)
    r_orig = r.copy()

    s_out, r_out, a, scale, info = slicot.sb03oy(False, True, 1, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = r
    assert u[1, 0] == 0.0 or abs(u[1, 0]) < 1e-14

    x = u @ u.T

    rhs = -1 * scale**2 * r_orig @ r_orig.T
    residual = s_orig @ x + x @ s_orig.T - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_discrete_basic():
    """
    Test discrete-time Lyapunov equation with no transpose.

    Solves: S'*X*S - X = -ISGN*scale^2*R'*R for Cholesky factor U where X = U'*U.

    Uses convergent matrix S with eigenvalue modulus 0.8 (< 1).
    """
    s = make_stable_discrete_2x2(0.8, np.pi/4)
    s_orig = s.copy()
    r = np.array([[1.0, 0.3], [0.0, 0.8]], order='F', dtype=np.float64)
    r_orig = r.copy()

    s_out, r_out, a, scale, info = slicot.sb03oy(True, False, 1, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = r
    assert u[1, 0] == 0.0 or abs(u[1, 0]) < 1e-14

    x = u.T @ u

    rhs = -1 * scale**2 * r_orig.T @ r_orig
    residual = s_orig.T @ x @ s_orig - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_discrete_transpose():
    """
    Test discrete-time Lyapunov equation with transpose.

    Solves: S*X*S' - X = -ISGN*scale^2*R*R' for Cholesky factor U where X = U*U'.

    Uses convergent matrix S with eigenvalue modulus 0.6.
    """
    s = make_stable_discrete_2x2(0.6, np.pi/3)
    s_orig = s.copy()
    r = np.array([[1.5, 0.2], [0.0, 1.2]], order='F', dtype=np.float64)
    r_orig = r.copy()

    s_out, r_out, a, scale, info = slicot.sb03oy(True, True, 1, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = r
    assert u[1, 0] == 0.0 or abs(u[1, 0]) < 1e-14

    x = u @ u.T

    rhs = -1 * scale**2 * r_orig @ r_orig.T
    residual = s_orig @ x @ s_orig.T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_isgn_minus_one():
    """
    Test with ISGN=-1 for continuous-time.

    For ISGN=-1 and continuous: -S must be stable (eigenvalues have positive real).
    So we use S with positive real part.
    """
    s = make_stable_continuous_2x2(0.5, 1.0)
    s_orig = s.copy()
    r = np.array([[1.0, 0.4], [0.0, 0.9]], order='F', dtype=np.float64)
    r_orig = r.copy()

    s_out, r_out, a, scale, info = slicot.sb03oy(False, False, -1, s, r)

    assert info == 0
    assert 0 < scale <= 1.0

    u = r
    x = u.T @ u

    rhs = -(-1) * scale**2 * r_orig.T @ r_orig
    residual = s_orig.T @ x + x @ s_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_output_a_upper_triangular():
    """
    Test that output matrix A is upper triangular.
    """
    s = make_stable_continuous_2x2(-1.0, 1.5)
    r = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=np.float64)

    s_out, r_out, a, scale, info = slicot.sb03oy(False, False, 1, s, r)

    assert info == 0
    np.testing.assert_allclose(a[1, 0], 0.0, atol=1e-14)


def test_sb03oy_continuous_unstable_returns_info2():
    """
    Test that unstable S (for ISGN=1) returns info=2.

    For continuous-time with ISGN=1, S must be stable (negative real parts).
    Positive real parts should return info=2.
    """
    s = make_stable_continuous_2x2(0.5, 1.0)
    r = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)

    s_out, r_out, a, scale, info = slicot.sb03oy(False, False, 1, s, r)

    assert info == 2


def test_sb03oy_real_eigenvalues_returns_info4():
    """
    Test that S with real eigenvalues returns info=4.

    S must have complex conjugate eigenvalues for this routine.
    """
    s = np.array([[1.0, 0.0], [0.0, 2.0]], order='F', dtype=np.float64)
    r = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)

    s_out, r_out, a, scale, info = slicot.sb03oy(False, False, 1, s, r)

    assert info == 4


def test_sb03oy_discrete_not_convergent_returns_info2():
    """
    Test that discrete-time with non-convergent S (with ISGN=1) returns info=2.

    For discrete-time with ISGN=1, eigenvalue moduli must be < 1.
    """
    s = make_stable_discrete_2x2(1.2, np.pi/4)
    r = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)

    s_out, r_out, a, scale, info = slicot.sb03oy(True, False, 1, s, r)

    assert info == 2


def test_sb03oy_b_u_relation_no_transpose():
    """
    Test the B*U = U*S relation for LTRANS=False.

    After solving, S contains B such that B*U = U*S (original S).
    """
    s = make_stable_continuous_2x2(-1.0, 2.0)
    s_orig = s.copy()
    r = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=np.float64)

    s_out, r_out, a, scale, info = slicot.sb03oy(False, False, 1, s, r)

    assert info == 0

    b = s
    u = r

    lhs = b @ u
    rhs = u @ s_orig
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_sb03oy_a_u_relation_no_transpose():
    """
    Test the A*U = scale^2*R relation for LTRANS=False.

    After solving, A satisfies A*U/scale = scale*R.
    """
    s = make_stable_continuous_2x2(-1.0, 2.0)
    r = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=np.float64)
    r_orig = r.copy()

    s_out, r_out, a, scale, info = slicot.sb03oy(False, False, 1, s, r)

    assert info == 0

    u = r

    lhs = a @ u
    rhs = scale**2 * r_orig
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)
