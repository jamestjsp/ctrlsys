import numpy as np
import pytest


def test_mb04zd_html_example():
    """
    Test MB04ZD using HTML documentation example.

    From MB04ZD.html example:
    N=3, COMPU='N'

    Hamiltonian matrix H = [[A, G], [Q, -A^T]]
    Square-reduced form: Q'A' - A'^T Q' = 0
    """
    n = 3
    compu = 'N'

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=float, order='F')

    g_upper = np.array([
        [1.0, 1.0, 1.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    q_lower = np.array([
        [7.0, 0.0, 0.0],
        [6.0, 8.0, 0.0],
        [5.0, 4.0, 9.0]
    ], dtype=float, order='F')

    qg = np.zeros((n, n+1), dtype=float, order='F')
    for i in range(n):
        for j in range(i+1):
            qg[i, j] = q_lower[i, j]
    for i in range(n):
        for j in range(i, n):
            qg[i, j+1] = g_upper[i, j]

    a_expected = np.array([
        [1.0000,  3.3485,  0.3436],
        [6.7566, 11.0750, -0.3014],
        [2.3478,  1.6899, -2.3868]
    ], dtype=float, order='F')

    g_expected_upper = np.array([
        [1.0000, 1.9126, -0.1072],
        [0.0,    8.4479, -1.0790],
        [0.0,    0.0,    -2.9871]
    ], dtype=float, order='F')

    q_expected_lower = np.array([
        [ 7.0000,  0.0,     0.0],
        [ 8.6275, 16.2238,  0.0],
        [-0.6352, -0.1403,  1.2371]
    ], dtype=float, order='F')

    from ctrlsys import mb04zd

    a_out, qg_out, u_out, info = mb04zd(compu, n, a, qg)

    assert info == 0
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)

    for i in range(n):
        for j in range(i+1):
            np.testing.assert_allclose(qg_out[i, j], q_expected_lower[i, j],
                                       rtol=1e-3, atol=1e-4)
    for i in range(n):
        for j in range(i, n):
            np.testing.assert_allclose(qg_out[i, j+1], g_expected_upper[i, j],
                                       rtol=1e-3, atol=1e-4)


def test_mb04zd_with_transform():
    """
    Test MB04ZD with COMPU='I' to compute transformation matrix.

    Validates U is orthogonal symplectic: U*U^T = I (first N rows).
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n = 4
    compu = 'I'

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n+1), dtype=float, order='F')
    for i in range(n):
        for j in range(i+1):
            qg[i, j] = np.random.randn()
            if i != j:
                qg[j, i] = qg[i, j]
    for i in range(n):
        for j in range(i, n):
            qg[i, j+1] = np.random.randn()
            if i != j:
                qg[j, i+1] = qg[i, j+1]

    from ctrlsys import mb04zd

    a_out, qg_out, u_out, info = mb04zd(compu, n, a, qg)

    assert info == 0
    assert u_out.shape == (n, 2*n)

    u1 = u_out[:, :n]
    u2 = u_out[:, n:]
    identity_check = u1 @ u1.T + u2 @ u2.T
    np.testing.assert_allclose(identity_check, np.eye(n), rtol=1e-13, atol=1e-14)

    symplectic_check = u1 @ u2.T - u2 @ u1.T
    np.testing.assert_allclose(symplectic_check, np.zeros((n, n)),
                               rtol=1e-13, atol=1e-14)


def test_mb04zd_square_reduced_property():
    """
    Test MB04ZD square-reduced property: Q'*A' - A'^T*Q' = 0.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n = 5
    compu = 'N'

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n+1), dtype=float, order='F')
    q_sym = np.random.randn(n, n)
    q_sym = (q_sym + q_sym.T) / 2
    for i in range(n):
        for j in range(i+1):
            qg[i, j] = q_sym[i, j]
    g_sym = np.random.randn(n, n)
    g_sym = (g_sym + g_sym.T) / 2
    for i in range(n):
        for j in range(i, n):
            qg[i, j+1] = g_sym[i, j]

    from ctrlsys import mb04zd

    a_out, qg_out, u_out, info = mb04zd(compu, n, a, qg)

    assert info == 0

    q_prime = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1):
            q_prime[i, j] = qg_out[i, j]
            q_prime[j, i] = qg_out[i, j]

    a_prime = a_out

    residual = q_prime @ a_prime - a_prime.T @ q_prime
    np.testing.assert_allclose(residual, np.zeros((n, n)), rtol=1e-11, atol=1e-12)


def test_mb04zd_accumulate_transform():
    """
    Test MB04ZD with COMPU='A' (accumulate transformation).

    Start with identity symplectic S, accumulate U, verify result.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n = 3
    compu = 'A'

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n+1), dtype=float, order='F')
    q_sym = np.random.randn(n, n)
    q_sym = (q_sym + q_sym.T) / 2
    for i in range(n):
        for j in range(i+1):
            qg[i, j] = q_sym[i, j]
    g_sym = np.random.randn(n, n)
    g_sym = (g_sym + g_sym.T) / 2
    for i in range(n):
        for j in range(i, n):
            qg[i, j+1] = g_sym[i, j]

    u_in = np.zeros((n, 2*n), dtype=float, order='F')
    u_in[:, :n] = np.eye(n)

    from ctrlsys import mb04zd

    a_out, qg_out, u_out, info = mb04zd(compu, n, a, qg, u_in)

    assert info == 0
    assert u_out.shape == (n, 2*n)

    u1 = u_out[:, :n]
    u2 = u_out[:, n:]
    identity_check = u1 @ u1.T + u2 @ u2.T
    np.testing.assert_allclose(identity_check, np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb04zd_similarity_transformation():
    """
    Test MB04ZD similarity: H' = U^T H U (eigenvalue preservation).

    Verify eigenvalues of square of Hamiltonian are preserved.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n = 4
    compu = 'I'

    a = np.random.randn(n, n).astype(float, order='F')
    qg = np.zeros((n, n+1), dtype=float, order='F')
    q_sym = np.random.randn(n, n)
    q_sym = (q_sym + q_sym.T) / 2
    for i in range(n):
        for j in range(i+1):
            qg[i, j] = q_sym[i, j]
    g_sym = np.random.randn(n, n)
    g_sym = (g_sym + g_sym.T) / 2
    for i in range(n):
        for j in range(i, n):
            qg[i, j+1] = g_sym[i, j]

    q_mat = np.zeros((n, n), dtype=float)
    g_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1):
            q_mat[i, j] = qg[i, j]
            q_mat[j, i] = qg[i, j]
    for i in range(n):
        for j in range(i, n):
            g_mat[i, j] = qg[i, j+1]
            g_mat[j, i] = qg[i, j+1]

    h_orig = np.zeros((2*n, 2*n), dtype=float)
    h_orig[:n, :n] = a
    h_orig[:n, n:] = g_mat
    h_orig[n:, :n] = q_mat
    h_orig[n:, n:] = -a.T

    h2_orig = h_orig @ h_orig
    eig_orig = np.sort(np.linalg.eigvals(h2_orig).real)

    from ctrlsys import mb04zd

    a_copy = a.copy(order='F')
    qg_copy = qg.copy(order='F')
    a_out, qg_out, u_out, info = mb04zd(compu, n, a_copy, qg_copy)

    assert info == 0

    q_prime = np.zeros((n, n), dtype=float)
    g_prime = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1):
            q_prime[i, j] = qg_out[i, j]
            q_prime[j, i] = qg_out[i, j]
    for i in range(n):
        for j in range(i, n):
            g_prime[i, j] = qg_out[i, j+1]
            g_prime[j, i] = qg_out[i, j+1]

    h_prime = np.zeros((2*n, 2*n), dtype=float)
    h_prime[:n, :n] = a_out
    h_prime[:n, n:] = g_prime
    h_prime[n:, :n] = q_prime
    h_prime[n:, n:] = -a_out.T

    h2_prime = h_prime @ h_prime
    eig_prime = np.sort(np.linalg.eigvals(h2_prime).real)

    np.testing.assert_allclose(eig_prime, eig_orig, rtol=1e-10, atol=1e-11)


def test_mb04zd_n_zero():
    """
    Test MB04ZD with N=0 (quick return).
    """
    n = 0
    compu = 'N'

    a = np.zeros((1, 1), dtype=float, order='F')
    qg = np.zeros((1, 2), dtype=float, order='F')

    from ctrlsys import mb04zd

    a_out, qg_out, u_out, info = mb04zd(compu, n, a, qg)

    assert info == 0


def test_mb04zd_invalid_compu():
    """
    Test MB04ZD with invalid COMPU parameter.
    """
    n = 3
    compu = 'X'

    a = np.eye(n, dtype=float, order='F')
    qg = np.zeros((n, n+1), dtype=float, order='F')

    from ctrlsys import mb04zd

    a_out, qg_out, u_out, info = mb04zd(compu, n, a, qg)

    assert info == -1


def test_mb04zd_n_one():
    """
    Test MB04ZD with N=1 (edge case, no loop iterations).
    """
    n = 1
    compu = 'I'

    a = np.array([[2.0]], dtype=float, order='F')
    qg = np.array([[1.0, 3.0]], dtype=float, order='F')

    from ctrlsys import mb04zd

    a_out, qg_out, u_out, info = mb04zd(compu, n, a, qg)

    assert info == 0
    np.testing.assert_allclose(a_out, a, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(qg_out, qg, rtol=1e-14, atol=1e-15)
    assert u_out.shape == (1, 2)
    np.testing.assert_allclose(u_out[0, 0], 1.0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(u_out[0, 1], 0.0, rtol=1e-14, atol=1e-15)
