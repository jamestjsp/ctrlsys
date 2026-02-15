import numpy as np
import pytest


def test_mb04xy_basic():
    """
    Test MB04XY with identity-like Householder transformations.

    When taup and tauq contain zeros, no Householder reflectors are applied
    (since taup[i]=0 means identity transformation).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n = 4, 3
    p = min(m, n)

    x = np.random.randn(m, n).astype(float, order='F')
    taup = np.zeros(p, dtype=float)
    tauq = np.zeros(p, dtype=float)

    u = np.eye(m, dtype=float, order='F')
    v = np.eye(n, dtype=float, order='F')
    u_orig = u.copy()
    v_orig = v.copy()
    inul = np.ones(max(m, n), dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('A', 'A', m, n, x, taup, tauq, u, v, inul)

    assert info == 0
    np.testing.assert_allclose(u_out, u_orig, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out, v_orig, rtol=1e-14, atol=1e-14)


def test_mb04xy_selective_columns():
    """
    Test MB04XY with selective column transformation via INUL.

    Only columns where INUL[i] = True should be transformed.
    With taup=0, no transformation happens regardless of INUL.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, n = 5, 4
    p = min(m, n)

    x = np.random.randn(m, n).astype(float, order='F')
    taup = np.zeros(p, dtype=float)
    tauq = np.zeros(p, dtype=float)

    u = np.random.randn(m, m).astype(float, order='F')
    u_orig = u.copy()

    inul = np.array([True, False, True, False, False], dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('A', 'N', m, n, x, taup, tauq, u, None, inul)

    assert info == 0
    np.testing.assert_allclose(u_out, u_orig, rtol=1e-14, atol=1e-14)


def test_mb04xy_jobu_s():
    """
    Test MB04XY with JOBU='S' (U has min(M,N) columns).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, n = 5, 3
    p = min(m, n)

    x = np.random.randn(m, n).astype(float, order='F')
    taup = np.zeros(p, dtype=float)
    tauq = np.zeros(p, dtype=float)

    u = np.eye(m, p, dtype=float, order='F')
    u_orig = u.copy()
    inul = np.ones(max(m, n), dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('S', 'N', m, n, x, taup, tauq, u, None, inul)

    assert info == 0
    assert u_out.shape == (m, p)
    np.testing.assert_allclose(u_out, u_orig, rtol=1e-14, atol=1e-14)


def test_mb04xy_jobv_s():
    """
    Test MB04XY with JOBV='S' (V has min(M,N) columns).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m, n = 3, 5
    p = min(m, n)

    x = np.random.randn(m, n).astype(float, order='F')
    taup = np.zeros(p, dtype=float)
    tauq = np.zeros(p, dtype=float)

    v = np.eye(n, p, dtype=float, order='F')
    v_orig = v.copy()
    inul = np.ones(max(m, n), dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('N', 'S', m, n, x, taup, tauq, None, v, inul)

    assert info == 0
    assert v_out.shape == (n, p)
    np.testing.assert_allclose(v_out, v_orig, rtol=1e-14, atol=1e-14)


def test_mb04xy_no_transform():
    """
    Test MB04XY with JOBU='N' and JOBV='N' (no transformation).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m, n = 4, 4

    x = np.random.randn(m, n).astype(float, order='F')
    taup = np.zeros(min(m, n), dtype=float)
    tauq = np.zeros(min(m, n), dtype=float)

    inul = np.ones(max(m, n), dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('N', 'N', m, n, x, taup, tauq, None, None, inul)

    assert info == 0


def test_mb04xy_wide_matrix():
    """
    Test MB04XY with wide matrix (M < N).

    For M < N, the offset IOFF = 1 is used in the Fortran code.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    m, n = 3, 5
    p = min(m, n)

    x = np.random.randn(m, n).astype(float, order='F')
    taup = np.zeros(p, dtype=float)
    tauq = np.zeros(p, dtype=float)

    u = np.eye(m, dtype=float, order='F')
    v = np.eye(n, dtype=float, order='F')
    u_orig = u.copy()
    v_orig = v.copy()
    inul = np.ones(max(m, n), dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('A', 'A', m, n, x, taup, tauq, u, v, inul)

    assert info == 0
    np.testing.assert_allclose(u_out, u_orig, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out, v_orig, rtol=1e-14, atol=1e-14)


def test_mb04xy_tall_matrix():
    """
    Test MB04XY with tall matrix (M > N).

    For M >= N, the offset IOFF = 0 is used in the Fortran code.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    m, n = 5, 3
    p = min(m, n)

    x = np.random.randn(m, n).astype(float, order='F')
    taup = np.zeros(p, dtype=float)
    tauq = np.zeros(p, dtype=float)

    u = np.eye(m, dtype=float, order='F')
    v = np.eye(n, dtype=float, order='F')
    u_orig = u.copy()
    v_orig = v.copy()
    inul = np.ones(max(m, n), dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('A', 'A', m, n, x, taup, tauq, u, v, inul)

    assert info == 0
    np.testing.assert_allclose(u_out, u_orig, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v_out, v_orig, rtol=1e-14, atol=1e-14)


def test_mb04xy_zero_dimensions():
    """
    Test MB04XY with zero dimensions (quick return).
    """
    from ctrlsys import mb04xy

    x = np.zeros((1, 1), dtype=float, order='F')
    taup = np.zeros(0, dtype=float)
    tauq = np.zeros(0, dtype=float)
    inul = np.zeros(1, dtype=bool)

    u_out, v_out, info = mb04xy('N', 'N', 0, 0, x, taup, tauq, None, None, inul)

    assert info == 0


def test_mb04xy_invalid_jobu():
    """
    Test MB04XY with invalid JOBU parameter.
    """
    from ctrlsys import mb04xy

    m, n = 3, 3
    x = np.eye(m, n, dtype=float, order='F')
    taup = np.zeros(min(m, n), dtype=float)
    tauq = np.zeros(min(m, n), dtype=float)
    inul = np.ones(max(m, n), dtype=bool)

    u_out, v_out, info = mb04xy('X', 'N', m, n, x, taup, tauq, None, None, inul)

    assert info == -1


def test_mb04xy_invalid_jobv():
    """
    Test MB04XY with invalid JOBV parameter.
    """
    from ctrlsys import mb04xy

    m, n = 3, 3
    x = np.eye(m, n, dtype=float, order='F')
    taup = np.zeros(min(m, n), dtype=float)
    tauq = np.zeros(min(m, n), dtype=float)
    inul = np.ones(max(m, n), dtype=bool)

    u_out, v_out, info = mb04xy('N', 'X', m, n, x, taup, tauq, None, None, inul)

    assert info == -2


def test_mb04xy_x_restored():
    """
    Test that MB04XY restores X matrix after modification.

    Per the documentation: "X is modified by the routine but restored on exit."

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    m, n = 4, 3
    p = min(m, n)

    x = np.random.randn(m, n).astype(float, order='F')
    x_orig = x.copy()
    taup = np.zeros(p, dtype=float)
    tauq = np.zeros(p, dtype=float)

    u = np.eye(m, dtype=float, order='F')
    inul = np.ones(max(m, n), dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('A', 'N', m, n, x, taup, tauq, u, None, inul)

    assert info == 0
    np.testing.assert_allclose(x, x_orig, rtol=1e-14, atol=1e-14)


def test_mb04xy_single_householder():
    """
    Test MB04XY with a single non-trivial Householder reflector.

    For M>=N (tall matrix), DGEBRD stores P_l reflector in X[l:m, l] where
    the diagonal element X[l,l] is temporarily set to 1 during the application.

    The reflector is: H = I - tau * v * v' where v = [1; x[l+1:m, l]]

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    m, n = 4, 3
    p = min(m, n)

    x = np.zeros((m, n), dtype=float, order='F')
    x[0, 0] = 0.0
    x[1, 0] = 0.5
    x[2, 0] = 0.3
    x[3, 0] = 0.2

    v = np.array([1.0, 0.5, 0.3, 0.2])
    tau = 2.0 / np.dot(v, v)
    taup = np.array([tau, 0.0, 0.0], dtype=float)
    tauq = np.zeros(p, dtype=float)

    u = np.eye(m, dtype=float, order='F')
    inul = np.ones(max(m, n), dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('A', 'N', m, n, x, taup, tauq, u, None, inul)

    assert info == 0

    np.testing.assert_allclose(u_out.T @ u_out, np.eye(m), rtol=1e-13, atol=1e-13)

    h = np.eye(m) - tau * np.outer(v, v)
    u_expected = h
    np.testing.assert_allclose(u_out, u_expected, rtol=1e-13, atol=1e-13)


def test_mb04xy_householder_properties():
    """
    Test MB04XY with valid Householder reflectors.

    When taup[i] != 0, a Householder transformation H = I - taup*v*v' is applied.
    The v vector is stored in x[l:m, l] where x[l,l] is implicitly 1.

    We verify:
    - Result U remains orthogonal (U'U = I)
    - Transformation matches direct Householder computation

    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    m, n = 4, 3
    p = min(m, n)

    x = np.zeros((m, n), dtype=float, order='F')

    x[0, 0] = 0.0
    x[1, 0] = 0.5
    x[2, 0] = 0.5
    x[3, 0] = 0.5

    v = np.array([1.0, 0.5, 0.5, 0.5])
    tau = 2.0 / np.dot(v, v)
    taup = np.array([tau, 0.0, 0.0], dtype=float)
    tauq = np.zeros(p, dtype=float)

    u = np.eye(m, dtype=float, order='F')
    inul = np.ones(max(m, n), dtype=bool)

    from ctrlsys import mb04xy

    u_out, v_out, info = mb04xy('A', 'N', m, n, x, taup, tauq, u, None, inul)

    assert info == 0
    np.testing.assert_allclose(u_out.T @ u_out, np.eye(m), rtol=1e-13, atol=1e-13)
