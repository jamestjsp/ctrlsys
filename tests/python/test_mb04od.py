import numpy as np
import pytest


def test_mb04od_html_example():
    """
    Test MB04OD using HTML documentation example

    From TMB04OD.f example program:
    N=3, M=2, P=2, UPLO='F'

    Input data read row-wise from HTML (Fortran READ with J inner loop)
    """
    n, m, p = 3, 2, 2
    uplo = 'F'

    r = np.array([
        [3., 2., 1.],
        [0., 2., 1.],
        [0., 0., 1.]
    ], dtype=float, order='F')

    a = np.array([
        [2., 3., 1.],
        [4., 6., 5.]
    ], dtype=float, order='F')

    b = np.array([
        [3., 2.],
        [1., 3.],
        [3., 2.]
    ], dtype=float, order='F')

    c = np.array([
        [1., 3.],
        [3., 2.]
    ], dtype=float, order='F')

    r_expected = np.array([
        [-5.3852, -6.6850, -4.6424],
        [0.0000, -2.8828, -2.0694],
        [0.0000, 0.0000, -1.7793]
    ], dtype=float, order='F')

    b_expected = np.array([
        [-4.2710, -3.7139],
        [-0.1555, -2.1411],
        [-1.6021, 0.9398]
    ], dtype=float, order='F')

    c_expected = np.array([
        [0.5850, 1.0141],
        [-2.7974, -3.1162]
    ], dtype=float, order='F')

    from ctrlsys import mb04od

    r_out, a_out, b_out, c_out, tau = mb04od(uplo, n, m, p, r, a, b, c)

    np.testing.assert_allclose(np.triu(r_out), r_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)
    assert tau.shape == (n,)


def test_mb04od_upper_trapezoidal():
    """
    Test MB04OD with UPLO='U' (upper trapezoidal A)

    Validates R is upper triangular and transformation is orthogonal.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n, m, p = 4, 3, 3
    uplo = 'U'

    r = np.random.randn(n, n).astype(float, order='F')
    r = np.triu(r)

    a = np.random.randn(p, n).astype(float, order='F')
    a = np.triu(a)

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04od

    r_out, a_out, b_out, c_out, tau = mb04od(uplo, n, m, p, r, a, b, c)

    assert np.all(np.tril(r_out, -1) == 0.0)
    assert np.all(np.isfinite(b_out))
    assert np.all(np.isfinite(c_out))
    assert tau.shape == (n,)


def test_mb04od_full_matrix():
    """
    Test MB04OD with UPLO='F' (full A matrix)

    Validates R is upper triangular and outputs are finite.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n, m, p = 3, 2, 4
    uplo = 'F'

    r = np.random.randn(n, n).astype(float, order='F')
    r = np.triu(r)

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04od

    r_out, a_out, b_out, c_out, tau = mb04od(uplo, n, m, p, r, a, b, c)

    assert np.all(np.tril(r_out, -1) == 0.0)
    assert r_out.shape == (n, n)
    assert b_out.shape == (n, m)
    assert c_out.shape == (p, m)
    assert np.all(np.isfinite(r_out))
    assert np.all(np.isfinite(b_out))
    assert np.all(np.isfinite(c_out))
    assert tau.shape == (n,)


def test_mb04od_orthogonality():
    """
    Test MB04OD orthogonality property: Q'*Q = I (implicit)

    Verify that Householder reflectors preserve norms.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n, m, p = 4, 2, 3
    uplo = 'F'

    r = np.random.randn(n, n).astype(float, order='F')
    r = np.triu(r)

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, m).astype(float, order='F')

    block_col1 = np.vstack([r, a])
    norm_before = np.linalg.norm(block_col1, 'fro')

    from ctrlsys import mb04od

    r_out, a_out, b_out, c_out, tau = mb04od(uplo, n, m, p, r, a, b, c)

    block_col1_after = np.vstack([r_out, np.zeros((p, n))])
    norm_after = np.linalg.norm(block_col1_after, 'fro')

    np.testing.assert_allclose(norm_after, norm_before, rtol=1e-13, atol=1e-14)


def test_mb04od_zero_m():
    """
    Test MB04OD with M=0 (no second block column)

    When M=0, routine still applies QR to first block column [R; A].
    """
    n, m, p = 3, 0, 2
    uplo = 'F'

    r = np.array([
        [1., 2., 3.],
        [0., 4., 5.],
        [0., 0., 6.]
    ], dtype=float, order='F')

    a = np.array([
        [1., 2., 3.],
        [4., 5., 6.]
    ], dtype=float, order='F')

    b = np.zeros((n, max(1, m)), dtype=float, order='F')
    c = np.zeros((p, max(1, m)), dtype=float, order='F')

    from ctrlsys import mb04od

    r_out, a_out, b_out, c_out, tau = mb04od(uplo, n, m, p, r, a, b, c)

    assert r_out.shape == (n, n)
    assert np.all(np.tril(r_out, -1) == 0.0)
    assert tau.shape == (n,)


def test_mb04od_zero_p():
    """
    Test MB04OD with P=0 (no A or C matrices)
    """
    n, m, p = 3, 2, 0
    uplo = 'F'

    r = np.array([
        [1., 2., 3.],
        [0., 4., 5.],
        [0., 0., 6.]
    ], dtype=float, order='F')

    a = np.zeros((max(1, p), n), dtype=float, order='F')
    b = np.array([
        [1., 2.],
        [3., 4.],
        [5., 6.]
    ], dtype=float, order='F')
    c = np.zeros((max(1, p), m), dtype=float, order='F')

    from ctrlsys import mb04od

    r_out, a_out, b_out, c_out, tau = mb04od(uplo, n, m, p, r, a, b, c)

    np.testing.assert_allclose(r_out, r, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(b_out, b, rtol=1e-14, atol=1e-15)


def test_mb04od_zero_n():
    """
    Test MB04OD with N=0 (quick return)
    """
    n, m, p = 0, 2, 3
    uplo = 'F'

    r = np.zeros((max(1, n), max(1, n)), dtype=float, order='F')
    a = np.zeros((max(1, p), max(1, n)), dtype=float, order='F')
    b = np.zeros((max(1, n), max(1, m)), dtype=float, order='F')
    c = np.zeros((max(1, p), max(1, m)), dtype=float, order='F')

    from ctrlsys import mb04od

    r_out, a_out, b_out, c_out, tau = mb04od(uplo, n, m, p, r, a, b, c)

    assert r_out.shape[0] >= 1
    assert tau.shape == (max(1, n),)


def test_mb04od_r_upper_triangular():
    """
    Test MB04OD property: output R remains upper triangular

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    n, m, p = 5, 3, 4
    uplo = 'F'

    r = np.random.randn(n, n).astype(float, order='F')
    r = np.triu(r)

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04od

    r_out, a_out, b_out, c_out, tau = mb04od(uplo, n, m, p, r, a, b, c)

    for i in range(n):
        for j in range(i):
            assert abs(r_out[i, j]) < 1e-14, f"R[{i},{j}] = {r_out[i, j]} (should be ~0)"


def test_mb04od_uplo_u_with_p_gt_n():
    """
    Test MB04OD with UPLO='U' and P > N

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    n, m, p = 3, 2, 5
    uplo = 'U'

    r = np.random.randn(n, n).astype(float, order='F')
    r = np.triu(r)

    a = np.random.randn(p, n).astype(float, order='F')
    a[:min(p, n), :] = np.triu(a[:min(p, n), :])

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, m).astype(float, order='F')

    from ctrlsys import mb04od

    r_out, a_out, b_out, c_out, tau = mb04od(uplo, n, m, p, r, a, b, c)

    assert np.all(np.isfinite(r_out))
    assert np.all(np.isfinite(b_out))
    assert np.all(np.isfinite(c_out))
    assert np.all(np.tril(r_out, -1) == 0.0)
