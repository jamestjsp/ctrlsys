import numpy as np
import pytest


def test_mb02ed_column_html_doc_example():
    """
    Validate MB02ED using HTML documentation example.

    Tests: T*X = B with block Toeplitz T (TYPET='C').
    Data from SLICOT-Reference/doc/MB02ED.html
    """
    n = 3
    k = 3
    nrhs = 2

    t = np.array([
        [3.0, 1.0, 0.2],
        [1.0, 4.0, 0.4],
        [0.2, 0.4, 5.0],
        [0.1, 0.1, 0.2],
        [0.2, 0.04, 0.03],
        [0.05, 0.2, 0.1],
        [0.1, 0.03, 0.1],
        [0.04, 0.02, 0.2],
        [0.01, 0.03, 0.02],
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ], order='F', dtype=float)

    x_expected = np.array([
        [0.2408, 0.4816],
        [0.1558, 0.3116],
        [0.1534, 0.3068],
        [0.2302, 0.4603],
        [0.1467, 0.2934],
        [0.1537, 0.3075],
        [0.2349, 0.4698],
        [0.1498, 0.2995],
        [0.1653, 0.3307],
    ], order='F', dtype=float)

    from ctrlsys import mb02ed

    x, t_out, info = mb02ed('C', k, n, nrhs, t.copy(), b.copy())

    assert info == 0
    np.testing.assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)


def test_mb02ed_row_mode():
    """
    Validate MB02ED in row mode: X*T = B.

    Tests: solve X*T = B where T is block Toeplitz.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n = 2
    k = 2
    nrhs = 3

    t = np.eye(k, n * k, order='F', dtype=float)
    t[0, 0] = 5.0
    t[1, 1] = 5.0
    t[0, 2] = 0.1
    t[1, 3] = 0.1

    b = np.random.randn(nrhs, n * k).astype(float, order='F')
    b_orig = b.copy()

    from ctrlsys import mb02ed

    x, t_out, info = mb02ed('R', k, n, nrhs, t.copy(), b.copy())

    assert info == 0
    assert x.shape == (nrhs, n * k)


def test_mb02ed_identity_toeplitz():
    """
    Validate MB02ED with identity block Toeplitz.

    When T = I, X should equal B (up to numerical tolerance).
    """
    n = 2
    k = 2
    nrhs = 2

    t = np.zeros((n * k, k), order='F', dtype=float)
    t[0, 0] = 1.0
    t[1, 1] = 1.0

    b = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ], order='F', dtype=float)
    b_expected = b.copy()

    from ctrlsys import mb02ed

    x, t_out, info = mb02ed('C', k, n, nrhs, t.copy(), b.copy())

    assert info == 0
    np.testing.assert_allclose(x, b_expected, rtol=1e-10, atol=1e-12)


def test_mb02ed_solution_satisfies_equation():
    """
    Validate that the solution X satisfies T*X = B (column mode).

    Uses the HTML doc example where we know the expected solution.
    The solution column is exactly 2x the first column.
    """
    n = 3
    k = 3
    nrhs = 2

    t = np.array([
        [3.0, 1.0, 0.2],
        [1.0, 4.0, 0.4],
        [0.2, 0.4, 5.0],
        [0.1, 0.1, 0.2],
        [0.2, 0.04, 0.03],
        [0.05, 0.2, 0.1],
        [0.1, 0.03, 0.1],
        [0.04, 0.02, 0.2],
        [0.01, 0.03, 0.02],
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ], order='F', dtype=float)

    from ctrlsys import mb02ed

    x, t_out, info = mb02ed('C', k, n, nrhs, t.copy(), b.copy())

    assert info == 0

    np.testing.assert_allclose(x[:, 1], 2.0 * x[:, 0], rtol=1e-12)


def test_mb02ed_edge_case_n_one():
    """
    Validate edge case: single block (N=1).
    """
    n = 1
    k = 2
    nrhs = 1

    t = np.array([
        [4.0, 1.0],
        [1.0, 3.0],
    ], order='F', dtype=float)

    b = np.array([
        [5.0],
        [4.0],
    ], order='F', dtype=float)

    from ctrlsys import mb02ed

    x, t_out, info = mb02ed('C', k, n, nrhs, t.copy(), b.copy())

    assert info == 0

    x_expected = np.linalg.solve(t, b)
    np.testing.assert_allclose(x, x_expected, rtol=1e-12, atol=1e-13)


def test_mb02ed_edge_case_k_one():
    """
    Validate edge case: block size 1 (scalar Toeplitz).
    """
    n = 3
    k = 1
    nrhs = 1

    t = np.array([[5.0], [0.5], [0.1]], order='F', dtype=float)

    b = np.array([[1.0], [2.0], [3.0]], order='F', dtype=float)

    from ctrlsys import mb02ed

    x, t_out, info = mb02ed('C', k, n, nrhs, t.copy(), b.copy())

    assert info == 0
    assert x.shape == (n * k, nrhs)


def test_mb02ed_error_not_positive_definite():
    """
    Validate error handling: matrix not positive definite.

    When off-diagonal blocks dominate, the algorithm fails with info=1.
    """
    n = 2
    k = 2
    nrhs = 1

    t = np.zeros((n * k, k), order='F', dtype=float)
    t[0, 0] = 0.1
    t[1, 1] = 0.1
    t[2, 0] = 10.0
    t[3, 1] = 10.0

    b = np.ones((n * k, nrhs), order='F', dtype=float)

    from ctrlsys import mb02ed

    x, t_out, info = mb02ed('C', k, n, nrhs, t.copy(), b.copy())

    assert info == 1


def test_mb02ed_error_invalid_typet():
    """
    Validate error handling: invalid TYPET parameter.
    """
    t = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)

    from ctrlsys import mb02ed

    with pytest.raises((ValueError, RuntimeError)):
        mb02ed('X', 1, 1, 1, t, b)
