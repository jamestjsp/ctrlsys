import numpy as np
import pytest


def test_mb01rb_side_left_upper():
    """
    Test MB01RB: R = alpha*R + beta*A*B (SIDE='L', UPLO='U', TRANS='N')

    Validates upper triangle computation for left-side multiplication.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    m, n = 4, 3
    alpha, beta = 2.0, 0.5

    r = np.random.randn(m, m).astype(float, order='F')
    r = np.triu(r)
    a = np.random.randn(m, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    r_original = r.copy()

    r_expected = alpha * r_original + beta * (a @ b)
    r_expected = np.triu(r_expected)

    from ctrlsys import mb01rb

    r_out, info = mb01rb('L', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), r_expected, rtol=1e-14, atol=1e-15)


def test_mb01rb_side_left_lower():
    """
    Test MB01RB: R = alpha*R + beta*A*B (SIDE='L', UPLO='L', TRANS='N')

    Validates lower triangle computation for left-side multiplication.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    m, n = 4, 3
    alpha, beta = 1.5, -0.5

    r = np.random.randn(m, m).astype(float, order='F')
    r = np.tril(r)
    a = np.random.randn(m, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    r_original = r.copy()

    r_expected = alpha * r_original + beta * (a @ b)
    r_expected = np.tril(r_expected)

    from ctrlsys import mb01rb

    r_out, info = mb01rb('L', 'L', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.tril(r_out), r_expected, rtol=1e-14, atol=1e-15)


def test_mb01rb_side_right_upper():
    """
    Test MB01RB: R = alpha*R + beta*B*A (SIDE='R', UPLO='U', TRANS='N')

    Validates upper triangle computation for right-side multiplication.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    m, n = 4, 3
    alpha, beta = 1.0, 1.0

    r = np.random.randn(m, m).astype(float, order='F')
    r = np.triu(r)
    a = np.random.randn(n, m).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    r_original = r.copy()

    r_expected = alpha * r_original + beta * (b @ a)
    r_expected = np.triu(r_expected)

    from ctrlsys import mb01rb

    r_out, info = mb01rb('R', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), r_expected, rtol=1e-14, atol=1e-15)


def test_mb01rb_transpose_left_upper():
    """
    Test MB01RB: R = alpha*R + beta*A'*B (SIDE='L', UPLO='U', TRANS='T')

    Validates upper triangle with transpose.
    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    m, n = 4, 5
    alpha, beta = 0.5, 2.0

    r = np.random.randn(m, m).astype(float, order='F')
    r = np.triu(r)
    a = np.random.randn(n, m).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    r_original = r.copy()

    r_expected = alpha * r_original + beta * (a.T @ b)
    r_expected = np.triu(r_expected)

    from ctrlsys import mb01rb

    r_out, info = mb01rb('L', 'U', 'T', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), r_expected, rtol=1e-14, atol=1e-15)


def test_mb01rb_alpha_zero():
    """
    Test MB01RB with alpha=0: R = beta*A*B

    When alpha=0, R is not referenced on input (special case).
    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)

    m, n = 3, 2
    alpha, beta = 0.0, 1.0

    r = np.random.randn(m, m).astype(float, order='F')
    a = np.random.randn(m, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    r_expected = beta * (a @ b)
    r_expected = np.triu(r_expected)

    from ctrlsys import mb01rb

    r_out, info = mb01rb('L', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), r_expected, rtol=1e-14, atol=1e-15)


def test_mb01rb_beta_zero():
    """
    Test MB01RB with beta=0: R = alpha*R

    When beta=0, A and B are not referenced (special case).
    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)

    m, n = 3, 2
    alpha, beta = 2.5, 0.0

    r = np.random.randn(m, m).astype(float, order='F')
    r = np.triu(r)
    a = np.random.randn(m, n).astype(float, order='F')
    b = np.random.randn(n, m).astype(float, order='F')

    r_expected = alpha * r
    r_expected = np.triu(r_expected)

    from ctrlsys import mb01rb

    r_out, info = mb01rb('L', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), r_expected, rtol=1e-14, atol=1e-15)


def test_mb01rb_zero_dimensions():
    """
    Test MB01RB with M=0 (quick return)
    """
    m, n = 0, 3
    alpha, beta = 1.0, 1.0

    r = np.zeros((max(1, m), max(1, m)), dtype=float, order='F')
    a = np.zeros((max(1, m), max(1, n)), dtype=float, order='F')
    b = np.zeros((max(1, n), max(1, m)), dtype=float, order='F')

    from ctrlsys import mb01rb

    r_out, info = mb01rb('L', 'U', 'N', m, n, alpha, beta, r, a, b)

    assert info == 0


def test_mb01rb_symmetric_property():
    """
    Test MB01RB symmetric property: when B = X*A' and X is symmetric

    MB01RB only updates UPLO triangle. Lower triangle remains zeros.
    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)

    m, n = 4, 3
    alpha, beta = 0.0, 1.0

    r = np.zeros((m, m), dtype=float, order='F')
    a = np.random.randn(m, n).astype(float, order='F')
    x = np.random.randn(n, n).astype(float, order='F')
    x = (x + x.T) / 2.0
    b = (x @ a.T).astype(float, order='F')

    r_expected = a @ b
    r_upper = np.triu(r_expected)

    from ctrlsys import mb01rb

    r_out, info = mb01rb('L', 'U', 'N', m, n, alpha, beta, r.copy(), a, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), r_upper, rtol=1e-14, atol=1e-15)


def test_mb01rb_error_invalid_side():
    """
    Test MB01RB error handling: invalid SIDE parameter
    """
    m, n = 3, 2
    alpha, beta = 1.0, 1.0

    r = np.zeros((m, m), dtype=float, order='F')
    a = np.zeros((m, n), dtype=float, order='F')
    b = np.zeros((n, m), dtype=float, order='F')

    from ctrlsys import mb01rb

    with pytest.raises(ValueError, match="Parameter 1"):
        mb01rb('X', 'U', 'N', m, n, alpha, beta, r, a, b)


def test_mb01rb_error_invalid_uplo():
    """
    Test MB01RB error handling: invalid UPLO parameter
    """
    m, n = 3, 2
    alpha, beta = 1.0, 1.0

    r = np.zeros((m, m), dtype=float, order='F')
    a = np.zeros((m, n), dtype=float, order='F')
    b = np.zeros((n, m), dtype=float, order='F')

    from ctrlsys import mb01rb

    with pytest.raises(ValueError, match="Parameter 2"):
        mb01rb('L', 'X', 'N', m, n, alpha, beta, r, a, b)


def test_mb01rb_error_negative_m():
    """
    Test MB01RB error handling: negative M
    """
    m, n = -1, 2
    alpha, beta = 1.0, 1.0

    r = np.zeros((1, 1), dtype=float, order='F')
    a = np.zeros((1, 1), dtype=float, order='F')
    b = np.zeros((1, 1), dtype=float, order='F')

    from ctrlsys import mb01rb

    with pytest.raises(ValueError, match="m"):
        mb01rb('L', 'U', 'N', m, n, alpha, beta, r, a, b)
