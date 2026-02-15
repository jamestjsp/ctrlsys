"""Tests for TF01QD: Markov parameters from transfer function matrix.

TF01QD computes N Markov parameters M(1), M(2), ..., M(N) from a
multivariable system whose transfer function matrix G(z) is given
in ARMA (autoregressive moving-average) form.

The (i,j)-th element of G(z) has the form:
           MA(1)z^{-1} + MA(2)z^{-2} + ... + MA(r)z^{-r}
  G_ij(z) = --------------------------------------------
            1 + AR(1)z^{-1} + AR(2)z^{-2} + ... + AR(r)z^{-r}
"""

import numpy as np
import pytest
from ctrlsys import tf01qd


def test_html_doc_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Tests NC=2, NB=2, N=8 system with transfer function elements
    of orders 2, 1, 3, 4.

    Data from SLICOT-Reference/doc/TF01QD.html
    """
    nc = 2  # outputs
    nb = 2  # inputs
    n = 8   # Markov parameters to compute

    # Orders of transfer function elements (row by row)
    # G(1,1): order 2, G(1,2): order 1, G(2,1): order 3, G(2,2): order 4
    iord = np.array([2, 1, 3, 4], dtype=np.int32)

    # MA (numerator) coefficients - concatenated row by row
    # G(1,1): MA = [1.0, -0.5] (order 2)
    # G(1,2): MA = [1.0] (order 1)
    # G(2,1): MA = [0.5, -0.4, 0.3] (order 3)
    # G(2,2): MA = [1.0, 0.5, -0.5, 0.0] (order 4)
    ma = np.array([1.0, -0.5, 1.0, 0.5, -0.4, 0.3, 1.0, 0.5, -0.5, 0.0],
                  dtype=float)

    # AR (denominator) coefficients - concatenated row by row
    # G(1,1): AR = [0.6, -0.2] (order 2)
    # G(1,2): AR = [-0.8] (order 1)
    # G(2,1): AR = [0.8, 0.4, 0.1] (order 3)
    # G(2,2): AR = [-0.8, 0.6, 0.0, -0.2] (order 4)
    ar = np.array([0.6, -0.2, -0.8, 0.8, 0.4, 0.1, -0.8, 0.6, 0.0, -0.2],
                  dtype=float)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0, f"Expected info=0, got {info}"
    assert h.shape == (nc, n * nb), f"Expected shape ({nc}, {n * nb}), got {h.shape}"

    # Expected Markov parameters from HTML doc
    m1_expected = np.array([[1.0, 1.0], [0.5, 1.0]], order='F', dtype=float)
    m2_expected = np.array([[-1.1, 0.8], [-0.8, 1.3]], order='F', dtype=float)
    m3_expected = np.array([[0.86, 0.64], [0.74, -0.06]], order='F', dtype=float)
    m4_expected = np.array([[-0.736, 0.512], [-0.322, -0.828]], order='F', dtype=float)
    m5_expected = np.array([[0.6136, 0.4096], [0.0416, -0.4264]], order='F', dtype=float)
    m6_expected = np.array([[-0.5154, 0.3277], [0.0215, 0.4157]], order='F', dtype=float)
    m7_expected = np.array([[0.4319, 0.2621], [-0.0017, 0.5764]], order='F', dtype=float)
    m8_expected = np.array([[-0.3622, 0.2097], [-0.0114, 0.0461]], order='F', dtype=float)

    # Validate each M(k)
    np.testing.assert_allclose(h[:, 0:2], m1_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 2:4], m2_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 4:6], m3_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 6:8], m4_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 8:10], m5_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 10:12], m6_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 12:14], m7_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(h[:, 14:16], m8_expected, rtol=1e-3, atol=1e-4)


def test_siso_first_order():
    """
    Validate SISO first-order system.

    G(z) = MA(1)z^{-1} / (1 + AR(1)z^{-1})
         = 0.5 z^{-1} / (1 - 0.8 z^{-1})

    The Markov parameters are: M(1)=0.5, M(k) = -AR(1)*M(k-1) = 0.8*M(k-1)
    So M(k) = 0.5 * 0.8^{k-1}
    """
    nc = 1
    nb = 1
    n = 5
    iord = np.array([1], dtype=np.int32)
    ma = np.array([0.5], dtype=float)
    ar = np.array([-0.8], dtype=float)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0
    assert h.shape == (1, 5)

    # M(k) = 0.5 * 0.8^(k-1)
    for k in range(1, n + 1):
        expected = 0.5 * (0.8 ** (k - 1))
        np.testing.assert_allclose(h[0, k - 1], expected, rtol=1e-14)


def test_markov_recurrence_relation():
    """
    Validate Markov parameter recurrence relation for k > r.

    For k > r: M(k+r) = -sum_{p=1}^{r} AR(p) * M(k+r-p)

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    nc = 1
    nb = 1
    n = 10

    r = 3
    iord = np.array([r], dtype=np.int32)
    ma = np.random.randn(r)
    ar = np.random.randn(r)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)
    assert info == 0

    # Verify recurrence for k > r
    for k in range(r + 1, n + 1):
        m_expected = 0.0
        for p in range(1, r + 1):
            m_expected -= ar[p - 1] * h[0, k - p - 1]
        np.testing.assert_allclose(h[0, k - 1], m_expected, rtol=1e-14)


def test_first_markov_parameter():
    """
    Validate M_ij(1) = MA(1) for all elements.

    The first Markov parameter equals the first MA coefficient.
    """
    nc = 2
    nb = 2
    n = 1

    iord = np.array([2, 3, 1, 2], dtype=np.int32)
    # First MA coefficients: 1.0, 2.0, 3.0, 4.0
    ma = np.array([1.0, 0.5, 2.0, 0.1, 0.2, 3.0, 4.0, 0.3], dtype=float)
    ar = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0

    # M(1) should be [[1.0, 2.0], [3.0, 4.0]]
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
    np.testing.assert_allclose(h, expected, rtol=1e-14)


def test_mimo_system():
    """
    Validate MIMO system with mixed orders.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    nc = 2
    nb = 3
    n = 6

    # Different orders for each transfer function element
    iord = np.array([2, 1, 3, 2, 2, 1], dtype=np.int32)  # nc*nb = 6 elements
    total_coeffs = sum(iord)  # 2+1+3+2+2+1 = 11

    ma = np.random.randn(total_coeffs)
    ar = np.random.randn(total_coeffs)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0
    assert h.shape == (nc, n * nb)

    # Verify first Markov parameter: M(1)_ij = first MA coeff of G_ij
    nl = 0
    for i in range(nc):
        for j in range(nb):
            idx = i * nb + j
            order = iord[idx]
            np.testing.assert_allclose(h[i, j], ma[nl], rtol=1e-14)
            nl += order


def test_zero_ar_coefficients():
    """
    Validate behavior with zero AR coefficients (FIR filter).

    When all AR(p) = 0, G(z) is a finite impulse response:
    M(k) = MA(k) for k <= r, M(k) = 0 for k > r
    """
    nc = 1
    nb = 1
    n = 6

    r = 3
    iord = np.array([r], dtype=np.int32)
    ma = np.array([1.0, 2.0, 3.0], dtype=float)
    ar = np.array([0.0, 0.0, 0.0], dtype=float)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0

    # M(1)=1, M(2)=2, M(3)=3 (directly from MA)
    np.testing.assert_allclose(h[0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(h[0, 1], 2.0, rtol=1e-14)
    np.testing.assert_allclose(h[0, 2], 3.0, rtol=1e-14)

    # M(4), M(5), M(6) = 0 (no feedback)
    np.testing.assert_allclose(h[0, 3], 0.0, atol=1e-15)
    np.testing.assert_allclose(h[0, 4], 0.0, atol=1e-15)
    np.testing.assert_allclose(h[0, 5], 0.0, atol=1e-15)


def test_single_output():
    """
    Validate NC=1 (single output) case.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    nc = 1
    nb = 3
    n = 4

    iord = np.array([2, 1, 2], dtype=np.int32)
    ma = np.random.randn(5)
    ar = np.random.randn(5)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0
    assert h.shape == (1, 12)


def test_single_input():
    """
    Validate NB=1 (single input) case.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    nc = 3
    nb = 1
    n = 4

    iord = np.array([2, 1, 3], dtype=np.int32)
    ma = np.random.randn(6)
    ar = np.random.randn(6)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0
    assert h.shape == (3, 4)


def test_quick_return_n_zero():
    """
    Validate quick return when N=0.
    """
    nc = 2
    nb = 2
    n = 0

    iord = np.array([1, 1, 1, 1], dtype=np.int32)
    ma = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    ar = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0
    assert h.shape[1] == 0


def test_quick_return_nc_zero():
    """
    Validate quick return when NC=0.
    """
    nc = 0
    nb = 2
    n = 5

    iord = np.array([], dtype=np.int32)
    ma = np.array([], dtype=float)
    ar = np.array([], dtype=float)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0
    assert h.shape[0] == 0


def test_quick_return_nb_zero():
    """
    Validate quick return when NB=0.
    """
    nc = 2
    nb = 0
    n = 5

    iord = np.array([], dtype=np.int32)
    ma = np.array([], dtype=float)
    ar = np.array([], dtype=float)

    h, info = tf01qd(nc, nb, n, iord, ar, ma)

    assert info == 0
    assert h.shape[1] == 0


def test_negative_nc_error():
    """
    Validate negative NC raises error.
    """
    with pytest.raises(ValueError):
        tf01qd(-1, 2, 5, np.array([1], dtype=np.int32),
               np.array([0.5], dtype=float), np.array([1.0], dtype=float))


def test_negative_nb_error():
    """
    Validate negative NB raises error.
    """
    with pytest.raises(ValueError):
        tf01qd(2, -1, 5, np.array([1], dtype=np.int32),
               np.array([0.5], dtype=float), np.array([1.0], dtype=float))


def test_negative_n_error():
    """
    Validate negative N raises error.
    """
    with pytest.raises(ValueError):
        tf01qd(2, 2, -1, np.array([1, 1, 1, 1], dtype=np.int32),
               np.array([0.5, 0.5, 0.5, 0.5], dtype=float),
               np.array([1.0, 1.0, 1.0, 1.0], dtype=float))
