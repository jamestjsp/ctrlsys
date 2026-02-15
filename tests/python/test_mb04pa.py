"""
Tests for MB04PA: Special reduction of (skew-)Hamiltonian like matrix.

MB04PA is an auxiliary routine called by MB04PB for reducing a Hamiltonian
or skew-Hamiltonian like matrix using orthogonal symplectic transformation.

Tests numerical correctness with reproducible random data.
Uses numpy only - no scipy/control.
"""

import numpy as np


def test_mb04pa_quick_return():
    """Test quick return when n+k <= 0."""
    from ctrlsys import mb04pa

    n, k, nb = 0, 0, 0
    lda = 1
    ldqg = 1
    ldxa = 1
    ldxg = 1
    ldxq = 1
    ldya = 1

    a = np.zeros((1, 1), order='F')
    qg = np.zeros((1, 2), order='F')
    xa = np.zeros((1, 1), order='F')
    xg = np.zeros((1, 1), order='F')
    xq = np.zeros((1, 1), order='F')
    ya = np.zeros((1, 1), order='F')
    cs = np.zeros(1, order='F')
    tau = np.zeros(1, order='F')
    dwork = np.zeros(1, order='F')

    a_out, qg_out, xa_out, xg_out, xq_out, ya_out, cs_out, tau_out = mb04pa(
        True, n, k, nb, a, qg, xa, xg, xq, ya, cs, tau, dwork
    )

    assert dwork[0] == 1.0


def test_mb04pa_hamiltonian_basic():
    """
    Test Hamiltonian reduction with small matrix.

    Tests that the routine produces modified arrays without crashing.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04pa

    np.random.seed(42)

    n = 5
    k = 2
    nb = 2
    lda = k + n
    ldqg = n + k

    a = np.random.randn(lda, n).astype(float, order='F')
    qg = np.random.randn(ldqg, n + 1).astype(float, order='F')
    xa = np.zeros((n, 2 * nb), order='F')
    xg = np.zeros((k + n, 2 * nb), order='F')
    xq = np.zeros((n, 2 * nb), order='F')
    ya = np.zeros((k + n, 2 * nb), order='F')
    cs = np.zeros(2 * nb, order='F')
    tau = np.zeros(nb, order='F')
    dwork = np.zeros(3 * nb, order='F')

    a_orig = a.copy()
    qg_orig = qg.copy()

    a_out, qg_out, xa_out, xg_out, xq_out, ya_out, cs_out, tau_out = mb04pa(
        True, n, k, nb, a, qg, xa, xg, xq, ya, cs, tau, dwork
    )

    assert not np.allclose(a_out, a_orig, rtol=1e-14)
    assert not np.allclose(qg_out, qg_orig, rtol=1e-14)


def test_mb04pa_skew_hamiltonian_basic():
    """
    Test skew-Hamiltonian reduction with small matrix.

    Tests the LHAM=False branch which handles skew-Hamiltonian matrices.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04pa

    np.random.seed(123)

    n = 5
    k = 2
    nb = 2
    lda = k + n
    ldqg = n + k

    a = np.random.randn(lda, n).astype(float, order='F')
    qg = np.random.randn(ldqg, n + 1).astype(float, order='F')
    xa = np.zeros((n, 2 * nb), order='F')
    xg = np.zeros((k + n, 2 * nb), order='F')
    xq = np.zeros((n, 2 * nb), order='F')
    ya = np.zeros((k + n, 2 * nb), order='F')
    cs = np.zeros(2 * nb, order='F')
    tau = np.zeros(nb, order='F')
    dwork = np.zeros(3 * nb, order='F')

    a_orig = a.copy()
    qg_orig = qg.copy()

    a_out, qg_out, xa_out, xg_out, xq_out, ya_out, cs_out, tau_out = mb04pa(
        False, n, k, nb, a, qg, xa, xg, xq, ya, cs, tau, dwork
    )

    assert not np.allclose(a_out, a_orig, rtol=1e-14)
    assert not np.allclose(qg_out, qg_orig, rtol=1e-14)


def test_mb04pa_cs_tau_outputs():
    """
    Test that CS and TAU outputs are properly computed.

    The CS array should contain cosines and sines of Givens rotations.
    The TAU array should contain scalar factors of reflectors.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04pa

    np.random.seed(456)

    n = 6
    k = 1
    nb = 3
    lda = k + n
    ldqg = n + k

    a = np.random.randn(lda, n).astype(float, order='F')
    qg = np.random.randn(ldqg, n + 1).astype(float, order='F')
    xa = np.zeros((n, 2 * nb), order='F')
    xg = np.zeros((k + n, 2 * nb), order='F')
    xq = np.zeros((n, 2 * nb), order='F')
    ya = np.zeros((k + n, 2 * nb), order='F')
    cs = np.zeros(2 * nb, order='F')
    tau = np.zeros(nb, order='F')
    dwork = np.zeros(3 * nb, order='F')

    a_out, qg_out, xa_out, xg_out, xq_out, ya_out, cs_out, tau_out = mb04pa(
        True, n, k, nb, a, qg, xa, xg, xq, ya, cs, tau, dwork
    )

    for i in range(nb):
        c = cs_out[2 * i]
        s = cs_out[2 * i + 1]
        np.testing.assert_allclose(c * c + s * s, 1.0, rtol=1e-14)


def test_mb04pa_single_block():
    """
    Test with nb=1 (single column/row reduction).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04pa

    np.random.seed(789)

    n = 4
    k = 1
    nb = 1
    lda = k + n
    ldqg = n + k

    a = np.random.randn(lda, n).astype(float, order='F')
    qg = np.random.randn(ldqg, n + 1).astype(float, order='F')
    xa = np.zeros((n, 2 * nb), order='F')
    xg = np.zeros((k + n, 2 * nb), order='F')
    xq = np.zeros((n, 2 * nb), order='F')
    ya = np.zeros((k + n, 2 * nb), order='F')
    cs = np.zeros(2 * nb, order='F')
    tau = np.zeros(nb, order='F')
    dwork = np.zeros(3 * nb, order='F')

    a_out, qg_out, xa_out, xg_out, xq_out, ya_out, cs_out, tau_out = mb04pa(
        True, n, k, nb, a, qg, xa, xg, xq, ya, cs, tau, dwork
    )

    c = cs_out[0]
    s = cs_out[1]
    np.testing.assert_allclose(c * c + s * s, 1.0, rtol=1e-14)


def test_mb04pa_larger_k():
    """
    Test with larger offset k.

    Random seed: 1234 (for reproducibility)
    """
    from ctrlsys import mb04pa

    np.random.seed(1234)

    n = 4
    k = 3
    nb = 2
    lda = k + n
    ldqg = n + k

    a = np.random.randn(lda, n).astype(float, order='F')
    qg = np.random.randn(ldqg, n + 1).astype(float, order='F')
    xa = np.zeros((n, 2 * nb), order='F')
    xg = np.zeros((k + n, 2 * nb), order='F')
    xq = np.zeros((n, 2 * nb), order='F')
    ya = np.zeros((k + n, 2 * nb), order='F')
    cs = np.zeros(2 * nb, order='F')
    tau = np.zeros(nb, order='F')
    dwork = np.zeros(3 * nb, order='F')

    a_out, qg_out, xa_out, xg_out, xq_out, ya_out, cs_out, tau_out = mb04pa(
        True, n, k, nb, a, qg, xa, xg, xq, ya, cs, tau, dwork
    )

    for i in range(nb):
        c = cs_out[2 * i]
        s = cs_out[2 * i + 1]
        np.testing.assert_allclose(c * c + s * s, 1.0, rtol=1e-14)


def test_mb04pa_output_dimensions():
    """
    Test that output arrays have correct dimensions.

    Random seed: 5678 (for reproducibility)
    """
    from ctrlsys import mb04pa

    np.random.seed(5678)

    n = 5
    k = 2
    nb = 2
    lda = k + n
    ldqg = n + k

    a = np.random.randn(lda, n).astype(float, order='F')
    qg = np.random.randn(ldqg, n + 1).astype(float, order='F')
    xa = np.zeros((n, 2 * nb), order='F')
    xg = np.zeros((k + n, 2 * nb), order='F')
    xq = np.zeros((n, 2 * nb), order='F')
    ya = np.zeros((k + n, 2 * nb), order='F')
    cs = np.zeros(2 * nb, order='F')
    tau = np.zeros(nb, order='F')
    dwork = np.zeros(3 * nb, order='F')

    a_out, qg_out, xa_out, xg_out, xq_out, ya_out, cs_out, tau_out = mb04pa(
        True, n, k, nb, a, qg, xa, xg, xq, ya, cs, tau, dwork
    )

    assert a_out.shape == (lda, n)
    assert qg_out.shape == (ldqg, n + 1)
    assert xa_out.shape == (n, 2 * nb)
    assert xg_out.shape == (k + n, 2 * nb)
    assert xq_out.shape == (n, 2 * nb)
    assert ya_out.shape == (k + n, 2 * nb)
    assert cs_out.shape == (2 * nb,)
    assert tau_out.shape == (nb,)
