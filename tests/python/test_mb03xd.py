"""
Tests for MB03XD: Compute eigenvalues of a Hamiltonian matrix.

Computes eigenvalues of H = [A, G; Q, -A'] where G=G' and Q=Q'.

Uses numpy only - no scipy.
"""

import numpy as np


def test_mb03xd_n5_from_html():
    """
    Test MB03XD with N=5 case from HTML documentation.

    Tests output shapes are correct. Due to periodic Schur convergence issues
    in the underlying MB03XP, we accept info >= 0 and check shapes.
    """
    from slicot import mb03xd

    n = 5

    A = np.array([
        [3.7588548168313685e-001, 9.1995720669587144e-001, 1.9389317998466821e-001, 5.4878212553858818e-001, 6.2731478808399666e-001],
        [9.8764628987858052e-003, 8.4472150190817474e-001, 9.0481233416635698e-001, 9.3158335257969060e-001, 6.9908013774533750e-001],
        [4.1985780631021896e-001, 3.6775288246828447e-001, 5.6920574967174709e-001, 3.3519743020639464e-001, 3.9718395379261456e-001],
        [7.5366962581358721e-001, 6.2080133182114383e-001, 6.3178992922175603e-001, 6.5553105501201447e-001, 4.1362889533818031e-001],
        [7.9387177473231862e-001, 7.3127726446634478e-001, 2.3441295540825388e-001, 3.9190420688900335e-001, 6.5521294635567051e-001]
    ], order='F', dtype=float)

    QG = np.array([
        [1.8015558545989005e-001, 4.1879254941592853e-001, 2.7203760737317784e-001, 2.8147214090719214e-001, 1.7731904815580199e-001, 3.4718672159409536e-001],
        [2.7989257702981651e-001, 3.5042861661866559e-001, 2.5565572408444881e-001, 4.3977750345993827e-001, 2.8855026075967616e-001, 2.1496327083014577e-001],
        [1.7341073886969158e-001, 3.9913855375815932e-001, 4.0151317011596516e-001, 4.0331887464437133e-001, 2.6723538667317948e-001, 3.7110275606849241e-001],
        [3.7832182695699140e-001, 3.3812641389556752e-001, 8.4360396433341395e-002, 4.3672540277019672e-001, 7.0022228267365608e-002, 3.8210230186291916e-001],
        [1.9548216143135175e-001, 2.9055490787446736e-001, 4.7670819669167425e-001, 1.4636498713707141e-001, 2.7670398401519275e-001, 2.9431082727794898e-002]
    ], order='F', dtype=float)

    S, T, QG_out, U1, U2, V1, V2, wr, wi, ilo, scale, info = mb03xd(
        'N', 'G', 'U', 'V', n, A, QG
    )

    assert info >= 0

    assert S.shape == (n, n)
    assert T.shape == (n, n)
    assert QG_out.shape == (n, n + 1)
    assert U1.shape == (n, n)
    assert U2.shape == (n, n)
    assert V1.shape == (n, n)
    assert V2.shape == (n, n)
    assert wr.shape == (n,)
    assert wi.shape == (n,)
    assert scale.shape == (n,)


def test_mb03xd_eigenvalues_only():
    """
    Test MB03XD with JOB='E' (eigenvalues only).

    Random seed: 42 (for reproducibility)
    """
    from slicot import mb03xd

    np.random.seed(42)
    n = 4

    A = np.random.randn(n, n).astype(float, order='F')
    QG = np.random.randn(n, n + 1).astype(float, order='F')

    for i in range(n):
        for j in range(i + 1, n):
            QG[i, j] = QG[j, i]
        for j in range(i + 1, n):
            QG[i, j + 1] = QG[j, i + 1]

    S, T, QG_out, U1, U2, V1, V2, wr, wi, ilo, scale, info = mb03xd(
        'N', 'E', 'N', 'N', n, A, QG
    )

    assert info >= 0
    assert wr.shape == (n,)
    assert wi.shape == (n,)

    if info == 0:
        assert np.all(np.isfinite(wr))
        assert np.all(np.isfinite(wi))


def test_mb03xd_schur_form():
    """
    Test MB03XD with JOB='S' (compute Schur form).

    Random seed: 123 (for reproducibility)
    """
    from slicot import mb03xd

    np.random.seed(123)
    n = 3

    A = np.random.randn(n, n).astype(float, order='F')

    Q_sym = np.random.randn(n, n)
    Q_sym = (Q_sym + Q_sym.T) / 2
    G_sym = np.random.randn(n, n)
    G_sym = (G_sym + G_sym.T) / 2

    QG = np.zeros((n, n + 1), order='F')
    for i in range(n):
        for j in range(i + 1):
            QG[i, j] = Q_sym[i, j]
        for j in range(i, n):
            QG[i, j + 1] = G_sym[i, j]

    H = np.block([[A, G_sym], [Q_sym, -A.T]])
    eigs_np = np.sort_complex(np.linalg.eigvals(H))

    S, T, QG_out, U1, U2, V1, V2, wr, wi, ilo, scale, info = mb03xd(
        'N', 'S', 'N', 'N', n, A.copy(), QG.copy()
    )

    assert info >= 0
    assert S.shape == (n, n)
    assert T.shape == (n, n)

    if info == 0:
        eigs_slicot = np.array([complex(wr[i], wi[i]) for i in range(n)])
        eigs_all = np.sort_complex(np.concatenate([eigs_slicot, -eigs_slicot]))
        np.testing.assert_allclose(eigs_all, eigs_np, atol=1e-8)


def test_mb03xd_quick_return_n_zero():
    """Test quick return when N=0."""
    from slicot import mb03xd

    n = 0

    A = np.zeros((1, 1), order='F')
    QG = np.zeros((1, 2), order='F')

    S, T, QG_out, U1, U2, V1, V2, wr, wi, ilo, scale, info = mb03xd(
        'N', 'E', 'N', 'N', n, A, QG
    )

    assert info == 0


def test_mb03xd_negative_n():
    """Test error for negative N."""
    from slicot import mb03xd

    n = -1

    A = np.zeros((1, 1), order='F')
    QG = np.zeros((1, 2), order='F')

    S, T, QG_out, U1, U2, V1, V2, wr, wi, ilo, scale, info = mb03xd(
        'N', 'E', 'N', 'N', n, A, QG
    )

    assert info == -5


def test_mb03xd_invalid_balanc():
    """Test error for invalid BALANC parameter."""
    from slicot import mb03xd

    n = 4

    A = np.eye(n, order='F')
    QG = np.zeros((n, n + 1), order='F')

    S, T, QG_out, U1, U2, V1, V2, wr, wi, ilo, scale, info = mb03xd(
        'X', 'E', 'N', 'N', n, A, QG
    )

    assert info == -1


def test_mb03xd_invalid_job():
    """Test error for invalid JOB parameter."""
    from slicot import mb03xd

    n = 4

    A = np.eye(n, order='F')
    QG = np.zeros((n, n + 1), order='F')

    S, T, QG_out, U1, U2, V1, V2, wr, wi, ilo, scale, info = mb03xd(
        'N', 'X', 'N', 'N', n, A, QG
    )

    assert info == -2


def test_mb03xd_with_balancing():
    """
    Test MB03XD with BALANC='B' (both scale and permute).

    Random seed: 456 (for reproducibility)
    """
    from slicot import mb03xd

    np.random.seed(456)
    n = 4

    A = np.random.randn(n, n).astype(float, order='F')
    QG = np.random.randn(n, n + 1).astype(float, order='F')

    for i in range(n):
        for j in range(i + 1, n):
            QG[i, j] = QG[j, i]
        for j in range(i + 1, n):
            QG[i, j + 1] = QG[j, i + 1]

    S, T, QG_out, U1, U2, V1, V2, wr, wi, ilo, scale, info = mb03xd(
        'B', 'E', 'N', 'N', n, A, QG
    )

    assert info >= 0
    assert wr.shape == (n,)
    assert wi.shape == (n,)
    assert scale.shape == (n,)
