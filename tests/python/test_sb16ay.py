"""
Tests for SB16AY - Cholesky factors of frequency-weighted controllability
and observability Grammians for controller reduction.

SB16AY computes for given state-space representations (A,B,C,D) and
(Ac,Bc,Cc,Dc) of the open-loop system G and feedback controller K,
the Cholesky factors S and R of the frequency-weighted Grammians
P = S*S' (controllability) and Q = R'*R (observability).

The controller must stabilize the closed-loop system.
Ac must be in block-diagonal real Schur form: Ac = diag(Ac1, Ac2),
where Ac1 contains unstable eigenvalues and Ac2 contains stable eigenvalues.
"""
import numpy as np
import pytest
import slicot


def test_no_weighting_continuous():
    """
    Test WEIGHT='N' for continuous-time.

    When no weighting is used:
    - Controllability Grammian P solves: Ac2*P + P*Ac2' + scalec^2*Bc2*Bc2' = 0
    - Observability Grammian Q solves: Ac2'*Q + Q*Ac2 + scaleo^2*Cc2'*Cc2 = 0

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n, m, p = 2, 1, 1
    nc, ncs = 3, 3

    a = np.array([[-1.0, 0.5], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([
        [-0.5, 0.1, 0.0],
        [0.0, -1.0, 0.2],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.5], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[0.5, 0.3, 0.2]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    scalec, scaleo, s, r, info = slicot.sb16ay(
        dico='C', jobc='S', jobo='S', weight='N',
        n=n, m=m, p=p, nc=nc, ncs=ncs,
        a=a, b=b, c=c, d=d,
        ac=ac, bc=bc, cc=cc, dc=dc
    )

    assert info == 0
    assert scalec > 0
    assert scaleo > 0

    s_upper = np.triu(s)
    r_upper = np.triu(r)
    p_gram = s_upper @ s_upper.T
    q_gram = r_upper.T @ r_upper

    bc2 = bc
    cc2 = cc

    res_c = ac @ p_gram + p_gram @ ac.T + scalec**2 * bc2 @ bc2.T
    res_o = ac.T @ q_gram + q_gram @ ac + scaleo**2 * cc2.T @ cc2

    np.testing.assert_allclose(res_c, 0.0, atol=1e-10)
    np.testing.assert_allclose(res_o, 0.0, atol=1e-10)


def test_no_weighting_discrete():
    """
    Test WEIGHT='N' for discrete-time.

    When no weighting is used:
    - Controllability Grammian P solves: Ac2*P*Ac2' - P + scalec^2*Bc2*Bc2' = 0
    - Observability Grammian Q solves: Ac2'*Q*Ac2 - Q + scaleo^2*Cc2'*Cc2 = 0

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n, m, p = 2, 1, 1
    nc, ncs = 3, 3

    a = np.array([[0.5, 0.1], [0.0, 0.4]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([
        [0.3, 0.1, 0.0],
        [0.0, 0.4, 0.1],
        [0.0, 0.0, 0.2]
    ], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.5], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[0.5, 0.3, 0.2]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    scalec, scaleo, s, r, info = slicot.sb16ay(
        dico='D', jobc='S', jobo='S', weight='N',
        n=n, m=m, p=p, nc=nc, ncs=ncs,
        a=a, b=b, c=c, d=d,
        ac=ac, bc=bc, cc=cc, dc=dc
    )

    assert info == 0
    assert scalec > 0
    assert scaleo > 0

    s_upper = np.triu(s)
    r_upper = np.triu(r)
    p_gram = s_upper @ s_upper.T
    q_gram = r_upper.T @ r_upper

    bc2 = bc
    cc2 = cc

    res_c = ac @ p_gram @ ac.T - p_gram + scalec**2 * bc2 @ bc2.T
    res_o = ac.T @ q_gram @ ac - q_gram + scaleo**2 * cc2.T @ cc2

    np.testing.assert_allclose(res_c, 0.0, atol=1e-10)
    np.testing.assert_allclose(res_o, 0.0, atol=1e-10)


def test_zero_ncs():
    """
    Test with NCS=0 (quick return).
    """
    n, m, p = 2, 1, 1
    nc, ncs = 2, 0

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([[1.5, 0.1], [0.0, 2.0]], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    scalec, scaleo, s, r, info = slicot.sb16ay(
        dico='C', jobc='S', jobo='S', weight='N',
        n=n, m=m, p=p, nc=nc, ncs=ncs,
        a=a, b=b, c=c, d=d,
        ac=ac, bc=bc, cc=cc, dc=dc
    )

    assert info == 0
    assert scalec == pytest.approx(1.0)
    assert scaleo == pytest.approx(1.0)


def test_invalid_dico():
    """Test invalid DICO parameter returns info=-1."""
    n, m, p = 2, 1, 1
    nc, ncs = 2, 2

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)
    ac = np.array([[-0.5, 0.0], [0.0, -1.0]], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    scalec, scaleo, s, r, info = slicot.sb16ay(
        dico='X', jobc='S', jobo='S', weight='N',
        n=n, m=m, p=p, nc=nc, ncs=ncs,
        a=a, b=b, c=c, d=d,
        ac=ac, bc=bc, cc=cc, dc=dc
    )

    assert info == -1


def test_invalid_ncs():
    """Test NCS > NC returns info=-9."""
    n, m, p = 2, 1, 1
    nc, ncs = 2, 3

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)
    ac = np.array([[-0.5, 0.0], [0.0, -1.0]], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    scalec, scaleo, s, r, info = slicot.sb16ay(
        dico='C', jobc='S', jobo='S', weight='N',
        n=n, m=m, p=p, nc=nc, ncs=ncs,
        a=a, b=b, c=c, d=d,
        ac=ac, bc=bc, cc=cc, dc=dc
    )

    assert info == -9


def test_unstable_ac2_continuous():
    """
    Test that unstable Ac2 (stable part) returns info=5 for continuous.
    """
    n, m, p = 2, 1, 1
    nc, ncs = 2, 2

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([[1.0, 0.0], [0.0, 2.0]], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    scalec, scaleo, s, r, info = slicot.sb16ay(
        dico='C', jobc='S', jobo='S', weight='N',
        n=n, m=m, p=p, nc=nc, ncs=ncs,
        a=a, b=b, c=c, d=d,
        ac=ac, bc=bc, cc=cc, dc=dc
    )

    assert info == 5


def test_cholesky_symmetry_continuous():
    """
    Test that S*S' is symmetric positive semi-definite (controllability)
    and R'*R is symmetric positive semi-definite (observability).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n, m, p = 2, 1, 1
    nc, ncs = 4, 4

    a = np.array([[-1.0, 0.2], [-0.1, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.5]], order='F', dtype=np.float64)
    c = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([
        [-0.5, 0.1, 0.0, 0.0],
        [0.0, -0.8, 0.1, 0.0],
        [0.0, 0.0, -1.2, 0.1],
        [0.0, 0.0, 0.0, -1.5]
    ], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.5], [0.3], [0.1]], order='F', dtype=np.float64)
    cc = np.array([[0.5, 0.4, 0.3, 0.2]], order='F', dtype=np.float64)
    dc = np.array([[0.05]], order='F', dtype=np.float64)

    scalec, scaleo, s, r, info = slicot.sb16ay(
        dico='C', jobc='S', jobo='S', weight='N',
        n=n, m=m, p=p, nc=nc, ncs=ncs,
        a=a, b=b, c=c, d=d,
        ac=ac, bc=bc, cc=cc, dc=dc
    )

    assert info == 0

    s_upper = np.triu(s)
    r_upper = np.triu(r)
    p_gram = s_upper @ s_upper.T
    q_gram = r_upper.T @ r_upper

    np.testing.assert_allclose(p_gram, p_gram.T, rtol=1e-14)
    np.testing.assert_allclose(q_gram, q_gram.T, rtol=1e-14)

    eig_p = np.linalg.eigvalsh(p_gram)
    eig_q = np.linalg.eigvalsh(q_gram)
    assert np.all(eig_p >= -1e-14)
    assert np.all(eig_q >= -1e-14)
