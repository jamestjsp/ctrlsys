"""
Tests for MB04QC: Apply symplectic block reflector to matrices.

Applies the orthogonal symplectic block reflector Q or Q^T to a real
2m-by-n matrix [op(A); op(B)] from the left.

Uses numpy only - no scipy.
"""

import numpy as np
from numpy.testing import assert_allclose
from fortran_reference import run_fortran_driver


def _fortran_matrix_assignment(name, matrix):
    values = np.asarray(matrix, order="F").ravel(order="F")
    chunks = [
        ", ".join(f"{value:.17e}".replace("e", "d") for value in values[i:i + 4])
        for i in range(0, len(values), 4)
    ]
    return (
        f"  {name} = reshape((/ &\n"
        + ", &\n".join(f"       {chunk}" for chunk in chunks)
        + f" &\n  /), shape({name}))\n"
    )


def _nontrivial_reflector_case():
    m, n, k = 5, 4, 2
    i = np.arange(1, m + 1, dtype=float)[:, None]
    j = np.arange(1, k + 1, dtype=float)[None, :]
    v = 0.1 * (i + 2*j)
    w = 0.2 * (2*i - j)

    ii = np.arange(1, k + 1, dtype=float)[:, None]
    jj_rs = np.arange(1, 6*k + 1, dtype=float)[None, :]
    rs = 0.01 * (ii + jj_rs)
    jj_t = np.arange(1, 9*k + 1, dtype=float)[None, :]
    t = 0.005 * (2*ii - jj_t)

    ia = np.arange(1, m + 1, dtype=float)[:, None]
    ja = np.arange(1, n + 1, dtype=float)[None, :]
    a = np.sin(ia + ja)
    b = np.cos(ia - ja)
    return (
        m, n, k,
        v.astype(float, order="F"),
        w.astype(float, order="F"),
        rs.astype(float, order="F"),
        t.astype(float, order="F"),
        a.astype(float, order="F"),
        b.astype(float, order="F"),
    )


def _mb04qc_nontrivial_fortran_source():
    m, n, k, v, w, rs, t, a, b = _nontrivial_reflector_case()
    ldwork = 9 * n * k
    source = f"""
program main
  implicit none
  integer, parameter :: m={m}, n={n}, k={k}, ldv=m, ldw=m
  integer, parameter :: ldrs=k, ldt=k, lda=m, ldb=m, ldwork={ldwork}
  double precision v(ldv,k), w(ldw,k), rs(ldrs,6*k), t(ldt,9*k)
  double precision a(lda,n), b(ldb,n), dwork(ldwork)
"""
    source += _fortran_matrix_assignment("v", v)
    source += _fortran_matrix_assignment("w", w)
    source += _fortran_matrix_assignment("rs", rs)
    source += _fortran_matrix_assignment("t", t)
    source += _fortran_matrix_assignment("a", a)
    source += _fortran_matrix_assignment("b", b)
    source += """
  call MB04QC('N','N','N','N','F','C','C',m,n,k,v,ldv,w,ldw,rs,ldrs,t,ldt,a,lda,b,ldb,dwork)
  print '(*(ES24.16,1X))', a
  print '(*(ES24.16,1X))', b
end program main
"""
    return source


def test_mb04qc_nontrivial_reflector_matches_fortran_reference(tmp_path):
    from ctrlsys import mb04qc

    m, n, k, v, w, rs, t, a, b = _nontrivial_reflector_case()
    output = run_fortran_driver(_mb04qc_nontrivial_fortran_source(), tmp_path)
    values = np.array(output.split(), dtype=float)
    a_f = values[:m*n].reshape((m, n), order="F")
    b_f = values[m*n:].reshape((m, n), order="F")

    a_out, b_out = mb04qc(
        "N", "N", "N", "N", "F", "C", "C",
        m, n, k, v, w, rs, t, a.copy(order="F"), b.copy(order="F")
    )

    assert_allclose(a_out, a_f, rtol=1e-12, atol=1e-12)
    assert_allclose(b_out, b_f, rtol=1e-12, atol=1e-12)


def test_mb04qc_basic():
    """
    Test MB04QC with identity-like block reflector.

    When R, S, T are zero matrices, the reflector is identity.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(42)
    m, n, k = 4, 3, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')
    A_orig = A.copy()
    B_orig = B.copy()

    A_copy = A.copy()
    B_copy = B.copy()
    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A_copy, B_copy
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)

    np.testing.assert_allclose(A_out, A_orig, atol=1e-12)
    np.testing.assert_allclose(B_out, B_orig, atol=1e-12)


def test_mb04qc_transpose_q():
    """
    Test MB04QC with TRANQ='T' (apply Q^T).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(123)
    m, n, k = 4, 3, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'T', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_transpose_a():
    """
    Test MB04QC with TRANA='T' (A stored transposed).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(456)
    m, n, k = 4, 3, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A_T = np.random.randn(n, m).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'T', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A_T.copy(), B.copy()
    )

    assert A_out.shape == (n, m)
    assert B_out.shape == (m, n)


def test_mb04qc_strab_n():
    """
    Test MB04QC with STRAB='N' (no zero structure).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(789)
    m, n, k = 5, 4, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i:, i] = np.random.randn(m - i)
        W[i:, i] = np.random.randn(m - i)

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'N', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_rowwise_storage():
    """
    Test MB04QC with STOREV='R' and STOREW='R'.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(111)
    m, n, k = 4, 3, 2

    V = np.zeros((k, m), order='F', dtype=float)
    W = np.zeros((k, m), order='F', dtype=float)
    for i in range(k):
        V[i, i] = 1.0
        W[i, i] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'R', 'R', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_k_equals_1():
    """
    Test MB04QC with K=1 (single reflector).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(222)
    m, n, k = 5, 4, 1

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    V[0, 0] = 1.0
    W[0, 0] = 1.0

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_m_equals_k():
    """
    Test MB04QC with M=K (degenerate case).

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import mb04qc

    np.random.seed(333)
    m, n, k = 3, 4, 3

    V = np.eye(m, k, order='F', dtype=float)
    W = np.eye(m, k, order='F', dtype=float)

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)


def test_mb04qc_n_equals_zero():
    """
    Test MB04QC with N=0 (quick return).
    """
    from ctrlsys import mb04qc

    m, n, k = 4, 0, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)

    rs = np.zeros((k, 6*k), order='F', dtype=float)
    t = np.zeros((k, 9*k), order='F', dtype=float)

    A = np.zeros((m, 1), order='F', dtype=float)
    B = np.zeros((m, 1), order='F', dtype=float)

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == A.shape


def test_mb04qc_with_mb04qf():
    """
    Test MB04QC integration with MB04QF block factors.

    MB04QF computes the RS and T factors, MB04QC applies them.
    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import mb04qf, mb04qc

    np.random.seed(444)
    m, n, k = 5, 4, 2

    V = np.zeros((m, k), order='F', dtype=float)
    W = np.zeros((m, k), order='F', dtype=float)
    for i in range(k):
        V[i:, i] = np.random.randn(m - i)
        W[i:, i] = np.random.randn(m - i)
        W[i, i] = 0.0

    cs = np.zeros(2*k, order='F', dtype=float)
    for i in range(k):
        theta = np.random.uniform(0, 2*np.pi)
        cs[2*i] = np.cos(theta)
        cs[2*i + 1] = np.sin(theta)

    tau = np.zeros(k, order='F', dtype=float)

    rs, t, info_qf = mb04qf('F', 'C', 'C', m, k, V.copy(), W.copy(), cs, tau)
    assert info_qf == 0

    A = np.random.randn(m, n).astype(float, order='F')
    B = np.random.randn(m, n).astype(float, order='F')
    A_orig = A.copy()
    B_orig = B.copy()

    A_out, B_out = mb04qc(
        'Z', 'N', 'N', 'N', 'F', 'C', 'C', m, n, k, V, W, rs, t, A.copy(), B.copy()
    )

    assert A_out.shape == (m, n)
    assert B_out.shape == (m, n)

    A_round, B_round = mb04qc(
        'Z', 'N', 'N', 'T', 'F', 'C', 'C', m, n, k, V, W, rs, t, A_out.copy(), B_out.copy()
    )
    np.testing.assert_allclose(A_round, A_orig, atol=1e-12)
    np.testing.assert_allclose(B_round, B_orig, atol=1e-12)
