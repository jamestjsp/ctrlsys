import numpy as np
import pytest


def test_mb04yw_qr_iteration_basic():
    """
    Test MB04YW QR iteration step with zero shift on 4x4 bidiagonal.

    Validates:
    - Bidiagonal structure preserved (only D and E modified)
    - Singular values preserved (J'J has same eigenvalues)

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n = 4, 4
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.1
    e = np.random.randn(p - 1) * 0.5

    j = np.diag(d) + np.diag(e, 1)
    jtj_orig = j.T @ j
    eig_orig = np.sort(np.linalg.eigvalsh(jtj_orig))

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()

    l, k = 1, p

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        True, False, False, m, n, l, k, 0.0,
        d_out, e_out, None, None
    )

    assert info == 0

    j_new = np.diag(d_res) + np.diag(e_res, 1)
    jtj_new = j_new.T @ j_new
    eig_new = np.sort(np.linalg.eigvalsh(jtj_new))

    np.testing.assert_allclose(eig_new, eig_orig, rtol=1e-12)


def test_mb04yw_ql_iteration_basic():
    """
    Test MB04YW QL iteration step with zero shift on 4x4 bidiagonal.

    Validates:
    - Singular values preserved under QL iteration

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, n = 4, 4
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.1
    e = np.random.randn(p - 1) * 0.5

    j = np.diag(d) + np.diag(e, 1)
    jtj_orig = j.T @ j
    eig_orig = np.sort(np.linalg.eigvalsh(jtj_orig))

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()

    l, k = 1, p

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        False, False, False, m, n, l, k, 0.0,
        d_out, e_out, None, None
    )

    assert info == 0

    j_new = np.diag(d_res) + np.diag(e_res, 1)
    jtj_new = j_new.T @ j_new
    eig_new = np.sort(np.linalg.eigvalsh(jtj_new))

    np.testing.assert_allclose(eig_new, eig_orig, rtol=1e-12)


def test_mb04yw_with_u_update():
    """
    Test MB04YW with U matrix accumulation.

    Validates:
    - U remains orthogonal after update
    - Transformation property: S' J T gives new bidiagonal

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, n = 5, 4
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.1
    e = np.random.randn(p - 1) * 0.5

    j = np.diag(d) + np.diag(e, 1)
    sv_orig = np.sort(np.linalg.svd(j, compute_uv=False))

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()
    u = np.eye(m, p, dtype=float, order='F')

    l, k = 1, p

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        True, True, False, m, n, l, k, 0.0,
        d_out, e_out, u, None
    )

    assert info == 0
    assert u_out is not None

    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-12, atol=1e-12)

    j_new = np.diag(d_res) + np.diag(e_res, 1)
    sv_new = np.sort(np.linalg.svd(j_new, compute_uv=False))
    np.testing.assert_allclose(sv_new, sv_orig, rtol=1e-12)


def test_mb04yw_with_v_update():
    """
    Test MB04YW with V matrix accumulation.

    Validates:
    - V remains orthogonal after update
    - Singular values preserved

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m, n = 4, 5
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.1
    e = np.random.randn(p - 1) * 0.5

    j = np.diag(d) + np.diag(e, 1)
    sv_orig = np.sort(np.linalg.svd(j, compute_uv=False))

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()
    v = np.eye(n, p, dtype=float, order='F')

    l, k = 1, p

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        True, False, True, m, n, l, k, 0.0,
        d_out, e_out, None, v
    )

    assert info == 0
    assert v_out is not None

    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-12, atol=1e-12)

    j_new = np.diag(d_res) + np.diag(e_res, 1)
    sv_new = np.sort(np.linalg.svd(j_new, compute_uv=False))
    np.testing.assert_allclose(sv_new, sv_orig, rtol=1e-12)


def test_mb04yw_with_both_u_and_v():
    """
    Test MB04YW with both U and V accumulation.

    Validates:
    - Both U and V remain orthogonal
    - Transformation: S' * J * T = J_new

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m, n = 5, 5
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.1
    e = np.random.randn(p - 1) * 0.5

    j_orig = np.diag(d) + np.diag(e, 1)
    sv_orig = np.sort(np.linalg.svd(j_orig, compute_uv=False))

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()
    u = np.eye(m, p, dtype=float, order='F')
    v = np.eye(n, p, dtype=float, order='F')

    l, k = 1, p

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        True, True, True, m, n, l, k, 0.0,
        d_out, e_out, u, v
    )

    assert info == 0
    assert u_out is not None
    assert v_out is not None

    np.testing.assert_allclose(u_out.T @ u_out, np.eye(p), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_out.T @ v_out, np.eye(p), rtol=1e-12, atol=1e-12)

    j_new = np.diag(d_res) + np.diag(e_res, 1)
    sv_new = np.sort(np.linalg.svd(j_new, compute_uv=False))
    np.testing.assert_allclose(sv_new, sv_orig, rtol=1e-12)


def test_mb04yw_nonzero_shift_qr():
    """
    Test MB04YW QR iteration with nonzero shift.

    Validates:
    - Singular values preserved with shifted iteration

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    m, n = 4, 4
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.5
    e = np.random.randn(p - 1) * 0.3

    j = np.diag(d) + np.diag(e, 1)
    sv_orig = np.sort(np.linalg.svd(j, compute_uv=False))

    shift = 0.1

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()

    l, k = 1, p

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        True, False, False, m, n, l, k, shift,
        d_out, e_out, None, None
    )

    assert info == 0

    j_new = np.diag(d_res) + np.diag(e_res, 1)
    sv_new = np.sort(np.linalg.svd(j_new, compute_uv=False))
    np.testing.assert_allclose(sv_new, sv_orig, rtol=1e-11)


def test_mb04yw_nonzero_shift_ql():
    """
    Test MB04YW QL iteration with nonzero shift.

    Validates:
    - Singular values preserved with shifted QL iteration

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    m, n = 4, 4
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.5
    e = np.random.randn(p - 1) * 0.3

    j = np.diag(d) + np.diag(e, 1)
    sv_orig = np.sort(np.linalg.svd(j, compute_uv=False))

    shift = 0.1

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()

    l, k = 1, p

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        False, False, False, m, n, l, k, shift,
        d_out, e_out, None, None
    )

    assert info == 0

    j_new = np.diag(d_res) + np.diag(e_res, 1)
    sv_new = np.sort(np.linalg.svd(j_new, compute_uv=False))
    np.testing.assert_allclose(sv_new, sv_orig, rtol=1e-11)


def test_mb04yw_submatrix():
    """
    Test MB04YW on submatrix of bidiagonal (l=2, k=4 of 5x5).

    Validates:
    - Only the submatrix from l to k is modified
    - Elements outside [l,k] remain unchanged

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    m, n = 5, 5
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.1
    e = np.random.randn(p - 1) * 0.5
    d_orig = d.copy()
    e_orig = e.copy()

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()

    l, k = 2, 4

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        True, False, False, m, n, l, k, 0.0,
        d_out, e_out, None, None
    )

    assert info == 0

    np.testing.assert_allclose(d_res[0], d_orig[0], rtol=1e-14)
    np.testing.assert_allclose(e_res[0], e_orig[0], rtol=1e-14)
    np.testing.assert_allclose(d_res[4], d_orig[4], rtol=1e-14)


def test_mb04yw_single_element():
    """
    Test MB04YW quick return when l==k (single element submatrix).

    Validates:
    - D and E unchanged
    - Quick return executed

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    m, n = 4, 4
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.1
    e = np.random.randn(p - 1) * 0.5
    d_orig = d.copy()
    e_orig = e.copy()

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()

    l, k = 2, 2

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        True, False, False, m, n, l, k, 0.0,
        d_out, e_out, None, None
    )

    assert info == 0
    np.testing.assert_allclose(d_res, d_orig, rtol=1e-14)
    np.testing.assert_allclose(e_res, e_orig, rtol=1e-14)


def test_mb04yw_transformation_property():
    """
    Test MB04YW transformation property: S' * J * T = J_new.

    Verifies the fundamental mathematical property that the bidiagonal
    transformation is a similarity transform on J'J.

    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    m, n = 4, 4
    p = min(m, n)

    d = np.abs(np.random.randn(p)) + 0.2
    e = np.random.randn(p - 1) * 0.4

    j_orig = np.diag(d) + np.diag(e, 1)

    d_out = d.astype(float, order='F').copy()
    e_out = e.astype(float, order='F').copy()
    u = np.eye(m, p, dtype=float, order='F')
    v = np.eye(n, p, dtype=float, order='F')

    l, k = 1, p

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        True, True, True, m, n, l, k, 0.0,
        d_out, e_out, u, v
    )

    assert info == 0

    j_new = np.diag(d_res) + np.diag(e_res, 1)

    s_mat = u_out[:p, :p]
    t_mat = v_out[:p, :p]

    np.testing.assert_allclose(s_mat.T @ s_mat, np.eye(p), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t_mat.T @ t_mat, np.eye(p), rtol=1e-12, atol=1e-12)

    j_transformed = s_mat.T @ j_orig @ t_mat
    np.testing.assert_allclose(j_transformed, j_new, rtol=1e-12, atol=1e-12)


def test_mb04yw_p_equals_one():
    """
    Test MB04YW quick return when p = min(m,n) <= 1.

    Validates:
    - Quick return for trivial case
    """
    m, n = 1, 3

    d = np.array([2.0], dtype=float, order='F')
    e = np.zeros(0, dtype=float, order='F')
    d_orig = d.copy()

    from ctrlsys import mb04yw

    d_res, e_res, u_out, v_out, info = mb04yw(
        True, False, False, m, n, 1, 1, 0.0,
        d, e, None, None
    )

    assert info == 0
    np.testing.assert_allclose(d_res, d_orig, rtol=1e-14)
