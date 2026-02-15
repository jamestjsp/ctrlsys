import numpy as np
import pytest


def test_mb04ud_html_example():
    """
    Test MB04UD using HTML documentation example.

    From MB04UD.html:
    M=4, N=4, TOL=0.0

    Input A (row-wise): Fortran READ(NIN, FMT=*)((A(I,J), J=1,N), I=1,M)
    Input E (row-wise): Fortran READ(NIN, FMT=*)((E(I,J), J=1,N), I=1,M)

    Expected results:
    - Transformed A is upper triangular
    - Transformed E is in column echelon form (upper triangular for full rank)
    - rank(E) = 4
    - ISTAIR = [1, 2, 3, 4]
    """
    m, n = 4, 4

    a = np.array([
        [2.0, 0.0, 2.0, -2.0],
        [0.0, -2.0, 0.0, 2.0],
        [2.0, 0.0, -2.0, 0.0],
        [2.0, -2.0, 0.0, 2.0]
    ], dtype=float, order='F')

    e = np.array([
        [1.0, 0.0, 1.0, -1.0],
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0, 0.0],
        [1.0, -1.0, 0.0, 1.0]
    ], dtype=float, order='F')

    a_expected = np.array([
        [0.5164, 1.0328, 1.1547, -2.3094],
        [0.0000, -2.5820, 0.0000, -1.1547],
        [0.0000, 0.0000, -3.4641, 0.0000],
        [0.0000, 0.0000, 0.0000, -3.4641]
    ], dtype=float, order='F')

    e_expected = np.array([
        [0.2582, 0.5164, 0.5774, -1.1547],
        [0.0000, -1.2910, 0.0000, -0.5774],
        [0.0000, 0.0000, -1.7321, 0.0000],
        [0.0000, 0.0000, 0.0000, -1.7321]
    ], dtype=float, order='F')

    istair_expected = np.array([1, 2, 3, 4], dtype=np.int32)

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('N', 'N', m, n, a, e)

    assert info == 0
    assert ranke == 4
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(e_out, e_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(istair, istair_expected)


def test_mb04ud_with_q_and_z():
    """
    Test MB04UD with JOBQ='I' and JOBZ='I' to compute transformation matrices.

    Validates orthogonality: Q'*Q = I and Z'*Z = I
    And transformation: Q'*A*Z = A_out, Q'*E*Z = E_out

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n = 4, 4

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    a_orig = a.copy()
    e_orig = e.copy()

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('I', 'I', m, n, a, e)

    assert info == 0

    np.testing.assert_allclose(q_out.T @ q_out, np.eye(m), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z_out.T @ z_out, np.eye(n), rtol=1e-14, atol=1e-14)

    a_transformed = q_out.T @ a_orig @ z_out
    e_transformed = q_out.T @ e_orig @ z_out
    np.testing.assert_allclose(a_out, a_transformed, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e_transformed, rtol=1e-14, atol=1e-14)


def test_mb04ud_column_echelon_form():
    """
    Test MB04UD produces correct column echelon form.

    Column echelon form properties:
    - First (N-r) columns are zero where r = rank(E)
    - Last nonzero element in each remaining column has increasing row indices

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, n = 5, 4

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('N', 'N', m, n, a, e)

    assert info == 0
    assert 0 <= ranke <= min(m, n)

    for j in range(n - ranke):
        col_norm = np.linalg.norm(e_out[:, j])
        assert col_norm < 1e-10, f"Column {j} should be zero but norm={col_norm}"


def test_mb04ud_rank_deficient():
    """
    Test MB04UD with rank-deficient E matrix.

    E = [1, 2; 2, 4; 3, 6] has rank 1.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, n = 3, 2

    a = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ], dtype=float, order='F')

    e = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0]
    ], dtype=float, order='F')

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('I', 'I', m, n, a, e)

    assert info == 0
    assert ranke == 1

    assert np.linalg.norm(e_out[:, 0]) < 1e-10


def test_mb04ud_update_q():
    """
    Test MB04UD with JOBQ='U' to update existing Q matrix.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m, n = 4, 3

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    q_init = np.eye(m, dtype=float, order='F')

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('U', 'N', m, n, a, e, q=q_init)

    assert info == 0
    np.testing.assert_allclose(q_out.T @ q_out, np.eye(m), rtol=1e-14, atol=1e-14)


def test_mb04ud_update_z():
    """
    Test MB04UD with JOBZ='U' to update existing Z matrix.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    m, n = 3, 4

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    z_init = np.eye(n, dtype=float, order='F')

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('N', 'U', m, n, a, e, z=z_init)

    assert info == 0
    np.testing.assert_allclose(z_out.T @ z_out, np.eye(n), rtol=1e-14, atol=1e-14)


def test_mb04ud_zero_dimensions():
    """
    Test MB04UD with M=0 or N=0 (quick return).
    """
    from ctrlsys import mb04ud

    a = np.zeros((1, 1), dtype=float, order='F')
    e = np.zeros((1, 1), dtype=float, order='F')
    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('N', 'N', 0, 0, a, e)
    assert info == 0
    assert ranke == 0


def test_mb04ud_wide_matrix():
    """
    Test MB04UD with wide matrix (M < N).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m, n = 3, 5

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('I', 'I', m, n, a, e)

    assert info == 0
    assert q_out.shape == (m, m)
    assert z_out.shape == (n, n)
    assert istair.shape == (m,)
    assert 0 <= ranke <= min(m, n)


def test_mb04ud_tall_matrix():
    """
    Test MB04UD with tall matrix (M > N).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    m, n = 5, 3

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('I', 'I', m, n, a, e)

    assert info == 0
    assert q_out.shape == (m, m)
    assert z_out.shape == (n, n)
    assert istair.shape == (m,)
    assert 0 <= ranke <= min(m, n)


def test_mb04ud_istair_structure():
    """
    Test MB04UD ISTAIR array structure.

    ISTAIR(i) = +j if E(i,j) is a corner point
    ISTAIR(i) = -j if E(i,j) is on boundary but not a corner

    For full-rank square matrix: all entries positive (corners).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    m, n = 4, 4

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('N', 'N', m, n, a, e)

    assert info == 0

    if ranke == min(m, n):
        for i in range(m - ranke, m):
            assert istair[i] > 0, f"ISTAIR[{i}] should be positive (corner)"


def test_mb04ud_tolerance():
    """
    Test MB04UD with custom tolerance.

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    m, n = 4, 4

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('N', 'N', m, n, a, e, tol=1e-8)

    assert info == 0
    assert 0 <= ranke <= min(m, n)


def test_mb04ud_frobenius_norm_preservation():
    """
    Test that orthogonal transformations preserve Frobenius norm.

    ||A||_F = ||Q'*A*Z||_F for orthogonal Q, Z.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    m, n = 4, 4

    a = np.random.randn(m, n).astype(float, order='F')
    e = np.random.randn(m, n).astype(float, order='F')

    a_norm_before = np.linalg.norm(a, 'fro')
    e_norm_before = np.linalg.norm(e, 'fro')

    from ctrlsys import mb04ud

    a_out, e_out, q_out, z_out, ranke, istair, info = mb04ud('N', 'N', m, n, a, e)

    assert info == 0

    a_norm_after = np.linalg.norm(a_out, 'fro')
    e_norm_after = np.linalg.norm(e_out, 'fro')

    np.testing.assert_allclose(a_norm_after, a_norm_before, rtol=1e-14)
    np.testing.assert_allclose(e_norm_after, e_norm_before, rtol=1e-14)
