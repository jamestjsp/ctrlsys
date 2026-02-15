"""
Tests for TG01LY - Finite-infinite decomposition of structured descriptor system.

Reduces a regular pole pencil A-lambda*E from structured form:
    A = [ A11 A12 A13 ]     E = [ E11  0  0 ]
        [ A21 A22 A23 ]         [  0   0  0 ]
        [ A31  0   0  ]         [  0   0  0 ]

to finite-infinite separated form:
    Q'*A*Z = [ Af  *  ]     Q'*E*Z = [ Ef  *  ]
             [ 0   Ai ]              [ 0   Ei ]
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_tg01ly_basic():
    """
    Test basic finite-infinite decomposition with structured input.

    Creates a descriptor system with 4 states, 1 finite eigenvalue, 1 infinite.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import tg01ly

    np.random.seed(42)

    n = 4
    m = 2
    p = 2
    ranke = 2
    rnka22 = 1

    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 1.0
    a[0, 1] = 2.0
    a[0, 2] = 0.5
    a[0, 3] = 0.3
    a[1, 0] = 0.0
    a[1, 1] = 3.0
    a[1, 2] = 0.7
    a[1, 3] = 0.2
    a[2, 0] = 0.1
    a[2, 1] = 0.0
    a[2, 2] = 2.0
    a[2, 3] = 0.4
    a[3, 0] = 0.5
    a[3, 1] = 0.0
    a[3, 2] = 0.0
    a[3, 3] = 0.0

    e = np.zeros((n, n), order='F', dtype=float)
    e[0, 0] = 1.0
    e[0, 1] = 0.5
    e[1, 1] = 2.0

    b = np.array([
        [1.0, 0.5],
        [0.0, 1.0],
        [0.3, 0.2],
        [0.1, 0.4]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, 0.5, 0.2],
        [0.0, 1.0, 0.3, 0.1]
    ], order='F', dtype=float)

    compq = True
    compz = True
    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    result = tg01ly(compq, compz, ranke, rnka22, a, e, b, c, q, z)

    a_out, e_out, b_out, c_out, q_out, z_out, nf, niblck, iblck, info = result

    assert info == 0, f"tg01ly failed with info={info}"
    assert nf >= 0 and nf <= n
    assert niblck >= 0

    qtq = q_out.T @ q_out
    ztz = z_out.T @ z_out
    assert_allclose(qtq, np.eye(n), rtol=1e-14, atol=1e-14)
    assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14)


def test_tg01ly_ranke_equals_n():
    """
    Test quick return when ranke = n (no infinite eigenvalues).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import tg01ly

    np.random.seed(123)

    n = 3
    m = 1
    p = 1
    ranke = n
    rnka22 = 0

    a = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.5, 0.2],
        [0.0, 2.0, 0.3],
        [0.0, 0.0, 3.0]
    ], order='F', dtype=float)

    b = np.array([[1.0], [2.0], [3.0]], order='F', dtype=float)
    c = np.array([[1.0, 2.0, 3.0]], order='F', dtype=float)

    compq = False
    compz = False
    q = np.zeros((1, 1), order='F', dtype=float)
    z = np.zeros((1, 1), order='F', dtype=float)

    result = tg01ly(compq, compz, ranke, rnka22, a, e, b, c, q, z)

    a_out, e_out, b_out, c_out, q_out, z_out, nf, niblck, iblck, info = result

    assert info == 0
    assert nf == ranke
    assert niblck == 0


def test_tg01ly_workspace_query():
    """
    Test workspace query functionality (ldwork = -1).
    """
    from ctrlsys import tg01ly

    n = 5
    m = 2
    p = 2
    ranke = 2
    rnka22 = 1

    a = np.zeros((n, n), order='F', dtype=float)
    e = np.zeros((n, n), order='F', dtype=float)
    b = np.zeros((n, m), order='F', dtype=float)
    c = np.zeros((p, n), order='F', dtype=float)
    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    result = tg01ly(True, True, ranke, rnka22, a, e, b, c, q, z, ldwork=-1)

    a_out, e_out, b_out, c_out, q_out, z_out, nf, niblck, iblck, info = result


def test_tg01ly_transformation_properties():
    """
    Validate orthogonal transformation properties: Q'Q = I, Z'Z = I.

    Uses a well-conditioned regular pencil for reliable convergence.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import tg01ly

    np.random.seed(456)

    n = 4
    m = 2
    p = 2
    ranke = 2
    rnka22 = 1

    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 2.0
    a[0, 1] = 1.0
    a[0, 2] = 0.5
    a[0, 3] = 0.3
    a[1, 0] = 0.0
    a[1, 1] = 3.0
    a[1, 2] = 0.7
    a[1, 3] = 0.4
    a[2, 0] = 0.1
    a[2, 1] = 0.0
    a[2, 2] = 2.5
    a[2, 3] = 0.5
    a[3, 0] = 0.4
    a[3, 1] = 0.0
    a[3, 2] = 0.0
    a[3, 3] = 0.0

    e = np.zeros((n, n), order='F', dtype=float)
    e[0, 0] = 1.0
    e[0, 1] = 0.3
    e[1, 1] = 2.0

    b = np.array([
        [1.0, 0.5],
        [0.0, 1.0],
        [0.3, 0.2],
        [0.1, 0.4]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, 0.5, 0.2],
        [0.0, 1.0, 0.3, 0.1]
    ], order='F', dtype=float)

    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    result = tg01ly(True, True, ranke, rnka22, a, e, b, c, q, z)

    a_out, e_out, b_out, c_out, q_out, z_out, nf, niblck, iblck, info = result

    assert info == 0, f"tg01ly returned info={info}, expected 0"

    qtq = q_out.T @ q_out
    ztz = z_out.T @ z_out
    assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)
    assert_allclose(ztz, np.eye(n), rtol=1e-13, atol=1e-14)

    assert nf >= 0 and nf <= n
    assert niblck >= 0


def test_tg01ly_block_structure():
    """
    Validate staircase block structure in output.

    The output should have zeros in Ai below the block diagonal.
    Uses well-conditioned regular pencil.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tg01ly

    np.random.seed(789)

    n = 5
    m = 2
    p = 2
    ranke = 2
    rnka22 = 2

    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 1.0
    a[0, 1] = 0.5
    a[0, 2] = 0.3
    a[0, 3] = 0.2
    a[0, 4] = 0.1
    a[1, 1] = 2.0
    a[1, 2] = 0.4
    a[1, 3] = 0.25
    a[1, 4] = 0.15
    a[2, 0] = 0.1
    a[2, 2] = 3.0
    a[2, 3] = 0.5
    a[2, 4] = 0.2
    a[3, 0] = 0.2
    a[3, 3] = 2.5
    a[3, 4] = 0.3
    a[4, 0] = 0.3

    e = np.zeros((n, n), order='F', dtype=float)
    e[0, 0] = 1.0
    e[0, 1] = 0.2
    e[1, 1] = 2.0

    b = np.array([
        [1.0, 0.5],
        [0.0, 1.0],
        [0.3, 0.2],
        [0.1, 0.4],
        [0.2, 0.1]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 0.0, 0.5, 0.2, 0.1],
        [0.0, 1.0, 0.3, 0.1, 0.05]
    ], order='F', dtype=float)

    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    result = tg01ly(True, True, ranke, rnka22, a, e, b, c, q, z)

    a_out, e_out, b_out, c_out, q_out, z_out, nf, niblck, iblck, info = result

    assert info == 0, f"tg01ly returned info={info}, expected 0"
    assert nf >= 0

    if nf < n:
        for i in range(nf, n):
            for j in range(nf):
                assert_allclose(a_out[i, j], 0.0, atol=1e-11)


def test_tg01ly_invalid_params():
    """
    Test error handling for invalid parameters.
    """
    from ctrlsys import tg01ly

    n = 3
    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, 1), order='F', dtype=float)
    c = np.ones((1, n), order='F', dtype=float)
    q = np.eye(n, order='F', dtype=float)
    z = np.eye(n, order='F', dtype=float)

    result = tg01ly(True, True, n + 1, 0, a, e, b, c, q, z)
    _, _, _, _, _, _, nf, niblck, iblck, info = result
    assert info < 0


def test_tg01ly_no_accumulate():
    """
    Test with COMPQ=False, COMPZ=False (no transformation accumulation).

    Random seed: 321 (for reproducibility)
    """
    from ctrlsys import tg01ly

    np.random.seed(321)

    n = 4
    m = 1
    p = 1
    ranke = 2
    rnka22 = 1

    a = np.zeros((n, n), order='F', dtype=float)
    a[0, 0] = 1.0
    a[0, 1] = 0.5
    a[0, 2] = 0.3
    a[0, 3] = 0.2
    a[1, 1] = 2.0
    a[1, 2] = 0.4
    a[1, 3] = 0.25
    a[2, 0] = 0.15
    a[2, 2] = 1.5
    a[2, 3] = 0.35
    a[3, 0] = 0.2

    e = np.zeros((n, n), order='F', dtype=float)
    e[0, 0] = 1.0
    e[0, 1] = 0.2
    e[1, 1] = 1.5

    b = np.random.randn(n, m).astype(float, order='F')
    c = np.random.randn(p, n).astype(float, order='F')

    q = np.zeros((1, 1), order='F', dtype=float)
    z = np.zeros((1, 1), order='F', dtype=float)

    result = tg01ly(False, False, ranke, rnka22, a, e, b, c, q, z)

    a_out, e_out, b_out, c_out, q_out, z_out, nf, niblck, iblck, info = result

    assert info == 0
    assert nf >= 0
