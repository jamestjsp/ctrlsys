"""
Tests for SB03MU: Discrete Sylvester equation solver for small matrices.

Solves: ISGN*op(TL)*X*op(TR) - X = SCALE*B
where TL is N1-by-N1, TR is N2-by-N2, N1,N2 in {0,1,2}.

Tests:
1. 1x1 case (scalar equation)
2. 2x2 case (full Gaussian elimination)
3. 1x2 and 2x1 cases
4. Transpose operations
5. Sign variation (ISGN=1 and -1)
6. Equation residual validation

Random seed: 42, 123, 456, 789, 888 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb03mu_1x1_basic():
    """
    Validate 1x1 case: SGN*TL11*X*TR11 - X = SCALE*B11.

    Simple scalar equation: (SGN*tl*tr - 1)*x = scale*b
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03mu

    tl = np.array([[2.0]], order='F', dtype=float)
    tr = np.array([[3.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(False, False, 1, tl, tr, b)

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify equation: 1*TL*X*TR - X = scale*B
    residual = tl @ x @ tr - x - scale * b
    assert_allclose(residual, 0.0, atol=1e-14)

    # Verify xnorm is abs(x)
    assert_allclose(xnorm, np.abs(x[0, 0]), rtol=1e-14)


def test_sb03mu_1x1_negative_sign():
    """
    Validate 1x1 case with ISGN=-1: -TL*X*TR - X = SCALE*B.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03mu

    tl = np.array([[2.0]], order='F', dtype=float)
    tr = np.array([[3.0]], order='F', dtype=float)
    b = np.array([[5.0]], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(False, False, -1, tl, tr, b)

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify equation: -TL*X*TR - X = scale*B
    residual = -tl @ x @ tr - x - scale * b
    assert_allclose(residual, 0.0, atol=1e-14)


def test_sb03mu_2x2_basic():
    """
    Validate 2x2 case with full 4x4 system solved by Gaussian elimination.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03mu

    np.random.seed(456)

    tl = np.array([
        [2.0, 0.3],
        [0.4, 1.5]
    ], order='F', dtype=float)

    tr = np.array([
        [1.8, 0.2],
        [0.1, 2.2]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.5],
        [0.3, 0.8]
    ], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(False, False, 1, tl, tr, b)

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify equation: TL*X*TR - X = scale*B
    residual = tl @ x @ tr - x - scale * b
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)


def test_sb03mu_1x2_case():
    """
    Validate 1x2 case: TL11*[X11 X12]*op(TR) - [X11 X12] = scale*[B11 B12].

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03mu

    tl = np.array([[2.5]], order='F', dtype=float)

    tr = np.array([
        [1.5, 0.3],
        [0.2, 1.8]
    ], order='F', dtype=float)

    b = np.array([[1.0, 0.5]], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(False, False, 1, tl, tr, b)

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify equation: TL*X*TR - X = scale*B
    residual = tl @ x @ tr - x - scale * b
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)


def test_sb03mu_2x1_case():
    """
    Validate 2x1 case: op(TL)*[X11; X21]*TR11 - [X11; X21] = scale*[B11; B21].

    Random seed: 888 (for reproducibility)
    """
    from ctrlsys import sb03mu

    tl = np.array([
        [2.0, 0.4],
        [0.3, 1.6]
    ], order='F', dtype=float)

    tr = np.array([[2.5]], order='F', dtype=float)

    b = np.array([[1.0], [0.7]], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(False, False, 1, tl, tr, b)

    assert info == 0
    assert scale > 0 and scale <= 1.0

    # Verify equation: TL*X*TR - X = scale*B
    residual = tl @ x @ tr - x - scale * b
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)


def test_sb03mu_transpose_left():
    """
    Validate with LTRANL=True: TL'*X*TR - X = scale*B.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03mu

    tl = np.array([
        [2.0, 0.3],
        [0.4, 1.5]
    ], order='F', dtype=float)

    tr = np.array([
        [1.8, 0.2],
        [0.1, 2.2]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.5],
        [0.3, 0.8]
    ], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(True, False, 1, tl, tr, b)

    assert info == 0

    # Verify equation: TL'*X*TR - X = scale*B
    residual = tl.T @ x @ tr - x - scale * b
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)


def test_sb03mu_transpose_right():
    """
    Validate with LTRANR=True: TL*X*TR' - X = scale*B.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03mu

    tl = np.array([
        [2.0, 0.3],
        [0.4, 1.5]
    ], order='F', dtype=float)

    tr = np.array([
        [1.8, 0.2],
        [0.1, 2.2]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.5],
        [0.3, 0.8]
    ], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(False, True, 1, tl, tr, b)

    assert info == 0

    # Verify equation: TL*X*TR' - X = scale*B
    residual = tl @ x @ tr.T - x - scale * b
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)


def test_sb03mu_both_transpose():
    """
    Validate with both transposes: TL'*X*TR' - X = scale*B.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03mu

    tl = np.array([
        [2.0, 0.3],
        [0.4, 1.5]
    ], order='F', dtype=float)

    tr = np.array([
        [1.8, 0.2],
        [0.1, 2.2]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.5],
        [0.3, 0.8]
    ], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(True, True, 1, tl, tr, b)

    assert info == 0

    # Verify equation: TL'*X*TR' - X = scale*B
    residual = tl.T @ x @ tr.T - x - scale * b
    assert_allclose(residual, np.zeros_like(residual), atol=1e-13)


def test_sb03mu_zero_dimensions():
    """
    Validate N1=0 or N2=0 returns immediately with xnorm=0.
    """
    from ctrlsys import sb03mu

    # Empty TL (N1=0)
    tl = np.array([], order='F', dtype=float).reshape(0, 0)
    tr = np.array([[2.0]], order='F', dtype=float)
    b = np.array([], order='F', dtype=float).reshape(0, 1)

    x, scale, xnorm, info = sb03mu(False, False, 1, tl, tr, b)

    assert info == 0
    assert xnorm == 0.0
    assert x.shape == (0, 1)


def test_sb03mu_singular():
    """
    Validate behavior when eigenvalue product exactly equals 1 (singular system).

    When ISGN*TL*TR = 1, the coefficient tau1 = ISGN*TL*TR - 1 = 0,
    which is below SMLNUM threshold and triggers info=1.
    """
    from ctrlsys import sb03mu

    # TL=1, TR=1 => TL*TR-1 = 0, exactly singular!
    tl = np.array([[1.0]], order='F', dtype=float)
    tr = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(False, False, 1, tl, tr, b)

    # Exactly singular case should return info=1 (warning)
    assert info == 1


def test_sb03mu_xnorm_2x2():
    """
    Validate xnorm computation for 2x2 case.

    xnorm is the infinity norm: max over rows of sum of absolute column values.
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03mu

    tl = np.array([
        [3.0, 0.5],
        [0.2, 2.5]
    ], order='F', dtype=float)

    tr = np.array([
        [2.0, 0.3],
        [0.1, 1.8]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.5],
        [0.3, 0.8]
    ], order='F', dtype=float)

    x, scale, xnorm, info = sb03mu(False, False, 1, tl, tr, b)

    assert info == 0

    # xnorm = max(|x11|+|x12|, |x21|+|x22|) for 2x2
    expected_xnorm = max(
        np.abs(x[0, 0]) + np.abs(x[0, 1]),
        np.abs(x[1, 0]) + np.abs(x[1, 1])
    )
    assert_allclose(xnorm, expected_xnorm, rtol=1e-14)
