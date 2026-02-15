import numpy as np
import pytest
from ctrlsys import md03by


def test_md03by_basic_gauss_newton():
    """Test basic Gauss-Newton case (PAR=0, acceptable step)."""
    n = 3

    # QR factorization from simple overdetermined system
    # R from QR of 5x3 matrix (upper triangular, nonincreasing diagonal)
    r = np.array([
        [10.0,  2.0,  1.0],
        [ 0.0,  8.0,  0.5],
        [ 0.0,  0.0,  5.0]
    ], dtype=float, order='F')
    r_orig = r.copy()

    ipvt = np.array([1, 2, 3], dtype=np.int32)
    diag = np.array([1.0, 1.0, 1.0], dtype=float)
    qtb = np.array([5.0, 4.0, 2.5], dtype=float)
    delta = 10.0  # Large trust radius - accept Gauss-Newton step
    par = 0.0
    tol = 0.0

    r_out, par_out, rank, x, rx, info = md03by(
        cond='N', n=n, r=r, ipvt=ipvt, diag=diag,
        qtb=qtb, delta=delta, par=par, rank=0, tol=tol
    )

    assert info == 0
    assert rank == 3

    # For large delta, should get Gauss-Newton solution (PAR=0)
    assert par_out == 0.0

    # x should satisfy R*x = qtb (use original R, not modified)
    r_x = r_orig @ x
    np.testing.assert_allclose(r_x, qtb, rtol=1e-12)

    # rx should be -R*P'*x = -R*x (identity permutation)
    expected_rx = -r_orig @ x
    np.testing.assert_allclose(rx, expected_rx, rtol=1e-12)

    # Check ||D*x|| <= delta
    dxnorm = np.linalg.norm(diag * x)
    assert dxnorm <= delta * 1.1


def test_md03by_small_trust_region():
    """Test with small trust region requiring PAR > 0."""
    n = 3

    r = np.array([
        [10.0,  2.0,  1.0],
        [ 0.0,  8.0,  0.5],
        [ 0.0,  0.0,  5.0]
    ], dtype=float, order='F')

    ipvt = np.array([1, 2, 3], dtype=np.int32)
    diag = np.array([1.0, 1.0, 1.0], dtype=float)
    qtb = np.array([10.0, 8.0, 5.0], dtype=float)
    delta = 0.5  # Small trust radius - forces damping
    par = 0.0
    tol = 0.0

    r_out, par_out, rank, x, rx, info = md03by(
        cond='N', n=n, r=r, ipvt=ipvt, diag=diag,
        qtb=qtb, delta=delta, par=par, rank=0, tol=tol
    )

    assert info == 0
    assert rank == 3

    # Should find PAR > 0 due to small trust radius
    assert par_out > 0.0

    # Check ||D*x|| ≈ delta (should be on trust region boundary)
    dxnorm = np.linalg.norm(diag * x)
    np.testing.assert_allclose(dxnorm, delta, rtol=0.11)  # 10% tolerance from algorithm


def test_md03by_with_permutation():
    """Test with non-identity permutation."""
    n = 3

    r = np.array([
        [12.0,  3.0,  2.0],
        [ 0.0,  9.0,  1.5],
        [ 0.0,  0.0,  6.0]
    ], dtype=float, order='F')

    ipvt = np.array([3, 1, 2], dtype=np.int32)  # Permutation
    diag = np.array([2.0, 1.5, 1.0], dtype=float)
    qtb = np.array([6.0, 4.5, 3.0], dtype=float)
    delta = 5.0
    par = 0.0
    tol = 0.0

    r_out, par_out, rank, x, rx, info = md03by(
        cond='N', n=n, r=r, ipvt=ipvt, diag=diag,
        qtb=qtb, delta=delta, par=par, rank=0, tol=tol
    )

    assert info == 0
    assert rank == 3

    # x should be permuted correctly
    # P*R*P'*(P*x) = Q'*b where P*x is the internal solution
    # Verify dimensions and basic properties
    assert x.shape == (n,)
    assert rx.shape == (n,)

    # Check ||D*x|| constraint
    dxnorm = np.linalg.norm(diag * x)
    assert dxnorm <= delta * 1.1


def test_md03by_condition_estimation():
    """Test with condition estimation (COND='E')."""
    n = 4

    r = np.array([
        [15.0,  3.0,  2.0,  1.0],
        [ 0.0, 10.0,  1.5,  0.8],
        [ 0.0,  0.0,  7.0,  0.5],
        [ 0.0,  0.0,  0.0,  4.0]
    ], dtype=float, order='F')

    ipvt = np.array([1, 2, 3, 4], dtype=np.int32)
    diag = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    qtb = np.array([7.5, 5.0, 3.5, 2.0], dtype=float)
    delta = 3.0
    par = 0.0
    tol = 1e-10

    r_out, par_out, rank, x, rx, info = md03by(
        cond='E', n=n, r=r, ipvt=ipvt, diag=diag,
        qtb=qtb, delta=delta, par=par, rank=0, tol=tol
    )

    assert info == 0
    assert rank <= n
    assert rank > 0  # Should be full rank for this well-conditioned case

    # Check solution properties
    dxnorm = np.linalg.norm(diag * x)
    assert dxnorm <= delta * 1.1


def test_md03by_rank_deficient():
    """Test with rank-deficient R matrix."""
    n = 4

    # Make last diagonal element very small (near-singular)
    r = np.array([
        [10.0,  2.0,  1.0,  0.5],
        [ 0.0,  8.0,  1.0,  0.3],
        [ 0.0,  0.0,  5.0,  0.2],
        [ 0.0,  0.0,  0.0,  1e-14]  # Nearly zero - rank deficient
    ], dtype=float, order='F')

    ipvt = np.array([1, 2, 3, 4], dtype=np.int32)
    diag = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    qtb = np.array([5.0, 4.0, 2.5, 0.0], dtype=float)
    delta = 5.0
    par = 0.0
    tol = 1e-10

    r_out, par_out, rank, x, rx, info = md03by(
        cond='E', n=n, r=r, ipvt=ipvt, diag=diag,
        qtb=qtb, delta=delta, par=par, rank=0, tol=tol
    )

    assert info == 0
    # Should detect reduced rank
    assert rank < n
    assert rank >= 3  # First 3 columns are well-conditioned


def test_md03by_use_provided_rank():
    """Test COND='U' mode with provided rank."""
    n = 4

    r = np.array([
        [10.0,  2.0,  1.0,  0.5],
        [ 0.0,  8.0,  1.0,  0.3],
        [ 0.0,  0.0,  5.0,  0.2],
        [ 0.0,  0.0,  0.0,  3.0]
    ], dtype=float, order='F')

    ipvt = np.array([1, 2, 3, 4], dtype=np.int32)
    diag = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    qtb = np.array([5.0, 4.0, 2.5, 1.5], dtype=float)
    delta = 5.0
    par = 0.0
    rank_in = 3  # User-provided rank
    tol = 0.0

    r_out, par_out, rank, x, rx, info = md03by(
        cond='U', n=n, r=r, ipvt=ipvt, diag=diag,
        qtb=qtb, delta=delta, par=par, rank=rank_in, tol=tol
    )

    assert info == 0
    # Output rank should be for the S matrix (may differ from input)
    assert rank > 0


def test_md03by_zero_dimension():
    """Test with N=0 (quick return)."""
    n = 0

    r = np.zeros((1, 0), dtype=float, order='F')
    ipvt = np.array([], dtype=np.int32)
    diag = np.array([], dtype=float)
    qtb = np.array([], dtype=float)
    delta = 1.0
    par = 0.0
    tol = 0.0

    r_out, par_out, rank, x, rx, info = md03by(
        cond='N', n=n, r=r, ipvt=ipvt, diag=diag,
        qtb=qtb, delta=delta, par=par, rank=0, tol=tol
    )

    assert info == 0
    assert par_out == 0.0
    assert rank == 0
    assert x.shape == (0,)
    assert rx.shape == (0,)


def test_md03by_error_invalid_delta():
    """Test error handling for invalid DELTA."""
    n = 3

    r = np.eye(3, dtype=float, order='F')
    ipvt = np.array([1, 2, 3], dtype=np.int32)
    diag = np.ones(3, dtype=float)
    qtb = np.ones(3, dtype=float)
    delta = -1.0  # Invalid: must be positive
    par = 0.0
    tol = 0.0

    with pytest.raises(ValueError, match="delta"):
        md03by(cond='N', n=n, r=r, ipvt=ipvt, diag=diag,
               qtb=qtb, delta=delta, par=par, rank=0, tol=tol)


def test_md03by_error_invalid_par():
    """Test error handling for invalid PAR."""
    n = 3

    r = np.eye(3, dtype=float, order='F')
    ipvt = np.array([1, 2, 3], dtype=np.int32)
    diag = np.ones(3, dtype=float)
    qtb = np.ones(3, dtype=float)
    delta = 1.0
    par = -0.5  # Invalid: must be non-negative
    tol = 0.0

    with pytest.raises(ValueError, match="par"):
        md03by(cond='N', n=n, r=r, ipvt=ipvt, diag=diag,
               qtb=qtb, delta=delta, par=par, rank=0, tol=tol)


def test_md03by_error_zero_diag():
    """Test error handling for zero diagonal element."""
    n = 3

    r = np.eye(3, dtype=float, order='F')
    ipvt = np.array([1, 2, 3], dtype=np.int32)
    diag = np.array([1.0, 0.0, 1.0], dtype=float)  # Zero element
    qtb = np.ones(3, dtype=float)
    delta = 1.0
    par = 0.0
    tol = 0.0

    with pytest.raises(ValueError, match="parameter 6"):
        md03by(cond='N', n=n, r=r, ipvt=ipvt, diag=diag,
               qtb=qtb, delta=delta, par=par, rank=0, tol=tol)
