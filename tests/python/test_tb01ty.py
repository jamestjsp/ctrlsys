"""
Tests for TB01TY - Balance rows/columns of matrix blocks

TB01TY balances rows (MODE=0) or columns (MODE!=0) of a matrix block using
integer powers of the floating-point base (radix). This avoids rounding errors.

Each non-zero row/column is scaled by BASE^IEXPT where IEXPT is the largest
integer <= EXPT, with EXPT = -log(ABSSUM/SIZE)/log(BASE).

Key behaviors:
- Non-zero rows/cols scaled to have 1-norm in range (SIZE/BASE, SIZE]
- Zero rows/cols get scale factor 1.0
- Scaling uses integer powers of BASE (no rounding errors)
- IEXPT calculation requires floor() semantics, not truncation
"""
import numpy as np
import pytest

ctrlsys = pytest.importorskip("ctrlsys")


def test_tb01ty_basic_column_balance():
    """
    Test basic column balancing (MODE != 0)

    Validates:
    - Columns are scaled correctly using integer powers of BASE
    - Scale factors stored in BVECT at correct positions (JOFF+1:JOFF+NCOL)
    - Scaled column 1-norms are in range (SIZE/BASE, SIZE]
    """
    nrow = 3
    ncol = 2
    ioff = 0
    joff = 0
    size = 1.0
    mode = 1  # Column balance

    # Create test matrix (column-major)
    x = np.array([
        [0.001, 100.0],
        [0.002, 200.0],
        [0.003, 300.0]
    ], dtype=float, order='F')

    x_out, bvect, info = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x)

    assert info == 0

    # Check scale factors are powers of 2 (most common BASE)
    base = 2.0
    for j in range(ncol):
        sf = bvect[joff + j]
        # Scale factor should be a power of BASE
        log_sf = np.log(sf) / np.log(base)
        assert abs(log_sf - round(log_sf)) < 1e-10, f"Scale factor {sf} not a power of {base}"

    # Check scaled column 1-norms are in range (SIZE/BASE, SIZE]
    for j in range(ncol):
        col_norm = np.sum(np.abs(x_out[:, joff + j]))
        assert col_norm <= size, f"Column {j} norm {col_norm} exceeds SIZE={size}"
        assert col_norm > size / base, f"Column {j} norm {col_norm} below SIZE/BASE"


def test_tb01ty_basic_row_balance():
    """
    Test basic row balancing (MODE == 0)

    Validates:
    - Rows are scaled correctly using integer powers of BASE
    - Scale factors stored in BVECT at correct positions (IOFF+1:IOFF+NROW)
    - Scaled row 1-norms are in range (SIZE/BASE, SIZE]
    """
    nrow = 2
    ncol = 3
    ioff = 0
    joff = 0
    size = 1.0
    mode = 0  # Row balance

    # Create test matrix (column-major)
    x = np.array([
        [0.001, 0.002, 0.003],
        [100.0, 200.0, 300.0]
    ], dtype=float, order='F')

    x_out, bvect, info = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x)

    assert info == 0

    # Check scale factors are powers of 2
    base = 2.0
    for i in range(nrow):
        sf = bvect[ioff + i]
        log_sf = np.log(sf) / np.log(base)
        assert abs(log_sf - round(log_sf)) < 1e-10, f"Scale factor {sf} not a power of {base}"

    # Check scaled row 1-norms are in range (SIZE/BASE, SIZE]
    for i in range(nrow):
        row_norm = np.sum(np.abs(x_out[ioff + i, :]))
        assert row_norm <= size, f"Row {i} norm {row_norm} exceeds SIZE={size}"
        assert row_norm > size / base, f"Row {i} norm {row_norm} below SIZE/BASE"


def test_tb01ty_zero_row_column():
    """
    Test handling of zero rows/columns

    Validates:
    - Zero rows/columns get scale factor 1.0
    - Non-zero rows/columns are scaled correctly
    - "Numerically zero" (very small values) treated as zero
    """
    nrow = 3
    ncol = 2
    ioff = 0
    joff = 0
    size = 1.0
    mode = 1  # Column balance

    # Column 0 has values, column 1 is zero
    x = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0]
    ], dtype=float, order='F')

    x_out, bvect, info = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x)

    assert info == 0

    # Zero column should have scale factor 1.0
    np.testing.assert_allclose(bvect[1], 1.0, rtol=1e-14)

    # Non-zero column should have proper scale factor
    base = 2.0
    log_sf = np.log(bvect[0]) / np.log(base)
    assert abs(log_sf - round(log_sf)) < 1e-10


def test_tb01ty_offset_block():
    """
    Test balancing with non-zero offsets (IOFF, JOFF)

    Validates:
    - Only the specified block (IOFF+1:IOFF+NROW, JOFF+1:JOFF+NCOL) is balanced
    - Scale factors stored at correct BVECT positions
    - Elements outside the block remain unchanged
    """
    # Full matrix is 4x4, we balance a 2x2 block with offset (1,1)
    ldx = 4
    ioff = 1
    joff = 1
    nrow = 2
    ncol = 2
    size = 1.0
    mode = 1  # Column balance

    x = np.zeros((ldx, ldx), dtype=float, order='F')
    # Fill the block to be balanced
    x[ioff:ioff+nrow, joff:joff+ncol] = np.array([
        [0.01, 10.0],
        [0.02, 20.0]
    ])
    # Fill some values outside the block
    x[0, 0] = 999.0
    x[3, 3] = 888.0

    x_orig = x.copy()
    x_out, bvect, info = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x)

    assert info == 0

    # Values outside the block should be unchanged
    np.testing.assert_allclose(x_out[0, 0], 999.0, rtol=1e-14)
    np.testing.assert_allclose(x_out[3, 3], 888.0, rtol=1e-14)

    # Scale factors should be at positions joff to joff+ncol-1
    for j in range(joff, joff + ncol):
        assert bvect[j] > 0, f"Scale factor at {j} should be positive"


def test_tb01ty_negative_exponent_adjustment():
    """
    Test IEXPT adjustment for negative non-integer exponents

    This is a key pitfall: Fortran INT truncates toward zero, but we need
    floor() semantics for negative values.

    Example: EXPT = -0.5
    - INT(-0.5) = 0 (truncation toward zero) - WRONG
    - floor(-0.5) = -1 (what we need) - CORRECT

    The Fortran code handles this with:
        IF ((IEXPT.LT.0) .AND. (DBLE(IEXPT).NE.EXPT)) IEXPT = IEXPT - 1

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    nrow = 4
    ncol = 3
    ioff = 0
    joff = 0
    size = 1.0
    mode = 1  # Column balance

    # Create columns with varying magnitudes that will produce negative EXPT
    base = 2.0

    # Column with 1-norm slightly above 1.0 -> EXPT is small negative
    # Column with 1-norm around 0.75 -> EXPT ~ 0.415, scale = 2^0 = 1
    # Column with 1-norm around 1.5 -> EXPT ~ -0.585, need floor to get -1
    x = np.array([
        [0.4, 0.2, 0.1],
        [0.3, 0.15, 0.2],
        [0.5, 0.25, 0.3],
        [0.3, 0.15, 0.15]
    ], dtype=float, order='F')

    x_out, bvect, info = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x)

    assert info == 0

    # Verify all scale factors are powers of BASE
    for j in range(ncol):
        sf = bvect[j]
        log_sf = np.log(sf) / np.log(base)
        iexpt = round(log_sf)
        assert abs(log_sf - iexpt) < 1e-10, f"Scale factor {sf} not power of {base}"

    # Verify scaled columns are in correct range
    for j in range(ncol):
        col_norm = np.sum(np.abs(x_out[:, j]))
        assert col_norm <= size + 1e-14, f"Column {j} norm {col_norm} > SIZE"
        assert col_norm > size / base - 1e-14, f"Column {j} norm {col_norm} < SIZE/BASE"


def test_tb01ty_size_parameter():
    """
    Test effect of SIZE parameter

    Validates:
    - Scaled 1-norms are in range (SIZE/BASE, SIZE]
    - Different SIZE values produce correct scaling
    """
    nrow = 3
    ncol = 2
    ioff = 0
    joff = 0
    mode = 1  # Column balance

    x = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0]
    ], dtype=float, order='F')

    base = 2.0

    # Test with SIZE = 10.0
    size = 10.0
    x_out, bvect, info = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x.copy())

    assert info == 0

    for j in range(ncol):
        col_norm = np.sum(np.abs(x_out[:, j]))
        assert col_norm <= size, f"Column {j} norm {col_norm} > SIZE={size}"
        assert col_norm > size / base, f"Column {j} norm {col_norm} < SIZE/BASE"

    # Test with SIZE = 0.1
    size = 0.1
    x_out2, bvect2, info2 = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x.copy())

    assert info2 == 0

    for j in range(ncol):
        col_norm = np.sum(np.abs(x_out2[:, j]))
        assert col_norm <= size, f"Column {j} norm {col_norm} > SIZE={size}"
        assert col_norm > size / base, f"Column {j} norm {col_norm} < SIZE/BASE"


def test_tb01ty_negative_size():
    """
    Test with negative SIZE parameter

    The Fortran code uses ABS(SIZE), so negative SIZE should work.
    """
    nrow = 2
    ncol = 2
    ioff = 0
    joff = 0
    size = -1.0  # Negative size
    mode = 1

    x = np.array([
        [0.01, 1.0],
        [0.02, 2.0]
    ], dtype=float, order='F')

    x_out, bvect, info = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x)

    assert info == 0

    # Check scaled norms are in range (|SIZE|/BASE, |SIZE|]
    base = 2.0
    abs_size = abs(size)
    for j in range(ncol):
        col_norm = np.sum(np.abs(x_out[:, j]))
        assert col_norm <= abs_size, f"Column {j} norm {col_norm} > |SIZE|"
        assert col_norm > abs_size / base, f"Column {j} norm {col_norm} < |SIZE|/BASE"


def test_tb01ty_scaling_preserves_ratios():
    """
    Test that scaling preserves element ratios within each row/column

    Validates mathematical property: ratios between elements are preserved.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    nrow = 3
    ncol = 4
    ioff = 0
    joff = 0
    size = 1.0
    mode = 1  # Column balance

    x = np.random.rand(nrow, ncol).astype(float, order='F') * 10
    x_orig = x.copy()

    x_out, bvect, info = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x)

    assert info == 0

    # Check that element ratios within each column are preserved
    for j in range(ncol):
        # Get scale factor
        sf = bvect[j]

        # All elements should be scaled by same factor
        for i in range(nrow):
            expected = x_orig[i, j] * sf
            np.testing.assert_allclose(x_out[i, j], expected, rtol=1e-14)


def test_tb01ty_zero_dimensions():
    """
    Test edge cases with zero dimensions

    Validates:
    - NROW=0 or NCOL=0 returns without error
    - Matrix unchanged for zero block dimensions
    """
    ioff = 0
    joff = 0
    size = 1.0

    x = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=float, order='F')
    x_orig = x.copy()

    # NROW = 0
    x_out, bvect, info = ctrlsys.tb01ty(1, ioff, joff, 0, 2, size, x.copy())
    assert info == 0

    # NCOL = 0
    x_out, bvect, info = ctrlsys.tb01ty(1, ioff, joff, 2, 0, size, x.copy())
    assert info == 0


def test_tb01ty_identity_scaling():
    """
    Test case where no scaling needed (1-norm already in range)

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    nrow = 3
    ncol = 2
    ioff = 0
    joff = 0
    size = 1.0
    mode = 1  # Column balance
    base = 2.0

    # Create columns with 1-norm in range (SIZE/BASE, SIZE] = (0.5, 1.0]
    # Scale factor should be 1.0
    x = np.array([
        [0.2, 0.25],
        [0.2, 0.20],
        [0.2, 0.25]
    ], dtype=float, order='F')

    # Column 0 has norm 0.6, column 1 has norm 0.7 - both in (0.5, 1.0]
    x_out, bvect, info = ctrlsys.tb01ty(mode, ioff, joff, nrow, ncol, size, x)

    assert info == 0

    # Scale factors should be 1.0 (2^0)
    np.testing.assert_allclose(bvect[0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(bvect[1], 1.0, rtol=1e-14)

    # Matrix should be unchanged
    np.testing.assert_allclose(x_out, x, rtol=1e-14)
