"""
Tests for UE01MD: Machine-specific parameters for SLICOT routines.

UE01MD provides an extension of the LAPACK routine ILAENV to return
machine-specific parameters for SLICOT routines (block sizes, crossover
points, number of shifts).

Parameters returned based on ISPEC:
- ISPEC=1: Optimal blocksize
- ISPEC=2: Minimum block size
- ISPEC=3: Crossover point
- ISPEC=4: Number of shifts (product eigenvalue routine)
- ISPEC=8: Crossover point for multishift QR
"""

import numpy as np
import pytest


def test_ue01md_blocksize_mb04st():
    """
    Test ISPEC=1 (optimal blocksize) for MB04ST/MB04SB routines.

    For routine names with C2='4S' and C3='B', the block size should be
    derived from ILAENV for DGEQRF.
    """
    from ctrlsys import ue01md

    ispec = 1
    name = "MB04SB"
    opts = ""
    n1, n2, n3 = 100, 100, -1

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)
    assert result >= 1


def test_ue01md_blocksize_mb04tt():
    """
    Test ISPEC=1 for MB04TT (C2='4T', C3='T').

    Block size derived from ILAENV for DGEHRD divided by 4.
    """
    from ctrlsys import ue01md

    ispec = 1
    name = "MB04TT"
    opts = ""
    n1, n2, n3 = 100, 100, 100

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)
    assert result >= 1


def test_ue01md_blocksize_mb04pb():
    """
    Test ISPEC=1 for MB04PB (C2='4P', C3='B').

    Block size derived from ILAENV for DGEHRD divided by 2.
    """
    from ctrlsys import ue01md

    ispec = 1
    name = "MB04PB"
    opts = ""
    n1, n2, n3 = 100, 100, 100

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)
    assert result >= 1


def test_ue01md_blocksize_mb04wd():
    """
    Test ISPEC=1 for MB04WD (C2='4W', C3='D').

    Block size derived from ILAENV for DORGQR divided by 2.
    """
    from ctrlsys import ue01md

    ispec = 1
    name = "MB04WD"
    opts = ""
    n1, n2, n3 = 100, 100, 100

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)
    assert result >= 1


def test_ue01md_blocksize_mb04qb():
    """
    Test ISPEC=1 for MB04QB (C2='4Q', C3='B').

    Block size derived from ILAENV for DORMQR divided by 2.
    """
    from ctrlsys import ue01md

    ispec = 1
    name = "MB04QB"
    opts = ""
    n1, n2, n3 = 100, 100, 100

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)
    assert result >= 1


def test_ue01md_blocksize_mb04rb():
    """
    Test ISPEC=1 for MB04RB (C2='4R', C3='B').

    Block size derived from ILAENV for DGEHRD divided by 2.
    """
    from ctrlsys import ue01md

    ispec = 1
    name = "MB04RB"
    opts = ""
    n1, n2, n3 = 100, 100, 100

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)
    assert result >= 1


def test_ue01md_minblocksize():
    """
    Test ISPEC=2 (minimum block size).

    For MB04SB, minimum block size derived from ILAENV for DGEQRF.
    """
    from ctrlsys import ue01md

    ispec = 2
    name = "MB04SB"
    opts = ""
    n1, n2, n3 = 100, 100, -1

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)
    assert result >= 2


def test_ue01md_crossover():
    """
    Test ISPEC=3 (crossover point).

    For MB04SB, crossover from ILAENV for DGEQRF.
    """
    from ctrlsys import ue01md

    ispec = 3
    name = "MB04SB"
    opts = ""
    n1, n2, n3 = 100, 100, -1

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)
    assert result >= 0


def test_ue01md_num_shifts():
    """
    Test ISPEC=4 (number of shifts for product eigenvalue routine).

    Calls ILAENV with DHSEQR.
    """
    from ctrlsys import ue01md

    ispec = 4
    name = "MB03XP"
    opts = "SS"
    n1, n2, n3 = 100, 1, 100

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)


def test_ue01md_multishift_crossover():
    """
    Test ISPEC=8 (crossover point for multishift QR).

    Calls ILAENV with DHSEQR.
    """
    from ctrlsys import ue01md

    ispec = 8
    name = "MB03XP"
    opts = "SS"
    n1, n2, n3 = 100, 1, 100

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)


def test_ue01md_invalid_ispec():
    """
    Test with invalid ISPEC returns -1.
    """
    from ctrlsys import ue01md

    ispec = 99
    name = "MB04SB"
    opts = ""
    n1, n2, n3 = 100, 100, -1

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert result == -1


def test_ue01md_lowercase_name():
    """
    Test that routine handles lowercase names (converts to uppercase).
    """
    from ctrlsys import ue01md

    ispec = 1
    name = "mb04sb"
    opts = ""
    n1, n2, n3 = 100, 100, -1

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert isinstance(result, int)
    assert result >= 1


def test_ue01md_unknown_routine_returns_one():
    """
    Test that unknown routine patterns return blocksize of 1.

    For unrecognized C2/C3 combinations, the routine returns the default value.
    """
    from ctrlsys import ue01md

    ispec = 1
    name = "AB01MD"
    opts = ""
    n1, n2, n3 = 100, 100, -1

    result = ue01md(ispec, name, opts, n1, n2, n3)

    assert result == 1


def test_ue01md_consistency_lower_upper():
    """
    Test that lowercase and uppercase names produce same result.
    """
    from ctrlsys import ue01md

    ispec = 1
    opts = ""
    n1, n2, n3 = 100, 100, -1

    result_upper = ue01md(ispec, "MB04SB", opts, n1, n2, n3)
    result_lower = ue01md(ispec, "mb04sb", opts, n1, n2, n3)

    assert result_upper == result_lower
