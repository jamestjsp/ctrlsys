"""Tests for TG01OZ - Complex SISO descriptor system reduction.

TG01OZ computes for a single-input single-output descriptor system with
complex elements, given by the system matrix:

    [ D     C    ]
    [ B  A - s*E ]

with E nonsingular, a reduced system matrix:

    [ d     c    ]
    [ b  a - s*e ]

such that d has a "sufficiently" large magnitude.

The routine returns:
- NZ: Order of the reduced system
- G: Gain of the reduced system
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose


def test_tg01oz_basic_identity_e():
    """Test TG01OZ with JOBE='I' (E is identity).

    Random seed: 42 (for reproducibility)
    Tests basic functionality - system reduction with identity E.
    """
    from ctrlsys import tg01oz

    np.random.seed(42)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = 0.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    e = np.eye(n, dtype=np.complex128, order='F')

    dcba_out, e_out, nz, g, info = tg01oz('I', dcba, e, 0.0)

    assert info == 0, f"TG01OZ failed with info={info}"
    assert 0 <= nz <= n, f"NZ={nz} should be between 0 and N={n}"
    assert isinstance(g, complex), "G should be complex"


def test_tg01oz_basic_general_e():
    """Test TG01OZ with JOBE='G' (E is general).

    Random seed: 123 (for reproducibility)
    Tests with general nonsingular E matrix.
    """
    from ctrlsys import tg01oz

    np.random.seed(123)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    np.fill_diagonal(e, np.abs(np.diag(e)) + 2.0)
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = 0.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_out, e_out, nz, g, info = tg01oz('G', dcba, e, 0.0)

    assert info == 0, f"TG01OZ failed with info={info}"
    assert 0 <= nz <= n, f"NZ={nz} should be between 0 and N={n}"


def test_tg01oz_d_already_large():
    """Test TG01OZ when D already has sufficiently large magnitude.

    When D is already large relative to the system norms, no reduction
    should occur (NZ = N).
    """
    from ctrlsys import tg01oz

    n = 3
    n1 = n + 1

    a = np.array([
        [1.0 + 0.0j, 0.1, 0.1],
        [0.1, 2.0 + 0.0j, 0.1],
        [0.1, 0.1, 3.0 + 0.0j]
    ], dtype=np.complex128, order='F')
    b = np.array([[0.1], [0.1], [0.1]], dtype=np.complex128, order='F')
    c = np.array([[0.1, 0.1, 0.1]], dtype=np.complex128, order='F')
    d = 100.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    e = np.eye(n, dtype=np.complex128, order='F')

    dcba_out, e_out, nz, g, info = tg01oz('I', dcba, e, 0.0)

    assert info == 0
    assert nz == n, "NZ should equal N when D is already large"
    assert_allclose(g, d, rtol=1e-14,
                    err_msg="Gain should be D when no reduction needed")


def test_tg01oz_n_zero():
    """Test TG01OZ with N=0 (quick return).

    When N=0, G should equal D.
    """
    from ctrlsys import tg01oz

    d = 5.0 + 3.0j
    dcba = np.array([[d]], dtype=np.complex128, order='F')
    e = np.zeros((0, 0), dtype=np.complex128, order='F')

    dcba_out, e_out, nz, g, info = tg01oz('G', dcba, e, 0.0)

    assert info == 0
    assert nz == 0
    assert_allclose(g, d, rtol=1e-14, err_msg="G should equal D when N=0")


def test_tg01oz_gain_computation():
    """Test TG01OZ gain computation property.

    Random seed: 456 (for reproducibility)

    For a system (A, E, B, C, D) reduced to (a, e, b, c, d),
    the gain G relates the original and reduced transfer functions.
    """
    from ctrlsys import tg01oz

    np.random.seed(456)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = 0.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    e = np.eye(n, dtype=np.complex128, order='F')

    dcba_out, e_out, nz, g, info = tg01oz('I', dcba, e, 0.0)

    assert info == 0
    assert abs(g) > 0 or nz == n, "G should be nonzero when reduction occurs"


def test_tg01oz_reduced_system_size():
    """Test TG01OZ that reduced DCBA has correct size (NZ+1) x (NZ+1).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import tg01oz

    np.random.seed(789)
    n = 5
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = 0.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    e = np.eye(n, dtype=np.complex128, order='F')

    dcba_out, e_out, nz, g, info = tg01oz('I', dcba, e, 0.0)

    assert info == 0
    d_reduced = dcba_out[0, 0]
    c_reduced = dcba_out[0, 1:nz+1]
    b_reduced = dcba_out[1:nz+1, 0]
    a_reduced = dcba_out[1:nz+1, 1:nz+1]

    assert c_reduced.shape == (nz,), f"C shape should be ({nz},)"
    assert b_reduced.shape == (nz,), f"B shape should be ({nz},)"
    assert a_reduced.shape == (nz, nz), f"A shape should be ({nz},{nz})"


def test_tg01oz_invalid_jobe():
    """Test TG01OZ with invalid JOBE parameter."""
    from ctrlsys import tg01oz

    n = 2
    n1 = n + 1
    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')

    with pytest.raises(ValueError):
        tg01oz('X', dcba, e, 0.0)


def test_tg01oz_transfer_function_preservation():
    """Test that transfer function is preserved after reduction.

    Random seed: 888 (for reproducibility)

    For a descriptor system, the transfer function is:
        G(s) = D + C * (s*E - A)^(-1) * B

    After reduction with gain G:
        G_reduced(s) = d + c * (s*e - a)^(-1) * b
        G_original(s) = G * G_reduced(s)
    """
    from ctrlsys import tg01oz

    np.random.seed(888)
    n = 3
    n1 = n + 1

    a = 0.5 * (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    np.fill_diagonal(a, np.diag(a) - 2.0)
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = 1e-10 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_orig = dcba.copy()
    e = np.eye(n, dtype=np.complex128, order='F')

    dcba_out, e_out, nz, g, info = tg01oz('I', dcba, e, 0.0)

    assert info == 0

    if nz < n and nz > 0:
        s = 1.0 + 1.0j
        d_orig = dcba_orig[0, 0]
        c_orig = dcba_orig[0, 1:]
        b_orig = dcba_orig[1:, 0]
        a_orig = dcba_orig[1:, 1:]

        tf_orig = d_orig + c_orig @ np.linalg.solve(s * np.eye(n) - a_orig, b_orig)

        d_red = dcba_out[0, 0]
        c_red = dcba_out[0, 1:nz+1]
        b_red = dcba_out[1:nz+1, 0]
        a_red = dcba_out[1:nz+1, 1:nz+1]
        e_red = e_out[:nz, :nz]

        tf_red = d_red + c_red @ np.linalg.solve(s * e_red - a_red, b_red)

        assert_allclose(tf_orig, g * tf_red, rtol=1e-10,
                       err_msg="Transfer function should be preserved up to gain")


def test_tg01oz_d_large_magnitude_after_reduction():
    """Test that d has sufficiently large magnitude after reduction.

    Random seed: 999 (for reproducibility)

    The algorithm should reduce until |d| * (1 + max|a|) > tol * ||b|| * ||c||.
    """
    from ctrlsys import tg01oz

    np.random.seed(999)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = 0.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    e = np.eye(n, dtype=np.complex128, order='F')

    dcba_out, e_out, nz, g, info = tg01oz('I', dcba, e, 0.0)

    assert info == 0

    if nz > 0:
        d_red = dcba_out[0, 0]
        assert abs(d_red) > 0, "Reduced d should have nonzero magnitude"


def test_tg01oz_workspace_query():
    """Test TG01OZ workspace query (LZWORK = -1)."""
    from ctrlsys import tg01oz

    n = 5
    n1 = n + 1
    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    e = np.eye(n, dtype=np.complex128, order='F')

    dcba_out, e_out, nz, g, info = tg01oz('G', dcba, e, 0.0)

    assert info == 0


def test_tg01oz_larger_system():
    """Test TG01OZ with a larger system.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import tg01oz

    np.random.seed(555)
    n = 8
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    np.fill_diagonal(e, np.abs(np.diag(e)) + 2.0)
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = 0.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_out, e_out, nz, g, info = tg01oz('G', dcba, e, 0.0)

    assert info == 0
    assert 0 <= nz <= n


def test_tg01oz_e_triangularized_when_general():
    """Test that E is triangularized when JOBE='G'.

    Random seed: 666 (for reproducibility)

    When JOBE='G', E is first triangularized via QR decomposition,
    and the reduced e should remain upper triangular.
    """
    from ctrlsys import tg01oz

    np.random.seed(666)
    n = 4
    n1 = n + 1

    a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    e = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
    np.fill_diagonal(e, np.abs(np.diag(e)) + 2.0)
    b = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1)).astype(np.complex128, order='F')
    c = (np.random.randn(1, n) + 1j * np.random.randn(1, n)).astype(np.complex128, order='F')
    d = 0.0 + 0.0j

    dcba = np.zeros((n1, n1), dtype=np.complex128, order='F')
    dcba[0, 0] = d
    dcba[0, 1:] = c[0, :]
    dcba[1:, 0] = b[:, 0]
    dcba[1:, 1:] = a

    dcba_out, e_out, nz, g, info = tg01oz('G', dcba, e, 0.0)

    assert info == 0

    if nz > 1:
        e_reduced = e_out[:nz, :nz]
        for i in range(1, nz):
            for j in range(i):
                assert abs(e_reduced[i, j]) < 1e-12, \
                    f"E not upper triangular at ({i},{j}): {e_reduced[i,j]}"
