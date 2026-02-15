"""Tests for TG01AZ - Complex descriptor system pencil balancing.

TG01AZ is the complex version of TG01AD. It balances the matrices of the
system pencil S = (A B; C 0) - lambda*(E 0; 0 0) by diagonal similarity
transformations (Dl*A*Dr - lambda Dl*E*Dr, Dl*B, C*Dr).
"""
import pytest
import numpy as np
from ctrlsys import tg01az


def test_tg01az_basic_example():
    """Test TG01AZ with example from SLICOT HTML documentation.

    L=4, N=4, M=2, P=2, JOB='A', THRESH=0.0
    Tests descriptor system pencil balancing with complex matrices.
    """
    l, n, m, p = 4, 4, 2, 2
    job = 'A'
    thresh = 0.0

    a = np.array([
        [  -1.0+0j,    0.0+0j,    0.0+0j,  0.003+0j],
        [   0.0+0j,    0.0+0j,    0.1+0j,   0.02+0j],
        [ 100.0+0j,   10.0+0j,    0.0+0j,    0.4+0j],
        [   0.0+0j,    0.0+0j,    0.0+0j,    0.0+0j]
    ], dtype=np.complex128, order='F')

    e = np.array([
        [   1.0+0j,   0.2+0j,    0.0+0j,   0.0+0j],
        [   0.0+0j,   1.0+0j,    0.0+0j,  0.01+0j],
        [ 300.0+0j,  90.0+0j,    6.0+0j,   0.3+0j],
        [   0.0+0j,   0.0+0j,   20.0+0j,   0.0+0j]
    ], dtype=np.complex128, order='F')

    b = np.array([
        [    10.0+0j,      0.0+0j],
        [     0.0+0j,      0.0+0j],
        [     0.0+0j,   1000.0+0j],
        [ 10000.0+0j,  10000.0+0j]
    ], dtype=np.complex128, order='F')

    c = np.array([
        [-0.1+0j,   0.0+0j,   0.001+0j,    0.0+0j],
        [ 0.0+0j,  0.01+0j,  -0.001+0j, 0.0001+0j]
    ], dtype=np.complex128, order='F')

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0, f"TG01AZ failed with info={info}"

    a_expected = np.array([
        [-1.0000+0j,  0.0000+0j,  0.0000+0j,  0.3000+0j],
        [ 0.0000+0j,  0.0000+0j,  1.0000+0j,  2.0000+0j],
        [ 1.0000+0j,  0.1000+0j,  0.0000+0j,  0.4000+0j],
        [ 0.0000+0j,  0.0000+0j,  0.0000+0j,  0.0000+0j]
    ], dtype=np.complex128, order='F')

    e_expected = np.array([
        [1.0000+0j,  0.2000+0j,  0.0000+0j,  0.0000+0j],
        [0.0000+0j,  1.0000+0j,  0.0000+0j,  1.0000+0j],
        [3.0000+0j,  0.9000+0j,  0.6000+0j,  0.3000+0j],
        [0.0000+0j,  0.0000+0j,  0.2000+0j,  0.0000+0j]
    ], dtype=np.complex128, order='F')

    b_expected = np.array([
        [100.0000+0j,   0.0000+0j],
        [  0.0000+0j,   0.0000+0j],
        [  0.0000+0j, 100.0000+0j],
        [100.0000+0j, 100.0000+0j]
    ], dtype=np.complex128, order='F')

    c_expected = np.array([
        [-0.0100+0j,  0.0000+0j,  0.0010+0j,  0.0000+0j],
        [ 0.0000+0j,  0.0010+0j, -0.0010+0j,  0.0010+0j]
    ], dtype=np.complex128, order='F')

    lscale_expected = np.array([10.0, 10.0, 0.1, 0.01], dtype=np.float64)
    rscale_expected = np.array([0.1, 0.1, 1.0, 10.0], dtype=np.float64)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(e_out, e_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(c_out, c_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(lscale, lscale_expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(rscale, rscale_expected, rtol=1e-3, atol=1e-4)


def test_tg01az_scaling_property():
    """Verify balancing property: transformed matrices satisfy Dl*A*Dr.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    l, n, m, p = 4, 4, 2, 2
    job = 'A'
    thresh = 0.0

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0

    dl = np.diag(lscale)
    dr = np.diag(rscale)

    a_check = dl @ a_orig @ dr
    e_check = dl @ e_orig @ dr
    b_check = dl @ b_orig
    c_check = c_orig @ dr

    np.testing.assert_allclose(a_out, a_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(b_out, b_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(c_out, c_check, rtol=1e-14, atol=1e-14)


def test_tg01az_job_n_only_ae():
    """Test JOB='N': balance only A and E, not B and C.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    l, n, m, p = 3, 3, 2, 2
    job = 'N'
    thresh = 0.0

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    e_orig = e.copy()

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0

    dl = np.diag(lscale)
    dr = np.diag(rscale)

    a_check = dl @ a_orig @ dr
    e_check = dl @ e_orig @ dr

    np.testing.assert_allclose(a_out, a_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e_check, rtol=1e-14, atol=1e-14)


def test_tg01az_job_b_only_abe():
    """Test JOB='B': balance A, E, and B only (not C).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    l, n, m, p = 3, 3, 2, 2
    job = 'B'
    thresh = 0.0

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0

    dl = np.diag(lscale)
    dr = np.diag(rscale)

    a_check = dl @ a_orig @ dr
    e_check = dl @ e_orig @ dr
    b_check = dl @ b_orig

    np.testing.assert_allclose(a_out, a_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(b_out, b_check, rtol=1e-14, atol=1e-14)


def test_tg01az_job_c_only_aec():
    """Test JOB='C': balance A, E, and C only (not B).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    l, n, m, p = 3, 3, 2, 2
    job = 'C'
    thresh = 0.0

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    c_orig = c.copy()

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0

    dl = np.diag(lscale)
    dr = np.diag(rscale)

    a_check = dl @ a_orig @ dr
    e_check = dl @ e_orig @ dr
    c_check = c_orig @ dr

    np.testing.assert_allclose(a_out, a_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(c_out, c_check, rtol=1e-14, atol=1e-14)


def test_tg01az_zero_dimensions():
    """Test TG01AZ with zero dimensions."""
    l, n, m, p = 0, 0, 0, 0
    job = 'A'
    thresh = 0.0

    a = np.array([], dtype=np.complex128).reshape(0, 0, order='F')
    e = np.array([], dtype=np.complex128).reshape(0, 0, order='F')
    b = np.array([], dtype=np.complex128).reshape(0, 0, order='F')
    c = np.array([], dtype=np.complex128).reshape(0, 0, order='F')

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0


def test_tg01az_zero_m_and_p():
    """Test TG01AZ with M=0 and P=0 (no B/C matrices).

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    l, n, m, p = 3, 3, 0, 0
    job = 'A'
    thresh = 0.0

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = np.array([], dtype=np.complex128).reshape(l, 0, order='F')
    c = np.array([], dtype=np.complex128).reshape(0, n, order='F')

    a_orig = a.copy()
    e_orig = e.copy()

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0

    dl = np.diag(lscale)
    dr = np.diag(rscale)
    a_check = dl @ a_orig @ dr
    e_check = dl @ e_orig @ dr

    np.testing.assert_allclose(a_out, a_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e_check, rtol=1e-14, atol=1e-14)


def test_tg01az_invalid_job():
    """Test TG01AZ with invalid JOB parameter."""
    l, n, m, p = 2, 2, 1, 1
    job = 'X'
    thresh = 0.0

    a = np.eye(2, dtype=np.complex128, order='F')
    e = np.eye(2, dtype=np.complex128, order='F')
    b = np.ones((2, 1), dtype=np.complex128, order='F')
    c = np.ones((1, 2), dtype=np.complex128, order='F')

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == -1


def test_tg01az_negative_thresh():
    """Test TG01AZ with negative THRESH (should fail with info=-6)."""
    l, n, m, p = 2, 2, 1, 1
    job = 'A'
    thresh = -1.0

    a = np.eye(2, dtype=np.complex128, order='F')
    e = np.eye(2, dtype=np.complex128, order='F')
    b = np.ones((2, 1), dtype=np.complex128, order='F')
    c = np.ones((1, 2), dtype=np.complex128, order='F')

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == -6


def test_tg01az_rectangular_ln_different():
    """Test TG01AZ with L != N (rectangular A and E).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    l, n, m, p = 3, 5, 2, 2
    job = 'A'
    thresh = 0.0

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0
    assert len(lscale) == l
    assert len(rscale) == n

    dl = np.diag(lscale)
    dr = np.diag(rscale)

    a_check = dl @ a_orig @ dr
    e_check = dl @ e_orig @ dr
    b_check = dl @ b_orig
    c_check = c_orig @ dr

    np.testing.assert_allclose(a_out, a_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(b_out, b_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(c_out, c_check, rtol=1e-14, atol=1e-14)


def test_tg01az_positive_thresh():
    """Test TG01AZ with positive threshold (ignores small elements).

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    l, n, m, p = 4, 4, 2, 2
    job = 'A'
    thresh = 0.5

    a = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    e = (np.random.randn(l, n) + 1j * np.random.randn(l, n)).astype(np.complex128, order='F')
    b = (np.random.randn(l, m) + 1j * np.random.randn(l, m)).astype(np.complex128, order='F')
    c = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0

    dl = np.diag(lscale)
    dr = np.diag(rscale)

    a_check = dl @ a_orig @ dr
    e_check = dl @ e_orig @ dr
    b_check = dl @ b_orig
    c_check = c_orig @ dr

    np.testing.assert_allclose(a_out, a_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(b_out, b_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(c_out, c_check, rtol=1e-14, atol=1e-14)


def test_tg01az_complex_imaginary_parts():
    """Test TG01AZ with significant imaginary parts.

    Random seed: 444 (for reproducibility)
    Verifies the CABS1 norm (|re| + |im|) is used correctly.
    """
    np.random.seed(444)
    l, n, m, p = 3, 3, 2, 2
    job = 'A'
    thresh = 0.0

    a = np.array([
        [1+2j, 0+0j, 0+0j],
        [0+0j, 3+4j, 0+0j],
        [0+0j, 0+0j, 5+6j]
    ], dtype=np.complex128, order='F')
    e = np.eye(3, dtype=np.complex128, order='F')
    b = np.ones((3, 2), dtype=np.complex128, order='F')
    c = np.ones((2, 3), dtype=np.complex128, order='F')

    a_orig = a.copy()
    e_orig = e.copy()
    b_orig = b.copy()
    c_orig = c.copy()

    a_out, e_out, b_out, c_out, lscale, rscale, info = tg01az(
        job, l, n, m, p, thresh, a, e, b, c
    )

    assert info == 0

    dl = np.diag(lscale)
    dr = np.diag(rscale)

    a_check = dl @ a_orig @ dr
    e_check = dl @ e_orig @ dr
    b_check = dl @ b_orig
    c_check = c_orig @ dr

    np.testing.assert_allclose(a_out, a_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(e_out, e_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(b_out, b_check, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(c_out, c_check, rtol=1e-14, atol=1e-14)
