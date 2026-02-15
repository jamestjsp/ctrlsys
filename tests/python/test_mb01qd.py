#!/usr/bin/env python3
"""
pytest tests for MB01QD - Matrix scaling with overflow/underflow prevention.
"""
import pytest
import numpy as np
from ctrlsys import mb01qd


def test_mb01qd_full_matrix():
    """Test MB01QD with full matrix (type='G')."""
    m, n = 3, 3
    # Column-major storage: col0, col1, col2
    a = np.array([
        [1.0, 4.0, 7.0],
        [2.0, 5.0, 8.0],
        [3.0, 6.0, 9.0]
    ], dtype=np.float64, order='F')

    cfrom, cto = 2.0, 4.0
    a_result, info = mb01qd(b'G', m, n, 0, 0, cfrom, cto, a)

    expected = np.array([
        [2.0, 8.0, 14.0],
        [4.0, 10.0, 16.0],
        [6.0, 12.0, 18.0]
    ], dtype=np.float64, order='F')

    assert info == 0
    np.testing.assert_allclose(a_result, expected, rtol=1e-14)


def test_mb01qd_lower_triangular():
    """Test MB01QD with lower triangular matrix (type='L')."""
    m, n = 3, 3
    # Column-major: only lower triangle scaled
    a = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float64, order='F')

    cfrom, cto = 1.0, 2.0
    a_result, info = mb01qd(b'L', m, n, 0, 0, cfrom, cto, a)

    expected = np.array([
        [2.0, 0.0, 0.0],
        [4.0, 6.0, 0.0],
        [8.0, 10.0, 12.0]
    ], dtype=np.float64, order='F')

    assert info == 0
    np.testing.assert_allclose(a_result, expected, rtol=1e-14)


def test_mb01qd_upper_triangular():
    """Test MB01QD with upper triangular matrix (type='U')."""
    m, n = 3, 3
    a = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], dtype=np.float64, order='F')

    cfrom, cto = 1.0, 0.5
    a_result, info = mb01qd(b'U', m, n, 0, 0, cfrom, cto, a)

    expected = np.array([
        [0.5, 1.0, 1.5],
        [0.0, 2.0, 2.5],
        [0.0, 0.0, 3.0]
    ], dtype=np.float64, order='F')

    assert info == 0
    np.testing.assert_allclose(a_result, expected, rtol=1e-14)


def test_mb01qd_upper_hessenberg():
    """Test MB01QD with upper Hessenberg matrix (type='H')."""
    m, n = 3, 3
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0]
    ], dtype=np.float64, order='F')

    cfrom, cto = 1.0, 3.0
    a_result, info = mb01qd(b'H', m, n, 0, 0, cfrom, cto, a)

    expected = np.array([
        [3.0, 6.0, 9.0],
        [12.0, 15.0, 18.0],
        [0.0, 21.0, 24.0]
    ], dtype=np.float64, order='F')

    assert info == 0
    np.testing.assert_allclose(a_result, expected, rtol=1e-14)


def test_mb01qd_empty_matrix():
    """Test MB01QD with empty matrix (m=0)."""
    m, n = 0, 3
    a = np.array([[]], dtype=np.float64, order='F').reshape(0, 3)

    cfrom, cto = 2.0, 4.0
    a_result, info = mb01qd(b'G', m, n, 0, 0, cfrom, cto, a)

    assert info == 0
    assert a_result.shape == (0, 3)


def test_mb01qd_block_lower_triangular():
    """Test MB01QD with block lower triangular structure."""
    m, n = 4, 4
    a = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ], dtype=np.float64, order='F')

    cfrom, cto = 1.0, 2.0
    nrows = np.array([2, 2], dtype=np.int32)
    a_result, info = mb01qd(b'L', m, n, 0, 0, cfrom, cto, a, nrows)

    expected = np.array([
        [2.0, 4.0, 0.0, 0.0],
        [6.0, 8.0, 0.0, 0.0],
        [10.0, 12.0, 14.0, 16.0],
        [18.0, 20.0, 22.0, 24.0]
    ], dtype=np.float64, order='F')

    assert info == 0
    np.testing.assert_allclose(a_result, expected, rtol=1e-14)


def test_mb01qd_overflow_prevention():
    """Test MB01QD handles large scaling factors safely."""
    m, n = 2, 2
    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=np.float64, order='F')

    # Large scaling factor within float64 range (max ~1.8e308)
    cfrom = 1e-100
    cto = 1e100
    a_result, info = mb01qd(b'G', m, n, 0, 0, cfrom, cto, a)

    # Result should be scaled by 1e200
    expected = np.array([
        [1e200, 2e200],
        [3e200, 4e200]
    ], dtype=np.float64, order='F')

    assert info == 0
    # Check relative error since absolute values are huge
    np.testing.assert_allclose(a_result / expected, 1.0, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
