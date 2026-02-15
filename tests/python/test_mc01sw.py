"""
Tests for mc01sw - extract mantissa and exponent.

MC01SW finds M and E such that A = M * B^E where 1 <= |M| < B.
"""

import numpy as np
from ctrlsys import mc01sw


def test_mc01sw_basic():
    """Test basic case: 1000 = 1.0 * 10^3."""
    m, e = mc01sw(1000.0, 10)
    assert abs(m - 1.0) < 1e-14
    assert e == 3
    # Verify: A = M * B^E
    assert abs(1000.0 - m * (10 ** e)) < 1e-10


def test_mc01sw_zero():
    """Test zero input: returns M=0, E=0."""
    m, e = mc01sw(0.0, 10)
    assert m == 0.0
    assert e == 0


def test_mc01sw_negative():
    """Test negative input: -1000 = -1.0 * 10^3."""
    m, e = mc01sw(-1000.0, 10)
    assert abs(m + 1.0) < 1e-14  # m = -1.0
    assert e == 3
    # Verify: A = M * B^E
    assert abs(-1000.0 - m * (10 ** e)) < 1e-10


def test_mc01sw_small_value():
    """Test small value: 0.001 = 1.0 * 10^-3."""
    m, e = mc01sw(0.001, 10)
    assert abs(m - 1.0) < 1e-14
    assert e == -3
    # Verify: A = M * B^E
    assert abs(0.001 - m * (10 ** e)) < 1e-15


def test_mc01sw_fractional():
    """Test fractional mantissa: 500 = 5.0 * 10^2."""
    m, e = mc01sw(500.0, 10)
    assert abs(m - 5.0) < 1e-14
    assert e == 2
    assert abs(500.0 - m * (10 ** e)) < 1e-10


def test_mc01sw_base_2():
    """Test with base 2: 8 = 1.0 * 2^3."""
    m, e = mc01sw(8.0, 2)
    assert abs(m - 1.0) < 1e-14
    assert e == 3
    assert abs(8.0 - m * (2 ** e)) < 1e-14


def test_mc01sw_base_2_fractional():
    """Test with base 2 fractional: 5 = 1.25 * 2^2."""
    m, e = mc01sw(5.0, 2)
    # m should be 1.25 (since 5 = 1.25 * 4)
    assert abs(m - 1.25) < 1e-14
    assert e == 2
    assert abs(5.0 - m * (2 ** e)) < 1e-14


def test_mc01sw_mantissa_bounds():
    """
    Validate mantissa bounds: 1 <= |M| < B.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    for _ in range(100):
        a = np.random.randn() * 10 ** np.random.randint(-10, 10)
        b = np.random.randint(2, 20)
        m, e = mc01sw(a, b)

        if a == 0.0:
            assert m == 0.0 and e == 0
        else:
            # 1 <= |M| < B
            assert abs(m) >= 1.0, f"Failed: |{m}| < 1 for a={a}, b={b}"
            assert abs(m) < b, f"Failed: |{m}| >= {b} for a={a}, b={b}"
            # A = M * B^E
            assert abs(a - m * (b ** e)) < abs(a) * 1e-14, f"Failed reconstruction for a={a}"


def test_mc01sw_reconstruction_property():
    """
    Validate mathematical property: A = M * B^E exactly.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    test_cases = [
        (1.0, 10),
        (100.0, 10),
        (0.01, 10),
        (-256.0, 2),
        (81.0, 3),  # 81 = 1.0 * 3^4
        (2.0, 10),
        (np.pi, 10),
        (np.e, 2),
    ]

    for a, b in test_cases:
        m, e = mc01sw(a, b)
        reconstructed = m * (b ** e)
        np.testing.assert_allclose(a, reconstructed, rtol=1e-14,
            err_msg=f"Failed for a={a}, b={b}")
