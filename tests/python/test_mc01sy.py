"""
Tests for mc01sy - reconstruct real number from mantissa and exponent.
"""
import numpy as np
import pytest


def test_mc01sy_basic_reconstruction():
    """
    Test basic reconstruction: A = M * B^E.
    Uses mc01sw to decompose, then mc01sy to reconstruct.
    """
    from ctrlsys import mc01sw, mc01sy

    test_values = [1024.0, 0.5, -256.0, 1.5, 0.125]
    base = 2

    for original in test_values:
        m, e = mc01sw(original, base)
        reconstructed, ovflow = mc01sy(m, e, base)
        assert not ovflow
        np.testing.assert_allclose(reconstructed, original, rtol=1e-14)


def test_mc01sy_zero_mantissa():
    """
    Test that M = 0 returns A = 0 regardless of exponent.
    """
    from ctrlsys import mc01sy

    a, ovflow = mc01sy(0.0, 100, 2)
    assert not ovflow
    assert a == 0.0

    a, ovflow = mc01sy(0.0, -100, 2)
    assert not ovflow
    assert a == 0.0


def test_mc01sy_zero_exponent():
    """
    Test that E = 0 returns A = M (identity case).
    """
    from ctrlsys import mc01sy

    test_values = [1.0, 1.5, -2.5, 0.5]

    for m in test_values:
        a, ovflow = mc01sy(m, 0, 2)
        assert not ovflow
        assert a == m


def test_mc01sy_base_10():
    """
    Test reconstruction with base 10.
    """
    from ctrlsys import mc01sw, mc01sy

    test_values = [1000.0, 0.001, -5000.0, 1.23456]
    base = 10

    for original in test_values:
        m, e = mc01sw(original, base)
        reconstructed, ovflow = mc01sy(m, e, base)
        assert not ovflow
        np.testing.assert_allclose(reconstructed, original, rtol=1e-14)


def test_mc01sy_negative_mantissa():
    """
    Test reconstruction with negative mantissa.
    """
    from ctrlsys import mc01sy

    # -1.5 * 2^3 = -12.0
    a, ovflow = mc01sy(-1.5, 3, 2)
    assert not ovflow
    np.testing.assert_allclose(a, -12.0, rtol=1e-14)


def test_mc01sy_invalid_base():
    """
    Test that base < 2 raises ValueError.
    """
    from ctrlsys import mc01sy

    with pytest.raises(ValueError, match="Base b must be >= 2"):
        mc01sy(1.0, 1, 1)

    with pytest.raises(ValueError, match="Base b must be >= 2"):
        mc01sy(1.0, 1, 0)


def test_mc01sy_roundtrip_random():
    """
    Test round-trip reconstruction with random values.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc01sw, mc01sy

    np.random.seed(42)

    for _ in range(20):
        original = np.random.uniform(-1e10, 1e10)
        if abs(original) < 1e-15:
            continue

        for base in [2, 10, 16]:
            m, e = mc01sw(original, base)
            reconstructed, ovflow = mc01sy(m, e, base)
            if not ovflow:
                np.testing.assert_allclose(reconstructed, original, rtol=1e-13)
