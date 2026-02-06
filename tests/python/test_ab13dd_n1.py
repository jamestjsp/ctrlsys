"""Test for ab13dd segfault with n=1 (single-state systems) - Issue #9"""
import numpy as np
import pytest
from slicot import ab13dd


def test_ab13dd_n1_segfault():
    """Minimal reproducer from issue #9: single-state system G(s) = 1/(s+1)"""
    n, m, p = 1, 1, 1
    A = np.asfortranarray([[-1.0]])
    E = np.asfortranarray([[1.0]])
    B = np.asfortranarray([[1.0]])
    C = np.asfortranarray([[1.0]])
    D = np.asfortranarray([[0.0]])
    fpeak_in = np.asfortranarray([0.0, 1.0])

    result = ab13dd('C', 'I', 'N', 'Z', n, m, p, fpeak_in, A, E, B, C, D, 0.0)

    gpeak, fpeak, info = result
    assert info == 0, f"Expected info=0, got {info}"
    # L-infinity norm of 1/(s+1) is 1 at omega=0
    assert gpeak[0] == pytest.approx(1.0, rel=0.01), f"Expected gpeak[0]=1.0, got {gpeak[0]}"


def test_ab13dd_n1_continuous_withd():
    """Single-state with D matrix"""
    n, m, p = 1, 1, 1
    A = np.asfortranarray([[-1.0]])
    E = np.asfortranarray([[1.0]])
    B = np.asfortranarray([[1.0]])
    C = np.asfortranarray([[1.0]])
    D = np.asfortranarray([[0.5]])  # nonzero D
    fpeak_in = np.asfortranarray([0.0, 1.0])

    result = ab13dd('C', 'I', 'N', 'D', n, m, p, fpeak_in, A, E, B, C, D, 0.0)
    gpeak, fpeak, info = result
    assert info == 0


def test_ab13dd_n1_discrete():
    """Single-state discrete system"""
    n, m, p = 1, 1, 1
    A = np.asfortranarray([[0.5]])  # stable discrete pole
    E = np.asfortranarray([[1.0]])
    B = np.asfortranarray([[1.0]])
    C = np.asfortranarray([[1.0]])
    D = np.asfortranarray([[0.0]])
    fpeak_in = np.asfortranarray([0.0, 1.0])

    result = ab13dd('D', 'I', 'N', 'Z', n, m, p, fpeak_in, A, E, B, C, D, 1e-9)
    gpeak, fpeak, info = result
    assert info == 0


if __name__ == "__main__":
    test_ab13dd_n1_segfault()
    print("Test passed!")
