import pytest
import numpy as np
from ctrlsys import ab13ed
from numpy.testing import assert_allclose

class TestAB13ED:
    def test_basic_stable_scalar(self):
        """
        Test with a scalar stable system A = [-1].
        Distance to imaginary axis is |-1 - 0| = 1.
        """
        A = np.array([[-1.0]], order='F')
        low, high, info = ab13ed(A)
        
        assert info == 0
        assert_allclose(low, 1.0, rtol=1e-5)
        # Theoretically exact for scalar, but allow bound slack
        assert low <= 1.0 <= high
        assert abs(high - low) < 1e-4

    def test_basic_unstable_scalar(self):
        """
        Test with a scalar unstable system A = [1].
        Distance to imaginary axis is |1| = 1.
        """
        A = np.array([[1.0]], order='F')
        low, high, info = ab13ed(A)
        
        assert info == 0
        assert_allclose(low, 1.0, rtol=1e-5)

    def test_diag_stable(self):
        """
        Test diagonal stable matrix.
        A = diag(-2, -5, -3)
        Distance should be min(|-2|, |-5|, |-3|) = 2.
        """
        A = np.diag([-2.0, -5.0, -3.0])
        A = np.asfortranarray(A)
        low, high, info = ab13ed(A)
        
        assert info == 0
        # Expected beta = 2.0
        assert low <= 2.0 + 1e-5
        assert high >= 2.0 - 1e-5
        
    def test_complex_pair_stable(self):
        """
        Test stable matrix with complex eigenvalues.
        A = [[-1, 10], [-10, -1]]
        Eigenvalues are -1 +/- 10i. Real part is -1.
        Distance to imaginary axis is |-1| = 1.
        """
        A = np.array([[-1.0, 10.0], [-10.0, -1.0]], order='F')
        low, high, info = ab13ed(A)
        
        assert info == 0
        assert_allclose(low, 1.0, rtol=1e-2) # Approximate
        assert high >= 1.0

    def test_invalid_args(self):
        """Test invalid arguments handling."""
        A = np.eye(2, order='F')
        # Check if it raises TypeError for invalid args if we were to pass wrong types
        # But here we just check Slicot error handling if exposed
        # Currently python wrapper might not catch strict parameter errors unless we added checks
        pass
    
    def test_tolerance(self):
        """Test with specific tolerance."""
        A = np.diag([-2.0, -3.0])
        A = np.asfortranarray(A)
        # Pass a tolerance
        low, high, info = ab13ed(A, tol=1e-5)
        assert info == 0
        assert abs(high - low) < 1.0 # Loose check, just confirming it runs
        assert low <= 2.0 <= high
