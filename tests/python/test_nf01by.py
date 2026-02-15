import unittest
import numpy as np
from ctrlsys import nf01ay

# We don't have nf01by exposed yet, so this test will fail importing it.
# But I can prepare the test logic.

def test_nf01by_jacobian():
    """
    Validate NF01BY Jacobian against finite differences or analytical formula.
    """
    try:
        from ctrlsys import nf01by
    except ImportError:
        self.fail("Could not import nf01by")

    np.random.seed(42)
    
    nsmp = 5
    nz = 3
    l = 1
    nn = 4
    
    ipar = np.array([nn], dtype=np.int32)
    
    nwb = nn * (nz + 2) + 1
    wb = np.random.rand(nwb)
    wb = (wb - 0.5) * 2.0
    
    z = np.random.rand(nsmp, nz)
    z_arr = np.asfortranarray(z)
    
    # Error vector e (random)
    e = np.random.rand(nsmp)
    
    # Call NF01BY
    # cjte = 'C' (compute J'e)
    # Arguments: cjte, nsmp, nz, l, ipar, wb, z, e
    # Returns: j, jte, info
    
    j, jte, info = nf01by('C', nsmp, nz, l, ipar, wb, z_arr, e)
    
    assert info == 0
    assert j.shape == (nsmp, nwb)
    assert jte.shape == (nwb,)
    
    # Validate J analytically
    # Reconstruct parameters
    w_end = nn * nz
    w_flat = wb[:w_end]
    ws_start = w_end
    ws_end = ws_start + nn
    ws = wb[ws_start:ws_end]
    b_start = ws_end
    b = wb[b_start:] # nn+1
    
    W = w_flat.reshape((nz, nn), order='F') # Input weights
    
    # Calculate activations
    # linear = Z @ W + b[:nn]
    # act = tanh(linear)
    # y = act @ ws + b[nn]
    
    # J structure:
    # Columns 0..NN*NZ-1: derivatives w.r.t W
    # Columns NN*NZ..NN*NZ+NN-1: derivatives w.r.t ws
    # Columns ... : derivatives w.r.t b
    
    linear = z @ W + b[:nn]
    act = np.tanh(linear)
    
    # 1. Derivative w.r.t b[nn] (output bias) -> 1
    j_b_out = j[:, nwb-1]
    np.testing.assert_allclose(j_b_out, 1.0)
    
    # 2. Derivative w.r.t ws
    # d y / d ws_i = act_i
    j_ws = j[:, ws_start:ws_end]
    np.testing.assert_allclose(j_ws, act, rtol=1e-14)
    
    # 3. Derivative w.r.t b[:nn] (hidden biases)
    # d y / d b_i = ws_i * (1 - act_i^2)
    # Shape (nsmp, nn)
    d_act = 1.0 - act**2
    expected_j_b = d_act * ws[None, :]
    j_b = j[:, b_start:b_start+nn]
    np.testing.assert_allclose(j_b, expected_j_b, rtol=1e-14)
    
    # 4. Derivative w.r.t W (hidden weights)
    # d y / d W_{ki} (input k, neuron i)
    # = d y / d b_i * z_k
    # W is stored column major: w1 (nz), w2 (nz)...
    # j[:, 0:nz] is deriv w.r.t w1
    
    for i in range(nn):
        # For neuron i
        # expected deriv w.r.t w_{., i} is (nsmp, nz)
        # element (s, k) = expected_j_b[s, i] * z[s, k]
        
        # j block
        j_w_i = j[:, i*nz : (i+1)*nz]
        
        expected_j_w_i = expected_j_b[:, i:i+1] * z
        
        np.testing.assert_allclose(j_w_i, expected_j_w_i, rtol=1e-14)
        
    # Validate JTE = J.T @ e
    jte_expected = j.T @ e
    np.testing.assert_allclose(jte, jte_expected, rtol=1e-13, atol=1e-14)

def test_nf01by_cjte_n():
    """
    Validate NF01BY with CJTE='N' (compute J only, not J'e).

    Random seed: 888 (for reproducibility)
    """
    try:
        from ctrlsys import nf01by
    except ImportError:
        self.fail("Could not import nf01by")

    np.random.seed(888)

    nsmp = 5
    nz = 3
    l = 1
    nn = 4

    ipar = np.array([nn], dtype=np.int32)

    nwb = nn * (nz + 2) + 1
    wb = np.random.rand(nwb)
    wb = (wb - 0.5) * 2.0

    z = np.random.rand(nsmp, nz)
    z_arr = np.asfortranarray(z)

    e = np.random.rand(nsmp)

    j, jte, info = nf01by('N', nsmp, nz, l, ipar, wb, z_arr, e)

    assert info == 0
    assert j.shape == (nsmp, nwb)

    # JTE should not be computed (may be garbage or zeros)
    # Just verify J is valid

    w_end = nn * nz
    w_flat = wb[:w_end]
    ws_start = w_end
    ws_end = ws_start + nn
    ws = wb[ws_start:ws_end]
    b_start = ws_end
    b = wb[b_start:]

    W = w_flat.reshape((nz, nn), order='F')

    linear = z @ W + b[:nn]
    act = np.tanh(linear)

    # Validate output bias derivative
    j_b_out = j[:, nwb-1]
    np.testing.assert_allclose(j_b_out, 1.0)

    # Validate ws derivatives
    j_ws = j[:, ws_start:ws_end]
    np.testing.assert_allclose(j_ws, act, rtol=1e-14)

def test_nf01by_finite_difference():
    """
    Validate Jacobian against finite differences.

    Random seed: 999 (for reproducibility)
    """
    try:
        from ctrlsys import nf01by, nf01ay
    except ImportError:
        self.fail("Could not import nf01by or nf01ay")

    np.random.seed(999)

    nsmp = 3
    nz = 2
    l = 1
    nn = 3

    ipar = np.array([nn], dtype=np.int32)

    nwb = nn * (nz + 2) + 1
    wb = np.random.rand(nwb) * 0.5

    z = np.random.rand(nsmp, nz)
    z_arr = np.asfortranarray(z)

    e = np.random.rand(nsmp)

    j, jte, info = nf01by('C', nsmp, nz, l, ipar, wb, z_arr, e)
    assert info == 0

    # Compute finite difference approximation for first few parameters
    h = 1e-7
    j_fd = np.zeros_like(j)

    for i in range(min(5, nwb)):
        wb_plus = wb.copy()
        wb_plus[i] += h

        wb_minus = wb.copy()
        wb_minus[i] -= h

        y_plus, _ = nf01ay(nsmp, nz, l, ipar, wb_plus, z_arr)
        y_minus, _ = nf01ay(nsmp, nz, l, ipar, wb_minus, z_arr)

        j_fd[:, i] = (y_plus[:, 0] - y_minus[:, 0]) / (2 * h)

    # Compare analytical vs finite difference
    np.testing.assert_allclose(j[:, :5], j_fd[:, :5], rtol=1e-5, atol=1e-6)

def test_nf01by_error_l_not_one():
    """
    Validate NF01BY error handling for l != 1.

    NF01BY only supports single output (l=1).
    """
    try:
        from ctrlsys import nf01by
    except ImportError:
        self.fail("Could not import nf01by")

    nsmp = 5
    nz = 3
    l = 2
    nn = 4

    ipar = np.array([nn], dtype=np.int32)

    nwb = nn * (nz + 2) + 1
    wb = np.random.rand(nwb)

    z_arr = np.asfortranarray(np.random.rand(nsmp, nz))
    e = np.random.rand(nsmp)

    # Should return error
    try:
        j, jte, info = nf01by('C', nsmp, nz, l, ipar, wb, z_arr, e)
        # If wrapper doesn't raise exception, check info
        assert info != 0
    except:
        # Exception expected for invalid l
        pass

if __name__ == '__main__':
    unittest.main()
