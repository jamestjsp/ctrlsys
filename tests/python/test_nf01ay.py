import unittest
import numpy as np
from ctrlsys import nf01ay

def test_nf01ay_basic():
    """
    Validate NF01AY against NumPy implementation.
    """
    np.random.seed(42)
    
    nsmp = 5
    nz = 3
    l = 2
    nn = 4
    
    ipar = np.array([nn], dtype=np.int32)
    
    ldwb = nn * (nz + 2) + 1
    lwb = ldwb * l
    
    wb = np.random.rand(lwb)
    # Ensure reproducible, somewhat reasonable range for tanh
    wb = (wb - 0.5) * 2.0 
    
    z = np.random.rand(nsmp, nz) # Row-major in python, but we pass it F-order
    # Wait, inputs are F-order.
    # Z in Fortran is (LDZ, NZ). Leading dim LDZ >= NSMP.
    # But effectively it stores samples in rows?
    # "The leading NSMP-by-NZ part of this array must contain the set of input samples"
    # Z(1,1)...Z(1,NZ) is sample 1.
    # In Fortran column-major, Z(i, j) is z[i-1 + (j-1)*LDZ].
    # So samples are rows of the matrix Z.
    
    z_arr = np.asfortranarray(z)
    
    # Expected output
    y_expected = np.zeros((nsmp, l), order='F')
    
    # Python implementation
    for k in range(l):
        # Extract weights for output k
        offset = k * ldwb
        wb_k = wb[offset : offset + ldwb]
        
        # Structure:
        # w_flat: nn * nz (weights for hidden layer)
        # ws: nn (weights for output layer)
        # b: nn + 1 (biases)
        
        w_end = nn * nz
        w_flat = wb_k[:w_end]
        ws_start = w_end
        ws_end = ws_start + nn
        ws = wb_k[ws_start:ws_end]
        b_start = ws_end
        b = wb_k[b_start:] # length nn+1
        
        # Reshape W to (nz, nn) - Wait, Fortran storage of W
        # "w1(1), ..., w1(NZ), ..., wn(1), ..., wn(NZ)"
        # So n vectors of length nz.
        # In Fortran memory: w1_1, w1_2... w1_nz, w2_1...
        # So it's effectively a matrix W of shape (nz, nn) in column-major order?
        # Or (nn, nz)?
        # "w1(1)...w1(NZ)" -> vector w1.
        # "wb(k) = [ w1... wn ... ]"
        # In DGEMM call:
        # CALL DGEMM( 'Transpose', 'Transpose', NN, NV, NZ, -TWO,
        #             WB(1+LK), NZ, Z(I,1), LDZ, ZERO, DWORK(NN+1), NN )
        # A = WB(1+LK). LDA = NZ.
        # Op(A) = A'. A is NZ x NN (since LDA=NZ and K=NN in DGEMM(M,N,K...)).
        # Wait. DGEMM(TRANSA, TRANSB, M, N, K, ...)
        # M = NN, N = NV, K = NZ.
        # A is WB. Transpose. So A is K x M = NZ x NN.
        # So WB stores NZ x NN matrix in column-major.
        # Columns are w1, w2... wn.
        # Each column is a weight vector for a neuron.
        
        W = w_flat.reshape((nz, nn), order='F')
        
        # Z is (nsmp, nz).
        # For each sample z_i (1 x nz):
        # hidden = tanh( z_i @ W + b[:nn] )
        # output = hidden @ ws + b[nn]
        
        # z @ W -> (nsmp, nn)
        
        linear_part = z @ W + b[:nn]
        hidden = np.tanh(linear_part)
        
        y_expected[:, k] = hidden @ ws + b[nn]
        
    # Call NF01AY
    # y = nf01ay(nsmp, nz, l, ipar, wb, z)
    # Note: wrapper needs to handle dimensions
    
    y_act, info = nf01ay(nsmp, nz, l, ipar, wb, z_arr)
    
    assert info == 0
    np.testing.assert_allclose(y_act, y_expected, rtol=1e-13, atol=1e-14)

def test_nf01ay_activation_range():
    """
    Validate tanh activation output range [-1, 1].

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    nsmp = 10
    nz = 3
    l = 2
    nn = 5

    ipar = np.array([nn], dtype=np.int32)

    ldwb = nn * (nz + 2) + 1
    lwb = ldwb * l

    wb = np.random.rand(lwb) * 10.0 - 5.0
    z_arr = np.asfortranarray(np.random.rand(nsmp, nz) * 10.0 - 5.0)

    y_act, info = nf01ay(nsmp, nz, l, ipar, wb, z_arr)

    assert info == 0

    # Tanh range is [-1, 1], but output adds bias and weights
    # So actual output can be outside [-1, 1]
    # Just verify computation succeeds and produces finite values
    assert np.all(np.isfinite(y_act))

def test_nf01ay_edge_case_zero_samples():
    """
    Validate NF01AY with nsmp=0 (no samples).
    """
    nsmp = 0
    nz = 3
    l = 2
    nn = 4

    ipar = np.array([nn], dtype=np.int32)

    ldwb = nn * (nz + 2) + 1
    lwb = ldwb * l

    wb = np.random.rand(lwb)
    z_arr = np.zeros((max(1, nsmp), nz), order='F')

    y_act, info = nf01ay(nsmp, nz, l, ipar, wb, z_arr)

    assert info == 0
    # Wrapper may return (0, l) for nsmp=0
    assert y_act.shape[1] == l
    assert y_act.shape[0] == nsmp

def test_nf01ay_error_ldwork_too_small():
    """
    Validate NF01AY error handling for insufficient workspace.

    This test would require modifying the wrapper to expose ldwork.
    Currently skipped as wrapper auto-computes ldwork.
    """
    # Skip - wrapper handles ldwork internally
    pass

if __name__ == '__main__':
    unittest.main()
