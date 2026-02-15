"""
Tests for SB03MW - Solve 2x2 continuous-time Lyapunov equation.

Solves for the 2-by-2 symmetric matrix X in:
    op(T)'*X + X*op(T) = SCALE*B

where T is 2-by-2, B is symmetric 2-by-2, and op(T) = T or T'.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from ctrlsys import sb03mw

def test_sb03mw_basic():
    """
    Test basic functionality for stable T.
    
    Solve T'*X + X*T = B
    
    T = [[-2, 1],
         [ 0, -3]]
    B = [[ 4, 0],
         [ 0, 9]]
         
    Analytic check or using scipy scalar logic.
    """
    t = np.array([[-2.0, 0.0], [1.0, -3.0]], order='F') # T is Upper in C (row 0: -2, 1; row 1: 0, -3) -> Wait, Fortran order.
    # Logic T:
    # -2  1
    #  0 -3
    # Fortran storage: [-2, 0, 1, -3]
    t = np.array([[-2.0, 0.0], [1.0, -3.0]], order='F')
    
    # B symmetric:
    # 1 0
    # 0 1
    b = np.eye(2, order='F')
    
    # Solve T'*X + X*T = scale*B (with ltran=False -> op(T)=T)
    # But SB03MW def: op(T)'*X + X*op(T) = scale*B
    # If ltran=False, op(T) = T. Eq: T'*X + X*T = scale*B.
    
    x, scale, xnorm, info = sb03mw(ltran=False, lupper=True, t=t, b=b)
    
    assert info == 0
    assert scale == 1.0 # Well scaled
    
    # SB03MW returns optimal solution in upper or lower triangle
    # We must symmetrize x to check full residual
    x_full = np.triu(x) + np.triu(x, 1).T
    
    # Check residual
    # T'*X + X*T - scale*B
    res = t.T @ x_full + x_full @ t - scale * b
    assert_allclose(res, np.zeros((2,2)), atol=1e-14)

def test_sb03mw_transpose():
    """
    Test with op(T) = T' (ltran=True).
    
    Eq: T*X + X*T' = scale*B
    """
    # Stable T
    t = np.array([[-1.0, 0.0], [-2.0, -3.0]], order='F')
    b = np.array([[2.0, 1.0], [1.0, 2.0]], order='F')
    
    x, scale, xnorm, info = sb03mw(ltran=True, lupper=True, t=t, b=b)
    
    assert info == 0
    
    x_full = np.triu(x) + np.triu(x, 1).T
    
    # Check residual: T*X + X*T' - scale*B
    res = t @ x_full + x_full @ t.T - scale * b
    assert_allclose(res, np.zeros((2,2)), atol=1e-14)

def test_sb03mw_unstable_scaling():
    """
    Test with T demanding scaling (near singular or overflow).
    """
    # T small, B large
    eps = np.finfo(float).eps
    t = np.array([[-eps, 0.0], [0.0, -eps]], order='F')
    b = np.eye(2, order='F')
    
    x, scale, xnorm, info = sb03mw(ltran=False, lupper=True, t=t, b=b)
    
    # X should be huge if scale were 1.
    # T*X + X*T = sc*B => -2*eps*X = sc*I => X = -sc/(2*eps)*I
    # Scale might be 1.0 if no overflow occurs (double max is very large)
    
    x_full = np.triu(x) + np.triu(x, 1).T
    
    # Residual with scaling
    res = t.T @ x_full + x_full @ t - scale * b
    assert_allclose(res, np.zeros((2,2)), atol=1e-14)

def test_sb03mw_singular():
    """
    Test with singular T (eigenvalue 0).
    INFO should be 1.
    """
    t = np.array([[0.0, 0.0], [0.0, -1.0]], order='F')
    b = np.eye(2, order='F')
    
    x, scale, xnorm, info = sb03mw(ltran=False, lupper=True, t=t, b=b)
    
    assert info == 1
