#!/usr/bin/env python3
"""Debug script to trace MB03LF NEIG calculation."""
import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/ctrlsys.c')
from python.ctrlsys import mb03lf

n = 8
m = n // 2

# Z matrix (8x8) from HTML doc
z = np.array([
    [3.1472, 4.5751, -0.7824, 1.7874, -2.2308, -0.6126, 2.0936, 4.5974],
    [4.0579, 4.6489, 4.1574, 2.5774, -4.5383, -1.1844, 2.5469, -1.5961],
    [-3.7301, -3.4239, 2.9221, 2.4313, -4.0287, 2.6552, -2.2397, 0.8527],
    [4.1338, 4.7059, 4.5949, -1.0777, 3.2346, 2.9520, 1.7970, -2.7619],
    [1.3236, 4.5717, 1.5574, 1.5548, 1.9483, -3.1313, 1.5510, 2.5127],
    [-4.0246, -0.1462, -4.6429, -3.2881, -1.8290, -0.1024, -3.3739, -2.4490],
    [-2.2150, 3.0028, 3.4913, 2.0605, 4.5022, -0.5441, -3.8100, 0.0596],
    [0.4688, -3.5811, 4.3399, -4.6817, -4.6555, 1.4631, -0.0164, 1.9908]
], order='F', dtype=float)

# B matrix (4x4) from HTML doc
b = np.array([
    [0.6882, -3.3782, -3.3435, 1.8921],
    [-0.3061, 2.9428, 1.0198, 2.4815],
    [-4.8810, -1.8878, -2.3703, -0.4946],
    [-1.6288, 0.2853, 1.5408, -4.1618]
], order='F', dtype=float)

# FG matrix (4x5) from HTML doc
fg = np.array([
    [-2.4013, -2.7102, 0.3834, -3.9335, 3.1730],
    [-3.1815, -2.3620, 4.9613, 4.6190, 3.6869],
    [3.6929, 0.7970, 0.4986, -4.9537, -4.1556],
    [3.5303, 1.2206, -1.4905, 0.1325, -1.0022]
], order='F', dtype=float)

# Call MB03LF
z_out, neig, q, u, alphar, alphai, beta, iwarn, info = mb03lf('C', 'C', 'P', z, b, fg)

print(f"info = {info}")
print(f"neig = {neig} (expected 3)")
print(f"iwarn = {iwarn}")
print()
print("Eigenvalues (alphar/beta + alphai/beta * i):")
for i in range(m):
    re = alphar[i] / beta[i] if beta[i] != 0 else float('inf')
    im = alphai[i] / beta[i] if beta[i] != 0 else float('inf')
    print(f"  [{i}] ({alphar[i]:.4f} + {alphai[i]:.4f}i) / {beta[i]:.4f} = {re:.4f} + {im:.4f}i")
print()
print("Due to skew-Ham/Ham structure, -lambda is also an eigenvalue:")
for i in range(m):
    re = -alphar[i] / beta[i] if beta[i] != 0 else float('inf')
    im = -alphai[i] / beta[i] if beta[i] != 0 else float('inf')
    print(f"  -[{i}] = {re:.4f} + {im:.4f}i")
print()

# Count eigenvalues with strictly negative real part
count = 0
for i in range(m):
    # Original eigenvalue
    re = alphar[i] / beta[i] if beta[i] != 0 else 0
    if re < 0:
        count += 1
        print(f"  lambda[{i}] has negative real part: {re:.4f}")
    # Its negative
    neg_re = -alphar[i] / beta[i] if beta[i] != 0 else 0
    if neg_re < 0:
        count += 1
        print(f"  -lambda[{i}] has negative real part: {neg_re:.4f}")
print(f"\nTotal eigenvalues with negative real part: {count}")
