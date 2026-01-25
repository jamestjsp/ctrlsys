# SLICOT Workflows

## 1. LQR Controller Design

Solve continuous-time algebraic Riccati equation for optimal state feedback.

```python
import numpy as np
from slicot import sb02md

n = 2
A = np.array([[0, 1], [-2, -3]], dtype=float, order='F')
B = np.array([[0], [1]], dtype=float, order='F')
Q = np.eye(n, dtype=float, order='F')
R = np.array([[1.0]], dtype=float, order='F')

# Form G = B @ inv(R) @ B.T
G = B @ np.linalg.solve(R, B.T)
G = np.asfortranarray(G)

# Solve ARE: A'X + XA - XGX + Q = 0
rcond, wr, wi, S, U, info = sb02md(
    'C',  # continuous
    'D',  # H matrix form
    'U',  # upper triangle
    'N',  # no scaling
    'S',  # stable eigenvalues first
    n, A.copy(), G, Q
)

X = Q  # Solution overwrites Q
K = np.linalg.solve(R, B.T @ X)  # Optimal gain
```

## 2. Pole Placement

Assign closed-loop eigenvalues using state feedback.

```python
import numpy as np
from slicot import sb01bd

n, m = 3, 1
A = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]], dtype=float, order='F')
B = np.array([[0], [0], [1]], dtype=float, order='F')

# Desired poles
wr = np.array([-1.0, -2.0, -3.0], dtype=float)
wi = np.zeros(3, dtype=float)

F, wr_out, wi_out, nfp, nap, nup, Z, iwarn, info = sb01bd(
    'C',     # continuous
    n, m,
    len(wr), # number of poles to assign
    0.0,     # alpha (stability threshold)
    A.copy(), B, wr, wi, 0.0
)

# Closed-loop: A + B @ F has eigenvalues at -1, -2, -3
```

## 3. Model Reduction (Balanced Truncation)

Reduce system order using Hankel-norm approximation.

```python
import numpy as np
from slicot import ab09ad

n, m, p = 6, 1, 1
# Assume A, B, C, D are defined (order n system)
A = ...  # n x n, stable
B = ...  # n x m
C = ...  # p x n
D = ...  # p x m

nr = 3  # desired reduced order

Ar, Br, Cr, Dr, ns, hsv, info = ab09ad(
    'C',    # continuous
    'B',    # balance for B and C
    'A',    # order selection: all singular values
    n, m, p,
    A.copy(), B.copy(), C.copy(), D.copy(),
    nr, 0.0
)
# hsv contains Hankel singular values
# ns is actual reduced order achieved
```

## 4. System Identification (N4SID/MOESP)

Estimate state-space model from input-output data.

```python
import numpy as np
from slicot import ib01ad, ib01bd

# Input-output data: u (nsmp x m), y (nsmp x l)
nsmp, m, l = 1000, 1, 1
nobr = 15  # block rows
u = ...    # input data, Fortran order
y = ...    # output data, Fortran order

# Step 1: Preprocessing and order estimation
n_est, r, sv, iwarn, info = ib01ad(
    'M',    # MOESP method
    'C',    # Cholesky algorithm
    'N',    # no B,D via MOESP
    'O',    # one batch
    'N',    # no connection
    'N',    # no control
    nobr, m, l, u, y, 0.0, -1.0
)

# Step 2: Estimate system matrices
n = n_est  # or choose based on sv
A, C, B, D, Q, Ry, S, K, iwarn, info = ib01bd(
    'C',    # combined method
    'A',    # all matrices
    'K',    # compute Kalman gain
    nobr, n, m, l, nsmp, r, 0.0
)
```

## 5. H-infinity Controller

Design H-infinity optimal controller.

```python
import numpy as np
from slicot import sb10ad

# Generalized plant P partitioned as:
#   [A  | B1  B2 ]
#   [C1 | D11 D12]
#   [C2 | D21 D22]

n = 4       # states
m1, m2 = 1, 1  # disturbance inputs, control inputs
p1, p2 = 1, 1  # performance outputs, measurements

# Define matrices...
A = ...
B = np.hstack([B1, B2])
C = np.vstack([C1, C2])
D = np.block([[D11, D12], [D21, D22]])

ncon = m2   # control inputs
nmeas = p2  # measurements
gamma = 1.0 # H-inf bound

Ak, Bk, Ck, Dk, rcond, info = sb10ad(
    n, m1 + m2, p1 + p2,
    ncon, nmeas, gamma,
    A.copy(), B.copy(), C.copy(), D.copy()
)
# Returns controller K: u = Ck @ xk + Dk @ y
```

## 6. Controllability Check

Verify controllability and find controllable realization.

```python
import numpy as np
from slicot import ab01nd

n, m = 4, 2
A = ...  # n x n
B = ...  # n x m

ncont, indcon, nblk, Z, tau, info = ab01nd(
    'I',    # form transformation matrix
    n, m,
    A.copy(), B.copy(), 0.0
)

if ncont == n:
    print("System is controllable")
else:
    print(f"Controllable subspace dimension: {ncont}")
    # A[:ncont, :ncont] is controllable part
```

## 7. Continuous to Discrete Conversion

Bilinear (Tustin) transformation.

```python
import numpy as np
from slicot import ab04md

n, m, p = 2, 1, 1
Ts = 0.1  # sampling time
alpha = 2.0 / Ts
beta = 1.0

A = np.array([[0, 1], [-2, -3]], dtype=float, order='F')
B = np.array([[0], [1]], dtype=float, order='F')
C = np.array([[1, 0]], dtype=float, order='F')
D = np.array([[0]], dtype=float, order='F')

info = ab04md(
    'C',    # continuous to discrete
    n, m, p,
    alpha, beta,
    A, B, C, D  # modified in-place
)
# A, B, C, D now contain discrete-time system
```

## 8. Lyapunov Equation

Solve A'X + XA + Q = 0 (or discrete variant).

```python
import numpy as np
from slicot import sb03md

n = 3
A = np.array([[-1, 0.5, 0], [0, -2, 0.5], [0, 0, -3]], dtype=float, order='F')
C = np.eye(n, dtype=float, order='F')  # Q = C'C

scale, sep, ferr, wr, wi, info = sb03md(
    'C',    # continuous
    'X',    # solve for X
    'N',    # not factored
    'T',    # transpose form
    n,
    A.copy(),
    C,      # Q on input, X on output
    0.0
)
X = C  # Solution
```
