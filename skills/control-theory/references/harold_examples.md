# Harold Control Theory Code Examples

A practical reference of control theory code examples using harold, extracted from the harold codebase test suite and documentation.

## 1. Transfer Function Creation and Manipulation

### SISO Transfer Functions

```python
from harold import Transfer
import numpy as np

# Simple SISO transfer function: 1/(s^2 + 2s + 1)
G = Transfer(1, [1, 2, 1])

# SISO with numerator dynamics: (s + 1)/(s^2 + s + 1)
G = Transfer([1, 1], [1, 1, 1])

# Static gain as transfer function
G = Transfer(5)

# Discrete-time transfer function with sampling period 0.1s
G = Transfer([1, 1], [1, 1, 1], dt=0.1)

# Transfer function with integrator: s/(s^2 + s)
G = Transfer([1, 0], [1, 1, 0])
```

### MIMO Transfer Functions

```python
# 2x2 MIMO system with individual numerators and denominators
num = [[[1, 3, 2], [1, 3]],
       [[1], [1, 0]]]
den = [[[1, 2, 1], [1, 3, 3]],
       [[1, 0, 0], [1, 2, 3, 4]]]
G = Transfer(num, den)

# MIMO with common denominator
num = [[[1, 3, 2], [1, 3]],
       [[1], [1, 0]]]
den = [1, 2, 3, 4, 5]  # Common denominator for all elements
G = Transfer(num, den)

# Static gain matrix (no dynamics)
G = Transfer(np.eye(3))
G = Transfer(np.arange(1, 10).reshape(3, 3))
```

### Transfer Function Algebra

```python
# Addition and subtraction
G = Transfer(1, [1, 2, 1])
H = Transfer(1, [1, 3])
F = G + H  # Parallel connection

# Multiplication (series connection)
F = G * H  # Series connection

# Matrix multiplication (@ operator for MIMO)
G = Transfer([[1, [1, 1]]], [[[1, 2, 1], [1, 1]]])
H = Transfer([[[1, 3]], [1]], [1, 2, 1])
F = G @ H  # Matrix product

# Scalar multiplication
F = 0.5 * G
F = G / 2

# Array multiplication
G = Transfer([[1, 2]], [1, 1])
H = np.array([2, 1]) * G  # Element-wise gain
```

## 2. State-Space Representation and Analysis

### State-Space Creation

```python
from harold import State
import numpy as np

# SISO state-space: dx/dt = Ax + Bu, y = Cx + Du
A = np.array([[0, 1], [-4, -5]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])
G = State(A, B, C, D)

# Strictly proper (D=0) - omit D matrix
G = State(A, B, C)

# Static gain
G = State(5)  # Scalar gain
G = State(np.eye(3))  # 3x3 identity gain

# Discrete-time state-space with sampling period
G = State(A, B, C, D, dt=0.1)

# MIMO state-space (3 states, 2 inputs, 2 outputs)
A = np.random.rand(3, 3)
B = np.random.rand(3, 2)
C = np.random.rand(2, 3)
D = np.zeros((2, 2))
G = State(A, B, C, D)
```

### State-Space Algebra

```python
# Series connection
G1 = State([[0, 1], [-1, -2]], [[0], [1]], [[1, 0]], [[0]])
G2 = State([[-3]], [[1]], [[1]], [[0]])
F = G1 @ G2  # Output of G2 feeds into G1

# Parallel connection
F = G1 + G2

# Feedback connection
from harold import feedback
CL = feedback(G1, G2)  # Negative feedback: G1 / (1 + G1*G2)

# Scalar operations
F = 2 * G1
F = G1 / 2
```

## 3. System Analysis

### Poles, Zeros, and Stability

```python
from harold import Transfer, State, transmission_zeros
import numpy as np

# Poles and zeros of transfer function
G = Transfer([1, 1], [1, 3, 3, 1])
poles = G.poles  # Array of pole locations
zeros = G.zeros  # Array of zero locations
# Check stability: all poles must have negative real parts
is_stable = np.all(poles.real < 0)

# Poles of state-space system
A = np.array([[0, 1], [-4, -5]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
G = State(A, B, C)
poles = G.poles  # Eigenvalues of A matrix

# Transmission zeros for MIMO systems
A = np.random.rand(4, 4)
B = np.random.rand(4, 2)
C = np.random.rand(3, 4)
D = np.zeros((3, 2))
zeros = transmission_zeros(A, B, C, D)

# Discrete-time stability (poles inside unit circle)
G = Transfer([0.1, 0.1], [1, -0.8], dt=0.1)
poles = G.poles
# For discrete systems, stability requires |poles| < 1
is_stable = np.all(np.abs(poles) < 1)
```

### DC Gain and Frequency Response

```python
from harold import frequency_response

# DC gain (steady-state gain)
G = Transfer(100, [1, 10, 100])
dc = G.dcgain

# Frequency response
G = Transfer([1, 1], [1, 2, 1])
freq_response, frequencies = frequency_response(G)

# Frequency response for discrete system
G = Transfer([1, -1], [1, -2, 1], dt=0.1)
freq_response, frequencies = frequency_response(G)

# MIMO frequency response
A = -3*np.eye(5) + np.random.rand(5, 5)
B = np.random.rand(5, 3)
C = np.random.rand(4, 5)
G = State(A, B, C)
freq_response, frequencies = frequency_response(G)
```

### Visualizing Frequency Response with Plots

```python
from harold import bode_plot, nyquist_plot
import matplotlib.pyplot as plt

# Single system Bode plot
G = Transfer([1, 1], [1, 2, 1])
fig = bode_plot(G, use_db=True)
plt.show()

# Nyquist plot
fig = nyquist_plot(G)
plt.show()

# Compare multiple systems on same Bode plot
G1 = Transfer(1, [1, 1])
G2 = Transfer(1, [1, 2])
G3 = Transfer(2, [1, 1, 1])
fig = bode_plot([G1, G2, G3], use_db=True)
plt.show()

# Customize frequency range
w = np.logspace(-2, 2, 100)  # 0.01 to 100 rad/s
fig = bode_plot(G, w=w, use_db=True, use_hz=False)
plt.show()
```

### System Norms

```python
from harold import system_norm

# H-infinity norm (peak frequency response)
G = Transfer([100], [1, 10, 100])
hinf_norm = system_norm(G)  # Default: p=np.inf

# H2 norm (RMS frequency response)
h2_norm = system_norm(G, p=2)

# Works for state-space too
F = transfer_to_state(G)
hinf_norm = system_norm(F)
```

## 4. Control Design

### LQR Design

```python
from harold import lqr, State
import numpy as np

# State weighting LQR for a simple 2-state system
A = np.array([[0, 1], [-1, -2]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
H = State(A, B, C)
Q = np.eye(2)  # State weight matrix (must match number of states)
k, x, eigs = lqr(H, Q)  # Returns gain, solution, closed-loop poles

# MIMO LQR with state weighting
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [4e-6, 0, 0, 0.05],
              [0, 0, -1e-4, 0]])
B = np.array([[0, 0], [1e-5/3, 0], [0, 0], [0, 0.01]])
C = np.eye(4)
H = State(A, B, C)
k, x, eigs = lqr(H[:, 1], np.eye(4))

# Output weighting LQR
k, x, eigs = lqr(H, Q=np.eye(4), weight_on='output')

# Discrete-time LQR
H = State(A, B, C, dt=0.1)
k, x, eigs = lqr(H, np.eye(4))
```

### Pole Placement with Ackermann's Formula

```python
from harold import ackermann, haroldcompanion, State
import numpy as np

# Ackermann pole placement for controllable system
A = haroldcompanion([1, 6, 5, 1])  # Companion matrix
B = np.eye(3)[:, [-1]]
desired_poles = [-10, -9, -8]
K = ackermann((A, B), desired_poles)

# Can also pass State object
G = State(A, B, [[1, 0, 0]])
K = ackermann(G, desired_poles)
```

## 5. Discretization and Sampling

### Discretization Methods

```python
from harold import discretize, State, Transfer
import numpy as np

# Zero-order hold (ZOH) discretization
A = np.eye(2)
B = 0.5 * np.ones((2, 1))
C = np.array([[0.75, 1.0], [1.0, 1.0]])
G = State(A, B, C)
dt = 0.5
Gd = discretize(G, dt=dt, method='zoh')

# First-order hold (FOH)
G = Transfer(1, [1, 0, 0])
Gd = discretize(G, dt=0.1, method='foh')

# Bilinear (Tustin) transformation
G = Transfer([1, 0.5, 9], [1, 5, 9])
Gd = discretize(G, dt=0.5, method='bilinear')

# Tustin with frequency prewarping (preserve response at 3 rad/s)
Gd = discretize(G, dt=0.5, prewarp_at=3/2/np.pi, method='bilinear')

# Forward/backward Euler
Gd = discretize(G, dt=0.1, method='>>')  # Forward Euler
Gd = discretize(G, dt=0.1, method='<<')  # Backward Euler

# Custom LFT discretization
Q = np.array([[1, 0.5], [0.5, 0]])  # Custom discretization matrix
Gd = discretize(G, dt=0.25, method='lft', q=Q)
```

### Undiscretization

```python
from harold import undiscretize

# Convert discrete system back to continuous
Gd = Transfer([0.048, 0.047], [1, -0.905], dt=0.1)
Gd.DiscretizedWith = 'foh'  # Specify original method
Gc = undiscretize(Gd)
```

## 6. Time-Domain Simulation

### Step and Impulse Response

```python
from harold import simulate_step_response, simulate_impulse_response, State, Transfer
import numpy as np

# Step response of continuous system
G = Transfer(1, [1, 1, 1])
y, t = simulate_step_response(G)

# Step response with custom time vector
t_custom = [0, 1, 2, 3]
y, t = simulate_step_response(G, t_custom)

# Impulse response
G = State(4)
y, t = simulate_impulse_response(G)

# Discrete-time step response
G = State(4, dt=0.01)
y, t = simulate_step_response(G)
```

### Visualizing Time Response with Plots

```python
from harold import step_response_plot, impulse_response_plot
import matplotlib.pyplot as plt

# Step response plot
G = Transfer(1, [1, 1, 1])
ax = step_response_plot(G)
plt.show()

# Impulse response plot
ax = impulse_response_plot(G)
plt.show()

# Compare multiple systems
G1 = Transfer(1, [1, 1, 1])
G2 = Transfer(1, [1, 2, 1])
ax = step_response_plot([G1, G2])
plt.show()

# Custom time vector
t = np.linspace(0, 10, 200)
ax = step_response_plot(G, t=t)
plt.show()

# Discrete-time systems (automatically uses step plot)
Gd = Transfer([0.1, 0.05], [1, -0.8, 0.15], dt=0.1)
ax = step_response_plot(Gd)
plt.show()
```

### General Linear Simulation

```python
from harold import simulate_linear_system, State, Transfer
import numpy as np

# Simulate with custom input
G = Transfer(1, [1, 1, 1], dt=0.01)
u = [1, 2, 3]  # Input sequence
y, t = simulate_linear_system(G, u)

# MIMO simulation with initial conditions
A = np.eye(3)
B = np.ones([3, 1])
C = np.ones([1, 3])
G = State(A, B, C, dt=0.01)
u = np.ones([10, 1])  # 10 time steps, 1 input
x0 = np.arange(3)  # Initial state
y, t = simulate_linear_system(G, u, x0=x0)

# Continuous-time simulation (requires time vector)
G = Transfer(1, [1, 1, 1])
t = np.linspace(0, 10, 100)
u = np.ones(100)
y, t = simulate_linear_system(G, u, t)
```

## 7. Conversions and Realizations

### Transfer Function ↔ State-Space

```python
from harold import transfer_to_state, state_to_transfer, Transfer, State
import numpy as np

# Transfer to state-space (creates controllable canonical form)
G = Transfer([1, -8, 28, -58, 67, -30],
             [1, 2, 3, 2, 3, 4, 5, 6])
H = transfer_to_state(G)

# State-space to transfer function
A = np.array([[0, 1], [-1, -2]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
G = State(A, B, C)
H = state_to_transfer(G)

# MIMO conversions work too
num = [[[1], [2]], [[3], [4]]]
den = [[[1, 1], [1, 2]], [[1, 3], [1, 4]]]
G = Transfer(num, den)
H = transfer_to_state(G)
```

### Minimal Realization

```python
from harold import minimal_realization, Transfer, State
import numpy as np

# Reduce transfer function with pole-zero cancellation
G = Transfer([1., -8., 28., -58., 67., -30.],
             np.poly([1, 2, 3., 2, 3., 4, 1+2j, 1-2j]))
H = minimal_realization(G)  # Removes canceled poles

# Reduce state-space system
A = np.array([[-6.5, 0.5, 6.5, -6.5],
              [-0.5, -5.5, -5.5, 5.5],
              [-0.5, 0.5, 0.5, -6.5],
              [-0.5, 0.5, -5.5, -0.5]])
B = np.array([[0., 1., 0.], [2., 1., 2.], [3., 4., 3.], [3., 2., 3.]])
C = np.array([[1., 1., 0., 0.]])
G = State(A, B, C)
H = minimal_realization(G)  # Returns minimal realization
```

### Canonical Realizations

```python
from harold import hessenberg_realization, Transfer
import numpy as np

# Hessenberg (upper or lower) realization
G = Transfer([1., -8., 28., -58., 67., -30.],
             np.poly([1, 2, 3., 2, 3., 4, 1+1j, 1-1j]), dt=0.1)

# Controllable Hessenberg form
H, T = hessenberg_realization(G, compute_T=True, form='c', invert=True)

# Observable Hessenberg form
H = hessenberg_realization(G, form='o', invert=True)
```

## 8. System Properties and Tests

### Controllability and Observability

```python
from harold import (controllability_matrix, observability_matrix,
                   is_kalman_controllable, is_kalman_observable,
                   kalman_decomposition, State)
import numpy as np

# Controllability matrix and test
A = np.arange(1, 5).reshape(2, 2)
B = np.array([[5], [7]])
Wc, rank, col_indices = controllability_matrix((A, B))
is_controllable = is_kalman_controllable((A, B))

# Observability matrix and test
C = np.array([[5, 7]])
Wo, rank, row_indices = observability_matrix((A, C))
is_observable = is_kalman_observable((A, C))

# Kalman decomposition (separate controllable/uncontrollable parts)
A = np.array([[2, 1, 1], [5, 3, 6], [-5, -1, -4]])
B = np.array([[1], [0], [0]])
C = np.array([[1, 0, 0]])
G = State(A, B, C)
F = kalman_decomposition(G)  # Returns transformed system
```

### Controllability Indices

```python
from harold import controllability_indices
import numpy as np

# Find controllability indices for MIMO system
A = np.array([[1.38, -0.21, 6.72, -5.68],
              [-0.58, -4.29, 0, 0.68],
              [1.07, 4.27, -6.65, 5.89],
              [0.05, 4.27, 1.34, -2.10]])
B = np.array([[0, 0], [5.68, 0], [1.14, -3.15], [1.14, 0]])
indices = controllability_indices(A, B)  # e.g., [2, 2]
```

## 9. Utility Functions

### Random System Generation

```python
from harold import random_state_model
import numpy as np

# Generate random SISO system with 5 states
G = random_state_model(5)

# Generate stable random system (all poles in LHP)
G = random_state_model(5, stable=True)

# Generate MIMO system (5 states, 2 inputs, 4 outputs)
G = random_state_model(5, 2, 4)

# Generate discrete-time random system
G = random_state_model(11, dt=0.1)  # Poles inside unit circle

# Generate with specific pole distribution
# prob_dist = [real_stable, real_unstable, complex_stable, complex_unstable]
G = random_state_model(11, stable=False, prob_dist=[0, 0, 0.5, 0.5])
```

### Matrix Utilities

```python
from harold import matrix_slice, concatenate_state_matrices, e_i, haroldcompanion
import numpy as np

# Extract A, B, C, D from augmented matrix
M = np.array([[-6.5, 0.5, 1.],
              [-0.5, -5.5, 2.],
              [1., 1., 0.]])
A, B, C, D = matrix_slice(M, (1, 1), corner='sw')  # 2x2 A, others sized accordingly

# Concatenate state-space matrices back into augmented form
G = State(np.eye(2), [[1], [0]], [[1, 0]], [[0]])
M = concatenate_state_matrices(G)  # Returns [A | B; C | D]

# Generate standard basis vector
vec = e_i(5, 2)  # [0, 0, 1, 0, 0] (3rd element is 1)
vec = e_i(5, [0, 3])  # Multiple indices: [1, 0, 0, 1, 0]
row = e_i(5, 2, output='r')  # Row vector instead of column

# Create companion matrix from polynomial coefficients
A = haroldcompanion([1, 6, 5, 1])  # Companion for s^3 + 6s^2 + 5s + 1
```

This comprehensive reference covers the main control theory operations available in harold. All examples are taken from the actual harold codebase tests and demonstrate practical, working code patterns.
