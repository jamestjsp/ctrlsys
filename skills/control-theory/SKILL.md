---
name: control-theory
description: This skill should be used when working with control theory problems, implementing linear time-invariant (LTI) systems, designing controllers, analyzing system dynamics, or learning about computer-controlled systems using slicot.
license: MIT
---

# Control Theory with SLICOT

This skill provides practical guidance for implementing and analyzing linear time-invariant (LTI) control systems using the slicot library (Python bindings for SLICOT), grounded in IFAC computer control principles.

## Purpose

Control theory is fundamental to engineering systems from simple regulators to complex autonomous systems. This skill helps translate theoretical control concepts into practical implementations using slicot, covering system representations, analysis, design, and discretization.

**Key SLICOT routines used:**
- `sb02md` - Riccati equation solver for LQR design
- `ab04md` - Bilinear (Tustin) discretization
- `tf01md` - Discrete-time state-space simulation
- `tb05ad` - Frequency response evaluation

## When to Use This Skill

To apply this skill, first determine the control theory task:

- **System Representation**: Creating and manipulating transfer functions or state-space models (SISO/MIMO)
- **System Analysis**: Computing poles, zeros, stability, frequency response, norms, controllability, observability
- **Control Design**: Pole placement, LQR optimization, observer design, Kalman filtering
- **Discretization**: Converting continuous systems to discrete-time (ZOH, FOH, Tustin, etc.)
- **Simulation**: Time-domain and frequency-domain responses
- **Learning**: Understanding IFAC computer control principles and best practices

This skill activates when users request help with these domains using slicot, or when they need to understand the theoretical foundations of computer-controlled systems.

## How to Use This Skill

### Step 1: Identify the System Type

Determine whether the problem involves:
- **Continuous or discrete-time** systems
- **SISO** (single-input single-output) or **MIMO** (multi-input multi-output)
- **Transfer function** or **state-space** representation

### Step 2: Reference Practical Code Examples

The scripts in `scripts/` provide working code patterns organized by functionality:
- `lqr_design.py` - LQR controller design using `sb02md`
- `discretize_controller.py` - Discretization using `ab04md`
- `analyze_system.py` - System analysis with frequency response via `tb05ad`
- `pid_controller.py` - PID control simulation using slicot

All arrays use Fortran column-major order (`order='F'` in NumPy) as required by SLICOT.

### Step 3: Understand Theoretical Foundations

For deeper understanding of computer-controlled systems theory, reference `IFAC_Computer_Control.md`. This comprehensive guide covers:
- Sampling and reconstruction (Chapters 2-3)
- Mathematical models and frequency response (Chapters 4-5)
- Control design and specification methods (Chapters 6-10)
- Practical implementation issues (Chapters 11-12)
- Real-time control and performance (Chapters 13-14)

Use the IFAC reference when:
- Implementing discretization methods (section on aliasing, prewarping)
- Designing controllers for sampled systems
- Understanding practical constraints (quantization, saturation, timing)
- Selecting between design approaches (H∞, LQ, pole placement)

### Step 4: Build Incrementally

When implementing control systems:

1. Start with a simple representation (create transfer function or state-space model)
2. Verify system properties (check stability, poles/zeros, controllability)
3. Design the controller (pole placement, LQR, or observer)
4. Discretize for implementation (choose appropriate method and sampling period)
5. Validate in simulation (step response, frequency response, closed-loop performance)
6. Implement in real-time (account for quantization, saturation, numerical issues)

## Key Concepts from SLICOT

SLICOT works with state-space representation (A, B, C, D matrices) using raw numpy arrays:

### State-Space Models
- A, B, C, D matrix representation in Fortran column-major order
- Natural for time-domain simulation and design
- MIMO with arbitrary dimensions
- Properties: poles (A eigenvalues via `np.linalg.eigvals`)

```python
import numpy as np

# Create state-space matrices (Fortran order)
A = np.array([[0, 1], [-1, -2]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)
C = np.array([[1, 0]], order='F', dtype=float)
D = np.array([[0]], order='F', dtype=float)

# Check stability via eigenvalues
poles = np.linalg.eigvals(A)
is_stable = np.all(poles.real < 0)  # Continuous: negative real parts
print(f"Stable: {is_stable}, Poles: {poles}")
```

### Transfer Function to State-Space Conversion
Use a helper function to convert from transfer function to controllable canonical form:

```python
def tf_to_ss(num, den):
    """Convert SISO transfer function to controllable canonical state-space."""
    num = np.atleast_1d(num).astype(float)
    den = np.atleast_1d(den).astype(float)
    den = den / den[0]
    num = num / den[0]
    n = len(den) - 1
    if n == 0:
        return (np.zeros((0, 0), order='F'),
                np.zeros((0, 1), order='F'),
                np.zeros((1, 0), order='F'),
                np.array([[num[0]]], order='F'))
    num_padded = np.zeros(n + 1)
    num_padded[n + 1 - len(num):] = num
    A = np.zeros((n, n), order='F')
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]
    B = np.zeros((n, 1), order='F')
    B[-1, 0] = 1.0
    C = np.zeros((1, n), order='F')
    d0 = num_padded[0]
    for i in range(n):
        C[0, n - 1 - i] = num_padded[i + 1] - d0 * den[i + 1]
    D = np.array([[d0]], order='F')
    return A, B, C, D
```

## Common Control Theory Workflows

### Discretize a Continuous Controller

Use `ab04md` for bilinear (Tustin) transformation:

```python
from slicot import ab04md
import numpy as np

# Continuous state-space (A, B, C, D)
A = np.array([[0, 1], [-1, -2]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)
C = np.array([[1, 0]], order='F', dtype=float)
D = np.array([[0]], order='F', dtype=float)

dt = 0.01  # Sampling period
# Tustin transformation: alpha=1, beta=2/dt
A_d, B_d, C_d, D_d, info = ab04md('C', A.copy(), B.copy(), C.copy(), D.copy(), alpha=1.0, beta=2.0/dt)
```

### Design a Discrete-Time LQR Controller

Use `sb02md` to solve the Riccati equation:

```python
from slicot import sb02md, ab04md
import numpy as np

# Continuous system
A = np.array([[0, 1], [-1, -2]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)
C = np.eye(2, order='F', dtype=float)
D = np.zeros((2, 1), order='F', dtype=float)

# Discretize first
dt = 0.01
A_d, B_d, C_d, D_d, _ = ab04md('C', A.copy(), B.copy(), C.copy(), D.copy(), alpha=1.0, beta=2.0/dt)

# LQR design: solve discrete Riccati equation
n = A_d.shape[0]
Q = np.eye(n, order='F', dtype=float)  # State weight
R = np.array([[1.0]], order='F', dtype=float)  # Input weight
R_inv = np.linalg.inv(R)
G = (B_d @ R_inv @ B_d.T).astype(float, order='F')

X, rcond, wr, wi, S, U, info = sb02md('D', 'D', 'U', 'N', 'U', n, A_d.copy(), G.copy(), Q.copy())
K = np.linalg.inv(R + B_d.T @ X @ B_d) @ B_d.T @ X @ A_d  # Feedback gain
```

### Analyze Discrete System Stability

For discrete-time systems, stability requires poles inside unit circle (|z| < 1):

```python
import numpy as np

# Discrete system (from ab04md output)
poles = np.linalg.eigvals(A_d)
is_stable = np.all(np.abs(poles) < 1)  # Discrete: inside unit circle
if is_stable:
    print(f"System stable. Poles: {poles}")
else:
    unstable = poles[np.abs(poles) >= 1]
    print(f"Unstable poles: {unstable}")
```

### Test Controllability Before Design

Before applying pole placement or LQR, verify the system is controllable:

```python
import numpy as np

def controllability_matrix(A, B):
    """Compute controllability matrix [B, AB, A^2B, ...]."""
    n = A.shape[0]
    m = B.shape[1]
    C_mat = np.zeros((n, n * m))
    for i in range(n):
        C_mat[:, i*m:(i+1)*m] = np.linalg.matrix_power(A, i) @ B
    return C_mat

A = np.array([[0, 1], [-1, -2]], dtype=float)
B = np.array([[0], [1]], dtype=float)
C_mat = controllability_matrix(A, B)
rank = np.linalg.matrix_rank(C_mat)
if rank == A.shape[0]:
    print("System is controllable - pole placement will work")
else:
    print("Uncontrollable modes exist - design will fail")
```

### Visualize Controller Performance

After designing a controller, use slicot and matplotlib to analyze closed-loop performance:

```python
from slicot import tf01md, ab04md, tb05ad
import matplotlib.pyplot as plt
import numpy as np

# Helper functions
def tf_to_ss(num, den):
    # ... (see above)
    pass

def ss_series(sys1, sys2):
    """Connect two state-space systems in series."""
    A1, B1, C1, D1 = sys1
    A2, B2, C2, D2 = sys2
    n1, n2 = A1.shape[0], A2.shape[0]
    A = np.zeros((n1 + n2, n1 + n2), order='F', dtype=float)
    A[:n1, :n1] = A1
    A[n1:, :n1] = B2 @ C1
    A[n1:, n1:] = A2
    B = np.vstack([B1, B2 @ D1]).astype(float, order='F')
    C = np.hstack([D2 @ C1, C2]).astype(float, order='F')
    D = (D2 @ D1).astype(float, order='F')
    return A, B, C, D

def ss_feedback(sys, k=1.0):
    """Negative unity feedback: L / (1 + k*L)."""
    A, B, C, D = sys
    inv_term = 1.0 / (1.0 + k * D[0, 0])
    Af = (A - k * inv_term * B @ C).astype(float, order='F')
    Bf = (inv_term * B).astype(float, order='F')
    Cf = (inv_term * C).astype(float, order='F')
    Df = (inv_term * D).astype(float, order='F')
    return Af, Bf, Cf, Df

# Plant and controller
G = tf_to_ss([1], [1, 2, 1])  # 1/(s^2 + 2s + 1)
C = tf_to_ss([2.0, 1.0], [1, 0])  # PI controller

# Open-loop and closed-loop
L = ss_series(G, C)
T = ss_feedback(L, 1.0)

# Simulate step response
dt = 0.01
A_d, B_d, C_d, D_d, _ = ab04md('C', T[0].copy(), T[1].copy(), T[2].copy(), T[3].copy(), alpha=1.0, beta=2.0/dt)
t = np.arange(0, 10, dt)
u = np.ones((1, len(t)), order='F', dtype=float)
y, _, _ = tf01md(A_d, B_d, C_d, D_d, u, np.zeros(A_d.shape[0]))

plt.plot(t, y.flatten())
plt.title('Closed-Loop Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.grid()
plt.show()
```

This workflow demonstrates the complete design cycle:
1. Define plant and controller as state-space
2. Form open-loop and closed-loop systems
3. Discretize and simulate step response
4. Verify stability via eigenvalues

## References and Standards

- **IFAC Computer Control**: Comprehensive reference on computer-controlled systems, discretization, and practical implementation
- **SLICOT Test Suite**: `tests/python/test_*.py` contains working examples for all routines
- **SLICOT Documentation**: See `skills/slicot-control/SKILL.md` for routine reference
- **State-Space Theory**: Properties of A, B, C, D matrices and their interpretation in control design

## Tips for Success

1. **Always verify stability first**: Check eigenvalues before relying on analysis
2. **Test controllability/observability**: Design only works on controllable/observable subsystems
3. **Choose discretization carefully**: Use `ab04md` with Tustin for bandwidth-critical applications
4. **Validate in simulation**: Always simulate step response using `tf01md`
5. **Use Fortran column-major order**: All arrays must use `order='F'` for SLICOT
6. **Account for implementation constraints**: Quantization, saturation, computational limits (see IFAC Ch. 11-12)
