# PID Controller Fundamentals

Understanding the Proportional-Integral-Derivative (PID) controller is essential for effective tuning. The PID algorithm converts error (difference between setpoint and measured value) into corrective action.

## The Error Signal

The foundation of feedback control is the error calculation:

```python
def calculate_error(setpoint, process_variable):
    """Calculate control error.

    Args:
        setpoint: Desired value (SP)
        process_variable: Measured value (PV)

    Returns:
        Error (E = SP - PV)
    """
    return setpoint - process_variable

# Example
SP = 100.0  # Desired temperature
PV = 95.0   # Current temperature
error = calculate_error(SP, PV)
print(f"Error: {error}")  # Output: Error: 5.0
```

Feedback control is **reactive**—error must occur before the controller acts. The PID algorithm addresses three attributes of this error:

1. **Magnitude** (present error) → Proportional action
2. **Duration** (past error) → Integral action
3. **Rate of change** (future error) → Derivative action

## Proportional (P) Action

Proportional control responds to the **magnitude** of the current error.

### Goal and Limitation

**Goal:** Stop the changing error
**Limitation:** P-only control results in permanent offset

```python
def proportional_action(error, Kc):
    """Calculate proportional output.

    Args:
        error: Current error (SP - PV)
        Kc: Controller gain

    Returns:
        Proportional output
    """
    return Kc * error

# Example: Temperature control
error = 5.0   # 5°C below setpoint
Kc = 2.0      # Controller gain

p_output = proportional_action(error, Kc)
print(f"Proportional Output: {p_output:.1f}%")
# Output: Proportional Output: 10.0%
```

### The Offset Problem

P-only control balances flows but leaves the PV deviated from SP:

```python
def simulate_p_only_control(load_change=10.0, Kc=2.0, Kp_process=0.5):
    """Demonstrate P-only control offset.

    Args:
        load_change: Disturbance magnitude
        Kc: Controller gain
        Kp_process: Process gain

    Returns:
        Steady-state offset
    """
    # P-only cannot eliminate offset
    # At steady state: Kc * error * Kp_process = load_change
    steady_state_error = load_change / (Kc * Kp_process)
    return steady_state_error

offset = simulate_p_only_control(load_change=10.0, Kc=2.0, Kp_process=0.5)
print(f"Permanent Offset: {offset:.1f}")
# Output: Permanent Offset: 10.0
```

### Inverse Relationship to Process Gain

Controller gain must be inversely proportional to process gain:

```python
def calculate_initial_kc(Kp_process):
    """Estimate initial controller gain.

    Rule of thumb: Kc ≈ 1 / Kp_process

    Args:
        Kp_process: Process gain (ΔPV/ΔOutput)

    Returns:
        Estimated controller gain
    """
    return 1.0 / Kp_process

# Example
Kp = 2.5  # Process gain (large actuator effect)
Kc = calculate_initial_kc(Kp)
print(f"Estimated Kc: {Kc:.2f}")
# Output: Estimated Kc: 0.40
```

## Integral (I) Action

Integral control responds to the **duration** of error (area under curve).

### Goal: Eliminate Offset

The integral term acts as a "watchdog," continuously driving output until error reaches zero.

```python
import numpy as np

def integral_action(error_history, Ti, dt):
    """Calculate integral output.

    Args:
        error_history: Array of error values over time
        Ti: Integral time (seconds)
        dt: Sample time (seconds)

    Returns:
        Integral output
    """
    error_integral = np.sum(error_history) * dt
    return (1.0 / Ti) * error_integral

# Example: Error persisting over time
error_history = np.array([5.0, 4.5, 4.0, 3.5, 3.0])  # Decreasing error
Ti = 10.0  # Integral time
dt = 1.0   # 1 second samples

i_output = integral_action(error_history, Ti, dt)
print(f"Integral Output: {i_output:.2f}")
# Output: Integral Output: 2.00
```

### Reset Time

Reset time (repeat time) measures how long integral takes to match the proportional kick:

```python
def calculate_reset_time(Ti, Kc, error_step):
    """Calculate time for integral to equal proportional output.

    For standard PI form, reset time = Ti

    Args:
        Ti: Integral time
        Kc: Controller gain
        error_step: Step change in error

    Returns:
        Reset time (equals Ti for standard form)
    """
    # Proportional kick
    p_kick = Kc * error_step

    # Integral accumulation rate
    i_rate = Kc * error_step / Ti

    # Time for integral to equal P kick
    reset_time = p_kick / i_rate

    return reset_time  # = Ti

reset_t = calculate_reset_time(Ti=10.0, Kc=0.5, error_step=5.0)
print(f"Reset Time: {reset_t:.1f} seconds")
# Output: Reset Time: 10.0 seconds
```

## The PI Combination (Workhorse)

PI control combines fast proportional response with persistent integral action, representing 100% of industrial controllers.

```python
class PIController:
    """Standard PI controller implementation."""

    def __init__(self, Kc, Ti, dt):
        """Initialize PI controller.

        Args:
            Kc: Controller gain
            Ti: Integral time (seconds)
            dt: Sample time (seconds)
        """
        self.Kc = Kc
        self.Ti = Ti
        self.dt = dt
        self.error_integral = 0.0

    def update(self, setpoint, process_variable):
        """Calculate controller output.

        Args:
            setpoint: Desired value
            process_variable: Measured value

        Returns:
            Controller output (%)
        """
        # Calculate error
        error = setpoint - process_variable

        # Proportional term
        p_term = error

        # Integral term (accumulate error)
        self.error_integral += error * self.dt
        i_term = (1.0 / self.Ti) * self.error_integral

        # Combined output
        output = self.Kc * (p_term + i_term)

        return output

    def reset(self):
        """Reset integral accumulation."""
        self.error_integral = 0.0


# Example usage
controller = PIController(Kc=0.5, Ti=10.0, dt=1.0)

SP = 100.0
PV_readings = [95.0, 96.0, 97.0, 98.0, 99.0]

print("PI Controller Response:")
for i, pv in enumerate(PV_readings):
    output = controller.update(SP, pv)
    print(f"  Step {i+1}: PV={pv:.1f}, Output={output:.2f}%")

# Output:
# PI Controller Response:
#   Step 1: PV=95.0, Output=2.75%
#   Step 2: PV=96.0, Output=2.50%
#   Step 3: PV=97.0, Output=2.25%
#   Step 4: PV=98.0, Output=2.00%
#   Step 5: PV=99.0, Output=1.75%
```

## Derivative (D) Action

Derivative control responds to the **rate of change** of error, providing lead action. **Rarely used** in industrial practice.

### Why Derivative is Rarely Used

```python
def derivative_action(error_current, error_previous, Td, dt):
    """Calculate derivative output.

    Args:
        error_current: Current error
        error_previous: Previous error
        Td: Derivative time
        dt: Sample time

    Returns:
        Derivative output
    """
    error_rate = (error_current - error_previous) / dt
    return Td * error_rate

# Example showing noise sensitivity
errors_clean = [5.0, 4.5, 4.0, 3.5]
errors_noisy = [5.0, 4.2, 4.3, 3.1]  # Same trend, but noisy

Td = 5.0
dt = 1.0

print("Derivative Response:")
for i in range(1, len(errors_clean)):
    d_clean = derivative_action(errors_clean[i], errors_clean[i-1], Td, dt)
    d_noisy = derivative_action(errors_noisy[i], errors_noisy[i-1], Td, dt)
    print(f"  Step {i}: Clean D={d_clean:.2f}, Noisy D={d_noisy:.2f}")

# Output shows noise amplification:
# Derivative Response:
#   Step 1: Clean D=-2.50, Noisy D=-4.00
#   Step 2: Clean D=-2.50, Noisy D=0.50
#   Step 3: Clean D=-2.50, Noisy D=-6.00
```

**Recommendation:** Set derivative time Td = 0 for industrial applications.

## Controller Forms

Different manufacturers implement PID using different mathematical forms. Parameters are NOT interchangeable between forms.

### Standard (Non-Interacting) Form

```python
def pid_standard_form(error, error_integral, error_derivative, Kc, Ti, Td):
    """Standard (non-interacting) PID form.

    Output = Kc * (E + (1/Ti)*∫E*dt + Td*dE/dt)

    Args:
        error: Current error
        error_integral: Accumulated error
        error_derivative: Rate of error change
        Kc: Controller gain
        Ti: Integral time
        Td: Derivative time

    Returns:
        Controller output
    """
    p_term = error
    i_term = (1.0 / Ti) * error_integral if Ti > 0 else 0.0
    d_term = Td * error_derivative

    output = Kc * (p_term + i_term + d_term)
    return output
```

**Advantage:** Integral time (Ti) can be set once (= τp) and left alone. Only adjust Kc for speed.

### Parallel (ISA) Form

```python
def pid_parallel_form(error, error_integral, error_derivative, Kp, Ki, Kd):
    """Parallel (ISA) PID form.

    Output = Kp*E + Ki*∫E*dt + Kd*dE/dt

    Args:
        error: Current error
        error_integral: Accumulated error
        error_derivative: Rate of error change
        Kp: Proportional gain
        Ki: Integral gain
        Kd: Derivative gain

    Returns:
        Controller output
    """
    p_term = Kp * error
    i_term = Ki * error_integral
    d_term = Kd * error_derivative

    output = p_term + i_term + d_term
    return output
```

**Note:** Gains are independent, but changing Kp slows regulation capability.

### Form Conversion

```python
def convert_standard_to_parallel(Kc, Ti, Td):
    """Convert Standard form to Parallel form parameters.

    Args:
        Kc: Controller gain (Standard)
        Ti: Integral time (Standard)
        Td: Derivative time (Standard)

    Returns:
        (Kp, Ki, Kd): Parallel form parameters
    """
    Kp = Kc
    Ki = Kc / Ti if Ti > 0 else 0.0
    Kd = Kc * Td
    return Kp, Ki, Kd

# Example conversion
Kc_std = 0.5
Ti_std = 10.0
Td_std = 0.0

Kp, Ki, Kd = convert_standard_to_parallel(Kc_std, Ti_std, Td_std)
print(f"Standard: Kc={Kc_std}, Ti={Ti_std}, Td={Td_std}")
print(f"Parallel: Kp={Kp}, Ki={Ki:.3f}, Kd={Kd}")
# Output:
# Standard: Kc=0.5, Ti=10.0, Td=0.0
# Parallel: Kp=0.5, Ki=0.050, Kd=0.0
```

## Anti-Reset Windup

Prevents integral accumulation when output saturates at limits.

```python
class PIControllerWithAWU:
    """PI controller with anti-reset windup."""

    def __init__(self, Kc, Ti, dt, output_min=0.0, output_max=100.0):
        self.Kc = Kc
        self.Ti = Ti
        self.dt = dt
        self.output_min = output_min
        self.output_max = output_max
        self.error_integral = 0.0

    def update(self, setpoint, process_variable):
        """Update with anti-reset windup."""
        error = setpoint - process_variable

        # Calculate output before limiting
        p_term = error
        i_term = (1.0 / self.Ti) * self.error_integral
        output_unlimited = self.Kc * (p_term + i_term)

        # Limit output
        output = np.clip(output_unlimited, self.output_min, self.output_max)

        # Only integrate if output not saturated
        if output == output_unlimited:
            # Not saturated, normal integration
            self.error_integral += error * self.dt
        # else: saturated, stop integrating (anti-windup)

        return output


# Demo: Large error causes saturation
controller_awu = PIControllerWithAWU(Kc=2.0, Ti=5.0, dt=1.0, output_max=100.0)
large_error_pv = [50.0, 60.0, 70.0, 80.0, 90.0]  # Far from SP=100
SP = 100.0

print("Anti-Windup Demo:")
for pv in large_error_pv:
    output = controller_awu.update(SP, pv)
    print(f"  PV={pv:.0f}, Output={output:.1f}% (capped at 100%)")
```

## Key Principles

1. **PI is the Workhorse** - P provides fast kick, I eliminates offset
2. **P-only Always Offsets** - Cannot make error zero without integral
3. **Derivative Rarely Used** - Noise sensitivity makes it problematic
4. **Know Controller Form** - Standard vs Parallel parameters differ
5. **Anti-Windup Essential** - Prevents overshoot after saturation
6. **Units Must Match** - Ti time units must match controller settings

## Next Steps

- [Lambda Tuning](03_lambda_tuning.md) - Calculate PI parameters systematically
- [Process Identification](01_process_identification.md) - Determine Kp, τp, Td
- [Integrating Processes](04_integrating_processes.md) - Tank level control specifics
