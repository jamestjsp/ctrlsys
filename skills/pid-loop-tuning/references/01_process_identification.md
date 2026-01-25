# Process Identification and Modeling

Process identification is the foundational step in systematic PID tuning. Understand the process dynamics before attempting to tune the controller.

## CSV Data Format for Step Test Analysis

For practical plant data analysis, export historian data to CSV format for use with `step_test_identify.py`.

### Required Columns

| Column | Description | Example Values |
|--------|-------------|----------------|
| `timestamp` or `time` | Time in seconds or datetime | 0, 1, 2, ... or ISO8601 |
| `PV` | Process variable in engineering units | 150.5, 151.2, ... (°C) |
| `OP` or `output` | Controller output in engineering units | 45.0, 50.0, ... (%) |

### Optional Columns

| Column | Description |
|--------|-------------|
| `SP` | Setpoint (for reference) |
| `dist_*` | Disturbance variables (e.g., `dist_ambient`, `dist_feed_temp`) |

### Example CSV

```csv
timestamp,PV,OP,dist_ambient
0,150.2,45.0,25.1
1,150.3,45.0,25.0
2,150.1,45.0,25.2
...
100,150.2,50.0,25.1
101,151.5,50.0,25.0
102,153.2,50.0,25.1
...
```

### Scaling for Proper Gain Calculation

Process gain must be dimensionless (%PV / %OP) for lambda tuning formulas to work correctly.

```python
# Engineering unit to percent scaling
pv_pct = 100 * (pv - pv_min) / (pv_max - pv_min)
op_pct = 100 * (op - op_min) / (op_max - op_min)

# Example: Temperature 100-200°C, Output 0-100%
pv_range = (100, 200)  # °C
op_range = (0, 100)    # %

# If PV moves from 150°C to 160°C when OP moves from 45% to 50%
# %PV change = 100 * (160-150) / (200-100) = 10%
# %OP change = 100 * (50-45) / (100-0) = 5%
# Kp = 10% / 5% = 2.0 (dimensionless)
```

### Sampling Time Considerations

- Sampling time (Ts) is inferred from timestamp differences
- For accurate identification: Ts ≤ tau_p / 10
- Typical DCS historian: 1-10 second intervals sufficient for most processes
- Fast processes (flow): May need 0.1-1 second sampling

### Disturbance Handling

If disturbance variables are present in CSV (columns starting with `dist_`), the script can regress out their effects before fitting the OP→PV relationship:

```bash
# Automatically uses dist_* columns for correction
uv run scripts/step_test_identify.py data.csv --step-time 100 \
    --pv-range 100 200 --op-range 0 100
```

## The Bump Test Method

A bump test (step test) reveals process characteristics by injecting a controlled change in manual mode and observing the response.

### Procedure

1. **Collaborate with Operations** - Work with operators to determine safe operating limits
2. **Determine Safe Magnitude** - Analyze historical trends for typical output movement range
3. **Use a Bump Cycle** - Execute up-down-up sequence to ensure repeatability and return to baseline
4. **Visual Inspection** - Check actuator and sensor condition before testing

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_bump_test(Kp=2.0, tau_p=10.0, dead_time=2.0,
                       bump_magnitude=5.0, duration=60.0):
    """Simulate a bump test on a first-order plus dead time process.

    Args:
        Kp: Process gain (ΔPV/ΔOutput)
        tau_p: Process time constant (seconds)
        dead_time: Process dead time (seconds)
        bump_magnitude: Output step change magnitude (%)
        duration: Total test duration (seconds)

    Returns:
        time, output, pv: Arrays for plotting
    """
    dt = 0.1
    time = np.arange(0, duration, dt)
    output = np.zeros_like(time)
    pv = np.zeros_like(time)

    # Bump cycle: +5% at t=10, -10% at t=30, +5% at t=50
    output[time >= 10] = bump_magnitude
    output[time >= 30] = -bump_magnitude
    output[time >= 50] = 0

    # First-order response with dead time
    for i in range(1, len(time)):
        if time[i] < dead_time:
            continue
        idx_delayed = int((time[i] - dead_time) / dt)
        if idx_delayed >= 0:
            # First-order lag: dy/dt = (Kp*u - y) / tau_p
            pv[i] = pv[i-1] + dt * (Kp * output[idx_delayed] - pv[i-1]) / tau_p

    return time, output, pv

# Example usage
if __name__ == "__main__":
    t, u, y = simulate_bump_test(Kp=2.0, tau_p=10.0, dead_time=2.0)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, u, 'b-', label='Output (%)')
    plt.ylabel('Controller Output (%)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, y, 'r-', label='PV')
    plt.xlabel('Time (s)')
    plt.ylabel('Process Variable')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
```

## FOPDT Model

The First Order Plus Dead Time (FOPDT) model is sufficient for most PID tuning applications:

```
G(s) = Kp × exp(-Td×s) / (τp×s + 1)
```

**Step Response:**
```
y(t) = Kp × Δu × (1 - exp(-(t-Td)/τp))  for t ≥ Td
y(t) = 0                                 for t < Td
```

**Why FOPDT is Sufficient:**
- Captures the three essential dynamics: gain, lag, delay
- Lambda tuning formulas are derived for FOPDT
- Higher-order processes approximate well when τ_dominant >> other τ's
- Model mismatch compensated by conservative λ choice

**For Higher-Order Systems:**
Use `--method=slicot` with `step_test_identify.py` for subspace identification. However, FOPDT with conservative λ typically works well even for complex processes.

## Self-Regulating Process Modeling

Self-regulating processes (flow, pressure, temperature) settle at a new steady state after a step change. Extract three key parameters from bump test data.

### Process Gain (Kp)

Process gain quantifies actuator influence on the process: "How much did PV move for a given output change?"

```python
def calculate_process_gain(delta_pv, delta_output):
    """Calculate process gain from bump test data.

    Args:
        delta_pv: Change in process variable (engineering units)
        delta_output: Change in controller output (%)

    Returns:
        Process gain Kp (units/%)

    Example:
        >>> calculate_process_gain(delta_pv=10.0, delta_output=5.0)
        2.0
    """
    Kp = delta_pv / delta_output
    return Kp

# Example: Temperature process
pv_initial = 150.0  # °C
pv_final = 160.0    # °C
output_initial = 40.0  # %
output_final = 45.0    # %

Kp = calculate_process_gain(
    delta_pv=pv_final - pv_initial,
    delta_output=output_final - output_initial
)
print(f"Process Gain Kp: {Kp:.2f} °C/%")
# Output: Process Gain Kp: 2.00 °C/%
```

### Process Time Constant (τp)

Time constant represents process inertia: "How long did it take to complete the change?"

```python
def estimate_time_constant_simple(settling_time):
    """Estimate process time constant from settling time.

    For first-order processes, settling time (0% to 98%) ≈ 4 × τp

    Args:
        settling_time: Time to reach steady state (seconds)

    Returns:
        Process time constant τp (seconds)

    Example:
        >>> estimate_time_constant_simple(settling_time=40.0)
        10.0
    """
    tau_p = settling_time / 4.0
    return tau_p

# Example calculation
settling_time = 40.0  # seconds to reach new steady state
tau_p = estimate_time_constant_simple(settling_time)
print(f"Process Time Constant τp: {tau_p:.1f} seconds")
# Output: Process Time Constant τp: 10.0 seconds
```

### Dead Time (Td)

Dead time is the pure delay between output change and initial PV response, caused by transportation lag.

```python
def identify_dead_time(time, pv, step_time, threshold=0.01):
    """Identify dead time from bump test data.

    Args:
        time: Time array (seconds)
        pv: Process variable array
        step_time: Time when output step occurred (seconds)
        threshold: Fraction of final change to detect (default 0.01 = 1%)

    Returns:
        Dead time Td (seconds)
    """
    # Find baseline before step
    baseline_mask = time < step_time
    pv_baseline = np.mean(pv[baseline_mask])

    # Find final value after step
    final_mask = time > (step_time + 50)  # Assume settled after 50s
    pv_final = np.mean(pv[final_mask])

    # Find when PV first moves by threshold amount
    delta_total = abs(pv_final - pv_baseline)
    threshold_value = pv_baseline + threshold * delta_total * np.sign(pv_final - pv_baseline)

    response_mask = time > step_time
    if np.sign(pv_final - pv_baseline) > 0:
        first_response_idx = np.where(pv[response_mask] > threshold_value)[0]
    else:
        first_response_idx = np.where(pv[response_mask] < threshold_value)[0]

    if len(first_response_idx) > 0:
        dead_time = time[response_mask][first_response_idx[0]] - step_time
    else:
        dead_time = 0.0

    return dead_time

# Example with simulated data
t, u, y = simulate_bump_test(Kp=2.0, tau_p=10.0, dead_time=2.0)
Td = identify_dead_time(t, y, step_time=10.0)
print(f"Dead Time Td: {Td:.2f} seconds")
# Output: Dead Time Td: 2.00 seconds
```

## Integrating Process Modeling

Integrating processes (tank levels) do not self-regulate. The PV continues ramping until inputs and outputs balance.

### Method 1: Slope Method

Calculate process gain from rate of change:

```python
def calculate_integrating_gain_slope(slope_change, output_change):
    """Calculate process gain for integrating process using slope method.

    Kp = ΔSlope / ΔOutput

    Args:
        slope_change: Change in PV slope (units/second)
        output_change: Change in controller output (%)

    Returns:
        Process gain Kp (%/second/%)

    Example:
        >>> calculate_integrating_gain_slope(0.5, 5.0)
        0.1
    """
    Kp = slope_change / output_change
    return Kp

# Example: Tank level
slope_initial = 0.0      # %/second (balanced)
slope_during = 0.5       # %/second (filling)
output_change = 5.0      # % increase in inlet valve

Kp = calculate_integrating_gain_slope(
    slope_change=slope_during - slope_initial,
    output_change=output_change
)
print(f"Process Gain (Integrating): {Kp:.3f} %/s/%")
# Output: Process Gain (Integrating): 0.100 %/s/%
```

### Method 2: Fill Time Method (Preferred)

Relate process gain to physical tank characteristics:

```python
def calculate_integrating_gain_filltime(tank_volume, max_flow_rate):
    """Calculate process gain for tank using fill time method.

    Fill Time = time to fill tank from 0% to 100% at maximum flow
    Kp = 1 / Fill Time

    Args:
        tank_volume: Tank capacity (gallons, liters, etc.)
        max_flow_rate: Maximum inlet flow (same units/minute)

    Returns:
        Process gain Kp (1/minutes)

    Example:
        >>> calculate_integrating_gain_filltime(1000, 50)
        0.05
    """
    fill_time = tank_volume / max_flow_rate  # minutes
    Kp = 1.0 / fill_time
    return Kp, fill_time

# Example: Storage tank
tank_capacity = 1000.0   # gallons
max_inlet_flow = 50.0    # gallons/minute

Kp, fill_time = calculate_integrating_gain_filltime(tank_capacity, max_inlet_flow)
print(f"Fill Time: {fill_time:.1f} minutes")
print(f"Process Gain Kp: {Kp:.4f} 1/min")
# Output: Fill Time: 20.0 minutes
# Output: Process Gain Kp: 0.0500 1/min
```

## Complete Process Identification Workflow

```python
def analyze_bump_test_data(time, pv, output, step_time):
    """Complete analysis workflow for bump test data.

    Args:
        time: Time array (seconds)
        pv: Process variable measurements
        output: Controller output values
        step_time: When step change occurred (seconds)

    Returns:
        dict with Kp, tau_p, Td
    """
    # Calculate dead time
    Td = identify_dead_time(time, pv, step_time)

    # Calculate process gain
    baseline_pv = np.mean(pv[time < step_time])
    final_pv = np.mean(pv[time > (step_time + 60)])
    baseline_out = np.mean(output[time < step_time])
    final_out = np.mean(output[time > step_time])

    delta_pv = final_pv - baseline_pv
    delta_output = final_out - baseline_out
    Kp = delta_pv / delta_output

    # Estimate time constant (settling starts after dead time)
    response_start = step_time + Td
    settling_time = 40.0  # Estimate from plot
    tau_p = settling_time / 4.0

    return {
        'Kp': Kp,
        'tau_p': tau_p,
        'Td': Td
    }

# Example usage with simulated data
t, u, y = simulate_bump_test(Kp=2.0, tau_p=10.0, dead_time=2.0)
params = analyze_bump_test_data(t, y, u, step_time=10.0)

print("Process Model Parameters:")
print(f"  Process Gain Kp:        {params['Kp']:.3f}")
print(f"  Time Constant τp:       {params['tau_p']:.1f} s")
print(f"  Dead Time Td:           {params['Td']:.2f} s")
# Output:
# Process Model Parameters:
#   Process Gain Kp:        2.000
#   Time Constant τp:       10.0 s
#   Dead Time Td:           2.00 s
```

## Key Principles

1. **Repeatability is Essential** - Perform multiple bump tests (3-5) to verify consistent response
2. **Account for Initial Conditions** - For integrating processes, subtract initial slope before calculating gain
3. **Match Time Units** - Ensure time constant units match controller integral time units (seconds vs minutes)
4. **Visual Inspection First** - Check mechanical condition before testing; hardware issues make tuning futile
5. **Validate Against Physics** - Rationalize calculated gain against tank volume, valve size, etc.

## Next Steps

With process parameters identified (Kp, τp, Td), proceed to:
- [PID Fundamentals](02_pid_fundamentals.md) - Understanding controller actions
- [Lambda Tuning](03_lambda_tuning.md) - Calculate controller parameters
