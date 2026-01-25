# Integrating Processes (Tank Level Control)

Integrating processes do not self-regulate—the PV continues ramping until inputs and outputs balance. Tank level control is the most common example. **95% of industrial tank levels oscillate**, making systematic tuning critical.

## Characteristics of Integrating Processes

```python
def simulate_integrating_response(imbalance, duration=60.0, dt=0.1):
    """Simulate integrating process (tank level).

    Unlike self-regulating, level continues moving with sustained imbalance.

    Args:
        imbalance: Flow imbalance (inlet - outlet) %
        duration: Simulation time (seconds)
        dt: Time step (seconds)

    Returns:
        time, level: Arrays for plotting
    """
    import numpy as np

    time = np.arange(0, duration, dt)
    level = np.zeros_like(time)

    Kp_integrating = 0.1  # %/second per % imbalance

    for i in range(1, len(time)):
        # Integrating: dLevel/dt = Kp * imbalance
        level[i] = level[i-1] + Kp_integrating * imbalance * dt

    return time, level


# Demo: Sustained imbalance causes continuous ramp
t, lvl = simulate_integrating_response(imbalance=5.0, duration=30.0)
print(f"Integrating Process: Level after 30s = {lvl[-1]:.1f}% (keeps rising)")
# Output: Integrating Process: Level after 30s = 15.0% (keeps rising)
```

## Process Gain Calculation

### Method 1: Slope Method

```python
def calculate_integrating_kp_slope(slope_1, slope_2, delta_output):
    """Calculate process gain from slope change.

    Kp = ΔSlope / ΔOutput

    Args:
        slope_1: Initial slope (%/second)
        slope_2: Final slope after output change
        delta_output: Change in output (%)

    Returns:
        Process gain Kp (%/second/%)
    """
    delta_slope = slope_2 - slope_1
    Kp = delta_slope / delta_output
    return Kp


# Example: Tank bump test
slope_before = 0.0    # Balanced (no change)
slope_during = 0.5    # Rising at 0.5%/s
output_change = 5.0   # 5% valve opening increase

Kp_slope = calculate_integrating_kp_slope(slope_before, slope_during, output_change)
print(f"Process Gain (slope method): {Kp_slope:.3f} %/s/%")
# Output: Process Gain (slope method): 0.100 %/s/%
```

### Method 2: Fill Time (Preferred)

```python
def calculate_integrating_kp_filltime(tank_volume, max_flow, time_units='minutes'):
    """Calculate process gain from tank fill time.

    Kp = 1 / Fill_Time

    Args:
        tank_volume: Tank capacity (gallons, liters, etc.)
        max_flow: Maximum flow rate (same units/time)
        time_units: 'minutes' or 'seconds'

    Returns:
        (Kp, fill_time): Process gain and fill time
    """
    fill_time = tank_volume / max_flow  # time to fill 0-100%
    Kp = 1.0 / fill_time

    return Kp, fill_time


# Example: Storage tank
tank_capacity = 1000.0   # gallons
max_inlet_flow = 50.0    # gallons/minute

Kp, fill_time = calculate_integrating_kp_filltime(tank_capacity, max_inlet_flow)
print(f"Tank Fill Time: {fill_time:.1f} minutes")
print(f"Process Gain Kp: {Kp:.4f} 1/min")
# Output:
# Tank Fill Time: 20.0 minutes
# Process Gain Kp: 0.0500 1/min
```

## Lambda Tuning for Tanks (Arrest Rate)

For integrating processes, λ is the **arrest rate**—time for level to stop deviating.

```python
def lambda_tuning_integrating(Kp, lambda_arrest):
    """Calculate PI tuning for integrating process (tank level).

    Tuning Rules:
    - Kc = 2 / (Kp × λ_arrest)
    - Ti = 2 × λ_arrest

    Goal: Critically damped second-order response

    Args:
        Kp: Process gain (1/fill_time)
        lambda_arrest: Desired arrest rate (time units)

    Returns:
        (Kc, Ti): Controller gain and integral time
    """
    Kc = 2.0 / (Kp * lambda_arrest)
    Ti = 2.0 * lambda_arrest

    return Kc, Ti


# Example: Tank tuning
fill_time = 20.0  # minutes
Kp = 1.0 / fill_time
lambda_arrest = fill_time / 5.0  # Fast: fill_time / 5

Kc, Ti = lambda_tuning_integrating(Kp, lambda_arrest)

print("Tank Level PI Tuning:")
print(f"  Fill Time:              {fill_time:.1f} min")
print(f"  Process Gain (Kp):      {Kp:.4f} 1/min")
print(f"  Arrest Rate (λ):        {lambda_arrest:.1f} min")
print(f"\nPI Parameters:")
print(f"  Controller Gain (Kc):   {Kc:.2f}")
print(f"  Integral Time (Ti):     {Ti:.1f} min")

# Recovery time
recovery_time = 6 * lambda_arrest
print(f"\nExpected Recovery:        {recovery_time:.1f} min (~6λ)")

# Output:
# Tank Level PI Tuning:
#   Fill Time:              20.0 min
#   Process Gain (Kp):      0.0500 1/min
#   Arrest Rate (λ):        4.0 min
#
# PI Parameters:
#   Controller Gain (Kc):   10.00
#   Integral Time (Ti):     8.0 min
#
# Expected Recovery:        24.0 min (~6λ)
```

## Selecting Arrest Rate

```python
def select_arrest_rate(fill_time, objective='balanced'):
    """Recommend arrest rate based on control objective.

    Args:
        fill_time: Tank fill time (minutes)
        objective: 'fast', 'balanced', or 'slow'

    Returns:
        (lambda_arrest, speed_factor): Arrest rate and M factor
    """
    speed_factors = {
        'fast': 5.0,      # λ = fill_time / 5 (tight control)
        'balanced': 3.0,  # λ = fill_time / 3 (moderate)
        'slow': 2.0       # λ = fill_time / 2 (surge absorption)
    }

    M = speed_factors.get(objective, 3.0)
    lambda_arrest = fill_time / M

    return lambda_arrest, M


# Compare tuning strategies
fill_time = 30.0  # minutes
Kp = 1.0 / fill_time

print("Arrest Rate Selection:")
for obj in ['fast', 'balanced', 'slow']:
    lam_arr, M = select_arrest_rate(fill_time, obj)
    Kc, Ti = lambda_tuning_integrating(Kp, lam_arr)
    recovery = 6 * lam_arr

    print(f"\n{obj.capitalize()} Control (M={M}):")
    print(f"  λ_arrest = {lam_arr:.1f} min, Recovery = {recovery:.0f} min")
    print(f"  Kc = {Kc:.2f}, Ti = {Ti:.1f} min")

# Output shows tradeoff: Fast control → Higher Kc, Faster recovery
```

## Validation and Troubleshooting

```python
def validate_tank_tuning(Kc, Kp, Ti):
    """Validate tank tuning parameters.

    For ideal critically damped: Kc × Kp × Ti = 4

    Args:
        Kc: Controller gain
        Kp: Process gain
        Ti: Integral time

    Returns:
        dict with validation results
    """
    product = Kc * Kp * Ti
    ideal = 4.0
    error_pct = abs(product - ideal) / ideal * 100

    if error_pct < 5:
        status = "✓ Ideal (critically damped)"
    elif error_pct < 15:
        status = "✓ Acceptable"
    elif product > ideal:
        status = "⚠ Too slow (sluggish recovery)"
    else:
        status = "⚠ Too fast (likely oscillating)"

    return {
        'product': product,
        'ideal': ideal,
        'error_pct': error_pct,
        'status': status
    }


# Example validation
Kc_test = 10.0
Kp_test = 0.05
Ti_test = 8.0

result = validate_tank_tuning(Kc_test, Kp_test, Ti_test)
print(f"Validation: Kc×Kp×Ti = {result['product']:.2f}")
print(f"Status: {result['status']}")
# Output:
# Validation: Kc×Kp×Ti = 4.00
# Status: ✓ Ideal (critically damped)
```

## Setpoint Filtering

Tank tuning produces high Kc (often 10-12 vs 0.3-0.5 for self-regulating). Setpoint steps cause output spikes.

```python
def apply_setpoint_filter(sp_new, sp_current, filter_time, dt):
    """Apply first-order filter to setpoint changes.

    Args:
        sp_new: New setpoint target
        sp_current: Current filtered setpoint
        filter_time: Filter time constant (seconds)
        dt: Sample time (seconds)

    Returns:
        Filtered setpoint for this step
    """
    alpha = dt / (filter_time + dt)
    sp_filtered = sp_current + alpha * (sp_new - sp_current)
    return sp_filtered


# Demo: Smooth setpoint ramp
import numpy as np

sp_target = 60.0  # New setpoint
sp_start = 50.0   # Current setpoint
filter_tc = 10.0  # 10-second ramp
dt = 1.0

sp_current = sp_start
print("Setpoint Filter (reduces output spike):")
for t in range(0, 15, 2):
    sp_current = apply_setpoint_filter(sp_target, sp_current, filter_tc, dt)
    print(f"  t={t}s: SP={sp_current:.1f}%")

# Output shows gradual approach to target instead of step
```

## Key Differences: Self-Regulating vs Integrating

| Aspect | Self-Regulating | Integrating (Tank) |
|--------|----------------|-------------------|
| **Process Gain** | Kp = ΔPV/ΔOutput | Kp = 1/Fill_Time |
| **Lambda** | Closed-loop TC | Arrest rate |
| **Kc Formula** | (1/Kp)/(λ/τp) | 2/(Kp×λ) |
| **Ti Formula** | τp | 2×λ |
| **Typical Kc** | 0.3 - 0.5 | 10 - 12 |
| **Settling** | ~4λ | ~6λ |
| **Response** | First-order | Second-order critically damped |

## Next Steps

- [Lambda Tuning](03_lambda_tuning.md) - Self-regulating process tuning
- [Process Identification](01_process_identification.md) - Bump test methods
- [Nonlinearities](06_nonlinearities.md) - Tank-specific issues (stiction shows triangle waves)
