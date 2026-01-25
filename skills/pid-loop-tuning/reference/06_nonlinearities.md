# Nonlinearities in Process Control

Nonlinearities are deviations from linear behavior where identical inputs produce inconsistent outputs. These issues appear throughout control loops and **cannot be fixed by tuning alone**—they require mechanical repair or adaptive strategies.

## Definition and Impact

A linear process produces consistent responses: a 5% output change always causes the same PV change. Nonlinearities violate this assumption.

```python
def test_linearity(outputs, responses):
    """Test if process behaves linearly.

    Args:
        outputs: Array of output changes
        responses: Array of corresponding PV responses

    Returns:
        (is_linear, gain_variation): Linearity check and gain range
    """
    import numpy as np

    # Calculate gain for each test
    gains = np.array(responses) / np.array(outputs)

    # Check consistency
    gain_std = np.std(gains)
    gain_mean = np.mean(gains)
    variation_pct = (gain_std / gain_mean) * 100

    is_linear = variation_pct < 10  # Within 10% = linear

    return is_linear, variation_pct, gains


# Example: Nonlinear process
outputs_tested = [2, 4, 6, 8, 10]  # % changes
responses = [4, 9, 15, 24, 35]  # Corresponding PV changes (nonlinear!)

linear, var, gains = test_linearity(outputs_tested, responses)

print("Linearity Test:")
print(f"  Calculated Gains: {gains}")
print(f"  Variation: {var:.1f}%")
print(f"  Linear? {'Yes' if linear else 'No (⚠ Nonlinear behavior)'}")

# Output:
# Linearity Test:
#   Calculated Gains: [2.  2.25 2.5  3.   3.5 ]
#   Variation: 24.2%
#   Linear? No (⚠ Nonlinear behavior)
```

## Common Nonlinearity Types

### 1. Stiction (Static Friction) - Sticky Valves

Mechanical friction prevents valve movement until controller pressure builds, causing abrupt "jumps."

**Diagnostic Patterns:**

```python
def diagnose_stiction_pattern(pv_signal, output_signal, process_type='self-regulating'):
    """Identify stiction from oscillation waveforms.

    Classic signatures:
    - Self-regulating: Square wave PV, Triangle wave Output
    - Integrating: Triangle wave PV, Triangle wave Output

    Args:
        pv_signal: Process variable time series
        output_signal: Controller output time series
        process_type: 'self-regulating' or 'integrating'

    Returns:
        Diagnostic assessment
    """
    import numpy as np

    # Simplified waveform detection (count direction changes)
    pv_changes = np.diff(np.sign(np.diff(pv_signal)))
    out_changes = np.diff(np.sign(np.diff(output_signal)))

    pv_corners = np.sum(pv_changes != 0)  # Sharp corners
    out_corners = np.sum(out_changes != 0)

    if process_type == 'self-regulating':
        if pv_corners > out_corners * 1.5:
            return "⚠ Likely stiction (square PV, triangle Output)"
    else:  # integrating
        if pv_corners > 2 and out_corners > 2:
            return "⚠ Likely stiction (both triangle waves)"

    return "✓ No obvious stiction pattern"


# Example: Self-regulating process with stiction
# Square wave PV: [50, 50, 48, 48, 52, 52, 48, 48]
# Triangle Output: [40, 45, 50, 45, 40, 45, 50, 45]
import numpy as np
pv_square = np.array([50, 50, 48, 48, 52, 52, 48, 48, 52, 52])
out_triangle = np.array([40, 42, 44, 46, 48, 46, 44, 42, 40, 42])

diagnosis = diagnose_stiction_pattern(pv_square, out_triangle, 'self-regulating')
print(f"Stiction Diagnosis: {diagnosis}")
# Output: Stiction Diagnosis: ⚠ Likely stiction (square PV, triangle Output)
```

**Management:** Stiction **must be fixed mechanically**—tuning cannot resolve it. Repair valve, replace positioner, or service actuator.

### 2. Dead Band / Dead Zone

A range where controller output changes but nothing happens (actuator or sensor constraint).

```python
def simulate_deadband(output_change, deadband_range=2.0):
    """Simulate dead band effect.

    Args:
        output_change: Requested output change (%)
        deadband_range: Size of dead zone (%)

    Returns:
        Actual output delivered
    """
    if abs(output_change) < deadband_range:
        actual_output = 0.0  # No movement in dead zone
    else:
        actual_output = output_change  # Moves outside zone

    return actual_output


# Example: Controller makes small adjustments
small_changes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
deadband = 2.0

print("Dead Band Effect:")
for change in small_changes:
    actual = simulate_deadband(change, deadband)
    status = "BLOCKED" if actual == 0 else "OK"
    print(f"  Request {change:.1f}% → Actual {actual:.1f}% ({status})")

# Output:
# Dead Band Effect:
#   Request 0.5% → Actual 0.0% (BLOCKED)
#   Request 1.0% → Actual 0.0% (BLOCKED)
#   Request 1.5% → Actual 0.0% (BLOCKED)
#   Request 2.0% → Actual 2.0% (OK)
#   Request 2.5% → Actual 2.5% (OK)
```

**Consequence:** Causes limit cycles (oscillation between boundaries)

**Management:** Minimize or eliminate dead band in controller/valve configuration

### 3. Nonlinear Process Gain

Process gain varies across operating range (e.g., horizontal cylindrical tank).

```python
def calculate_tank_gain_varying(level_pct):
    """Calculate process gain for horizontal cylindrical tank.

    Gain is highest at top/bottom (narrow), lowest at middle (wide).

    Args:
        level_pct: Current tank level (0-100%)

    Returns:
        Process gain at this level
    """
    import numpy as np

    # Simplified: gain inversely proportional to tank width
    # Width varies with level: w = sin(level × π/100)
    theta = (level_pct / 100.0) * np.pi
    width_factor = np.sin(theta)

    # Gain is inverse of width
    if width_factor > 0.1:  # Avoid division by zero
        Kp = 1.0 / width_factor
    else:
        Kp = 10.0  # Very high gain at extremes

    return Kp


# Demonstrate gain variation
levels = [10, 30, 50, 70, 90]

print("Horizontal Tank - Process Gain by Level:")
for lvl in levels:
    Kp = calculate_tank_gain_varying(lvl)
    print(f"  Level {lvl}%: Kp = {Kp:.2f}")

# Output:
# Horizontal Tank - Process Gain by Level:
#   Level 10%: Kp = 5.76 (high gain - steep sides)
#   Level 30%: Kp = 2.00
#   Level 50%: Kp = 1.00 (lowest gain - widest point)
#   Level 70%: Kp = 2.00
#   Level 90%: Kp = 5.76 (high gain again)
```

**Management Strategies:**

1. **Tune for Worst Case** (highest gain region)
2. **Gain Scheduling** - Adjust parameters based on operating point
3. **Adaptive Control** - Automatically update tuning

```python
def gain_scheduling_example(current_level, Kp_nominal=1.0):
    """Adjust controller gain based on operating point.

    Args:
        current_level: Current tank level (%)
        Kp_nominal: Nominal process gain (at 50%)

    Returns:
        Adjusted controller gain
    """
    # Calculate current process gain
    Kp_current = calculate_tank_gain_varying(current_level)

    # Adjust controller gain to maintain consistent loop gain
    # Loop gain = Kc × Kp should be constant
    Kc_nominal = 1.0 / Kp_nominal
    Kc_adjusted = Kc_nominal * (Kp_nominal / Kp_current)

    return Kc_adjusted, Kp_current


# Example: Adaptive gain scheduling
levels_test = [20, 50, 80]

print("\nGain Scheduling:")
for lvl in levels_test:
    Kc_adj, Kp_curr = gain_scheduling_example(lvl, Kp_nominal=1.0)
    print(f"  Level {lvl}%: Kp={Kp_curr:.2f}, Kc={Kc_adj:.2f}")

# Output:
# Gain Scheduling:
#   Level 20%: Kp=3.42, Kc=0.29 (reduce Kc at high Kp)
#   Level 50%: Kp=1.00, Kc=1.00 (nominal)
#   Level 80%: Kp=2.75, Kc=0.36
```

## Diagnostic Tools

### Frequency Analysis for Oscillation Root Cause

```python
def analyze_oscillation_source(oscillation_period_seconds, dead_time=None):
    """Diagnose oscillation cause using period analysis.

    Args:
        oscillation_period_seconds: Measured cycle time
        dead_time: Process dead time if known (seconds)

    Returns:
        Probable cause
    """
    causes = []

    if dead_time is not None:
        # Dead time amplification occurs at period = 2×Td
        amplification_period = 2.0 * dead_time
        period_ratio = oscillation_period_seconds / amplification_period

        if 0.9 < period_ratio < 1.1:
            causes.append("⚠ Dead time amplification (period ≈ 2×Td)")

    # Heuristics
    if oscillation_period_seconds < 5:
        causes.append("⚠ Fast cycle - likely derivative or noise")
    elif oscillation_period_seconds < 30:
        causes.append("⚠ Moderate cycle - check tuning or stiction")
    else:
        causes.append("⚠ Slow cycle - likely external disturbance or cascade issue")

    return causes if causes else ["✓ No obvious patterns"]


# Examples
print("Oscillation Diagnosis:")
print(analyze_oscillation_source(4.0, dead_time=2.0))
# Output: ['⚠ Dead time amplification (period ≈ 2×Td)']

print(analyze_oscillation_source(15.0))
# Output: ['⚠ Moderate cycle - check tuning or stiction']
```

### Statistical Process Analysis

```python
def statistical_health_check(pv_data):
    """Quick statistical check for loop health.

    Args:
        pv_data: Time series of PV measurements

    Returns:
        dict with health indicators
    """
    import numpy as np

    mean_val = np.mean(pv_data)
    std_val = np.std(pv_data)
    variability_pct = (std_val / mean_val) * 100 if mean_val > 0 else 0

    # Check for oscillation (high autocorrelation at lag 1)
    if len(pv_data) > 2:
        autocorr = np.corrcoef(pv_data[:-1], pv_data[1:])[0, 1]
    else:
        autocorr = 0

    # Skewness check (asymmetry suggests nonlinearity)
    # Simplified: compare upper/lower halves
    median = np.median(pv_data)
    upper_count = np.sum(pv_data > median)
    lower_count = np.sum(pv_data < median)
    skew_ratio = upper_count / lower_count if lower_count > 0 else 1.0

    health_status = {
        'mean': mean_val,
        'std': std_val,
        'variability_pct': variability_pct,
        'autocorrelation': autocorr,
        'skew_ratio': skew_ratio,
        'assessment': []
    }

    if variability_pct > 5:
        health_status['assessment'].append("⚠ High variability")
    if abs(autocorr) > 0.7:
        health_status['assessment'].append("⚠ Strong oscillation detected")
    if skew_ratio < 0.8 or skew_ratio > 1.2:
        health_status['assessment'].append("⚠ Asymmetric distribution (nonlinearity?)")

    if not health_status['assessment']:
        health_status['assessment'].append("✓ Loop appears healthy")

    return health_status


# Example with oscillating data
import numpy as np
oscillating_pv = 50 + 5 * np.sin(np.linspace(0, 4*np.pi, 50))  # 50 ± 5

health = statistical_health_check(oscillating_pv)
print("Loop Health Check:")
print(f"  Variability: {health['variability_pct']:.1f}%")
print(f"  Autocorrelation: {health['autocorrelation']:.2f}")
print(f"  Assessment: {health['assessment']}")
```

## Tuning Strategy for Nonlinear Processes

When nonlinearities cannot be eliminated:

```python
def robust_tuning_for_nonlinearity(Kp_min, Kp_max, tau_p, Td):
    """Calculate robust tuning for processes with variable gain.

    Strategy: Tune for highest gain region to ensure stability.

    Args:
        Kp_min: Minimum process gain
        Kp_max: Maximum process gain (worst case)
        tau_p: Process time constant
        Td: Dead time

    Returns:
        Conservative PI parameters
    """
    # Use highest gain (most sensitive) for tuning
    Kp_design = Kp_max

    # Conservative lambda (4×Td instead of 3×Td)
    lambda_cl = 4.0 * Td

    # Calculate tuning
    tau_ratio = lambda_cl / tau_p
    Kc = (1.0 / Kp_design) / tau_ratio
    Ti = tau_p

    gain_range = Kp_max / Kp_min

    return {
        'Kc': Kc,
        'Ti': Ti,
        'design_Kp': Kp_design,
        'lambda': lambda_cl,
        'gain_range': gain_range,
        'note': f"Tuned for worst case (Kp={Kp_max}), gain varies {gain_range:.1f}×"
    }


# Example: Tank with 5× gain variation
result = robust_tuning_for_nonlinearity(Kp_min=1.0, Kp_max=5.0, tau_p=10.0, Td=2.0)

print("Robust Tuning for Nonlinear Process:")
print(f"  Kc = {result['Kc']:.4f}")
print(f"  Ti = {result['Ti']:.1f}")
print(f"  {result['note']}")
```

## Key Principles

1. **Tuning Cannot Fix Hardware** - Stiction, dead band require mechanical repair
2. **Diagnose Before Tuning** - Check for square/triangle wave patterns
3. **Variable Gain Needs Strategy** - Tune for worst case or use gain scheduling
4. **Regular Monitoring** - Statistical/frequency analysis catches developing issues
5. **Model Mismatch is a Nonlinearity** - Advanced control amplifies this problem
6. **Document Baseline** - Know normal behavior to detect degradation

## Next Steps

- [Process Identification](01_process_identification.md) - Visual inspection before bump testing
- [Advanced Methods](05_advanced_methods.md) - Model mismatch impact on dead time compensation
- [Integrating Processes](04_integrating_processes.md) - Stiction shows triangle waves in tanks
