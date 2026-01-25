# Advanced Control Methods (Dead Time Compensation)

When process dead time is large relative to the time constant, conventional PI control becomes ineffective. Advanced methods compensate by predicting the process response, allowing aggressive tuning without oscillation.

## When Advanced Control is Required

```python
def requires_dead_time_compensation(Td, lambda_cl):
    """Determine if dead time compensation is needed.

    Rule: If Td > 3×λ, conventional PI will oscillate

    Args:
        Td: Process dead time (seconds)
        lambda_cl: Desired closed-loop time constant

    Returns:
        (needs_dtc, reason): Whether DTC is required and explanation
    """
    ratio = Td / lambda_cl

    if ratio > 3.0:
        needs_dtc = True
        reason = f"Dead time too large (Td/λ={ratio:.2f} > 3.0)"
    else:
        needs_dtc = False
        reason = f"Conventional PI adequate (Td/λ={ratio:.2f} ≤ 3.0)"

    return needs_dtc, reason


# Example scenarios
scenarios = [
    ("Normal Loop", 2.0, 10.0),
    ("Long Pipe", 15.0, 10.0),
    ("Analyzer Loop", 45.0, 10.0)
]

print("Dead Time Compensation Assessment:")
for name, td, lam in scenarios:
    needs, reason = requires_dead_time_compensation(td, lam)
    print(f"  {name}: {reason}")

# Output:
# Dead Time Compensation Assessment:
#   Normal Loop: Conventional PI adequate (Td/λ=0.20 ≤ 3.0)
#   Long Pipe: Conventional PI adequate (Td/λ=1.50 ≤ 3.0)
#   Analyzer Loop: Dead time too large (Td/λ=4.50 > 3.0)
```

## Smith Predictor

The Smith Predictor uses a process model to simulate the response **without delay**, allowing PI tuning as if dead time didn't exist.

### Architecture

```
Setpoint → [PI Controller] → Output → [Real Process with Delay] → PV
                ↑                              ↓
                └──── [Estimated Load] ←──────┘
                      (PV - Model)

Where Model = [Process without Delay]
```

### Implementation

```python
class SmithPredictor:
    """Smith Predictor dead time compensator."""

    def __init__(self, Kc, Ti, Kp_model, tau_p_model, Td_model, dt):
        """Initialize Smith Predictor.

        Args:
            Kc: Controller gain
            Ti: Integral time
            Kp_model: Model process gain
            tau_p_model: Model time constant
            Td_model: Model dead time
            dt: Sample time
        """
        self.Kc = Kc
        self.Ti = Ti
        self.Kp_model = Kp_model
        self.tau_p_model = tau_p_model
        self.Td_model = Td_model
        self.dt = dt

        # State variables
        self.error_integral = 0.0
        self.pv_model_no_delay = 0.0
        self.output_history = []  # For dead time

    def update(self, setpoint, pv_actual, output_previous):
        """Calculate controller output using Smith Predictor.

        Args:
            setpoint: Desired value
            pv_actual: Measured process variable
            output_previous: Previous controller output

        Returns:
            Controller output
        """
        # Update model (first-order without delay)
        dpv_model = (self.Kp_model * output_previous - self.pv_model_no_delay) / self.tau_p_model
        self.pv_model_no_delay += dpv_model * self.dt

        # Estimate load disturbance
        # Load = Actual PV - Model prediction (with historical output)
        # Simplified: load_estimate = pv_actual - pv_model_no_delay
        load_estimate = pv_actual - self.pv_model_no_delay

        # Effective error (includes load compensation)
        error_effective = setpoint - (self.pv_model_no_delay + load_estimate)

        # PI control on effective error
        self.error_integral += error_effective * self.dt
        output = self.Kc * (error_effective + (1.0 / self.Ti) * self.error_integral)

        return output
```

### Tuning Smith Predictor

```python
def tune_smith_predictor(Kp, tau_p, Td, lambda_internal=None):
    """Calculate Smith Predictor parameters.

    The internal PI controller sees only gain and TC (no dead time),
    so it can be tuned aggressively.

    Args:
        Kp: Process gain
        tau_p: Process time constant
        Td: Process dead time
        lambda_internal: Lambda for internal loop (default: τp)

    Returns:
        (Kc, Ti, model_params): Tuning and model parameters
    """
    if lambda_internal is None:
        lambda_internal = tau_p  # Aggressive: tau_ratio = 1

    # Tune internal PI as if no dead time
    tau_ratio = lambda_internal / tau_p
    Kc = (1.0 / Kp) / tau_ratio
    Ti = tau_p

    model_params = {
        'Kp': Kp,
        'tau_p': tau_p,
        'Td': Td
    }

    return Kc, Ti, model_params


# Example
Kp = 2.0
tau_p = 10.0
Td = 30.0  # Large dead time (would require λ=90s for conventional PI)

Kc, Ti, model = tune_smith_predictor(Kp, tau_p, Td)

print("Smith Predictor Tuning:")
print(f"  Process: Kp={Kp}, τp={tau_p}s, Td={Td}s")
print(f"  Internal PI: Kc={Kc:.3f}, Ti={Ti:.1f}s")
print(f"  Model params: {model}")
print(f"\nNote: Much more aggressive than conventional PI!")
```

## Internal Model Control (IMC)

IMC uses a lead-lag compensator instead of PID. Mathematically equivalent to Smith Predictor but easier to implement in modern DCS.

### IMC Tuning (Lambda Method)

```python
def tune_imc(Kp, tau_p, Td, lambda_imc):
    """Calculate IMC parameters using lambda tuning.

    For first-order + dead time processes.

    Args:
        Kp: Process gain
        tau_p: Process time constant
        Td: Process dead time
        lambda_imc: Desired closed-loop time constant

    Returns:
        (Kc_equivalent, Ti_equivalent): Equivalent PI parameters
    """
    # IMC lambda tuning rules for FOPTD
    Kc = tau_p / (Kp * (lambda_imc + Td))
    Ti = tau_p

    return Kc, Ti


# Example: Compare conventional vs IMC lambda tuning
Kp = 2.0
tau_p = 10.0
Td = 15.0  # Significant dead time

# Conventional lambda tuning
lambda_conv = 3.0 * Td  # = 45s (very slow)
Kc_conv = (1.0 / Kp) / (lambda_conv / tau_p)
Ti_conv = tau_p

# IMC lambda tuning
lambda_imc = 15.0  # Can be much smaller!
Kc_imc, Ti_imc = tune_imc(Kp, tau_p, Td, lambda_imc)

print("Comparison: Conventional PI vs IMC")
print(f"\nConventional (λ={lambda_conv}s):")
print(f"  Kc={Kc_conv:.3f}, Ti={Ti_conv:.1f}s")

print(f"\nIMC (λ={lambda_imc}s):")
print(f"  Kc={Kc_imc:.3f}, Ti={Ti_imc:.1f}s")
print(f"  → {(Kc_imc/Kc_conv):.1f}× more aggressive!")
```

## Model Mismatch: The Achilles Heel

All advanced methods depend entirely on model accuracy. Mismatch causes instability.

```python
def analyze_model_mismatch(Kp_true, Kp_model, tau_p_true, tau_p_model):
    """Quantify model mismatch.

    Args:
        Kp_true: Actual process gain
        Kp_model: Model process gain
        tau_p_true: Actual time constant
        tau_p_model: Model time constant

    Returns:
        dict with mismatch analysis
    """
    gain_error = abs(Kp_true - Kp_model) / Kp_true * 100
    tc_error = abs(tau_p_true - tau_p_model) / tau_p_true * 100

    if gain_error > 20 or tc_error > 20:
        severity = "⚠ High mismatch - expect oscillation"
    elif gain_error > 10 or tc_error > 10:
        severity = "⚠ Moderate mismatch - tune conservatively"
    else:
        severity = "✓ Low mismatch - good performance expected"

    return {
        'gain_error_pct': gain_error,
        'tc_error_pct': tc_error,
        'severity': severity
    }


# Example: Process dynamics changed over time
Kp_original = 2.0
tau_p_original = 10.0

Kp_current = 2.5  # Gain increased 25%
tau_p_current = 12.0  # TC increased 20%

mismatch = analyze_model_mismatch(Kp_current, Kp_original, tau_p_current, tau_p_original)

print("Model Mismatch Analysis:")
print(f"  Gain error: {mismatch['gain_error_pct']:.1f}%")
print(f"  TC error: {mismatch['tc_error_pct']:.1f}%")
print(f"  {mismatch['severity']}")

# Output:
# Model Mismatch Analysis:
#   Gain error: 20.0%
#   TC error: 16.7%
#   ⚠ High mismatch - expect oscillation
```

## Preventive Maintenance: Regular Bump Tests

```python
def schedule_bump_test(last_test_days_ago, process_type='stable'):
    """Recommend bump test frequency.

    Args:
        last_test_days_ago: Days since last bump test
        process_type: 'stable', 'variable', or 'critical'

    Returns:
        (needs_test, reason): Whether test is due
    """
    test_intervals = {
        'stable': 180,    # 6 months
        'variable': 90,   # 3 months
        'critical': 30    # 1 month
    }

    interval = test_intervals.get(process_type, 90)
    needs_test = last_test_days_ago >= interval

    if needs_test:
        reason = f"Bump test overdue ({last_test_days_ago} days, limit {interval})"
    else:
        days_remaining = interval - last_test_days_ago
        reason = f"Next test in {days_remaining} days"

    return needs_test, reason


# Example
test_status = schedule_bump_test(last_test_days_ago=95, process_type='variable')
print(test_status[1])
# Output: Bump test overdue (95 days, limit 90)
```

## Key Principles

1. **Use When Td > 3×λ** - Otherwise conventional PI is adequate
2. **Model Accuracy Critical** - Mismatch causes oscillation or instability
3. **Regular Bump Tests** - Process dynamics change over time (wear, seasons, production lines)
4. **Smith = IMC** - Mathematically equivalent, choose based on DCS capability
5. **Lambda Still Applies** - Larger λ provides robustness against mismatch
6. **Not a Silver Bullet** - Cannot fix mechanical problems (stiction, etc.)

## Next Steps

- [Lambda Tuning](03_lambda_tuning.md) - Conventional PI tuning for normal processes
- [Nonlinearities](06_nonlinearities.md) - When dead time varies or mechanical issues exist
- [Process Identification](01_process_identification.md) - Accurate bump testing procedures
