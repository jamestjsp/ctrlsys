# Lambda Tuning (Direct Synthesis)

Lambda tuning is a systematic, model-based method for calculating PI controller parameters. It delivers predictable, non-oscillatory responses by mathematically linking process dynamics to desired closed-loop performance.

## Core Concept: The Lambda Parameter

Lambda (λ) is the **desired closed-loop time constant** - the single tuning "knob" that defines response speed.

```python
def calculate_tau_ratio(lambda_cl, tau_p):
    """Calculate tau ratio (speed parameter).

    Tau Ratio = Closed-Loop TC / Open-Loop TC

    Args:
        lambda_cl: Desired closed-loop time constant
        tau_p: Process time constant

    Returns:
        Tau ratio (dimensionless)
    """
    return lambda_cl / tau_p

# Example: Conservative tuning
lambda_val = 30.0  # seconds (desired response speed)
tau_p = 10.0       # seconds (process time constant)

tau_ratio = calculate_tau_ratio(lambda_val, tau_p)
print(f"Tau Ratio: {tau_ratio:.1f}")
# Output: Tau Ratio: 3.0 (conservative/safe)
```

**Interpretation:**
- τ_ratio < 1: Aggressive (fast), requires high model confidence
- τ_ratio = 1-2: Moderate speed
- τ_ratio = 3-4: Conservative (safe), handles model mismatch
- **Rule of Thumb:** λ ≥ 3 × Dead Time for robust tuning

## Self-Regulating Process Tuning

For processes that settle at a new steady state (flow, pressure, temperature).

### Tuning Formulas

```python
def lambda_tuning_self_regulating(Kp, tau_p, lambda_cl):
    """Calculate PI tuning for self-regulating process.

    Tuning Rules:
    - Kc = (1/Kp) / (λ/τp)  or  Kc = τp / (Kp × λ)
    - Ti = τp
    - Td = 0

    Args:
        Kp: Process gain (ΔPV/ΔOutput)
        tau_p: Process time constant (seconds)
        lambda_cl: Desired closed-loop time constant (seconds)

    Returns:
        (Kc, Ti, Td, tau_ratio): Tuning parameters
    """
    tau_ratio = lambda_cl / tau_p
    Kc = (1.0 / Kp) / tau_ratio  # or: tau_p / (Kp * lambda_cl)
    Ti = tau_p
    Td = 0.0

    return Kc, Ti, Td, tau_ratio


# Example: Temperature control
Kp = 2.0       # °C per % output change
tau_p = 10.0   # seconds
Td = 2.0       # seconds dead time
lambda_cl = 3.0 * Td  # Conservative: 3 × dead time

Kc, Ti, Td, tau_ratio = lambda_tuning_self_regulating(Kp, tau_p, lambda_cl)

print("Self-Regulating Process Tuning:")
print(f"  Process Gain (Kp):      {Kp:.2f}")
print(f"  Time Constant (τp):     {tau_p:.1f} s")
print(f"  Dead Time (Td):         {Td:.1f} s")
print(f"  Lambda (λ):             {lambda_cl:.1f} s")
print(f"  Tau Ratio:              {tau_ratio:.2f}")
print(f"\nCalculated PI Parameters:")
print(f"  Controller Gain (Kc):   {Kc:.4f}")
print(f"  Integral Time (Ti):     {Ti:.1f} s")
print(f"  Derivative Time (Td):   {Td:.1f} s")

# Expected settling time
settling_time = 4 * lambda_cl
print(f"\nExpected Settling Time: {settling_time:.1f} s (~4λ)")
```

### Response Validation

Verify tuning produces expected response:

```python
def validate_proportional_kick(tau_ratio):
    """Calculate expected proportional kick fraction.

    For tau_ratio = 2, initial P-kick should be 50% of total output needed.

    Args:
        tau_ratio: Lambda / tau_p

    Returns:
        Fraction of total output in P-kick
    """
    p_kick_fraction = 1.0 / tau_ratio
    return p_kick_fraction


# Example validation
ratios = [0.5, 1.0, 2.0, 3.0, 4.0]

print("Expected Proportional Kick:")
for ratio in ratios:
    kick_pct = validate_proportional_kick(ratio) * 100
    speed = "Very Fast" if ratio < 1 else ("Fast" if ratio < 2 else "Conservative")
    print(f"  τ_ratio={ratio:.1f}: {kick_pct:.0f}% of total ({speed})")

# Output:
# Expected Proportional Kick:
#   τ_ratio=0.5: 200% of total (Very Fast) ← Overshoot risk
#   τ_ratio=1.0: 100% of total (Fast)
#   τ_ratio=2.0: 50% of total (Fast)
#   τ_ratio=3.0: 33% of total (Conservative) ← Recommended
#   τ_ratio=4.0: 25% of total (Conservative)
```

## Model Mismatch and Lambda Selection

Real processes deviate from first-order models. Compensate with larger λ.

```python
def recommend_lambda(dead_time, model_confidence='medium'):
    """Recommend lambda based on dead time and model confidence.

    Args:
        dead_time: Process dead time (seconds)
        model_confidence: 'high', 'medium', or 'low'

    Returns:
        Recommended lambda (seconds)
    """
    multipliers = {
        'high': 2.0,    # λ = 2 × Td (requires accurate model)
        'medium': 3.0,  # λ = 3 × Td (standard robustness)
        'low': 4.0      # λ = 4 × Td (conservative, high mismatch)
    }

    multiplier = multipliers.get(model_confidence, 3.0)
    lambda_rec = multiplier * dead_time

    return lambda_rec, multiplier


# Example recommendations
Td = 5.0  # seconds

for conf in ['high', 'medium', 'low']:
    lambda_val, mult = recommend_lambda(Td, conf)
    print(f"{conf.capitalize()} confidence: λ = {mult}×Td = {lambda_val:.1f} s")

# Output:
# High confidence: λ = 2.0×Td = 10.0 s
# Medium confidence: λ = 3.0×Td = 15.0 s
# Low confidence: λ = 4.0×Td = 20.0 s
```

## Complete Tuning Workflow

```python
def complete_lambda_tuning_workflow(Kp, tau_p, Td, confidence='medium'):
    """Complete Direct Synthesis tuning workflow.

    Args:
        Kp: Process gain from bump test
        tau_p: Time constant from bump test
        Td: Dead time from bump test
        confidence: Model confidence level

    Returns:
        dict with tuning parameters and analysis
    """
    # Step 1: Recommend lambda based on dead time
    lambda_cl, multiplier = recommend_lambda(Td, confidence)

    # Step 2: Calculate tuning parameters
    Kc, Ti, Td_param, tau_ratio = lambda_tuning_self_regulating(Kp, tau_p, lambda_cl)

    # Step 3: Validation check
    validation_product = Kc * Kp * Ti
    ideal_product = 4.0
    product_error = abs(validation_product - ideal_product) / ideal_product * 100

    # Step 4: Expected response
    settling_time = 4 * lambda_cl
    p_kick_fraction = 1.0 / tau_ratio

    results = {
        'inputs': {'Kp': Kp, 'tau_p': tau_p, 'Td': Td, 'confidence': confidence},
        'lambda': lambda_cl,
        'lambda_rule': f"{multiplier}×Td",
        'tau_ratio': tau_ratio,
        'tuning': {'Kc': Kc, 'Ti': Ti, 'Td': 0.0},
        'validation_product': validation_product,
        'product_error_pct': product_error,
        'settling_time': settling_time,
        'p_kick_pct': p_kick_fraction * 100
    }

    return results


# Example: Complete workflow
process_params = {'Kp': 1.5, 'tau_p': 12.0, 'Td': 3.0}
results = complete_lambda_tuning_workflow(**process_params, confidence='medium')

print("="*60)
print("Lambda Tuning Workflow Results")
print("="*60)
print(f"\nProcess Parameters:")
print(f"  Kp = {results['inputs']['Kp']:.2f}")
print(f"  τp = {results['inputs']['tau_p']:.1f} s")
print(f"  Td = {results['inputs']['Td']:.1f} s")

print(f"\nLambda Selection:")
print(f"  Rule: λ = {results['lambda_rule']}")
print(f"  Lambda = {results['lambda']:.1f} s")
print(f"  Tau Ratio = {results['tau_ratio']:.2f}")

print(f"\nPI Tuning Parameters:")
print(f"  Kc = {results['tuning']['Kc']:.4f}")
print(f"  Ti = {results['tuning']['Ti']:.1f} s")
print(f"  Td = {results['tuning']['Td']:.1f} s")

print(f"\nValidation:")
print(f"  Kc×Kp×Ti = {results['validation_product']:.3f} (ideal=4.0)")
print(f"  Error = {results['product_error_pct']:.1f}%")

print(f"\nExpected Response:")
print(f"  Settling Time = {results['settling_time']:.1f} s (~4λ)")
print(f"  P-Kick = {results['p_kick_pct']:.0f}% of total output")
print("="*60)
```

## Dead Time Constraints

When dead time dominates, conventional PI fails and advanced methods are required.

```python
def check_dead_time_limit(Td, lambda_cl):
    """Check if dead time compensation is required.

    Rule: If Td > 3×λ, use Smith Predictor or IMC

    Args:
        Td: Dead time (seconds)
        lambda_cl: Desired closed-loop time constant

    Returns:
        (requires_dtc, ratio): Whether DTC needed and Td/λ ratio
    """
    ratio = Td / lambda_cl
    requires_dtc = ratio > 3.0

    return requires_dtc, ratio


# Example: Check various scenarios
scenarios = [
    ("Normal", 2.0, 10.0),
    ("Borderline", 3.0, 10.0),
    ("Needs DTC", 35.0, 10.0)
]

print("Dead Time Compensation Check:")
for name, td, lam in scenarios:
    needs_dtc, ratio = check_dead_time_limit(td, lam)
    status = "⚠ Use DTC" if needs_dtc else "✓ PI OK"
    print(f"  {name}: Td/λ={ratio:.2f} → {status}")

# Output:
# Dead Time Compensation Check:
#   Normal: Td/λ=0.20 → ✓ PI OK
#   Borderline: Td/λ=0.30 → ✓ PI OK
#   Needs DTC: Td/λ=3.50 → ⚠ Use DTC
```

## Key Principles

1. **Lambda is the Speed Knob** - Single parameter defines closed-loop response
2. **λ ≥ 3×Td** - Minimum robust rule for dead time
3. **Ti = τp Always** - Integral time equals process time constant
4. **Tau Ratio Defines Aggressiveness** - Higher = safer, lower = faster
5. **Validation Product = 4** - For ideal tuning: Kc × Kp × Ti ≈ 4
6. **Units Must Match** - Ti and τp must use same time units

## Next Steps

- [Integrating Processes](04_integrating_processes.md) - Tank level tuning with arrest rate
- [Advanced Methods](05_advanced_methods.md) - Dead time compensation for Td > 3×λ
- [PID Fundamentals](02_pid_fundamentals.md) - Understanding P, I, D actions
