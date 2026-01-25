---
name: pid-loop-tuning
description: A comprehensive guide for process engineers on the systematic methodology for tuning Proportional-Integral-Derivative (PID) feedback control loops, covering process identification, model-based tuning rules, and validation for both self-regulating and integrating processes.
license: "Creative Commons Attribution 4.0 International"
metadata:
  skill-version: "2.0"
  author: "James Joseph"
  keywords: "PID, process control, control loop, tuning, direct synthesis, lambda tuning, process identification, self-regulating, integrating, tank level"
---

# Systematic PID Loop Tuning

Transform PID tuning from guesswork into a structured, repeatable engineering discipline using model-based methods. This methodology applies universally to flow, pressure, temperature, level, and other process control applications.

## About This Skill

This skill provides practical PID tuning guidance grounded in industrial control engineering. The interactive tools use **slicot** (SLICOT control systems library with Python bindings) for:

- State-space representation and analysis
- Frequency domain analysis (Bode plots, poles/zeros) via `tb05ad`
- Closed-loop stability verification
- Controller discretization for embedded deployment via `ab04md`
- Step response simulation via `tf01md`

SLICOT provides industrial-strength numerical routines for control system analysis with raw numpy arrays in Fortran column-major order.

## The Four-Step Methodology

### 1. Identify the Process

Understand process dynamics before tuning. Execute a bump test (step test) in manual mode to reveal the process characteristics.

**Self-Regulating Processes** (flow, pressure, temperature):
- Extract process gain (Kp), time constant (τp), and dead time (Td)
- Use FOPDT (First Order Plus Dead Time) model—sufficient for most PID tuning

**Integrating Processes** (tank levels):
- Calculate gain from slope change or fill time
- Process never settles—requires different approach

#### CSV-Based Identification (Recommended)

For practical plant data, use `step_test_identify.py` with CSV files:

```bash
# Identify FOPDT from CSV step test data
uv run scripts/step_test_identify.py data.csv \
    --step-time 100 \
    --pv-range 0 200 \
    --op-range 0 100 \
    --plot
```

**CSV Format Requirements:**
- Required columns: `timestamp`, `PV`, `OP`
- Optional columns: `SP`, `dist_*` (disturbance variables)
- Sampling time inferred automatically from timestamp

**Scaling for Meaningful Gain:**
```
Gain (Kp) = %PV change / %OP change

Where:
  %PV = 100 × (PV - PV_min) / (PV_max - PV_min)
  %OP = 100 × (OP - OP_min) / (OP_max - OP_min)
```

This produces a dimensionless gain suitable for direct use in lambda tuning formulas.

**Identification Methods:**
- `--method=regression` (DEFAULT): Nonlinear least squares FOPDT fit with R² and RMSE metrics
- `--method=slicot`: SLICOT subspace ID (`ib01ad`/`ib01bd`) for higher-order systems

See [Process Identification](references/01_process_identification.md) for detailed procedures and CSV format documentation.

### 2. Understand the Controller

The PI combination (Proportional + Integral) is the workhorse of industrial control, representing 100% of common applications.

**Proportional (P) Action:**
- Responds to error magnitude (present error)
- Goal: Stop the changing error
- Limitation: Results in permanent offset when used alone

**Integral (I) Action:**
- Responds to error duration (past error/area under curve)
- Goal: Make error zero, eliminate offset
- Acts as "watchdog" until PV returns to setpoint

**Derivative (D) Action:**
- Responds to rate of change (future error prediction)
- Rarely used due to noise sensitivity
- Set to zero for most industrial applications

### 3. Apply Lambda Tuning (Direct Synthesis)

Lambda tuning is a systematic, model-based method that delivers predictable, non-oscillatory responses.

**Key Parameter: Lambda (λ)** - The desired closed-loop time constant
- Choose λ ≥ 3 × Dead Time for robust performance
- Larger λ = slower, safer response (conservative)
- Smaller λ = faster, aggressive response (requires high model confidence)

**Self-Regulating Process Tuning Rules:**

```python
def lambda_tuning_pi(Kp, tau_p, lambda_cl):
    """Calculate PI parameters using Direct Synthesis (Lambda Tuning).

    Args:
        Kp: Process gain (ΔPV/ΔOutput)
        tau_p: Process time constant (seconds)
        lambda_cl: Desired closed-loop time constant (seconds)

    Returns:
        (Kc, Ti): Controller gain and integral time
    """
    tau_ratio = lambda_cl / tau_p
    Kc = (1.0 / Kp) / tau_ratio
    Ti = tau_p
    return Kc, Ti

# Example calculation
Kp = 2.0      # Process gain
tau_p = 10.0  # Time constant
lambda_cl = 30.0  # Desired response time (3 × dead time)

Kc, Ti = lambda_tuning_pi(Kp, tau_p, lambda_cl)
print(f"Controller Gain Kc: {Kc:.3f}")
print(f"Integral Time Ti:   {Ti:.1f} seconds")
# Output:
# Controller Gain Kc: 0.167
# Integral Time Ti:   10.0 seconds
```

**Integrating Process Tuning Rules** (tanks):

```python
def tank_tuning_pi(Kp, lambda_arrest):
    """Calculate PI parameters for integrating processes.

    Args:
        Kp: Process gain (1/fill_time) or slope method
        lambda_arrest: Desired arrest rate (seconds)

    Returns:
        (Kc, Ti): Controller gain and integral time
    """
    Kc = 2.0 / (Kp * lambda_arrest)
    Ti = 2.0 * lambda_arrest
    return Kc, Ti

# Example: Tank level control
fill_time = 20.0  # minutes to fill tank 0-100%
Kp = 1.0 / fill_time  # Process gain
lambda_arrest = fill_time / 5.0  # Fast response: fill_time / 5

Kc, Ti = tank_tuning_pi(Kp, lambda_arrest)
print(f"Tank Controller Gain Kc: {Kc:.2f}")
print(f"Tank Integral Time Ti:   {Ti:.1f} min")
# Output:
# Tank Controller Gain Kc: 2.50
# Tank Integral Time Ti:   8.0 min
```

See [Lambda Tuning Methods](references/03_lambda_tuning.md) for detailed derivations and advanced scenarios.

### 4. Validate Performance

Test the tuned controller in automatic mode with a small setpoint change. Verify the response matches design specifications.

**Expected Response:**
- Self-regulating: Smooth first-order response, settling in ~4λ time constants
- Integrating: Critically damped response, level recovers in ~6 arrest rates
- No overshoot or oscillation (non-oscillatory by design)

**Troubleshooting:**

| Observation | Analysis | Corrective Action |
|-------------|----------|-------------------|
| PV overshoots or oscillates | Lambda too aggressive for model mismatch | Increase λ (3×Td → 4×Td) for more conservative tuning |
| Response very slow, sluggish | Incorrect model or wrong PID form | Verify Kp, τp calculations; check controller algorithm type |
| PV responds smoothly in expected time | Success | Document parameters as baseline |

## Critical Considerations

**Units Must Match:**
- Time constant and integral time must use same units (seconds or minutes)
- Mismatch causes tuning error factor of 60
- Verify controller documentation for expected units

**Dead Time Limits:**
- If Td > 3 × λ, conventional PI fails
- Use advanced methods: Smith Predictor or IMC
- See [Advanced Methods](references/05_advanced_methods.md)

**Model Mismatch:**
- Real processes deviate from first-order model
- Compensate by increasing λ (larger tau ratio)
- Robustness vs. speed tradeoff

**Nonlinearities:**
- Stiction, dead band, varying gain cannot be fixed by tuning
- Require mechanical repair or adaptive control
- See [Nonlinearities](references/06_nonlinearities.md)

## PID Controller Forms

**Critical:** DCS/PLC vendors implement PID differently. Using tuning parameters in the wrong form causes factor-of-Ti errors.

### ISA (Standard/Ideal) Form
```
u(t) = Kc × [e(t) + (1/Ti)∫e(t)dt + Td×de/dt]
```
- Kc multiplies all terms
- Used by: Honeywell, Emerson DeltaV, Siemens

### Parallel (Independent) Form
```
u(t) = Kp×e(t) + Ki∫e(t)dt + Kd×de/dt
```
- Each term has independent gain
- Used by: Allen-Bradley, some Yokogawa

### Conversion Table

| From | To | Kc/Kp | Ti/Ki | Td/Kd |
|------|----|-------|-------|-------|
| ISA → Parallel | | Kp = Kc | Ki = Kc/Ti | Kd = Kc×Td |
| Parallel → ISA | | Kc = Kp | Ti = Kp/Ki | Td = Kd/Kp |

```python
def isa_to_parallel(Kc, Ti, Td=0):
    """Convert ISA form to Parallel form."""
    Kp = Kc
    Ki = Kc / Ti if Ti > 0 else 0
    Kd = Kc * Td
    return Kp, Ki, Kd

def parallel_to_isa(Kp, Ki, Kd=0):
    """Convert Parallel form to ISA form."""
    Kc = Kp
    Ti = Kp / Ki if Ki > 0 else float('inf')
    Td = Kd / Kp if Kp > 0 else 0
    return Kc, Ti, Td

# Example: Lambda tuning gives ISA parameters
Kc, Ti = 0.938, 45.0  # ISA form from lambda tuning
Kp, Ki, Kd = isa_to_parallel(Kc, Ti)
print(f"Parallel: Kp={Kp:.3f}, Ki={Ki:.4f}, Kd={Kd}")
# Output: Parallel: Kp=0.938, Ki=0.0208, Kd=0
```

**Warning:** Entering ISA parameters into a Parallel controller (or vice versa) results in grossly incorrect integral action.

## Additional Resources

### Interactive Tools

- **[Notebooks](notebooks/)** - Jupyter notebooks for iterative tuning workflows
  - **pid_analysis_workflow.ipynb** - Complete control theory analysis using slicot: Bode plots, pole/zero analysis, frequency response, and discretization
  - Uses: slicot for state-space analysis (`tf01md`, `ab04md`, `tb05ad`), numpy for arrays, matplotlib for visualization
  - Use for: Visual validation, frequency domain analysis, documenting tuning sessions, discretization for embedded deployment

- **[Scripts](scripts/)** - Command-line calculation tools
  - **step_test_identify.py** - CSV-based FOPDT identification with proper scaling and regression fitting
  - **lambda_tuning_calculator.py** - Quick PI parameter calculation from process models

### Reference Documentation

- **[Process Identification](references/01_process_identification.md)** - Detailed bump test procedures, slope methods, fill time calculations
- **[PID Fundamentals](references/02_pid_fundamentals.md)** - Deep dive into P, I, D actions and controller forms
- **[Lambda Tuning](references/03_lambda_tuning.md)** - Complete derivation, tau ratio selection, frequency analysis
- **[Integrating Processes](references/04_integrating_processes.md)** - Tank level tuning, arrest rate selection, validation
- **[Advanced Methods](references/05_advanced_methods.md)** - Smith Predictor, IMC, dead time compensation
- **[Nonlinearities](references/06_nonlinearities.md)** - Stiction diagnosis, dead band, adaptive control

## Quick Reference

**Self-Regulating PI Tuning:**
- Kc = (1/Kp) / (λ/τp)
- Ti = τp
- Choose λ ≥ 3×Td

**Integrating PI Tuning:**
- Kc = 2 / (Kp × λ_arrest)
- Ti = 2 × λ_arrest
- λ_arrest = fill_time / M (M=5 for fast, M=2 for slow)

**Validation Check:**
- For ideal tuning: Kc × Kp × Ti = 4
