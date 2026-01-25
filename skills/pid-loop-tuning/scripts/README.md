# PID Loop Tuning Scripts

Executable Python tools for process identification and PID controller tuning.

## Installation

Scripts use `uv` for dependency management. Install dependencies:

```bash
uv pip install numpy scipy matplotlib control
```

Or run scripts directly with `uv run`:

```bash
uv run scripts/lambda_tuning_calculator.py --help
```

## Available Scripts

### 1. lambda_tuning_calculator.py

Interactive calculator for PI tuning parameters using Direct Synthesis (Lambda Tuning).

**Examples:**
```bash
# Self-regulating process (temperature, flow, pressure)
uv run scripts/lambda_tuning_calculator.py --type self --Kp 2.0 --tau_p 10.0 --lambda 30.0

# Tank level from fill time
uv run scripts/lambda_tuning_calculator.py --type tank --fill_time 20.0 --speed_factor 5

# Integrating process (direct parameters)
uv run scripts/lambda_tuning_calculator.py --type integrating --Kp 0.05 --lambda 4.0
```

### 2. bump_test_analysis.py

Analyze bump test data to extract process model parameters (Kp, τp, Td).

**Examples:**
```bash
# Run with simulated demo data
uv run scripts/bump_test_analysis.py --demo --plot

# Analyze real data from CSV files
uv run scripts/bump_test_analysis.py \
    --time_file time.csv \
    --pv_file pv.csv \
    --output_file output.csv \
    --step_time 10.0 \
    --plot
```

### 3. pid_bode_analysis.py

Advanced frequency domain analysis with Bode plots and stability margins.

**Usage:**
```bash
uv run scripts/pid_bode_analysis.py
```

## Quick Workflow

1. **Perform Bump Test** - Collect time-series data in manual mode
2. **Analyze Data** - Use `bump_test_analysis.py` to extract Kp, τp, Td
3. **Calculate Tuning** - Use `lambda_tuning_calculator.py` to get Kc, Ti
4. **Implement** - Enter parameters into DCS/PLC controller
5. **Validate** - Test with small setpoint change in automatic mode

## Dependencies

- **numpy** - Numerical computation
- **scipy** - Scientific computing (optional for advanced analysis)
- **matplotlib** - Plotting (optional, for --plot flags)
- **python-control** - Control systems library (for advanced analysis)

## Notes

- All time units must be consistent (seconds or minutes)
- Lambda (λ) should be ≥ 3 × dead time for robust tuning
- For tank levels, use fill time method for most accurate Kp
- Derivative (D) term should typically be set to zero
