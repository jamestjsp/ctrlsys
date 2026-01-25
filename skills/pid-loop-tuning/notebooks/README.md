# PID Tuning Notebooks

Interactive Jupyter notebooks for iterative PID controller design and analysis using control theory.

## Purpose

Notebooks provide a data science-style workflow for PID tuning that enables:
- **Iterative exploration** - Adjust parameters, immediately see results
- **Visual validation** - Bode plots, step responses, stability margins
- **Documentation** - Save complete analysis sessions with results
- **Reproducibility** - Share notebooks with team for peer review
- **Teaching** - Excellent tool for training engineers on control theory

## Available Notebooks

### pid_analysis_workflow.ipynb

Complete 7-step PID controller design and validation workflow:

1. **Specify Plant Model** - Define transfer function with dead time
2. **Plant Step Response** - Visualize open-loop behavior
3. **Set PID Parameters** - Enter tuning from Lambda calculations
4. **Closed-Loop Analysis** - Reference tracking and disturbance rejection
5. **Stability Margins** - Gain margin, phase margin, Bode plots
6. **Discretization** - Convert to discrete-time for embedded systems
7. **Discrete Validation** - Verify digital controller performance

**Use this for:**
- Advanced frequency domain analysis
- Stability margin verification
- Controller discretization for PLC/DCS implementation
- Academic-quality documentation of tuning decisions

## Quick Start

### Installation

```bash
# Install Jupyter and control theory packages
uv pip install jupyter numpy scipy matplotlib python-control

# Or install all at once
uv pip install jupyter numpy scipy matplotlib python-control ipykernel
```

### Launch Jupyter

```bash
# From skill root directory
cd /path/to/pid-loop-tuning

# Launch Jupyter Lab (recommended)
uv run jupyter lab notebooks/

# Or launch Jupyter Notebook (classic interface)
uv run jupyter notebook notebooks/
```

### Open and Run

1. Browser opens automatically at `http://localhost:8888`
2. Click `pid_analysis_workflow.ipynb`
3. Run cells sequentially: `Shift+Enter` or click ▶️ Run button
4. Modify parameters in cells 1 (Plant) and 2 (Controller)
5. Re-run analysis cells to see updated results

## Typical Workflow

### Step 1: Perform Bump Test (DCS/Field)

```bash
# Use scripts to analyze bump test data
uv run scripts/bump_test_analysis.py --demo --plot
# Extract: Kp=2.0, τp=10.0s, Td=2.0s
```

### Step 2: Calculate Tuning (Scripts or Manual)

```bash
# Use lambda tuning calculator
uv run scripts/lambda_tuning_calculator.py \
    --type self \
    --Kp 2.0 \
    --tau_p 10.0 \
    --lambda 6.0
# Get: Kc=0.833, Ti=10.0s
```

### Step 3: Validate in Notebook (Control Theory)

1. Open `pid_analysis_workflow.ipynb`
2. **Cell 1:** Enter plant transfer function from process model
3. **Cell 2:** Enter Kc, Ti from lambda tuning
4. **Run all cells** to see:
   - Step response (should be non-oscillatory)
   - Bode plot with stability margins (GM > 2, PM > 30°)
   - Frequency domain validation
5. **Iterate:** Adjust lambda if margins insufficient, recalculate, re-run

### Step 4: Implement in DCS

Use validated parameters from notebook in field controller.

### Step 5: Document Results

- Save notebook with results: `File → Save`
- Export to PDF: `File → Export → PDF`
- Commit to git for version control
- Share with team for peer review

## When to Use Notebooks vs Scripts

### Use Notebooks When:

✅ Need visual feedback (plots, graphs)
✅ Iteratively exploring tuning options
✅ Documenting a specific tuning session
✅ Teaching control theory concepts
✅ Advanced frequency domain analysis
✅ Verifying stability margins mathematically
✅ Discretizing controller for embedded systems

### Use Scripts When:

✅ Quick parameter calculation
✅ Batch processing bump test data
✅ Command-line automation
✅ Simple lambda tuning calculation
✅ Field work without Jupyter access

## Example: Complete Tuning Session

```python
# In pid_analysis_workflow.ipynb, Cell 1: Plant Model

import numpy as np
import control as ctrl

# From bump test: Kp=2.0, τp=10.0s, Td=2.0s
# First-order plus dead time (FOPTD) model
Kp_process = 2.0
tau_p = 10.0
Td = 2.0

# Transfer function: G(s) = Kp / (τp*s + 1) * e^(-Td*s)
num = [Kp_process]
den = [tau_p, 1]
plant = ctrl.TransferFunction(num, den)

# Pade approximation for dead time
pade_order = 3
num_delay, den_delay = ctrl.pade(Td, pade_order)
delay_tf = ctrl.TransferFunction(num_delay, den_delay)

G = ctrl.series(plant, delay_tf)
G
```

```python
# Cell 2: PID Controller Parameters

# From lambda tuning (λ = 3×Td = 6.0s)
Kc = 0.833  # Controller gain
Ti = 10.0   # Integral time (= τp)
Td = 0.0    # Derivative time (not used)

# PI controller: C(s) = Kc * (1 + 1/(Ti*s))
C = ctrl.TransferFunction([Kc*Ti, Kc], [Ti, 0])
C
```

**Result:** Notebook generates Bode plot showing GM=6.2dB, PM=42°, validates tuning is robust.

## Notebook Best Practices

### 1. Document Assumptions

Add markdown cells explaining:
- Where process parameters came from (bump test date, conditions)
- Why specific lambda value was chosen
- Any deviations from standard tuning rules

### 2. Version Control

```bash
# Save notebook with descriptive name
# Example: FIC-101_tuning_2025-10-20.ipynb

# Commit to git
git add notebooks/FIC-101_tuning_2025-10-20.ipynb
git commit -m "FIC-101 flow loop tuning validation"
```

### 3. Clear Outputs Before Committing

```bash
# Optional: Strip outputs to reduce file size
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

### 4. Use Markdown Liberally

Document:
- Equipment tag numbers
- Test conditions (flow rate, temperature, etc.)
- Assumptions made
- Results interpretation
- Recommendations

## Dependencies

| Package | Purpose | Installation |
|---------|---------|--------------|
| **jupyter** | Interactive notebook environment | `uv pip install jupyter` |
| **numpy** | Numerical computation | `uv pip install numpy` |
| **scipy** | Scientific computing | `uv pip install scipy` |
| **matplotlib** | Plotting and visualization | `uv pip install matplotlib` |
| **python-control** | Control systems analysis | `uv pip install control` |

### Optional Dependencies

| Package | Purpose | Installation |
|---------|---------|--------------|
| **pandas** | Data analysis (for bump test CSV) | `uv pip install pandas` |
| **seaborn** | Statistical visualization | `uv pip install seaborn` |

## Troubleshooting

### Jupyter Won't Start

```bash
# Ensure Jupyter is installed
uv pip install jupyter

# Try specifying full path
uv run python -m jupyter lab notebooks/
```

### Import Errors

```bash
# Install missing package
uv pip install control  # or numpy, scipy, matplotlib
```

### Plots Not Displaying

```python
# Add to first cell of notebook
%matplotlib inline
import matplotlib.pyplot as plt
```

### Kernel Crashes

```bash
# Update to latest python-control
uv pip install --upgrade control
```

## Additional Resources

- **[SKILL.md](../SKILL.md)** - Main PID tuning methodology
- **[Lambda Tuning](../reference/03_lambda_tuning.md)** - Tuning calculations explained
- **[Scripts](../scripts/)** - Command-line tools for quick calculations
- **[python-control docs](https://python-control.readthedocs.io/)** - Control package documentation

## Contributing Notebooks

Have a useful tuning notebook? Add it here!

**Guidelines:**
- Include clear markdown documentation
- Add descriptive cell headers
- Document all assumptions
- Include example with realistic parameters
- Test all cells run without errors
- Add entry to this README

**Suggested Topics:**
- Cascade control tuning example
- Smith Predictor implementation
- Tank level tuning with simulation
- Model mismatch sensitivity analysis
- Real bump test data analysis
