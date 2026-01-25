#!/usr/bin/env python3
"""
Step Test FOPDT Identification from CSV Data

Identifies First Order Plus Dead Time (FOPDT) model parameters from CSV step test data
with proper engineering unit scaling for meaningful process gain calculation.

Usage:
    uv run scripts/step_test_identify.py --help
    uv run scripts/step_test_identify.py --demo
    uv run scripts/step_test_identify.py data.csv --step-time 100 --pv-range 0 100 --op-range 0 100
"""

import argparse
import warnings
import numpy as np
from scipy.optimize import curve_fit


def load_csv(path):
    """Load CSV file and infer sampling time.

    Expected columns: timestamp, PV, OP (required), SP, dist_* (optional)

    Args:
        path: Path to CSV file

    Returns:
        dict with 'time', 'pv', 'op', 'Ts', and optional 'sp', 'disturbances'
    """
    import pandas as pd

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    time_col = cols_lower.get('timestamp') or cols_lower.get('time') or cols_lower.get('t')
    pv_col = cols_lower.get('pv') or cols_lower.get('process_variable')
    op_col = cols_lower.get('op') or cols_lower.get('output') or cols_lower.get('mv')

    if not all([time_col, pv_col, op_col]):
        raise ValueError("CSV must have timestamp/time, PV, and OP/output columns")

    time = df[time_col].values.astype(float)
    pv = df[pv_col].values.astype(float)
    op = df[op_col].values.astype(float)

    Ts = np.median(np.diff(time))

    result = {'time': time, 'pv': pv, 'op': op, 'Ts': Ts}

    if 'sp' in cols_lower:
        result['sp'] = df[cols_lower['sp']].values.astype(float)

    dist_cols = [c for c in df.columns if c.lower().startswith('dist')]
    if dist_cols:
        result['disturbances'] = df[dist_cols].values

    return result


def scale_data(pv, op, pv_range, op_range):
    """Scale PV and OP to 0-100% for meaningful gain calculation.

    Args:
        pv: Process variable in engineering units
        op: Controller output in engineering units
        pv_range: (min, max) in engineering units
        op_range: (min, max) in engineering units

    Returns:
        (pv_pct, op_pct): Scaled values in percent
    """
    pv_min, pv_max = pv_range
    op_min, op_max = op_range

    pv_pct = 100.0 * (pv - pv_min) / (pv_max - pv_min)
    op_pct = 100.0 * (op - op_min) / (op_max - op_min)

    return pv_pct, op_pct


def detrend_baseline(time, pv, step_time):
    """Remove baseline trend before step.

    Args:
        time: Time array
        pv: Process variable array
        step_time: When step occurred

    Returns:
        pv_detrended: PV with baseline trend removed
    """
    mask = time < step_time
    if np.sum(mask) < 2:
        return pv

    t_base = time[mask]
    pv_base = pv[mask]

    coeffs = np.polyfit(t_base, pv_base, 1)
    trend = np.polyval(coeffs, time)

    return pv - trend + pv_base[0]


def correct_disturbances(pv, disturbances, time, step_time):
    """Remove disturbance effects via regression.

    Args:
        pv: Process variable array
        disturbances: Array of disturbance variables (N x M)
        time: Time array
        step_time: When step occurred (use baseline for fitting)

    Returns:
        pv_corrected: PV with disturbance effects removed
    """
    if disturbances is None or len(disturbances) == 0:
        return pv

    mask = time < step_time
    if np.sum(mask) < 10:
        return pv

    X = disturbances[mask]
    y = pv[mask]

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        correction = disturbances @ coeffs
        return pv - correction + np.mean(pv[mask])
    except np.linalg.LinAlgError:
        return pv


def fopdt_step_response(t, Kp, tau_p, Td, delta_op, t_step):
    """FOPDT step response model.

    y(t) = Kp * delta_op * (1 - exp(-(t-t_step-Td)/tau_p)) for t >= t_step + Td

    Args:
        t: Time array
        Kp: Process gain
        tau_p: Time constant
        Td: Dead time
        delta_op: Step magnitude
        t_step: Step time

    Returns:
        Model response
    """
    y = np.zeros_like(t)
    t_eff = t - t_step - Td
    active = t_eff >= 0
    y[active] = Kp * delta_op * (1.0 - np.exp(-t_eff[active] / tau_p))
    return y


def fit_fopdt(time, pv, op, step_time, baseline_samples=None):
    """Fit FOPDT model using nonlinear least squares (DEFAULT method).

    Args:
        time: Time array
        pv: Process variable (scaled to %)
        op: Controller output (scaled to %)
        step_time: When step occurred
        baseline_samples: Number of samples for baseline (auto if None)

    Returns:
        dict with Kp, tau_p, Td, R2, RMSE, fit_pv
    """
    mask_before = time < step_time
    mask_after = time >= step_time

    n_before = np.sum(mask_before)
    n_after = np.sum(mask_after)

    if baseline_samples is None:
        baseline_samples = max(10, n_before // 2)

    pv_baseline = np.mean(pv[mask_before][-baseline_samples:])
    op_baseline = np.mean(op[mask_before][-baseline_samples:])

    op_after = np.mean(op[mask_after][:min(50, n_after // 5 + 1)])
    delta_op = op_after - op_baseline

    if abs(delta_op) < 0.1:
        raise ValueError("No significant step detected in OP")

    pv_centered = pv - pv_baseline
    time_shifted = time - step_time

    t_fit = time_shifted[mask_after]
    pv_fit = pv_centered[mask_after]

    pv_final_samples = max(10, n_after // 5)
    pv_final = np.mean(pv_fit[-pv_final_samples:])
    Kp_init = pv_final / delta_op if abs(delta_op) > 0.1 else 1.0

    response_duration = time[-1] - step_time

    pv_63_target = 0.632 * pv_final
    tau_p_init = response_duration / 4.0
    for t_val, pv_val in zip(t_fit, pv_fit):
        if abs(pv_val) >= abs(pv_63_target):
            tau_p_init = max(t_val, 1.0)
            break

    pv_threshold = 0.05 * abs(pv_final) if abs(pv_final) > 0.1 else 0.1
    Td_init = 0.0
    for t_val, pv_val in zip(t_fit, pv_fit):
        if abs(pv_val) > pv_threshold:
            Td_init = max(t_val - 0.5, 0.0)
            break

    def model(t, Kp, tau_p, Td):
        y = np.zeros_like(t)
        active = t >= Td
        tau_safe = max(tau_p, 0.01)
        y[active] = Kp * delta_op * (1.0 - np.exp(-(t[active] - Td) / tau_safe))
        return y

    bounds = (
        [-np.inf, 0.1, 0.0],
        [np.inf, response_duration * 2, response_duration / 3]
    )

    try:
        popt, _ = curve_fit(
            model, t_fit, pv_fit,
            p0=[Kp_init, tau_p_init, Td_init],
            bounds=bounds,
            maxfev=10000
        )
        Kp, tau_p, Td = popt
    except RuntimeError as e:
        warnings.warn(f"Curve fit failed: {e}. Using initial estimates.")
        Kp, tau_p, Td = Kp_init, tau_p_init, Td_init

    pv_model = model(t_fit, Kp, tau_p, Td)

    ss_res = np.sum((pv_fit - pv_model) ** 2)
    ss_tot = np.sum((pv_fit - np.mean(pv_fit)) ** 2)
    R2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    RMSE = np.sqrt(np.mean((pv_fit - pv_model) ** 2))

    fit_pv_full = np.zeros_like(time)
    fit_pv_full[mask_before] = pv_baseline
    fit_pv_full[mask_after] = pv_baseline + model(t_fit, Kp, tau_p, Td)

    return {
        'Kp': Kp,
        'tau_p': tau_p,
        'Td': Td,
        'R2': R2,
        'RMSE': RMSE,
        'delta_op': delta_op,
        'pv_baseline': pv_baseline,
        'fit_pv': fit_pv_full,
        'method': 'regression'
    }


def fit_slicot(time, pv, op, Ts, step_time, order=1):
    """Fit model using SLICOT subspace identification (OPTIONAL method).

    Uses ib01ad for preprocessing and ib01bd for state-space extraction.
    Converts discrete state-space to FOPDT parameters.

    Args:
        time: Time array
        pv: Process variable (scaled to %)
        op: Controller output (scaled to %)
        Ts: Sampling time
        step_time: When step occurred
        order: State-space order (default 1 for FOPDT equivalent)

    Returns:
        dict with Kp, tau_p, Td, A, B, C, D matrices
    """
    try:
        import slicot
    except ImportError:
        raise ImportError("SLICOT not available. Use --method=regression instead.")

    mask_after = time >= step_time
    u = op[mask_after].reshape(-1, 1).copy(order='F')
    y = pv[mask_after].reshape(-1, 1).copy(order='F')

    N = len(u)
    m = 1
    l = 1
    nobr = min(N // 4, 50)
    n = order

    try:
        R, _ = slicot.ib01ad(
            'M', 'N', 'C', 'N', 'N',
            nobr, m, l, N,
            u, y
        )

        A, C, B, D, _ = slicot.ib01bd(
            'M', 'C', 'N',
            nobr, n, m, l,
            R
        )
    except Exception as e:
        raise RuntimeError(f"SLICOT identification failed: {e}")

    if n == 1 and A.size == 1:
        a = A[0, 0]
        if abs(a) < 1.0 and a > 0:
            tau_p = -Ts / np.log(a)
        else:
            tau_p = Ts * 5
    else:
        eigs = np.linalg.eigvals(A)
        dominant = eigs[np.argmax(np.abs(eigs))]
        if np.abs(dominant) < 1.0 and np.real(dominant) > 0:
            tau_p = -Ts / np.log(np.abs(dominant))
        else:
            tau_p = Ts * 5

    D_eff = D[0, 0] if D.size > 0 else 0.0
    if n == 1:
        dc_gain = C[0, 0] * B[0, 0] / (1.0 - A[0, 0]) + D_eff
    else:
        try:
            dc_gain = C @ np.linalg.solve(np.eye(n) - A, B) + D_eff
            dc_gain = dc_gain[0, 0]
        except np.linalg.LinAlgError:
            dc_gain = D_eff

    mask_before = time < step_time
    op_baseline = np.mean(op[mask_before]) if np.sum(mask_before) > 0 else op[0]
    op_after = np.mean(op[mask_after][:50])
    delta_op = op_after - op_baseline

    Kp = dc_gain

    Td = estimate_dead_time_from_residual(time[mask_after], y.flatten(), u.flatten(), A, B, C, D, Ts)

    return {
        'Kp': Kp,
        'tau_p': tau_p,
        'Td': Td,
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'delta_op': delta_op,
        'method': 'slicot'
    }


def estimate_dead_time_from_residual(time, y, u, A, B, C, D, Ts):  # noqa: ARG001
    """Estimate dead time from model residual analysis.

    Shifts model output to find best alignment with measured data.

    Args:
        time: Time array (after step)
        y: Measured output
        u: Input
        A, B, C, D: State-space matrices
        Ts: Sampling time

    Returns:
        Estimated dead time
    """
    n = A.shape[0]
    N = len(u)

    y_model = np.zeros(N)
    x = np.zeros((n, 1))
    for k in range(N):
        y_model[k] = (C @ x + D * u[k])[0, 0]
        x = A @ x + B * u[k]

    max_shift = min(50, N // 4)
    best_corr = -np.inf
    best_shift = 0

    for shift in range(max_shift):
        if shift == 0:
            corr = np.corrcoef(y, y_model)[0, 1]
        else:
            corr = np.corrcoef(y[shift:], y_model[:-shift])[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return best_shift * Ts


def simulate_demo_data():
    """Generate demo FOPDT step test data."""
    Kp_true = 1.5
    tau_p_true = 30.0
    Td_true = 5.0
    Ts = 1.0

    t = np.arange(0, 300, Ts)
    op = np.zeros_like(t)
    pv = np.zeros_like(t)

    step_time = 50.0
    step_idx = int(step_time / Ts)
    op[step_idx:] = 60.0
    op[:step_idx] = 50.0

    pv_baseline = 50.0
    delta_op = 10.0

    for i in range(len(t)):
        t_eff = t[i] - step_time - Td_true
        if t_eff < 0:
            pv[i] = pv_baseline
        else:
            pv[i] = pv_baseline + Kp_true * delta_op * (1.0 - np.exp(-t_eff / tau_p_true))

    np.random.seed(42)
    pv += np.random.normal(0, 0.2, size=pv.shape)

    return {
        'time': t,
        'pv': pv,
        'op': op,
        'Ts': Ts,
        'step_time': step_time,
        'true_params': {'Kp': Kp_true, 'tau_p': tau_p_true, 'Td': Td_true}
    }


def plot_results(time, pv, op, fit_result, step_time, save_path=None):
    """Plot identification results."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(time, pv, 'b-', alpha=0.7, label='Measured PV')
    if 'fit_pv' in fit_result:
        ax1.plot(time, fit_result['fit_pv'], 'r--', linewidth=2, label='FOPDT Model')
    ax1.axvline(step_time, color='g', linestyle=':', alpha=0.7, label='Step Time')
    ax1.axvline(step_time + fit_result['Td'], color='orange', linestyle=':', alpha=0.7, label=f"Td={fit_result['Td']:.1f}s")
    ax1.set_ylabel('Process Variable (%)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Step Test FOPDT Identification')

    ax2.plot(time, op, 'g-', linewidth=2, label='Controller Output')
    ax2.axvline(step_time, color='g', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Controller Output (%)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    textstr = f"FOPDT Parameters:\n"
    textstr += f"  Kp = {fit_result['Kp']:.4f}\n"
    textstr += f"  τp = {fit_result['tau_p']:.2f} s\n"
    textstr += f"  Td = {fit_result['Td']:.2f} s\n"
    if 'R2' in fit_result:
        textstr += f"  R² = {fit_result['R2']:.4f}\n"
        textstr += f"  RMSE = {fit_result['RMSE']:.3f}%"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Step Test FOPDT Identification from CSV Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with demo data
  uv run scripts/step_test_identify.py --demo

  # Identify from CSV with engineering unit scaling
  uv run scripts/step_test_identify.py data.csv --step-time 100 \\
      --pv-range 0 200 --op-range 0 100

  # Use SLICOT subspace ID (optional, for higher-order systems)
  uv run scripts/step_test_identify.py data.csv --step-time 100 \\
      --pv-range 0 200 --op-range 0 100 --method slicot
"""
    )

    parser.add_argument('csv_file', nargs='?', help='CSV file with timestamp, PV, OP columns')
    parser.add_argument('--demo', action='store_true', help='Run with simulated demo data')
    parser.add_argument('--step-time', type=float, help='Time when step occurred')
    parser.add_argument('--pv-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='PV range in engineering units (default: auto from data)')
    parser.add_argument('--op-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='OP range in engineering units (default: 0 100)')
    parser.add_argument('--method', choices=['regression', 'slicot'], default='regression',
                        help='Identification method (default: regression)')
    parser.add_argument('--detrend', action='store_true',
                        help='Remove baseline trend before fitting')
    parser.add_argument('--plot', action='store_true', help='Display results plot')
    parser.add_argument('--save-plot', help='Save plot to file path')

    args = parser.parse_args()

    if args.demo:
        print("Running with simulated demo data...\n")
        data = simulate_demo_data()
        step_time = data['step_time']
        pv_range = (0, 100)
        op_range = (0, 100)
        pv_pct, op_pct = data['pv'], data['op']
        time = data['time']
        Ts = data['Ts']
        print(f"True parameters: Kp={data['true_params']['Kp']}, "
              f"τp={data['true_params']['tau_p']}, Td={data['true_params']['Td']}\n")
    elif args.csv_file:
        if not args.step_time:
            parser.error("--step-time is required when using CSV file")

        data = load_csv(args.csv_file)
        time = data['time']
        pv = data['pv']
        op = data['op']
        Ts = data['Ts']
        step_time = args.step_time

        pv_range = tuple(args.pv_range) if args.pv_range else (np.min(pv), np.max(pv))
        op_range = tuple(args.op_range) if args.op_range else (0, 100)

        pv_pct, op_pct = scale_data(pv, op, pv_range, op_range)

        if args.detrend:
            pv_pct = detrend_baseline(time, pv_pct, step_time)

        if 'disturbances' in data:
            pv_pct = correct_disturbances(pv_pct, data['disturbances'], time, step_time)

        print(f"Loaded {len(time)} samples, Ts={Ts:.3f}s")
        print(f"PV range: {pv_range[0]:.1f} to {pv_range[1]:.1f} (scaled to %)")
        print(f"OP range: {op_range[0]:.1f} to {op_range[1]:.1f} (scaled to %)\n")
    else:
        parser.print_help()
        return

    if args.method == 'regression':
        result = fit_fopdt(time, pv_pct, op_pct, step_time)
    else:
        result = fit_slicot(time, pv_pct, op_pct, Ts, step_time)

    print("=" * 60)
    print("FOPDT Identification Results")
    print("=" * 60)
    print(f"\nMethod: {result['method']}")
    print(f"\nProcess Model Parameters:")
    print(f"  Process Gain (Kp):      {result['Kp']:.4f} (%PV / %OP)")
    print(f"  Time Constant (τp):     {result['tau_p']:.2f} seconds")
    print(f"  Dead Time (Td):         {result['Td']:.2f} seconds")

    if 'R2' in result:
        print(f"\nFit Quality:")
        print(f"  R²:                     {result['R2']:.4f}")
        print(f"  RMSE:                   {result['RMSE']:.3f}%")

    print(f"\nStep Change:")
    print(f"  ΔOP:                    {result['delta_op']:.2f}%")

    print(f"\nRecommended Lambda (λ):")
    min_lambda = max(3.0 * result['Td'], result['tau_p'] * 0.5)
    print(f"  Minimum Robust:         {min_lambda:.2f} seconds")
    print(f"  Conservative (4×Td):    {4.0 * result['Td']:.2f} seconds")

    print("\n" + "=" * 60)
    print("Next: Use these parameters with lambda_tuning_calculator.py")
    print("=" * 60 + "\n")

    if args.plot or args.save_plot:
        plot_results(time, pv_pct, op_pct, result, step_time, args.save_plot)


if __name__ == "__main__":
    main()
