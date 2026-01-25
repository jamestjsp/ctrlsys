#!/usr/bin/env python3
"""
Bump Test Data Analysis for Process Identification

Analyzes bump test data to extract process model parameters (Kp, τp, Td)
for use in PID tuning calculations.

Usage:
    uv run scripts/bump_test_analysis.py --help
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def analyze_step_response(time, pv, output, step_time, pv_threshold=0.01):
    """Analyze bump test data to extract process parameters.

    Args:
        time: Time array
        pv: Process variable measurements
        output: Controller output values
        step_time: Time when step change occurred
        pv_threshold: Threshold for detecting response start (fraction of total change)

    Returns:
        dict with Kp, tau_p, Td, and additional analysis data
    """
    # Identify baseline and final values
    baseline_mask = time < step_time
    baseline_pv = np.mean(pv[baseline_mask])
    baseline_output = np.mean(output[baseline_mask[-1]:baseline_mask[-1]+10])

    # Find final settled value (last 20% of data)
    final_start_idx = int(0.8 * len(time))
    final_pv = np.mean(pv[final_start_idx:])
    final_output = np.mean(output[step_time <= time][0:10])

    delta_pv = final_pv - baseline_pv
    delta_output = final_output - baseline_output

    # Calculate process gain
    if abs(delta_output) > 1e-6:
        Kp = delta_pv / delta_output
    else:
        Kp = 0.0

    # Identify dead time (when PV first moves by threshold)
    threshold_value = baseline_pv + pv_threshold * delta_pv
    response_mask = time > step_time

    if delta_pv > 0:
        first_response = np.where(pv[response_mask] > threshold_value)[0]
    else:
        first_response = np.where(pv[response_mask] < threshold_value)[0]

    if len(first_response) > 0:
        response_start_idx = np.where(response_mask)[0][first_response[0]]
        Td = time[response_start_idx] - step_time
    else:
        Td = 0.0

    # Estimate time constant (time to reach 63.2% of final change)
    target_63 = baseline_pv + 0.632 * delta_pv
    after_dead_time = time > (step_time + Td)

    if delta_pv > 0:
        idx_63 = np.where(pv[after_dead_time] >= target_63)[0]
    else:
        idx_63 = np.where(pv[after_dead_time] <= target_63)[0]

    if len(idx_63) > 0:
        time_63_idx = np.where(after_dead_time)[0][idx_63[0]]
        tau_p = time[time_63_idx] - (step_time + Td)
    else:
        # Fallback: estimate from 98% settling time
        target_98 = baseline_pv + 0.98 * delta_pv
        if delta_pv > 0:
            idx_98 = np.where(pv >= target_98)[0]
        else:
            idx_98 = np.where(pv <= target_98)[0]

        if len(idx_98) > 0:
            settling_time = time[idx_98[0]] - (step_time + Td)
            tau_p = settling_time / 4.0  # Approximate
        else:
            tau_p = 0.0

    return {
        'Kp': Kp,
        'tau_p': tau_p,
        'Td': Td,
        'baseline_pv': baseline_pv,
        'final_pv': final_pv,
        'delta_pv': delta_pv,
        'baseline_output': baseline_output,
        'final_output': final_output,
        'delta_output': delta_output,
        'step_time': step_time
    }


def plot_bump_test(time, pv, output, analysis, save_path=None):
    """Create analysis plots for bump test data.

    Args:
        time: Time array
        pv: Process variable array
        output: Controller output array
        analysis: Analysis results dict
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot PV response
    ax1.plot(time, pv, 'b-', linewidth=2, label='Process Variable')
    ax1.axhline(analysis['baseline_pv'], color='g', linestyle='--', alpha=0.7, label='Baseline')
    ax1.axhline(analysis['final_pv'], color='r', linestyle='--', alpha=0.7, label='Final Value')

    # Mark dead time
    dead_time_end = analysis['step_time'] + analysis['Td']
    ax1.axvline(analysis['step_time'], color='orange', linestyle=':', alpha=0.7, label='Step Time')
    ax1.axvline(dead_time_end, color='purple', linestyle=':', alpha=0.7, label='Dead Time End')

    # Mark 63.2% point for time constant
    pv_63 = analysis['baseline_pv'] + 0.632 * analysis['delta_pv']
    tau_time = dead_time_end + analysis['tau_p']
    ax1.plot(tau_time, pv_63, 'ro', markersize=10, label=f'τp Point (63.2%)')

    ax1.set_ylabel('Process Variable')
    ax1.set_title('Bump Test Analysis - Process Variable Response')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Plot controller output
    ax2.plot(time, output, 'g-', linewidth=2, label='Controller Output')
    ax2.axvline(analysis['step_time'], color='orange', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Controller Output (%)')
    ax2.set_title('Controller Output')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    # Add text box with results
    textstr = f"""Process Model Parameters:
    Kp = {analysis['Kp']:.4f}
    τp = {analysis['tau_p']:.2f} s
    Td = {analysis['Td']:.2f} s

    ΔPV = {analysis['delta_pv']:.3f}
    ΔOutput = {analysis['delta_output']:.3f}%"""

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def simulate_example_data():
    """Generate example bump test data for demonstration."""
    Kp_true = 2.0
    tau_p_true = 10.0
    Td_true = 2.0

    dt = 0.1
    time = np.arange(0, 80, dt)
    output = np.zeros_like(time)
    pv = np.zeros_like(time)

    # Step change at t=10s
    step_time = 10.0
    output[time >= step_time] = 5.0  # 5% step

    # First-order + dead time response
    for i in range(1, len(time)):
        if time[i] < (step_time + Td_true):
            pv[i] = pv[i-1]
        else:
            idx_delayed = i - int(Td_true / dt)
            if idx_delayed >= 0:
                dpv_dt = (Kp_true * output[idx_delayed] - pv[i-1]) / tau_p_true
                pv[i] = pv[i-1] + dpv_dt * dt

    # Add small noise
    pv += np.random.normal(0, 0.02, size=pv.shape)

    return time, pv, output, step_time


def main():
    parser = argparse.ArgumentParser(
        description='Bump Test Data Analysis for Process Identification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--demo', action='store_true',
                        help='Run with simulated demo data')
    parser.add_argument('--time_file', help='CSV file with time data')
    parser.add_argument('--pv_file', help='CSV file with PV data')
    parser.add_argument('--output_file', help='CSV file with output data')
    parser.add_argument('--step_time', type=float, help='Time when step occurred')
    parser.add_argument('--plot', action='store_true', help='Display analysis plots')
    parser.add_argument('--save_plot', help='Save plot to file path')

    args = parser.parse_args()

    if args.demo:
        print("Running with simulated demo data...\n")
        time, pv, output, step_time = simulate_example_data()
    elif all([args.time_file, args.pv_file, args.output_file, args.step_time]):
        time = np.loadtxt(args.time_file, delimiter=',')
        pv = np.loadtxt(args.pv_file, delimiter=',')
        output = np.loadtxt(args.output_file, delimiter=',')
        step_time = args.step_time
    else:
        print("ERROR: Either use --demo or provide all data files and --step_time")
        parser.print_help()
        return

    # Analyze the data
    analysis = analyze_step_response(time, pv, output, step_time)

    # Print results
    print("=" * 60)
    print("Bump Test Analysis Results")
    print("=" * 60)
    print(f"\nProcess Model Parameters:")
    print(f"  Process Gain (Kp):            {analysis['Kp']:.4f}")
    print(f"  Time Constant (τp):           {analysis['tau_p']:.2f} seconds")
    print(f"  Dead Time (Td):               {analysis['Td']:.2f} seconds")

    print(f"\nMeasured Changes:")
    print(f"  ΔPV:                          {analysis['delta_pv']:.4f}")
    print(f"  ΔOutput:                      {analysis['delta_output']:.2f}%")
    print(f"  Baseline PV:                  {analysis['baseline_pv']:.4f}")
    print(f"  Final PV:                     {analysis['final_pv']:.4f}")

    print(f"\nRecommended Lambda (λ):")
    min_lambda = 3.0 * analysis['Td']
    print(f"  Minimum Robust (3×Td):        {min_lambda:.2f} seconds")
    print(f"  Conservative (4×Td):          {4.0 * analysis['Td']:.2f} seconds")

    print("\n" + "=" * 60)
    print("Next Step: Use these parameters with lambda_tuning_calculator.py")
    print("=" * 60 + "\n")

    # Plot if requested
    if args.plot or args.save_plot:
        plot_bump_test(time, pv, output, analysis, args.save_plot)


if __name__ == "__main__":
    main()
