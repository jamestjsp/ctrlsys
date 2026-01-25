#!/usr/bin/env python3
"""
Lambda Tuning Calculator for PID Controllers

Interactive calculator for computing PI controller parameters using
the Direct Synthesis (Lambda Tuning) method for both self-regulating
and integrating processes.

Usage:
    uv run scripts/lambda_tuning_calculator.py
"""

import argparse


def calculate_self_regulating(Kp: float, tau_p: float, lambda_cl: float) -> tuple[float, float, float]:
    """Calculate PI tuning for self-regulating process.

    Args:
        Kp: Process gain (ΔPV/ΔOutput)
        tau_p: Process time constant (seconds or minutes)
        lambda_cl: Desired closed-loop time constant (same units as tau_p)

    Returns:
        (Kc, Ti, tau_ratio): Controller gain, integral time, and tau ratio
    """
    tau_ratio = lambda_cl / tau_p
    Kc = (1.0 / Kp) / tau_ratio
    Ti = tau_p
    return Kc, Ti, tau_ratio


def calculate_integrating(Kp: float, lambda_arrest: float) -> tuple[float, float]:
    """Calculate PI tuning for integrating process (tank level).

    Args:
        Kp: Process gain (1/fill_time) in 1/time_units
        lambda_arrest: Desired arrest rate (same time units as Kp)

    Returns:
        (Kc, Ti): Controller gain and integral time
    """
    Kc = 2.0 / (Kp * lambda_arrest)
    Ti = 2.0 * lambda_arrest
    return Kc, Ti


def calculate_from_fill_time(fill_time: float, speed_factor: float = 5.0) -> tuple[float, float, float]:
    """Calculate integrating process tuning from tank fill time.

    Args:
        fill_time: Time to fill tank 0% to 100% (minutes)
        speed_factor: M factor for arrest rate (5=fast, 2=slow)

    Returns:
        (Kp, lambda_arrest, (Kc, Ti)): Process gain, arrest rate, and tuning
    """
    Kp = 1.0 / fill_time
    lambda_arrest = fill_time / speed_factor
    Kc, Ti = calculate_integrating(Kp, lambda_arrest)
    return Kp, lambda_arrest, (Kc, Ti)


def validate_tuning(Kc: float, Kp: float, Ti: float) -> dict:
    """Validate tuning parameters for standard PID form.

    Args:
        Kc: Controller gain
        Kp: Process gain
        Ti: Integral time

    Returns:
        dict with validation results
    """
    product = Kc * Kp * Ti
    ideal_product = 4.0
    error_pct = abs(product - ideal_product) / ideal_product * 100

    return {
        'product': product,
        'ideal': ideal_product,
        'error_percent': error_pct,
        'status': 'Ideal' if error_pct < 5 else ('Acceptable' if error_pct < 15 else 'Review')
    }


def main():
    parser = argparse.ArgumentParser(
        description='Lambda Tuning Calculator for PID Controllers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Self-Regulating Process:
    %(prog)s --type self --Kp 2.0 --tau_p 10.0 --lambda 30.0

  Integrating Process (from fill time):
    %(prog)s --type tank --fill_time 20.0 --speed_factor 5

  Integrating Process (direct):
    %(prog)s --type integrating --Kp 0.05 --lambda 4.0
        """
    )

    parser.add_argument('--type', required=True, choices=['self', 'integrating', 'tank'],
                        help='Process type: self-regulating, integrating, or tank')
    parser.add_argument('--Kp', type=float, help='Process gain')
    parser.add_argument('--tau_p', type=float, help='Process time constant (for self-regulating)')
    parser.add_argument('--lambda', dest='lambda_val', type=float,
                        help='Desired closed-loop time constant or arrest rate')
    parser.add_argument('--fill_time', type=float, help='Tank fill time (for tank type)')
    parser.add_argument('--speed_factor', type=float, default=5.0,
                        help='Speed factor M for tank (default: 5.0)')
    parser.add_argument('--time_units', default='seconds', help='Time units (default: seconds)')

    args = parser.parse_args()

    print("=" * 60)
    print("Lambda Tuning Calculator - Direct Synthesis Method")
    print("=" * 60)

    if args.type == 'self':
        # Self-regulating process
        if not all([args.Kp, args.tau_p, args.lambda_val]):
            parser.error("self type requires --Kp, --tau_p, and --lambda")

        Kc, Ti, tau_ratio = calculate_self_regulating(args.Kp, args.tau_p, args.lambda_val)

        print(f"\nProcess Type: Self-Regulating")
        print(f"\nInput Parameters:")
        print(f"  Process Gain (Kp):            {args.Kp:.4f}")
        print(f"  Time Constant (τp):           {args.tau_p:.2f} {args.time_units}")
        print(f"  Lambda (λ):                   {args.lambda_val:.2f} {args.time_units}")
        print(f"  Tau Ratio (λ/τp):             {tau_ratio:.2f}")

        print(f"\nCalculated PI Tuning Parameters:")
        print(f"  Controller Gain (Kc):         {Kc:.4f}")
        print(f"  Integral Time (Ti):           {Ti:.2f} {args.time_units}")
        print(f"  Derivative Time (Td):         0.0 {args.time_units}")

        validation = validate_tuning(Kc, args.Kp, Ti)
        print(f"\nValidation (Kc × Kp × Ti):")
        print(f"  Calculated Product:           {validation['product']:.3f}")
        print(f"  Ideal Product:                {validation['ideal']:.3f}")
        print(f"  Error:                        {validation['error_percent']:.1f}%")
        print(f"  Status:                       {validation['status']}")

        print(f"\nExpected Response:")
        settling_time = 4 * args.lambda_val
        print(f"  Settling Time (~4λ):          {settling_time:.1f} {args.time_units}")
        print(f"  Response Type:                Non-oscillatory first-order")

    elif args.type == 'tank':
        # Tank using fill time method
        if not args.fill_time:
            parser.error("tank type requires --fill_time")

        Kp, lambda_arrest, (Kc, Ti) = calculate_from_fill_time(args.fill_time, args.speed_factor)

        print(f"\nProcess Type: Integrating (Tank Level)")
        print(f"\nInput Parameters:")
        print(f"  Fill Time:                    {args.fill_time:.2f} {args.time_units}")
        print(f"  Speed Factor (M):             {args.speed_factor:.1f}")

        print(f"\nDerived Parameters:")
        print(f"  Process Gain (Kp):            {Kp:.4f} 1/{args.time_units}")
        print(f"  Arrest Rate (λ):              {lambda_arrest:.2f} {args.time_units}")

        print(f"\nCalculated PI Tuning Parameters:")
        print(f"  Controller Gain (Kc):         {Kc:.4f}")
        print(f"  Integral Time (Ti):           {Ti:.2f} {args.time_units}")

        validation = validate_tuning(Kc, Kp, Ti)
        print(f"\nValidation (Kc × Kp × Ti):")
        print(f"  Calculated Product:           {validation['product']:.3f}")
        print(f"  Ideal Product:                {validation['ideal']:.3f}")
        print(f"  Status:                       {validation['status']}")

        recovery_time = 6 * lambda_arrest
        print(f"\nExpected Response:")
        print(f"  Recovery Time (~6λ):          {recovery_time:.1f} {args.time_units}")
        print(f"  Response Type:                Critically damped second-order")

    else:  # integrating
        # Integrating process direct parameters
        if not all([args.Kp, args.lambda_val]):
            parser.error("integrating type requires --Kp and --lambda")

        Kc, Ti = calculate_integrating(args.Kp, args.lambda_val)

        print(f"\nProcess Type: Integrating")
        print(f"\nInput Parameters:")
        print(f"  Process Gain (Kp):            {args.Kp:.4f} 1/{args.time_units}")
        print(f"  Arrest Rate (λ):              {args.lambda_val:.2f} {args.time_units}")

        print(f"\nCalculated PI Tuning Parameters:")
        print(f"  Controller Gain (Kc):         {Kc:.4f}")
        print(f"  Integral Time (Ti):           {Ti:.2f} {args.time_units}")

        validation = validate_tuning(Kc, args.Kp, Ti)
        print(f"\nValidation (Kc × Kp × Ti):")
        print(f"  Calculated Product:           {validation['product']:.3f}")
        print(f"  Ideal Product:                {validation['ideal']:.3f}")
        print(f"  Status:                       {validation['status']}")

        recovery_time = 6 * args.lambda_val
        print(f"\nExpected Response:")
        print(f"  Recovery Time (~6λ):          {recovery_time:.1f} {args.time_units}")

    print("\n" + "=" * 60)
    print("Note: Ensure time units for τp and Ti match controller settings!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
