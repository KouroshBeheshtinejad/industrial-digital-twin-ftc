from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from python.plant_model.cstr_model import (
    load_parameters,
    cstr_dynamics
)


ROOT = Path(__file__).resolve().parents[2]

SETPOINT = 356.0

COOLANT_FLOWS = np.linspace(0.02, 0.50, 100)

INITIAL_STATE = np.array([
    64.50615287,
    355.28373576,
    305.90638203
])


def find_equilibrium(params, coolant_flow):
    """
    Find the steady-state operating point for a fixed coolant flow.

    The equilibrium is calculated by solving the three nonlinear
    steady-state equations using the process model.
    """

    def steady_state_residual(state):
        return cstr_dynamics(
            0.0,
            state,
            coolant_flow,
            params
        )

    from scipy.optimize import root

    solution = root(
        steady_state_residual,
        INITIAL_STATE,
        method="hybr"
    )

    if not solution.success:
        return None

    equilibrium_state = solution.x

    residual_norm = np.linalg.norm(
        steady_state_residual(equilibrium_state)
    )

    if (
        not np.all(np.isfinite(equilibrium_state))
        or residual_norm > 1e-5
        or equilibrium_state[0] <= 0
        or equilibrium_state[1] <= 0
        or equilibrium_state[2] <= 0
    ):
        return None

    return equilibrium_state


def analyze_operating_range(params):
    rows = []

    for coolant_flow in COOLANT_FLOWS:
        equilibrium = find_equilibrium(
            params,
            coolant_flow
        )

        if equilibrium is None:
            continue

        concentration, temperature, jacket_temperature = equilibrium

        rows.append({
            "coolant_flow": coolant_flow,
            "concentration_mol_m3": concentration,
            "reactor_temperature_K": temperature,
            "jacket_temperature_K": jacket_temperature,
            "temperature_error_K": SETPOINT - temperature,
            "setpoint_feasible": abs(temperature - SETPOINT) <= 0.1
        })

    return pd.DataFrame(rows)


def find_setpoint_flow(params):
    """
    Search for a coolant flow that produces the requested
    steady-state reactor temperature.
    """

    def temperature_error(coolant_flow):
        equilibrium = find_equilibrium(
            params,
            coolant_flow
        )

        if equilibrium is None:
            raise ValueError(
                "Equilibrium could not be calculated."
            )

        return equilibrium[1] - SETPOINT

    minimum_flow = params.minimum_coolant_flow
    maximum_flow = params.maximum_coolant_flow

    lower_error = temperature_error(minimum_flow)
    upper_error = temperature_error(maximum_flow)

    if lower_error * upper_error > 0:
        return None

    return brentq(
        temperature_error,
        minimum_flow,
        maximum_flow,
        xtol=1e-10
    )


def main():
    params = load_parameters(
        ROOT / "configs" / "plant_parameters" / "cstr_nominal.json"
    )

    results = analyze_operating_range(params)

    if results.empty:
        raise RuntimeError(
            "No valid steady-state operating points were found."
        )

    target_flow = find_setpoint_flow(params)

    print("\n========== ACTUATOR AUTHORITY ANALYSIS ==========")

    print(f"Temperature setpoint: {SETPOINT:.2f} K")
    print(
        f"Coolant flow range: "
        f"{params.minimum_coolant_flow:.3f} - "
        f"{params.maximum_coolant_flow:.3f} m^3/min"
    )

    print("\nOperating temperature range:")
    print(
        f"Minimum: {results['reactor_temperature_K'].min():.4f} K"
    )
    print(
        f"Maximum: {results['reactor_temperature_K'].max():.4f} K"
    )

    if target_flow is not None:
        target_state = find_equilibrium(
            params,
            target_flow
        )

        print("\nSetpoint operating point:")
        print(f"Required coolant flow: {target_flow:.6f} m^3/min")
        print(f"Reactor temperature: {target_state[1]:.6f} K")
        print(f"Concentration: {target_state[0]:.6f} mol/m^3")
        print(f"Jacket temperature: {target_state[2]:.6f} K")
    else:
        print(
            "\nThe requested setpoint is not reachable "
            "within the nominal coolant flow range."
        )

    stuck_flow = params.maximum_coolant_flow

    stuck_state = find_equilibrium(
        params,
        stuck_flow
    )

    print("\nStuck-at-maximum actuator condition:")
    print(f"Actual coolant flow: {stuck_flow:.4f} m^3/min")

    if stuck_state is not None:
        print(
            f"Steady-state reactor temperature: "
            f"{stuck_state[1]:.6f} K"
        )
        print(
            f"Steady-state tracking error: "
            f"{SETPOINT - stuck_state[1]:.6f} K"
        )

    output_tables = ROOT / "results" / "tables"
    output_figures = ROOT / "results" / "figures"

    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_tables / "actuator_authority_analysis.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        results["coolant_flow"],
        results["reactor_temperature_K"],
        linewidth=2,
        label="Steady-state reactor temperature"
    )

    ax.axhline(
        SETPOINT,
        color="black",
        linestyle="--",
        label="Temperature setpoint"
    )

    ax.axvline(
        stuck_flow,
        color="red",
        linestyle=":",
        label="Stuck actuator flow"
    )

    if target_flow is not None:
        ax.axvline(
            target_flow,
            color="green",
            linestyle="--",
            label="Required coolant flow"
        )

    ax.set_xlabel("Coolant flow (m³/min)")
    ax.set_ylabel("Reactor temperature (K)")
    ax.set_title("CSTR Steady-State Actuator Authority Analysis")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    figure_path = (
        output_figures / "actuator_authority_analysis.png"
    )

    plt.savefig(figure_path, dpi=300)
    plt.close()

    print("\nOutput files saved successfully.")
    print(
        f"Table: "
        f"{output_tables / 'actuator_authority_analysis.csv'}"
    )
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
