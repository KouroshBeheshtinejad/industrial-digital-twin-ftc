
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from python.plant_model.cstr_model import (
    load_parameters,
    cstr_dynamics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plant_parameters"
    / "cstr_nominal.json"
)

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def integrate_process(state, coolant_flow, params, start_time, end_time):

    solution = solve_ivp(
        fun=lambda t, x: cstr_dynamics(
            t,
            x,
            coolant_flow,
            params,
        ),
        t_span=(start_time, end_time),
        y0=state,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y[:, -1]


def simulate_model_mismatch(
    nominal_params,
    initial_state,
    simulation_time=40.0,
    sample_time=0.05,
    ua_multiplier=1.10,
    noise_std=0.05,
    random_seed=6000,
):

    plant_params = replace(
        nominal_params,
        heat_transfer_coefficient=(
            nominal_params.heat_transfer_coefficient
            * ua_multiplier
        ),
    )

    twin_params = nominal_params

    number_of_steps = int(simulation_time / sample_time) + 1

    time = np.linspace(
        0.0,
        simulation_time,
        number_of_steps,
    )

    plant_states = np.zeros((number_of_steps, 3))
    twin_states = np.zeros((number_of_steps, 3))

    measurements = np.zeros(number_of_steps)
    commanded_flows = np.zeros(number_of_steps)
    actual_flows = np.zeros(number_of_steps)

    plant_states[0] = initial_state
    twin_states[0] = initial_state

    rng = np.random.default_rng(random_seed)

    integral_error = 0.0
    coolant_bias = nominal_params.nominal_coolant_flow

    setpoint = 356.0
    kp = 0.03
    ki = 0.003

    for i in range(number_of_steps - 1):

        measurement = (
            plant_states[i, 1]
            + rng.normal(0.0, noise_std)
        )

        error = setpoint - measurement

        unconstrained_flow = (
            coolant_bias
            - kp * error
            - ki * integral_error
        )

        integration_allowed = (
            nominal_params.minimum_coolant_flow
            < unconstrained_flow
            < nominal_params.maximum_coolant_flow
            or (
                unconstrained_flow
                >= nominal_params.maximum_coolant_flow
                and error > 0
            )
            or (
                unconstrained_flow
                <= nominal_params.minimum_coolant_flow
                and error < 0
            )
        )

        if integration_allowed:
            integral_error += error * sample_time

        unconstrained_flow = (
            coolant_bias
            - kp * error
            - ki * integral_error
        )

        coolant_flow = np.clip(
            unconstrained_flow,
            nominal_params.minimum_coolant_flow,
            nominal_params.maximum_coolant_flow,
        )

        plant_states[i + 1] = integrate_process(
            plant_states[i],
            coolant_flow,
            plant_params,
            time[i],
            time[i + 1],
        )

        twin_states[i + 1] = integrate_process(
            twin_states[i],
            coolant_flow,
            twin_params,
            time[i],
            time[i + 1],
        )

        measurements[i] = measurement
        commanded_flows[i] = coolant_flow
        actual_flows[i] = coolant_flow

    measurements[-1] = (
        plant_states[-1, 1]
        + rng.normal(0.0, noise_std)
    )

    commanded_flows[-1] = commanded_flows[-2]
    actual_flows[-1] = actual_flows[-2]

    return {
        "time": time,
        "plant_states": plant_states,
        "twin_states": twin_states,
        "measurements": measurements,
        "commanded_flows": commanded_flows,
        "actual_flows": actual_flows,
        "plant_params": plant_params,
    }


def run_experiment():

    nominal_params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    number_of_runs = 30

    records = []

    for run_index in range(number_of_runs):

        results = simulate_model_mismatch(
            nominal_params=nominal_params,
            initial_state=initial_state,
            simulation_time=40.0,
            sample_time=0.05,
            ua_multiplier=1.10,
            noise_std=0.05,
            random_seed=6000 + run_index,
        )

        plant_states = results["plant_states"]
        twin_states = results["twin_states"]

        records.append({
            "time": results["time"],
            "r_T": (
                results["measurements"]
                - twin_states[:, 1]
            ),
            "r_Tj": (
                plant_states[:, 2]
                - twin_states[:, 2]
            ),
            "r_C": (
                plant_states[:, 0]
                - twin_states[:, 0]
            ),
            "r_u": (
                results["actual_flows"]
                - results["commanded_flows"]
            ),
        })

    return records, nominal_params


def calculate_statistics(records):

    rows = []

    for signal_name in ["r_T", "r_Tj", "r_C", "r_u"]:

        values = np.concatenate([
            record[signal_name]
            for record in records
        ])

        rows.append({
            "residual": signal_name,
            "mean": np.mean(values),
            "std": np.std(values),
            "max_absolute": np.max(np.abs(values)),
            "percentile_99_abs": np.percentile(
                np.abs(values),
                99,
            ),
        })

    return pd.DataFrame(rows)


def save_results(records, statistics):

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        TABLES_DIR
        / "fault_isolation_model_mismatch_signature.csv"
    )

    figure_path = (
        FIGURES_DIR
        / "fault_isolation_model_mismatch_signature.png"
    )

    statistics.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 12),
        sharex=True,
    )

    signal_names = ["r_T", "r_Tj", "r_C", "r_u"]

    for axis, signal_name in zip(axes, signal_names):

        record = records[0]

        axis.plot(
            record["time"],
            record[signal_name],
        )

        axis.axhline(0, linestyle=":")

        axis.set_ylabel(signal_name)
        axis.grid(True)

    axes[-1].set_xlabel("Time (min)")

    fig.suptitle(
        "Model Mismatch - Residual Signature"
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(f"\nCSV saved to: {csv_path}")
    print(f"Figure saved to: {figure_path}")


def main():

    records, nominal_params = run_experiment()

    statistics = calculate_statistics(records)

    print("\n========== MODEL MISMATCH SIGNATURE ==========")

    print(
        f"Nominal UA: "
        f"{nominal_params.heat_transfer_coefficient:.2f}"
    )

    print(
        f"Actual UA: "
        f"{nominal_params.heat_transfer_coefficient * 1.10:.2f}"
    )

    print("Model mismatch: 10%")
    print("Sensor bias: None")
    print("Actuator fault: None")

    print("\n========== RESIDUAL STATISTICS ==========")

    print(
        statistics.to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    save_results(records, statistics)


if __name__ == "__main__":
    main()
