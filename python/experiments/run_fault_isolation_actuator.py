from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from python.plant_model.cstr_model import load_parameters
from python.experiments.run_actuator_fault_detection import (
    simulate_actuator_fault,
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


def run_actuator_fault_experiment():

    params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    number_of_runs = 30

    simulation_time = 40.0
    sample_time = 0.05

    fault_time = 20.0
    stuck_flow = 0.50

    records = []

    for run_index in range(number_of_runs):

        results = simulate_actuator_fault(
            params=params,
            initial_state=initial_state,
            setpoint=356.0,
            kp=0.03,
            ki=0.003,
            simulation_time=simulation_time,
            sample_time=sample_time,
            fault_time=fault_time,
            stuck_flow=stuck_flow,
            noise_std=0.05,
            detection_threshold=0.75,
            persistence_samples=5,
            random_seed=5000 + run_index,
        )

        time = results["time"]

        plant_states = results["plant_states"]
        twin_states = results["twin_states"]

        measured_temperature = results["measured_temperature"]

        commanded_flows = results["commanded_flows"]
        actual_flows = results["actual_flows"]

        records.append({
            "time": time,
            "r_T": measured_temperature - twin_states[:, 1],
            "r_Tj": plant_states[:, 2] - twin_states[:, 2],
            "r_C": plant_states[:, 0] - twin_states[:, 0],
            "r_u": actual_flows - commanded_flows,
        })

    return records, fault_time


def calculate_statistics(records, fault_time):

    rows = []

    for signal_name in ["r_T", "r_Tj", "r_C", "r_u"]:

        healthy_values = np.concatenate([
            record[signal_name][record["time"] < fault_time]
            for record in records
        ])

        faulty_values = np.concatenate([
            record[signal_name][record["time"] >= fault_time]
            for record in records
        ])

        rows.append({
            "residual": signal_name,
            "healthy_mean": np.mean(healthy_values),
            "healthy_std": np.std(healthy_values),
            "faulty_mean": np.mean(faulty_values),
            "faulty_std": np.std(faulty_values),
            "faulty_max_absolute": np.max(
                np.abs(faulty_values)
            ),
        })

    return pd.DataFrame(rows)


def save_results(records, statistics, fault_time):

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        TABLES_DIR
        / "fault_isolation_actuator_signature.csv"
    )

    figure_path = (
        FIGURES_DIR
        / "fault_isolation_actuator_signature.png"
    )

    statistics.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 12),
        sharex=True,
    )

    signal_names = ["r_T", "r_Tj", "r_C", "r_u"]

    record = records[0]

    for axis, signal_name in zip(axes, signal_names):

        axis.plot(
            record["time"],
            record[signal_name],
        )

        axis.axvline(
            fault_time,
            linestyle="--",
            label="Actuator fault injection",
        )

        axis.axhline(0, linestyle=":")

        axis.set_ylabel(signal_name)
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Time (min)")

    fig.suptitle(
        "Actuator Fault - Residual Signature"
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

    records, fault_time = run_actuator_fault_experiment()

    statistics = calculate_statistics(
        records,
        fault_time,
    )

    print("\n========== ACTUATOR FAULT SIGNATURE ==========")

    print("Fault injection time:", fault_time, "min")
    print("Stuck actuator flow:", 0.50, "m^3/min")

    print("\n========== RESIDUAL STATISTICS ==========")

    print(
        statistics.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    save_results(
        records,
        statistics,
        fault_time,
    )


if __name__ == "__main__":
    main()
