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

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def run_baseline():

    params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    number_of_runs = 30

    simulation_time = 40.0
    sample_time = 0.05

    residual_records = []

    for run_index in range(number_of_runs):

        results = simulate_actuator_fault(
            params=params,
            initial_state=initial_state,
            setpoint=356.0,
            kp=0.03,
            ki=0.003,
            simulation_time=simulation_time,
            sample_time=sample_time,
            fault_time=simulation_time + 1.0,
            stuck_flow=None,
            noise_std=0.05,
            detection_threshold=0.75,
            persistence_samples=5,
            random_seed=3000 + run_index,
        )

        plant_states = results["plant_states"]
        twin_states = results["twin_states"]

        measured_temperature = results["measured_temperature"]

        commanded_flows = results["commanded_flows"]
        actual_flows = results["actual_flows"]

        residual_records.append({
            "r_T": measured_temperature - twin_states[:, 1],
            "r_Tj": plant_states[:, 2] - twin_states[:, 2],
            "r_C": plant_states[:, 0] - twin_states[:, 0],
            "r_u": actual_flows - commanded_flows,
        })

    return residual_records


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
                np.abs(values), 99
            ),
        })

    return pd.DataFrame(rows)


def save_results(records, statistics):

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        TABLES_DIR
        / "fault_isolation_healthy_baseline.csv"
    )

    figure_path = (
        FIGURES_DIR
        / "fault_isolation_healthy_baseline.png"
    )

    statistics.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(
        4, 1,
        figsize=(11, 12),
    )

    signal_names = ["r_T", "r_Tj", "r_C", "r_u"]

    for axis, signal_name in zip(axes, signal_names):

        values = records[0][signal_name]

        time = np.arange(len(values)) * 0.05

        axis.plot(time, values)

        axis.axhline(0, linestyle="--")

        axis.set_ylabel(signal_name)
        axis.grid(True)

    axes[-1].set_xlabel("Time (min)")

    fig.suptitle(
        "Healthy System - Residual Baseline"
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

    records = run_baseline()

    statistics = calculate_statistics(records)

    print("\n========== HEALTHY RESIDUAL STATISTICS ==========")

    print(
        statistics.to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    save_results(records, statistics)


if __name__ == "__main__":
    main()
