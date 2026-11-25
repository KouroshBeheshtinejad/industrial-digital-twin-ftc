import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

from python.plant_model.cstr_model import load_parameters
from python.experiments.integrated_simulation import (
    calculate_metrics,
    default_scenarios,
    run_integrated_simulation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "plant_parameters" / "cstr_nominal.json"


def save_csv(output_directory: Path, result) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "trajectory.csv"
    columns = [
        "time",
        "true_concentration",
        "true_temperature",
        "true_jacket_temperature",
        "estimated_concentration",
        "estimated_temperature",
        "estimated_jacket_temperature",
        "measured_temperature",
        "sensor_bias",
        "controller_command",
        "actual_coolant_flow",
        "temperature_residual",
        "actuator_residual",
        "sensor_alarm",
        "actuator_alarm",
        "diagnosed_fault",
        "ftc_mode",
        "control_temperature",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, time in enumerate(result.time):
            writer.writerow({
                "time": f"{time:.12g}",
                "true_concentration": f"{result.true_states[index, 0]:.12g}",
                "true_temperature": f"{result.true_states[index, 1]:.12g}",
                "true_jacket_temperature": f"{result.true_states[index, 2]:.12g}",
                "estimated_concentration": f"{result.estimated_states[index, 0]:.12g}",
                "estimated_temperature": f"{result.estimated_states[index, 1]:.12g}",
                "estimated_jacket_temperature": f"{result.estimated_states[index, 2]:.12g}",
                "measured_temperature": f"{result.measured_temperature[index]:.12g}",
                "sensor_bias": f"{result.sensor_bias[index]:.12g}",
                "controller_command": f"{result.controller_command[index]:.12g}",
                "actual_coolant_flow": f"{result.actual_coolant_flow[index]:.12g}",
                "temperature_residual": f"{result.temperature_residual[index]:.12g}",
                "actuator_residual": f"{result.actuator_residual[index]:.12g}",
                "sensor_alarm": bool(result.sensor_alarm[index]),
                "actuator_alarm": bool(result.actuator_alarm[index]),
                "diagnosed_fault": result.diagnosed_fault[index],
                "ftc_mode": result.ftc_mode[index],
                "control_temperature": f"{result.control_temperature[index]:.12g}",
            })
    return csv_path


def save_plots(output_directory: Path, result) -> Path:
    time = result.time
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(time, result.true_states[:, 1], label="Actual")
    axes[0].plot(time, result.estimated_states[:, 1], label="Estimated")
    axes[0].axhline(result.scenario.setpoint, color="black", linestyle="--", label="Setpoint")
    axes[0].set_ylabel("Reactor temperature (K)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time, result.controller_command, label="Controller command")
    axes[1].plot(time, result.actual_coolant_flow, label="Actual coolant flow")
    axes[1].set_ylabel("Coolant flow (m^3/min)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time, result.temperature_residual, label="Temperature residual")
    axes[2].axhline(result.scenario.temperature_threshold, color="orange", linestyle="--")
    axes[2].axhline(-result.scenario.temperature_threshold, color="orange", linestyle="--")
    axes[2].plot(time, result.actuator_residual, label="Actuator residual")
    axes[2].axhline(result.scenario.actuator_threshold, color="red", linestyle=":")
    axes[2].axhline(-result.scenario.actuator_threshold, color="red", linestyle=":")
    axes[2].set_ylabel("Residual")
    axes[2].legend()
    axes[2].grid(True)

    mode_names = [
        "normal",
        "sensor_reconfiguration",
        "actuator_reconfiguration",
        "safe_mode",
    ]
    mode_values = [
        mode_names.index(mode) if mode in mode_names else 3
        for mode in result.ftc_mode
    ]
    axes[3].step(time, mode_values, where="post")
    axes[3].set_yticks(range(len(mode_names)))
    axes[3].set_yticklabels(mode_names)
    axes[3].set_ylabel("FTC mode")
    axes[3].set_xlabel("Time (min)")
    axes[3].grid(True)

    fig.suptitle(f"Integrated CSTR Simulation: {result.scenario.name}")
    fig.tight_layout()
    plot_path = output_directory / "integrated_response.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run integrated CSTR Digital Twin and FTC simulations."
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        choices=[
            "healthy",
            "sensor_bias",
            "actuator_stuck",
            "simultaneous_faults",
            "all",
        ],
        default=["all"],
    )
    args = parser.parse_args()

    params = load_parameters(CONFIG_PATH)
    available = default_scenarios()
    if "all" in args.scenarios:
        selected = available
    else:
        selected = [
            scenario for scenario in available
            if scenario.name in args.scenarios
        ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_directory = (
        PROJECT_ROOT
        / "results"
        / "integrated_simulation"
        / timestamp
    )
    base_directory.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    generated = []
    for scenario in selected:
        result = run_integrated_simulation(params, scenario)
        scenario_directory = base_directory / scenario.name
        csv_path = save_csv(scenario_directory, result)
        plot_path = save_plots(scenario_directory, result)
        metrics_rows.append(calculate_metrics(result, params))
        generated.append((csv_path, plot_path))

    metrics_table = pd.DataFrame(metrics_rows)
    metrics_path = base_directory / "metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)

    print(metrics_table.to_string(index=False))
    print(f"\nMetrics: {metrics_path}")
    for csv_path, plot_path in generated:
        print(f"CSV: {csv_path}")
        print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
