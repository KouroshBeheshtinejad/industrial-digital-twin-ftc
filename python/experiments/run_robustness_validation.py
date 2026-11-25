import argparse
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python.plant_model.cstr_model import load_parameters
from python.experiments.integrated_simulation import (
    IntegratedScenario,
    calculate_metrics,
    run_integrated_simulation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "plant_parameters" / "cstr_nominal.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "robustness_validation"


def residual_statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "residual_mean": float(np.mean(values)),
        "residual_std": float(np.std(values)),
        "residual_min": float(np.min(values)),
        "residual_max": float(np.max(values)),
    }


def alarm_counts(result) -> tuple[int, int]:
    false_mask = result.time < result.scenario.fault_time
    false_sensor = int(np.count_nonzero(result.sensor_alarm & false_mask))
    false_actuator = int(np.count_nonzero(result.actuator_alarm & false_mask))
    return false_sensor, false_actuator


def append_aggregate(
    rows: list,
    experiment: str,
    condition: str,
    raw_rows: list,
    runs: int,
    seeds: list[int],
) -> None:
    frame = pd.DataFrame(raw_rows)
    condition_frame = frame[frame["condition"] == condition]
    numeric_columns = [
        column
        for column in condition_frame.columns
        if column not in ("experiment", "condition", "detected", "ftc_mode")
        and pd.api.types.is_numeric_dtype(condition_frame[column])
    ]

    summary = {
        "experiment": experiment,
        "condition": condition,
        "number_of_runs": runs,
        "random_seeds": ",".join(str(seed) for seed in seeds),
    }
    for column in numeric_columns:
        values = condition_frame[column].dropna().to_numpy(dtype=float)
        if len(values):
            summary[f"{column}_mean"] = float(np.mean(values))
            summary[f"{column}_std"] = float(np.std(values))
            summary[f"{column}_min"] = float(np.min(values))
            summary[f"{column}_max"] = float(np.max(values))

    if "detected" in condition_frame:
        summary["detection_rate"] = float(condition_frame["detected"].mean())
    if "false_sensor_alarm" in condition_frame:
        summary["false_sensor_alarm_rate"] = float(
            condition_frame["false_sensor_alarm"].mean()
        )
    if "false_actuator_alarm" in condition_frame:
        summary["false_actuator_alarm_rate"] = float(
            condition_frame["false_actuator_alarm"].mean()
        )
    rows.append(summary)


def run_validation(
    nominal_params,
    runs: int,
    simulation_time: float,
    sample_time: float,
):
    raw_rows = []
    aggregate_rows = []
    seed_base = 6000

    # Healthy measurement-noise robustness.
    noise_levels = [0.02, 0.05, 0.10, 0.20]
    for noise_index, noise_std in enumerate(no_noise := noise_levels):
        seeds = [seed_base + noise_index * runs + seed for seed in range(runs)]
        condition_rows = []
        for seed in seeds:
            scenario = IntegratedScenario(
                name="healthy",
                simulation_time=simulation_time,
                sample_time=sample_time,
                measurement_noise_std=noise_std,
                random_seed=seed,
            )
            result = run_integrated_simulation(nominal_params, scenario)
            metrics = calculate_metrics(result, nominal_params)
            false_sensor, false_actuator = alarm_counts(result)
            row = {
                "experiment": "healthy_noise",
                "condition": f"{noise_std:.2f} K",
                "seed": seed,
                "detected": bool(np.any(result.sensor_alarm) or np.any(result.actuator_alarm)),
                "false_sensor_alarm": bool(false_sensor > 0),
                "false_actuator_alarm": bool(false_actuator > 0),
                "false_sensor_alarm_count": false_sensor,
                "false_actuator_alarm_count": false_actuator,
                "IAE": metrics["IAE"],
                "ISE": metrics["ISE"],
                "maximum_temperature_deviation_K": float(
                    np.max(np.abs(result.true_states[:, 1] - scenario.setpoint))
                ),
                "final_temperature_error_K": metrics["final_temperature_error_K"],
            }
            row.update(residual_statistics(result.temperature_residual))
            raw_rows.append(row)
            condition_rows.append(row)
        append_aggregate(
            aggregate_rows,
            "healthy_noise",
            f"{noise_std:.2f} K",
            raw_rows,
            runs,
            seeds,
        )

    # Controlled UA mismatch: nominal twin, modified actual plant.
    ua_levels = [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15]
    for ua_index, ua_fraction in enumerate(ua_levels):
        seeds = [seed_base + 1000 + ua_index * runs + seed for seed in range(runs)]
        condition = f"{ua_fraction:+.0%}"
        for seed in seeds:
            actual_params = replace(
                nominal_params,
                heat_transfer_coefficient=(
                    nominal_params.heat_transfer_coefficient * (1.0 + ua_fraction)
                ),
            )
            scenario = IntegratedScenario(
                name="ua_mismatch",
                simulation_time=simulation_time,
                sample_time=sample_time,
                random_seed=seed,
            )
            result = run_integrated_simulation(
                nominal_params,
                scenario,
                actual_params=actual_params,
            )
            metrics = calculate_metrics(result, nominal_params)
            false_sensor, false_actuator = alarm_counts(result)
            sensor_detection = np.flatnonzero(result.sensor_alarm)
            actuator_detection = np.flatnonzero(result.actuator_alarm)
            sensor_time = (
                float(result.time[sensor_detection[0]])
                if len(sensor_detection) else float("nan")
            )
            actuator_time = (
                float(result.time[actuator_detection[0]])
                if len(actuator_detection) else float("nan")
            )
            row = {
                "experiment": "ua_mismatch",
                "condition": condition,
                "seed": seed,
                "detected": bool(np.any(result.sensor_alarm) or np.any(result.actuator_alarm)),
                "false_sensor_alarm": bool(false_sensor > 0),
                "false_actuator_alarm": bool(false_actuator > 0),
                "false_sensor_alarm_count": false_sensor,
                "false_actuator_alarm_count": false_actuator,
                "sensor_detection_time_min": sensor_time,
                "actuator_detection_time_min": actuator_time,
                "IAE": metrics["IAE"],
                "ISE": metrics["ISE"],
                "final_temperature_error_K": metrics["final_temperature_error_K"],
            }
            row.update(residual_statistics(result.temperature_residual))
            raw_rows.append(row)
        append_aggregate(
            aggregate_rows,
            "ua_mismatch",
            condition,
            raw_rows,
            runs,
            seeds,
        )

    # Sensor-bias sensitivity.
    sensor_biases = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00,
                     -0.25, -0.50, -0.75, -1.00, -1.50, -2.00]
    for bias_index, bias in enumerate(sensor_biases):
        seeds = [seed_base + 2000 + bias_index * runs + seed for seed in range(runs)]
        condition = f"{bias:+.2f} K"
        for seed in seeds:
            scenario = IntegratedScenario(
                name="sensor_bias",
                simulation_time=simulation_time,
                sample_time=sample_time,
                fault_time=30.0,
                sensor_fault_enabled=True,
                sensor_bias=bias,
                random_seed=seed,
            )
            result = run_integrated_simulation(nominal_params, scenario)
            metrics = calculate_metrics(result, nominal_params)
            false_sensor, false_actuator = alarm_counts(result)
            detection_indices = np.flatnonzero(result.sensor_alarm)
            detected = bool(len(detection_indices))
            detection_time = (
                float(result.time[detection_indices[0]])
                if detected else float("nan")
            )
            detection_delay = (
                detection_time - scenario.fault_time
                if detected else float("nan")
            )
            row = {
                "experiment": "sensor_bias",
                "condition": condition,
                "seed": seed,
                "detected": detected,
                "detection_time_min": detection_time,
                "detection_delay_min": detection_delay,
                "false_sensor_alarm": bool(false_sensor > 0),
                "false_actuator_alarm": bool(false_actuator > 0),
                "false_sensor_alarm_count": false_sensor,
                "false_actuator_alarm_count": false_actuator,
                "IAE": metrics["IAE"],
                "ISE": metrics["ISE"],
            }
            row.update(residual_statistics(result.temperature_residual))
            raw_rows.append(row)
        append_aggregate(
            aggregate_rows,
            "sensor_bias",
            condition,
            raw_rows,
            runs,
            seeds,
        )

    # Actuator stuck sensitivity.
    stuck_flows = [0.02, 0.30, 0.40, 0.45, 0.50]
    for stuck_index, stuck_flow in enumerate(stuck_flows):
        seeds = [seed_base + 3000 + stuck_index * runs + seed for seed in range(runs)]
        condition = f"{stuck_flow:.2f} m3/min"
        for seed in seeds:
            scenario = IntegratedScenario(
                name="actuator_stuck",
                simulation_time=simulation_time,
                sample_time=sample_time,
                fault_time=30.0,
                actuator_stuck_enabled=True,
                stuck_flow=stuck_flow,
                random_seed=seed,
            )
            result = run_integrated_simulation(nominal_params, scenario)
            metrics = calculate_metrics(result, nominal_params)
            false_sensor, false_actuator = alarm_counts(result)
            detection_indices = np.flatnonzero(result.actuator_alarm)
            detected = bool(len(detection_indices))
            detection_time = (
                float(result.time[detection_indices[0]])
                if detected else float("nan")
            )
            detection_delay = (
                detection_time - scenario.fault_time
                if detected else float("nan")
            )
            row = {
                "experiment": "actuator_stuck",
                "condition": condition,
                "seed": seed,
                "detected": detected,
                "detection_time_min": detection_time,
                "detection_delay_min": detection_delay,
                "false_sensor_alarm": bool(false_sensor > 0),
                "false_actuator_alarm": bool(false_actuator > 0),
                "false_sensor_alarm_count": false_sensor,
                "false_actuator_alarm_count": false_actuator,
                "IAE": metrics["IAE"],
                "ISE": metrics["ISE"],
                "final_temperature_error_K": metrics["final_temperature_error_K"],
                "maximum_coolant_flow": metrics["maximum_coolant_flow"],
                "minimum_coolant_flow": metrics["minimum_coolant_flow"],
                "safe_mode_active": bool("safe_mode" in result.ftc_mode),
            }
            row.update(residual_statistics(result.actuator_residual))
            raw_rows.append(row)
        append_aggregate(
            aggregate_rows,
            "actuator_stuck",
            condition,
            raw_rows,
            runs,
            seeds,
        )

    return pd.DataFrame(raw_rows), pd.DataFrame(aggregate_rows)


def save_summary_plots(aggregate: pd.DataFrame, output_directory: Path) -> list[Path]:
    plot_paths = []

    sensor = aggregate[aggregate["experiment"] == "sensor_bias"].copy()
    if not sensor.empty:
        sensor["bias"] = sensor["condition"].str.replace(" K", "", regex=False).astype(float)
        sensor = sensor.sort_values("bias")
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(
            sensor["bias"],
            sensor["detection_rate"],
            marker="o",
        )
        axis.set_xlabel("Sensor bias (K)")
        axis.set_ylabel("Detection rate")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True)
        fig.tight_layout()
        path = output_directory / "sensor_bias_detection_rate.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

    actuator = aggregate[aggregate["experiment"] == "actuator_stuck"].copy()
    if not actuator.empty:
        actuator["stuck_flow"] = (
            actuator["condition"].str.replace(" m3/min", "", regex=False).astype(float)
        )
        actuator = actuator.sort_values("stuck_flow")
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(
            actuator["stuck_flow"],
            actuator["detection_rate"],
            marker="o",
        )
        axis.set_xlabel("Stuck coolant flow (m3/min)")
        axis.set_ylabel("Detection rate")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True)
        fig.tight_layout()
        path = output_directory / "actuator_stuck_detection_rate.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

    ua = aggregate[aggregate["experiment"] == "ua_mismatch"].copy()
    if not ua.empty:
        ua["ua_fraction"] = ua["condition"].str.replace("%", "", regex=False).astype(float)
        ua = ua.sort_values("ua_fraction")
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(
            ua["ua_fraction"],
            ua["detection_rate"],
            marker="o",
            label="Any alarm",
        )
        axis.set_xlabel("UA mismatch (%)")
        axis.set_ylabel("Alarm/detection rate")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True)
        fig.tight_layout()
        path = output_directory / "ua_mismatch_alarm_rate.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

    return plot_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible CSTR robustness validation sweeps."
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--simulation-time", type=float, default=60.0)
    parser.add_argument("--sample-time", type=float, default=0.10)
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("runs must be at least one.")
    if args.simulation_time <= 0:
        raise ValueError("simulation-time must be positive.")
    if args.sample_time <= 0:
        raise ValueError("sample-time must be positive.")

    nominal_params = load_parameters(CONFIG_PATH)
    raw, aggregate = run_validation(
        nominal_params,
        args.runs,
        args.simulation_time,
        args.sample_time,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "raw_results.csv"
    aggregate_path = RESULTS_DIR / "aggregated_results.csv"
    raw.to_csv(raw_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    plot_paths = save_summary_plots(aggregate, RESULTS_DIR)

    print(aggregate.to_string(index=False))
    print(f"\nRaw CSV: {raw_path}")
    print(f"Aggregated CSV: {aggregate_path}")
    for plot_path in plot_paths:
        print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
