from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from python.plant_model.cstr_model import load_parameters
from python.experiments.run_sensor_fault_detection import (
    simulate_sensor_fault,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plant_parameters"
    / "cstr_nominal.json"
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def run_sensitivity_analysis():

    params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    sensor_biases = [
        0.25,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
    ]

    number_of_runs = 30

    fault_time = 30.0
    simulation_time = 80.0
    sample_time = 0.05

    detection_threshold = 0.75
    persistence_samples = 5

    noise_std = 0.05

    all_results = []

    print("\n========== SENSOR FAULT SENSITIVITY ANALYSIS ==========")

    for bias in sensor_biases:

        detection_delays = []
        false_alarm_count = 0
        detected_count = 0

        for run_index in range(number_of_runs):

            random_seed = 1000 + run_index

            results = simulate_sensor_fault(
                params=params,
                initial_state=initial_state,
                setpoint=356.0,
                kp=0.03,
                ki=0.003,
                simulation_time=simulation_time,
                sample_time=sample_time,
                fault_time=fault_time,
                sensor_bias=bias,
                noise_std=noise_std,
                detection_threshold=detection_threshold,
                persistence_samples=persistence_samples,
                random_seed=random_seed,
            )

            time = results["time"]

            detection_time = results["detection_time"]

            pre_fault_mask = time < fault_time

            false_alarm = np.any(
                results["alarm"][pre_fault_mask]
            )

            if false_alarm:
                false_alarm_count += 1

            if detection_time is not None:
                detected_count += 1

                delay = detection_time - fault_time

                detection_delays.append(delay)

        detection_rate = (
            detected_count / number_of_runs
        ) * 100.0

        false_alarm_rate = (
            false_alarm_count / number_of_runs
        ) * 100.0

        if detection_delays:

            mean_delay = np.mean(detection_delays)

            std_delay = np.std(
                detection_delays,
                ddof=1
            ) if len(detection_delays) > 1 else 0.0

        else:

            mean_delay = np.nan
            std_delay = np.nan

        all_results.append({
            "sensor_bias_K": bias,
            "detection_rate_percent": detection_rate,
            "false_alarm_rate_percent": false_alarm_rate,
            "mean_detection_delay_min": mean_delay,
            "std_detection_delay_min": std_delay,
            "detected_runs": detected_count,
            "total_runs": number_of_runs,
        })

        print(
            f"\nBias: {bias:.2f} K | "
            f"Detection: {detected_count}/{number_of_runs} | "
            f"Detection Rate: {detection_rate:.1f}% | "
            f"False Alarm Rate: {false_alarm_rate:.1f}% | "
            f"Mean Delay: {mean_delay:.4f} min"
        )

    return pd.DataFrame(all_results)


def save_results(data):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        TABLES_DIR
        / "cstr_sensor_fault_sensitivity.csv"
    )

    figure_path = (
        FIGURES_DIR
        / "cstr_sensor_fault_sensitivity.png"
    )

    data.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 13),
        sharex=True,
    )

    biases = data["sensor_bias_K"]

    axes[0].plot(
        biases,
        data["detection_rate_percent"],
        marker="o",
    )

    axes[0].set_ylabel("Detection Rate (%)")
    axes[0].set_ylim(-5, 105)
    axes[0].set_title("Sensor Fault Detection Rate")
    axes[0].grid(True)

    axes[1].plot(
        biases,
        data["false_alarm_rate_percent"],
        marker="s",
    )

    axes[1].set_ylabel("False Alarm Rate (%)")
    axes[1].set_ylim(-5, 105)
    axes[1].set_title("False Alarm Rate")
    axes[1].grid(True)

    axes[2].errorbar(
        biases,
        data["mean_detection_delay_min"],
        yerr=data["std_detection_delay_min"],
        marker="o",
        capsize=4,
    )

    axes[2].set_xlabel("Injected Sensor Bias (K)")
    axes[2].set_ylabel("Detection Delay (min)")
    axes[2].set_title("Detection Delay")
    axes[2].grid(True)

    fig.suptitle(
        "CSTR - Sensor Fault Sensitivity Analysis"
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

    results = run_sensitivity_analysis()

    print("\n========== FINAL SENSITIVITY TABLE ==========")

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    save_results(results)


if __name__ == "__main__":
    main()