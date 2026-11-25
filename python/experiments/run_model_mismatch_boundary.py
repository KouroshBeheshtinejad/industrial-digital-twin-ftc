from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from python.plant_model.cstr_model import load_parameters

from python.experiments.run_model_mismatch_analysis import (
    simulate_model_mismatch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plant_parameters"
    / "cstr_nominal.json"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "figures"


def maximum_consecutive_exceedances(values, threshold):
    exceedances = np.abs(values) > threshold

    current_run = 0
    maximum_run = 0

    for exceeded in exceedances:
        if exceeded:
            current_run += 1
            maximum_run = max(maximum_run, current_run)
        else:
            current_run = 0

    return maximum_run


def main():

    params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    mismatch_levels = np.arange(8, 16, 1)

    seeds = range(20)

    threshold = 0.75
    persistence_samples = 5

    summary = []

    print("\n========== MODEL MISMATCH BOUNDARY ANALYSIS ==========")

    print(
        "\nMismatch | False Alarms | "
        "Mean Max Residual | Mean Max Run"
    )

    for mismatch_percentage in mismatch_levels:

        maximum_residuals = []
        maximum_runs = []

        false_alarms = 0

        for seed in seeds:

            results = simulate_model_mismatch(
                nominal_params=params,
                initial_state=initial_state,
                setpoint=356.0,
                kp=0.03,
                ki=0.003,
                simulation_time=80.0,
                sample_time=0.05,
                plant_ua_multiplier=(
                    1.0 + mismatch_percentage / 100.0
                ),
                noise_std=0.05,
                detection_threshold=threshold,
                persistence_samples=persistence_samples,
                random_seed=seed,
            )

            residuals = results["residuals"]

            max_residual = np.max(np.abs(residuals))

            max_run = maximum_consecutive_exceedances(
                residuals,
                threshold,
            )

            maximum_residuals.append(max_residual)
            maximum_runs.append(max_run)

            if results["detection_time"] is not None:
                false_alarms += 1

        false_alarm_rate = false_alarms / len(seeds)

        mean_max_residual = np.mean(maximum_residuals)
        std_max_residual = np.std(maximum_residuals)

        mean_max_run = np.mean(maximum_runs)

        summary.append({
            "mismatch": mismatch_percentage,
            "false_alarm_rate": false_alarm_rate,
            "mean_max_residual": mean_max_residual,
            "std_max_residual": std_max_residual,
            "mean_max_run": mean_max_run,
        })

        print(
            f"{mismatch_percentage:7.2f}% | "
            f"{false_alarms:2d}/20 | "
            f"{mean_max_residual:17.6f} K | "
            f"{mean_max_run:14.2f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mismatch_values = [
        item["mismatch"] for item in summary
    ]

    false_alarm_rates = [
        item["false_alarm_rate"] * 100
        for item in summary
    ]

    mean_residuals = [
        item["mean_max_residual"]
        for item in summary
    ]

    residual_stds = [
        item["std_max_residual"]
        for item in summary
    ]

    mean_runs = [
        item["mean_max_run"]
        for item in summary
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 12),
    )

    axes[0].plot(
        mismatch_values,
        false_alarm_rates,
        marker="o",
    )
    axes[0].set_ylabel("False alarm rate (%)")
    axes[0].set_xlabel("Plant UA mismatch (%)")
    axes[0].set_ylim(-5, 105)
    axes[0].grid(True)

    axes[1].errorbar(
        mismatch_values,
        mean_residuals,
        yerr=residual_stds,
        marker="o",
        capsize=4,
    )
    axes[1].axhline(
        threshold,
        linestyle="--",
        label="Detection threshold",
    )
    axes[1].set_ylabel("Mean maximum absolute residual (K)")
    axes[1].set_xlabel("Plant UA mismatch (%)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        mismatch_values,
        mean_runs,
        marker="o",
    )
    axes[2].axhline(
        persistence_samples,
        linestyle="--",
        label="Required consecutive samples",
    )
    axes[2].set_ylabel("Mean maximum consecutive exceedances")
    axes[2].set_xlabel("Plant UA mismatch (%)")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle("CSTR - Model Mismatch Detection Boundary")
    fig.tight_layout()

    output_path = (
        RESULTS_DIR / "cstr_model_mismatch_boundary.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.show()

    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
