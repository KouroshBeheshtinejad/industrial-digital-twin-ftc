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


def main():

    params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    mismatch_levels = [
        0.0,
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
    ]

    seeds = range(10)

    threshold = 0.5

    summary = []

    print("\n========== MODEL MISMATCH SENSITIVITY ==========")

    print(
        "\nMismatch | False Alarms | "
        "Max Residual Mean | Max Residual Std"
    )

    for mismatch in mismatch_levels:

        maximum_residuals = []
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
                plant_ua_multiplier=1.0 + mismatch,
                noise_std=0.05,
                detection_threshold=threshold,
                persistence_samples=5,
                random_seed=seed,
            )

            maximum_residual = np.max(
                np.abs(results["residuals"])
            )

            maximum_residuals.append(maximum_residual)

            if results["detection_time"] is not None:
                false_alarms += 1

        maximum_residuals = np.asarray(maximum_residuals)

        mean_residual = np.mean(maximum_residuals)
        std_residual = np.std(maximum_residuals)

        false_alarm_rate = false_alarms / len(seeds)

        summary.append({
            "mismatch": mismatch * 100,
            "false_alarm_rate": false_alarm_rate,
            "mean_max_residual": mean_residual,
            "std_max_residual": std_residual,
        })

        print(
            f"{mismatch * 100:7.2f}% | "
            f"{false_alarms:3d}/10 | "
            f"{mean_residual:16.6f} K | "
            f"{std_residual:15.6f} K"
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

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 9),
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
    axes[1].set_xlabel("Plant UA mismatch (%)")
    axes[1].set_ylabel("Maximum absolute residual (K)")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle("Fault Detector Sensitivity to Model Mismatch")
    fig.tight_layout()

    output_path = (
        RESULTS_DIR / "cstr_model_mismatch_sensitivity.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.show()

    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
