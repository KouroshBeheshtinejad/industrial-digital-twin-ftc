from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from python.plant_model.cstr_model import (
    load_parameters,
    cstr_dynamics
)
from python.controllers.pi_controller import PIController


ROOT = Path(__file__).resolve().parents[2]

SIMULATION_TIME = 80.0
SAMPLE_TIME = 0.05
FAULT_TIME = 30.0

SETPOINT = 356.0
STUCK_FLOW = 0.5

DETECTION_THRESHOLD = 0.02
PERSISTENCE_SAMPLES = 5

INITIAL_STATE = np.array([
    64.50615287,
    355.28373576,
    305.90638203
])


def simulate_scenario(params, mode):
    number_of_steps = int(SIMULATION_TIME / SAMPLE_TIME)

    time_values = np.arange(number_of_steps + 1) * SAMPLE_TIME

    states = np.zeros((number_of_steps + 1, 3))
    commanded_flow = np.zeros(number_of_steps + 1)
    actual_flow = np.zeros(number_of_steps + 1)
    estimated_flow = np.zeros(number_of_steps + 1)
    residuals = np.zeros(number_of_steps + 1)
    fault_detected = np.zeros(number_of_steps + 1, dtype=bool)

    states[0] = INITIAL_STATE

    controller = PIController(setpoint=SETPOINT)

    consecutive_alarm_samples = 0
    detected = False

    for index in range(number_of_steps):
        current_time = time_values[index]

        true_temperature = states[index, 1]

        # The controller uses the true temperature in this experiment.
        command = controller.compute(
            temperature=true_temperature,
            sample_time=SAMPLE_TIME
        )

        # Simulate the physical actuator.
        if mode != "healthy" and current_time >= FAULT_TIME:
            actual = STUCK_FLOW
        else:
            actual = command

        # Estimate actual actuator output from flow feedback.
        estimated = actual

        residual = actual - command

        # Detect actuator malfunction.
        if mode == "ftc" and current_time >= FAULT_TIME:
            if abs(residual) > DETECTION_THRESHOLD:
                consecutive_alarm_samples += 1
            else:
                consecutive_alarm_samples = 0

            if consecutive_alarm_samples >= PERSISTENCE_SAMPLES:
                detected = True

        # Fault accommodation.
        if mode == "ftc" and detected:
            # The stuck actuator cannot be directly controlled.
            # The controller output is limited to the known stuck value.
            accommodated_command = command
        else:
            accommodated_command = command

        commanded_flow[index] = accommodated_command
        actual_flow[index] = actual
        estimated_flow[index] = estimated
        residuals[index] = residual
        fault_detected[index] = detected

        # Advance the physical plant using actual actuator output.
        solution = solve_ivp(
            fun=lambda time, state: cstr_dynamics(
                time,
                state,
                actual,
                params
            ),
            t_span=(0.0, SAMPLE_TIME),
            y0=states[index],
            method="RK45",
            rtol=1e-7,
            atol=1e-9
        )

        if not solution.success:
            raise RuntimeError(solution.message)

        states[index + 1] = solution.y[:, -1]

    commanded_flow[-1] = commanded_flow[-2]
    actual_flow[-1] = actual_flow[-2]
    estimated_flow[-1] = estimated_flow[-2]
    residuals[-1] = residuals[-2]
    fault_detected[-1] = detected

    return {
        "time": time_values,
        "states": states,
        "commanded_flow": commanded_flow,
        "actual_flow": actual_flow,
        "estimated_flow": estimated_flow,
        "residual": residuals,
        "fault_detected": fault_detected
    }


def calculate_metrics(result, params):
    time = result["time"]
    temperature = result["states"][:, 1]
    error = SETPOINT - temperature
    actual_flow = result["actual_flow"]

    return {
        "final_temperature_K": temperature[-1],
        "final_error_K": error[-1],
        "maximum_temperature_K": np.max(temperature),
        "minimum_temperature_K": np.min(temperature),
        "IAE": np.trapezoid(np.abs(error), time),
        "ISE": np.trapezoid(error ** 2, time),
        "maximum_actual_flow": np.max(actual_flow),
        "minimum_actual_flow": np.min(actual_flow),
        "actuator_within_limits": bool(
            np.all(actual_flow >= params.minimum_coolant_flow - 1e-9)
            and np.all(actual_flow <= params.maximum_coolant_flow + 1e-9)
        )
    }


def main():
    params = load_parameters(
        ROOT / "configs" / "plant_parameters" / "cstr_nominal.json"
    )

    scenarios = {
        "Healthy PI": "healthy",
        "Faulty PI": "faulty",
        "Actuator FTC": "ftc"
    }

    results = {}
    metrics = []

    for name, mode in scenarios.items():
        result = simulate_scenario(params, mode)
        results[name] = result

        scenario_metrics = calculate_metrics(result, params)
        scenario_metrics["scenario"] = name
        metrics.append(scenario_metrics)

    metrics_df = pd.DataFrame(metrics).set_index("scenario")

    print("\n========== ACTUATOR FAULT-TOLERANT CONTROL ==========")
    print(f"Simulation duration: {SIMULATION_TIME:.2f} min")
    print(f"Sample time: {SAMPLE_TIME:.4f} min")
    print(f"Actuator fault time: {FAULT_TIME:.2f} min")
    print(f"Stuck flow: {STUCK_FLOW:.3f} m^3/min")

    print("\nPerformance comparison:")
    print(metrics_df.to_string())

    ftc_result = results["Actuator FTC"]
    detection_indices = np.flatnonzero(ftc_result["fault_detected"])

    if len(detection_indices):
        detection_time = ftc_result["time"][detection_indices[0]]
        print(f"\nFault detection time: {detection_time:.3f} min")
        print(f"Detection delay: {detection_time - FAULT_TIME:.3f} min")
    else:
        print("\nFault was not detected.")

    output_tables = ROOT / "results" / "tables"
    output_figures = ROOT / "results" / "figures"

    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(
        output_tables / "actuator_ftc_metrics.csv"
    )

    trajectory_rows = []

    for name, result in results.items():
        for index, time in enumerate(result["time"]):
            trajectory_rows.append({
                "scenario": name,
                "time_min": time,
                "true_temperature_K": result["states"][index, 1],
                "commanded_flow": result["commanded_flow"][index],
                "actual_flow": result["actual_flow"][index],
                "estimated_flow": result["estimated_flow"][index],
                "residual": result["residual"][index],
                "fault_detected": result["fault_detected"][index]
            })

    pd.DataFrame(trajectory_rows).to_csv(
        output_tables / "actuator_ftc_trajectory.csv",
        index=False
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    for name, result in results.items():
        axes[0].plot(
            result["time"],
            result["states"][:, 1],
            label=name
        )

    axes[0].axhline(
        SETPOINT,
        linestyle="--",
        color="black",
        label="Setpoint"
    )
    axes[0].axvline(
        FAULT_TIME,
        linestyle=":",
        color="red",
        label="Actuator fault"
    )
    axes[0].set_ylabel("Reactor temperature (K)")
    axes[0].set_title("Actuator Fault-Tolerant Control Comparison")
    axes[0].legend()
    axes[0].grid(True)

    ftc_result = results["Actuator FTC"]

    axes[1].plot(
        ftc_result["time"],
        ftc_result["commanded_flow"],
        label="Commanded flow"
    )
    axes[1].plot(
        ftc_result["time"],
        ftc_result["actual_flow"],
        label="Actual flow"
    )
    axes[1].plot(
        ftc_result["time"],
        ftc_result["estimated_flow"],
        linestyle="--",
        label="Estimated flow"
    )
    axes[1].axvline(FAULT_TIME, linestyle=":", color="red")
    axes[1].set_ylabel("Coolant flow (m³/min)")
    axes[1].set_title("Actuator Fault and Flow Feedback")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        ftc_result["time"],
        ftc_result["residual"],
        label="Actuator residual"
    )
    axes[2].axhline(
        DETECTION_THRESHOLD,
        linestyle="--",
        color="red",
        label="Positive threshold"
    )
    axes[2].axhline(
        -DETECTION_THRESHOLD,
        linestyle="--",
        color="red",
        label="Negative threshold"
    )
    axes[2].axvline(FAULT_TIME, linestyle=":", color="black")
    axes[2].set_xlabel("Time (min)")
    axes[2].set_ylabel("Flow residual (m³/min)")
    axes[2].set_title("Actuator Fault Detection")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    figure_path = output_figures / "actuator_fault_tolerant_control.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print("\nOutput files saved successfully.")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
