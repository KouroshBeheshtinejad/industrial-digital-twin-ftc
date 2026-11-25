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
from python.estimation.state_estimator import ModelBasedStateEstimator


ROOT = Path(__file__).resolve().parents[2]

SIMULATION_TIME = 80.0
SAMPLE_TIME = 0.05
FAULT_TIME = 30.0
SENSOR_BIAS = 2.0
DETECTION_THRESHOLD = 0.75
PERSISTENCE_SAMPLES = 5
SETPOINT = 356.0

INITIAL_STATE = np.array([
    64.50615287,
    355.28373576,
    305.90638203
])


def simulate_scenario(params, mode):
    number_of_steps = int(SIMULATION_TIME / SAMPLE_TIME)

    time_values = np.arange(
        number_of_steps + 1
    ) * SAMPLE_TIME

    states = np.zeros((number_of_steps + 1, 3))
    measured_temperature = np.zeros(number_of_steps + 1)
    estimated_temperature = np.zeros(number_of_steps + 1)
    coolant_flow = np.zeros(number_of_steps + 1)
    residuals = np.zeros(number_of_steps + 1)
    fault_detected = np.zeros(number_of_steps + 1, dtype=bool)

    states[0] = INITIAL_STATE

    controller = PIController(setpoint=SETPOINT)

    estimator = ModelBasedStateEstimator(
        params=params,
        initial_state=INITIAL_STATE
    )

    consecutive_alarm_samples = 0
    detected = False

    for index in range(number_of_steps):
        current_time = time_values[index]
        true_temperature = states[index, 1]

        sensor_fault_active = current_time >= FAULT_TIME

        bias = (
            SENSOR_BIAS
            if mode != "healthy" and sensor_fault_active
            else 0.0
        )

        measurement = true_temperature + bias

        # Fault detection using the model-based temperature estimate.
        residual = measurement - estimator.estimated_temperature

        if mode == "ftc" and sensor_fault_active:
            if abs(residual) > DETECTION_THRESHOLD:
                consecutive_alarm_samples += 1
            else:
                consecutive_alarm_samples = 0

            if consecutive_alarm_samples >= PERSISTENCE_SAMPLES:
                detected = True

        # Select the temperature used by the controller.
        if mode == "healthy":
            control_temperature = measurement

        elif mode == "faulty":
            control_temperature = measurement

        elif mode == "ftc":
            if detected:
                control_temperature = estimator.estimated_temperature
            else:
                control_temperature = measurement

        else:
            raise ValueError(f"Unknown simulation mode: {mode}")

        # Calculate the coolant command.
        command = controller.compute(
            temperature=control_temperature,
            sample_time=SAMPLE_TIME
        )

        coolant_flow[index] = command
        measured_temperature[index] = measurement
        estimated_temperature[index] = estimator.estimated_temperature
        residuals[index] = residual
        fault_detected[index] = detected

        # In this experiment, the actuator is assumed healthy.
        actual_coolant_flow = command

        # Advance the physical plant.
        solution = solve_ivp(
            fun=lambda time, state: cstr_dynamics(
                time,
                state,
                actual_coolant_flow,
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

        # Advance the estimator using the actual actuator input.
        estimator.predict(
            actual_coolant_flow=actual_coolant_flow,
            sample_time=SAMPLE_TIME
        )

    final_bias = (
        SENSOR_BIAS
        if mode != "healthy" and SIMULATION_TIME >= FAULT_TIME
        else 0.0
    )

    measured_temperature[-1] = states[-1, 1] + final_bias

    estimated_temperature[-1] = estimator.estimated_temperature
    coolant_flow[-1] = coolant_flow[-2]
    residuals[-1] = (
        measured_temperature[-1] - estimated_temperature[-1]
    )
    fault_detected[-1] = detected

    return {
        "time": time_values,
        "states": states,
        "measured_temperature": measured_temperature,
        "estimated_temperature": estimated_temperature,
        "coolant_flow": coolant_flow,
        "residual": residuals,
        "fault_detected": fault_detected
    }


def calculate_metrics(result, params):
    time = result["time"]
    temperature = result["states"][:, 1]
    error = SETPOINT - temperature
    flow = result["coolant_flow"]

    return {
        "final_temperature_K": temperature[-1],
        "final_error_K": error[-1],
        "maximum_temperature_K": np.max(temperature),
        "minimum_temperature_K": np.min(temperature),
        "IAE": np.trapezoid(np.abs(error), time),
        "ISE": np.trapezoid(error ** 2, time),
        "maximum_coolant_flow": np.max(flow),
        "minimum_coolant_flow": np.min(flow),
        "actuator_within_limits": bool(
            np.all(flow >= params.minimum_coolant_flow - 1e-9)
            and np.all(flow <= params.maximum_coolant_flow + 1e-9)
        )
    }


def main():
    params = load_parameters(
        ROOT / "configs" / "plant_parameters" / "cstr_nominal.json"
    )

    scenarios = {
        "Healthy PI": "healthy",
        "Faulty PI": "faulty",
        "Sensor FTC": "ftc"
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

    print("\n========== SENSOR FAULT-TOLERANT CONTROL ==========")
    print(f"Simulation duration: {SIMULATION_TIME:.2f} min")
    print(f"Sample time: {SAMPLE_TIME:.4f} min")
    print(f"Sensor fault time: {FAULT_TIME:.2f} min")
    print(f"Sensor bias: {SENSOR_BIAS:.2f} K")

    print("\nPerformance comparison:")
    print(metrics_df.to_string())

    ftc_result = results["Sensor FTC"]
    detection_indices = np.flatnonzero(ftc_result["fault_detected"])

    if len(detection_indices):
        detection_time = ftc_result["time"][detection_indices[0]]
        print(f"\nFault detection time: {detection_time:.3f} min")
        print(
            f"Detection delay: "
            f"{detection_time - FAULT_TIME:.3f} min"
        )
    else:
        print("\nFault was not detected.")

    output_tables = ROOT / "results" / "tables"
    output_figures = ROOT / "results" / "figures"

    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(
        output_tables / "sensor_ftc_metrics.csv"
    )

    trajectory_rows = []

    for name, result in results.items():
        for index, time in enumerate(result["time"]):
            trajectory_rows.append({
                "scenario": name,
                "time_min": time,
                "true_temperature_K": result["states"][index, 1],
                "measured_temperature_K": result["measured_temperature"][index],
                "estimated_temperature_K": result["estimated_temperature"][index],
                "coolant_flow": result["coolant_flow"][index],
                "residual_K": result["residual"][index],
                "fault_detected": result["fault_detected"][index]
            })

    pd.DataFrame(trajectory_rows).to_csv(
        output_tables / "sensor_ftc_trajectory.csv",
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
        label="Sensor fault"
    )
    axes[0].set_ylabel("Reactor temperature (K)")
    axes[0].set_title("Sensor Fault-Tolerant Control Comparison")
    axes[0].legend()
    axes[0].grid(True)

    ftc_result = results["Sensor FTC"]

    axes[1].plot(
        ftc_result["time"],
        ftc_result["measured_temperature"],
        label="Measured temperature"
    )
    axes[1].plot(
        ftc_result["time"],
        ftc_result["estimated_temperature"],
        label="Estimated temperature"
    )
    axes[1].plot(
        ftc_result["time"],
        ftc_result["states"][:, 1],
        linestyle="--",
        label="True temperature"
    )
    axes[1].axvline(FAULT_TIME, linestyle=":", color="red")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_title("Sensor Measurement and State Estimation")
    axes[1].legend()
    axes[1].grid(True)

    for name, result in results.items():
        axes[2].plot(
            result["time"],
            result["coolant_flow"],
            label=name
        )

    axes[2].set_xlabel("Time (min)")
    axes[2].set_ylabel("Coolant flow (m³/min)")
    axes[2].set_title("Control Input Comparison")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    figure_path = output_figures / "sensor_fault_tolerant_control.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print("\nOutput files saved successfully.")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
