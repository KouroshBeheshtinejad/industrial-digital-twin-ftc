from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from python.plant_model.cstr_model import load_parameters
from python.experiments.run_sensor_fault_detection import (
    integrate_process,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plant_parameters"
    / "cstr_nominal.json"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "figures"


def simulate_actuator_fault(
    params,
    initial_state,
    setpoint=356.0,
    kp=0.03,
    ki=0.003,
    simulation_time=80.0,
    sample_time=0.05,
    fault_time=30.0,
    stuck_flow=None,
    noise_std=0.05,
    detection_threshold=0.75,
    persistence_samples=5,
    random_seed=42,
):

    if stuck_flow is None:
        stuck_flow = params.maximum_coolant_flow

    number_of_steps = int(
        simulation_time / sample_time
    ) + 1

    time = np.linspace(
        0.0,
        simulation_time,
        number_of_steps,
    )

    plant_states = np.zeros((number_of_steps, 3))
    twin_states = np.zeros((number_of_steps, 3))

    measured_temperature = np.zeros(number_of_steps)

    commanded_flows = np.zeros(number_of_steps)
    actual_flows = np.zeros(number_of_steps)

    residuals = np.zeros(number_of_steps)

    alarm = np.zeros(number_of_steps, dtype=bool)
    fault_active = np.zeros(number_of_steps, dtype=bool)

    plant_states[0] = initial_state
    twin_states[0] = initial_state

    rng = np.random.default_rng(random_seed)

    integral_error = 0.0

    coolant_bias = params.nominal_coolant_flow

    consecutive_exceedances = 0
    detection_time = None

    for i in range(number_of_steps - 1):

        current_time = time[i]

        actual_temperature = plant_states[i, 1]

        measurement = (
            actual_temperature
            + rng.normal(0.0, noise_std)
        )

        # PI controller uses the healthy sensor.
        error = setpoint - measurement

        unconstrained_flow = (
            coolant_bias
            - kp * error
            - ki * integral_error
        )

        # Conditional integration anti-windup.
        integration_allowed = (
            params.minimum_coolant_flow
            < unconstrained_flow
            < params.maximum_coolant_flow
            or (
                unconstrained_flow >= params.maximum_coolant_flow
                and error > 0
            )
            or (
                unconstrained_flow <= params.minimum_coolant_flow
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

        commanded_flow = np.clip(
            unconstrained_flow,
            params.minimum_coolant_flow,
            params.maximum_coolant_flow,
        )

        # Inject a stuck-at-maximum actuator fault.
        is_fault_active = current_time >= fault_time

        if is_fault_active:
            actual_flow = stuck_flow
        else:
            actual_flow = commanded_flow

        # The physical plant receives the actual actuator output.
        plant_states[i + 1] = integrate_process(
            plant_states[i],
            actual_flow,
            params,
            time[i],
            time[i + 1],
        )

        # The digital twin assumes a healthy actuator.
        # It receives the controller command.
        twin_states[i + 1] = integrate_process(
            twin_states[i],
            commanded_flow,
            params,
            time[i],
            time[i + 1],
        )

        # Model-based residual.
        residual = measurement - twin_states[i, 1]

        # Persistent threshold-based fault detection.
        if abs(residual) > detection_threshold:
            consecutive_exceedances += 1
        else:
            consecutive_exceedances = 0

        if (
            consecutive_exceedances >= persistence_samples
            and detection_time is None
        ):
            detection_time = current_time

        measured_temperature[i] = measurement

        commanded_flows[i] = commanded_flow
        actual_flows[i] = actual_flow

        residuals[i] = residual

        fault_active[i] = is_fault_active
        alarm[i] = detection_time is not None

    # Store final sample.
    measured_temperature[-1] = (
        plant_states[-1, 1]
        + rng.normal(0.0, noise_std)
    )

    residuals[-1] = (
        measured_temperature[-1]
        - twin_states[-1, 1]
    )

    commanded_flows[-1] = commanded_flows[-2]
    actual_flows[-1] = actual_flows[-2]

    fault_active[-1] = time[-1] >= fault_time
    alarm[-1] = detection_time is not None

    return {
        "time": time,
        "plant_states": plant_states,
        "twin_states": twin_states,
        "measured_temperature": measured_temperature,
        "commanded_flows": commanded_flows,
        "actual_flows": actual_flows,
        "residuals": residuals,
        "fault_active": fault_active,
        "alarm": alarm,
        "detection_time": detection_time,
    }


def main():

    params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    fault_time = 30.0

    stuck_flow = params.maximum_coolant_flow

    detection_threshold = 0.75

    results = simulate_actuator_fault(
        params=params,
        initial_state=initial_state,
        setpoint=356.0,
        kp=0.03,
        ki=0.003,
        simulation_time=80.0,
        sample_time=0.05,
        fault_time=fault_time,
        stuck_flow=stuck_flow,
        noise_std=0.05,
        detection_threshold=detection_threshold,
        persistence_samples=5,
        random_seed=42,
    )

    time = results["time"]

    plant_temperature = results["plant_states"][:, 1]
    twin_temperature = results["twin_states"][:, 1]

    detection_time = results["detection_time"]

    pre_fault_mask = time < fault_time

    false_alarm = np.any(
        results["alarm"][pre_fault_mask]
    )

    within_limits = np.all(
        (results["actual_flows"] >= params.minimum_coolant_flow)
        & (results["actual_flows"] <= params.maximum_coolant_flow)
    )

    print("\n========== ACTUATOR FAULT DETECTION RESULTS ==========")

    print("Fault type: Actuator stuck at maximum")
    print(f"Fault injection time: {fault_time:.2f} min")
    print(f"Stuck actuator flow: {stuck_flow:.6f} m^3/min")
    print("Sensor fault: None")
    print("Measurement noise std: 0.05 K")
    print(f"Detection threshold: {detection_threshold:.2f} K")

    print("\nDetection results:")

    if detection_time is None:
        print("Fault detected: False")
        print("Detection delay: Not detected")
    else:
        print("Fault detected: True")
        print(f"Detection time: {detection_time:.3f} min")
        print(
            f"Detection delay: "
            f"{detection_time - fault_time:.3f} min"
        )

    print(f"False alarm before fault: {false_alarm}")

    print("\nFinal process values:")

    print(
        f"Actual reactor temperature: "
        f"{plant_temperature[-1]:.6f} K"
    )

    print(
        f"Twin predicted temperature: "
        f"{twin_temperature[-1]:.6f} K"
    )

    print(
        f"Final residual: "
        f"{results['residuals'][-1]:.6f} K"
    )

    print(
        f"Final commanded flow: "
        f"{results['commanded_flows'][-1]:.6f} m^3/min"
    )

    print(
        f"Final actual flow: "
        f"{results['actual_flows'][-1]:.6f} m^3/min"
    )

    print("\nActuator constraints:")
    print(f"Within limits: {within_limits}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 13),
        sharex=True,
    )

    axes[0].plot(
        time,
        plant_temperature,
        label="Actual plant temperature",
    )

    axes[0].plot(
        time,
        twin_temperature,
        "--",
        label="Digital twin prediction",
    )

    axes[0].plot(
        time,
        results["measured_temperature"],
        alpha=0.4,
        label="Sensor measurement",
    )

    axes[0].axvline(
        fault_time,
        linestyle=":",
        label="Fault injection",
    )

    axes[0].set_ylabel("Temperature (K)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time,
        results["residuals"],
        label="Residual",
    )

    axes[1].axhline(
        detection_threshold,
        linestyle="--",
        label="Positive threshold",
    )

    axes[1].axhline(
        -detection_threshold,
        linestyle="--",
        label="Negative threshold",
    )

    axes[1].axvline(
        fault_time,
        linestyle=":",
        label="Fault injection",
    )

    if detection_time is not None:
        axes[1].axvline(
            detection_time,
            linestyle="-.",
            label="Detection time",
        )

    axes[1].set_ylabel("Residual (K)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        time,
        results["commanded_flows"],
        label="Commanded coolant flow",
    )

    axes[2].plot(
        time,
        results["actual_flows"],
        "--",
        label="Actual coolant flow",
    )

    axes[2].axvline(
        fault_time,
        linestyle=":",
        label="Fault injection",
    )

    axes[2].set_ylabel("Coolant flow (m³/min)")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].step(
        time,
        results["alarm"].astype(int),
        where="post",
        label="Fault alarm",
    )

    axes[3].set_ylabel("Alarm")
    axes[3].set_xlabel("Time (min)")
    axes[3].set_yticks([0, 1])
    axes[3].legend()
    axes[3].grid(True)

    fig.suptitle(
        "CSTR - Model-Based Actuator Fault Detection"
    )

    fig.tight_layout()

    output_path = (
        RESULTS_DIR
        / "cstr_actuator_fault_detection.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
