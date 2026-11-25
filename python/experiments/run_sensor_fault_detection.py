from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from python.plant_model.cstr_model import (
    load_parameters,
    cstr_dynamics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plant_parameters"
    / "cstr_nominal.json"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "figures"


def integrate_process(state, coolant_flow, params, start_time, end_time):
    solution = solve_ivp(
        fun=lambda t, x: cstr_dynamics(
            t, x, coolant_flow, params
        ),
        t_span=(start_time, end_time),
        y0=state,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y[:, -1]


def simulate_sensor_fault(
    params,
    initial_state,
    setpoint=356.0,
    kp=0.03,
    ki=0.003,
    simulation_time=80.0,
    sample_time=0.05,
    fault_time=30.0,
    sensor_bias=2.0,
    noise_std=0.05,
    detection_threshold=0.75,
    persistence_samples=5,
    random_seed=42,
):
    number_of_steps = int(simulation_time / sample_time) + 1

    time = np.linspace(
        0.0,
        simulation_time,
        number_of_steps,
    )

    plant_states = np.zeros((number_of_steps, 3))
    twin_states = np.zeros((number_of_steps, 3))

    measured_temperature = np.zeros(number_of_steps)
    coolant_flows = np.zeros(number_of_steps)
    residuals = np.zeros(number_of_steps)
    fault_active = np.zeros(number_of_steps, dtype=bool)
    alarm = np.zeros(number_of_steps, dtype=bool)

    plant_states[0] = initial_state
    twin_states[0] = initial_state

    rng = np.random.default_rng(random_seed)

    integral_error = 0.0
    coolant_bias = params.nominal_coolant_flow
    consecutive_exceedances = 0

    detection_time = None

    for i in range(number_of_steps - 1):

        current_time = time[i]

        # Simulate sensor bias and measurement noise.
        is_fault_active = current_time >= fault_time

        active_bias = sensor_bias if is_fault_active else 0.0

        actual_temperature = plant_states[i, 1]

        measurement = (
            actual_temperature
            + active_bias
            + rng.normal(0.0, noise_std)
        )

        # PI controller uses the sensor measurement.
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

        coolant_flow = np.clip(
            unconstrained_flow,
            params.minimum_coolant_flow,
            params.maximum_coolant_flow,
        )

        # Advance the physical plant.
        plant_states[i + 1] = integrate_process(
            plant_states[i],
            coolant_flow,
            params,
            time[i],
            time[i + 1],
        )

        # Advance the independent digital twin using the same
        # applied actuator command, not the measured temperature.
        twin_states[i + 1] = integrate_process(
            twin_states[i],
            coolant_flow,
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
        coolant_flows[i] = coolant_flow
        residuals[i] = residual
        fault_active[i] = is_fault_active
        alarm[i] = detection_time is not None

    # Store the final measurement and residual.
    final_fault_active = time[-1] >= fault_time

    final_bias = sensor_bias if final_fault_active else 0.0

    measured_temperature[-1] = (
        plant_states[-1, 1]
        + final_bias
        + rng.normal(0.0, noise_std)
    )

    residuals[-1] = (
        measured_temperature[-1] - twin_states[-1, 1]
    )

    coolant_flows[-1] = coolant_flows[-2]
    fault_active[-1] = final_fault_active
    alarm[-1] = detection_time is not None

    return {
        "time": time,
        "plant_states": plant_states,
        "twin_states": twin_states,
        "measured_temperature": measured_temperature,
        "coolant_flows": coolant_flows,
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
    sensor_bias = 2.0
    detection_threshold = 0.75

    results = simulate_sensor_fault(
        params=params,
        initial_state=initial_state,
        setpoint=356.0,
        kp=0.03,
        ki=0.003,
        simulation_time=80.0,
        sample_time=0.05,
        fault_time=fault_time,
        sensor_bias=sensor_bias,
        noise_std=0.05,
        detection_threshold=detection_threshold,
        persistence_samples=5,
    )

    time = results["time"]
    plant_temperature = results["plant_states"][:, 1]
    twin_temperature = results["twin_states"][:, 1]
    measured_temperature = results["measured_temperature"]
    residuals = results["residuals"]
    coolant_flows = results["coolant_flows"]

    detection_time = results["detection_time"]

    pre_fault_mask = time < fault_time

    false_alarm = np.any(
        results["alarm"][pre_fault_mask]
    )

    print("\n========== SENSOR FAULT DETECTION RESULTS ==========")

    print(f"Fault type: Constant sensor bias")
    print(f"Fault injection time: {fault_time:.2f} min")
    print(f"Injected sensor bias: {sensor_bias:.2f} K")
    print(f"Measurement noise std: 0.05 K")
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
    print(f"Actual reactor temperature: {plant_temperature[-1]:.6f} K")
    print(f"Twin predicted temperature: {twin_temperature[-1]:.6f} K")
    print(f"Measured temperature: {measured_temperature[-1]:.6f} K")
    print(f"Final residual: {residuals[-1]:.6f} K")
    print(f"Final coolant flow: {coolant_flows[-1]:.6f} m^3/min")

    print("\nActuator constraints:")
    print(
        "Within limits:",
        np.all(
            (coolant_flows >= params.minimum_coolant_flow)
            & (coolant_flows <= params.maximum_coolant_flow)
        ),
    )

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
        measured_temperature,
        alpha=0.5,
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
        residuals,
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
        coolant_flows,
        label="Coolant flow",
    )
    axes[2].axhline(
        params.minimum_coolant_flow,
        linestyle="--",
        label="Minimum limit",
    )
    axes[2].axhline(
        params.maximum_coolant_flow,
        linestyle="--",
        label="Maximum limit",
    )
    axes[2].axvline(
        fault_time,
        linestyle=":",
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

    fig.suptitle("CSTR - Model-Based Sensor Fault Detection")
    fig.tight_layout()

    output_path = (
        RESULTS_DIR / "cstr_sensor_fault_detection.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.show()

    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
