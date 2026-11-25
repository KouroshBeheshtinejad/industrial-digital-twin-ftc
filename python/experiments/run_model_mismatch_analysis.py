from dataclasses import replace
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


def integrate_process(
    state,
    coolant_flow,
    params,
    start_time,
    end_time,
):
    solution = solve_ivp(
        fun=lambda t, x: cstr_dynamics(
            t,
            x,
            coolant_flow,
            params,
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


def simulate_model_mismatch(
    nominal_params,
    initial_state,
    setpoint=356.0,
    kp=0.03,
    ki=0.003,
    simulation_time=80.0,
    sample_time=0.05,
    plant_ua_multiplier=1.05,
    noise_std=0.05,
    detection_threshold=0.5,
    persistence_samples=5,
    random_seed=42,
):
    # The physical plant has a different heat-transfer coefficient.
    plant_params = replace(
        nominal_params,
        heat_transfer_coefficient=(
            nominal_params.heat_transfer_coefficient
            * plant_ua_multiplier
        ),
    )

    # The digital twin retains the nominal parameters.
    twin_params = nominal_params

    number_of_steps = int(simulation_time / sample_time) + 1

    time = np.linspace(
        0.0,
        simulation_time,
        number_of_steps,
    )

    plant_states = np.zeros((number_of_steps, 3))
    twin_states = np.zeros((number_of_steps, 3))

    measurements = np.zeros(number_of_steps)
    coolant_flows = np.zeros(number_of_steps)
    residuals = np.zeros(number_of_steps)
    alarms = np.zeros(number_of_steps, dtype=bool)

    plant_states[0] = initial_state
    twin_states[0] = initial_state

    rng = np.random.default_rng(random_seed)

    integral_error = 0.0
    coolant_bias = nominal_params.nominal_coolant_flow

    consecutive_exceedances = 0
    detection_time = None

    for i in range(number_of_steps - 1):

        current_time = time[i]

        # Healthy sensor: no injected bias.
        measurement = (
            plant_states[i, 1]
            + rng.normal(0.0, noise_std)
        )

        # PI controller uses the healthy plant measurement.
        error = setpoint - measurement

        unconstrained_flow = (
            coolant_bias
            - kp * error
            - ki * integral_error
        )

        # Conditional integration anti-windup.
        integration_allowed = (
            nominal_params.minimum_coolant_flow
            < unconstrained_flow
            < nominal_params.maximum_coolant_flow
            or (
                unconstrained_flow >= nominal_params.maximum_coolant_flow
                and error > 0
            )
            or (
                unconstrained_flow <= nominal_params.minimum_coolant_flow
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
            nominal_params.minimum_coolant_flow,
            nominal_params.maximum_coolant_flow,
        )

        # Advance the actual plant with modified parameters.
        plant_states[i + 1] = integrate_process(
            plant_states[i],
            coolant_flow,
            plant_params,
            time[i],
            time[i + 1],
        )

        # Advance the digital twin with nominal parameters.
        # Both receive the same applied actuator command.
        twin_states[i + 1] = integrate_process(
            twin_states[i],
            coolant_flow,
            twin_params,
            time[i],
            time[i + 1],
        )

        # Residual uses measurement and twin prediction only.
        residual = measurement - twin_states[i, 1]

        if abs(residual) > detection_threshold:
            consecutive_exceedances += 1
        else:
            consecutive_exceedances = 0

        if (
            consecutive_exceedances >= persistence_samples
            and detection_time is None
        ):
            detection_time = current_time

        measurements[i] = measurement
        coolant_flows[i] = coolant_flow
        residuals[i] = residual
        alarms[i] = detection_time is not None

    # Final sample.
    measurements[-1] = (
        plant_states[-1, 1]
        + rng.normal(0.0, noise_std)
    )

    residuals[-1] = (
        measurements[-1] - twin_states[-1, 1]
    )

    coolant_flows[-1] = coolant_flows[-2]
    alarms[-1] = detection_time is not None

    return {
        "time": time,
        "plant_states": plant_states,
        "twin_states": twin_states,
        "measurements": measurements,
        "coolant_flows": coolant_flows,
        "residuals": residuals,
        "alarms": alarms,
        "detection_time": detection_time,
        "plant_params": plant_params,
    }


def main():

    params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    mismatch_percentage = 5.0
    detection_threshold = 0.5

    results = simulate_model_mismatch(
        nominal_params=params,
        initial_state=initial_state,
        setpoint=356.0,
        kp=0.03,
        ki=0.003,
        simulation_time=80.0,
        sample_time=0.05,
        plant_ua_multiplier=1.05,
        noise_std=0.05,
        detection_threshold=detection_threshold,
        persistence_samples=5,
    )

    time = results["time"]
    plant_temperature = results["plant_states"][:, 1]
    twin_temperature = results["twin_states"][:, 1]
    measurements = results["measurements"]
    residuals = results["residuals"]
    coolant_flows = results["coolant_flows"]

    detection_time = results["detection_time"]

    false_alarm = detection_time is not None

    print("\n========== MODEL MISMATCH ANALYSIS ==========")

    print("\nModel configuration:")
    print(
        f"Nominal UA: "
        f"{params.heat_transfer_coefficient:.2f}"
    )
    print(
        f"Actual plant UA: "
        f"{results['plant_params'].heat_transfer_coefficient:.2f}"
    )
    print(
        f"Model mismatch: "
        f"{mismatch_percentage:.2f}%"
    )

    print("\nSensor configuration:")
    print("Sensor bias: 0.00 K")
    print("Sensor noise std: 0.05 K")
    print("Injected sensor fault: None")

    print("\nDetection results:")

    if detection_time is None:
        print("Fault alarm: False")
        print("False alarm caused by model mismatch: False")
    else:
        print("Fault alarm: True")
        print(f"Detection time: {detection_time:.3f} min")
        print("False alarm caused by model mismatch: True")

    print("\nResidual statistics:")
    print(f"Maximum absolute residual: {np.max(np.abs(residuals)):.6f} K")
    print(f"Mean residual: {np.mean(residuals):.6f} K")
    print(f"Final residual: {residuals[-1]:.6f} K")

    print("\nFinal process values:")
    print(f"Actual plant temperature: {plant_temperature[-1]:.6f} K")
    print(f"Twin predicted temperature: {twin_temperature[-1]:.6f} K")
    print(f"Sensor measurement: {measurements[-1]:.6f} K")
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
        measurements,
        alpha=0.35,
        label="Healthy sensor measurement",
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
    axes[2].set_ylabel("Coolant flow (m³/min)")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].step(
        time,
        results["alarms"].astype(int),
        where="post",
        label="Fault alarm",
    )
    axes[3].set_ylabel("Alarm")
    axes[3].set_xlabel("Time (min)")
    axes[3].set_yticks([0, 1])
    axes[3].legend()
    axes[3].grid(True)

    fig.suptitle("CSTR - Digital Twin Model Mismatch Analysis")
    fig.tight_layout()

    output_path = (
        RESULTS_DIR / "cstr_model_mismatch_analysis.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.show()

    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()