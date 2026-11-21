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


def simulate_with_disturbance(
    params,
    initial_state,
    setpoint,
    kp,
    ki,
    simulation_time=100.0,
    sample_time=0.05,
    disturbance_time=20.0,
    disturbed_feed_temperature=351.0,
):
    number_of_steps = int(simulation_time / sample_time) + 1

    time = np.linspace(
        0.0,
        simulation_time,
        number_of_steps,
    )

    states = np.zeros((number_of_steps, 3))
    coolant_flows = np.zeros(number_of_steps)
    feed_temperatures = np.zeros(number_of_steps)

    states[0] = initial_state

    integral_error = 0.0
    coolant_bias = params.nominal_coolant_flow

    for i in range(number_of_steps - 1):

        current_time = time[i]
        current_state = states[i]

        # Apply the feed-temperature disturbance.
        if current_time >= disturbance_time:
            active_params = replace(
                params,
                feed_temperature=disturbed_feed_temperature,
            )
        else:
            active_params = params

        temperature = current_state[1]
        error = setpoint - temperature

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

        solution = solve_ivp(
            fun=lambda t, x: cstr_dynamics(
                t,
                x,
                coolant_flow,
                active_params,
            ),
            t_span=(time[i], time[i + 1]),
            y0=current_state,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )

        if not solution.success:
            raise RuntimeError(solution.message)

        states[i + 1] = solution.y[:, -1]
        coolant_flows[i] = coolant_flow
        feed_temperatures[i] = active_params.feed_temperature

    coolant_flows[-1] = coolant_flows[-2]
    feed_temperatures[-1] = (
        disturbed_feed_temperature
        if time[-1] >= disturbance_time
        else params.feed_temperature
    )

    return time, states, coolant_flows, feed_temperatures


def calculate_metrics(
    time,
    temperature,
    setpoint,
    disturbance_time,
):
    error = setpoint - temperature

    post_disturbance = time >= disturbance_time

    disturbance_temperature = temperature[
        np.searchsorted(time, disturbance_time)
    ]

    peak_deviation = np.max(
        np.abs(error[post_disturbance])
    )

    band = 0.1

    outside_band_indices = np.where(
        post_disturbance
        & (np.abs(error) > band)
    )[0]

    if len(outside_band_indices) == 0:
        settling_time = 0.0
    elif outside_band_indices[-1] == len(time) - 1:
        settling_time = None
    else:
        settling_time = (
            time[outside_band_indices[-1] + 1]
            - disturbance_time
        )

    iae = np.trapezoid(
        np.abs(error[post_disturbance]),
        time[post_disturbance],
    )

    ise = np.trapezoid(
        error[post_disturbance] ** 2,
        time[post_disturbance],
    )

    return {
        "disturbance_temperature": disturbance_temperature,
        "peak_deviation": peak_deviation,
        "settling_time": settling_time,
        "iae": iae,
        "ise": ise,
        "final_error": error[-1],
    }


def main():

    params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    setpoint = 356.0
    kp = 0.03
    ki = 0.003

    disturbance_time = 20.0
    disturbed_feed_temperature = 351.0

    time, states, coolant_flows, feed_temperatures = (
        simulate_with_disturbance(
            params=params,
            initial_state=initial_state,
            setpoint=setpoint,
            kp=kp,
            ki=ki,
            simulation_time=100.0,
            sample_time=0.05,
            disturbance_time=disturbance_time,
            disturbed_feed_temperature=disturbed_feed_temperature,
        )
    )

    concentration = states[:, 0]
    temperature = states[:, 1]
    jacket_temperature = states[:, 2]

    metrics = calculate_metrics(
        time,
        temperature,
        setpoint,
        disturbance_time,
    )

    print("\n========== DISTURBANCE REJECTION RESULTS ==========")

    print(f"Temperature setpoint: {setpoint:.4f} K")
    print(f"Disturbance time: {disturbance_time:.2f} min")
    print(
        f"Feed temperature: {params.feed_temperature:.2f} "
        f"-> {disturbed_feed_temperature:.2f} K"
    )

    print("\nFinal process values:")
    print(f"Concentration: {concentration[-1]:.6f} mol/m^3")
    print(f"Reactor temperature: {temperature[-1]:.6f} K")
    print(f"Jacket temperature: {jacket_temperature[-1]:.6f} K")
    print(f"Coolant flow: {coolant_flows[-1]:.6f} m^3/min")

    print("\nPerformance metrics:")
    print(
        f"Temperature at disturbance: "
        f"{metrics['disturbance_temperature']:.6f} K"
    )
    print(
        f"Peak absolute deviation: "
        f"{metrics['peak_deviation']:.6f} K"
    )

    if metrics["settling_time"] is None:
        print("Settling time: Not settled within simulation")
    else:
        print(
            f"Settling time (±0.1 K): "
            f"{metrics['settling_time']:.3f} min"
        )

    print(f"IAE: {metrics['iae']:.6f} K.min")
    print(f"ISE: {metrics['ise']:.6f} K^2.min")
    print(f"Final tracking error: {metrics['final_error']:.6f} K")

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
        3,
        1,
        figsize=(11, 10),
        sharex=True,
    )

    axes[0].plot(
        time,
        temperature,
        label="Reactor temperature",
    )
    axes[0].axhline(
        setpoint,
        linestyle="--",
        label="Setpoint",
    )
    axes[0].axvline(
        disturbance_time,
        linestyle=":",
        label="Disturbance",
    )
    axes[0].set_ylabel("Temperature (K)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time,
        coolant_flows,
        label="Coolant flow",
    )
    axes[1].axhline(
        params.minimum_coolant_flow,
        linestyle="--",
        label="Minimum limit",
    )
    axes[1].axhline(
        params.maximum_coolant_flow,
        linestyle="--",
        label="Maximum limit",
    )
    axes[1].axvline(
        disturbance_time,
        linestyle=":",
    )
    axes[1].set_ylabel("Coolant flow (m³/min)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        time,
        feed_temperatures,
        label="Feed temperature",
    )
    axes[2].axvline(
        disturbance_time,
        linestyle=":",
    )
    axes[2].set_ylabel("Feed temperature (K)")
    axes[2].set_xlabel("Time (min)")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle("CSTR PI Control - Disturbance Rejection")
    fig.tight_layout()

    output_path = (
        RESULTS_DIR / "cstr_disturbance_rejection.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.show()

    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
