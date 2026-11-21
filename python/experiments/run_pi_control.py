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


def simulate_closed_loop(
    params,
    initial_state,
    setpoint,
    kp,
    ki,
    simulation_time=80.0,
    sample_time=0.05,
):
    number_of_steps = int(simulation_time / sample_time) + 1

    time = np.linspace(
        0.0,
        simulation_time,
        number_of_steps,
    )

    states = np.zeros((number_of_steps, 3))
    coolant_flows = np.zeros(number_of_steps)
    errors = np.zeros(number_of_steps)

    states[0] = initial_state

    integral_error = 0.0

    coolant_bias = params.nominal_coolant_flow

    for i in range(number_of_steps - 1):
        current_state = states[i]

        temperature = current_state[1]
        error = setpoint - temperature

        # PI controller with negative process gain.
        unconstrained_flow = (
            coolant_bias
            - kp * error
            - ki * integral_error
        )

        # Actuator saturation.
        coolant_flow = np.clip(
            unconstrained_flow,
            params.minimum_coolant_flow,
            params.maximum_coolant_flow,
        )

        # Conditional integration anti-windup.
        integration_allowed = (
            params.minimum_coolant_flow < unconstrained_flow < params.maximum_coolant_flow
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

        # Recalculate the control signal after integration.
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

        # Zero-order-hold actuator command over one sample.
        solution = solve_ivp(
            fun=lambda t, x: cstr_dynamics(
                t,
                x,
                coolant_flow,
                params,
            ),
            t_span=(
                time[i],
                time[i + 1],
            ),
            y0=current_state,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )

        if not solution.success:
            raise RuntimeError(solution.message)

        states[i + 1] = solution.y[:, -1]

        coolant_flows[i] = coolant_flow
        errors[i] = error

    coolant_flows[-1] = coolant_flows[-2]
    errors[-1] = setpoint - states[-1, 1]

    return time, states, coolant_flows, errors


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

    time, states, coolant_flow, errors = simulate_closed_loop(
        params=params,
        initial_state=initial_state,
        setpoint=setpoint,
        kp=kp,
        ki=ki,
        simulation_time=80.0,
        sample_time=0.05,
    )

    concentration = states[:, 0]
    reactor_temperature = states[:, 1]
    jacket_temperature = states[:, 2]

    print("\n========== PI CONTROL RESULTS ==========")

    print(f"Temperature setpoint: {setpoint:.4f} K")
    print(f"Proportional gain: {kp:.6f}")
    print(f"Integral gain: {ki:.6f}")

    print("\nFinal process values:")
    print(f"Concentration: {concentration[-1]:.6f} mol/m^3")
    print(f"Reactor temperature: {reactor_temperature[-1]:.6f} K")
    print(f"Jacket temperature: {jacket_temperature[-1]:.6f} K")
    print(f"Coolant flow: {coolant_flow[-1]:.6f} m^3/min")

    print("\nPerformance metrics:")
    print(f"Final tracking error: {errors[-1]:.6f} K")
    print(f"Maximum temperature: {np.max(reactor_temperature):.6f} K")
    print(f"Minimum temperature: {np.min(reactor_temperature):.6f} K")
    print(f"Maximum coolant flow: {np.max(coolant_flow):.6f} m^3/min")
    print(f"Minimum coolant flow: {np.min(coolant_flow):.6f} m^3/min")

    print("\nActuator constraints:")
    print(
        "Within limits:",
        bool(
            np.all(coolant_flow >= params.minimum_coolant_flow)
            and np.all(coolant_flow <= params.maximum_coolant_flow)
        ),
    )

    # Plot results.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
    )

    axes[0].plot(
        time,
        reactor_temperature,
        label="Reactor temperature",
    )

    axes[0].axhline(
        setpoint,
        linestyle="--",
        label="Temperature setpoint",
    )

    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("CSTR PI Closed-Loop Control")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time,
        coolant_flow,
        label="Coolant flow",
    )

    axes[1].axhline(
        params.minimum_coolant_flow,
        linestyle="--",
        label="Minimum actuator limit",
    )

    axes[1].axhline(
        params.maximum_coolant_flow,
        linestyle="--",
        label="Maximum actuator limit",
    )

    axes[1].set_ylabel("Coolant flow (m^3/min)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        time,
        errors,
        label="Tracking error",
    )

    axes[2].set_xlabel("Time (min)")
    axes[2].set_ylabel("Error (K)")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()

    output_directory = PROJECT_ROOT / "results" / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_directory / "cstr_pi_control.png",
        dpi=150,
    )

    plt.show()


if __name__ == "__main__":
    main()
