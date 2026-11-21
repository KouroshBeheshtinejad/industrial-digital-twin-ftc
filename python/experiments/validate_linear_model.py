from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

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


def calculate_jacobian_state(function, state):
    state = np.asarray(state, dtype=float)
    n = len(state)
    jacobian = np.zeros((n, n))

    for i in range(n):
        step = 1e-5 * max(1.0, abs(state[i]))

        x_plus = state.copy()
        x_minus = state.copy()

        x_plus[i] += step
        x_minus[i] -= step

        jacobian[:, i] = (
            function(x_plus) - function(x_minus)
        ) / (2.0 * step)

    return jacobian


def calculate_jacobian_input(function, input_value):
    step = 1e-6 * max(1.0, abs(input_value))

    return (
        function(input_value + step)
        - function(input_value - step)
    ) / (2.0 * step)


def main():
    params = load_parameters(CONFIG_PATH)

    nominal_input = params.nominal_coolant_flow
    input_change = 0.005
    changed_input = nominal_input + input_change

    simulation_time = 40.0
    number_of_points = 1001

    time_points = np.linspace(
        0.0,
        simulation_time,
        number_of_points,
    )

    def plant_dynamics(state, input_value):
        return cstr_dynamics(
            0.0,
            state,
            input_value,
            params,
        )

    # Find the nominal operating point.
    equilibrium = root(
        lambda state: plant_dynamics(state, nominal_input),
        np.array([100.0, 350.0, 300.0]),
    )

    if not equilibrium.success:
        raise RuntimeError(equilibrium.message)

    x_star = equilibrium.x

    # Calculate linearized state-space matrices.
    A = calculate_jacobian_state(
        lambda state: plant_dynamics(state, nominal_input),
        x_star,
    )

    B = calculate_jacobian_input(
        lambda input_value: plant_dynamics(x_star, input_value),
        nominal_input,
    ).reshape(3, 1)

    # Simulate the nonlinear plant after the input step.
    nonlinear_solution = solve_ivp(
        fun=lambda t, x: plant_dynamics(x, changed_input),
        t_span=(0.0, simulation_time),
        y0=x_star,
        t_eval=time_points,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not nonlinear_solution.success:
        raise RuntimeError(nonlinear_solution.message)

    # Simulate the linearized deviation model.
    def linear_dynamics(time, delta_state):
        return (
            A @ delta_state
            + B[:, 0] * input_change
        )

    linear_solution = solve_ivp(
        fun=linear_dynamics,
        t_span=(0.0, simulation_time),
        y0=np.zeros(3),
        t_eval=time_points,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not linear_solution.success:
        raise RuntimeError(linear_solution.message)

    # Convert deviations into absolute reactor temperatures.
    nonlinear_temperature = nonlinear_solution.y[1]

    linear_temperature = (
        x_star[1] + linear_solution.y[1]
    )

    temperature_difference = (
        nonlinear_temperature - linear_temperature
    )

    maximum_error = np.max(
        np.abs(temperature_difference)
    )

    print("\n========== LINEAR MODEL VALIDATION ==========")

    print(f"Nominal coolant flow: {nominal_input:.6f} m^3/min")
    print(f"Changed coolant flow: {changed_input:.6f} m^3/min")
    print(f"Input perturbation: {input_change:.6f} m^3/min")

    print("\nNominal reactor temperature:")
    print(f"{x_star[1]:.6f} K")

    print("\nFinal nonlinear temperature:")
    print(f"{nonlinear_temperature[-1]:.6f} K")

    print("\nFinal linear temperature:")
    print(f"{linear_temperature[-1]:.6f} K")

    print("\nMaximum absolute temperature error:")
    print(f"{maximum_error:.8f} K")

    # Plot the comparison.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    axes[0].plot(
        time_points,
        nonlinear_temperature,
        label="Nonlinear CSTR",
    )

    axes[0].plot(
        time_points,
        linear_temperature,
        "--",
        label="Linearized CSTR",
    )

    axes[0].set_ylabel("Reactor Temperature (K)")
    axes[0].set_title("Nonlinear vs Linearized Model")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time_points,
        temperature_difference,
        label="Temperature prediction error",
    )

    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Temperature Error (K)")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    output_directory = PROJECT_ROOT / "results" / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_directory / "linear_model_validation.png",
        dpi=150,
    )

    plt.show()


if __name__ == "__main__":
    main()
