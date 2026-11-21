
from pathlib import Path

import numpy as np
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


def calculate_jacobian(function, state):
    """Calculate a numerical Jacobian using central differences."""

    state = np.asarray(state, dtype=float)
    number_of_states = len(state)

    jacobian = np.zeros(
        (number_of_states, number_of_states)
    )

    for i in range(number_of_states):
        step = 1e-5 * max(1.0, abs(state[i]))

        state_plus = state.copy()
        state_minus = state.copy()

        state_plus[i] += step
        state_minus[i] -= step

        jacobian[:, i] = (
            function(state_plus) - function(state_minus)
        ) / (2.0 * step)

    return jacobian


def main():
    params = load_parameters(CONFIG_PATH)

    coolant_flow = params.nominal_coolant_flow

    initial_guess = np.array([
        100.0,
        350.0,
        300.0,
    ])

    def dynamics(state):
        return cstr_dynamics(
            0.0,
            state,
            coolant_flow,
            params,
        )

    equilibrium = root(
        dynamics,
        initial_guess,
        method="hybr",
    )

    if not equilibrium.success:
        raise RuntimeError(
            f"Equilibrium solver failed: {equilibrium.message}"
        )

    operating_point = equilibrium.x
    residual = dynamics(operating_point)

    if not np.all(np.isfinite(operating_point)):
        raise RuntimeError("Non-finite equilibrium state.")

    if (
        operating_point[0] < 0
        or operating_point[1] <= 0
        or operating_point[2] <= 0
    ):
        raise RuntimeError(
            "The computed equilibrium is physically invalid."
        )

    if np.linalg.norm(residual, ord=np.inf) > 1e-7:
        raise RuntimeError(
            "Equilibrium residual is too large."
        )

    jacobian = calculate_jacobian(
        dynamics,
        operating_point,
    )

    eigenvalues = np.linalg.eigvals(jacobian)

    is_stable = np.all(
        np.real(eigenvalues) < 0
    )

    print("\n========== CSTR OPERATING POINT ==========")

    print(f"Coolant flow: {coolant_flow:.6f} m^3/min")

    print(
        f"Concentration: {operating_point[0]:.6f} mol/m^3"
    )

    print(
        f"Reactor temperature: {operating_point[1]:.6f} K"
    )

    print(
        f"Jacket temperature: {operating_point[2]:.6f} K"
    )

    print("\n========== EQUILIBRIUM VALIDATION ==========")

    print("State derivatives:")
    print(residual)

    print(
        f"Infinity norm of residual: "
        f"{np.linalg.norm(residual, ord=np.inf):.3e}"
    )

    print("\n========== JACOBIAN MATRIX ==========")
    print(jacobian)

    print("\n========== EIGENVALUE ANALYSIS ==========")

    for index, eigenvalue in enumerate(eigenvalues, start=1):
        print(f"Eigenvalue {index}: {eigenvalue:.8f}")

    print("\n========== LOCAL STABILITY ==========")

    if is_stable:
        print("The operating point is locally asymptotically stable.")
    else:
        print("The operating point is NOT locally asymptotically stable.")


if __name__ == "__main__":
    main()
